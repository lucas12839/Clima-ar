"""El tiempo y el pronóstico de cada localidad (24/11/2026).

Antes la página mostraba el tiempo de Bahía Blanca para cualquier localidad."""
import json
import os
from datetime import datetime, timedelta, timezone

os.environ.setdefault("CLIMAAR_REMOTE_DATA_ENABLED", "0")
import app  # noqa: E402

LOCS = {l["id"]: l for l in json.load(open(app.LOCALIDADES_FILE, encoding="utf-8"))["localidades"]}


def hace(m):
    return (datetime.now(timezone.utc) - timedelta(minutes=m)).isoformat()


def modelo(temp):
    return {"temperatura": temp, "viento_vel": 20, "viento_dir": 200, "rafagas": 40, "humedad": 30,
            "presion": 1012, "nubosidad": 20, "codigo_tiempo": 1, "sensacion_termica": temp, "precipitacion": 0}


def preparar(tmp_path, monkeypatch, bcp=(), cwop=(), sazb=None):
    obs = tmp_path / "obs"
    obs.mkdir()
    (obs / "bcp_actual.json").write_text(json.dumps({"stations": list(bcp)}))
    (obs / "cwop_actual.json").write_text(json.dumps({"stations": list(cwop)}))
    meteo = tmp_path / "meteo.json"
    meteo.write_text(json.dumps({"observacion_sazb": sazb or {}, "ambiente_actual": {}}))
    monkeypatch.setattr(app, "OBS_DIR", obs)
    monkeypatch.setattr(app, "METEO_FILE", meteo)
    monkeypatch.setattr(app, "get_openmeteo_data", lambda: modelo(20))
    temps = {"bahia_blanca": 20, "ingeniero_white": 19, "general_cerri": 21, "punta_alta": 18,
             "tornquist": 25, "medanos": 24, "sierra_de_la_ventana": 22}
    monkeypatch.setattr(app, "get_openmeteo_locales", lambda: {k: modelo(v) for k, v in temps.items()})


PUNTA_ALTA = {"station_id": 76, "nombre": "PUNTA ALTA", "observation_time": None,
              "variables": {"temperatura_c": 17.4, "viento_kmh": 31.0, "rafaga_15min_kmh": 52.0,
                            "direccion_viento": "Sur"}}
SIERRAS = {"station_id": 97, "nombre": "SIERRAS GRANDES", "observation_time": None,
           "variables": {"temperatura_c": 12.0}}
NOROESTE = {"indicativo": "LU4DRH-11", "nombre": "BAHIA BLANCA NOROESTE (CWOP)", "lat": -38.6893, "lon": -62.3342,
            "observation_time": None, "variables": {"temperatura_c": 23.3, "viento_kmh": 10}}


def test_cada_localidad_tiene_su_temperatura(tmp_path, monkeypatch):
    preparar(tmp_path, monkeypatch, bcp=[dict(PUNTA_ALTA, observation_time=hace(4)),
                                         dict(SIERRAS, observation_time=hace(4))],
             cwop=[dict(NOROESTE, observation_time=hace(3))])
    t = {k: app._tiempo_actual(v) for k, v in LOCS.items()}
    # Punta Alta: su propia estación (viento y ráfaga de la misma).
    assert t["punta_alta"]["temperatura"] == 17.4
    assert (t["punta_alta"]["viento_vel"], t["punta_alta"]["rafagas"]) == (31, 52)
    assert t["punta_alta"]["fuentes"]["temperatura"].startswith("Punta Alta (hace")
    # Cerri: la estación Noroeste está a 5,5 km.
    assert t["general_cerri"]["temperatura"] == 23.3
    assert "Bahía Blanca Noroeste, a 5 km" in t["general_cerri"]["fuentes"]["temperatura"]
    # Tornquist y Médanos: sin estación cerca, modelo en su punto.
    assert t["tornquist"]["temperatura"] == 25 and "temperatura" not in t["tornquist"]["fuentes"]
    assert t["medanos"]["temperatura"] == 24
    # Sierra de la Ventana: Sierras Grandes está a 10,5 km sobre las sierras: no cuenta.
    assert t["sierra_de_la_ventana"]["temperatura"] == 22
    # Ya no son todas iguales.
    assert len({v["temperatura"] for v in t.values()}) >= 5


def test_estacion_vieja_no_cuenta_y_aeropuerto_solo_cerca(tmp_path, monkeypatch):
    sazb = {"disponible": True, "obsTime_utc": hace(20), "temperatura_c": 19.0, "presion_hpa": 1009,
            "viento_kt": 10, "direccion_viento": 180}
    preparar(tmp_path, monkeypatch, bcp=[dict(PUNTA_ALTA, observation_time=hace(60))], sazb=sazb)
    pa = app._tiempo_actual(LOCS["punta_alta"])
    assert pa["temperatura"] == 18                 # estación vieja: modelo
    assert pa["presion"] == 1012                   # el aeropuerto está a más de 12 km
    white = app._tiempo_actual(LOCS["ingeniero_white"])
    assert white["presion"] == 1009                # aeropuerto a 11,7 km


def test_bahia_igual_que_antes(tmp_path, monkeypatch):
    preparar(tmp_path, monkeypatch)
    assert app._tiempo_actual(LOCS["bahia_blanca"])["temperatura"] == app._tiempo_actual()["temperatura"] == 20


def test_openmeteo_un_solo_pedido_para_las_7(monkeypatch):
    pedidos = []

    class Resp:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    def get(url, timeout):
        pedidos.append(url)
        n = url.split("latitude=")[1].split("&")[0].count(",") + 1
        if "current=" in url:
            return Resp([{"current": {"temperature_2m": 10 + i}} for i in range(n)])
        return Resp([{"daily": {"time": ["2026-11-24", "2026-11-25"], "temperature_2m_max": [20 + i, 21 + i]}}
                     for i in range(n)])

    monkeypatch.setattr(app.requests, "get", get)
    for c in app._LOCALES_CACHE.values():
        c.update(ts=0.0, data=None, fallo=0.0)
    act = app.get_openmeteo_locales()
    assert act["bahia_blanca"]["temperatura"] == 10 and act["sierra_de_la_ventana"]["temperatura"] == 16
    app.get_openmeteo_locales()                            # guardado: no pide de nuevo
    pron = app.get_openmeteo_forecast_locales()
    assert pron["tornquist"][1]["maxima"] == 25
    assert len(pedidos) == 2
    for c in app._LOCALES_CACHE.values():
        c.update(ts=0.0, data=None, fallo=0.0)


def test_openmeteo_caido_no_rompe(monkeypatch):
    def get(url, timeout):
        raise app.requests.ConnectionError("sin red")

    monkeypatch.setattr(app.requests, "get", get)
    for c in app._LOCALES_CACHE.values():
        c.update(ts=0.0, data=None, fallo=0.0)
    assert app.get_openmeteo_locales() == {}
    assert app.get_openmeteo_forecast_locales() == {}


def test_modelo_caido_no_se_reintenta_en_cada_pedido(monkeypatch):
    intentos = []

    def get(url, timeout):
        intentos.append(url)
        raise app.requests.ConnectionError("sin red")

    monkeypatch.setattr(app.requests, "get", get)
    for c in app._LOCALES_CACHE.values():
        c.update(ts=0.0, data=None, fallo=0.0)
    monkeypatch.setitem(app._CURRENT_CACHE, "fallo", 0.0)
    monkeypatch.setitem(app._CURRENT_CACHE, "data", None)
    monkeypatch.setitem(app._CURRENT_CACHE, "ts", 0.0)
    for _ in range(5):
        app.get_openmeteo_locales()
        app.get_openmeteo_data()
    assert len(intentos) == 2          # uno de cada uno; los demás esperan el minuto
    for c in app._LOCALES_CACHE.values():
        c.update(ts=0.0, data=None, fallo=0.0)
    app._CURRENT_CACHE["fallo"] = 0.0


def test_anemometro_trabado_no_se_muestra(tmp_path, monkeypatch):
    preparar(tmp_path, monkeypatch, bcp=[dict(PUNTA_ALTA, observation_time=hace(4),
                                              variables=dict(PUNTA_ALTA["variables"], viento_kmh=0.0,
                                                             rafaga_15min_kmh=None))])
    monkeypatch.setattr(app, "_vientos_trabados", lambda: {"PUNTA ALTA"})
    t = app._tiempo_actual(LOCS["punta_alta"])
    assert t["temperatura"] == 17.4                        # la temperatura sí
    assert (t["viento_vel"], t["rafagas"]) == (20, 40)     # el viento, del modelo
