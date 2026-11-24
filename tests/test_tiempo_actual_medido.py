"""El tiempo ahora: el servidor solo muestra lo que calcula la cadena (24/11/2026).

Lo medido y controlado sale de senales_superficie.json ("tiempo_ahora"); lo que
no mide nadie, del modelo europeo en el punto de la localidad
(estado_meteorologico.json, "modelo_localidades")."""
import json
import os
from datetime import datetime, timedelta, timezone

os.environ.setdefault("CLIMAAR_REMOTE_DATA_ENABLED", "0")
import app  # noqa: E402

LOCS = {l["id"]: l for l in json.load(open(app.LOCALIDADES_FILE, encoding="utf-8"))["localidades"]}


def hace(m):
    return (datetime.now(timezone.utc) - timedelta(minutes=m)).isoformat()


def hora_actual():
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M")


def modelo(**por_loc):
    h = hora_actual()
    return {"modelo": "ECMWF IFS", "actualizado_utc": hace(10), "localidades": {
        k: {"horas": [h], "temperature_2m": [v[0]], "wind_speed_10m": [v[1]], "wind_gusts_10m": [v[2]],
            "wind_direction_10m": [200], "relative_humidity_2m": [40], "pressure_msl": [1012],
            "cloud_cover": [20], "precipitation": [0], "weather_code": [1]}
        for k, v in por_loc.items()}}


def preparar(tmp_path, monkeypatch, tiempo_ahora=None, modelo_locs=None, sazb=None):
    ia = tmp_path / "ia"
    (ia / "estaciones").mkdir(parents=True)
    (ia / "estaciones" / "senales_superficie.json").write_text(json.dumps(
        {"generado_utc": hace(2), "tiempo_ahora": tiempo_ahora or {}}))
    meteo = tmp_path / "meteo.json"
    meteo.write_text(json.dumps({"observacion_sazb": sazb or {}, "modelo_localidades": modelo_locs}))
    monkeypatch.setattr(app, "IA_DIR", ia)
    monkeypatch.setattr(app, "METEO_FILE", meteo)
    monkeypatch.setattr(app, "get_openmeteo_locales", lambda: {})


def med(valor, estacion, km=0.5, minutos=4, red="bcp", **extra):
    return dict(valor=valor, estacion=estacion, km=km, utc=hace(minutos), red=red, **extra)


def test_cada_localidad_su_dato_y_mismo_viento_que_la_tarjeta(tmp_path, monkeypatch):
    preparar(tmp_path, monkeypatch, tiempo_ahora={
        "bahia_blanca": {"temperatura": med(19.0, "BAHIA BLANCA CENTRO"),
                         "viento": med(21.0, "BAHIA BLANCA CENTRO", direccion=90.0, rafaga=33.0)},
        "punta_alta": {"temperatura": med(18.9, "PUNTA ALTA", km=0.4),
                       "viento": med(16.0, "GRUMBEIN", km=14.0, direccion=67.5, rafaga=None)},
    }, modelo_locs=modelo(bahia_blanca=(20, 25, 40), punta_alta=(21, 30, 61), tornquist=(23, 15, 35)))
    b = app._tiempo_actual(LOCS["bahia_blanca"])
    assert (b["temperatura"], b["viento_vel"], b["viento_dir"], b["rafagas"]) == (19.0, 21.0, 90.0, 33.0)
    assert b["fuentes"]["rafagas"] == b["fuentes"]["viento"]      # misma estación (sin rafaga_estacion)
    pa = app._tiempo_actual(LOCS["punta_alta"])
    assert (pa["temperatura"], pa["viento_vel"], pa["rafagas"]) == (18.9, 16.0, None)
    assert "Grünbein, a 14 km" in pa["fuentes"]["viento"]
    assert "rafagas" not in pa["fuentes"]                  # no se mezcla la ráfaga del modelo
    t = app._tiempo_actual(LOCS["tornquist"])               # sin estación: modelo de su punto
    assert (t["temperatura"], t["viento_vel"], t["rafagas"]) == (23, 15, 35)
    assert t["fuentes"] == {} and t["fuente_modelo"] == "modelo europeo"


def test_medicion_vieja_no_se_muestra(tmp_path, monkeypatch):
    preparar(tmp_path, monkeypatch, tiempo_ahora={
        "bahia_blanca": {"temperatura": med(19.0, "BAHIA BLANCA CENTRO", minutos=50)}},
        modelo_locs=modelo(bahia_blanca=(20, 25, 40)))
    b = app._tiempo_actual(LOCS["bahia_blanca"])
    assert b["temperatura"] == 20 and "temperatura" not in b["fuentes"]


def test_presion_del_aeropuerto_solo_cerca(tmp_path, monkeypatch):
    sazb = {"disponible": True, "obsTime_utc": hace(20), "presion_hpa": 1009}
    preparar(tmp_path, monkeypatch, modelo_locs=modelo(ingeniero_white=(20, 10, 20), punta_alta=(20, 10, 20)),
             sazb=sazb)
    assert app._tiempo_actual(LOCS["ingeniero_white"])["presion"] == 1009
    assert app._tiempo_actual(LOCS["punta_alta"])["presion"] == 1012


def test_sin_modelo_de_la_cadena_usa_el_respaldo(tmp_path, monkeypatch):
    preparar(tmp_path, monkeypatch)
    monkeypatch.setattr(app, "get_openmeteo_locales", lambda: {"medanos": {"temperatura": 22, "viento_vel": 9}})
    m = app._tiempo_actual(LOCS["medanos"])
    assert m["temperatura"] == 22 and m["fuente_modelo"] == "modelo (Open-Meteo)"


def test_sin_nada_no_disponible(tmp_path, monkeypatch):
    preparar(tmp_path, monkeypatch)
    assert app._tiempo_actual(LOCS["medanos"]) == {"disponible": False}
