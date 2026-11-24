"""El tiempo ahora: lo medido en Bahía primero, el modelo de relleno (24/11/2026)."""
import json
import os
from datetime import datetime, timedelta, timezone

os.environ.setdefault("CLIMAAR_REMOTE_DATA_ENABLED", "0")
import app  # noqa: E402

MODELO = {"temperatura": 24, "viento_vel": 30, "viento_dir": 220, "rafagas": 47, "humedad": 20,
          "presion": 1010, "nubosidad": 10, "codigo_tiempo": 1, "sensacion_termica": 23, "precipitacion": 0}


def preparar(tmp_path, monkeypatch, bcp=None, cwop=None, sazb=None, modelo=MODELO):
    obs = tmp_path / "obs"
    obs.mkdir()
    (obs / "bcp_actual.json").write_text(json.dumps({"stations": bcp or []}))
    (obs / "cwop_actual.json").write_text(json.dumps({"stations": cwop or []}))
    meteo = tmp_path / "meteo.json"
    meteo.write_text(json.dumps({"observacion_sazb": sazb or {}, "ambiente_actual": {}}))
    monkeypatch.setattr(app, "OBS_DIR", obs)
    monkeypatch.setattr(app, "METEO_FILE", meteo)
    monkeypatch.setattr(app, "get_openmeteo_data", lambda: modelo)


def hace(minutos):
    return (datetime.now(timezone.utc) - timedelta(minutes=minutos)).isoformat()


CENTRO = {"station_id": 104, "observation_time": None,
          "variables": {"temperatura_c": 22.5, "viento_kmh": 23.0, "direccion_viento": "Sudsudoeste",
                        "rafaga_15min_kmh": None}}
NOROESTE = {"indicativo": "LU4DRH-11", "observation_time": None,
            "variables": {"temperatura_c": 23.9, "viento_kmh": 14.5, "direccion_viento_grados": 229.0,
                          "rafaga_15min_kmh": 20.9, "humedad_relativa_pct": 19.0, "lluvia_1h_mm": 0.0}}


def test_medido_gana_al_modelo(tmp_path, monkeypatch):
    preparar(tmp_path, monkeypatch, bcp=[dict(CENTRO, observation_time=hace(5))],
             cwop=[dict(NOROESTE, observation_time=hace(3))])
    t = app._tiempo_actual()
    assert t["temperatura"] == 22.5                       # Centro
    # 24/11/2026: el viento sale del Centro (primero en el orden); la ráfaga de
    # la Noroeste no se mezcla con el viento de otra estación.
    assert (t["viento_vel"], t["rafagas"]) == (23, None)
    assert "Centro" in t["fuentes"]["viento"] and "rafagas" not in t["fuentes"]
    assert t["humedad"] == 19
    assert t["presion"] == 1010                           # sin aeropuerto: modelo
    assert t["sensacion_termica"] is None                 # la del modelo no va con temperatura medida
    assert "presion" not in t["fuentes"]


def test_rafaga_menor_que_el_viento_no_se_muestra(tmp_path, monkeypatch):
    centro = dict(CENTRO, observation_time=hace(3),
                  variables=dict(CENTRO["variables"], viento_kmh=20.0, rafaga_15min_kmh=12.0))
    preparar(tmp_path, monkeypatch, bcp=[centro])
    t = app._tiempo_actual()
    assert (t["viento_vel"], t["rafagas"]) == (20, None)


def test_viento_de_estaciones_cwop_no_se_usa(tmp_path, monkeypatch):
    # Caso real 24/11/2026: Noroeste 3,2 km/h con 33 km/h en el Centro (reparada).
    nor = dict(NOROESTE, observation_time=hace(3),
               variables=dict(NOROESTE["variables"], viento_kmh=3.2, rafaga_15min_kmh=3.2))
    preparar(tmp_path, monkeypatch, cwop=[nor])
    t = app._tiempo_actual()
    assert (t["viento_vel"], t["rafagas"]) == (30, 47)          # modelo
    assert t["temperatura"] == 23.9 and "Noroeste" in t["fuentes"]["temperatura"]


def test_viento_medido_sin_rafaga_no_muestra_la_del_modelo(tmp_path, monkeypatch):
    preparar(tmp_path, monkeypatch, bcp=[dict(CENTRO, observation_time=hace(5))])
    t = app._tiempo_actual()
    assert t["viento_vel"] == 23 and t["viento_dir"] == 202.5
    assert t["rafagas"] is None


def test_estaciones_viejas_no_entran(tmp_path, monkeypatch):
    preparar(tmp_path, monkeypatch, bcp=[dict(CENTRO, observation_time=hace(40))],
             cwop=[dict(NOROESTE, observation_time=hace(45))])
    t = app._tiempo_actual()
    assert (t["temperatura"], t["viento_vel"], t["rafagas"]) == (24, 30, 47)
    assert t["fuentes"] == {}


def test_aeropuerto_para_presion(tmp_path, monkeypatch):
    preparar(tmp_path, monkeypatch, sazb={"disponible": True, "obsTime_utc": hace(50), "temperatura_c": 21,
                                          "viento_kt": 16, "direccion_viento": 210, "presion_hpa": 1011})
    t = app._tiempo_actual()
    assert t["presion"] == 1011 and t["viento_vel"] == 30 and t["rafagas"] is None


def test_sin_nada(tmp_path, monkeypatch):
    preparar(tmp_path, monkeypatch, modelo={})
    assert app._tiempo_actual() == {"disponible": False}
