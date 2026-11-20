"""Punto 8: ambiente medido por el satélite (cálculos y ciclo, sin red)."""
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from goes19 import ambiente as amb  # noqa: E402
import ambiente_satelite as ams  # noqa: E402

FRAME = datetime(2026, 11, 20, 20, 20, tzinfo=timezone.utc)
PRES = np.array([1100.0, 1013.9, 1000.0, 925.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 200.0, 100.0])


def atmosfera_estandar(t0_c=20.0, gradiente=6.5):
    """Temperatura (K) por nivel con gradiente constante por km."""
    t = []
    for p in PRES:
        z = 287.04 * 288.0 / 9.80665 * math.log(1000.0 / p) / 1000.0  # km aprox
        t.append(273.15 + t0_c - gradiente * z)
    return np.array(t)


def test_perfil_gradiente_y_nivel_cero():
    t = atmosfera_estandar(20.0, 6.5)
    hr = np.full(PRES.size, 50.0)
    d = amb.derivados_perfil(PRES, t, hr)
    assert 6.0 <= d["gradiente_700_500_c_km"] <= 7.5  # perfil sintético aproximado
    # 20 °C con 6,5 °C/km -> 0 °C cerca de 3 km
    assert 2800 <= d["nivel_0c"]["altura_m"] <= 3300
    assert d["humedad_700_pct"] == 50 and d["humedad_media_700_500_pct"] == 50
    assert d["rocio_850_c"] < d["temperatura_850_c"]


def test_perfil_ignora_niveles_bajo_el_suelo():
    t = atmosfera_estandar(20.0, 6.5)
    t[0] = 400.0  # 1100 hPa: extrapolado, absurdo
    assert amb.derivados_perfil(PRES, t, None)["nivel_0c"]["altura_m"] < 3300


def test_rocio():
    assert amb.rocio_c(20.0, 100.0) == pytest.approx(20.0, abs=0.01)
    assert amb.rocio_c(20.0, 50.0) == pytest.approx(9.3, abs=0.2)
    assert amb.rocio_c(None, 50.0) is None


def ventanas_sinteticas(n=40, despejados=30, cape=800.0, tpw=30.0):
    lat = np.full(n, -38.72) + np.linspace(-0.05, 0.05, n)
    lon = np.full(n, -62.27)
    dqf = np.full(n, 4.0)
    dqf[:despejados] = 0.0
    nan = np.full(n, np.nan)

    def ventana(datos):
        datos = {k: np.where(dqf == 0, v, np.nan) if v.ndim == 1 else
                 np.where((dqf == 0)[:, None], v, np.nan) for k, v in datos.items()}
        return {"datos": datos, "dqf": dqf.copy(), "latitude": lat, "longitude": lon,
                "distance_km": np.full(n, 10.0), "presion_hpa": PRES}

    perfil_t = np.tile(atmosfera_estandar(22.0, 7.8), (n, 1))
    perfil_h = np.tile(np.array([90, 90, 90, 90, 90, 30, 25, 20, 20, 20, 20, 20.0]), (n, 1))
    return {
        "inestabilidad": ventana({"CAPE": np.full(n, cape), "LI": np.full(n, -3.0), "KI": np.full(n, 30.0),
                                  "TT": np.full(n, 52.0), "SI": np.full(n, -1.0)}),
        "agua_precipitable": ventana({"TPW": np.full(n, tpw)}),
        "perfil_temperatura": ventana({"LVT": perfil_t}),
        "perfil_humedad": ventana({"LVM": perfil_h}),
    }, nan


def test_localidad_con_cielo_despejado():
    v, _ = ventanas_sinteticas()
    r = amb.por_localidades(v, [{"id": "bahia_blanca", "lat": -38.72, "lon": -62.27}], 25.0)["bahia_blanca"]
    assert r["cape_jkg"] == 800 and r["agua_precipitable_mm"] == 30.0
    assert r["pixeles_despejados"]["inestabilidad"] == 30
    lec = r["lectura"]
    assert lec["inestabilidad"] == "moderada"
    assert lec["agua_en_la_columna"] == "alta"
    assert lec["capas_medias"] == "muy_inestables"
    assert lec["aire_seco_en_altura"] == "si"   # húmedo abajo, seco en 700-500


def test_nublado_no_inventa_valores():
    v, _ = ventanas_sinteticas(despejados=0)
    r = amb.por_localidades(v, [{"id": "x", "lat": -38.72, "lon": -62.27}], 25.0)["x"]
    assert "cape_jkg" not in r and "agua_precipitable_mm" not in r and "perfil" not in r
    assert r["lectura"] == {}


def test_region_mas_inestable():
    v, _ = ventanas_sinteticas()
    v["inestabilidad"]["datos"]["CAPE"][5] = 2500.0
    r = amb.region(v)
    assert r["mas_inestable"]["cape_jkg"] == 2500
    assert r["fraccion_despejada"] == pytest.approx(0.75)


def test_memoria_ultima_medicion_despejada():
    hora = FRAME - timedelta(minutes=8)
    medido = ams.con_memoria({"cape_jkg": 900.0, "agua_precipitable_mm": 31.0}, None, hora, FRAME)
    assert medido["origen"]["inestabilidad"] == "ahora"
    # una hora después se nubló: se usa la última medición despejada con su edad
    luego = FRAME + timedelta(minutes=60)
    nublado = ams.con_memoria({}, medido, luego, luego)
    assert nublado["valores"]["cape_jkg"] == 900.0
    assert nublado["origen"]["inestabilidad"].startswith("ultima_medicion_despejada_hace_68")
    assert nublado["ultimo_despejado"]["inestabilidad"]["edad_min"] == 68
    # más de 6 h: ya no representa el ambiente
    tarde = FRAME + timedelta(hours=7)
    assert ams.con_memoria({}, nublado, tarde, tarde)["valores"] == {"pixeles_despejados": {}}


def test_memoria_nunca_usa_dato_posterior_al_radar():
    futuro = {"ultimo_despejado": {"inestabilidad": {"hora_utc": "2026-11-20T21:00:00Z",
                                                     "valores": {"cape_jkg": 1.0}}}}
    assert "cape_jkg" not in ams.con_memoria({}, futuro, FRAME, FRAME)["valores"]


def test_comparacion_con_modelo():
    medido = {"cape_jkg": 300.0, "agua_precipitable_mm": 28.0,
              "perfil": {"temperatura_500_c": -17.0, "nivel_0c": {"altura_m": 3200}}}
    modelo = {"tiempo_utc": "2026-11-20T20:00", "cape": 900.0, "total_column_integrated_water_vapour": 21.0,
              "temperature_500hPa": -20.5, "freezing_level_height": 2600}
    c = ams.comparar(medido, modelo)
    assert c["variables"]["agua_precipitable_mm"]["diferencia"] == 7.0
    assert c["variables"]["nivel_0c_m"]["diferencia"] == 600.0
    texto = " ".join(c["lectura"])
    assert "más agua" in texto and "menos energía" in texto and "más cálido" in texto
    assert ams.comparar(medido, None)["estado"] == "sin_modelo"


def test_registro_modelo_hora_mas_cercana():
    meteo = {"ambiente_futuro": {"puntos": [{"punto": "BB_CENTRO", "datos": [
        {"tiempo_utc": "2026-11-20T19:00", "cape": 1}, {"tiempo_utc": "2026-11-20T20:00", "cape": 2},
        {"tiempo_utc": "2026-11-20T21:00", "cape": 3}]}]}}
    assert ams.registro_modelo(meteo, FRAME)["cape"] == 2
    assert ams.registro_modelo(meteo, FRAME + timedelta(hours=5)) is None


def _calcular(monkeypatch, v, estado_ok=True, previo=None):
    archivos = {k: {"obj": {"key": f"x/{k}_{FRAME:%H%M}.nc", "url": "u", "start_time_utc": "2026-11-20T20:10:21Z",
                            "end_time_utc": "2026-11-20T20:19:52Z"}, "mismo_barrido_que_c13": True}
                for k in amb.PRODUCTOS}
    monkeypatch.setattr(ams, "buscar_archivos", lambda frame, inicio: archivos)
    llamadas = []

    def leer(origen, variables):
        llamadas.append(variables)
        if not estado_ok:
            raise OSError("NOAA no responde")
        return next(x for k, x in v.items() if amb.PRODUCTOS[k][1] == variables)

    monkeypatch.setattr(amb, "leer_ventana", leer)
    locs = [{"id": "bahia_blanca", "nombre": "Bahía Blanca", "lat": -38.72, "lon": -62.27}]
    meteo = {"ambiente_futuro": {"puntos": [{"punto": "BB_CENTRO", "datos": [
        {"tiempo_utc": "2026-11-20T20:00", "cape": 500.0, "total_column_integrated_water_vapour": 25.0}]}]}}
    return ams.calcular(FRAME, "2026-11-20T20:10:21Z", previo, meteo, locs), llamadas


def test_ciclo_completo_y_refresco_sin_bajar_de_nuevo(monkeypatch):
    v, _ = ventanas_sinteticas()
    salida, llamadas = _calcular(monkeypatch, v)
    assert salida["disponible"] and salida["en_paralelo"]
    assert all(p["estado"] == "ok" for p in salida["productos"].values())
    assert all(p["estado"] == "ok" for p in salida["productos"].values())
    bb = salida["localidades"]["bahia_blanca"]
    assert bb["valores"]["cape_jkg"] == 800 and bb["origen"]["perfiles"] == "ahora"
    assert salida["comparacion_modelo_bahia"]["variables"]["agua_precipitable_mm"]["diferencia"] == 5.0
    assert len(llamadas) == 4
    # refresco entre imágenes: mismos archivos -> no se lee nada
    salida2, llamadas2 = _calcular(monkeypatch, v, previo=salida)
    assert salida2["reutilizado"] is True and llamadas2 == []
    json.dumps(ams.limpiar(salida), allow_nan=False)


def test_satelite_caido_conserva_lo_ultimo_despejado(monkeypatch):
    v, _ = ventanas_sinteticas()
    bueno, _ = _calcular(monkeypatch, v)
    bueno["claves_archivos"] = {}   # otro ciclo
    caido, _ = _calcular(monkeypatch, v, estado_ok=False, previo=bueno)
    assert caido["disponible"] is False and "sin productos" in caido["motivo"]
    assert all(p["estado"] == "error" for p in caido["productos"].values())
    assert caido["localidades"]["bahia_blanca"]["valores"]["cape_jkg"] == 800  # memoria


def test_producto_viejo_no_se_usa(monkeypatch):
    from goes19 import por_tormenta as pt
    v, _ = ventanas_sinteticas()
    archivos = {k: {"obj": {"key": f"x/{k}.nc", "url": "u", "end_time_utc": "2026-11-20T19:40:00Z"}}
                for k in amb.PRODUCTOS}
    ventanas, estado = ams.leer_productos(archivos, FRAME)
    assert ventanas == {} and all(e["estado"] == "no_sincronizado" for e in estado.values())
    assert pt.DESFASE_MAX_MIN == 20.0


def test_main_nunca_corta_el_ciclo(tmp_path, monkeypatch):
    monkeypatch.setattr(ams, "SALIDA", tmp_path / "a.json")
    monkeypatch.setattr(ams, "RADAR_TS", tmp_path / "no_existe.txt")
    assert ams.main() == 0
    d = json.loads((tmp_path / "a.json").read_text())
    assert d["disponible"] is False and d["en_paralelo"] is True
