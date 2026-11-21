"""Etapa A: compensar el atraso del radar con lo medido después de la imagen
(estaciones, posición y movimiento de los topes de GOES-19, crecimiento medido)."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import corrector_estaciones as ce
import corrector_satelite as cs
import nowcast_capas_nucleo as nc
import nowcast_flujo_optico as nf

SHAPE = (512, 512)
KM_PX = nf.km_por_pixel(512)
T0 = datetime(2026, 11, 20, 22, 0, tzinfo=timezone.utc)


def disco(cx, cy, r, valor, fondo=0.0):
    h, w = SHAPE
    gy, gx = np.ogrid[:h, :w]
    c = np.full(SHAPE, fondo, dtype=np.float32)
    c[(gx - cx) ** 2 + (gy - cy) ** 2 <= r ** 2] = valor
    return c


def centro(campo, umbral=20.0):
    ys, xs = np.nonzero(campo >= umbral)
    return float(xs.mean()), float(ys.mean())


# ---------------------------------------------------------------- estaciones (punto 1)

def test_lluvia_ventana_reinicio_y_lectura_erronea():
    t = lambda hm: datetime(2026, 11, 21, int(hm[:2]), int(hm[3:]), tzinfo=timezone.utc)  # noqa: E731
    # Reinicio a medianoche local (03 UTC): cuenta lo nuevo.
    pts = [(t("02:50"), 6.4), (t("03:05"), 0.4), (t("03:10"), 1.0)]
    assert ce.lluvia_ventana(pts, t("02:58"), t("03:10"))[0] == pytest.approx(1.0)
    # Bajada sin cruzar la medianoche: error de lectura, no 6 mm nuevos.
    pts = [(t("02:34"), 6.0), (t("02:40"), 6.0), (t("02:47"), 0.0), (t("02:50"), 6.0)]
    assert ce.lluvia_ventana(pts, t("02:40"), t("02:50"))[0] == 0.0


def _est(series, coords, dudosas=()):
    return {"series": series, "coords": coords, "dudosas": list(dudosas)}


def _serie(ahora, valores, paso=5):
    n = len(valores)
    return [((ahora - timedelta(minutes=paso * (n - 1 - i))).isoformat(), v) for i, v in enumerate(valores)]


def test_estacion_agrega_lluvia_medida_y_la_mueve():
    ahora = T0 + timedelta(minutes=20)
    lat, lon = -38.72, -62.27
    est = _est({"X": _serie(ahora, [1.0, 1.0, 1.6, 2.4])}, {"X": (lat, lon)})
    campos = {20.0: np.zeros(SHAPE, np.float32), 40.0: np.zeros(SHAPE, np.float32)}
    flujo = {"u": np.full(SHAPE, 1.0, np.float32), "v": np.zeros(SHAPE, np.float32)}
    info = {}
    out = ce.corregir_campos(campos, est, ahora, 20.0, flujo, SHAPE, None, None, info)
    x, y = ce.a_pixel(lat, lon, SHAPE)
    assert out[20.0][int(round(y)), int(round(x))] >= 35.0      # 1,4 mm en 10 min ≈ 8 mm/h
    assert info["agregadas"][0]["estacion"] == "X"
    # 20 min después la lluvia medida se movió 20 px al este.
    assert out[40.0][int(round(y)), int(round(x + 20))] >= 35.0
    assert out[40.0][int(round(y)), int(round(x))] == 0.0


def test_estacion_sin_lluvia_saca_la_tormenta_que_no_esta():
    ahora = T0 + timedelta(minutes=20)
    lat, lon = -38.72, -62.27
    x, y = ce.a_pixel(lat, lon, SHAPE)
    est = _est({"X": _serie(ahora, [3.0, 3.0, 3.0, 3.0, 3.0])}, {"X": (lat, lon)})
    tormenta = disco(x, y, 15, 45.0)
    out = ce.corregir_campos({20.0: tormenta}, est, ahora, 20.0, None, SHAPE, None, tormenta)
    assert out[20.0][int(round(y)), int(round(x))] == 0.0
    assert out[20.0][int(round(y)), int(round(x + 12))] == 45.0   # fuera del disco chico no se toca
    # Estación dudosa o vieja: no se usa.
    dud = _est(est["series"], est["coords"], ["X"])
    assert ce.corregir_campos({20.0: tormenta}, dud, ahora, 20.0, None, SHAPE, None, tormenta)[20.0] is tormenta
    vieja = ce.corregir_campos({20.0: tormenta}, est, ahora + timedelta(minutes=30), 20.0, None, SHAPE, None, tormenta)
    assert vieja[20.0] is tormenta


# ---------------------------------------------------------------- satélite (puntos 2-4)

def _sat(cx, cy, temp=-65.0, sigma=15.0):
    """Tope de tormenta: más frío en el centro (como uno real; un disco parejo no tiene textura)."""
    h, w = SHAPE
    gy, gx = np.ogrid[:h, :w]
    return (10.0 + (temp - 10.0) * np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma ** 2))).astype(np.float32)


def test_posicion_corregida_con_el_corrimiento_de_los_topes():
    """El radar dice 1 px/min al este; los topes midieron 1,5 px/min en los 20 min siguientes:
    ahora la tormenta está donde dicen los topes, no donde suponía el radar."""
    radar = disco(150, 256, 10, 50.0)
    u = np.full(SHAPE, 1.0, np.float32)
    v = np.zeros(SHAPE, np.float32)
    s_radar, s_nuevo = _sat(150, 256), _sat(180, 256)
    info = {}
    prep = cs.preparar(radar, u, v, s_radar, s_nuevo, None, 20.0, None, KM_PX, {"peso_posicion": 1.0}, None, info)
    assert info["posicion"] is True
    x, _ = centro(prep["base"])
    solo_radar = nc.serie(radar, u, v, [20.0])[20.0]
    assert centro(solo_radar)[0] == pytest.approx(170, abs=2)
    # Más adelante que lo que suponía el radar, sin pasarse de lo que midieron los topes.
    assert centro(solo_radar)[0] + 2 < x <= 181
    # La trayectoria sigue el mismo tramo corregido.
    xt, _ = cs.mover_track(150, 256, prep, 20.0)
    assert 172 < xt <= 181


def test_sin_satelite_o_sin_tramo_es_igual_al_radar():
    radar = disco(150, 256, 10, 50.0)
    u = np.full(SHAPE, 1.0, np.float32)
    v = np.zeros(SHAPE, np.float32)
    base = nc.campos_escenarios(radar, u, v, [20.0, 40.0])
    for args in ((None, None, 20.0), (_sat(150, 256), _sat(160, 256), 2.0)):   # sin barrido / tramo corto
        out = cs.pronostico(radar, u, v, [20.0, 40.0], args[0], args[1], None, args[2], None, KM_PX,
                            {"peso_posicion": 1.0})
        assert all(np.array_equal(out[t], base[t]) for t in base)


def test_diferencia_imposible_no_entra():
    radar = disco(150, 256, 10, 50.0)
    u = np.full(SHAPE, 0.5, np.float32)
    v = np.zeros(SHAPE, np.float32)
    # Topes que "saltan" 120 px en 10 min (680 km/h): se acota a la diferencia máxima.
    prep = cs.preparar(radar, u, v, _sat(150, 256), _sat(270, 256), None, 10.0, None, KM_PX,
                       {"peso_posicion": 1.0, "diferencia_max_kmh": 60.0})
    x, _ = centro(prep["base"])
    assert x <= 150 + 10 * (0.5 + 60 / 60 / KM_PX) + 3


def test_crecimiento_medido_sube_y_baja():
    radar = disco(150, 256, 12, 40.0)
    u = np.zeros(SHAPE, np.float32)
    v = np.zeros(SHAPE, np.float32)
    p = {"peso_posicion": 1.0, "dbz_por_c": 0.5, "cambio_max_dbz": 10.0}
    crece = cs.preparar(radar, u, v, _sat(150, 256, -45.0), _sat(150, 256, -65.0), None, 15.0, None, KM_PX, p)
    baja = cs.preparar(radar, u, v, _sat(150, 256, -65.0), _sat(150, 256, -45.0), None, 15.0, None, KM_PX, p)
    assert crece["base"][256, 150] > 40.0 > baja["base"][256, 150]
    assert crece["base"][256, 300] == 0.0   # nunca crea eco donde no hay


# ---------------------------------------------------------------- ciclo completo

@pytest.fixture
def ciclo(tmp_path, monkeypatch):
    import localidades_radar as lr
    import nowcast_capas as ncap

    loc = next(l for l in lr.cargar_localidades() if l["id"] == "bahia_blanca")
    bx, by = nf.latlon_to_pixel(loc["lat"], loc["lon"], 512, 512)
    hist = tmp_path / "historico"
    hist.mkdir()
    campos = {}
    for i, n in enumerate((20, 10, 0)):
        p = hist / f"{T0 - timedelta(minutes=n):%Y%m%d%H%M%S}.png"
        p.write_bytes(b"x")
        campos[p.name] = np.where(disco(bx - 60 + 10 * i, by, 8, 50.0) > 0, 50.0, np.nan).astype(np.float32)
    act = tmp_path / "actual.png"
    act.write_bytes(b"x")
    campos["actual.png"] = campos[f"{T0:%Y%m%d%H%M%S}.png"]
    ts = tmp_path / "radar_timestamp.txt"
    ts.write_text(nf.iso(T0), encoding="utf-8")
    monkeypatch.setattr(nf, "HISTORICO", hist)
    monkeypatch.setattr(nf, "ACTUAL", act)
    monkeypatch.setattr(nf, "TIMESTAMP_FILE", ts)
    monkeypatch.setattr(nf, "cargar_dbz", lambda p: campos.get(p.name))
    monkeypatch.setattr(nc, "cargar_clutter_total", lambda shape: np.zeros(shape, dtype=bool))
    monkeypatch.setattr(nc, "cargar_intermitente", lambda shape: None)
    monkeypatch.setattr(nf, "frame_previo", lambda dt, historico=None: (
        (hist / f"{dt - timedelta(minutes=10):%Y%m%d%H%M%S}.png", dt - timedelta(minutes=10))
        if (hist / f"{dt - timedelta(minutes=10):%Y%m%d%H%M%S}.png").exists() else None))
    for nombre in ("SALIDA", "VERIFICACION", "GOES", "SENALES", "METEO", "CONFIG", "C13_AHORA", "C13_RADAR",
                   "C13_AHORA_PREVIA"):
        monkeypatch.setattr(ncap, nombre, tmp_path / f"{nombre.lower()}.{'npz' if nombre.startswith('C13') else 'json'}")
    return {"ncap": ncap, "tmp": tmp_path, "bx": bx, "by": by, "campos": campos}


def _guardar_sat(path, cx, cy, hora, temp=-65.0):
    """Barrido sintético: tope frío en el píxel (cx, cy) de la grilla del radar."""
    import corrector_tiempo_real as ct
    lat, lon = np.meshgrid(np.linspace(-41, -36.5, 300), np.linspace(-65.5, -59, 300), indexing="ij")
    x, y = ct.a_pixel(lat.ravel(), lon.ravel(), 512, 512)
    t = (10.0 + (temp - 10.0) * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 15.0 ** 2))).astype(np.float32)
    np.savez(path, lat=lat.ravel().astype(np.float32), lon=lon.ravel().astype(np.float32), temp_c=t,
             hora=np.array(nf.iso(hora)))


def test_ciclo_con_satelite_activo_y_sin_actividad(ciclo, monkeypatch):
    import corrector_tiempo_real as ct
    ncap = ciclo["ncap"]
    monkeypatch.setattr(ct, "desplazamiento_paralaje_km_por_km", lambda: (0.0, 0.0))
    monkeypatch.setattr(ct, "_PARALAJE", None)
    ncap.CONFIG.write_text(json.dumps({"satelite_etapa_a": {"activo": True, "parametros": {"peso_posicion": 1.0}}}),
                           encoding="utf-8")
    ahora = T0 + timedelta(minutes=22)
    bx, by = ciclo["bx"], ciclo["by"]
    _guardar_sat(ncap.C13_RADAR, bx - 40, by, T0)
    _guardar_sat(ncap.C13_AHORA, bx - 40 + 27, by, T0 + timedelta(minutes=18))
    _guardar_sat(ncap.C13_AHORA_PREVIA, bx - 40 + 12, by, T0 + timedelta(minutes=8))
    s = ncap.ejecutar(ahora)
    info = s["satelite_etapa_a"]
    assert info["usado"] is True and info["posicion"] is True
    assert info["tramo_min"] == pytest.approx(18.0, abs=0.1)
    # Sin barridos: falla segura, como antes.
    for p in (ncap.C13_RADAR, ncap.C13_AHORA, ncap.C13_AHORA_PREVIA):
        p.unlink()
    s2 = ncap.ejecutar(ahora)
    assert s2["satelite_etapa_a"]["usado"] is False
    assert "localidades" in s2
    # Sin actividad: radar vacío, nada se rompe.
    for k in list(ciclo["campos"]):
        ciclo["campos"][k] = np.full(SHAPE, np.nan, np.float32)
    _guardar_sat(ncap.C13_AHORA, bx, by, T0 + timedelta(minutes=18))
    s3 = ncap.ejecutar(ahora)
    assert s3["movimiento"] is None and "localidades" in s3


def test_trayectoria_con_satelite(ciclo, monkeypatch):
    import trayectorias_capas as tc
    track = {"track_id": 7, "x": ciclo["bx"] - 40, "y": ciclo["by"], "pixeles": 200}
    sin = tc.calcular(T0, [track])
    assert sin["satelite"]["usado"] is False
    sat = {"radar": _sat(ciclo["bx"] - 40, ciclo["by"]), "nuevo": _sat(ciclo["bx"] - 40 + 27, ciclo["by"]),
           "previo": None, "tramo": 18.0, "entre": None}
    con = tc.calcular(T0, [track], satelite=sat, parametros_satelite={"peso_posicion": 1.0})
    assert con["satelite"]["usado"] is True
    h_sin = {h["minutos"]: h["x"] for h in sin["tracks"]["7"]["horizontes"]}
    h_con = {h["minutos"]: h["x"] for h in con["tracks"]["7"]["horizontes"]}
    # Radar: 1 px/min; topes: 1,5 px/min en los 18 min medidos -> hasta 9 px más adelante.
    assert 1.5 < h_con[20] - h_sin[20] <= 10
