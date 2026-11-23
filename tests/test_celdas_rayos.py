"""Paso 2: celdas seguidas con sus rayos (sin red)."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import celdas_rayos as cr  # noqa: E402

KM = 1.0
SHAPE = (200, 200)


def rayos_en(cx, cy, n=30, dispersion=2.0, edades=None, seed=1):
    rng = np.random.default_rng(seed)
    x = cx + dispersion * rng.standard_normal(n)
    y = cy + dispersion * rng.standard_normal(n)
    e = np.asarray(edades) if edades is not None else rng.uniform(0, 10, n)
    return x, y, e


def test_agrupa_rayos_en_celdas_separadas():
    x1, y1, e1 = rayos_en(50, 50)
    x2, y2, e2 = rayos_en(150, 150, seed=2)
    ruido = (np.array([100.0]), np.array([20.0]), np.array([1.0]))  # un rayo suelto no es celda
    x = np.concatenate([x1, x2, ruido[0]]); y = np.concatenate([y1, y2, ruido[1]])
    e = np.concatenate([e1, e2, ruido[2]])
    c = cr.celdas(x, y, e, SHAPE, KM)
    assert len(c) == 2
    assert sorted(round(k["x"] / 50) for k in c) == [1, 3]


def test_rayos_viejos_no_cuentan():
    x, y, _ = rayos_en(50, 50)
    assert cr.celdas(x, y, np.full(x.size, 15.0), SHAPE, KM) == []


def test_movimiento_propio_de_los_rayos():
    # Rayos viejos (6-10 min) en x=40, nuevos (0-4 min) en x=50: 10 px en ~6 min hacia el este.
    xv, yv, _ = rayos_en(40, 100, n=20, dispersion=0.5, edades=np.full(20, 8.0))
    xn, yn, _ = rayos_en(50, 100, n=20, dispersion=0.5, edades=np.full(20, 2.0))
    c = cr.celdas(np.r_[xv, xn], np.r_[yv, yn], np.r_[np.full(20, 8.0), np.full(20, 2.0)], SHAPE, KM)
    vx, vy = c[0]["movimiento_rayos_px_min"]
    assert vx == pytest.approx(10 / 6, abs=0.2) and abs(vy) < 0.3
    assert c[0]["x"] > 45   # el centro pesa más los rayos nuevos


def test_probabilidad_alta_si_viene_derecho_y_baja_si_pasa_lejos():
    u = np.full(SHAPE, 1.0, dtype=np.float32)   # 1 px/min = 60 km/h hacia el este
    v = np.zeros(SHAPE, dtype=np.float32)
    celda = [{"x": 80.0, "y": 100.0, "radio_km": 3.0, "rayos": 30}]
    locs = {"en_el_camino": (100.0, 100.0, 4.0), "lejos": (100.0, 160.0, 4.0), "atras": (40.0, 100.0, 4.0)}
    p = cr.probabilidades(celda, u, v, KM, locs)
    assert p["en_el_camino"]["probabilidad"] > 0.8
    assert 10 <= p["en_el_camino"]["minutos"] <= 16
    assert p["lejos"]["probabilidad"] < 0.05 and p["atras"]["probabilidad"] < 0.05


def test_celda_del_radar_corregida_con_rayos():
    campo = np.zeros(SHAPE, dtype=np.float32)
    campo[95:105, 45:55] = 50.0                    # núcleo en x=50 en la imagen vieja
    u = np.full(SHAPE, 1.0, dtype=np.float32)
    v = np.zeros(SHAPE, dtype=np.float32)
    locs = {"loc": (100.0, 100.0, 3.0)}
    # Sin rayos: con 20 min de atraso estaría en x=70 -> llega a x=100 en ~25 min.
    sin = cr.probabilidades_celdas(campo, u, v, KM, 20.0, [], locs)
    # Los rayos dicen que ahora está en x=82 (se movió más rápido): llega antes y más seguro.
    con = cr.probabilidades_celdas(campo, u, v, KM, 20.0,
                                   [{"x": 82.0, "y": 100.0, "radio_km": 3.0, "rayos": 40}], locs)
    assert con["loc"]["celdas_corregidas_con_rayos"] == 1
    assert con["loc"]["minutos"] < sin["loc"]["minutos"]
    assert con["loc"]["probabilidad"] >= sin["loc"]["probabilidad"]


def test_celda_de_rayos_sin_nucleo_en_el_radar_entra_como_nueva():
    campo = np.zeros(SHAPE, dtype=np.float32)
    u = np.full(SHAPE, 1.0, dtype=np.float32)
    v = np.zeros(SHAPE, dtype=np.float32)
    r = cr.probabilidades_celdas(campo, u, v, KM, 20.0, [{"x": 90.0, "y": 100.0, "radio_km": 3.0, "rayos": 20}],
                                 {"loc": (100.0, 100.0, 3.0)})
    assert r["loc"]["probabilidad"] > 0.8


def test_sin_tormentas_probabilidad_cero():
    r = cr.probabilidades_celdas(np.zeros(SHAPE, np.float32), None, None, KM, 20.0, [], {"loc": (1.0, 1.0, 3.0)})
    assert r["loc"]["probabilidad"] == 0.0 and r["loc"]["minutos"] is None


def test_verificacion_en_vivo_cuenta_aciertos(monkeypatch):
    import celdas_localidad as cl
    t0 = datetime(2026, 11, 20, 21, 0, tzinfo=timezone.utc)
    zona = np.zeros(SHAPE, dtype=bool)
    zona[90:110, 90:110] = True
    con = np.zeros(SHAPE, np.float32)
    con[95:100, 95:100] = 50.0
    campos = {t0 + timedelta(minutes=m): (con if m == 20 else np.zeros(SHAPE, np.float32)) for m in (0, 10, 20, 30)}
    previo = {"pendientes": [{"ahora_utc": "2026-11-20T21:00:00Z",
                              "localidades": {"a": {"p": 0.8, "pr": 0.0, "b5": True},
                                              "b": {"p": 0.6, "pr": 0.9, "b5": False}}},
                             {"ahora_utc": "2026-11-20T21:20:00Z", "localidades": {"a": {"p": 0.9}}}]}
    out = cl.verificar(previo, campos, {"a": zona, "b": np.zeros(SHAPE, bool)}, t0 + timedelta(minutes=30))
    assert out["acumulado"]["celdas_probable"] == [1, 1, 0, 0]   # a: pasó; b: no pasó
    assert out["acumulado"]["combinado"] == [1, 0, 0, 1]         # solo a tenía el 5b
    assert out["acumulado"]["rayos"] == [0, 1, 1, 0]             # b avisó por rayos y no pasó
    assert len(out["pendientes"]) == 1                          # el de 21:20 todavía no se puede verificar


def test_idea2_solo_fuertes_o_con_rayos():
    """Idea 2: una celda débil (40 dBZ) sin rayos no entra; una fuerte (50 dBZ) o con rayos, sí."""
    u = np.full(SHAPE, 1.0, dtype=np.float32)
    v = np.zeros(SHAPE, dtype=np.float32)
    locs = {"loc": (100.0, 100.0, 3.0)}
    debil = np.zeros(SHAPE, dtype=np.float32)
    debil[95:105, 65:75] = 40.0
    fuerte = debil.copy()
    fuerte[98:102, 68:72] = 50.0
    par = {"solo_fuertes_dbz": 45.0}
    assert cr.probabilidades_celdas(debil, u, v, KM, 10.0, [], locs)["loc"]["probabilidad"] > 0.5
    assert cr.probabilidades_celdas(debil, u, v, KM, 10.0, [], locs, par)["loc"]["probabilidad"] == 0.0
    assert cr.probabilidades_celdas(fuerte, u, v, KM, 10.0, [], locs, par)["loc"]["probabilidad"] > 0.5
    con_rayos = cr.probabilidades_celdas(debil, u, v, KM, 10.0,
                                         [{"x": 80.0, "y": 100.0, "radio_km": 3.0, "rayos": 20}], locs, par)
    assert con_rayos["loc"]["probabilidad"] > 0.5 and con_rayos["loc"]["celdas_corregidas_con_rayos"] == 1


def test_idea2_verificacion_con_su_propia_verdad():
    """El aviso fuerte se valida con 45 dBZ o rayos dentro de la localidad, no con 35 dBZ."""
    import celdas_localidad as cl
    t0 = datetime(2026, 11, 20, 21, 0, tzinfo=timezone.utc)
    zona = np.zeros(SHAPE, dtype=bool)
    zona[90:110, 90:110] = True
    lluvia = np.zeros(SHAPE, np.float32)
    lluvia[95:100, 95:100] = 40.0                       # tormenta débil: no cuenta para la idea 2
    campos = {t0 + timedelta(minutes=m): lluvia for m in (0, 10, 20, 30)}
    previo = {"pendientes": [{"ahora_utc": "2026-11-20T21:00:00Z",
                              "localidades": {"a": {"p": 0.8, "pr": 0.0, "b5": True, "pf": 0.9},
                                              "b": {"p": 0.0, "pr": 0.0, "b5": False, "pf": 0.0}}}],
              "rayos_en_localidad": {"b": ["2026-11-20T21:15:00Z"]}}   # rayos sobre b sin aviso
    out = cl.verificar(previo, campos, {"a": zona, "b": np.zeros(SHAPE, bool)}, t0 + timedelta(minutes=30))
    assert out["acumulado_fuerte"] == [0, 1, 1, 0]   # a: avisó y no fue fuerte; b: rayos sin aviso
    assert out["resumen"]["aviso_fuerte"]["falsas_alarmas_pct"] == 100.0


def test_idea2_registra_rayos_dentro_de_la_localidad():
    import celdas_localidad as cl
    ahora = datetime(2026, 11, 20, 21, 0, tzinfo=timezone.utc)
    goes = {"glm": {"ventana": {"disponible": True, "fin_ventana_utc": "2026-11-20T20:59:00Z",
                                "rayos_10min": [[-38.72, -62.27, 1.0], [-37.0, -60.0, 2.0]]}}}
    locs = [{"id": "bahia", "lat": -38.7183, "lon": -62.2663, "radio_km": 10.0},
            {"id": "tornquist", "lat": -38.1, "lon": -62.2233, "radio_km": 4.0}]
    reg = cl.registrar_rayos({}, goes, locs, ahora)
    assert list(reg) == ["bahia"] and len(reg["bahia"]) == 2
    viejo = cl.registrar_rayos({"rayos_en_localidad": {"bahia": ["2026-11-20T15:00:00Z"]}}, {}, locs, ahora)
    assert viejo == {}                                # más de 3 h: se borra


def test_verdad_usa_lo_observado_limpio_con_el_satelite():
    """1.2: un eco de terreno de la noche (en el PNG, sin nube encima) no cuenta como tormenta
    si al ser la imagen actual el satélite lo había sacado."""
    import celdas_localidad as cl
    t0 = datetime(2026, 11, 21, 3, 0, tzinfo=timezone.utc)
    zona = np.zeros(SHAPE, dtype=bool)
    zona[90:110, 90:110] = True
    terreno = np.zeros(SHAPE, np.float32)
    terreno[95:100, 95:100] = 50.0
    campos = {t0 + timedelta(minutes=m): terreno for m in (0, 10, 20, 30)}
    observado = {cl.iso(t): {"a": 0.0} for t in campos}            # el satélite lo sacó
    pend = [{"ahora_utc": "2026-11-21T03:00:00Z", "localidades": {"a": {"p": 0.0, "pf": 0.0}}}]
    previo = {"pendientes": list(pend), "observado": observado}
    out = cl.verificar(previo, campos, {"a": zona}, t0 + timedelta(minutes=30))
    assert out["acumulado"]["celdas_probable"] == [0, 0, 0, 1]
    assert out["acumulado_fuerte"] == [0, 0, 0, 1]
    sin_obs = cl.verificar({"pendientes": list(pend)}, campos, {"a": zona}, t0 + timedelta(minutes=30))
    assert sin_obs["acumulado"]["celdas_probable"] == [0, 0, 1, 0]   # sin lo guardado, el PNG


def test_motor_muestra_solo_la_tormenta_electrica_encima():
    """Etapa E: de "justo sobre la localidad" se ve solo la tormenta eléctrica (rayos >= 0,7)."""
    import motor_decision_climaar as md
    celdas = {"bahia_blanca": {"probabilidad_solo_rayos": 0.82, "minutos_solo_rayos": 12},
              "punta_alta": {"probabilidad_solo_rayos": 0.5, "minutos_solo_rayos": 5},
              "sierra_de_la_ventana": {"probabilidad_solo_rayos": 0.95, "minutos_solo_rayos": 0}}
    s = md.sobre_la_localidad("bahia_blanca", celdas, True)
    assert s["minutos"] == 12 and s["que"] == "tormenta eléctrica"
    assert md.sobre_la_localidad("punta_alta", celdas, True) is None          # poco probable
    assert md.sobre_la_localidad("sierra_de_la_ventana", celdas, True) is None  # excluida
    assert md.sobre_la_localidad("bahia_blanca", celdas, False) is None       # radar viejo
