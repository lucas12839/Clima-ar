"""Velocidad del seguimiento con flujo óptico (24/11/2026).

El centro de una tormenta se corre cuando cambia de forma y daba velocidades
imposibles (tormenta 673 a 102 km/h, máximo 400 km/h). La velocidad sale del
flujo óptico y se suaviza con la anterior; un salto del centro de más de
120 km/h deja el tramo marcado como dudoso.
"""
import math

import numpy as np

import seguimiento_temporal as st

T0 = "2026-11-13T18:00:00Z"
T1 = "2026-11-13T18:10:00Z"


def kmh(vx, vy):
    return st.px_min_a_kmh(math.hypot(vx, vy))


def test_con_flujo_usa_el_flujo_aunque_el_centro_salte():
    flujo = (st.kmh_a_px_min(40.0), 0.0)            # 40 km/h al este
    centro = (st.kmh_a_px_min(150.0), 0.0)          # el centro "saltó" 150 km/h
    vx, vy, dudoso, fuente = st.elegir_velocidad(*centro, flujo)
    assert abs(kmh(vx, vy) - 40.0) < 0.5
    assert dudoso is True
    assert fuente == "FLUJO"


def test_con_flujo_y_centro_coherente_no_es_dudoso():
    flujo = (st.kmh_a_px_min(40.0), 0.0)
    vx, vy, dudoso, _ = st.elegir_velocidad(st.kmh_a_px_min(45.0), 0.0, flujo)
    assert not dudoso
    assert abs(kmh(vx, vy) - 40.0) < 0.5


def test_suaviza_con_la_velocidad_anterior():
    flujo = (st.kmh_a_px_min(60.0), 0.0)
    previa = (st.kmh_a_px_min(30.0), 0.0)
    vx, vy, _, fuente = st.elegir_velocidad(st.kmh_a_px_min(60.0), 0.0, flujo, previa)
    assert abs(kmh(vx, vy) - (0.4 * 60 + 0.6 * 30)) < 0.5
    assert fuente == "FLUJO_SUAVIZADA"


def test_sin_flujo_salto_imposible_mantiene_la_anterior():
    previa = (st.kmh_a_px_min(35.0), 0.0)
    vx, vy, dudoso, fuente = st.elegir_velocidad(st.kmh_a_px_min(200.0), 0.0, None, previa)
    assert (vx, vy) == previa
    assert dudoso and fuente == "ANTERIOR"
    vx, vy, dudoso, fuente = st.elegir_velocidad(st.kmh_a_px_min(200.0), 0.0, None, None)
    assert (vx, vy) == (0.0, 0.0) and dudoso and fuente == "SIN_DATO"


def test_sin_flujo_centro_razonable_se_usa():
    vx, vy, dudoso, fuente = st.elegir_velocidad(st.kmh_a_px_min(50.0), 0.0, None, None)
    assert abs(kmh(vx, vy) - 50.0) < 0.5
    assert not dudoso and fuente == "CENTRO"


def test_velocidad_flujo_nucleo_promedia_el_rectangulo():
    u = np.zeros((100, 100), dtype=np.float32)
    v = np.zeros((100, 100), dtype=np.float32)
    u[40:60, 40:60] = 1.0
    vec = st.velocidad_flujo_nucleo((u, v), {"x": 50, "y": 50, "ancho": 10, "alto": 10})
    assert abs(vec[0] - 1.0) < 1e-6 and vec[1] == 0.0
    assert st.velocidad_flujo_nucleo(None, {"x": 1, "y": 1}) is None


def nucleo(x, y, pixeles=400, ancho=30, alto=20):
    return {"x": x, "y": y, "pixeles": pixeles, "ancho": ancho, "alto": alto,
            "intensidad": 5, "nivel_intensidad": "FUERTE", "estado": "DINAMICO", "confianza": 0.9}


def test_seguimiento_con_flujo_no_da_velocidades_imposibles(monkeypatch):
    """La tormenta cambia de forma: el centro se corre 20 px (113 km/h) pero el
    eco se mueve 5 px en 10 min (28 km/h). Se informa el flujo y queda dudoso
    solo si el centro pasa los 120 km/h."""
    campo = (np.full((512, 512), 0.5, dtype=np.float32), np.zeros((512, 512), dtype=np.float32))
    monkeypatch.setattr(st, "flujo_entre", lambda previo, actual, intervalo: campo)
    monkeypatch.setattr(st, "ruta_frame", lambda ts, actual=None: "previo.png")
    estado = st.estado_nuevo()
    st.procesar_frame_secuencia(estado, [nucleo(200, 200)], T0, None)
    st.procesar_frame_secuencia(estado, [nucleo(220, 200, pixeles=420, ancho=40)], T1, T0,
                                frame_actual="actual.png")
    track = estado["tracks"][0]
    assert abs(st.px_min_a_kmh(track["velocidad_pixeles_min"]) - 28.2) < 1.0
    assert st.px_min_a_kmh(track["velocidad_centro_pixeles_min"]) > 100
    assert track["fuente_velocidad"] == "FLUJO"
    assert track["movimiento_dudoso"] is False
    assert track["direccion"] == "E"


def test_seguimiento_sin_imagenes_funciona_como_antes():
    estado = st.estado_nuevo()
    st.procesar_frame_secuencia(estado, [nucleo(200, 200)], T0, None)
    st.procesar_frame_secuencia(estado, [nucleo(205, 200)], T1, T0)
    track = estado["tracks"][0]
    assert abs(st.px_min_a_kmh(track["velocidad_pixeles_min"]) - 28.2) < 1.0
    assert track["fuente_velocidad"] == "CENTRO"


def _track_con_historial(fuente):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    # El centro salta: 0, 20, 0, 20 px (cambio de forma); el flujo dice 0,5 px/min al este.
    hist = [{"timestamp": f"2026-11-13T18:{m:02d}:00Z", "x": 100 + (20 if i % 2 else 0), "y": 100,
             "pixeles": 300, "intensidad": 5} for i, m in enumerate((0, 10, 20, 30))]
    return {"track_id": 1, "historial": hist, "ultimo_frame": "2026-11-13T18:30:00Z",
            "fuente_velocidad": fuente, "velocidad_x_pixeles_min": 0.5, "velocidad_y_pixeles_min": 0.0}


def test_prediccion_evolucion_usa_la_velocidad_del_flujo():
    import prediccion_evolucion as pe
    con = pe.estimate_motion(_track_con_historial("FLUJO_SUAVIZADA"))
    sin = pe.estimate_motion(_track_con_historial("CENTRO"))
    assert abs(con["vx"] - 0.5) < abs(sin["vx"] - 0.5)
    # Con el historial recortado (prueba hacia atrás) no usa la velocidad de ahora.
    t = _track_con_historial("FLUJO_SUAVIZADA")
    recortado = pe.estimate_motion(t, pe.recent_history(t)[:3])
    t["fuente_velocidad"] = "CENTRO"
    assert recortado["vx"] == pe.estimate_motion(t, pe.recent_history(t)[:3])["vx"]
