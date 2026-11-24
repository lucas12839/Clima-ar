"""División y unión de tormentas: el sistema conserva movimiento y tamaño."""
import math

import seguimiento_temporal as st

T0 = "2026-11-23T20:00:00Z"
T1 = "2026-11-23T20:10:00Z"


def nucleo(x, y, pixeles, lado=20):
    return {"x": x, "y": y, "pixeles": pixeles, "ancho": lado, "alto": lado,
            "intensidad": 4, "nivel_intensidad": "MODERADO", "estado": "DINAMICO", "confianza": 0.9}


def track_en_movimiento(tid, x, y, pixeles, vx=1.0, vy=0.0, lado=20):
    t = st.crear_track(tid, nucleo(x, y, pixeles, lado), T0)
    t["historial"] = [
        {"timestamp": "2026-11-23T19:50:00Z", "x": x - vx * 10, "y": y - vy * 10, "pixeles": pixeles,
         "ancho": lado, "alto": lado, "intensidad": 4},
        t["historial"][0],
    ]
    t["velocidad_x_pixeles_min"], t["velocidad_y_pixeles_min"] = vx, vy
    t["velocidad_pixeles_min"] = math.hypot(vx, vy)
    return t


def test_division_limpia_conserva_movimiento_y_tamano():
    padre = track_en_movimiento(1, 100, 100, 400, lado=30)
    # Se movió 10 px al este y se partió en dos mitades iguales (norte y sur).
    a, b = nucleo(110, 95, 200, 15), nucleo(110, 105, 200, 15)
    ajuste = st.ajustar_division(padre, a, b, 10.0)
    assert ajuste is not None
    st.actualizar_track(padre, a, T1, 10.0, "ACEPTABLE")
    assert abs(padre["velocidad_x_pixeles_min"] - 1.0) < 1e-6
    assert abs(padre["velocidad_y_pixeles_min"]) < 1e-6
    assert abs(padre["cambio_area_porcentaje"]) < 1e-6  # no "se debilita"
    hijo = st.crear_track(2, b, T1)
    st.heredar_division(hijo, padre, ajuste)
    assert hijo["rumbo_grados"] == padre["rumbo_grados"]
    assert len(hijo["historial"]) == 3
    assert hijo["historial"][-1]["x"] == 110 and hijo["historial"][-1]["y"] == 105
    assert abs(hijo["historial"][-2]["y"] - 105) < 1e-6  # su pasado, en su lugar
    assert hijo["primer_frame"] == T1  # el número nuevo no "nace antes"


def test_pedazo_que_cae_afuera_no_es_division():
    padre = track_en_movimiento(1, 100, 100, 400, lado=20)
    a, b = nucleo(110, 100, 380, 20), nucleo(160, 100, 100, 10)
    assert st.ajustar_division(padre, a, b, 10.0) is None
    assert padre["x"] == 100 and padre["pixeles"] == 400  # no se tocó


def test_union_limpia_conserva_movimiento_y_tamano():
    p = track_en_movimiento(1, 100, 100, 300, lado=20)
    f = track_en_movimiento(2, 100, 110, 300, lado=20)
    nuevo = nucleo(110, 105, 600, 30)
    assert st.ajustar_union(p, [f], nuevo, 10.0) is True
    st.actualizar_track(p, nuevo, T1, 10.0, "ACEPTABLE")
    assert abs(p["velocidad_x_pixeles_min"] - 1.0) < 1e-6
    assert abs(p["velocidad_y_pixeles_min"]) < 1e-6  # sin salto al sur
    assert abs(p["cambio_area_porcentaje"]) < 1e-6  # no "crece"


def test_union_que_no_cierra_no_se_ajusta():
    p = track_en_movimiento(1, 100, 100, 300)
    f = track_en_movimiento(2, 100, 110, 300)
    chico = nucleo(110, 105, 200, 20)  # quedó un tercio de la suma
    assert st.ajustar_union(p, [f], chico, 10.0) is False
    assert p["x"] == 100 and p["pixeles"] == 300


def test_secuencia_sin_actividad_y_con_division():
    estado = st.estado_nuevo()
    assert st.procesar_frame_secuencia(estado, [], T0, None)["nucleos_actuales"] == 0
    st.procesar_frame_secuencia(estado, [nucleo(100, 100, 400, 30)], T0, None)
    salida = st.procesar_frame_secuencia(
        estado, [nucleo(110, 95, 200, 15), nucleo(110, 105, 200, 15)], T1, T0)
    assert salida["nucleos_actuales"] == 2
    ids = [t["track_id"] for t in estado["tracks"]]
    assert len(ids) == len(set(ids))
