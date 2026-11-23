"""Control en vivo del error de la trayectoria del 5b (23/11/2026)."""
from datetime import datetime, timedelta, timezone

import scripts.trayectorias_capas as tc

T0 = datetime(2026, 11, 23, 19, 0, tzinfo=timezone.utc)


def _salida(x, y, t0):
    return {"disponible": True, "tracks": {"7": {"horizontes": [
        {"minutos": m, "timestamp_utc": tc.iso(t0 + timedelta(minutes=m)), "x": x + m, "y": y}
        for m in (5, 10, 20, 30, 60)]}}}


def test_mide_el_error_de_la_misma_tormenta_a_10_min():
    v = tc.verificar({}, T0, [{"track_id": 7, "x": 100.0, "y": 100.0}], _salida(100, 100, T0))
    assert len(v["pendientes"]) == 3 and v["registros"] == []
    t1 = T0 + timedelta(minutes=10)
    # quedó 3 px más al sur de lo pronosticado (110, 100)
    v = tc.verificar(v, t1, [{"track_id": 7, "x": 110.0, "y": 103.0}], {"disponible": False})
    assert len(v["registros"]) == 1
    r = v["registros"][0]
    assert r["horizonte_minutos"] == 10 and abs(r["error_km"] - 3 * tc.KM_POR_PX) < 0.01
    assert v["resumen"]["10"]["muestras"] == 1


def test_un_refresco_de_la_misma_imagen_no_suma_nada():
    v = tc.verificar({}, T0, [{"track_id": 7, "x": 1.0, "y": 1.0}], _salida(1, 1, T0))
    assert tc.verificar(v, T0, [], _salida(1, 1, T0)) is v


def test_otra_tormenta_o_imagen_perdida_no_se_compara():
    v = tc.verificar({}, T0, [{"track_id": 7, "x": 1.0, "y": 1.0}], _salida(1, 1, T0))
    # a los 10 min la 7 ya no está (se deshizo) y aparece la 8 en el mismo lugar
    v = tc.verificar(v, T0 + timedelta(minutes=10), [{"track_id": 8, "x": 11.0, "y": 1.0}], {})
    assert v["registros"] == []
    # se pierde la imagen de los 20 min: ese pronóstico se descarta, no queda colgado
    v = tc.verificar(v, T0 + timedelta(minutes=40), [], {})
    assert v["pendientes"] == [] and v["registros"] == []
