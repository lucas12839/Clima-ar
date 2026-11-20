"""Punto 3: salto de rayos (2σ), asignación exclusiva y tamaño de rayos."""
import numpy as np
import pytest

from goes19 import rayos_tormenta as rt


def _edades(tasas_por_min):
    """Edades de rayos para tramos de 2 min (del más viejo al más reciente)."""
    edades = []
    for k, tasa in enumerate(tasas_por_min):
        hasta = 14 - 2 * k
        n = int(round(tasa * 2))
        edades += list(np.linspace(hasta - 1.9, hasta - 0.1, n)) if n else []
    return np.array(edades)


def test_salto_clasico():
    r = rt.salto_2sigma(_edades([2, 3, 2, 3, 4, 5, 16]), 14.0)
    assert r["estado"] == "ok"
    assert r["tasas_por_min"] == [2, 3, 2, 3, 4, 5, 16]
    assert r["dfrdt_actual"] == 5.5
    assert r["salto"] is True
    assert r["nivel_sigma"] > 2


def test_sin_salto_con_frecuencia_alta_pero_pareja():
    r = rt.salto_2sigma(_edades([20, 22, 19, 21, 20, 22, 21]), 14.0)
    assert r["salto"] is False


def test_aumento_brusco_pero_debajo_de_10_por_minuto_no_es_salto():
    r = rt.salto_2sigma(_edades([0.5, 1, 0.5, 1, 0.5, 1, 6]), 14.0)
    assert r["salto"] is False
    assert r["nivel_sigma"] > 2  # se informa para calibrar con casos reales


def test_sin_cobertura_suficiente_no_se_calcula():
    assert rt.salto_2sigma(_edades([1] * 7), 9.0)["estado"] == "cobertura_insuficiente"


def _tormenta(lat, lon, radio=10.0, este=0.0, norte=0.0):
    return {"latitud": lat, "longitud": lon, "radio_km": radio, "este_km_min": este, "norte_km_min": norte}


def test_cada_rayo_cuenta_para_una_sola_tormenta():
    a = _tormenta(-38.7, -62.30)
    b = _tormenta(-38.7, -62.10)   # a ~17 km: los radios de 10 km se superponen
    lon = np.array([-62.29, -62.21, -62.19, -62.11, -61.50])
    rayos = {"minutos": np.full(5, 1.0), "latitude": np.full(5, -38.7), "longitude": lon}
    asignado = rt.asignar(rayos, [a, b])
    assert asignado.tolist() == [0, 0, 1, 1, -1]


def test_asignacion_sigue_el_movimiento_y_no_usa_rayos_del_futuro():
    t = _tormenta(-38.7, -62.3, radio=8.0, este=1.0)   # va al este a 1 km/min
    lon_hace_12 = -62.3 - 12.0 / (111.195 * np.cos(np.radians(-38.7)))
    rayos = {"minutos": np.array([12.0, 12.0, -1.0, 15.0]),
             "latitude": np.full(4, -38.7), "longitude": np.array([lon_hace_12, -62.3, -62.3, -62.3])}
    assert rt.asignar(rayos, [t]).tolist() == [0, -1, -1, -1]


def test_tamano_de_rayos():
    r = rt.tamano(np.array([120.0, 250.0, 900.0, np.nan]), np.array([10.0, 20.0, 30.0, 40.0]))
    assert r["cantidad"] == 3 and r["area_minima_km2"] == 120.0
    assert r["fraccion_chicos"] == pytest.approx(0.667, abs=0.001)
    assert r["energia_total_fj"] == 100.0
    assert rt.tamano(np.array([]), np.array([])) == {"cantidad": 0}


def test_por_tormenta_completo():
    t = _tormenta(-38.7, -62.3)
    edades = _edades([2, 3, 2, 3, 4, 5, 16])
    n = edades.size
    rayos = {"cobertura_min": 14.0, "ventana_min": 14.0, "minutos": edades,
             "latitude": np.full(n, -38.7), "longitude": np.full(n, -62.3),
             "area_km2": np.where(edades < 2, 150.0, 800.0), "energia_fj": np.ones(n)}
    r = rt.por_tormenta(rayos, rt.asignar(rayos, [t]), 0, "2026-11-19T23:30:00Z")
    assert r["estado"] == "ok" and r["fin_utc"] == "2026-11-19T23:30:00Z"
    assert r["cobertura_min"] == 10.0
    assert r["cantidad"] == int(np.sum(edades < 10))
    assert r["salto_rayos"]["salto"] is True
    assert r["tamano_rayos"]["area_minima_km2"] == 150.0
    assert rt.por_tormenta(None, np.zeros(0, int), 0, None)["estado"] == "no_disponible"
