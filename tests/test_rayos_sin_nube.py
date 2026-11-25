"""Rayo sin nube de tormenta = falso (25/11/2026: 'Rayos aislados' en Médanos con cielo despejado)."""
import numpy as np

from goes19 import glm_ventana as gv


def grilla(temp_fria_en=None):
    lat, lon = np.meshgrid(np.linspace(-40, -37.5, 120), np.linspace(-64, -60, 160), indexing="ij")
    temp = np.full(lat.shape, 12.0)           # cielo despejado (suelo tibio)
    if temp_fria_en:
        la, lo = temp_fria_en
        cerca = (np.abs(lat - la) < 0.08) & (np.abs(lon - lo) < 0.1)
        temp[cerca] = -45.0                    # tope de tormenta
    return {"lat": lat, "lon": lon, "temp_c": temp}


def test_cielo_despejado_saca_el_rayo():
    # Caso real: rayo de las 6:41 junto a Médanos, sin nubes.
    ok = gv.con_nube(np.array([-38.846]), np.array([-62.669]), grilla())
    assert not ok[0]


def test_rayo_bajo_tormenta_queda():
    c13 = grilla(temp_fria_en=(-38.9, -62.7))
    ok = gv.con_nube(np.array([-38.846, -39.6]), np.array([-62.669, -59.64]), c13)
    assert ok.tolist() == [True, False]        # el segundo está lejos de la nube


def test_sin_imagen_no_se_puede_controlar():
    assert gv.con_nube(np.array([-38.8]), np.array([-62.6]), None).tolist() == [True]


def test_filtrar_saca_de_todos_los_campos():
    res = [{"lat": np.array([-38.846, -38.9]), "lon": np.array([-62.669, -62.7]),
            "area_km2": np.array([50.0, 80.0]), "energia_fj": np.array([10.0, 20.0])}]
    sacados = gv.filtrar_sin_nube(res, grilla(temp_fria_en=(-38.9, -62.7)))
    assert sacados == 0
    sacados = gv.filtrar_sin_nube(res, grilla())
    assert sacados == 2 and res[0]["lat"].size == res[0]["area_km2"].size == 0
