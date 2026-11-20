"""Corrector del pronóstico con rayos y satélite (datos medidos mientras falta el radar)."""
from datetime import datetime, timedelta, timezone

import numpy as np

import corrector_tiempo_real as ct

SHAPE = (512, 512)


def latlon_de_pixel(x, y):
    cx = cy = 255.5
    lon = ct.RADAR_LON + (x - cx) / cx * ct.RADAR_RADIUS_KM / (111.320 * np.cos(np.radians(ct.RADAR_LAT)))
    lat = ct.RADAR_LAT - (y - cy) / cy * ct.RADAR_RADIUS_KM / 110.574
    return lat, lon


def test_quita_lluvia_donde_el_satelite_no_ve_nube():
    campo = np.zeros(SHAPE, np.float32)
    campo[100:120, 100:120] = 30.0           # lluvia pronosticada (por ejemplo, prolongada por ANVIL)
    sat = np.full(SHAPE, 10.0, np.float32)   # cielo sin nube alta: tope tibio
    c, cambios = ct.corregir(campo, sat, [], [])
    assert c.max() == 0.0 and cambios["quitado_sin_nube"] == 400


def test_no_quita_si_hay_nube_fria_cerca():
    campo = np.zeros(SHAPE, np.float32)
    campo[100:120, 100:120] = 30.0
    sat = np.full(SHAPE, 10.0, np.float32)
    sat[95:125, 95:125] = -45.0
    c, _ = ct.corregir(campo, sat, [], [])
    assert (c[100:120, 100:120] == 30.0).all()


def test_rayos_solos_son_tormenta_electrica_no_lluvia():
    campo = np.zeros(SHAPE, np.float32)
    lat, lon = latlon_de_pixel(300.0, 300.0)
    # Rayos con tope tibio (o sin satélite): NO se inventa lluvia, solo zona eléctrica.
    for sat in (None, np.full(SHAPE, -10.0, np.float32)):
        c, cambios = ct.corregir(campo, sat, [lat, lat], [lon, lon])
        assert c.max() == 0.0 and not cambios.get("agregado_con_rayos")
    assert ct.zona_electrica([lat], [lon], SHAPE)[300, 300]
    # Rayos con tope frío encima: tormenta que llueve.
    c, cambios = ct.corregir(campo, np.full(SHAPE, -55.0, np.float32), [lat, lat], [lon, lon])
    assert c[300, 300] == ct.REGLAS_DEFECTO["dbz_rayos"] and cambios["agregado_con_rayos"] > 0
    # Sin satélite (falta el dato) nunca se borra nada.
    campo[10:20, 10:20] = 30.0
    c, cambios = ct.corregir(campo, None, [], [])
    assert (c[10:20, 10:20] == 30.0).all() and "quitado_sin_nube" not in cambios


def test_paralaje_corre_la_nube_hacia_el_norte_y_oeste():
    dx, dy = ct.desplazamiento_paralaje_km_por_km()
    assert dx > 0 and dy > 0  # el satélite la ve al este y al sur: se corrige al revés
    lat, lon = latlon_de_pixel(np.array([300.0]), np.array([300.0]))
    g = ct.grilla_satelite(lat, lon, np.array([-60.0]), SHAPE)
    y, x = np.unravel_index(np.nanargmin(g), g.shape)
    assert y < 300 and x < 300


def test_satelite_publicado_y_rayos_de_la_ventana():
    t0 = datetime(2026, 11, 19, 22, 0, tzinfo=timezone.utc)
    inicios = [t0 - timedelta(minutes=m) for m in (20, 10, 0)]
    assert ct.elegir_satelite(inicios, t0) == 1  # el de las 22:00 todavía no se publicó
    fines = np.array([np.datetime64((t0 - timedelta(minutes=m)).replace(tzinfo=None)) for m in (15, 5, 1)])
    lat, lon = ct.rayos_ventana(fines, [1, 2, 3], [4, 5, 6], t0.replace(tzinfo=None))
    assert list(lat) == [2, 3]
