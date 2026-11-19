import unittest

import numpy as np

from goes19 import glm_ventana as gv

LOCS = [{"id": "bahia_blanca", "lat": -38.7183, "lon": -62.2663},
        {"id": "punta_alta", "lat": -38.88, "lon": -62.075}]


class ResumirTest(unittest.TestCase):
    def test_sin_rayos(self):
        r = gv.resumir([], [], [], LOCS, 20)
        self.assertTrue(r["disponible"])
        self.assertEqual(r["tasa_30km_por_min"], 0.0)
        self.assertIsNone(r["por_localidad"]["bahia_blanca"]["distancia_km"])

    def test_salto_y_localidad(self):
        # 30 rayos recientes sobre Punta Alta, 2 previos.
        lat = [-38.88] * 30 + [-38.88] * 2
        lon = [-62.075] * 32
        edad = [3.0] * 30 + [15.0] * 2
        r = gv.resumir(lat, lon, edad, LOCS, 20)
        pa = r["por_localidad"]["punta_alta"]
        self.assertEqual(pa["tasa_30km_por_min"], 3.0)
        self.assertEqual(pa["tasa_30km_previa_por_min"], 0.2)
        self.assertEqual(pa["rayos_15km_10min"], 30)
        # Bahía Blanca está a ~25 km: cuenta "a <= 30 km" pero no "sobre".
        bb = r["por_localidad"]["bahia_blanca"]
        self.assertEqual(bb["rayos_15km_10min"], 0)
        self.assertGreater(bb["distancia_km"], 20)

    def test_cobertura_corta_no_da_tasa_previa(self):
        r = gv.resumir([-38.7], [-62.27], [1.0], LOCS, 8)
        self.assertIsNone(r["por_localidad"]["bahia_blanca"]["tasa_30km_previa_por_min"])
        self.assertEqual(r["por_localidad"]["bahia_blanca"]["tasa_30km_por_min"], 0.12)


if __name__ == "__main__":
    unittest.main()
