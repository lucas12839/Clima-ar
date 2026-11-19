import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import localidades_radar as lr
import motor_decision_climaar as motor
import nowcast_flujo_optico as nf
from scripts.rma10_features import latlon_to_pixel

LOCS = [
    {"id": "bahia_blanca", "nombre": "Bahía Blanca", "lat": -38.7183, "lon": -62.2663, "radio_km": 10.0},
    {"id": "punta_alta", "nombre": "Punta Alta", "lat": -38.88, "lon": -62.075, "radio_km": 6.0},
]
T0 = datetime(2026, 11, 13, 6, 0, tzinfo=timezone.utc)


def campo_con_celda(lat, lon, dbz=50.0, radio_px=6):
    c = np.zeros((512, 512), dtype=np.float32)
    x, y = latlon_to_pixel(lat, lon, 512, 512)
    yy, xx = np.ogrid[:512, :512]
    c[(xx - x) ** 2 + (yy - y) ** 2 <= radio_px ** 2] = dbz
    return c


class LocalidadesTest(unittest.TestCase):
    def test_lluvia_sobre_punta_alta_y_no_sobre_bahia(self):
        c = campo_con_celda(-38.88, -62.075)
        with tempfile.TemporaryDirectory() as tmp:
            h = Path(tmp) / "h.json"
            for i in range(6):
                r = lr.analizar(T0 + timedelta(minutes=10 * i), c, None, 0.0, None, LOCS, h)
        self.assertGreater(r["punta_alta"]["mm_h_medio"], 5)
        self.assertEqual(r["bahia_blanca"]["mm_h_medio"], 0.0)
        self.assertGreater(r["punta_alta"]["lluvia_1h_mm"], 5)
        self.assertGreaterEqual(r["punta_alta"]["minutos_lluvia_intensa"], 50)

    def test_llegada_con_movimiento(self):
        # Celda 15 km al oeste de Punta Alta moviéndose al este a ~60 km/h.
        c = campo_con_celda(-38.88, -62.25)
        km = nf.km_por_pixel(512)
        u = np.full(c.shape, 1.0 / km, dtype=np.float32)      # px/min = 60 km/h
        v = np.zeros(c.shape, dtype=np.float32)
        r = lr.llegadas(c, {"u": u, "v": v, "valido": True}, LOCS)
        pa = r["punta_alta"]
        self.assertIsNotNone(pa["eta_min"])
        self.assertLessEqual(pa["eta_min"], 20)
        self.assertAlmostEqual(pa["velocidad_local_kmh"], 60, delta=3)

    def test_confiabilidad_y_tendencia(self):
        tabla = {"global": {"20-40": {"35-40": {"casos": 195, "llego": 63, "se_desvio": 110, "se_deshizo": 22}}},
                 "por_localidad": {}}
        conf = motor.confiabilidad_llegada("punta_alta", 30, 38, tabla)
        self.assertEqual(conf["base"], "region")
        self.assertAlmostEqual(conf["fraccion"], 0.32, places=2)
        self.assertEqual(motor.tendencia(55, 45), "se intensifican")
        self.assertEqual(motor.tendencia(40, 50), "se debilitan")
        self.assertIsNone(motor.tendencia(40, None))


if __name__ == "__main__":
    unittest.main()


class EstadoFuentesTest(unittest.TestCase):
    def test_estado_fuentes_tiene_todas(self):
        r = motor.estado_fuentes()
        self.assertEqual(set(r["fuentes"]), {"radar", "satelite", "rayos", "modelo", "estaciones"})
        self.assertIn("a_ciegas", r)

    def test_edad(self):
        ahora = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(motor._edad_min("2026-11-19T11:30:00Z", ahora), 30.0)
        self.assertIsNone(motor._edad_min(None, ahora))
