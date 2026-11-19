import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "estaciones"))
import colector_cwop as cc  # noqa: E402

AHORA = datetime(2026, 11, 19, 17, 55, tzinfo=timezone.utc)
PAQUETE = ("LU4DRH-11>APRS,TCPIP*,qAC,T2SPAIN:@191750z3841.36S/06220.05W_084/008g010t068"
           "r000p000P000h61b100810Ecowitt WS")


class ParsearTest(unittest.TestCase):
    def test_paquete_real(self):
        p = cc.parsear(PAQUETE, AHORA)
        v = p["variables"]
        self.assertEqual(p["observation_time"], "2026-11-19T17:50:00+00:00")
        self.assertAlmostEqual(p["lat"], -38.6893, places=3)
        self.assertEqual(v["temperatura_c"], 20.0)
        self.assertEqual(v["viento_kmh"], 12.9)
        self.assertEqual(v["rafaga_15min_kmh"], 16.1)
        self.assertEqual(v["presion_hpa"], 1008.1)
        self.assertEqual(v["precipitacion_dia_mm"], 0.0)

    def test_valores_faltantes(self):
        p = cc.parsear("X1>APRS:@191750z3841.36S/06220.05W_.../...g...t068r...P012h..b.....", AHORA)
        self.assertIsNone(p["variables"]["viento_kmh"])
        self.assertIsNone(p["variables"]["presion_hpa"])
        self.assertEqual(p["variables"]["precipitacion_dia_mm"], 3.0)

    def test_basura_no_rompe(self):
        self.assertIsNone(cc.parsear("# aprsc 3.0.1", AHORA))
        self.assertIsNone(cc.parsear("LU4DRH-11>APRS:>status texto", AHORA))


class SalidaTest(unittest.TestCase):
    CFG = {"estaciones": [{"indicativo": "LU4DRH-11", "nombre": "BB NO (CWOP)", "lat": -38.69, "lon": -62.33}]}

    def test_valida_y_atrasada(self):
        rec = {"LU4DRH-11": cc.parsear(PAQUETE, AHORA)}
        s = cc.armar_salida(self.CFG, rec, {}, AHORA)
        self.assertEqual(s["stations"][0]["estado"], "valida")
        tarde = datetime(2026, 11, 19, 18, 30, tzinfo=timezone.utc)
        s2 = cc.armar_salida(self.CFG, {}, s, tarde)
        self.assertEqual(s2["stations"][0]["estado"], "atrasada")

    def test_sin_datos(self):
        s = cc.armar_salida(self.CFG, {}, {}, AHORA)
        self.assertEqual(s["stations"][0]["estado"], "sin_datos")


if __name__ == "__main__":
    unittest.main()
