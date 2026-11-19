import unittest

import numpy as np

import nowcast_flujo_optico as nf
import motor_decision_climaar as motor
import tormenta_intelligence as engine
from georreferenciar_rma10 import confianza_alcance


def celda(shape, cx, cy, radio=8, dbz=50.0):
    h, w = shape
    gy, gx = np.ogrid[:h, :w]
    campo = np.full(shape, np.nan, dtype=np.float32)
    campo[(gx - cx) ** 2 + (gy - cy) ** 2 <= radio ** 2] = dbz
    return campo


class FlujoOpticoTest(unittest.TestCase):
    def setUp(self):
        self.shape = (512, 512)
        self.clutter = np.zeros(self.shape, dtype=bool)
        self.bx, self.by = nf.latlon_to_pixel(nf.BAHIA_LAT, nf.BAHIA_LON, 512, 512)

    def test_estima_velocidad_y_direccion_de_una_celda(self):
        # 10 px en 10 min hacia el este = 1 px/min ≈ 56 km/h.
        prev = celda(self.shape, 150, 250)
        act = celda(self.shape, 160, 250)
        res = nf.pronosticar(prev, act, 10.0, self.clutter)
        mov = res["flujo"]
        self.assertTrue(mov["valido"])
        self.assertAlmostEqual(mov["direccion_hacia_grados"], 90, delta=15)
        self.assertAlmostEqual(mov["velocidad_media_kmh"], 56, delta=15)
        # A 30 min la celda debería estar ~30 px más al este.
        f30 = res["campos"][30]
        ys, xs = np.where(f30 >= 35)
        self.assertAlmostEqual(float(xs.mean()), 190, delta=6)

    def test_celda_que_viene_hacia_la_ciudad_tiene_eta(self):
        x0 = self.bx - 40
        prev = celda(self.shape, x0 - 10, self.by)
        act = celda(self.shape, x0, self.by)
        res = nf.pronosticar(prev, act, 10.0, self.clutter)
        imp = nf.impacto_ciudad(res["actual"], res["flujo"])
        self.assertIsNotNone(imp["eta_tormenta_min"])
        # Borde de la celda a ~32 px del borde de la zona (10 km ≈ 10.6 px) a 1 px/min.
        self.assertTrue(15 <= imp["eta_tormenta_min"] <= 35, imp["eta_tormenta_min"])

    def test_celda_que_se_aleja_no_tiene_eta(self):
        x0 = self.bx - 40
        prev = celda(self.shape, x0 + 10, self.by)
        act = celda(self.shape, x0, self.by)
        res = nf.pronosticar(prev, act, 10.0, self.clutter)
        imp = nf.impacto_ciudad(res["actual"], res["flujo"])
        self.assertIsNone(imp["eta_tormenta_min"])

    def test_sin_ecos_no_es_valido(self):
        vacio = np.full(self.shape, np.nan, dtype=np.float32)
        res = nf.pronosticar(vacio, vacio, 10.0, self.clutter)
        self.assertFalse(res["flujo"]["valido"])

    def test_clutter_se_ignora(self):
        clutter = np.zeros(self.shape, dtype=bool)
        clutter[240:270, 240:270] = True
        campo = celda(self.shape, 255, 255, radio=10)
        limpio = nf.campo_limpio(campo, clutter)
        self.assertEqual(float(limpio[240:270, 240:270].max()), 0.0)

    def test_factores_de_confianza(self):
        self.assertEqual(nf.factor_edad(10), 1.0)
        self.assertEqual(nf.factor_edad(30), 1.0)
        self.assertAlmostEqual(nf.factor_edad(45), 0.7, places=2)
        self.assertEqual(nf.factor_edad(61), 0.0)
        self.assertEqual(nf.factor_alcance(100), 1.0)
        self.assertAlmostEqual(nf.factor_alcance(240), 0.5)
        self.assertEqual(confianza_alcance(150), 1.0)
        self.assertAlmostEqual(confianza_alcance(195), 0.75, places=2)


class MotorFlujoTest(unittest.TestCase):
    def flujo(self, eta, conf=0.8, dbz=48.0, ts="2026-11-19T05:00:00Z"):
        return {
            "disponible": True,
            "radar_timestamp": ts,
            "confianza": conf,
            "movimiento": {"velocidad_media_kmh": 40, "direccion_hacia_grados": 90},
            "bahia_blanca": {"eta_tormenta_desde_ahora_min": eta, "dbz_al_llegar": dbz},
        }

    def test_llegada_cercana_suma_evidencia(self):
        score, ok, res = motor.optical_flow_evidence(
            self.flujo(10), {"frame_actual_utc": "2026-11-19T05:00:00+00:00"}
        )
        self.assertTrue(ok)
        self.assertGreater(score, 0.5)
        self.assertEqual(res["eta_tormenta_min"], 10)

    def test_sin_llegada_no_suma(self):
        score, ok, _ = motor.optical_flow_evidence(
            self.flujo(None), {"frame_actual_utc": "2026-11-19T05:00:00Z"}
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_llegada_de_lluvia_en_el_umbral_no_suma(self):
        score, ok, _ = motor.optical_flow_evidence(
            self.flujo(10, dbz=37.0), {"frame_actual_utc": "2026-11-19T05:00:00Z"}
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_baja_confianza_no_suma(self):
        score, ok, _ = motor.optical_flow_evidence(
            self.flujo(10, conf=0.3), {"frame_actual_utc": "2026-11-19T05:00:00Z"}
        )
        self.assertFalse(ok)

    def test_desincronizado_no_suma(self):
        score, ok, res = motor.optical_flow_evidence(
            self.flujo(10), {"frame_actual_utc": "2026-11-19T04:50:00Z"}
        )
        self.assertFalse(ok)
        self.assertEqual(res["motivo"], "DESINCRONIZADO")


class EscalaEdadRadarTest(unittest.TestCase):
    def nowcast(self, minutos, alcance=None):
        from datetime import datetime, timedelta, timezone

        frame = datetime.now(timezone.utc) - timedelta(minutes=minutos)
        nc = {
            "frame_actual_utc": frame.isoformat(),
            "actividad": True,
            "movimiento_hacia_bahia": True,
            "eta_minutos": 120.0,
            "distancia_km": 80.0,
            "velocidad_kmh": 40.0,
            "confianza_movimiento": 1.0,
        }
        if alcance is not None:
            nc["confianza_alcance"] = alcance
        return nc

    def test_hasta_30_min_es_vigente(self):
        v = engine.apply_radar_age(self.nowcast(25))["vigencia_radar"]
        self.assertEqual(v["modo"], "radar_vigente")

    def test_45_min_baja_la_confianza(self):
        nc = engine.apply_radar_age(self.nowcast(45))
        self.assertEqual(nc["vigencia_radar"]["modo"], "estimado_desde_ultimo_radar_valido")
        self.assertAlmostEqual(nc["confianza_movimiento"], 0.7, delta=0.03)

    def test_mas_de_60_min_es_obsoleto_sin_actividad(self):
        nc = engine.apply_radar_age(self.nowcast(70))
        self.assertEqual(nc["vigencia_radar"]["modo"], "radar_obsoleto")
        self.assertTrue(nc["vigencia_radar"]["radar_atrasado"])
        self.assertFalse(nc["actividad"])

    def test_lejos_del_radar_baja_la_confianza(self):
        nc = engine.apply_radar_age(self.nowcast(5, alcance=0.6))
        self.assertAlmostEqual(nc["confianza_movimiento"], 0.6, places=3)


if __name__ == "__main__":
    unittest.main()
