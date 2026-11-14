import unittest
from datetime import datetime, timedelta, timezone

from scripts import meteorologia_fusion as fusion


class MeteorologiaCalidadTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)
        self.radar = {"disponible": True, "tiles_ok": True}
        self.ambiente = [{"datos": {"temperature_2m": 20}} for _ in range(9)]

    def sazb(self, age):
        return {
            "disponible": True,
            "obsTime_utc": (self.now - timedelta(minutes=age)).isoformat(),
        }

    def test_sazb_fresca_aporta_puntos(self):
        out = fusion.calcular_calidad(
            self.radar, self.sazb(10), self.ambiente, now=self.now
        )
        self.assertEqual(out["score"], 100)
        self.assertTrue(out["sazb_valido"])
        self.assertTrue(out["sazb_fresco"])

    def test_sazb_vieja_no_suma_puntos_por_disponibilidad(self):
        out = fusion.calcular_calidad(
            self.radar, self.sazb(56), self.ambiente, now=self.now
        )
        self.assertEqual(out["score"], 80)
        self.assertEqual(out["nivel"], "medio")
        self.assertTrue(out["sazb_disponible"])
        self.assertFalse(out["sazb_valido"])
        self.assertFalse(out["sazb_fresco"])
        self.assertEqual(out["sazb_edad_minutos"], 56.0)

    def test_recalcula_edad_y_no_confia_en_edad_cacheada(self):
        sazb = self.sazb(56)
        sazb["edad_minutos"] = 1
        out = fusion.calcular_calidad(
            self.radar, sazb, self.ambiente, now=self.now
        )
        self.assertEqual(out["sazb_edad_minutos"], 56.0)
        self.assertFalse(out["sazb_fresco"])
        self.assertEqual(out["score"], 80)

    def test_timestamp_futuro_no_se_considera_fresco(self):
        sazb = {
            "disponible": True,
            "obsTime_utc": (self.now + timedelta(minutes=4)).isoformat(),
        }
        out = fusion.calcular_calidad(
            self.radar, sazb, self.ambiente, now=self.now
        )
        self.assertEqual(out["score"], 80)
        self.assertFalse(out["sazb_fresco"])
        self.assertEqual(out["sazb_edad_minutos"], -4.0)

    def test_sazb_sin_timestamp_no_se_considera_fresco(self):
        out = fusion.calcular_calidad(
            self.radar, {"disponible": True, "edad_minutos": 1},
            self.ambiente, now=self.now
        )
        self.assertEqual(out["score"], 80)
        self.assertFalse(out["sazb_fresco"])

    def test_sazb_no_disponible_no_suma_puntos(self):
        out = fusion.calcular_calidad(
            self.radar, {"disponible": False}, self.ambiente, now=self.now
        )
        self.assertEqual(out["score"], 80)
        self.assertFalse(out["sazb_disponible"])


    def test_sazb_vencida_no_reemplaza_superficie_ecmwf(self):
        sazb = self.sazb(25.6)
        sazb.update({
            "temperatura_c": 5,
            "punto_rocio_c": 4,
            "presion_hpa": 990,
            "viento_kt": 30,
            "direccion_viento": 270,
        })
        actual = {
            "temperature_2m": 18,
            "dew_point_2m": 12,
            "pressure_msl": 1015,
            "wind_speed_10m": 20,
            "wind_direction_10m": 180,
            "wind_gusts_10m": 28,
        }
        out = fusion.fusionar_actual_ecmwf_sazb(actual, sazb, now=self.now)
        self.assertEqual(out["fuente_superficie_actual"], "ECMWF")
        self.assertEqual(out["estado_observacion_actual"], "observacion_sazb_vencida")
        self.assertEqual(out["temperature_2m"], 18)
        self.assertEqual(out["wind_speed_10m"], 20)
        self.assertFalse(out["observacion_sazb_fresca"])
        self.assertEqual(out["observacion_sazb_edad_minutos"], 25.6)

    def test_sazb_exactamente_15_minutos_sigue_vigente(self):
        sazb = self.sazb(15)
        out = fusion.calcular_calidad(
            self.radar, sazb, self.ambiente, now=self.now
        )
        self.assertTrue(out["sazb_fresco"])
        self.assertEqual(out["score"], 100)

    def test_fusion_marca_observacion_futura_como_no_fresca(self):
        sazb = self.sazb(-4)
        out = fusion.fusionar_actual_ecmwf_sazb(
            {"wind_speed_10m": 12, "wind_direction_10m": 180},
            sazb, now=self.now
        )
        self.assertFalse(out["observacion_sazb_fresca"])
        self.assertEqual(
            out["estado_observacion_actual"], "observacion_sazb_vencida"
        )


if __name__ == "__main__":
    unittest.main()
