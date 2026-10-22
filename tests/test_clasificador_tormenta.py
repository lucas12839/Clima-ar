import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import clasificador_tormenta as clf  # noqa: E402
import tormenta_intelligence as engine  # noqa: E402

WHEN = datetime(2026, 12, 16, 19, 0, tzinfo=timezone.utc)

CALM = dict(
    temperature_2m=15, relative_humidity_2m=81, dew_point_2m=12,
    pressure_msl=1018, wind_speed_10m=11, wind_direction_10m=50,
    wind_gusts_10m=22, precipitation=0,
)
STORM = dict(
    temperature_2m=33, relative_humidity_2m=60, dew_point_2m=22,
    pressure_msl=1003, wind_speed_10m=45, wind_direction_10m=350,
    wind_gusts_10m=95, precipitation=12,
)


class ClasificadorTormentaTest(unittest.TestCase):
    def test_modelo_diferencia_ambientes(self):
        calm = clf.predict(CALM, WHEN)
        storm = clf.predict(STORM, WHEN)
        self.assertIsNotNone(calm)
        self.assertEqual(calm["clase"], "sin_tormenta")
        self.assertEqual(storm["clase"], "tormenta_severa")
        self.assertGreater(
            storm["indice_severidad_historica"],
            calm["indice_severidad_historica"],
        )

    def test_sistema_que_no_amenaza_es_sin_tormenta(self):
        storm = clf.predict(STORM, WHEN)
        result = engine.classify_weather(
            {"amenaza": False}, "rojo", {"x": 0.9}, 0.9, storm,
            {"intensidad_nivel_max": 6},
        )
        self.assertEqual(result, "sin_tormenta")

    def test_eco_debil_no_eleva_la_clase(self):
        storm = clf.predict(STORM, WHEN)
        result = engine.classify_weather(
            {"amenaza": True, "nivel": "alta"}, "rojo", {"x": 0.9}, 0.9,
            storm, {"intensidad_nivel_max": 2},
        )
        self.assertEqual(result, "tormenta")

    def test_severidad_limitada_por_clase(self):
        self.assertEqual(
            engine.cap_severity_by_class("rojo", "tormenta"), "amarillo"
        )
        self.assertEqual(
            engine.cap_severity_by_class("naranja", "sin_tormenta"), "verde"
        )


if __name__ == "__main__":
    unittest.main()
