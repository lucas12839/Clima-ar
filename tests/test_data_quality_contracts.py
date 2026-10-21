import unittest
from datetime import datetime, timezone

from app import sanitize_ml_probabilities, sanitize_storm_probabilities
from scripts.motor_decision_climaar import intelligence_score
from scripts.prediccion_rma10_ml import get_calibrated_probability
from scripts.tormenta_intelligence_timestamp import temporal_interpolation


class ProbabilityContractTests(unittest.TestCase):
    def test_uncalibrated_ml_output_is_not_published_as_probability(self):
        self.assertIsNone(
            get_calibrated_probability(
                0.8,
                {"enabled": False},
            )
        )

    def test_calibrated_ml_output_is_retained(self):
        value = get_calibrated_probability(
            0.8,
            {"enabled": True, "a": 1.0, "b": 0.0},
        )
        self.assertAlmostEqual(value, 0.8)

    def test_api_hides_only_uncalibrated_horizons(self):
        payload = {
            "calibracion_probabilistica": {
                "por_modelo": {
                    "5m_strengthened": {"aplicada": True},
                    "5m_toward_bahia": {"aplicada": False},
                }
            },
            "predicciones": [
                {
                    "horizontes": {
                        "5": {
                            "strengthened": 80.0,
                            "weakened": 20.0,
                            "toward_bahia": 90.0,
                        }
                    }
                }
            ],
        }

        result = sanitize_ml_probabilities(payload)
        horizon = result["predicciones"][0]["horizontes"]["5"]
        self.assertEqual(horizon["strengthened"], 80.0)
        self.assertIsNone(horizon["weakened"])
        self.assertIsNone(horizon["toward_bahia"])

    def test_uncalibrated_storm_probabilities_are_hidden(self):
        result = sanitize_storm_probabilities(
            {
                "probabilidades": {"tormenta": 0.8},
                "peligros": {"granizo": 0.6},
                "ia_v8": {"probabilidad_impacto": 0.7},
            }
        )
        self.assertIsNone(result["probabilidades"]["tormenta"])
        self.assertIsNone(result["peligros"]["granizo"])
        self.assertIsNone(result["ia_v8"]["probabilidad_impacto"])

    def test_decision_engine_uses_heuristic_indices_not_legacy_probabilities(self):
        result = intelligence_score(
            {
                "probabilidades": {"tormenta": 0.99},
                "indices_heuristicos": {
                    "tormenta": 0.25,
                    "peligros": {"granizo": 0.4},
                },
            }
        )
        self.assertEqual(result, 0.4)


class TemporalInterpolationTests(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        self.rows = [
            {
                "tiempo_utc": self.start.isoformat(),
                "temperature_2m": 10.0,
            },
            {
                "tiempo_utc": datetime(
                    2024, 1, 1, 1, tzinfo=timezone.utc
                ).isoformat(),
                "temperature_2m": 20.0,
            },
        ]

    def test_interpolates_between_observed_times(self):
        target = datetime(2024, 1, 1, 0, 30, tzinfo=timezone.utc)
        result = temporal_interpolation(self.rows, target)
        self.assertTrue(result["interpolacion_temporal"])
        self.assertEqual(result["temperature_2m"], 15.0)

    def test_does_not_extrapolate_before_or_after_available_data(self):
        before = datetime(2023, 12, 31, 23, 30, tzinfo=timezone.utc)
        after = datetime(2024, 1, 1, 1, 30, tzinfo=timezone.utc)
        self.assertIsNone(temporal_interpolation(self.rows, before))
        self.assertIsNone(temporal_interpolation(self.rows, after))


if __name__ == "__main__":
    unittest.main()
