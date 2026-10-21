import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import app
from app import sanitize_ml_probabilities, sanitize_storm_probabilities
from scripts.motor_decision_climaar import intelligence_score
from scripts.prediccion_rma10_ml import get_calibrated_probability
from scripts import procesador_radar
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

    def test_storm_api_does_not_return_legacy_heuristics_as_probabilities(self):
        with (
            patch.object(app, "STORM_FILE", app.BASE_DIR / "data/ia/inteligencia_tormenta.json"),
            patch.object(app, "radar_metadata", return_value={}),
            patch.object(app, "radar_timestamp", return_value="2026-10-21T19:40:00Z"),
        ):
            payload = app.tormenta()

        self.assertFalse(payload["probabilidades_calibradas"])
        self.assertTrue(
            all(
                value is None
                for value in payload["datos"]["probabilidades"].values()
            )
        )

    def test_model_api_applies_legacy_per_model_calibration_metrics(self):
        path = app.BASE_DIR / "data/radar/seguimiento/prediccion_ml_rma10.json"
        source = app.read_json(path)

        with (
            patch.object(app, "ML_PREDICTION_FILE", path),
            patch.object(app, "radar_metadata", return_value={}),
        ):
            payload = app.modelo()

        track = payload["predicciones"][0]
        for target in ("strengthened", "weakened", "toward_bahia"):
            key = f"5m_{target}"
            selection = source["calibracion_probabilistica"]["metricas"][key][
                "selection"
            ]
            value = track["horizontes"]["5"][target]
            if selection == "CALIBRATED":
                self.assertIsNotNone(value)
            else:
                self.assertIsNone(value)

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


class StaticEchoMaskTests(unittest.TestCase):
    def test_intensifying_echo_is_preserved_inside_static_mask(self):
        static_mask = np.ones((3, 3), dtype=bool)
        previous = np.ones((3, 3), dtype=np.uint8)
        current = previous.copy()
        current[1, 1] = 3

        with (
            patch.object(procesador_radar, "cargar_mascara", return_value=static_mask),
            patch.object(procesador_radar, "cargar_info_mascara", return_value={}),
            patch.object(
                procesador_radar,
                "_cargar_intensidades_previas",
                return_value=[previous],
            ),
        ):
            result, removed, _ = procesador_radar.aplicar_mascara_estatica(current)

        self.assertEqual(removed, 8)
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(result[1, 1], 3)


if __name__ == "__main__":
    unittest.main()
