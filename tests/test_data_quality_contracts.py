import json
import shutil
import unittest
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import app
from app import sanitize_ml_probabilities, sanitize_storm_probabilities
from scripts.motor_decision_climaar import intelligence_score
from scripts.prediccion_rma10_ml import get_calibrated_probability
from scripts import procesador_radar
from scripts import conectar_evolucion_trayectoria as evolution_connector
from scripts import validar_integridad_climaar as integrity_validator
from scripts.prediccion_evolucion import predict_track
from scripts.tormenta_intelligence_timestamp import temporal_interpolation
from scraper import capturar_radar_smn


class EvolutionGapTests(unittest.TestCase):
    def test_short_gap_keeps_motion_forecast(self):
        start = datetime(2026, 10, 21, 19, 0, tzinfo=timezone.utc)
        track = {
            "track_id": 1,
            "historial": [
                {"timestamp": start.isoformat(), "x": 10, "y": 20},
                {
                    "timestamp": (start + timedelta(minutes=10)).isoformat(),
                    "x": 20,
                    "y": 25,
                },
            ],
        }

        prediction, _ = predict_track(track)

        self.assertTrue(prediction["pronostico_disponible"])
        self.assertEqual(prediction["max_gap_temporal_minutos"], 10)
        self.assertEqual(
            [item["horizonte_minutos"] for item in prediction["horizontes"]],
            [5, 10, 20, 30],
        )

    def test_long_but_supported_gap_degrades_confidence(self):
        start = datetime(2026, 10, 21, 19, 0, tzinfo=timezone.utc)

        def prediction_for_gap(gap):
            return predict_track(
                {
                    "track_id": 1,
                    "historial": [
                        {"timestamp": start.isoformat(), "x": 10, "y": 20},
                        {
                            "timestamp": (
                                start + timedelta(minutes=gap)
                            ).isoformat(),
                            "x": 20,
                            "y": 25,
                        },
                    ],
                }
            )[0]

        short_gap = prediction_for_gap(10)
        long_gap = prediction_for_gap(120)

        self.assertTrue(long_gap["pronostico_disponible"])
        self.assertEqual(long_gap["max_gap_temporal_minutos"], 120)
        self.assertLess(
            long_gap["confianza_prediccion"],
            short_gap["confianza_prediccion"],
        )
        self.assertGreater(
            long_gap["horizontes"][0]["incertidumbre_pixeles"],
            short_gap["horizontes"][0]["incertidumbre_pixeles"],
        )

    def test_gap_beyond_supported_interval_does_not_invent_motion(self):
        start = datetime(2026, 10, 21, 19, 0, tzinfo=timezone.utc)
        prediction, _ = predict_track(
            {
                "track_id": 1,
                "historial": [
                    {"timestamp": start.isoformat(), "x": 10, "y": 20},
                    {
                        "timestamp": (start + timedelta(minutes=181)).isoformat(),
                        "x": 20,
                        "y": 25,
                    },
                ],
            }
        )

        self.assertFalse(prediction["pronostico_disponible"])
        self.assertEqual(prediction["motivo"], "SIN_MOVIMIENTO_VALIDO")


class RadarFrameSelectionTests(unittest.TestCase):
    def test_selects_a_newer_frame(self):
        last_frame = datetime(2026, 10, 21, 19, 30, tzinfo=timezone.utc)
        with patch.object(
            capturar_radar_smn,
            "timestamp_persistido",
            return_value=last_frame,
        ):
            selected = capturar_radar_smn.seleccionar_mas_reciente(
                [
                    ("20261021163500", "older.png"),
                    ("20261021164000", "newer.png"),
                ]
            )

        self.assertEqual(selected, ("20261021164000", "newer.png"))

    def test_rejects_duplicate_and_stale_frames(self):
        last_frame = datetime(2026, 10, 21, 19, 40, tzinfo=timezone.utc)
        with patch.object(
            capturar_radar_smn,
            "timestamp_persistido",
            return_value=last_frame,
        ):
            for timestamp in ("20261021164000", "20261021163000"):
                with self.subTest(timestamp=timestamp):
                    self.assertIsNone(
                        capturar_radar_smn.seleccionar_mas_reciente(
                            [(timestamp, "old.png")]
                        )
                    )


class EvolutionConnectorTests(unittest.TestCase):
    def test_preserves_principal_without_valid_motion(self):
        timestamp = "2026-10-21T19:40:00Z"
        radar = {
            "actualizado": timestamp,
            "nowcast": {
                "frame_actual_utc": timestamp,
                "nucleo_principal_id": 17,
                "track_principal_candidato_id": 17,
            },
        }
        evolution = {
            "radar_timestamp": timestamp,
            "nucleos": [
                {
                    "track_id": 17,
                    "pronostico_disponible": False,
                    "motivo": "SIN_MOVIMIENTO_VALIDO",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            radar_path = Path(directory) / "radar_nowcast.json"
            evolution_path = Path(directory) / "prediccion_evolucion.json"
            radar_path.write_text(json.dumps(radar), encoding="utf-8")
            evolution_path.write_text(json.dumps(evolution), encoding="utf-8")

            with (
                patch.object(evolution_connector, "RADAR_FILE", radar_path),
                patch.object(evolution_connector, "EVOLUTION_FILE", evolution_path),
            ):
                evolution_connector.main()

            result = json.loads(radar_path.read_text(encoding="utf-8"))["nowcast"]

        self.assertEqual(result["track_principal_id"], 17)
        self.assertEqual(result["proyecciones"], {})
        self.assertFalse(
            result["proyecciones_metadata"]["pronostico_disponible"]
        )
        self.assertEqual(
            result["proyecciones_metadata"]["motivo_no_disponible"],
            "SIN_MOVIMIENTO_VALIDO",
        )

    def test_rejects_unrecognized_missing_forecast(self):
        nowcast = {
            "nucleo_principal_id": 17,
            "track_principal_candidato_id": 17,
        }
        evolution = {
            "nucleos": [
                {
                    "track_id": 17,
                    "pronostico_disponible": False,
                    "motivo": "DATOS_CORRUPTOS",
                }
            ]
        }

        with self.assertRaisesRegex(RuntimeError, "estado de evolución"):
            evolution_connector.select_principal_track(nowcast, evolution)

    def test_valid_motion_still_requires_and_builds_all_horizons(self):
        timestamp = "2026-10-21T19:40:00Z"
        track = {
            "track_id": 17,
            "pronostico_disponible": True,
            "horizontes": [
                {
                    "horizonte_minutos": horizon,
                    "x_estimado": 250 + horizon,
                    "y_estimado": 250,
                }
                for horizon in (5, 10, 20, 30)
            ],
        }
        radar = {
            "actualizado": timestamp,
            "nowcast": {
                "frame_actual_utc": timestamp,
                "nucleo_principal_id": 17,
                "track_principal_candidato_id": 17,
            },
        }
        evolution = {"radar_timestamp": timestamp, "nucleos": [track]}

        with tempfile.TemporaryDirectory() as directory:
            radar_path = Path(directory) / "radar_nowcast.json"
            evolution_path = Path(directory) / "prediccion_evolucion.json"
            radar_path.write_text(json.dumps(radar), encoding="utf-8")
            evolution_path.write_text(json.dumps(evolution), encoding="utf-8")

            with (
                patch.object(evolution_connector, "RADAR_FILE", radar_path),
                patch.object(evolution_connector, "EVOLUTION_FILE", evolution_path),
            ):
                evolution_connector.main()

            result = json.loads(radar_path.read_text(encoding="utf-8"))["nowcast"]

        self.assertEqual(set(result["proyecciones"]), {"5", "10", "20", "30"})
        self.assertTrue(
            result["proyecciones_metadata"]["pronostico_disponible"]
        )

    def test_integrity_validator_accepts_explicit_no_motion_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            radar_dir = root / "data/radar"
            track_dir = radar_dir / "seguimiento"
            ia_dir = root / "data/ia"
            track_dir.mkdir(parents=True)
            ia_dir.mkdir(parents=True)

            paths = (
                (integrity_validator.RADAR / "radar_timestamp.txt", radar_dir / "radar_timestamp.txt"),
                (integrity_validator.RADAR / "resultado_procesamiento.json", radar_dir / "resultado_procesamiento.json"),
                (integrity_validator.TRACK / "seguimiento_temporal.json", track_dir / "seguimiento_temporal.json"),
                (integrity_validator.TRACK / "estado.json", track_dir / "estado.json"),
                (integrity_validator.RADAR / "radar_nowcast.json", radar_dir / "radar_nowcast.json"),
                (integrity_validator.TRACK / "prediccion_ml_rma10.json", track_dir / "prediccion_ml_rma10.json"),
                (integrity_validator.TRACK / "prediccion_evolucion.json", track_dir / "prediccion_evolucion.json"),
                (integrity_validator.IA / "estado_meteorologico.json", ia_dir / "estado_meteorologico.json"),
                (integrity_validator.IA / "inteligencia_tormenta.json", ia_dir / "inteligencia_tormenta.json"),
                (integrity_validator.IA / "resultado_final.json", ia_dir / "resultado_final.json"),
            )
            for source, destination in paths:
                shutil.copy2(source, destination)

            candidate = json.loads(
                (track_dir / "seguimiento_temporal.json").read_text(
                    encoding="utf-8"
                )
            )["track_principal_candidato_id"]

            evolution_path = track_dir / "prediccion_evolucion.json"
            evolution = json.loads(evolution_path.read_text(encoding="utf-8"))
            evolution_track = next(
                item
                for item in evolution["nucleos"]
                if str(item["track_id"]) == str(candidate)
            )
            evolution_track.update(
                {
                    "pronostico_disponible": False,
                    "motivo": "SIN_MOVIMIENTO_VALIDO",
                }
            )
            evolution_path.write_text(
                json.dumps(evolution),
                encoding="utf-8",
            )

            nowcast_path = radar_dir / "radar_nowcast.json"
            radar = json.loads(nowcast_path.read_text(encoding="utf-8"))
            nowcast = radar["nowcast"]
            nowcast["proyecciones"] = {}
            nowcast["proyecciones_metadata"].update(
                {
                    "track_id": candidate,
                    "pronostico_disponible": False,
                    "motivo_no_disponible": "SIN_MOVIMIENTO_VALIDO",
                }
            )
            nowcast_path.write_text(json.dumps(radar), encoding="utf-8")

            with (
                patch.object(integrity_validator, "BASE", root),
                patch.object(integrity_validator, "RADAR", radar_dir),
                patch.object(integrity_validator, "TRACK", track_dir),
                patch.object(integrity_validator, "IA", ia_dir),
            ):
                integrity_validator.main()


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
        self.assertEqual(
            payload["tipo_probabilidades"],
            "heuristicas_no_calibradas",
        )
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

        self.assertEqual(
            payload["calibracion"],
            payload["calibracion_probabilistica"],
        )

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
