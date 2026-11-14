import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.estaciones.observation_network_encoder import build_observation_batch
from scripts.estaciones.colector_bcp import is_fresh_age as bcp_is_fresh
from scripts.estaciones.colector_smn_synop import is_fresh_age as smn_is_fresh


class ObservationFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)

    def write_json(self, path, data):
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_colectores_comparten_limite_inclusivo_de_15_minutos(self):
        for checker in (bcp_is_fresh, smn_is_fresh):
            self.assertTrue(checker(0))
            self.assertTrue(checker(15))
            self.assertFalse(checker(15.001))
            self.assertFalse(checker(-0.001))
            self.assertFalse(checker(None))

    def test_encoder_recalcula_edad_desde_timestamp_y_acepta_15_exactos(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            stations = [
                {"name": "BCP 15", "network": "bcp", "station_number": 1,
                 "active": True, "latitude": -38.7, "longitude": -62.2},
                {"name": "SMN 14", "network": "smn", "wigos": "w1",
                 "active": True, "latitude": -38.8, "longitude": -62.3},
                {"name": "BCP 15+ ", "network": "bcp", "station_number": 2,
                 "active": True, "latitude": -38.6, "longitude": -62.1},
                {"name": "SMN futuro", "network": "smn", "wigos": "w2",
                 "active": True, "latitude": -38.5, "longitude": -62.0},
                {"name": "BCP sin hora", "network": "bcp", "station_number": 3,
                 "active": True, "latitude": -38.4, "longitude": -62.4},
            ]
            self.write_json(tmp / "catalog.json", {"stations": stations})
            self.write_json(tmp / "bcp.json", {"stations": [
                {"station_id": 1,
                 "observation_time": (self.now - timedelta(minutes=15)).isoformat(),
                 "edad_minutos": 999, "fresh": False, "estado": "descartada",
                 "variables": {"temperatura_c": 20}},
                {"station_id": 2,
                 "observation_time": (self.now - timedelta(minutes=15.01)).isoformat(),
                 "edad_minutos": 1, "fresh": True, "estado": "valida",
                 "variables": {"temperatura_c": 30}},
                {"station_id": 3, "edad_minutos": 1, "fresh": True,
                 "estado": "valida", "variables": {"temperatura_c": 40}},
            ]})
            self.write_json(tmp / "smn.json", {"stations": [
                {"wigos": "w1",
                 "observation_time": (self.now - timedelta(minutes=14.9)).isoformat(),
                 "edad_minutos": 200, "fresh": False, "estado": "descartada",
                 "variables": {}},
                {"wigos": "w2",
                 "observation_time": (self.now + timedelta(minutes=1)).isoformat(),
                 "edad_minutos": 0, "fresh": True, "estado": "valida",
                 "variables": {}},
            ]})
            batch = build_observation_batch(
                tmp / "catalog.json", tmp / "bcp.json", tmp / "smn.json",
                now=self.now,
            )
            self.assertEqual(batch["station_count"], 5)
            self.assertEqual(batch["valid_count"], 2)
            self.assertEqual(batch["ages_minutes"][0], 15.0)
            self.assertAlmostEqual(batch["ages_minutes"][1], 14.9)
            self.assertIsNone(batch["ages_minutes"][2])
            self.assertIsNone(batch["ages_minutes"][3])
            self.assertIsNone(batch["ages_minutes"][4])


if __name__ == "__main__":
    unittest.main()
