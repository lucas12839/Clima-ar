import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import importlib.util

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/estaciones/validar_red_observaciones.py"
spec = importlib.util.spec_from_file_location("validar_red", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class StationValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.catalog_path = self.root / "catalog.json"
        self.bcp_path = self.root / "bcp.json"
        self.smn_path = self.root / "smn.json"
        self.output_path = self.root / "out.json"
        self.now = datetime.now(timezone.utc).replace(microsecond=0)
        module.CATALOG = self.catalog_path
        module.BCP = self.bcp_path
        module.SMN = self.smn_path
        module.OUTPUT = self.output_path

    def tearDown(self):
        self.tmp.cleanup()

    def write_inputs(self, catalog, bcp, smn):
        self.catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
        self.bcp_path.write_text(json.dumps({"stations": bcp}), encoding="utf-8")
        self.smn_path.write_text(json.dumps({"stations": smn}), encoding="utf-8")

    def test_timestamp_utc_y_naive(self):
        self.assertIsNone(module.parse_time("2026-11-14T12:00:00"))
        self.assertIsNotNone(module.parse_time("2026-11-14T12:00:00Z"))

    def test_red_real_14_estaciones(self):
        from pathlib import Path
        # Usa el catálogo real de GitHub si el test se ejecuta dentro del repo.
        catalog_file = Path(__file__).resolve().parents[1] / "data/ia/estaciones/catalogo_estaciones.json"
        if not catalog_file.exists():
            self.skipTest("Catálogo real no disponible en este paquete aislado")
        catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
        bcp, smn = [], []
        for station in catalog["stations"]:
            rec = {"observation_time": (self.now - timedelta(minutes=1)).isoformat()}
            if station["network"] == "bcp":
                rec["station_id"] = int(station["collector_id"])
                bcp.append(rec)
            else:
                rec["wigos"] = station["wigos_id"]
                smn.append(rec)
        self.write_inputs(catalog, bcp, smn)
        self.assertEqual(module.main(), 0)
        result = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual((result["station_count"], result["valid_count"]), (14, 14))
        self.assertEqual(result["bcp"], {"total": 9, "valid": 9})
        self.assertEqual(result["smn"], {"total": 5, "valid": 5})

    def test_descarta_vencida_futura_y_timestamp_ausente(self):
        catalog = {"station_count": 3, "stations": [
            {"network": "bcp", "collector_id": "1", "name": "A", "active": True},
            {"network": "bcp", "collector_id": "2", "name": "B", "active": True},
            {"network": "smn", "wigos_id": "W", "name": "C", "active": True},
        ]}
        bcp = [
            {"station_id": 1, "observation_time": (self.now - timedelta(minutes=16)).isoformat()},
            {"station_id": 2, "observation_time": (self.now + timedelta(minutes=1)).isoformat()},
        ]
        smn = [{"wigos": "W"}]
        self.write_inputs(catalog, bcp, smn)
        self.assertEqual(module.main(), 0)
        result = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(result["valid_count"], 0)
        self.assertEqual([x["state"] for x in result["stations"]],
                         ["vencida", "timestamp_futuro", "timestamp_invalido_o_ausente"])

    def test_observacion_reciente_se_acepta(self):
        catalog = {"station_count": 1, "stations": [
            {"network": "bcp", "collector_id": "1", "name": "A", "active": True}
        ]}
        bcp = [{"station_id": 1, "observation_time": (self.now - timedelta(minutes=14)).isoformat()}]
        self.write_inputs(catalog, bcp, [])
        self.assertEqual(module.main(), 0)
        result = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(result["valid_count"], 1)


if __name__ == "__main__":
    unittest.main()
