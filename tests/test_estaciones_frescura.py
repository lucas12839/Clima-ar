import json
import unittest
from pathlib import Path

from scripts.estaciones.colector_bcp import is_fresh_age as bcp_is_fresh
from scripts.estaciones.colector_smn_synop import is_fresh_age as smn_is_fresh

CATALOG = Path(__file__).resolve().parents[1] / "data/ia/estaciones/catalogo_estaciones.json"


class ObservationFreshnessTests(unittest.TestCase):
    def test_colector_bcp_escribe_dentro_del_repositorio_actual(self):
        from scripts.estaciones.colector_bcp import OUTPUT, BASE
        self.assertEqual(OUTPUT, BASE / "data/ia/estaciones/observaciones/bcp_actual.json")
        self.assertNotEqual(str(OUTPUT), "/opt/Clima-ar/data/ia/estaciones/observaciones/bcp_actual.json")

    def test_colectores_comparten_limite_inclusivo_de_15_minutos(self):
        for checker in (bcp_is_fresh, smn_is_fresh):
            self.assertTrue(checker(0))
            self.assertTrue(checker(15))
            self.assertFalse(checker(15.001))
            self.assertFalse(checker(-0.001))
            self.assertFalse(checker(None))

    def test_catalogo_real_conecta_las_estaciones_con_sus_colectores(self):
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        stations = catalog["stations"]
        bcp_catalog = [s for s in stations if s["network"] == "bcp" and s.get("active")]
        smn_catalog = [s for s in stations if s["network"] == "smn" and s.get("active")]

        from scripts.estaciones.colector_bcp import STATIONS as BCP_STATIONS
        from scripts.estaciones.colector_smn_synop import STATIONS as SMN_STATIONS
        self.assertEqual(
            {str(s["collector_id"]) for s in bcp_catalog},
            {str(k) for k in BCP_STATIONS},
        )
        self.assertEqual(
            {s["wigos_id"] for s in smn_catalog},
            set(SMN_STATIONS),
        )
        self.assertEqual(len(bcp_catalog) + len(smn_catalog), 14)


if __name__ == "__main__":
    unittest.main()
