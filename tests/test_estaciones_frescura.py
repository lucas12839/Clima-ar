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

    def test_cada_red_tiene_su_limite_inclusivo(self):
        # BCP mide cada minuto: 15 min. SMN es horario y llega 20-25 min tarde: 90 min.
        for checker, limite in ((bcp_is_fresh, 15), (smn_is_fresh, 90)):
            self.assertTrue(checker(0))
            self.assertTrue(checker(limite))
            self.assertFalse(checker(limite + 0.001))
            self.assertFalse(checker(-0.001))
            self.assertFalse(checker(None))

    def test_bahia_aero_con_25_minutos_entra(self):
        # Antes (15 min para todos) el SMN no entraba nunca.
        self.assertTrue(smn_is_fresh(25))

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
        # 23/11/2026: las del SMN, Cabildo, San Eloy, Pasman y Pigüé quedaron desactivadas.
        self.assertEqual(smn_catalog, [])
        self.assertTrue({s["wigos_id"] for s in stations if s["network"] == "smn"} <= set(SMN_STATIONS))
        self.assertEqual(len(bcp_catalog), 10)


if __name__ == "__main__":
    unittest.main()


def test_colector_bcp_en_paralelo_reintenta_y_mantiene_el_orden(tmp_path, monkeypatch):
    import json as _json
    import scripts.estaciones.colector_bcp as cb
    llamadas = {}

    def falso(sid, name):
        llamadas[sid] = llamadas.get(sid, 0) + 1
        if sid == 1 and llamadas[sid] == 1:
            raise TimeoutError("corte de red")
        if sid == 15:
            raise TimeoutError("caída")
        return {"ok": True, "station_id": sid, "nombre": name, "estado": "valida", "fresh": True}

    monkeypatch.setattr(cb, "collect", falso)
    monkeypatch.setattr(cb, "OUTPUT", tmp_path / "bcp.json")
    monkeypatch.setattr(cb.time, "sleep", lambda s: None)
    cb.main()
    d = _json.loads((tmp_path / "bcp.json").read_text())
    assert [s["station_id"] for s in d["stations"]] == list(cb.STATIONS)
    assert llamadas[1] == 2 and d["stations"][0]["estado"] == "valida"   # se recuperó
    caida = [s for s in d["stations"] if s["station_id"] == 15][0]
    assert caida["estado"] == "error" and llamadas[15] == 2
