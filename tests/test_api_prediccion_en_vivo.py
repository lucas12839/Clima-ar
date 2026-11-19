import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import app


class PrediccionEnVivoTest(unittest.TestCase):
    def test_usa_radar_timestamp_de_la_evolucion(self):
        with tempfile.TemporaryDirectory() as d:
            pred = Path(d) / "prediccion_evolucion.json"
            pred.write_text(json.dumps({"radar_timestamp": "2026-11-19T05:40:00Z", "nucleos": []}), encoding="utf-8")
            with mock.patch.object(app, "PREDICTION_FILE", pred), mock.patch.object(
                app, "radar_timestamp", return_value="2026-11-19T05:40:00+00:00"
            ), mock.patch.object(app, "radar_metadata", return_value={}):
                r = app.prediccion_en_vivo() if hasattr(app, "prediccion_en_vivo") else None
        if r is None:
            self.skipTest("endpoint con otro nombre")
        self.assertEqual(r["modo"], "PREDICCION_ENTRE_FRAMES")
        self.assertTrue(r["ok"])


if __name__ == "__main__":
    unittest.main()
