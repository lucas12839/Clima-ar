import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from goes19 import topes_localidad as tl

LOCS = [{"id": "bahia_blanca", "lat": -38.7183, "lon": -62.2663}]


class TopesTest(unittest.TestCase):
    def test_minimo_y_enfriamiento(self):
        lat = np.array([-38.72, -38.72, -39.5])
        lon = np.array([-62.27, -62.30, -62.27])
        with tempfile.TemporaryDirectory() as tmp:
            previo = Path(tmp) / "s.json"
            previo.write_text(json.dumps({"version": "1.6", "products": {"abi_c13": {
                "source": {"end_time_utc": "2026-11-19T10:00:00Z"},
                "por_localidad": {"bahia_blanca": {"p10_c_30km": -30.0}}}}}))
            r = tl.por_localidad(np.array([223.15, 233.15, 290.0]), lat, lon,
                                 "2026-11-19T10:10:00Z", previo, LOCS)["bahia_blanca"]
        self.assertEqual(r["min_c_15km"], -50.0)
        self.assertLess(r["p10_c_30km"], -48)
        self.assertGreater(r["enfriamiento_c_10min"], 15)

    def test_sin_estado_previo(self):
        r = tl.por_localidad(np.array([250.0]), np.array([-38.72]), np.array([-62.27]),
                             "2026-11-19T10:10:00Z", Path("/no/existe.json"), LOCS)["bahia_blanca"]
        self.assertNotIn("enfriamiento_c_10min", r)


if __name__ == "__main__":
    unittest.main()
