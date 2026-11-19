import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import meteorologia_fusion as mf


class RespaldoEcmwfTest(unittest.TestCase):
    def escribir(self, d, minutos):
        p = Path(d) / "ambiente_futuro.json"
        p.write_text(json.dumps({
            "actualizado_utc": (datetime.now(timezone.utc) - timedelta(minutes=minutos)).isoformat(),
            "puntos": [{"punto": "BB_CENTRO", "datos": []}],
        }), encoding="utf-8")
        return p

    def test_usa_pronostico_reciente(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(mf, "AMBIENTE_FILE", self.escribir(d, 40)):
            puntos, info = mf.cargar_respaldo_ecmwf("timeout")
        self.assertIsNotNone(puntos)
        self.assertTrue(info["usado"])
        self.assertAlmostEqual(info["edad_minutos"], 40, delta=1)

    def test_rechaza_pronostico_viejo(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(mf, "AMBIENTE_FILE", self.escribir(d, 400)):
            puntos, info = mf.cargar_respaldo_ecmwf("timeout")
        self.assertIsNone(puntos)
        self.assertFalse(info["usado"])

    def test_sin_archivo(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(mf, "AMBIENTE_FILE", Path(d) / "no.json"):
            puntos, info = mf.cargar_respaldo_ecmwf("timeout")
        self.assertIsNone(puntos)


if __name__ == "__main__":
    unittest.main()
