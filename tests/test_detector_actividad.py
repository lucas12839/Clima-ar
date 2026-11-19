import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import detector_sinarame as det  # noqa: E402


AHORA = datetime(2026, 11, 19, 5, 0, tzinfo=timezone.utc)


class ActividadTest(unittest.TestCase):
    def test_ignora_runs_fantasma_viejos(self):
        runs = [
            {"created_at": "2026-11-07T07:22:08Z"},  # fantasma
            {"created_at": "2026-11-19T04:55:00Z"},  # real
            {"created_at": None},
        ]
        self.assertEqual(det.contar_runs_recientes(runs, AHORA, 60), 1)

    def test_github_activos_solo_cuenta_recientes(self):
        respuestas = {
            "queued": [{"created_at": "2026-11-07T07:22:08Z"}] * 17,
            "pending": [{"created_at": "2026-11-19T04:58:00Z"}],
            "waiting": [],
            "in_progress": [],
        }

        gh = det.GitHub("x/y", "token")

        def fake_req(metodo, ruta, cuerpo=None, accept=None):
            estado = ruta.split("status=")[1].split("&")[0]
            return json.dumps({"workflow_runs": respuestas[estado]}).encode()

        gh._req = fake_req
        self.assertEqual(gh.activos("probar-radar-smn.yml", ahora=AHORA), 1)


if __name__ == "__main__":
    unittest.main()
