import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import informe_aciertos as ia  # noqa: E402


class InformeTest(unittest.TestCase):
    def test_restar_y_contador_reiniciado(self):
        actual = {"nowcast": {"capas|hasta_10min": {"20": [5, 2, 1, 10], "35": [1, 1, 1, 1], "error_abs": 30.0, "n": 6}},
                  "llegadas": {"lluvia|probable|localidad": [3, 1, 1, 9]},
                  "alarma": {"acierto": 4, "falsa_alarma": 2, "errores_min": [1, 2, 3, 9]}}
        base = {"nowcast": {"capas|hasta_10min": {"20": [2, 1, 0, 5], "35": [2, 0, 0, 0], "error_abs": 10.0, "n": 2}},
                "llegadas": {"lluvia|probable|localidad": [1, 0, 0, 4]},
                "alarma": {"acierto": 2, "falsa_alarma": 2, "errores_min": [1, 2]}}
        d = ia.restar(actual, base)
        self.assertEqual(d["nowcast"]["capas|hasta_10min"]["20"], [3, 1, 1, 5])
        self.assertEqual(d["nowcast"]["capas|hasta_10min"]["35"], [1, 1, 1, 1])   # volvió atrás: desde cero
        self.assertEqual(d["llegadas"]["lluvia|probable|localidad"], [2, 1, 1, 5])
        self.assertEqual(d["alarma"], {"acierto": 2, "falsa_alarma": 0, "errores_min": [3, 9]})

    def test_mes_nuevo_cierra_el_anterior(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            viejos = (ia.VERIF_NOWCAST, ia.ALARMA, ia.VERIF_PUNTAJE, ia.FACTOR, ia.DIR, ia.BASE_MENSUAL)
            ia.VERIF_NOWCAST, ia.ALARMA = tmp / "v.json", tmp / "a.json"
            ia.VERIF_PUNTAJE, ia.FACTOR = tmp / "p.json", tmp / "f.json"
            ia.DIR, ia.BASE_MENSUAL = tmp / "inf", tmp / "inf" / "base.json"

            def poner(a):
                ia.VERIF_NOWCAST.write_text(json.dumps({"acumulado": {"capas|hasta_20min": {
                    "20": [a, 0, 0, 1], "35": [a, 0, 0, 1], "error_abs": 0.0, "n": 0}}}))
            try:
                poner(10)
                ia.main(datetime(2026, 11, 30, 12, tzinfo=timezone.utc))
                poner(14)
                # 1/12 01:00 UTC = 30/11 22:00 en Argentina: sigue noviembre.
                ia.main(datetime(2026, 12, 1, 1, tzinfo=timezone.utc))
                self.assertFalse((tmp / "inf" / "2026-12.json").exists())
                poner(15)
                ia.main(datetime(2026, 12, 1, 4, tzinfo=timezone.utc))   # ya es diciembre
                poner(19)
                ia.main(datetime(2026, 12, 2, 4, tzinfo=timezone.utc))
                nov = json.loads((tmp / "inf" / "2026-11.json").read_text())
                dic = json.loads((tmp / "inf" / "2026-12.json").read_text())
                md = (tmp / "inf" / "2026-12.md").read_text()
            finally:
                ia.VERIF_NOWCAST, ia.ALARMA, ia.VERIF_PUNTAJE, ia.FACTOR, ia.DIR, ia.BASE_MENSUAL = viejos
        self.assertTrue(nov["cerrado"])
        self.assertEqual(nov["nowcast"]["capas"]["hasta_20min"]["casos_lluvia"], 15)
        self.assertEqual(dic["nowcast"]["capas"]["hasta_20min"]["casos_lluvia"], 4)
        self.assertIn("hasta 20 min", md)


if __name__ == "__main__":
    unittest.main()
