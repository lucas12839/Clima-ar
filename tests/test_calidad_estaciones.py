import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "estaciones"))
import calidad_y_calibracion as cc  # noqa: E402

AHORA = datetime(2026, 11, 13, 18, 30, tzinfo=timezone.utc)


def serie(temps, lluvias=None, paso=10):
    return [{"utc": (AHORA - timedelta(minutes=paso * (len(temps) - 1 - i))).isoformat(),
             "temp": t, "lluvia_dia": (lluvias[i] if lluvias else 0.0)} for i, t in enumerate(temps)]


class ControlesTest(unittest.TestCase):
    def test_imposibles(self):
        self.assertTrue(cc.imposibles({"temperatura_c": 85}))
        self.assertFalse(cc.imposibles({"temperatura_c": 22, "rafaga_kmh": 120}))

    def test_trabada(self):
        self.assertTrue(cc.trabada(serie([15.2] * 19), True, AHORA))
        self.assertFalse(cc.trabada(serie([15.2] * 19), False, AHORA))
        self.assertFalse(cc.trabada(serie([15.2] * 18 + [15.3]), True, AHORA))

    def test_acumulado_que_baja(self):
        self.assertTrue(cc.acumulado_inconsistente(serie([15] * 3, [10.0, 12.0, 4.0])))
        self.assertFalse(cc.acumulado_inconsistente(serie([15] * 3, [10.0, 12.0, 12.0])))

    def test_desacuerdo_con_radar(self):
        # Radar (sin corregir) 20 mm x 0,4 = 8 mm encima y la estación 0: falta.
        self.assertIsNotNone(cc.desacuerdo_lluvia(0.0, 20.0, 45, 0.4))
        # Estación 15 mm sin eco cerca: falta.
        self.assertIsNotNone(cc.desacuerdo_lluvia(15.0, 0.0, 5, 0.4))
        # Lluvia en los dos: bien.
        self.assertIsNone(cc.desacuerdo_lluvia(6.0, 20.0, 45, 0.4))

    def test_factor_necesita_datos_suficientes(self):
        filas = [{"confiable": True, "distancia_radar_km": 10, "estacion_mm": 4, "radar_mm": 10,
                  "dia_local": f"2026-11-{10 + i % 3}"} for i in range(19)]
        self.assertFalse(cc.calcular_factor(filas)["valido"])
        filas.append(dict(filas[0]))
        f = cc.calcular_factor(filas)
        self.assertTrue(f["valido"])
        self.assertAlmostEqual(f["factor"], 0.4)

    def test_factor_ignora_dudosas_y_lejanas(self):
        filas = [{"confiable": False, "distancia_radar_km": 10, "estacion_mm": 50, "radar_mm": 1,
                  "dia_local": "2026-11-10"},
                 {"confiable": True, "distancia_radar_km": 150, "estacion_mm": 50, "radar_mm": 1,
                  "dia_local": "2026-11-10"}]
        self.assertEqual(cc.calcular_factor(filas)["horas"], 0)


class CicloCompletoTest(unittest.TestCase):
    def test_dos_faltas_vuelven_dudosa_la_lluvia_y_se_registra(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            senales = {"estaciones": [{"nombre": "GRUMBEIN", "utc": AHORA.isoformat(), "lluvia_1h_mm": 0.0,
                                       "temperatura_c": 20.0}]}
            nowcast = {"puntos_control": {"bcp:GRUMBEIN": {"nombre": "GRUMBEIN", "distancia_radar_km": 9,
                                                           "lluvia_1h_sin_corregir_mm": 30.0, "dbz_pico": 50}}}
            (tmp / "s.json").write_text(json.dumps(senales))
            (tmp / "n.json").write_text(json.dumps(nowcast))
            (tmp / "h.json").write_text(json.dumps({}))
            viejos = (cc.SENALES, cc.NOWCAST, cc.HISTORIAL_SUP, cc.CALIDAD, cc.REGISTRO, cc.FACTOR)
            cc.SENALES, cc.NOWCAST, cc.HISTORIAL_SUP = tmp / "s.json", tmp / "n.json", tmp / "h.json"
            cc.CALIDAD, cc.REGISTRO, cc.FACTOR = tmp / "c.json", tmp / "r.json", tmp / "f.json"
            try:
                cc.main(AHORA)
                primera = json.loads((tmp / "c.json").read_text())["estaciones"]["GRUMBEIN"]
                senales["estaciones"][0]["utc"] = (AHORA + timedelta(hours=1)).isoformat()
                (tmp / "s.json").write_text(json.dumps(senales))
                cc.main(AHORA + timedelta(hours=1))
                segunda = json.loads((tmp / "c.json").read_text())["estaciones"]["GRUMBEIN"]
                registro = json.loads((tmp / "r.json").read_text())
            finally:
                cc.SENALES, cc.NOWCAST, cc.HISTORIAL_SUP, cc.CALIDAD, cc.REGISTRO, cc.FACTOR = viejos
        self.assertEqual(primera["estado_lluvia"], "confiable")
        self.assertEqual(segunda["estado_lluvia"], "dudosa")
        self.assertEqual(len(registro), 2)
        self.assertFalse(registro[1]["confiable"])


if __name__ == "__main__":
    unittest.main()
