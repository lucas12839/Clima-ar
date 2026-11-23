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
                  "dia_local": f"2026-11-{10 + i % 3}", "ventana": "alineada"} for i in range(19)]
        self.assertFalse(cc.calcular_factor(filas)["valido"])
        filas.append(dict(filas[0]))
        f = cc.calcular_factor(filas)
        self.assertTrue(f["valido"])
        self.assertAlmostEqual(f["factor"], 0.4)

    def test_factor_ignora_dudosas_y_lejanas(self):
        filas = [{"confiable": False, "distancia_radar_km": 10, "estacion_mm": 50, "radar_mm": 1,
                  "dia_local": "2026-11-10", "ventana": "alineada"},
                 {"confiable": True, "distancia_radar_km": 150, "estacion_mm": 50, "radar_mm": 1,
                  "dia_local": "2026-11-10", "ventana": "alineada"},
                 {"confiable": True, "distancia_radar_km": 10, "estacion_mm": 50, "radar_mm": 1,
                  "dia_local": "2026-11-10", "ventana": "desfasada"}]
        f = cc.calcular_factor(filas)
        self.assertEqual(f["horas"], 0)
        self.assertEqual(f["por_distancia"]["100_160_km"]["horas"], 1)

    def test_factor_necesita_lluvia_suficiente(self):
        filas = [{"confiable": True, "distancia_radar_km": 10, "estacion_mm": 1.0, "radar_mm": 2.5,
                  "dia_local": f"2026-11-{10 + i % 3}", "ventana": "alineada"} for i in range(25)]
        self.assertFalse(cc.calcular_factor(filas)["valido"])   # 25 mm < 30 mm

    def test_lluvia_entre_usa_la_misma_hora(self):
        fin = datetime(2026, 11, 13, 18, 0, tzinfo=timezone.utc)
        s = [{"utc": (fin + timedelta(minutes=m)).isoformat(), "lluvia_dia": v}
             for m, v in ((-70, 0.0), (-55, 1.0), (-30, 4.0), (0, 9.0), (20, 30.0))]
        self.assertEqual(cc.lluvia_entre(s, fin - timedelta(hours=1), fin), 9.0)   # no cuenta lo de después
        self.assertIsNone(cc.lluvia_entre(s[:3], fin - timedelta(hours=1), fin))   # sin dato al final
        # Reinicio a la medianoche local (03:00 UTC) y bajada errónea sin medianoche.
        f2 = datetime(2026, 11, 13, 3, 30, tzinfo=timezone.utc)
        s2 = [{"utc": (f2 + timedelta(minutes=m)).isoformat(), "lluvia_dia": v}
              for m, v in ((-60, 10.0), (-40, 12.0), (-20, 1.0), (-10, 0.0), (0, 2.0))]
        self.assertEqual(cc.lluvia_entre(s2, f2 - timedelta(hours=1), f2), 2.0 + 1.0 + 1.0)

    def test_radar_hora_completa(self):
        fin = datetime(2026, 11, 13, 18, 0, tzinfo=timezone.utc)
        h = [{"utc": (fin - timedelta(minutes=m)).isoformat()} for m in (0, 10, 20, 30, 40, 50, 60)]
        self.assertTrue(cc.radar_hora_completa(h, fin))
        # Hueco de 30 min (como el 23/11): la suma del radar queda corta.
        self.assertFalse(cc.radar_hora_completa([r for i, r in enumerate(h) if i not in (2, 3)], fin))
        self.assertFalse(cc.radar_hora_completa(h[1:], fin))   # la imagen de la hora no está


class CicloCompletoTest(unittest.TestCase):
    def test_dos_faltas_vuelven_dudosa_la_lluvia_y_se_registra(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            senales = {"estaciones": [{"nombre": "GRUMBEIN", "utc": AHORA.isoformat(), "lluvia_1h_mm": 0.0,
                                       "temperatura_c": 20.0}]}
            nowcast = {"puntos_control": {"bcp:GRUMBEIN": {"nombre": "GRUMBEIN", "distancia_radar_km": 9,
                                                           "lluvia_1h_sin_corregir_mm": 30.0, "dbz_pico": 50}}}
            def preparar(frame):
                nowcast["radar_timestamp"] = frame.isoformat()
                (tmp / "n.json").write_text(json.dumps(nowcast))
                (tmp / "p.json").write_text(json.dumps(
                    [{"utc": (frame - timedelta(minutes=m)).isoformat()} for m in range(0, 70, 10)]))
                (tmp / "h.json").write_text(json.dumps({"GRUMBEIN": [
                    {"utc": (frame - timedelta(minutes=m)).isoformat(), "lluvia_dia": 0.0, "temp": 20.0}
                    for m in range(60, -10, -10)]}))
            (tmp / "s.json").write_text(json.dumps(senales))
            viejos = (cc.SENALES, cc.NOWCAST, cc.HISTORIAL_SUP, cc.CALIDAD, cc.REGISTRO, cc.FACTOR,
                      cc.HISTORIAL_PUNTOS)
            cc.SENALES, cc.NOWCAST, cc.HISTORIAL_SUP = tmp / "s.json", tmp / "n.json", tmp / "h.json"
            cc.CALIDAD, cc.REGISTRO, cc.FACTOR = tmp / "c.json", tmp / "r.json", tmp / "f.json"
            cc.HISTORIAL_PUNTOS = tmp / "p.json"
            frame = AHORA.replace(minute=0)   # imagen de la hora en punto, llega 30 min tarde
            try:
                preparar(frame)
                cc.main(AHORA)
                primera = json.loads((tmp / "c.json").read_text())["estaciones"]["GRUMBEIN"]
                senales["estaciones"][0]["utc"] = (AHORA + timedelta(hours=1)).isoformat()
                (tmp / "s.json").write_text(json.dumps(senales))
                preparar(frame + timedelta(hours=1))
                cc.main(AHORA + timedelta(hours=1))
                segunda = json.loads((tmp / "c.json").read_text())["estaciones"]["GRUMBEIN"]
                registro = json.loads((tmp / "r.json").read_text())
            finally:
                (cc.SENALES, cc.NOWCAST, cc.HISTORIAL_SUP, cc.CALIDAD, cc.REGISTRO, cc.FACTOR,
                 cc.HISTORIAL_PUNTOS) = viejos
        self.assertEqual(primera["estado_lluvia"], "confiable")
        self.assertEqual(segunda["estado_lluvia"], "dudosa")
        self.assertEqual(len(registro), 2)
        self.assertFalse(registro[1]["confiable"])
        self.assertEqual(registro[0]["hora"], "2026-11-13T18:00:00Z")
        self.assertEqual(registro[0]["ventana"], "alineada")


if __name__ == "__main__":
    unittest.main()
