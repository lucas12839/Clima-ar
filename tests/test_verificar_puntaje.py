import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ / "scripts" / "estaciones"))
import verificar_puntaje as vp  # noqa: E402

FRAME = datetime(2026, 11, 21, 19, 0, tzinfo=timezone.utc)


def serie(valores, viento=None):
    """Una lectura cada 10 min desde 20 min antes de la imagen hasta 50 min después."""
    return [{"utc": (FRAME + timedelta(minutes=m)).isoformat(), "lluvia_dia": v, "rafaga": None,
             "viento": (viento or {}).get(m, 10.0)} for m, v in zip(range(-20, 60, 10), valores)]


class MedidoTest(unittest.TestCase):
    def test_viento_y_cobertura(self):
        s = serie([0] * 8, {10: 95.0, 50: 120.0})
        self.assertEqual(vp.viento_max(s, FRAME - timedelta(minutes=10), FRAME + timedelta(minutes=40)), 95.0)
        self.assertTrue(vp.cubre(s, FRAME - timedelta(minutes=10), FRAME + timedelta(minutes=40)))
        self.assertFalse(vp.cubre(s[:3], FRAME - timedelta(minutes=10), FRAME + timedelta(minutes=40)))

    def test_rayos_necesitan_las_tres_imagenes(self):
        g = {vp.iso(FRAME + timedelta(minutes=10 * i)): {"tormentas": [{"track_id": 7, "rayos_10min": 4}]}
             for i in (1, 2, 3)}
        self.assertEqual(vp.rayos_despues(g, FRAME, 7), 12)
        self.assertEqual(vp.rayos_despues(g, FRAME, 8), 0)      # la tormenta ya no estaba: 0 rayos
        del g[vp.iso(FRAME + timedelta(minutes=20))]
        self.assertIsNone(vp.rayos_despues(g, FRAME, 7))        # falta una imagen: no se sabe

    def test_resumen_por_grado(self):
        acc = {}
        for grado, viento in ((3, 95.0), (3, 50.0), (0, 20.0)):
            vp.sumar(acc, {"grados": {"rafagas": grado, "lluvia": 0, "granizo": 0},
                           "medido": {"viento_kmh": viento, "lluvia_mm": 1.0, "rayos": None}})
        r = vp.resumen(acc)["rafagas"]
        self.assertEqual(r["3"]["viento_kmh"]["pct_ge_90"], 50.0)
        self.assertEqual(r["0"]["viento_kmh"]["pct_ge_60"], 0.0)
        self.assertNotIn("viento_kmh", vp.resumen(acc)["granizo"]["0"])   # el granizo no se mide


class CicloTest(unittest.TestCase):
    def test_ciclo_una_vez_por_imagen(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            viejos = (vp.PUNTAJE_DIR, vp.GOES_DIR, vp.HISTORIAL_SUP, vp.CALIDAD, vp.SALIDA, vp.CASOS_DIR)
            vp.PUNTAJE_DIR, vp.GOES_DIR = tmp / "p", tmp / "g"
            vp.HISTORIAL_SUP, vp.CALIDAD = tmp / "h.json", tmp / "c.json"
            vp.SALIDA, vp.CASOS_DIR = tmp / "v.json", tmp / "casos"
            vp.PUNTAJE_DIR.mkdir()
            linea = {"frame_radar_utc": "2026-11-21T19:00:00Z",
                     "tormentas": [{"track_id": 7, "lat": -38.7561, "lon": -62.1065, "granizo": [0.3, 0],
                                    "rafagas": [0.6, 3], "lluvia": [0.7, 4], "sospecha": False}]}
            # Dos líneas de la misma imagen (refresco): se cuenta una vez.
            (vp.PUNTAJE_DIR / "2026-11-21.jsonl").write_text(json.dumps(linea) + "\n" + json.dumps(linea) + "\n")
            vp.HISTORIAL_SUP.write_text(json.dumps({"GRUMBEIN": serie([0, 0, 2, 8, 15, 20, 24, 25], {10: 92.0})}))
            vp.CALIDAD.write_text(json.dumps({"estaciones": {}}))
            try:
                ahora = FRAME + timedelta(minutes=60)
                vp.main(ahora)
                vp.main(ahora + timedelta(minutes=5))   # la misma imagen no se vuelve a contar
                d = json.loads(vp.SALIDA.read_text())
                casos = (vp.CASOS_DIR / "2026-11.jsonl").read_text().splitlines()
            finally:
                vp.PUNTAJE_DIR, vp.GOES_DIR, vp.HISTORIAL_SUP, vp.CALIDAD, vp.SALIDA, vp.CASOS_DIR = viejos
        self.assertEqual(len(casos), 1)
        caso = json.loads(casos[0])
        self.assertEqual(caso["estacion"], "GRUMBEIN")
        self.assertEqual(caso["medido"]["lluvia_mm"], 24.0)       # de 10 min antes a 40 min después
        self.assertEqual(caso["medido"]["viento_kmh"], 92.0)
        r = d["resumen_por_mes"]["2026-11"]
        self.assertEqual(r["lluvia"]["4"]["lluvia_mm"]["pct_ge_20"], 100.0)
        self.assertEqual(r["rafagas"]["3"]["viento_kmh"]["pct_ge_90"], 100.0)


if __name__ == "__main__":
    unittest.main()
