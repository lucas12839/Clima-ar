import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from scraper import capturar_radar_smn as sc


class RegistroLlegadasTest(unittest.TestCase):
    def test_registra_atraso_de_sinarame_y_del_pipeline(self):
        frame = datetime(2026, 11, 19, 4, 20, tzinfo=timezone.utc)
        inicio = frame + timedelta(minutes=6)
        ahora = inicio + timedelta(minutes=2)
        with tempfile.TemporaryDirectory() as tmp:
            destino = Path(tmp) / "llegadas.jsonl"
            with mock.patch.object(sc, "LLEGADAS", destino), mock.patch.dict(
                os.environ, {"CLIMAAR_INICIO_RUN_UTC": "2026-11-19T04:26:00Z"}
            ):
                sc.registrar_llegada(frame, ahora)
                sc.registrar_llegada(frame + timedelta(minutes=10), ahora + timedelta(minutes=10))
            lineas = destino.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lineas), 2)
        r = json.loads(lineas[0])
        self.assertEqual(r["atraso_sinarame_min"], 6.0)
        self.assertEqual(r["atraso_pipeline_min"], 2.0)
        self.assertEqual(r["edad_al_capturar_min"], 8.0)

    def test_sin_inicio_de_run_no_falla(self):
        frame = datetime(2026, 11, 19, 4, 20, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            destino = Path(tmp) / "llegadas.jsonl"
            env = {k: v for k, v in os.environ.items() if k != "CLIMAAR_INICIO_RUN_UTC"}
            with mock.patch.object(sc, "LLEGADAS", destino), mock.patch.dict(os.environ, env, clear=True):
                r = sc.registrar_llegada(frame, frame + timedelta(minutes=3))
        self.assertIsNone(r["atraso_sinarame_min"])
        self.assertEqual(r["edad_al_capturar_min"], 3.0)

    def test_limita_el_tamano_del_registro(self):
        frame = datetime(2026, 11, 19, 4, 20, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            destino = Path(tmp) / "llegadas.jsonl"
            with mock.patch.object(sc, "LLEGADAS", destino), mock.patch.object(sc, "MAX_LINEAS_LLEGADAS", 3):
                for i in range(5):
                    sc.registrar_llegada(frame + timedelta(minutes=10 * i), frame + timedelta(minutes=10 * i + 1))
            self.assertEqual(len(destino.read_text(encoding="utf-8").splitlines()), 3)


class PodaHistoricoTest(unittest.TestCase):
    def test_borra_solo_frames_viejos_con_nombre_valido(self):
        ahora = datetime(2026, 11, 19, 5, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            viejo = d / "20261001000000.png"
            nuevo = d / "20261118000000.png"
            otro = d / "notas.png"
            for f in (viejo, nuevo, otro):
                f.write_bytes(b"x")
            self.assertEqual(sc.podar_historico(ahora, 30, d), 1)
            self.assertFalse(viejo.exists())
            self.assertTrue(nuevo.exists())
            self.assertTrue(otro.exists())


class NavegadorTest(unittest.TestCase):
    def test_usa_canal_configurado(self):
        pw = mock.MagicMock()
        with mock.patch.dict(os.environ, {"CLIMAAR_CHROMIUM_CHANNEL": "chrome"}):
            sc.lanzar_navegador(pw)
        pw.chromium.launch.assert_called_once_with(headless=True, channel="chrome")

    def test_si_el_canal_falla_usa_chromium_de_playwright(self):
        pw = mock.MagicMock()
        pw.chromium.launch.side_effect = [RuntimeError("sin chrome"), "browser"]
        with mock.patch.dict(os.environ, {"CLIMAAR_CHROMIUM_CHANNEL": "chrome"}):
            self.assertEqual(sc.lanzar_navegador(pw), "browser")
        self.assertEqual(pw.chromium.launch.call_args_list[-1], mock.call(headless=True))


if __name__ == "__main__":
    unittest.main()
