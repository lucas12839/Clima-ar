"""El radar (timestamp real de SINARAME) es el reloj meteorologico."""

import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import detector_sinarame as det  # noqa: E402
import radar_estado as est  # noqa: E402
import tormenta_intelligence as engine  # noqa: E402

T0 = datetime(2026, 10, 22, 2, 0, tzinfo=timezone.utc)


def iso(dt):
    return dt.isoformat()


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.estado = self.dir / "estado_radar.json"

    def tearDown(self):
        self.tmp.cleanup()


class IdempotenciaTest(_Base):
    def test_mismo_timestamp_no_se_procesa_dos_veces(self):
        ok, _ = est.reclamar(iso(T0), self.estado, T0)
        self.assertTrue(ok)
        est.completar(iso(T0), self.estado, T0)
        ok, motivo = est.reclamar(iso(T0), self.estado, T0)
        self.assertFalse(ok)
        self.assertEqual(motivo, "ya_procesado")
        # tampoco uno mas viejo
        ok, _ = est.reclamar(iso(T0 - timedelta(minutes=10)), self.estado, T0)
        self.assertFalse(ok)

    def test_frame_nuevo_si_dispara_procesamiento(self):
        est.reclamar(iso(T0), self.estado, T0)
        est.completar(iso(T0), self.estado, T0)
        nuevo = T0 + timedelta(minutes=45)
        ok, motivo = est.reclamar(iso(nuevo), self.estado, nuevo)
        self.assertTrue(ok)
        self.assertEqual(motivo, "reclamado")

    def test_frame_nuevo_durante_procesamiento_se_encola_sin_duplicar(self):
        est.reclamar(iso(T0), self.estado, T0)
        mas = T0 + timedelta(minutes=10)
        ok, motivo = est.reclamar(iso(mas), self.estado, T0 + timedelta(minutes=1))
        self.assertFalse(ok)
        self.assertEqual(motivo, "encolado")
        # el mismo frame en curso tampoco se duplica
        ok, motivo = est.reclamar(iso(T0), self.estado, T0 + timedelta(minutes=1))
        self.assertEqual((ok, motivo), (False, "en_proceso"))
        # estado integro: sigue procesando T0, pendiente T0+10
        s = est.leer(self.estado)
        self.assertEqual(s["procesando_radar_timestamp"], iso(T0))
        self.assertEqual(s["pendiente_radar_timestamp"], iso(mas))
        est.completar(iso(T0), self.estado, T0 + timedelta(minutes=2))
        s = est.leer(self.estado)
        self.assertEqual(s["last_processed_radar_timestamp"], iso(T0))
        self.assertEqual(s["pendiente_radar_timestamp"], iso(mas))
        # el pendiente luego se procesa una vez
        ok, _ = est.reclamar(iso(mas), self.estado, mas)
        self.assertTrue(ok)

    def test_reclamos_concurrentes_solo_uno_gana(self):
        resultados = []

        def intento():
            resultados.append(est.reclamar(iso(T0), self.estado, T0)[0])

        hilos = [threading.Thread(target=intento) for _ in range(12)]
        [h.start() for h in hilos]
        [h.join() for h in hilos]
        self.assertEqual(resultados.count(True), 1)
        json.loads(self.estado.read_text())  # JSON integro

    def test_reclamo_muerto_se_recupera(self):
        est.reclamar(iso(T0), self.estado, T0)
        tarde = T0 + timedelta(minutes=est.CLAIM_TTL_MIN + 1)
        ok, _ = est.reclamar(iso(T0), self.estado, tarde)
        self.assertTrue(ok)

    def test_fallo_libera_para_reintentar(self):
        est.reclamar(iso(T0), self.estado, T0)
        est.fallar(iso(T0), "boom", self.estado, T0)
        ok, _ = est.reclamar(iso(T0), self.estado, T0)
        self.assertTrue(ok)

    def test_completar_sin_reclamo_falla(self):
        with self.assertRaises(RuntimeError):
            est.completar(iso(T0), self.estado, T0)


class PublicacionAtomicaTest(_Base):
    def test_radar_activo_nunca_es_parcial_ni_invalido(self):
        activo = self.dir / "actual.png"
        est.publicar_frame_atomico(b"COMPLETO", activo, lambda d: (True, "ok"))
        self.assertEqual(activo.read_bytes(), b"COMPLETO")

        with self.assertRaises(ValueError):
            est.publicar_frame_atomico(
                b"PARCIAL", activo, lambda d: (False, "truncado")
            )
        self.assertEqual(activo.read_bytes(), b"COMPLETO")
        self.assertFalse((self.dir / "actual.png.tmp").exists())

    def test_validador_que_explota_no_toca_el_activo(self):
        activo = self.dir / "actual.png"
        est.publicar_frame_atomico(b"A", activo)

        def malo(_):
            raise OSError("corte")

        with self.assertRaises(OSError):
            est.publicar_frame_atomico(b"B", activo, malo)
        self.assertEqual(activo.read_bytes(), b"A")


class DetectorTest(_Base):
    def armar(self, frames, procesado):
        """frames: lista mutable con el timestamp publicado por SINARAME."""
        self.reloj = {"t": T0}
        self.radar = []
        self.refrescos = []
        self.procesado = procesado
        self.activo = False

        def consultar():
            v = frames[-1]
            if isinstance(v, Exception):
                raise v
            return v

        return det.Detector(
            consultar=consultar,
            leer_procesado=lambda: self.procesado["v"],
            despachar_radar=self.radar.append,
            despachar_refresco=lambda: self.refrescos.append(1),
            hay_actividad=lambda: self.activo,
            reloj=lambda: self.reloj["t"],
            estado_path=self.dir / "det.json",
            refresco_min=10,
            redespacho_min=10,
        )

    def avanzar(self, minutos):
        self.reloj["t"] += timedelta(minutes=minutos)

    def test_sin_cambios_no_dispara_procesamiento(self):
        frames = [iso(T0)]
        d = self.armar(frames, {"v": iso(T0)})
        for _ in range(5):
            d.tick()
            self.avanzar(1)
        self.assertEqual(self.radar, [])

    def test_frame_nuevo_dispara_una_sola_vez(self):
        frames = [iso(T0)]
        proc = {"v": iso(T0)}
        d = self.armar(frames, proc)
        d.tick()
        frames.append(iso(T0 + timedelta(minutes=45)))
        self.avanzar(45)
        self.assertEqual(d.tick(), "despachar")
        for _ in range(5):  # el pipeline todavia no termino
            self.avanzar(1)
            self.assertEqual(d.tick().split("+")[0], "en_proceso")
        self.assertEqual(self.radar, [frames[-1]])

    def test_cambia_al_nuevo_frame_y_no_reprocesa_el_anterior(self):
        frames = [iso(T0)]
        proc = {"v": None}
        d = self.armar(frames, proc)
        d.tick()
        self.assertEqual(self.radar, [iso(T0)])
        proc["v"] = iso(T0)  # el pipeline termino
        self.avanzar(5)
        d.tick()
        self.assertEqual(self.radar, [iso(T0)])  # sin duplicado
        nuevo = iso(T0 + timedelta(minutes=45))
        frames.append(nuevo)
        self.avanzar(40)
        d.tick()
        self.assertEqual(self.radar, [iso(T0), nuevo])

    def test_redespacho_si_el_pipeline_no_proceso(self):
        frames = [iso(T0)]
        d = self.armar(frames, {"v": None})
        d.tick()
        self.avanzar(11)
        d.tick()
        self.assertEqual(len(self.radar), 2)  # idempotente en el claim

    def test_sinarame_caido_no_detiene_la_inteligencia(self):
        frames = [RuntimeError("SINARAME 503")]
        d = self.armar(frames, {"v": iso(T0)})
        self.assertEqual(d.tick(), "error_sinarame+refresco")
        self.assertEqual(len(self.refrescos), 1)
        self.assertEqual(self.radar, [])

    def test_refresco_entre_frames_cada_n_minutos_sin_solaparse(self):
        frames = [iso(T0)]
        d = self.armar(frames, {"v": iso(T0)})
        d.tick()
        self.assertEqual(len(self.refrescos), 1)
        self.avanzar(5)
        d.tick()
        self.assertEqual(len(self.refrescos), 1)
        self.avanzar(6)
        self.activo = True  # hay un run en curso: no se apila otro
        d.tick()
        self.assertEqual(len(self.refrescos), 1)
        self.activo = False
        d.tick()
        self.assertEqual(len(self.refrescos), 2)


class EdadRadarEIATest(unittest.TestCase):
    def nowcast(self, minutos):
        frame = datetime.now(timezone.utc) - timedelta(minutes=minutos)
        return {
            "frame_actual_utc": frame.isoformat(),
            "actividad": True,
            "movimiento_hacia_bahia": True,
            "eta_minutos": 120.0,
            "distancia_km": 80.0,
            "velocidad_kmh": 40.0,
            "confianza_movimiento": 0.8,
        }

    def test_edad_explicita(self):
        v = engine.apply_radar_age(self.nowcast(5))["vigencia_radar"]
        self.assertEqual(v["modo"], "radar_vigente")
        self.assertAlmostEqual(v["radar_age"], 5, delta=0.5)
        self.assertEqual(v["radar_timestamp"] is not None, True)

    def test_radar_viejo_no_se_presenta_como_actual(self):
        nc = engine.apply_radar_age(self.nowcast(45))
        v = nc["vigencia_radar"]
        self.assertEqual(v["modo"], "estimado_desde_ultimo_radar_valido")
        self.assertTrue(v["estimado"])
        self.assertLess(nc["eta_minutos"], 120.0)
        o = engine.origen_datos(nc)
        self.assertFalse(o["observacion_radar"]["es_actual"])
        self.assertTrue(o["extrapolacion_nowcast"]["aplicada"])
        self.assertAlmostEqual(
            o["extrapolacion_nowcast"]["minutos_extrapolados"], 45, delta=0.5
        )

    def test_radar_obsoleto_mantiene_limite(self):
        nc = engine.apply_radar_age(self.nowcast(engine.RADAR_ESTIMACION_MAX_MIN + 30))
        self.assertEqual(nc["vigencia_radar"]["modo"], "radar_obsoleto")
        self.assertFalse(nc["actividad"])
        o = engine.origen_datos(nc)
        self.assertFalse(o["extrapolacion_nowcast"]["aplicada"])

    def test_snapshot_recalcula_edad_ahora(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            p = tmp / "e.json"
            est.reclamar(iso(T0), p, T0)
            est.completar(iso(T0), p, T0)
            s = est.snapshot(p, T0 + timedelta(minutes=30))
            self.assertEqual(s["radar_age_minutos"], 30.0)
            self.assertEqual(s["vigencia_radar"], "estimado_desde_ultimo_radar_valido")
        finally:
            shutil.rmtree(tmp)

    def test_ia_sigue_funcionando_con_radar_sin_actualizar(self):
        from unittest import mock

        tmp = Path(tempfile.mkdtemp())
        cwd = os.getcwd()
        try:
            ia = tmp / "data/ia"
            shutil.copytree(ROOT / "data/ia", ia)
            (tmp / "data/radar").mkdir(parents=True)
            shutil.copy(ROOT / "data/radar/radar_nowcast.json", tmp / "data/radar")
            shutil.copy(ROOT / "data/radar/estado_radar.json", tmp / "data/radar")
            os.chdir(tmp)
            with mock.patch.multiple(
                engine,
                IA_DIR=ia,
                RADAR_FILE=tmp / "data/radar/radar_nowcast.json",
                FUSION_FILE=ia / "estado_meteorologico.json",
                OUTPUT_FILE=ia / "inteligencia_tormenta.json",
                HISTORY_FILE=ia / "historial_inteligencia_tormenta.csv",
            ):
                engine.main()
            out = json.loads((ia / "inteligencia_tormenta.json").read_text())
        finally:
            os.chdir(cwd)
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertIn("vigencia_radar", out)
        self.assertIsNotNone(out["radar_timestamp"])
        self.assertIsNotNone(out["radar_age"])
        self.assertIn("observacion_radar", out["origen_datos"])
        self.assertIn("extrapolacion_nowcast", out["origen_datos"])
        self.assertIn("prediccion_ml", out["origen_datos"])
        self.assertEqual(
            out["origen_datos"]["collector"]["last_processed_radar_timestamp"],
            "2026-10-22T01:10:00+00:00",
        )
        if out["radar_age"] > engine.RADAR_ESTIMACION_MAX_MIN:
            self.assertEqual(out["vigencia_radar"]["modo"], "radar_obsoleto")


if __name__ == "__main__":
    unittest.main()


class ScraperDetectorTest(_Base):
    def test_modo_detectar_solo_informa_timestamps(self):
        sys.path.insert(0, str(ROOT))
        from scraper import capturar_radar_smn as sc

        salida = self.dir / "c.json"
        sc.emitir_candidatos(
            [("20261022000000", "https://x/cache/RMA10/a20261022000000.png")],
            salida,
        )
        datos = json.loads(salida.read_text())
        # SINARAME publica en hora argentina (UTC-3): 00:00 ART = 03:00 UTC
        self.assertEqual(
            datos["candidatos"][0]["utc"], "2026-10-22T03:00:00+00:00"
        )
