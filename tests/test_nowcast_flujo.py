import unittest
import pytest

import numpy as np

import nowcast_flujo_optico as nf
import motor_decision_climaar as motor
import tormenta_intelligence as engine
from georreferenciar_rma10 import confianza_alcance


def celda(shape, cx, cy, radio=8, dbz=50.0):
    h, w = shape
    gy, gx = np.ogrid[:h, :w]
    campo = np.full(shape, np.nan, dtype=np.float32)
    campo[(gx - cx) ** 2 + (gy - cy) ** 2 <= radio ** 2] = dbz
    return campo


class FlujoOpticoTest(unittest.TestCase):
    def setUp(self):
        self.shape = (512, 512)
        self.clutter = np.zeros(self.shape, dtype=bool)
        self.bx, self.by = nf.latlon_to_pixel(nf.BAHIA_LAT, nf.BAHIA_LON, 512, 512)

    def test_estima_velocidad_y_direccion_de_una_celda(self):
        # 10 px en 10 min hacia el este = 1 px/min ≈ 56 km/h.
        prev = celda(self.shape, 150, 250)
        act = celda(self.shape, 160, 250)
        res = nf.pronosticar(prev, act, 10.0, self.clutter)
        mov = res["flujo"]
        self.assertTrue(mov["valido"])
        self.assertAlmostEqual(mov["direccion_hacia_grados"], 90, delta=15)
        self.assertAlmostEqual(mov["velocidad_media_kmh"], 56, delta=15)
        # A 30 min la celda debería estar ~30 px más al este.
        f30 = res["campos"][30]
        ys, xs = np.where(f30 >= 35)
        self.assertAlmostEqual(float(xs.mean()), 190, delta=6)

    def test_celda_que_viene_hacia_la_ciudad_tiene_eta(self):
        x0 = self.bx - 40
        prev = celda(self.shape, x0 - 10, self.by)
        act = celda(self.shape, x0, self.by)
        res = nf.pronosticar(prev, act, 10.0, self.clutter)
        imp = nf.impacto_ciudad(res["actual"], res["flujo"])
        self.assertIsNotNone(imp["eta_tormenta_min"])
        # Borde de la celda a ~32 px del borde de la zona (10 km ≈ 10.6 px) a 1 px/min.
        self.assertTrue(15 <= imp["eta_tormenta_min"] <= 35, imp["eta_tormenta_min"])

    def test_celda_que_se_aleja_no_tiene_eta(self):
        x0 = self.bx - 40
        prev = celda(self.shape, x0 + 10, self.by)
        act = celda(self.shape, x0, self.by)
        res = nf.pronosticar(prev, act, 10.0, self.clutter)
        imp = nf.impacto_ciudad(res["actual"], res["flujo"])
        self.assertIsNone(imp["eta_tormenta_min"])

    def test_sin_ecos_no_es_valido(self):
        vacio = np.full(self.shape, np.nan, dtype=np.float32)
        res = nf.pronosticar(vacio, vacio, 10.0, self.clutter)
        self.assertFalse(res["flujo"]["valido"])

    def test_clutter_se_ignora(self):
        clutter = np.zeros(self.shape, dtype=bool)
        clutter[240:270, 240:270] = True
        campo = celda(self.shape, 255, 255, radio=10)
        limpio = nf.campo_limpio(campo, clutter)
        self.assertEqual(float(limpio[240:270, 240:270].max()), 0.0)

    def test_factores_de_confianza(self):
        self.assertEqual(nf.factor_edad(10), 1.0)
        self.assertEqual(nf.factor_edad(15), 1.0)
        self.assertAlmostEqual(nf.factor_edad(27.5), 0.7, places=2)
        self.assertEqual(nf.factor_edad(41), 0.0)
        self.assertEqual(nf.factor_alcance(100), 1.0)
        self.assertAlmostEqual(nf.factor_alcance(240), 0.5)
        self.assertEqual(confianza_alcance(150), 1.0)
        self.assertAlmostEqual(confianza_alcance(195), 0.75, places=2)


class MotorFlujoTest(unittest.TestCase):
    def flujo(self, eta, conf=0.8, dbz=48.0, ts="2026-11-19T05:00:00Z"):
        return {
            "disponible": True,
            "radar_timestamp": ts,
            "confianza": conf,
            "movimiento": {"velocidad_media_kmh": 40, "direccion_hacia_grados": 90},
            "bahia_blanca": {"eta_tormenta_desde_ahora_min": eta, "dbz_al_llegar": dbz},
        }

    def test_llegada_cercana_suma_evidencia(self):
        score, ok, res = motor.optical_flow_evidence(
            self.flujo(10), {"frame_actual_utc": "2026-11-19T05:00:00+00:00"}
        )
        self.assertTrue(ok)
        self.assertGreater(score, 0.5)
        self.assertEqual(res["eta_tormenta_min"], 10)

    def test_sin_llegada_no_suma(self):
        score, ok, _ = motor.optical_flow_evidence(
            self.flujo(None), {"frame_actual_utc": "2026-11-19T05:00:00Z"}
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_llegada_de_lluvia_en_el_umbral_no_suma(self):
        score, ok, _ = motor.optical_flow_evidence(
            self.flujo(10, dbz=37.0), {"frame_actual_utc": "2026-11-19T05:00:00Z"}
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_baja_confianza_no_suma(self):
        score, ok, _ = motor.optical_flow_evidence(
            self.flujo(10, conf=0.3), {"frame_actual_utc": "2026-11-19T05:00:00Z"}
        )
        self.assertFalse(ok)

    def test_desincronizado_no_suma(self):
        score, ok, res = motor.optical_flow_evidence(
            self.flujo(10), {"frame_actual_utc": "2026-11-19T04:50:00Z"}
        )
        self.assertFalse(ok)
        self.assertEqual(res["motivo"], "DESINCRONIZADO")


class EscalaEdadRadarTest(unittest.TestCase):
    def nowcast(self, minutos, alcance=None):
        from datetime import datetime, timedelta, timezone

        frame = datetime.now(timezone.utc) - timedelta(minutes=minutos)
        nc = {
            "frame_actual_utc": frame.isoformat(),
            "actividad": True,
            "movimiento_hacia_bahia": True,
            "eta_minutos": 120.0,
            "distancia_km": 80.0,
            "velocidad_kmh": 40.0,
            "confianza_movimiento": 1.0,
        }
        if alcance is not None:
            nc["confianza_alcance"] = alcance
        return nc

    def test_hasta_15_min_es_vigente(self):
        v = engine.apply_radar_age(self.nowcast(12))["vigencia_radar"]
        self.assertEqual(v["modo"], "radar_vigente")

    def test_27_min_baja_la_confianza(self):
        nc = engine.apply_radar_age(self.nowcast(27.5))
        self.assertEqual(nc["vigencia_radar"]["modo"], "estimado_desde_ultimo_radar_valido")
        self.assertAlmostEqual(nc["confianza_movimiento"], 0.7, delta=0.03)

    def test_mas_de_40_min_es_obsoleto_sin_actividad(self):
        nc = engine.apply_radar_age(self.nowcast(45))
        self.assertEqual(nc["vigencia_radar"]["modo"], "radar_obsoleto")
        self.assertTrue(nc["vigencia_radar"]["radar_atrasado"])
        self.assertFalse(nc["actividad"])

    def test_lejos_del_radar_baja_la_confianza(self):
        nc = engine.apply_radar_age(self.nowcast(5, alcance=0.6))
        self.assertAlmostEqual(nc["confianza_movimiento"], 0.6, places=3)


if __name__ == "__main__":
    unittest.main()


class VelocidadFisicaTest(unittest.TestCase):
    def test_movimientos_coinciden(self):
        import adaptar_rma10_intelligence as ad
        self.assertTrue(ad.movimientos_coinciden(1.0, 90, 1.2, 100))
        self.assertFalse(ad.movimientos_coinciden(2.0, 90, 0.9, 95))   # >50 %
        self.assertFalse(ad.movimientos_coinciden(1.0, 0, 1.0, 120))   # >60°
        self.assertFalse(ad.movimientos_coinciden(0.0, None, 1.0, 90))

    def test_flujo_casi_estacionario_es_valido(self):
        import adaptar_rma10_intelligence as ad
        campo = (np.zeros((512, 512), np.float32), np.zeros((512, 512), np.float32), 0.8)
        anterior = ad.CAMPO_FLUJO
        try:
            ad.CAMPO_FLUJO = campo
            r = ad.movimiento_por_flujo({"x": 200, "y": 200}, (5, 5, 2.0, 45, "seguimiento"))
        finally:
            ad.CAMPO_FLUJO = anterior
        self.assertEqual(r[4], "flujo_optico")
        self.assertEqual(r[2], 0.0)

    def test_seguimiento_rechaza_salto_sin_superposicion(self):
        import seguimiento_temporal as st
        a = {"x": 100.0, "y": 100.0, "pixeles": 200, "ancho": 16, "alto": 16, "bbox": [92, 92, 108, 108]}
        b = dict(a, x=160.0, y=100.0, bbox=[152, 92, 168, 108])  # 60 px en 10 min ≈ 340 km/h
        self.assertIsNone(st.compatibilidad(a, b, 10.0))

    def test_evolucion_descarta_saltos(self):
        import prediccion_evolucion as pe
        hist = [
            {"timestamp": "2026-11-12T20:00:00Z", "x": 100, "y": 100},
            {"timestamp": "2026-11-12T20:10:00Z", "x": 160, "y": 100},
            {"timestamp": "2026-11-12T20:20:00Z", "x": 168, "y": 100},
        ]
        muestras = pe.movement_samples(hist)
        self.assertEqual(len(muestras), 1)
        self.assertAlmostEqual(muestras[0]["speed"], 0.8)


def test_control_de_velocidad_con_varias_imagenes():
    """Caso 19/11 23:10: un solo par de imágenes dio 58 km/h y los vecinos 22-25."""
    import numpy as np

    flujo = {"velocidad_media_kmh": 58.3, "u": np.full((4, 4), 1.0, np.float32),
             "v": np.zeros((4, 4), np.float32)}
    f = nf.controlar_velocidad(dict(flujo), [22.8, 25.0])
    assert f["control_velocidad"]["corregida"] is True
    assert f["velocidad_media_kmh"] == 25.0
    assert float(f["u"][0, 0]) == pytest.approx(25.0 / 58.3, rel=1e-3)
    # Un cambio chico o sin pares previos no se toca.
    assert nf.controlar_velocidad(dict(flujo, velocidad_media_kmh=30.0), [22.8, 25.0])["control_velocidad"]["corregida"] is False
    assert nf.controlar_velocidad(dict(flujo), [])["velocidad_media_kmh"] == 58.3


def test_localidades_sin_ecos_sin_nube(tmp_path):
    """Etapa F: el análisis por localidad saca los ecos sin nube encima (canal 13 de la misma hora)."""
    import numpy as np
    from datetime import datetime, timezone
    import nowcast_flujo_optico as nf
    dbz = np.zeros((512, 512), np.float32)
    dbz[200:210, 200:210] = 55.0
    clutter = np.zeros((512, 512), bool)
    t = datetime(2026, 11, 21, 3, 0, tzinfo=timezone.utc)
    lat, lon = np.meshgrid(np.linspace(-41, -36.5, 120), np.linspace(-65, -59, 120), indexing="ij")
    c13 = tmp_path / "c13.npz"
    np.savez_compressed(c13, lat=lat.astype(np.float32), lon=lon.astype(np.float32),
                        temp_c=np.full(lat.shape, 15.0, np.float32), hora=np.array("2026-11-21T02:58:00Z"))
    limpio, info = nf.limpiar_para_localidades(dbz, clutter, t, c13)
    assert info["satelite_usado"] and limpio.max() == 0.0           # cielo despejado: eco falso
    np.savez_compressed(c13, lat=lat.astype(np.float32), lon=lon.astype(np.float32),
                        temp_c=np.full(lat.shape, -45.0, np.float32), hora=np.array("2026-11-21T02:58:00Z"))
    limpio, info = nf.limpiar_para_localidades(dbz, clutter, t, c13)
    assert limpio.max() == 55.0                                     # nube fría encima: se queda
    np.savez_compressed(c13, lat=lat.astype(np.float32), lon=lon.astype(np.float32),
                        temp_c=np.full(lat.shape, 15.0, np.float32), hora=np.array("2026-11-21T02:20:00Z"))
    limpio, info = nf.limpiar_para_localidades(dbz, clutter, t, c13)
    assert not info["satelite_usado"] and limpio.max() == 55.0      # satélite de otra hora: no se usa
    limpio, info = nf.limpiar_para_localidades(dbz, clutter, t, tmp_path / "no_existe.npz")
    assert not info["satelite_usado"]
