import unittest
from datetime import datetime, timezone

import clasificacion_tiempo as ct
from scripts.estaciones import senales_superficie as ss

AHORA = datetime(2026, 11, 19, 10, 0, tzinfo=timezone.utc)


def r(hhmm, **kw):
    return {"utc": f"2026-11-19T{hhmm}:00+00:00", **kw}


class SenalesTest(unittest.TestCase):
    def test_lluvia_desde_acumulado_diario(self):
        serie = [r("08:00", lluvia_dia=2.0), r("09:10", lluvia_dia=10.0), r("09:40", lluvia_dia=25.0), r("10:00", lluvia_dia=40.0)]
        self.assertEqual(ss.lluvia_en(serie, AHORA, 1), 30.0)
        self.assertEqual(ss.lluvia_en(serie, AHORA, 3), 38.0)

    def test_reinicio_a_medianoche_suma_lo_nuevo(self):
        serie = [r("09:20", lluvia_dia=12.0), r("09:40", lluvia_dia=1.5), r("10:00", lluvia_dia=4.0)]
        self.assertEqual(ss.lluvia_en(serie, AHORA, 1), 4.0)

    def test_frente_de_rafagas(self):
        serie = [r("09:35", temp=24.0, presion=1000.0), r("10:00", temp=18.5, presion=1003.1)]
        e = ss.resumen_estacion("BAHIA BLANCA CENTRO", serie, AHORA, {"BAHIA BLANCA CENTRO": (-38.71, -62.27)})
        self.assertTrue(e["frente_de_rafagas"])
        self.assertLess(e["distancia_bahia_km"], 2)


class ClasificacionConEstacionesTest(unittest.TestCase):
    def base(self, **kw):
        datos = dict(dbz_ciudad=30.0, dbz_llegada=None, eta_llegada_min=None, dbz_region=30.0,
                     distancia_region_km=None, velocidad_kmh=None, coherencia=None, minutos_lluvia_intensa=0)
        datos.update(kw)
        return ct.clasificar(**datos)

    def test_rafaga_medida_sube_a_potencialmente_severa(self):
        res = self.base(superficie={"rafaga_max_kmh": 85})
        self.assertEqual(res["nivel"], 8)
        self.assertFalse(res["amenazas"]["viento"]["en_calibracion"])
        self.assertIn("viento", res["etiqueta"])

    def test_lluvia_medida_confirma_inundacion(self):
        res = self.base(superficie={"lluvia_1h_mm": 35, "lluvia_3h_mm": 60})
        self.assertEqual(res["nivel"], 8)
        self.assertEqual(res["amenazas"]["inundacion"]["fuente"], "estaciones")

    def test_frente_de_rafagas_es_indicio(self):
        res = self.base(superficie={"frente_de_rafagas": True, "con_frente_de_rafagas": ["BAHIA BLANCA AERO"]})
        self.assertTrue(res["amenazas"]["viento"]["en_calibracion"])
        self.assertLess(res["nivel"], 8)

    def test_sin_radar_usa_lluvia_medida(self):
        res = ct.clasificar(dbz_ciudad=None, dbz_llegada=None, eta_llegada_min=None, dbz_region=None,
                            distancia_region_km=None, velocidad_kmh=None, coherencia=None,
                            minutos_lluvia_intensa=None, radar_disponible=False,
                            superficie={"lluvia_1h_mm": 6.0})
        self.assertEqual(res["tipo"], "lluvia_moderada")
        self.assertEqual(res["alcance"], "sin_radar")


if __name__ == "__main__":
    unittest.main()
