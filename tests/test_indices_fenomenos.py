import unittest

import clasificacion_tiempo as ct
import indices_fenomenos as fen


def radar(**kw):
    datos = dict(disponible=True, dbz_ciudad=0.0, dbz_30km=0.0, dbz_llegada=None,
                 eta_llegada_min=None, velocidad_kmh=None, coherencia=None,
                 minutos_lluvia_intensa=0.0, lluvia_3h_mm=0.0, lluvia_6h_mm=0.0)
    datos.update(kw)
    return datos


class CombinacionesTest(unittest.TestCase):
    """Cada fenómeno por separado: la combinación real se informa tal cual."""

    def test_sin_nada(self):
        r = fen.evaluar(radar=radar())
        self.assertEqual(r["resumen"], "Sin fenómenos")
        self.assertEqual(r["calificacion"]["grado"], 0)

    def test_lluvia_debil_con_viento_fuerte(self):
        r = fen.evaluar(radar=radar(dbz_ciudad=25), superficie={"rafaga_max_kmh": 75})
        self.assertEqual(r["fenomenos"]["lluvia"]["grado"], 1)
        self.assertEqual(r["fenomenos"]["viento"]["grado"], 3)
        self.assertEqual(r["resumen"], "Viento fuerte con lluvia débil")
        self.assertEqual(r["calificacion"]["etiqueta"], "Fuerte (viento)")

    def test_solo_viento_sin_lluvia(self):
        r = fen.evaluar(radar=radar(), superficie={"rafaga_max_kmh": 65, "lluvia_1h_mm": 0})
        self.assertEqual(r["resumen"], "Viento fuerte, sin lluvia")

    def test_viento_por_modelo_sin_lluvia_no_llega_a_severo(self):
        r = fen.evaluar(radar=radar(), ambiente={"wind_gusts_10m": 110, "wind_speed_10m": 45})
        self.assertEqual(r["fenomenos"]["viento"]["grado"], 3)
        self.assertEqual(r["fenomenos"]["viento"]["fuente"], "modelo")

    def test_tormenta_electrica_sin_lluvia_ni_viento(self):
        r = fen.evaluar(radar=radar(), rayos={"tasa_30km_por_min": 0.5, "tasa_30km_previa_por_min": 0.4})
        self.assertEqual(r["resumen"], "Rayos aislados, sin lluvia")
        self.assertEqual(r["fenomenos"]["viento"]["grado"], 0)

    def test_lluvia_sin_rayos(self):
        r = fen.evaluar(radar=radar(dbz_ciudad=35))
        self.assertEqual(r["resumen"], "Lluvia moderada")
        self.assertEqual(r["fenomenos"]["electrica"]["grado"], 0)

    def test_viento_y_lluvia_fuertes(self):
        r = fen.evaluar(radar=radar(dbz_ciudad=45), superficie={"rafaga_max_kmh": 80})
        self.assertEqual(r["resumen"], "Lluvia fuerte con viento fuerte")
        self.assertEqual(r["calificacion"]["etiqueta"], "Fuerte (lluvia, viento)")

    def test_llovizna(self):
        r = fen.evaluar(radar=radar(dbz_ciudad=17))
        self.assertEqual(r["resumen"], "Llovizna")


class SeveroTest(unittest.TestCase):
    def test_linea_rapida_es_posible_severa_hasta_calibrar(self):
        # 16/12/2023: línea a 75-84 km/h, coherencia 0,94-0,98, núcleos >=45 dBZ cerca.
        r = fen.evaluar(radar=radar(dbz_ciudad=36, dbz_30km=50, velocidad_kmh=78, coherencia=0.95))
        v = r["fenomenos"]["viento"]
        self.assertEqual(v["grado"], 4)
        self.assertEqual(v["categoria"], "posible severo")
        self.assertEqual(r["calificacion"]["etiqueta"], "Posible severa (viento)")

    def test_rafaga_medida_confirma_severa(self):
        r = fen.evaluar(radar=radar(dbz_ciudad=36, dbz_30km=50, velocidad_kmh=78, coherencia=0.95),
                        superficie={"rafaga_max_kmh": 120})
        self.assertEqual(r["fenomenos"]["viento"]["categoria"], "severo")
        self.assertEqual(r["calificacion"]["etiqueta"], "Severa (viento)")

    def test_lluvia_acumulada_por_radar_es_posible_severa(self):
        r = fen.evaluar(radar=radar(dbz_ciudad=40, lluvia_3h_mm=33))
        self.assertEqual(r["fenomenos"]["lluvia"]["categoria"], "posible severa")

    def test_lluvia_medida_confirma_severa(self):
        r = fen.evaluar(radar=radar(dbz_ciudad=40), superficie={"lluvia_1h_mm": 35, "lluvia_3h_mm": 60})
        self.assertEqual(r["fenomenos"]["lluvia"]["categoria"], "severa")
        self.assertEqual(r["calificacion"]["etiqueta"], "Severa (lluvia)")

    def test_granizo_por_intensidad(self):
        for dbz, cat in ((48, "sin señal"), (52, "posible"), (57, "probable"), (62, "posible severo")):
            self.assertEqual(fen.evaluar(radar=radar(dbz_ciudad=dbz))["fenomenos"]["granizo"]["categoria"], cat, dbz)

    def test_isoterma_alta_baja_el_granizo(self):
        r = fen.evaluar(radar=radar(dbz_ciudad=57), ambiente={"freezing_level_height": 4800})
        self.assertEqual(r["fenomenos"]["granizo"]["categoria"], "posible")

    def test_salto_de_rayos(self):
        r = fen.evaluar(radar=radar(dbz_ciudad=48),
                        rayos={"tasa_30km_por_min": 4.0, "tasa_30km_previa_por_min": 0.8})
        self.assertEqual(r["fenomenos"]["electrica"]["grado"], 4)
        self.assertTrue(r["fenomenos"]["electrica"]["salto"])


class LlegadaTest(unittest.TestCase):
    def test_informa_como_llega(self):
        r = fen.evaluar(radar=radar(dbz_ciudad=25, dbz_llegada=56, eta_llegada_min=25))
        self.assertEqual(r["minutos_hasta_llegada"], 25)
        self.assertEqual(r["resumen_ahora"], "Lluvia débil")
        self.assertEqual(r["resumen_al_llegar"], "Lluvia fuerte (torrencial) con granizo probable")
        self.assertEqual(r["fenomenos"]["granizo"]["cuando"], "al_llegar")

    def test_lejos_no_cuenta(self):
        r = fen.evaluar(radar=radar(dbz_llegada=56, eta_llegada_min=90))
        self.assertIsNone(r["al_llegar"])


class SinRadarTest(unittest.TestCase):
    def test_sin_radar_usa_estaciones_y_rayos(self):
        r = fen.evaluar(radar={"disponible": False}, superficie={"lluvia_1h_mm": 8, "rafaga_max_kmh": 55},
                        rayos={"flashes": 3, "distancia_km": 12})
        self.assertEqual(r["fenomenos"]["lluvia"]["grado"], 2)
        self.assertEqual(r["fenomenos"]["viento"]["grado"], 3)
        self.assertEqual(r["fenomenos"]["electrica"]["grado"], 1)
        self.assertFalse(r["radar_disponible"])


class IntegracionTest(unittest.TestCase):
    def base(self, **kw):
        datos = dict(dbz_ciudad=0.0, dbz_llegada=None, eta_llegada_min=None, dbz_region=0.0,
                     distancia_region_km=None, velocidad_kmh=None, coherencia=None,
                     minutos_lluvia_intensa=0.0)
        datos.update(kw)
        return ct.clasificar(**datos)

    def test_clasificar_incluye_fenomenos(self):
        r = self.base(dbz_ciudad=25, superficie={"rafaga_max_kmh": 75})
        self.assertEqual(r["resumen"], "Viento fuerte con lluvia débil")
        self.assertIn("calificacion", r)

    def test_severo_se_mantiene_40_min(self):
        t0, t1, t2 = "2026-11-19T10:00:00+00:00", "2026-11-19T10:30:00+00:00", "2026-11-19T10:50:00+00:00"
        severo = dict(dbz_ciudad=36, dbz_30km=50, velocidad_kmh=78, coherencia=0.95)
        r1 = ct.mantener_amenazas(self.base(**severo), None, t0, None)
        r2 = ct.mantener_amenazas(self.base(dbz_ciudad=30), r1, t1, t0)
        r3 = ct.mantener_amenazas(self.base(dbz_ciudad=30), r2, t2, t1)
        self.assertEqual(r2["calificacion"]["etiqueta"], "Posible severa (viento)")
        self.assertTrue(r2["fenomenos"]["viento"]["mantenido"])
        self.assertEqual(r2["ahora"]["viento"]["grado"], 0)
        self.assertEqual(r3["calificacion"]["etiqueta"], "Débil (lluvia)")

    def test_sin_radar(self):
        r = self.base(radar_disponible=False, superficie={"lluvia_1h_mm": 4})
        self.assertEqual(r["resumen"], "Lluvia moderada")


if __name__ == "__main__":
    unittest.main()


class SateliteTest(unittest.TestCase):
    def test_topes_muy_frios_suben_granizo(self):
        r = fen.evaluar(radar=radar(dbz_ciudad=52), satelite={"min_c_15km": -64})
        self.assertEqual(r["fenomenos"]["granizo"]["categoria"], "probable")

    def test_topes_solos_no_crean_granizo(self):
        r = fen.evaluar(radar=radar(dbz_ciudad=30), satelite={"min_c_15km": -70})
        self.assertEqual(r["fenomenos"]["granizo"]["grado"], 0)

    def test_sin_radar_nube_de_tormenta_es_lluvia_probable(self):
        r = fen.evaluar(radar={"disponible": False}, satelite={"min_c_15km": -55})
        self.assertEqual(r["fenomenos"]["lluvia"]["grado"], 2)
        self.assertFalse(r["fenomenos"]["lluvia"]["confirmado"])


class AmbienteTest(unittest.TestCase):
    # 16/12/2023 a las 12 UTC (data/eventos_severos/variables_convectivas_2023-12-16.csv)
    AMB = {"cape": 1830, "wind_speed_850hPa": 30, "wind_direction_850hPa": 0,
           "wind_speed_500hPa": 90, "wind_direction_500hPa": 280,
           "temperature_850hPa": 18.3, "temperature_500hPa": -10.3, "relative_humidity_500hPa": 50}

    def test_ambiente_no_crea_fenomenos(self):
        r = fen.evaluar(radar=radar(), ambiente=self.AMB)
        self.assertEqual(r["calificacion"]["grado"], 0)
        self.assertTrue(r["ambiente"]["favorable_severas"])

    def test_tormenta_en_ambiente_favorable_da_rafagas_posibles(self):
        r = fen.evaluar(radar=radar(dbz_ciudad=47), ambiente=self.AMB)
        self.assertEqual(r["fenomenos"]["viento"]["grado"], 3)
        self.assertFalse(r["fenomenos"]["viento"]["confirmado"])

    def test_granizo_sube_con_ambiente(self):
        r = fen.evaluar(radar=radar(dbz_ciudad=52), ambiente=self.AMB)
        self.assertEqual(r["fenomenos"]["granizo"]["categoria"], "probable")


class VelocidadEstableTest(unittest.TestCase):
    def test_salto_de_velocidad_no_da_severo(self):
        # 13/11/2026 05:20 UTC: 35 -> 105 km/h en 10 min (artefacto del flujo óptico).
        r = fen.evaluar(radar=radar(dbz_ciudad=40, dbz_30km=63, velocidad_kmh=105, coherencia=0.98,
                                    velocidad_previa_kmh=35))
        self.assertEqual(r["fenomenos"]["viento"]["grado"], 3)

    def test_velocidad_repetida_da_posible_severo(self):
        # 16/12/2023: 75 y 84 km/h en dos frames seguidos.
        r = fen.evaluar(radar=radar(dbz_ciudad=40, dbz_30km=50, velocidad_kmh=84, coherencia=0.96,
                                    velocidad_previa_kmh=75))
        self.assertEqual(r["fenomenos"]["viento"]["categoria"], "posible severo")

    def test_primer_frame_sin_previa_queda_en_fuerte(self):
        r = fen.evaluar(radar=radar(dbz_ciudad=40, dbz_30km=50, velocidad_kmh=84, coherencia=0.96,
                                    velocidad_previa_kmh=None))
        self.assertEqual(r["fenomenos"]["viento"]["grado"], 3)


class AcumuladoTest(unittest.TestCase):
    def test_si_ya_paro_no_es_severa(self):
        r = fen.evaluar(radar=radar(mm_h_area=0.0, lluvia_1h_mm=0.5, lluvia_3h_mm=4, lluvia_6h_mm=60))
        self.assertEqual(r["fenomenos"]["lluvia"]["grado"], 0)
        self.assertIn("anegamientos", r["resumen"])

    def test_si_sigue_lloviendo_es_posible_severa(self):
        r = fen.evaluar(radar=radar(mm_h_area=6.0, lluvia_1h_mm=8, lluvia_3h_mm=40, lluvia_6h_mm=60))
        self.assertEqual(r["fenomenos"]["lluvia"]["categoria"], "posible severa")


class ConfiabilidadTest(unittest.TestCase):
    def test_llegada_poco_confiable_no_sube_calificacion(self):
        conf = {"casos": 195, "llego": 63, "se_desvio": 110, "se_deshizo": 22}
        r = fen.evaluar(radar=radar(dbz_llegada=56, eta_llegada_min=30, confiabilidad_llegada=conf))
        self.assertEqual(r["llegada"]["verbo"], "puede llegar")
        self.assertIn("pasaron al costado", r["llegada"]["texto"])
        self.assertEqual(r["calificacion"]["grado"], 0)
        self.assertIsNotNone(r["al_llegar"])

    def test_llegada_confiable_si_cuenta(self):
        conf = {"casos": 31, "llego": 21, "se_desvio": 8, "se_deshizo": 2}
        r = fen.evaluar(radar=radar(dbz_llegada=56, eta_llegada_min=10, confiabilidad_llegada=conf))
        self.assertEqual(r["llegada"]["verbo"], "llega")
        self.assertEqual(r["fenomenos"]["granizo"]["cuando"], "al_llegar")
