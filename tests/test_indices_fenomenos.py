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
        r = fen.evaluar(radar=radar(), superficie={"rafaga_max_kmh": 75, "lluvia_1h_mm": 0})
        self.assertEqual(r["resumen"], "Viento fuerte, sin lluvia")
        # Cortes del SMN (etapa F): una ráfaga de 65 km/h es "moderado".
        r = fen.evaluar(radar=radar(), superficie={"rafaga_max_kmh": 65, "lluvia_1h_mm": 0})
        self.assertEqual(r["resumen"], "Viento moderado, sin lluvia")

    def test_viento_por_modelo_sin_lluvia_no_llega_a_severo(self):
        r = fen.evaluar(radar=radar(), ambiente={"wind_gusts_10m": 110, "wind_speed_10m": 45})
        self.assertEqual(r["fenomenos"]["viento"]["grado"], 3)
        self.assertEqual(r["fenomenos"]["viento"]["fuente"], "modelo")

    def test_tormenta_electrica_sin_lluvia_ni_viento(self):
        r = fen.evaluar(radar=radar(), rayos={"tasa_30km_por_min": 0.3, "tasa_30km_previa_por_min": 0.2})
        self.assertEqual(r["resumen"], "Rayos aislados, sin lluvia")
        r = fen.evaluar(radar=radar(), rayos={"tasa_30km_por_min": 1.0, "tasa_30km_previa_por_min": 0.8})
        self.assertEqual(r["fenomenos"]["electrica"]["grado"], 2)
        self.assertEqual(r["resumen"], "Rayos seguidos, sin lluvia")
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
    def test_linea_rapida_sin_arco_ya_no_da_viento(self):
        # La velocidad sola acertaba 15 %: ya no se muestra.
        r = fen.evaluar(radar=radar(dbz_ciudad=36, dbz_30km=50, velocidad_kmh=78, coherencia=0.95))
        self.assertLess(r["fenomenos"]["viento"]["grado"], 3)

    def test_eco_en_arco_encima_da_rafagas_fuertes_posibles(self):
        r = fen.evaluar(radar=radar(dbz_ciudad=36, dbz_30km=50,
                                    arco={"minutos": 0, "distancia_km": 3, "velocidad_kmh": 70}))
        v = r["fenomenos"]["viento"]
        self.assertEqual(v["grado"], 3)
        self.assertFalse(v["confirmado"])
        self.assertTrue(any("eco en arco" in x for x in v["razones"]))
        self.assertEqual(r["ahora"]["viento"]["grado"], 3)

    def test_eco_en_arco_llegando_va_al_llegar(self):
        r = fen.evaluar(radar=radar(dbz_ciudad=0, dbz_llegada=45, eta_llegada_min=30,
                                    arco={"minutos": 30, "distancia_km": 40, "velocidad_kmh": 80}))
        self.assertEqual(r["ahora"]["viento"]["grado"], 0)
        self.assertEqual(r["al_llegar"]["viento"]["grado"], 3)

    def test_arco_y_rafaga_medida_da_severo(self):
        r = fen.evaluar(radar=radar(dbz_ciudad=36, arco={"minutos": 0, "velocidad_kmh": 70}),
                        superficie={"rafaga_max_kmh": 120})
        self.assertEqual(r["fenomenos"]["viento"]["categoria"], "severo")

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
        for dbz, cat in ((48, "sin señal"), (52, "posible"), (57, "probable"), (62, "fuerte")):
            self.assertEqual(fen.evaluar(radar=radar(dbz_ciudad=dbz))["fenomenos"]["granizo"]["categoria"], cat, dbz)
        # Severo (>= 2 cm) necesita las dos pruebas aparte del radar: tope muy frío Y muchos rayos.
        rayos = {"tasa_30km_por_min": 4.0, "tasa_30km_previa_por_min": 3.5}
        r = fen.evaluar(radar=radar(dbz_ciudad=62), rayos=rayos)
        self.assertEqual(r["fenomenos"]["granizo"]["categoria"], "fuerte")
        r = fen.evaluar(radar=radar(dbz_ciudad=62), satelite={"min_c_15km": -65.0})
        self.assertEqual(r["fenomenos"]["granizo"]["grado"], 3)
        r = fen.evaluar(radar=radar(dbz_ciudad=62), rayos=rayos, satelite={"min_c_15km": -65.0})
        self.assertEqual(r["fenomenos"]["granizo"]["categoria"], "posible severo")
        # 57 dBZ subido a "fuerte" por el satélite no llega a severo: falta el núcleo de 60.
        r = fen.evaluar(radar=radar(dbz_ciudad=57), rayos=rayos, satelite={"min_c_15km": -65.0})
        self.assertLess(r["fenomenos"]["granizo"]["grado"], 4)

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


class EscalaVientoTest(unittest.TestCase):
    """Escala de Lucas (23/11/2026): 30 / 50 / 70 / 90 km/h de ráfaga."""

    def test_cortes(self):
        for kmh, grado in ((29, 0), (30, 1), (49, 1), (50, 2), (69, 2), (70, 3), (89, 3), (90, 4)):
            self.assertEqual(fen.grado_viento_kmh(kmh), grado, kmh)

    def test_sostenido_45_es_fuerte(self):
        r = fen.evaluar(radar=radar(), superficie={"rafaga_max_kmh": 50, "viento_max_kmh": 46})
        self.assertEqual(r["fenomenos"]["viento"]["grado"], 3)


class SinRadarTest(unittest.TestCase):
    def test_sin_radar_usa_estaciones_y_rayos(self):
        r = fen.evaluar(radar={"disponible": False}, superficie={"lluvia_1h_mm": 8, "rafaga_max_kmh": 72},
                        rayos={"flashes": 3, "distancia_km": 12})
        self.assertEqual(r["fenomenos"]["lluvia"]["grado"], 2)
        self.assertEqual(r["fenomenos"]["viento"]["grado"], 3)
        self.assertEqual(r["fenomenos"]["electrica"]["grado"], 2)
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
        severo = dict(dbz_ciudad=36, dbz_30km=50, superficie={"rafaga_max_kmh": 95})
        r1 = ct.mantener_amenazas(self.base(**severo), None, t0, None)
        r2 = ct.mantener_amenazas(self.base(dbz_ciudad=30), r1, t1, t0)
        r3 = ct.mantener_amenazas(self.base(dbz_ciudad=30), r2, t2, t1)
        self.assertEqual(r2["calificacion"]["etiqueta"], r1["calificacion"]["etiqueta"])
        self.assertEqual(r1["fenomenos"]["viento"]["grado"], 4)
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

    def test_sin_radar_nube_de_tormenta_con_rayos_es_lluvia_probable(self):
        # Punto 10 (simular_sin_radar.py): tope <= -60 °C y al menos un rayo a <= 15 km.
        r = fen.evaluar(radar={"disponible": False}, satelite={"min_c_15km": -62},
                        rayos={"rayos_15km_10min": 2})
        self.assertEqual(r["fenomenos"]["lluvia"]["grado"], 2)
        self.assertFalse(r["fenomenos"]["lluvia"]["confirmado"])

    def test_sin_radar_tope_solo_no_alcanza(self):
        # Antes: tope <= -50 °C = lluvia probable (88 % falsas alarmas en la simulación).
        for sat, ray in (({"min_c_15km": -55}, {"rayos_15km_10min": 3}), ({"min_c_15km": -65}, {})):
            r = fen.evaluar(radar={"disponible": False}, satelite=sat, rayos=ray)
            self.assertEqual(r["fenomenos"]["lluvia"]["grado"], 0)

    def test_sin_radar_usa_el_radar_viejo_movido(self):
        mov = {"dbz_p75": 38, "mm_h_medio": 6.0, "edad_min": 55}
        r = fen.evaluar(radar={"disponible": False, "movido": mov})
        self.assertGreaterEqual(r["fenomenos"]["lluvia"]["grado"], 1)
        self.assertEqual(r["fenomenos"]["lluvia"]["fuente"], "radar_movido")
        self.assertFalse(r["fenomenos"]["lluvia"]["confirmado"])
        # Más de 90 min: no está probado, no se usa.
        viejo = fen.evaluar(radar={"disponible": False, "movido": {**mov, "edad_min": 95}})
        self.assertEqual(viejo["fenomenos"]["lluvia"]["grado"], 0)


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


class GranizoConfirmacionTests(unittest.TestCase):
    """19/11 23:20: 56 dBZ, CAPE 0, tope -37 °C y casi sin rayos -> no "probable"."""

    RADAR = {"dbz_ciudad": 10.0, "dbz_llegada": 56.0, "eta_llegada_min": 8.4, "velocidad_kmh": 20.0,
             "coherencia": 0.9}

    def test_solo_radar_baja_un_grado(self):
        r = fen.evaluar(radar=self.RADAR, rayos={"tasa_30km_por_min": 0.1, "tasa_30km_previa_por_min": 0.1,
                                                 "distancia_km": 26.0},
                        ambiente={"cape": 0.0}, satelite={"min_c_15km": -37.0})
        g = r["al_llegar"]["granizo"]
        self.assertEqual(g["grado"], 1)
        self.assertTrue(any("solo lo indica el radar" in x for x in g["razones"]))

    def test_con_confirmacion_se_mantiene(self):
        for kwargs in ({"rayos": {"tasa_30km_por_min": 4.0, "tasa_30km_previa_por_min": 3.5, "distancia_km": 10.0},
                        "ambiente": {"cape": 0.0}, "satelite": {"min_c_15km": -37.0}},
                       {"ambiente": {"cape": 1200.0}, "satelite": {"min_c_15km": -37.0}},
                       {"ambiente": {"cape": 0.0}, "satelite": {"min_c_15km": -55.0}}):
            r = fen.evaluar(radar=self.RADAR, **kwargs)
            self.assertEqual(r["al_llegar"]["granizo"]["grado"], 2, kwargs)

    def test_sin_datos_no_se_baja(self):
        r = fen.evaluar(radar=self.RADAR)
        self.assertEqual(r["al_llegar"]["granizo"]["grado"], 2)
