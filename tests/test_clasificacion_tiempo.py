import unittest

import clasificacion_tiempo as ct


def base(**kw):
    datos = dict(
        dbz_ciudad=0.0, dbz_llegada=None, eta_llegada_min=None, dbz_region=0.0,
        distancia_region_km=None, velocidad_kmh=None, coherencia=None,
        minutos_lluvia_intensa=0.0,
    )
    datos.update(kw)
    return ct.clasificar(**datos)


class EscalaTest(unittest.TestCase):
    def test_escala_por_reflectividad(self):
        casos = [(0, "sin_precipitacion"), (15, "llovizna"), (25, "lluvia_debil"),
                 (35, "lluvia_moderada"), (42, "lluvia_fuerte"), (47, "tormenta_moderada"),
                 (52, "tormenta_fuerte")]
        for dbz, esperado in casos:
            self.assertEqual(base(dbz_ciudad=dbz)["tipo"], esperado, dbz)

    def test_conveccion_convierte_lluvia_fuerte_en_tormenta(self):
        r = base(dbz_ciudad=42, ambiente={"cape": 800})
        self.assertEqual(r["tipo"], "tormenta_debil")

    def test_granizo_sube_a_potencialmente_severa(self):
        r = base(dbz_ciudad=58)
        self.assertEqual(r["nivel"], 8)
        self.assertIn("granizo", r["amenazas"])
        self.assertIn("granizo", r["etiqueta"])

    def test_llegada_se_informa(self):
        r = base(dbz_ciudad=0, dbz_llegada=47, eta_llegada_min=20)
        self.assertEqual(r["alcance"], "acercandose")
        self.assertEqual(r["minutos_hasta_llegada"], 20)

    def test_viento_e_inundacion_son_indicios_en_evaluacion(self):
        r = base(dbz_ciudad=36, dbz_30km=50, lluvia_3h_mm=40, arco={"minutos": 0, "velocidad_kmh": 75})
        r = ct.mantener_amenazas(r, None, "2026-11-19T10:00:00+00:00", None)
        self.assertLess(r["nivel"], 8)
        self.assertEqual(r["indicios_en_evaluacion"], ["inundacion", "viento"])

    def test_linea_rapida_lejana_no_cuenta(self):
        r = base(dbz_ciudad=0, dbz_30km=10, dbz_region=55, velocidad_kmh=80, coherencia=0.95)
        self.assertNotIn("viento", r["amenazas"])

    def test_linea_rapida_sin_arco_no_es_amenaza(self):
        r = base(dbz_ciudad=36, dbz_30km=50, velocidad_kmh=80, coherencia=0.95)
        self.assertNotIn("viento", r["amenazas"])

    def test_arco_llegando_es_amenaza_de_viento(self):
        r = base(dbz_ciudad=0, dbz_llegada=45, eta_llegada_min=25, arco={"minutos": 25, "velocidad_kmh": 80})
        self.assertIn("viento", r["amenazas"])
        self.assertIn("25 min", r["amenazas"]["viento"]["razon"])

    def test_amenaza_se_mantiene_40_min(self):
        r1 = ct.mantener_amenazas(base(dbz_ciudad=58), None, "2026-11-19T10:00:00+00:00", None)
        r2 = ct.mantener_amenazas(base(dbz_ciudad=30), r1, "2026-11-19T10:30:00+00:00", "2026-11-19T10:00:00+00:00")
        r3 = ct.mantener_amenazas(base(dbz_ciudad=30), r2, "2026-11-19T10:50:00+00:00", "2026-11-19T10:30:00+00:00")
        self.assertIn("granizo", r2["amenazas"])
        self.assertTrue(r2["amenazas"]["granizo"]["mantenida"])
        self.assertNotIn("granizo", r3["amenazas"])

    def test_sin_radar_lo_dice(self):
        r = base(radar_disponible=False, rayos={"flashes": 12, "distancia_km": 20})
        self.assertEqual(r["alcance"], "sin_radar")
        self.assertEqual(r["tipo"], "tormenta_debil")


if __name__ == "__main__":
    unittest.main()
