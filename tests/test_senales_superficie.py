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
        # BCP reinicia el acumulado a la medianoche local (03 UTC).
        ahora = datetime(2026, 11, 19, 3, 30, tzinfo=timezone.utc)
        serie = [r("02:40", lluvia_dia=12.0), r("03:05", lluvia_dia=1.5), r("03:30", lluvia_dia=4.0)]
        self.assertEqual(ss.lluvia_en(serie, ahora, 1), 4.0)

    def test_bajada_sin_medianoche_es_lectura_erronea(self):
        # Caso La Vitícola 21/11 02:47 UTC: 6,0 -> 0,0 -> 6,0 no son 6 mm nuevos.
        serie = [r("09:20", lluvia_dia=6.0), r("09:30", lluvia_dia=0.0), r("09:40", lluvia_dia=6.0),
                 r("10:00", lluvia_dia=6.4)]
        self.assertEqual(ss.lluvia_en(serie, AHORA, 1), 0.4)

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


def test_viento_cwop_no_entra_al_resumen():
    est = [{"nombre": "BAHIA BLANCA CENTRO", "viento_kmh": 33.0, "rafaga_kmh": None, "lluvia_1h_mm": 0,
            "lluvia_3h_mm": 0, "frente_de_rafagas": False},
           {"nombre": "BAHIA BLANCA NOROESTE (CWOP)", "viento_kmh": 3.2, "rafaga_kmh": 3.2, "lluvia_1h_mm": 0.4,
            "lluvia_3h_mm": 0.4, "frente_de_rafagas": False}]
    r = ss.resumir(est)
    assert (r["viento_max_kmh"], r["rafaga_max_kmh"]) == (33.0, None)
    assert r["lluvia_1h_mm"] == 0.4                     # la lluvia de la CWOP sí cuenta
    solo = ss.resumir(est[1:])
    assert (solo["viento_max_kmh"], solo["rafaga_max_kmh"]) == (None, None)


def test_anemometro_trabado_en_cero():
    from datetime import timedelta
    ahora = datetime(2026, 11, 24, 21, 40, tzinfo=timezone.utc)

    def serie(vals):
        return [{"utc": (ahora - timedelta(minutes=10 * (len(vals) - i))).isoformat(), "viento": v}
                for i, v in enumerate(vals)]

    h = {"PUNTA ALTA": serie([0.0] * 20), "BAHIA BLANCA CENTRO": serie([25.0] * 20)}
    assert ss.vientos_trabados(h, ahora) == ["PUNTA ALTA"]
    # Noche calma en toda la región: nadie está trabado.
    calma = {"PUNTA ALTA": serie([0.0] * 20), "BAHIA BLANCA CENTRO": serie([3.0] * 20)}
    assert ss.vientos_trabados(calma, ahora) == []
    # Pocas mediciones o alguna distinta de cero: no.
    assert ss.vientos_trabados({"PUNTA ALTA": serie([0.0] * 8), "X": serie([25.0] * 8)}, ahora) == []
    assert ss.vientos_trabados({"PUNTA ALTA": serie([0.0] * 19 + [4.0]), "X": serie([25.0] * 20)}, ahora) == []


def test_sierras_grandes_no_representa_a_sierra_de_la_ventana(tmp_path):
    import json as js
    locs = tmp_path / "loc.json"
    locs.write_text(js.dumps({"localidades": [{"id": "sierra_de_la_ventana", "lat": -38.1378, "lon": -61.7936}]}))
    base = {"lluvia_1h_mm": 0, "lluvia_3h_mm": 0, "frente_de_rafagas": False, "red": "bcp",
            "utc": "2026-11-25T12:21:00+00:00"}
    es = [dict(base, nombre="SIERRAS GRANDES", lat=-38.1564, lon=-61.9118, viento_kmh=35.0, rafaga_kmh=52.0),
          dict(base, nombre="VILLA VENTANA", lat=-38.0820, lon=-61.9300, viento_kmh=27.0, rafaga_kmh=40.0)]
    z = ss.por_localidad(es, locs)["sierra_de_la_ventana"]
    assert (z["rafaga_max_kmh"], z["rafaga_estacion"], z["nombres"]) == (40.0, "VILLA VENTANA", ["VILLA VENTANA"])
    assert ss.tiempo_ahora(es, locs)["sierra_de_la_ventana"]["viento"]["estacion"] == "VILLA VENTANA"
