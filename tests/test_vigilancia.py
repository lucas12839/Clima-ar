import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vigilancia as vg  # noqa: E402

AHORA = datetime(2026, 11, 23, 14, 0, tzinfo=timezone.utc)


def hace(minutos):
    return (AHORA - timedelta(minutes=minutos)).isoformat()


SANO = {"latido": hace(1), "info": {"utc": hace(30), "clave_vence": "2027-02-21 03:00:00 UTC",
                                    "actualizacion": {"utc": hace(35), "resultado": "ok"}},
        "radar_ts": hace(15), "respaldo": [{"status": "completed", "conclusion": "success"}] * 3,
        "cwop": [{"created_at": hace(5), "status": "in_progress"}]}


def revisar(**cambios):
    d = dict(SANO, **cambios)
    return {p["clave"] for p in vg.revisar(AHORA, d["latido"], d["info"], d["radar_ts"], d["respaldo"], d["cwop"])}


class VigilanciaTest(unittest.TestCase):
    def test_todo_sano(self):
        self.assertEqual(revisar(), set())

    def test_lo_que_paso_el_23_11(self):
        # Clave vencida (401): sin latido; el respaldo falla en cada consulta; 6 h sin radar.
        p = revisar(latido=hace(370), radar_ts=hace(370),
                    respaldo=[{"status": "completed", "conclusion": "failure"}] * 3)
        self.assertEqual(p, {"latido", "radar", "respaldo"})

    def test_clave_por_vencer_y_vencida(self):
        info = dict(SANO["info"], clave_vence="2026-12-06 03:00:00 UTC")
        self.assertIn("clave_por_vencer", revisar(info=info))
        info = dict(SANO["info"], clave_vence="2026-11-20 03:00:00 UTC")
        self.assertIn("clave_vencida", revisar(info=info))

    def test_sinarame_atrasado_no_es_alarma_enseguida(self):
        self.assertEqual(revisar(radar_ts=hace(45)), set())          # detector sano: espera 60 min
        self.assertEqual(revisar(radar_ts=hace(65)), {"radar"})

    def test_actualizacion_del_vps(self):
        info = dict(SANO["info"], actualizacion={"utc": hace(40), "resultado": "error", "detalle": "x"})
        self.assertEqual(revisar(info=info), {"actualizacion_error"})
        info = dict(SANO["info"], actualizacion={"utc": hace(300), "resultado": "ok"})
        self.assertEqual(revisar(info=info), {"actualizacion_parada"})

    def test_reloj_cwop_cortado(self):
        self.assertEqual(revisar(cwop=[{"created_at": hace(50), "status": "completed"}]), {"reloj_cwop"})

    def test_claves_en_el_cuerpo(self):
        texto = vg.cuerpo([{"clave": "radar", "texto": "a"}, {"clave": "latido", "texto": "b"}], AHORA)
        self.assertEqual(vg.claves_de(texto), {"radar", "latido"})
        self.assertIn("@lucas12839", texto)

    def test_avisar_abre_actualiza_y_cierra(self):
        llamadas, issues = [], []

        class G(vg.GitHub):
            def api(self, metodo, ruta, cuerpo_json=None):
                llamadas.append((metodo, ruta))
                if metodo == "GET":
                    return issues
                if metodo == "POST" and ruta == "issues":
                    issues.append({"number": 7, "title": vg.TITULO, "body": cuerpo_json["body"]})
                if metodo == "PATCH" and cuerpo_json.get("body"):
                    issues[0]["body"] = cuerpo_json["body"]
                return {}

        g = G("o/r", "t")
        radar = [{"clave": "radar", "texto": "sin radar"}]
        self.assertEqual(vg.avisar(g, radar, AHORA), "abierto")
        self.assertEqual(vg.avisar(g, radar, AHORA), "sin_cambios")
        self.assertEqual(vg.avisar(g, radar + [{"clave": "latido", "texto": "x"}], AHORA), "actualizado")
        self.assertEqual(vg.avisar(g, [{"clave": "reloj_cwop", "texto": "x"}], AHORA), "cerrado")
        self.assertIn(("PATCH", "issues/7"), llamadas)


if __name__ == "__main__":
    unittest.main()


class SaludCadenaTest(unittest.TestCase):
    def test_salud_avisa_clave_por_vencer(self):
        import salud_cadena
        info = {"utc": AHORA.isoformat(), "commit": "abc", "clave_vence": "2026-12-06 03:00:00 UTC",
                "actualizacion": {"utc": AHORA.isoformat(), "resultado": "ok"}}
        r = salud_cadena.revisar(ahora=AHORA, latido=hace(1), info_detector=info)
        self.assertTrue(any("vence el 06/12/2026" in a for a in r["alertas"]))
        self.assertEqual(r["detector"]["commit"], "abc")
        r = salud_cadena.revisar(ahora=AHORA, latido=hace(1), info_detector=None)
        self.assertFalse(any("clave" in a for a in r["alertas"]))


class RelojExternoTest(unittest.TestCase):
    def test_reloj_externo_que_deja_de_llamar(self):
        d = SANO
        def r(externo):
            return {p["clave"] for p in vg.revisar(AHORA, d["latido"], d["info"], d["radar_ts"], d["respaldo"],
                                                    d["cwop"], externo)}
        self.assertEqual(r([]), set())                                         # todavía no se instaló
        self.assertEqual(r([{"created_at": hace(10 + 10 * i)} for i in range(20)]), set())
        self.assertEqual(r([{"created_at": hace(60 + 10 * i)} for i in range(20)]), {"reloj_externo"})


class IAAtrasadaTest(unittest.TestCase):
    """23/11/2026: 4 imágenes quedaron sin analizar por un error de la IA y nadie avisó."""

    def r(self, ia=None, corridas=None):
        d = SANO
        return {p["clave"] for p in vg.revisar(AHORA, d["latido"], d["info"], d["radar_ts"], d["respaldo"],
                                                d["cwop"], [], ia, corridas)}

    def test_ia_al_dia(self):
        self.assertEqual(self.r({"radar": SANO["radar_ts"], "utc": hace(2)}), set())

    def test_radar_nuevo_todavia_analizandose_no_es_alarma(self):
        self.assertEqual(self.r({"radar": hace(25), "utc": hace(6)}), set())

    def test_radar_sin_analisis(self):
        self.assertEqual(self.r({"radar": hace(25), "utc": hace(20)}), {"ia_atrasada"})

    def test_ia_que_no_publica(self):
        self.assertEqual(self.r({"radar": SANO["radar_ts"], "utc": hace(40)}), {"ia_atrasada"})

    def test_ia_que_falla_seguido(self):
        mal = [{"status": "completed", "conclusion": "failure"}] * 2
        self.assertEqual(self.r({"radar": SANO["radar_ts"], "utc": hace(2)}, mal), {"ia_falla"})
        bien = [{"status": "completed", "conclusion": "cancelled"}, {"status": "completed", "conclusion": "failure"},
                {"status": "completed", "conclusion": "success"}]
        self.assertEqual(self.r({"radar": SANO["radar_ts"], "utc": hace(2)}, bien), set())
