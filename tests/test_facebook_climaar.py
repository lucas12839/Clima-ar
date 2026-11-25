"""Publicaciones automáticas en Facebook (24/11/2026)."""
import io
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import facebook_climaar as fb  # noqa: E402

AHORA = datetime(2026, 11, 13, 18, 36, tzinfo=timezone.utc)   # 15:36 en Argentina


def loc(id_, nivel, minutos=None, nombre=None, lat=-38.88, lon=-62.07):
    return {"id": id_, "nombre": nombre or id_, "nivel": nivel, "etiqueta": "Posible severa" if nivel == 4 else "",
            "resumen": "Lluvia fuerte", "minutos_hasta_llegada": minutos, "sobre_la_localidad": False,
            "lat": lat, "lon": lon}


def resumen(*locs, edad=6):
    return {"localidades": list(locs), "frame_edad_minutos": edad, "frame_actual_utc": "2026-11-13T18:30:00+00:00",
            "radar": {"bounds": [[-40.89, -64.93], [-36.58, -59.40]]}, "ambiente": "Ambiente poco favorable a tormentas",
            "tiempo_actual": {"disponible": True, "temperatura": 18}}


def pub(minutos_antes, firma, nivel=2):
    return {"tipo": "advertencia", "utc": (AHORA - timedelta(minutes=minutos_antes)).isoformat(), "firma": firma,
            "nivel": nivel}


def test_sin_tormentas_no_publica():
    r = resumen(loc("bahia_blanca", 1))
    assert fb.situacion(r) is None
    assert fb.toca_tormenta({}, None, r, AHORA)[0] is False


def test_radar_viejo_no_publica():
    r = resumen(loc("punta_alta", 3), edad=35)
    toca, motivo = fb.toca_tormenta({}, fb.situacion(r), r, AHORA)
    assert not toca and "viejo" in motivo


def test_va_informando_sin_llenar_la_pagina():
    r = resumen(loc("punta_alta", 2), loc("medanos", 2))
    sit = fb.situacion(r)
    firma = sit["firma"]
    assert fb.toca_tormenta({}, sit, r, AHORA)[0]
    assert fb.toca_tormenta({"publicaciones": [pub(20, {})]}, sit, r, AHORA)[0] is False
    assert fb.toca_tormenta({"publicaciones": [pub(35, {})]}, sit, r, AHORA) == (True, "cambió la situación")
    assert fb.toca_tormenta({"publicaciones": [pub(60, firma)]}, sit, r, AHORA) == (False, "sin cambios")
    assert fb.toca_tormenta({"publicaciones": [pub(95, firma)]}, sit, r, AHORA)[0] is True
    e = {"publicaciones": [pub(40 + i * 31, {}) for i in range(12)][::-1]}
    assert "máximo 12" in fb.toca_tormenta(e, sit, r, AHORA)[1]


def test_severa_sin_tope_diario():
    r = resumen(loc("punta_alta", 4, minutos=10))
    sit = fb.situacion(r)
    e = {"publicaciones": [pub(40 + i * 31, {}) for i in range(12)][::-1]}
    assert fb.toca_tormenta(e, sit, r, AHORA)[0] is True


def test_fin_una_sola_vez_cuando_se_van():
    r = resumen(loc("punta_alta", 1))
    assert fb.situacion(r) is None
    assert not fb.toca_fin({}, None, r, AHORA)
    assert not fb.toca_fin({"publicaciones": [pub(20, {})]}, None, r, AHORA)
    assert fb.toca_fin({"publicaciones": [pub(40, {})]}, None, r, AHORA)
    fin = dict(pub(10, {}), tipo="fin")
    assert not fb.toca_fin({"publicaciones": [pub(60, {}), fin]}, None, r, AHORA)
    assert not fb.toca_fin({"publicaciones": [pub(400, {})]}, None, r, AHORA)
    viejo = resumen(loc("punta_alta", 1), edad=35)
    assert not fb.toca_fin({"publicaciones": [pub(40, {})]}, None, viejo, AHORA)
    r2 = resumen(loc("medanos", 2))
    assert fb.toca_tormenta({"publicaciones": [dict(pub(40, {}), tipo="fin")]}, fb.situacion(r2), r2, AHORA)[0]


def test_fuerte_cada_30_min_si_cambia_y_cada_hora_igual():
    r = resumen(loc("punta_alta", 3, minutos=25))
    sit = fb.situacion(r)
    firma = sit["firma"]
    assert fb.toca_tormenta({"publicaciones": [pub(20, {})]}, sit, r, AHORA)[0] is False
    assert fb.toca_tormenta({"publicaciones": [pub(35, {})]}, sit, r, AHORA)[0] is True
    assert fb.toca_tormenta({"publicaciones": [pub(35, firma, 3)]}, sit, r, AHORA)[0] is False
    assert fb.toca_tormenta({"publicaciones": [pub(65, firma, 3)]}, sit, r, AHORA)[0] is True


def test_noche_una_vez_desde_las_2030_con_pronostico_de_manana():
    antes = datetime(2026, 11, 24, 23, 20, tzinfo=timezone.utc)
    hora = datetime(2026, 11, 24, 23, 35, tzinfo=timezone.utc)
    tarde = datetime(2026, 11, 25, 2, 5, tzinfo=timezone.utc)
    assert not fb.toca_noche({}, antes)
    assert fb.toca_noche({}, hora)
    assert not fb.toca_noche({"noche_fecha": "2026-11-24"}, hora)
    assert not fb.toca_noche({}, tarde)
    # 25/11/2026: la de la noche cuenta esta noche y la madrugada (de 21 a 7), no mañana.
    base = datetime(2026, 11, 24, 20, 0)
    horas = [{"hora": (base + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M"), "temperatura": 19 - i,
              "prob_lluvia": 40 if i == 6 else 0, "lluvia_mm": 1.0 if i == 6 else 0, "codigo_tiempo": 1 if i < 6 else 61,
              "viento": 10 + (12 if i >= 6 else 0), "viento_direccion": 180, "rafagas": 30 + i}
             for i in range(16)]
    pron = {"horas": horas}
    t = fb.texto_noche(resumen(loc("bahia_blanca", 1)), pron, hora)
    assert t.startswith("Buenas noches, Bahía Blanca. Así sigue el tiempo esta noche y en la madrugada:")
    assert "a la medianoche 15°" in t and "al amanecer 9°" in t and "Mínima de 8° cerca de las 7" in t
    assert "de madrugada" in t and "hasta 40 % (unos 1,0 mm)" in t and "(aumenta de madrugada)" in t
    assert fb.URL_PUBLICA in t and "alerta" not in t.lower() and "Máxima" not in t
    assert fb.PIE_INFORME in t and t.endswith("#BahíaBlanca #Clima #Tiempo")
    from PIL import Image
    png = fb.imagen_noche(resumen(loc("bahia_blanca", 1)), pron, hora)
    assert Image.open(io.BytesIO(png)).size == (1080, 1080)


def test_cambio_detecta_subida_y_llegada():
    assert fb.cambio({"a": [3, False]}, {"a": [2, False]})
    assert fb.cambio({"a": [2, True]}, {"a": [2, False]})
    assert fb.cambio({"b": [2, False]}, {"a": [2, False]})
    assert not fb.cambio({"a": [2, False]}, {"a": [3, True]})


def test_manana_una_vez_por_dia_desde_las_730():
    temprano = datetime(2026, 11, 24, 10, 20, tzinfo=timezone.utc)   # 7:20
    hora = datetime(2026, 11, 24, 10, 35, tzinfo=timezone.utc)       # 7:35
    tarde = datetime(2026, 11, 24, 14, 5, tzinfo=timezone.utc)       # 11:05
    assert not fb.toca_manana({}, temprano)
    assert fb.toca_manana({}, hora)
    assert not fb.toca_manana({"manana_fecha": "2026-11-24"}, hora)
    assert not fb.toca_manana({}, tarde)


def test_textos_con_enlace_propio_y_sin_smn():
    r = resumen(loc("punta_alta", 4, minutos=25, nombre="Punta Alta"))
    t = fb.texto_tormenta(fb.situacion(r), r, AHORA)
    pron = {"hoy": {"maxima": 27, "minima": 14, "codigo_tiempo": 95, "prob_lluvia": 80, "lluvia_mm": 12,
                    "viento": 30, "rafagas": 55, "viento_direccion": 200}}
    m = fb.texto_manana(r, pron, AHORA)
    for texto in (t, m):
        assert fb.URL_PUBLICA in texto
        assert "smn" not in texto.lower()
        assert "alerta" not in texto.lower()
        assert "No reemplaza" not in texto and "#PuntaAlta" not in texto
    assert fb.PIE_INFORME in m and fb.PIE_INFORME not in t
    assert t.startswith("⚠️ ADVERTENCIA")
    r2 = resumen(loc("medanos", 2, nombre="Médanos"))
    assert fb.texto_tormenta(fb.situacion(r2), r2, AHORA).startswith("⚠️ ADVERTENCIA · Tormentas moderadas")
    assert "posible severo" in t and "Llega en ~25 min (cerca de las 16:01)" in t
    assert "Máxima 27°" in m and "80 %" in m


def test_imagenes_se_arman():
    from PIL import Image

    r = resumen(loc("punta_alta", 3, minutos=25, nombre="Punta Alta"), loc("bahia_blanca", 1, nombre="Bahía Blanca",
                                                                         lat=-38.72, lon=-62.27))
    radar = io.BytesIO()
    Image.new("RGBA", (512, 512), (0, 200, 0, 120)).save(radar, "PNG")
    png = fb.imagen_tormenta(fb.situacion(r), r, radar.getvalue(), AHORA)
    assert Image.open(io.BytesIO(png)).size == (1080, 1350)
    png = fb.imagen_manana(r, {"hoy": {"maxima": 27, "minima": 14}}, AHORA)
    png2 = fb.imagen_tormenta({"nivel": 0, "localidades": [], "firma": {}}, r, radar.getvalue(), AHORA, fin=True)
    assert Image.open(io.BytesIO(png2)).size == (1080, 1350)
    assert "Se alejaron" in fb.texto_fin(r, AHORA) and "alerta" not in fb.texto_fin(r, AHORA).lower()
    assert Image.open(io.BytesIO(png)).size == (1080, 1080)


def test_nunca_dice_alerta():
    assert fb.sin_alerta("Alerta amarilla y ALERTAS; alerta.") == "Advertencia amarilla y ADVERTENCIAS; advertencia."
    assert fb.sin_alerta("alertado") == "alertado"


def test_sin_webhook_no_publica(monkeypatch):
    monkeypatch.delenv("MAKE_WEBHOOK_URL", raising=False)
    with pytest.raises(RuntimeError, match="falta MAKE_WEBHOOK_URL"):
        fb.publicar("hola", b"png", "manana", AHORA)


def test_publicar_manda_los_tres_campos_a_make(tmp_path, monkeypatch):
    import requests

    monkeypatch.setenv("MAKE_WEBHOOK_URL", "https://hook.make.com/secreto")
    monkeypatch.setattr(fb, "PUBLICO", tmp_path / "publico")
    png = b"\x89PNG\r\n\x1a\nDATOS"
    enviado = {}

    class Resp:
        def __init__(self, code=200, content=b""):
            self.status_code, self.content, self.text = code, content, "Accepted"

    def get(url, timeout):
        enviado["revisada"] = url
        return Resp(200, (tmp_path / "publico" / url.rsplit("/", 1)[-1]).read_bytes())

    def post(url, data, files, timeout):
        enviado.update(url=url, data=data, files=files)
        return Resp()

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", post)
    url = fb.publicar("hola", png, "advertencia", AHORA)
    assert url == "https://climaar-bahiablanca.duckdns.org/publicaciones/20261113-153600-advertencia.png"
    assert enviado["revisada"] == url
    assert enviado["data"] == {"mensaje": "hola", "imagen_url": url, "tipo": "advertencia"}
    assert enviado["files"]["imagen"] == ("20261113-153600-advertencia.png", png, "image/png")

    # Imagen que no se ve desde afuera: no se le avisa a Make.
    monkeypatch.setattr(requests, "get", lambda url, timeout: Resp(404))
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("no tenía que avisar a Make"))
    with pytest.raises(RuntimeError, match="no se ve desde afuera"):
        fb.publicar("hola", png, "manana", AHORA)


def test_ciclo_apagado_no_publica_pero_registra(tmp_path, monkeypatch):
    monkeypatch.setattr(fb, "ESTADO", tmp_path / "estado.json")
    monkeypatch.setenv("FB_ACTIVO", "0")
    r = resumen(loc("punta_alta", 3, nombre="Punta Alta"))
    monkeypatch.setattr(fb, "pedir_json", lambda ruta, timeout=30: r if ruta == "/resumen" else {})
    monkeypatch.setattr(fb, "pedir_bytes", lambda ruta, timeout=30: None)
    monkeypatch.setattr(fb, "publicar", lambda *a: pytest.fail("no tenía que publicar"))
    hechos = fb.ciclo(ahora=AHORA)
    assert hechos == ["advertencia: armado sin publicar"]
    # Ya registrada: en el ciclo siguiente no se repite.
    assert fb.ciclo(ahora=AHORA + timedelta(minutes=5))[-1].startswith("advertencia: no")


def test_espera_si_el_servidor_se_esta_reiniciando(monkeypatch):
    import requests
    import time

    llamadas = []

    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    def get(url, timeout, **kw):
        llamadas.append(url)
        if len(llamadas) < 3:
            raise requests.ConnectionError("reiniciando")
        return Resp()

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    assert fb.pedir_json("/resumen") == {"ok": True}
    assert len(llamadas) == 3
    llamadas.clear()
    monkeypatch.setattr(requests, "get", lambda url, timeout, **kw: (_ for _ in ()).throw(requests.ConnectionError()))
    with pytest.raises(requests.ConnectionError):
        fb.pedir_json("/resumen")
