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
    return {"tipo": "tormenta", "utc": (AHORA - timedelta(minutes=minutos_antes)).isoformat(), "firma": firma,
            "nivel": nivel}


def test_sin_tormentas_no_publica():
    r = resumen(loc("bahia_blanca", 1))
    assert fb.situacion(r) is None
    assert fb.toca_tormenta({}, None, r, AHORA)[0] is False


def test_radar_viejo_no_publica():
    r = resumen(loc("punta_alta", 3), edad=35)
    toca, motivo = fb.toca_tormenta({}, fb.situacion(r), r, AHORA)
    assert not toca and "viejo" in motivo


def test_moderada_maximo_tres_por_dia_y_90_min():
    r = resumen(loc("punta_alta", 2), loc("medanos", 2))
    sit = fb.situacion(r)
    assert fb.toca_tormenta({}, sit, r, AHORA)[0]
    # 60 min después de la última: no.
    e = {"publicaciones": [pub(60, {"punta_alta": [2, False]})]}
    assert fb.toca_tormenta(e, sit, r, AHORA)[0] is False
    # 3 en el día: no, aunque cambie.
    e = {"publicaciones": [pub(300, {}), pub(200, {}), pub(100, {})]}
    assert fb.toca_tormenta(e, sit, r, AHORA)[0] is False


def test_moderada_sin_cambios_no_repite():
    r = resumen(loc("punta_alta", 2))
    sit = fb.situacion(r)
    e = {"publicaciones": [pub(120, {"punta_alta": [2, False]})]}
    assert fb.toca_tormenta(e, sit, r, AHORA) == (False, "sin cambios")


def test_fuerte_cada_30_min_si_cambia_y_cada_hora_igual():
    r = resumen(loc("punta_alta", 3, minutos=25))
    sit = fb.situacion(r)
    firma = sit["firma"]
    assert fb.toca_tormenta({"publicaciones": [pub(20, {})]}, sit, r, AHORA)[0] is False     # menos de 30 min
    assert fb.toca_tormenta({"publicaciones": [pub(35, {})]}, sit, r, AHORA)[0] is True      # cambió
    assert fb.toca_tormenta({"publicaciones": [pub(35, firma, 3)]}, sit, r, AHORA)[0] is False
    assert fb.toca_tormenta({"publicaciones": [pub(65, firma, 3)]}, sit, r, AHORA)[0] is True  # actualización


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
        assert "No es una alerta oficial" in texto
    assert t.startswith("⚠️ ADVERTENCIA")
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
    assert Image.open(io.BytesIO(png)).size == (1080, 1080)


def test_sin_webhook_no_publica(monkeypatch):
    monkeypatch.delenv("MAKE_WEBHOOK_URL", raising=False)
    with pytest.raises(RuntimeError, match="falta MAKE_WEBHOOK_URL"):
        fb.publicar("hola", b"png", "manana", AHORA)


def test_publicar_manda_texto_e_imagen_a_make(tmp_path, monkeypatch):
    import requests

    monkeypatch.setenv("MAKE_WEBHOOK_URL", "https://hook.make.com/secreto")
    monkeypatch.setattr(fb, "PUBLICO", tmp_path / "publico")
    enviado = {}

    class Resp:
        status_code = 200
        text = "Accepted"

    def post(url, json, timeout):
        enviado.update(url=url, json=json)
        return Resp()

    monkeypatch.setattr(requests, "post", post)
    url = fb.publicar("hola", b"PNG", "tormenta", AHORA)
    assert url == f"{fb.URL_PUBLICA}/publicaciones/20261113-153600-tormenta.png"
    assert (tmp_path / "publico" / "20261113-153600-tormenta.png").read_bytes() == b"PNG"
    assert enviado["json"] == {"mensaje": "hola", "imagen_url": url, "tipo": "tormenta"}


def test_ciclo_apagado_no_publica_pero_registra(tmp_path, monkeypatch):
    monkeypatch.setattr(fb, "ESTADO", tmp_path / "estado.json")
    monkeypatch.setenv("FB_ACTIVO", "0")
    r = resumen(loc("punta_alta", 3, nombre="Punta Alta"))
    monkeypatch.setattr(fb, "pedir_json", lambda ruta, timeout=30: r if ruta == "/resumen" else {})
    monkeypatch.setattr(fb, "pedir_bytes", lambda ruta, timeout=30: None)
    monkeypatch.setattr(fb, "publicar", lambda *a: pytest.fail("no tenía que publicar"))
    hechos = fb.ciclo(ahora=AHORA)
    assert hechos == ["tormenta: armado sin publicar"]
    # Ya registrada: en el ciclo siguiente no se repite.
    assert fb.ciclo(ahora=AHORA + timedelta(minutes=5))[-1].startswith("tormenta: no")
