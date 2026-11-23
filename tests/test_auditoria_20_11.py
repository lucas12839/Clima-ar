"""Auditoría de la cadena del 20/11/2026: cada arreglo con su prueba, para que
no vuelva a pasarse por alto."""
import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.estaciones import frescura  # noqa: E402
from scripts.estaciones import senales_superficie as ss  # noqa: E402
import prediccion_evolucion as pe  # noqa: E402

AHORA = datetime(2026, 11, 20, 19, 30, tzinfo=timezone.utc)


def hace(minutos):
    return (AHORA - timedelta(minutes=minutos)).isoformat()


# 1. Edad límite según cada cuánto mide la red ---------------------------------

def test_limites_por_red():
    assert frescura.es_fresca(hace(14), "bcp", AHORA)
    assert not frescura.es_fresca(hace(16), "bcp", AHORA)
    # SMN horario: llega 20-25 min tarde. Antes (15 min) nunca entraba.
    assert frescura.es_fresca(hace(25), "smn", AHORA)
    assert frescura.es_fresca(hace(89), "smn", AHORA)
    assert not frescura.es_fresca(hace(91), "smn", AHORA)
    assert frescura.es_fresca(hace(19), "cwop", AHORA)
    assert not frescura.es_fresca(hace(21), "cwop", AHORA)
    assert not frescura.es_fresca(hace(-1), "smn", AHORA)  # del futuro
    assert not frescura.es_fresca(None, "bcp", AHORA)
    assert frescura.sin_datos_hace_un_dia(25 * 60) and not frescura.sin_datos_hace_un_dia(60)


# 2. La edad se recalcula al usar el dato (CWOP de hace 2 h no es "fresca") ---

def test_cwop_vieja_marcada_valida_no_entra(tmp_path, monkeypatch):
    vacio = tmp_path / "vacio.json"
    vacio.write_text('{"stations": []}', encoding="utf-8")
    cwop = tmp_path / "cwop.json"
    cwop.write_text(json.dumps({"stations": [
        {"nombre": "VIEJA", "estado": "valida", "observation_time": hace(120), "variables": {"temperatura_c": 22.2}},
        {"nombre": "NUEVA", "estado": "valida", "observation_time": hace(5), "variables": {"temperatura_c": 15.0}},
    ]}), encoding="utf-8")
    smn = tmp_path / "smn.json"
    smn.write_text(json.dumps({"stations": [
        {"nombre": "BAHIA BLANCA AERO", "estado": "valida", "observation_time": hace(25),
         "variables": {"air_temperature": [{"value": 15.0}]}},
    ]}), encoding="utf-8")
    monkeypatch.setattr(ss, "BCP", vacio)
    monkeypatch.setattr(ss, "SMN", smn)
    monkeypatch.setattr(ss, "CWOP", cwop)
    nombres = {o["nombre"] for o in ss.observaciones_frescas(AHORA)}
    assert nombres == {"NUEVA", "BAHIA BLANCA AERO"}


def test_validar_red_smn_con_25_min_y_estaciones_muertas(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "validar_red_aud", ROOT / "scripts/estaciones/validar_red_observaciones.py")
    vr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vr)
    ahora = datetime.now(timezone.utc)
    cat = {"station_count": 3, "stations": [
        {"name": "AERO", "network": "smn", "wigos_id": "1", "active": True},
        {"name": "GRUMBEIN", "network": "bcp", "collector_id": 1, "active": True},
        {"name": "PASMAN", "network": "bcp", "collector_id": 83, "active": True},
    ]}
    (tmp_path / "c.json").write_text(json.dumps(cat))
    (tmp_path / "b.json").write_text(json.dumps({"stations": [
        {"station_id": 1, "observation_time": (ahora - timedelta(minutes=3)).isoformat()},
        {"station_id": 83, "observation_time": (ahora - timedelta(days=20)).isoformat()},
    ]}))
    (tmp_path / "s.json").write_text(json.dumps({"stations": [
        {"wigos": "1", "observation_time": (ahora - timedelta(minutes=25)).isoformat()}]}))
    vr.CATALOG, vr.BCP, vr.SMN, vr.OUTPUT = (tmp_path / "c.json", tmp_path / "b.json",
                                             tmp_path / "s.json", tmp_path / "o.json")
    assert vr.main() == 0
    out = json.loads((tmp_path / "o.json").read_text())
    assert out["smn"] == {"total": 1, "valid": 1}
    assert out["estaciones_sin_datos_24h"] == ["PASMAN"]
    assert out["max_age_minutes_por_red"] == {"bcp": 15.0, "smn": 90.0}


# 3. Numeración continua de tormentas -------------------------------------------

def test_prediccion_no_se_valida_contra_otra_tormenta_con_el_mismo_numero():
    pred = {"track_id": "5", "timestamp_origen": "2026-11-20T10:30:00Z",
            "timestamp_objetivo": "2026-11-20T18:40:00Z", "horizonte_desde_origen_minutos": 20,
            "x_estimado": 100.0, "y_estimado": 100.0, "incertidumbre_pixeles": 5}
    # La "5" actual nació a las 18:20, después de la predicción: es otra tormenta.
    track = {"track_id": 5, "primer_frame": "2026-11-20T18:20:00Z",
             "historial": [{"timestamp": "2026-11-20T18:40:00Z", "x": 300.0, "y": 300.0}]}
    pendientes, validadas = pe.validate_pending("2026-11-20T18:50:00Z", [track], [dict(pred)])
    assert validadas == []  # antes: "validada" con ~280 px de error
    assert len(pendientes) == 1
    # Si la tormenta ya existía cuando se hizo la predicción, sí se valida.
    track_vieja = dict(track, primer_frame="2026-11-20T10:00:00Z")
    _, validadas = pe.validate_pending("2026-11-20T18:50:00Z", [track_vieja], [dict(pred)])
    assert validadas and validadas[0]["validacion"]["modo"] == "MISMO_TRACK"


def test_pendiente_vieja_se_descarta():
    pred = {"track_id": "9", "timestamp_origen": "2026-11-20T10:30:00Z",
            "timestamp_objetivo": "2026-11-20T10:50:00Z", "horizonte_desde_origen_minutos": 20,
            "x_estimado": 1.0, "y_estimado": 1.0}
    pendientes, validadas = pe.validate_pending("2026-11-20T19:10:00Z", [], [pred])
    assert pendientes == [] and validadas == []
    # Con menos de una hora de atraso sigue esperando un frame atrasado.
    pred2 = dict(pred, timestamp_objetivo="2026-11-20T18:50:00Z")
    pendientes, _ = pe.validate_pending("2026-11-20T19:10:00Z", [], [pred2])
    assert len(pendientes) == 1


def test_registro_de_errores_viejo_se_descarta(tmp_path, monkeypatch):
    reg = [{"track_id": "1", "timestamp_objetivo": "2026-11-20T02:00:00Z", "horizonte_minutos": 10,
            "error_x_pixeles": 40, "error_y_pixeles": 1, "error_pixeles": 40}]
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"schema_version": 1, "registros": reg}))
    monkeypatch.setattr(pe, "CALIBRATION", p)
    assert pe.load_calibration() == []
    p.write_text(json.dumps({"schema_version": 2, "registros": reg}))
    assert len(pe.load_calibration()) == 1


def test_seguimiento_no_reinicia_la_numeracion(tmp_path, monkeypatch):
    """Cielo limpio en el estado previo -> reconstrucción. Antes volvía a 1."""
    frames = [ROOT / "data/radar/historico" / n
              for n in ("20261120185000.png", "20261120190000.png", "20261120191000.png")]
    if not all(f.exists() for f in frames):
        pytest.skip("sin las imágenes de radar de la prueba")
    radar = tmp_path / "data/radar"
    (radar / "historico").mkdir(parents=True)
    (radar / "seguimiento").mkdir()
    for f in frames:
        shutil.copy(f, radar / "historico" / f.name)
    shutil.copy(frames[-1], radar / "actual.png")
    for nombre in ("mascara_ecos_estaticos.png", "mascara_ecos_estaticos.json",
                   "mascara_ecos_estaticos_confianza.png"):
        if (ROOT / "data/radar" / nombre).exists():
            shutil.copy(ROOT / "data/radar" / nombre, radar / nombre)
    ts = datetime.strptime(frames[-1].stem, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    (radar / "radar_timestamp.txt").write_text(ts.isoformat())
    (radar / "seguimiento/estado.json").write_text(json.dumps({
        "schema_version": 8, "ultimo_radar_timestamp": (ts - timedelta(hours=3)).isoformat(),
        "siguiente_track_id": 500, "tracks": [], "tracks_activos": [], "ultimo_frame_nucleos": []}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT))
    import importlib
    import seguimiento_temporal as st
    importlib.reload(st)
    st.procesar()
    estado = json.loads((radar / "seguimiento/estado.json").read_text())
    ids = [t["track_id"] for t in estado["tracks"]]
    assert ids, "las imágenes de prueba deberían tener ecos"
    assert min(ids) >= 500
    assert estado["siguiente_track_id"] > 500


# 4-5. Velocidad y tamaño ---------------------------------------------------------

def test_resultado_final_se_escribe_compacto():
    for f in ("scripts/motor_decision_climaar.py", "scripts/tormenta_intelligence_timestamp.py",
              "scripts/tormenta_intelligence.py", "scripts/meteorologia_fusion.py"):
        texto = (ROOT / f).read_text(encoding="utf-8")
        assert 'separators=(",", ":")' in texto, f


# 6-10. Restos y versiones ------------------------------------------------------

def test_restos_borrados():
    for f in ("scripts/riesgo_severo_climaar.py", "scripts/acumular_features_rma10.py",
              "scripts/ciclo_radar.py", "data/radar/status.json", "data/radar/latest.png",
              "data/radar/radar_features_rma10.csv", "data/goes"):
        assert not (ROOT / f).exists(), f
    calidad = (ROOT / ".github/workflows/climaar-calidad.yml").read_text(encoding="utf-8")
    assert "riesgo_severo" not in calidad


def test_capturas_de_error_fuera_de_data():
    texto = (ROOT / "scraper/capturar_radar_smn.py").read_text(encoding="utf-8")
    assert 'OUT /\n                                f"captura_error_' not in texto
    assert "logs/" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_versiones_fijas():
    for linea in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#"):
            assert "==" in linea, linea


# Control de salud: detecta lo que se había pasado por alto ----------------------

def _escribir(base, rel, contenido):
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(contenido if isinstance(contenido, str) else json.dumps(contenido), encoding="utf-8")


def test_salud_cadena_avisa_y_tambien_dice_ok(tmp_path, monkeypatch):
    import salud_cadena as sc
    monkeypatch.setattr(sc, "DATA", tmp_path)
    # Situación mala (la del 20/11 antes de la auditoría)
    _escribir(tmp_path, "ia/estaciones/red_observaciones_estado.json",
              {"valid_count": 7, "station_count": 14, "bcp": {"total": 9, "valid": 7},
               "smn": {"total": 5, "valid": 0}, "estaciones_sin_datos_24h": ["PASMAN"]})
    _escribir(tmp_path, "ia/estaciones/observaciones/cwop_actual.json",
              {"stations": [{"observation_time": hace(120)}]})
    _escribir(tmp_path, "sazb/status.json", {"obsTime_utc": hace(30)})
    _escribir(tmp_path, "radar/radar_timestamp.txt", hace(18))
    _escribir(tmp_path, "radar/seguimiento/estado.json", {"siguiente_track_id": 3, "ultimo_radar_timestamp": hace(18)})
    _escribir(tmp_path, "radar/seguimiento/calibracion_prediccion.json",
              {"registros": [{"horizonte_minutos": 10, "error_pixeles": 42}] * 30})
    _escribir(tmp_path, "ia/resultado_final.json", "x" * 900 * 1024)
    _escribir(tmp_path, "radar/seguimiento/seguimiento_temporal.json",
              {"modo_seguimiento": "RECONSTRUCCION_HISTORICA"})
    malo = sc.revisar(AHORA, latido=hace(30), previo={"siguiente_numero_tormenta": 52,
                                                      "reconstrucciones_seguidas": 2})
    texto = " ".join(malo["alertas"])
    for clave in ("SMN", "PASMAN", "CWOP", "latido", "volvió atrás", "erra", "pesados", "desde cero"):
        assert clave in texto, clave
    assert malo["estado"] == "con_alertas"
    # Situación sana
    _escribir(tmp_path, "ia/estaciones/red_observaciones_estado.json",
              {"valid_count": 9, "station_count": 14, "bcp": {"total": 9, "valid": 8},
               "smn": {"total": 5, "valid": 1}, "estaciones_sin_datos_24h": []})
    _escribir(tmp_path, "ia/estaciones/observaciones/cwop_actual.json",
              {"stations": [{"observation_time": hace(8)}]})
    _escribir(tmp_path, "radar/seguimiento/estado.json", {"siguiente_track_id": 60})
    _escribir(tmp_path, "radar/seguimiento/calibracion_prediccion.json",
              {"registros": [{"horizonte_minutos": 10, "error_pixeles": 6}] * 30})
    _escribir(tmp_path, "ia/resultado_final.json", "{}")
    _escribir(tmp_path, "radar/seguimiento/seguimiento_temporal.json",
              {"modo_seguimiento": "CONTINUIDAD_ESTADO_PREVIO"})
    bueno = sc.revisar(AHORA, latido=hace(1), previo={"siguiente_numero_tormenta": 52})
    assert bueno["alertas"] == [] and bueno["estado"] == "ok"


def test_capturador_no_toca_el_estado_del_seguimiento():
    """Causa raíz: el capturador marcaba el frame nuevo en el estado del
    seguimiento antes de procesarlo y el seguimiento reconstruía todo desde
    cero en cada imagen (números de tormenta nuevos en cada frame)."""
    texto = (ROOT / "scraper/capturar_radar_smn.py").read_text(encoding="utf-8")
    assert "def guardar_estado_local" not in texto
    assert "guardar_estado_local(" not in texto


def test_seguimiento_sigue_las_mismas_tormentas_entre_imagenes(tmp_path, monkeypatch):
    # Imágenes fijas (tormentas del 20/11, 15:40-16:10 hora de Argentina). El
    # histórico se poda a los 30 días: después la prueba se saltea.
    nombres = ("20261120184000.png", "20261120185000.png", "20261120190000.png", "20261120191000.png")
    frames = [ROOT / "data/radar/historico" / n for n in nombres]
    if not all(f.exists() for f in frames):
        pytest.skip("sin las imágenes de radar de la prueba")
    radar = tmp_path / "data/radar"
    (radar / "historico").mkdir(parents=True)
    (radar / "seguimiento").mkdir()
    for nombre in ("mascara_ecos_estaticos.png", "mascara_ecos_estaticos.json",
                   "mascara_ecos_estaticos_confianza.png"):
        if (ROOT / "data/radar" / nombre).exists():
            shutil.copy(ROOT / "data/radar" / nombre, radar / nombre)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT))
    import importlib
    import seguimiento_temporal as st
    importlib.reload(st)

    def correr(frame):
        shutil.copy(frame, radar / "historico" / frame.name)
        shutil.copy(frame, radar / "actual.png")
        ts = datetime.strptime(frame.stem, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        (radar / "radar_timestamp.txt").write_text(ts.isoformat())
        return st.procesar()

    for f in frames[:3]:
        correr(f)
    antes = json.loads((radar / "seguimiento/estado.json").read_text())
    ids_antes = {t["track_id"] for t in antes["tracks"] if t.get("presente_en_ultimo_frame")}
    resultado = correr(frames[3])
    assert resultado["modo_seguimiento"] == "CONTINUIDAD_ESTADO_PREVIO"
    despues = json.loads((radar / "seguimiento/estado.json").read_text())
    ids_despues = {t["track_id"] for t in despues["tracks"] if t.get("presente_en_ultimo_frame")}
    # La mayoría de las tormentas conserva su número de una imagen a la otra.
    assert len(ids_antes & ids_despues) >= 0.5 * len(ids_antes)


def test_restos_no_vuelven_a_guardarse():
    for wf in ("climaar-intelligence.yml", "probar-radar-smn.yml"):
        texto = (ROOT / ".github/workflows" / wf).read_text(encoding="utf-8")
        assert "rm -f data/radar/status.json data/radar/latest.png data/radar/radar_features_rma10.csv" in texto


def test_salud_detecta_resultado_que_no_se_actualizo_y_guarda_desde_cuando(tmp_path, monkeypatch):
    """Re-auditoría 23/11/2026: un paso en paralelo que falla deja su archivo viejo."""
    import json as _json
    import shutil as _sh
    import salud_cadena as sc
    base = tmp_path / "data"
    _sh.copytree(sc.DATA, base, ignore=_sh.ignore_patterns("historico", "*.png", "historial"))
    monkeypatch.setattr(sc, "DATA", base)
    gen = _json.loads((base / "ia/salud_cadena.json").read_text())["generado_utc"]
    ahora = datetime.fromisoformat(gen)  # el momento del ciclo guardado
    r1 = sc.revisar(ahora=ahora, latido=None)
    frame = r1["frame_revisado"]
    assert r1["sin_actualizar_en_el_ciclo"] == [], r1["sin_actualizar_en_el_ciclo"]
    p = base / "ia/celdas_localidad.json"
    d = _json.loads(p.read_text())
    d["radar_timestamp"] = "2026-01-01T00:00:00Z"
    p.write_text(_json.dumps(d))
    r2 = sc.revisar(ahora=ahora, latido=None, previo=r1)
    assert "celdas_localidad.json" in r2["sin_actualizar_en_el_ciclo"]
    clave = [k for k in r2["alertas_desde"] if "No se actualizaron" in k][0]
    r3 = sc.revisar(ahora=ahora + timedelta(minutes=5), latido=None, previo=r2)
    # un paso que no se volvió a correr (archivo sin regenerar) también se ve
    r4 = sc.revisar(ahora=ahora + timedelta(minutes=30), latido=None, previo=r3)
    assert "nowcast_capas.json" in r4["sin_actualizar_en_el_ciclo"]
    assert r3["alertas_desde"][clave] == r2["alertas_desde"][clave]  # sigue desde la primera vez
    assert frame
