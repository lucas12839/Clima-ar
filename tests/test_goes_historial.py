"""Regla 7: casos guardados de satélite y rayos por tormenta."""
import json
from datetime import datetime, timezone

from goes19 import historial as h

FRAME = "2026-11-19T23:30:00Z"


def _estado(tormentas, estado="ok", frame=FRAME):
    return {
        "version": "1.7",
        "sincronizacion": {"inicio_barrido_referencia_utc": "2026-11-19T23:20:21Z",
                           "productos": {"abi_c13": {"estado": "sincronizado"}}},
        "glm": {"rayos_frame_radar": {"cobertura_min": 10.0}},
        "por_tormenta": {"estado": estado, "frame_radar_utc": frame,
                         "topes_sobresalientes_zona": 1, "tormentas": tormentas},
    }


T = {"track_id": 7, "latitud": -38.71234567, "longitud": -62.98765, "radio_km": 14.1, "dbz_max": 55.123,
     "movimiento_km_min": {"este": 0.9, "norte": 0.1},
     "c13": {"estado": "ok", "min_c": -61.2, "p10_c": -59.0, "enfriamiento_c_10min": 4.5,
             "sincronizacion": {"desfase_min": 1.6}},
     "c08": {"min_c": -60.0}, "altura_topes": {"max_km": 11.2}, "presion_topes": {"min_hpa": 210.0},
     "rayos": {"cantidad": 40, "cobertura_min": 10.0},
     "senales_severas": {"topes_sobresalientes": {"cantidad": 1, "mas_frio_c": -66.0, "max_diferencia_yunque_c": 8.1},
                         "vapor_menos_infrarrojo": {"max_c": 1.5}, "altura_topes_2km": {"max_km": 12.4, "p90_km": 11.8},
                         "fase": {"fraccion_hielo_en_nubes": 1.0},
                         "congelamiento": {"nivel": {"metros": 3900}, "tope_sobre_congelamiento_km": 7.9}},
     "altura_paralaje_km": 11.8}


def test_guarda_una_linea_compacta_por_imagen(tmp_path):
    assert h.guardar(_estado([T]), tmp_path).startswith("guardado: 1 tormentas")
    path = tmp_path / "2026-11-19.jsonl"
    caso = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    t = caso["tormentas"][0]
    assert caso["frame_radar_utc"] == FRAME and caso["topes_sobresalientes_zona"] == 1
    assert t["track_id"] == 7 and t["lat"] == -38.7123 and t["ot_cantidad"] == 1
    assert t["tope_sobre_0c_km"] == 7.9 and t["rayos_10min"] == 40 and t["c13_enfriamiento_c_10min"] == 4.5
    assert len(path.read_text(encoding="utf-8")) < 1500  # compacto
    # La misma imagen de radar no se repite (Intelligence puede correr dos veces).
    assert h.guardar(_estado([T]), tmp_path).startswith("ya guardado")
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_sin_tormentas_o_error_no_guarda_y_no_corta(tmp_path):
    assert h.guardar(_estado([], estado="sin_tormentas"), tmp_path) == "sin tormentas: no se guarda"
    assert not list(tmp_path.glob("*.jsonl"))
    assert h.guardar(_estado([T], frame="no-es-hora"), tmp_path).startswith("ERROR")


def test_leer_casos_y_limpieza_de_viejos(tmp_path):
    h.guardar(_estado([T]), tmp_path)
    h.guardar(_estado([T], frame="2026-11-20T00:10:00Z"), tmp_path)
    viejo = tmp_path / "2025-01-01.jsonl"
    viejo.write_text("{}\n", encoding="utf-8")
    h.guardar(_estado([T], frame="2026-11-20T00:20:00Z"), tmp_path)
    assert not viejo.exists()  # más de 400 días
    casos = h.leer(datetime(2026, 11, 19, 23, 0, tzinfo=timezone.utc),
                   datetime(2026, 11, 20, 0, 15, tzinfo=timezone.utc), tmp_path)
    assert [c["frame_radar_utc"] for c in casos] == [FRAME, "2026-11-20T00:10:00Z"]


def test_reemplaza_si_llega_mas_satelite_para_la_misma_imagen(tmp_path):
    sin = _estado([T])
    sin["sincronizacion"]["productos"] = {"abi_c13": None}
    assert h.guardar(sin, tmp_path).startswith("guardado")
    assert h.guardar(_estado([T]), tmp_path).startswith("reemplazado")
    renglones = (tmp_path / "2026-11-19.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(renglones) == 1 and json.loads(renglones[0])["productos"]["abi_c13"] == "sincronizado"
    # Al revés no: un ciclo sin satélite no pisa uno completo.
    assert h.guardar(sin, tmp_path).startswith("ya guardado")


def test_dias_cerrados_se_comprimen_y_se_siguen_leyendo(tmp_path):
    h.guardar(_estado([T]), tmp_path)                                   # 19/11
    h.guardar(_estado([T], frame="2026-11-22T10:00:00Z"), tmp_path)     # 22/11: el 19 ya cerró
    assert not (tmp_path / "2026-11-19.jsonl").exists()
    assert (tmp_path / "2026-11-19.jsonl.gz").exists()
    assert (tmp_path / "2026-11-22.jsonl").exists()                     # el día en curso sigue abierto
    casos = h.leer(datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc),
                   datetime(2026, 11, 22, 23, 0, tzinfo=timezone.utc), tmp_path)
    assert [c["frame_radar_utc"] for c in casos] == [FRAME, "2026-11-22T10:00:00Z"]
    # Lo lee también la verificación del puntaje.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import verificar_puntaje as vp
    assert vp.leer_jsonl(tmp_path / "2026-11-19.jsonl")[0]["frame_radar_utc"] == FRAME
