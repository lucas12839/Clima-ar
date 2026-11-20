"""Paso 1 (20/11/2026): el 5b pasa a lo que se ve (llegadas por zona y trayectorias)."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import motor_decision_climaar as md  # noqa: E402
import trayectorias_capas as tc  # noqa: E402

FRAME = datetime(2026, 11, 20, 21, 0, tzinfo=timezone.utc)


def capas(eta_zona=10, eta_loc=20, ts="2026-11-20T21:00:00+00:00", usado=True):
    return {
        "disponible": True, "radar_timestamp": ts,
        "llegada_proxima_hora": {"usado": usado},
        "localidades": {"bahia_blanca": {
            "nombre": "Bahía Blanca",
            "llegada_proxima_hora": {"tormenta": {"probable": {"localidad": eta_loc, "hasta_10km": eta_zona},
                                                  "posible": {"localidad": 10, "hasta_10km": 0}}},
            "pronostico_desde_ahora": {"10min": {"dbz": 30.0}, "20min": {"dbz": 51.7},
                                       "30min": {"dbz": 44.0}, "60min": {"dbz": 20.0}},
        }},
    }


FLUJO = {"radar_timestamp": "2026-11-20T21:00:00+00:00",
         "bahia_blanca": {"eta_tormenta_desde_ahora_min": 30.8, "dbz_al_llegar": 35.3},
         "localidades": {"bahia_blanca": {"llegada": {"eta_desde_ahora_min": 30.8, "dbz_al_llegar": 35.3,
                                                       "velocidad_local_kmh": 40.0}}}}


def test_llegada_que_se_ve_sale_del_5b_zona_10km_probable():
    o = md.llegadas_5b(FLUJO, capas())
    ll = o["localidades"]["bahia_blanca"]["llegada"]
    assert o["fuente_llegadas"] == "nowcast_capas"
    assert ll["eta_desde_ahora_min"] == 10          # zona (<= 10 km), no "justo encima" (20)
    assert ll["dbz_al_llegar"] == 51.7               # lo más fuerte desde que llega
    assert ll["eta_desde_ahora_min_flujo_optico"] == 30.8   # se conserva lo anterior
    assert ll["velocidad_local_kmh"] == 40.0
    assert o["bahia_blanca"]["eta_tormenta_desde_ahora_min"] == 10
    assert FLUJO["bahia_blanca"]["eta_tormenta_desde_ahora_min"] == 30.8   # no modifica la entrada


def test_sin_llegada_del_5b_no_hay_llegada():
    o = md.llegadas_5b(FLUJO, capas(eta_zona=None))
    ll = o["localidades"]["bahia_blanca"]["llegada"]
    assert ll["eta_desde_ahora_min"] is None and ll["dbz_al_llegar"] is None


@pytest.mark.parametrize("cambio", [{"ts": "2026-11-20T20:50:00+00:00"}, {"usado": False}])
def test_5b_desincronizado_o_sin_llegadas_sigue_el_flujo(cambio):
    o = md.llegadas_5b(FLUJO, capas(**cambio))
    assert o["fuente_llegadas"] == "flujo_optico"
    assert o["localidades"]["bahia_blanca"]["llegada"]["eta_desde_ahora_min"] == 30.8


def test_5b_ausente_sigue_el_flujo():
    assert md.llegadas_5b(FLUJO, {})["fuente_llegadas"] == "flujo_optico"


def test_confiabilidad_del_5b_cuenta_como_probable():
    import indices_fenomenos as fen
    d = fen.descripcion_llegada(md.CONFIABILIDAD_5B)
    assert d["probable"] is True and d["de_cada_10"] == 8


# Trayectorias ---------------------------------------------------------------

def test_trayectoria_sigue_el_campo_de_movimiento():
    u = np.full((100, 100), 0.5, dtype=np.float32)    # 0,5 px/min hacia el este
    v = np.full((100, 100), -0.25, dtype=np.float32)  # hacia el norte (y crece al sur)
    t = tc.trayectoria(u, v, 0.94, {"track_id": 7, "x": 20.0, "y": 60.0, "pixeles": 50}, FRAME)
    h30 = next(h for h in t["horizontes"] if h["minutos"] == 30)
    assert h30["x"] == pytest.approx(35.0, abs=0.01) and h30["y"] == pytest.approx(52.5, abs=0.01)
    assert t["rumbo_hacia_grados"] == pytest.approx(63.4, abs=0.2)   # ENE
    assert [h["minutos"] for h in t["horizontes"]] == [5, 10, 20, 30, 60]
    assert h30["timestamp_utc"] == "2026-11-20T21:30:00Z"


def test_cada_tormenta_con_su_propio_movimiento():
    u = np.zeros((100, 100), dtype=np.float32)
    u[:, 50:] = 1.0                                     # solo la mitad este se mueve
    v = np.zeros((100, 100), dtype=np.float32)
    quieta = tc.trayectoria(u, v, 1.0, {"track_id": 1, "x": 20.0, "y": 50.0, "pixeles": 10}, FRAME)
    rapida = tc.trayectoria(u, v, 1.0, {"track_id": 2, "x": 70.0, "y": 50.0, "pixeles": 10}, FRAME)
    assert quieta["velocidad_kmh"] == 0.0 and rapida["velocidad_kmh"] == pytest.approx(60.0)


def test_tracks_actuales_exige_la_imagen_del_ciclo():
    estado = {"ultimo_radar_timestamp": "2026-11-20T21:00:00Z", "tracks": [
        {"track_id": 1, "activo": True, "presente_en_ultimo_frame": True, "x": 1, "y": 1},
        {"track_id": 2, "activo": True, "presente_en_ultimo_frame": False, "x": 1, "y": 1}]}
    assert [t["track_id"] for t in tc.tracks_actuales(estado, "2026-11-20T21:00:00+00:00")] == [1]
    with pytest.raises(RuntimeError):
        tc.tracks_actuales(estado, "2026-11-20T21:10:00+00:00")


def test_conectar_usa_el_5b_y_si_falta_el_viejo(tmp_path, monkeypatch):
    import conectar_evolucion_trayectoria as ce
    t = {"disponible": True, "radar_timestamp": "2026-11-20T21:00:00Z", "tracks": {"7": {
        "track_id": 7, "rumbo_hacia_grados": 63.4, "velocidad_pixeles_min": 0.56,
        "horizontes": [{"minutos": m, "x": 100 + m, "y": 200.0, "incertidumbre_pixeles": 5} for m in (5, 10, 20, 30, 60)]}}}
    p = tmp_path / "t.json"
    p.write_text(json.dumps(t))
    monkeypatch.setattr(ce, "TRAYECTORIAS_FILE", p)
    r = ce.trayectoria_capas(7, FRAME)
    assert r["pronostico_disponible"] and ce.get_horizon_prediction(r, 30)["x_estimado"] == 130
    assert ce.get_horizon_prediction(r, 30)["direccion_estimada"] == "ENE"
    assert ce.trayectoria_capas(8, FRAME) is None                       # tormenta que no está
    assert ce.trayectoria_capas(7, datetime(2026, 11, 20, 21, 10, tzinfo=timezone.utc)) is None  # otra imagen
    p.unlink()
    assert ce.trayectoria_capas(7, FRAME) is None
