import json
from datetime import datetime, timedelta, timezone

import meteorologia_fusion as mf


def preparar(tmp_path, monkeypatch, edad_min):
    amb = tmp_path / "ambiente_futuro.json"
    ahora = datetime.now(timezone.utc)
    amb.write_text(json.dumps({
        "actualizado_utc": (ahora - timedelta(minutes=edad_min)).isoformat(),
        "puntos": [{"punto": "BB_CENTRO", "datos": [{"tiempo_utc": ahora.isoformat(), "temperatura_c": 20}]}],
    }), encoding="utf-8")
    monkeypatch.setattr(mf, "AMBIENTE_FILE", amb)
    monkeypatch.setattr(mf, "FUSION_FILE", tmp_path / "fusion.json")
    monkeypatch.setattr(mf, "cargar_radar", lambda: {"disponible": True, "actividad": False})
    monkeypatch.setattr(mf, "cargar_sazb", lambda: {"disponible": False})
    monkeypatch.setattr(mf, "append_historico", lambda *a: None)
    monkeypatch.setattr(mf, "calcular_calidad", lambda *a, **k: {"score": 50, "nivel": "x", "puntos_ecmwf_validos": 1})
    monkeypatch.setattr(mf, "fusionar_actual_ecmwf_sazb", lambda a, s, now=None: {})
    llamadas = []
    monkeypatch.setattr(mf, "consultar_ecmwf", lambda: llamadas.append(1) or (_ for _ in ()).throw(TimeoutError("caído")))
    return amb, llamadas


def test_pronostico_reciente_no_consulta(tmp_path, monkeypatch):
    amb, llamadas = preparar(tmp_path, monkeypatch, 10)
    antes = amb.read_text()
    assert mf.main() == 0
    f = json.loads((tmp_path / "fusion.json").read_text())
    assert llamadas == [] and f["estado"] == "operativo" and f["ecmwf_cache"]["usado"]
    assert amb.read_text() == antes  # conserva su hora real


def test_pronostico_viejo_consulta_y_usa_respaldo(tmp_path, monkeypatch):
    _, llamadas = preparar(tmp_path, monkeypatch, 90)
    assert mf.main() == 0
    f = json.loads((tmp_path / "fusion.json").read_text())
    assert llamadas == [1] and f["estado"] == "operativo_con_ecmwf_anterior"
    assert f["ecmwf_cache"] == {"usado": False}
