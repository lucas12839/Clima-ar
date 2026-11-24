"""Portada 24/11/2026: rayos recientes en /resumen y ruta del archivo de rayos."""
import json
import os
from datetime import datetime, timezone

os.environ.setdefault("CLIMAAR_REMOTE_DATA_ENABLED", "0")
import app  # noqa: E402


def test_ruta_rayos_dentro_de_data():
    # Antes faltaba "data/": en el VPS el mapa del satélite quedaba sin rayos.
    assert app.RAYOS_MAPA_FILE.relative_to(app.DATA_BASE_DIR).as_posix() == "data/goes19/rayos_mapa.json"


def test_rayos_recientes(tmp_path, monkeypatch):
    ahora = datetime.now(timezone.utc)
    t = ahora.timestamp()
    p = tmp_path / "rayos.json"
    p.write_text(json.dumps({"actualizado_utc": ahora.isoformat(), "rayos": [
        [-38.7, -62.2, int(t - 60)], [-38.7, -62.2, int(t - 1200)], [-38.7, -62.2, int(t - 5000)], ["x"]]}))
    monkeypatch.setattr(app, "RAYOS_MAPA_FILE", p)
    r = app._rayos_recientes()
    assert r["ultimos_10_min"] == 1 and r["ultima_hora"] == 2 and r["disponible"] is True


def test_rayos_viejos_no_disponibles(tmp_path, monkeypatch):
    p = tmp_path / "rayos.json"
    p.write_text(json.dumps({"actualizado_utc": "2026-01-01T00:00:00Z", "rayos": []}))
    monkeypatch.setattr(app, "RAYOS_MAPA_FILE", p)
    assert app._rayos_recientes()["disponible"] is False
    monkeypatch.setattr(app, "RAYOS_MAPA_FILE", tmp_path / "no.json")
    assert app._rayos_recientes()["disponible"] is False
