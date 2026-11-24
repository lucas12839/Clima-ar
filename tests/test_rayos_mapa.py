"""Rayos de la última hora para el mapa de la app (vista Meteorólogos)."""

import json
from datetime import datetime, timedelta, timezone

from goes19 import state

AHORA = datetime(2026, 11, 24, 7, 0, tzinfo=timezone.utc)
T = int(AHORA.timestamp())


def test_guarda_sin_repetidos_y_descarta_lo_viejo(tmp_path):
    destino = tmp_path / "rayos_mapa.json"
    state.guardar_rayos_mapa(
        [(-38.1, -62.2, T - 100), (-38.1, -62.2, T - 100), (-37.0, -62.0, T - 90 * 60)],
        AHORA,
        destino,
    )
    datos = json.loads(destino.read_text(encoding="utf-8"))
    assert datos["rayos"] == [[-38.1, -62.2, T - 100]]


def test_suma_lo_nuevo_a_lo_guardado_en_orden(tmp_path):
    destino = tmp_path / "rayos_mapa.json"
    state.guardar_rayos_mapa([(-38.1, -62.2, T - 100)], AHORA, destino)
    state.guardar_rayos_mapa([(-38.2, -62.3, T + 300)], AHORA + timedelta(minutes=5), destino)
    rayos = json.loads(destino.read_text(encoding="utf-8"))["rayos"]
    assert [r[2] for r in rayos] == [T - 100, T + 300]


def test_sin_rayos_igual_limpia_lo_viejo(tmp_path):
    destino = tmp_path / "rayos_mapa.json"
    state.guardar_rayos_mapa([(-38.1, -62.2, T - 100)], AHORA, destino)
    state.guardar_rayos_mapa([], AHORA + timedelta(hours=2), destino)
    assert json.loads(destino.read_text(encoding="utf-8"))["rayos"] == []
