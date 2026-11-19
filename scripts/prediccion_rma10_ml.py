#!/usr/bin/env python3
"""Predicción RMA10 por track SIN modelos entrenados.

ClimaAR no usa modelos entrenados en producción. Este script conserva el
contrato de ``prediccion_ml_rma10.json`` (lo leen Intelligence, la API,
la validación de integridad y los workflows), pero ya no carga ningún
``.joblib``: publica la geometría causal de cada track activo y deja todas
las probabilidades en ``None`` con la calibración marcada como no aplicada.

La predicción de evolución sale de las reglas físicas
(``prediccion_evolucion.py`` y ``conectar_evolucion_trayectoria.py``).
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rma10_features import build_v1_feature_dict, parse_utc  # noqa: E402

ROOT = REPO_ROOT
STATE_PATH = ROOT / "data/radar/seguimiento/estado.json"
RESULTADO_PATH = ROOT / "data/radar/resultado_procesamiento.json"
OUTPUT_PATH = ROOT / "data/radar/seguimiento/prediccion_ml_rma10.json"
RADAR_TIMESTAMP_PATH = ROOT / "data/radar/radar_timestamp.txt"

HORIZONS = (5, 10, 20, 30)
TARGETS = ("strengthened", "weakened", "toward_bahia")
MODEL_VERSION = "rma10-reglas-sin-entrenamiento-v1"
ESTADO_MODELO = "DESACTIVADO_SIN_ENTRENAMIENTO"
MOTIVO = "ClimaAR no usa modelos entrenados; la evolución sale de reglas físicas."


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"No existe: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"JSON inválido: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"JSON no es objeto: {path}")
    return data


def number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def get_dimensions(resultado: dict[str, Any]) -> tuple[int, int]:
    width = int(number(resultado.get("ancho_imagen", resultado.get("width", 512)), 512))
    height = int(number(resultado.get("alto_imagen", resultado.get("height", 512)), 512))
    if not (2 <= width <= 4096 and 2 <= height <= 4096):
        raise RuntimeError(f"Dimensiones RMA10 inválidas: {width}x{height}")
    return width, height


def read_radar_timestamp() -> str:
    if not RADAR_TIMESTAMP_PATH.exists():
        raise SystemExit("No existe radar_timestamp.txt")
    value = RADAR_TIMESTAMP_PATH.read_text(encoding="utf-8").strip()
    if parse_utc(value) is None:
        raise SystemExit(f"Timestamp RMA10 inválido: {value!r}")
    return value


def current_track_observation(track: dict[str, Any], radar_timestamp: str):
    history = track.get("historial", [])
    if not isinstance(history, list) or not history:
        return None, []
    valid_history = [item for item in history if isinstance(item, dict)]
    if not valid_history:
        return None, []
    valid_history.sort(
        key=lambda item: parse_utc(item.get("timestamp"))
        or datetime.min.replace(tzinfo=timezone.utc)
    )
    latest = valid_history[-1]
    latest_ts = parse_utc(latest.get("timestamp"))
    radar_dt = parse_utc(radar_timestamp)
    if latest_ts is None or radar_dt is None or latest_ts != radar_dt:
        return None, valid_history
    current = dict(track)
    current.update(latest)
    current["timestamp"] = radar_timestamp
    return current, valid_history


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def build_output(state: dict[str, Any], resultado: dict[str, Any], radar_timestamp: str) -> dict[str, Any]:
    width, height = get_dimensions(resultado)
    tracks = state.get("tracks", [])
    if not isinstance(tracks, list):
        tracks = []

    empty_horizons = {str(h): {t: None for t in TARGETS} for h in HORIZONS}
    empty_flags = {str(h): {t: False for t in TARGETS} for h in HORIZONS}

    predictions = []
    for track in tracks:
        if not isinstance(track, dict):
            continue
        if track.get("activo") is False or track.get("presente_en_ultimo_frame") is False:
            continue
        current, history = current_track_observation(track, radar_timestamp)
        if current is None:
            continue
        features = build_v1_feature_dict(current, history, width, height)
        predictions.append({
            "track_id": track.get("track_id", track.get("id")),
            "lat": round(features["lat"], 6),
            "lon": round(features["lon"], 6),
            "distance_km": round(features["distance_km"], 2),
            "bearing_to_bahia_deg": round(features["bearing_to_bahia_deg"], 1),
            "displacement_px": round(features["displacement_px"], 2),
            "speed_px_min": round(features["speed_px_min"], 4),
            "motion_bearing_deg": round(features["motion_bearing_deg"], 1),
            "motion_valid": bool(features["motion_valid"]),
            "horizontes": json.loads(json.dumps(empty_horizons)),
            "calibracion_por_horizonte": json.loads(json.dumps(empty_flags)),
        })

    keys = [f"{h}m_{t}" for h in HORIZONS for t in TARGETS]
    return {
        "ok": True,
        "modelo": MODEL_VERSION,
        "estado_modelo": ESTADO_MODELO,
        "motivo": MOTIVO,
        "generado_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "frame_radar": radar_timestamp,
        "tracks_analizados": len(predictions),
        "horizontes_min": list(HORIZONS),
        "calibracion_probabilistica": {
            "aplicada": False,
            "parcialmente_aplicada": False,
            "por_modelo": {key: {"aplicada": False, "motivo": MOTIVO} for key in keys},
            "metricas": {key: {"enabled": False, "selection": "DESACTIVADO"} for key in keys},
            "valores_calibrados": 0,
        },
        "predicciones": predictions,
    }


def main() -> None:
    state = load_json(STATE_PATH)
    resultado = load_json(RESULTADO_PATH)
    radar_timestamp = read_radar_timestamp()
    output = build_output(state, resultado, radar_timestamp)
    atomic_write_json(OUTPUT_PATH, output)
    print(json.dumps({
        "ok": True,
        "modelo": MODEL_VERSION,
        "tracks_analizados": output["tracks_analizados"],
        "output": str(OUTPUT_PATH),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
