#!/usr/bin/env python3
"""Runtime V1 RMA10: causal live features and independently calibrated outputs."""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rma10_features import (
    BAHIA_LAT,
    BAHIA_LON,
    RADAR_LAT,
    RADAR_LON,
    RADAR_RADIUS_KM,
    V1_FEATURES,
    bearing_deg,
    build_v1_feature_dict,
    distance_km,
    parse_utc,
    pixel_to_latlon,
)

ROOT = REPO_ROOT
MODEL_PATH = ROOT / "data/models/rma10_model.joblib"
STATE_PATH = ROOT / "data/radar/seguimiento/estado.json"
RESULTADO_PATH = ROOT / "data/radar/resultado_procesamiento.json"
OUTPUT_PATH = ROOT / "data/radar/seguimiento/prediccion_ml_rma10.json"
RADAR_TIMESTAMP_PATH = ROOT / "data/radar/radar_timestamp.txt"

FEATURES = list(V1_FEATURES)
HORIZONS = (5, 10, 20, 30)
TARGETS = ("strengthened", "weakened", "toward_bahia")
MODEL_VERSION = "rma10-v1-causal-2"
CALIBRATION_EPS = 1e-6
CALIBRATION_MIN_SAMPLES = 100


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


def get_probability(model: Any, row: pd.DataFrame) -> float:
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    if 1 not in classes:
        return 0.0
    return float(probabilities[classes.index(1)])


def logit(probability: float) -> float:
    p = min(1.0 - CALIBRATION_EPS, max(CALIBRATION_EPS, float(probability)))
    return math.log(p / (1.0 - p))


def sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def apply_probability_calibration(probability: float, calibration: dict[str, Any]) -> float:
    if not isinstance(calibration, dict):
        raise ValueError("Calibración inválida")
    a = number(calibration.get("a"), float("nan"))
    b = number(calibration.get("b"), float("nan"))
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Coeficientes de calibración inválidos")
    return sigmoid(a * logit(probability) + b)


def get_calibrated_probability(probability: float, calibration: dict[str, Any] | None):
    if not isinstance(calibration, dict) or calibration.get("enabled") is not True:
        return None
    return apply_probability_calibration(probability, calibration)


def read_radar_timestamp() -> str:
    if not RADAR_TIMESTAMP_PATH.exists():
        raise SystemExit("No existe radar_timestamp.txt")
    value = RADAR_TIMESTAMP_PATH.read_text(encoding="utf-8").strip()
    if parse_utc(value) is None:
        raise SystemExit(f"Timestamp RMA10 inválido: {value!r}")
    return value


def validate_bundle(bundle: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(bundle, dict):
        raise RuntimeError("Bundle del modelo V1 inválido")
    if bundle.get("model_version") != MODEL_VERSION:
        raise RuntimeError(
            f"Modelo V1 incompatible: {bundle.get('model_version')!r}; "
            f"se requiere {MODEL_VERSION!r}. Reentrenar antes de ejecutar."
        )
    if bundle.get("features") != FEATURES:
        raise RuntimeError("Features del modelo V1 no coinciden con el runtime")
    if tuple(bundle.get("horizons_min", ())) != HORIZONS:
        raise RuntimeError("Horizontes del modelo V1 incompatibles")
    if tuple(bundle.get("targets", ())) != TARGETS:
        raise RuntimeError("Targets del modelo V1 incompatibles")
    models = bundle.get("models")
    expected = {f"{h}m_{target}" for h in HORIZONS for target in TARGETS}
    if not isinstance(models, dict) or set(models) != expected:
        raise RuntimeError("El bundle V1 no contiene exactamente los 12 modelos")
    calibration = bundle.get("calibration")
    if not isinstance(calibration, dict) or set(calibration) != expected:
        raise RuntimeError("El bundle V1 no contiene la calibración de los 12 modelos")
    return models, calibration


def current_track_observation(track: dict[str, Any], radar_timestamp: str):
    history = track.get("historial", [])
    if not isinstance(history, list) or not history:
        return None, []
    valid_history = [item for item in history if isinstance(item, dict)]
    if not valid_history:
        return None, []
    valid_history.sort(key=lambda item: parse_utc(item.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc))
    latest = valid_history[-1]
    latest_ts = parse_utc(latest.get("timestamp"))
    radar_dt = parse_utc(radar_timestamp)
    if latest_ts is None or radar_dt is None or latest_ts != radar_dt:
        return None, valid_history
    current = dict(track)
    current.update(latest)
    current["timestamp"] = radar_timestamp
    return current, valid_history


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"No existe el modelo: {MODEL_PATH}")
    state = load_json(STATE_PATH)
    resultado = load_json(RESULTADO_PATH)
    bundle = joblib.load(MODEL_PATH)
    models, calibration = validate_bundle(bundle)

    radar_timestamp = read_radar_timestamp()
    radar_dt = parse_utc(radar_timestamp)
    width, height = get_dimensions(resultado)

    tracks = state.get("tracks", [])
    if not isinstance(tracks, list):
        tracks = []

    predictions = []
    calibrated_values = 0
    for track in tracks:
        if not isinstance(track, dict):
            continue
        if track.get("activo") is False or track.get("presente_en_ultimo_frame") is False:
            continue
        current, history = current_track_observation(track, radar_timestamp)
        if current is None:
            continue
        features = build_v1_feature_dict(current, history, width, height)
        row = pd.DataFrame([[features[name] for name in FEATURES]], columns=FEATURES)

        horizons = {}
        calibration_status = {}
        for horizon in HORIZONS:
            targets = {}
            calibration_status[str(horizon)] = {}
            for target in TARGETS:
                key = f"{horizon}m_{target}"
                raw_score = get_probability(models[key], row)
                calibrated = get_calibrated_probability(raw_score, calibration[key])
                if calibrated is None:
                    targets[target] = None
                    calibration_status[str(horizon)][target] = False
                else:
                    calibrated_values += 1
                    targets[target] = round(calibrated * 100.0, 1)
                    calibration_status[str(horizon)][target] = True
            horizons[str(horizon)] = targets

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
            "horizontes": horizons,
            "calibracion_por_horizonte": calibration_status,
        })

    expected_keys = [f"{h}m_{target}" for h in HORIZONS for target in TARGETS]
    calibrated_count = sum(calibration[key].get("enabled") is True for key in expected_keys)
    all_calibrated = calibrated_count == len(expected_keys)

    output = {
        "ok": True,
        "modelo": bundle["model_version"],
        "estado_modelo": "CALIBRADO" if all_calibrated else "CALIBRACION_PARCIAL_O_PENDIENTE",
        "generado_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "frame_radar": radar_timestamp,
        "tracks_analizados": len(predictions),
        "horizontes_min": list(HORIZONS),
        "features_version": bundle.get("features_version"),
        "dataset_sha256": bundle.get("dataset_sha256"),
        "calibracion_probabilistica": {
            "aplicada": all_calibrated,
            "parcialmente_aplicada": calibrated_count > 0 and not all_calibrated,
            "por_modelo": {
                key: {
                    "aplicada": calibration[key].get("enabled") is True,
                    "motivo": calibration[key].get("reason"),
                    "muestras_calibracion": calibration[key].get("calibration_samples"),
                    "muestras_test": calibration[key].get("test_samples"),
                    "brier_test_sin_calibrar": calibration[key].get("brier_raw_test"),
                    "brier_test_calibrado": calibration[key].get("brier_calibrated_test"),
                }
                for key in expected_keys
            },
            "metricas": {
                key: {
                    **calibration[key],
                    "selection": "CALIBRATED" if calibration[key].get("enabled") is True else "UNCALIBRATED",
                }
                for key in expected_keys
            },
            "eventos": bundle.get("event_split"),
            "minimo_muestras": CALIBRATION_MIN_SAMPLES,
            "valores_calibrados": calibrated_values,
            "metricas_independientes": bundle.get("validation_metrics"),
        },
        "predicciones": predictions,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "ok": True,
        "tracks_analizados": len(predictions),
        "calibracion_probabilistica": all_calibrated,
        "modelos_calibrados": calibrated_count,
        "output": str(OUTPUT_PATH),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
