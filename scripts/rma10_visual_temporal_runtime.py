#!/usr/bin/env python3
"""Live runtime for the causal RMA10 visual-temporal model."""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

BASE = Path(__file__).resolve().parents[1]
SCRIPTS = BASE / "scripts"
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts.rma10_features import build_v1_feature_dict, parse_utc
import entrenar_modelo_visual_temporal_rma10 as trainer  # noqa: E402

MODEL_PATH = BASE / "data/models/rma10_visual_temporal_model.joblib"
INTEL_PATH = BASE / "data/ia/inteligencia_tormenta.json"
ESTADO_PATH = BASE / "data/radar/seguimiento/estado.json"
SEGUIMIENTO_PATH = BASE / "data/radar/seguimiento/seguimiento_temporal.json"
TIMESTAMP_PATH = BASE / "data/radar/radar_timestamp.txt"
RESULTADO_PATH = BASE / "data/radar/resultado_procesamiento.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"No existe archivo requerido: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"JSON inválido {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON no es un objeto: {path}")
    return data


def parse_ts(value):
    return parse_utc(value)


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def get_dimensions(resultado):
    width = int(resultado.get("ancho_imagen", resultado.get("width", 512)))
    height = int(resultado.get("alto_imagen", resultado.get("height", 512)))
    if not (2 <= width <= 4096 and 2 <= height <= 4096):
        raise RuntimeError(f"Dimensiones RMA10 inválidas: {width}x{height}")
    return width, height


def track_history_for_trainer(track):
    history = []
    for item in track.get("historial", []):
        if not isinstance(item, dict):
            continue
        timestamp = parse_ts(item.get("timestamp", item.get("frame_utc")))
        if timestamp is None:
            continue
        try:
            x = float(item["x"])
            y = float(item["y"])
        except (KeyError, TypeError, ValueError):
            continue
        history.append({
            "timestamp": timestamp,
            "x": x,
            "y": y,
            "intensidad": float(item.get("intensidad", item.get("intensity", track.get("intensidad", 0))) or 0),
            "pixeles": float(item.get("pixeles", item.get("area_px", track.get("pixeles", 0))) or 0),
            "ancho": float(item.get("ancho", item.get("width_px", track.get("ancho", 0))) or 0),
            "alto": float(item.get("alto", item.get("height_px", track.get("alto", 0))) or 0),
        })
    history.sort(key=lambda item: item["timestamp"])
    if not history:
        raise RuntimeError("El track principal no tiene historial utilizable")
    return history


def principal_track(estado, seguimiento, radar_dt):
    principal_id = seguimiento.get("track_principal_candidato_id")
    if principal_id is None:
        raise RuntimeError("Tracking no definió track_principal_candidato_id")
    principal_text = trainer.canonical_track_id(principal_id)
    for track in estado.get("tracks", []):
        if not isinstance(track, dict):
            continue
        track_id = trainer.canonical_track_id(track.get("track_id", track.get("id")))
        if track_id != principal_text:
            continue
        if track.get("activo") is False or track.get("presente_en_ultimo_frame") is False:
            raise RuntimeError("El track principal no está activo/presente en el último frame")
        history = track_history_for_trainer(track)
        if history[-1]["timestamp"] != radar_dt:
            raise RuntimeError("El historial del track principal no coincide con el timestamp RMA10")
        return track, history
    raise RuntimeError(f"El track principal {principal_id} no existe entre los tracks activos")


def build_runtime_visual_feature(radar_dt, history, image_index):
    image_times = [item[0] for item in image_index]
    sequence = trainer.find_visual_sequence(
        radar_dt,
        history,
        image_index,
        image_times,
    )
    if sequence is None:
        raise RuntimeError("No hay una secuencia visual causal y reciente de al menos 3 frames")
    frames, valid_mask, elapsed, gaps, selected = sequence
    visual = trainer.feature_vector(frames, valid_mask, elapsed, gaps)
    visual = np.asarray(visual, dtype=np.float32).reshape(-1)
    anchor = trainer.causal_position(history, radar_dt)
    if anchor is None:
        raise RuntimeError("No se pudo obtener la posición causal actual del track")
    return visual, selected, anchor


def build_v1_base(history, width, height):
    current = dict(history[-1])
    features = build_v1_feature_dict(current, history, width, height)
    values = np.asarray([features[name] for name in trainer.V1_FEATURES], dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("Features V1 live no finitas")
    return values


def build_final_vector(track, history, radar_dt, image_index, payload, width, height):
    visual_raw, selected, anchor = build_runtime_visual_feature(radar_dt, history, image_index)
    expected_visual = int(payload["visual_raw_features"])
    if visual_raw.size != expected_visual:
        raise RuntimeError(f"Features visuales incompatibles: runtime={visual_raw.size} modelo={expected_visual}")
    pca = payload["pca"]
    visual_pca = pca.transform(visual_raw.reshape(1, -1))[0].astype(np.float32)
    components = int(payload["pca_components"])
    if visual_pca.size != components:
        raise RuntimeError(f"PCA incompatible: runtime={visual_pca.size} modelo={components}")
    base = build_v1_base(history, width, height)
    vector = np.concatenate([base, visual_pca, np.asarray([1.0], dtype=np.float32)]).astype(np.float32)
    expected_final = len(trainer.V1_FEATURES) + components + 1
    if vector.size != expected_final or not np.isfinite(vector).all():
        raise RuntimeError(f"Vector final inválido: runtime={vector.size} esperado={expected_final}")
    return vector, selected, anchor


def validate_bundle(payload):
    if not isinstance(payload, dict):
        raise RuntimeError("Payload del modelo visual inválido")

    # The deployed production bundle may predate the current trainer version.
    # Accept only the two explicitly known versions, then enforce the complete
    # structural contract below. Unknown versions remain rejected.
    supported_versions = {
        trainer.MODEL_VERSION,
        "rma10-visual-temporal-final-ensemble-1",
    }
    model_version = payload.get("model_version")
    if model_version not in supported_versions:
        raise RuntimeError(
            f"Versión de modelo incompatible: {model_version!r}; "
            f"versiones admitidas: {sorted(supported_versions)!r}"
        )
    if payload.get("features_v1") != list(trainer.V1_FEATURES):
        raise RuntimeError("Features V1 del modelo visual no coinciden con el runtime")
    if payload.get("features_version") != "rma10-causal-features-v2":
        raise RuntimeError("Versión de features causales incompatible")
    if int(payload.get("sequence_len", 0)) != trainer.SEQUENCE_LEN:
        raise RuntimeError("Longitud de secuencia incompatible")
    if int(payload.get("min_sequence_len", 0)) != trainer.MIN_SEQUENCE_LEN:
        raise RuntimeError("Mínimo de secuencia incompatible")
    if payload.get("horizons_min") != list(trainer.HORIZONS) or payload.get("targets") != list(trainer.TARGETS):
        raise RuntimeError("Horizontes/targets incompatibles")
    pca = payload.get("pca")
    if not hasattr(pca, "transform"):
        raise RuntimeError("PCA ausente o inválido")

    try:
        raw_features = int(payload["visual_raw_features"])
        pca_components = int(payload["pca_components"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Dimensiones visuales ausentes o inválidas en el bundle") from exc

    if raw_features <= 0 or pca_components <= 0:
        raise RuntimeError("Dimensiones visuales deben ser positivas")
    if getattr(pca, "n_features_in_", raw_features) != raw_features:
        raise RuntimeError("PCA entrenado con una cantidad de features visuales distinta")
    if getattr(pca, "n_components_", pca_components) != pca_components:
        raise RuntimeError("Cantidad de componentes PCA no coincide con el bundle")
    if getattr(pca, "components_", np.empty((pca_components,))).shape[0] != pca_components:
        raise RuntimeError("Estructura interna PCA incompatible")

    models = payload.get("models")
    calibration = payload.get("calibration")
    expected = {f"{h}m_{target}" for h in trainer.HORIZONS for target in trainer.TARGETS}
    if not isinstance(models, dict) or set(models) != expected:
        raise RuntimeError("El modelo visual no contiene exactamente 12 ensembles")
    if not isinstance(calibration, dict) or set(calibration) != expected:
        raise RuntimeError("El modelo visual no contiene exactamente 12 calibraciones")

    expected_model_features = len(trainer.V1_FEATURES) + pca_components + 1
    for key in sorted(expected):
        ensemble = models[key]
        if not isinstance(ensemble, (list, tuple)) or not ensemble:
            raise RuntimeError(f"Ensemble visual inválido: {key}")
        for item in ensemble:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise RuntimeError(f"Componente de ensemble inválido: {key}")
            _, estimator, weight = item
            try:
                numeric_weight = float(weight)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Peso inválido en ensemble: {key}") from exc
            if not math.isfinite(numeric_weight) or numeric_weight <= 0:
                raise RuntimeError(f"Peso no positivo o no finito en ensemble: {key}")
            fitted_features = getattr(estimator, "n_features_in_", None)
            if fitted_features is not None and int(fitted_features) != expected_model_features:
                raise RuntimeError(
                    f"Dimensión incompatible en {key}: modelo={fitted_features}, "
                    f"runtime={expected_model_features}"
                )

    if not payload.get("config", {}).get("motion_compensated"):
        raise RuntimeError("El modelo no declara secuencia visual compensada por movimiento")
    return models, calibration


def positive_probability(ensemble, vector):
    if not isinstance(ensemble, (list, tuple)) or not ensemble:
        raise RuntimeError("Ensemble visual inválido")
    weighted = 0.0
    total_weight = 0.0
    for item in ensemble:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise RuntimeError("Elemento de ensemble visual inválido")
        _, model, weight = item
        weight = float(weight)
        if weight <= 0:
            continue
        probabilities = model.predict_proba(vector.reshape(1, -1))[0]
        classes = list(model.classes_)
        probability = float(probabilities[classes.index(1)]) if 1 in classes else 0.0
        weighted += probability * weight
        total_weight += weight
    if total_weight <= 0:
        raise RuntimeError("Ensemble visual sin pesos válidos")
    return float(weighted / total_weight)


def apply_calibration(raw_probability, calibration):
    if not isinstance(calibration, dict) or calibration.get("enabled") is not True or calibration.get("approved") is not True:
        return None
    a = float(calibration.get("a"))
    b = float(calibration.get("b"))
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    eps = 1e-6
    p = min(1.0 - eps, max(eps, float(raw_probability)))
    logit = math.log(p / (1.0 - p))
    z = max(-60.0, min(60.0, a * logit + b))
    return 1.0 / (1.0 + math.exp(-z))


def main():
    intel = load_json(INTEL_PATH)
    estado = load_json(ESTADO_PATH)
    seguimiento = load_json(SEGUIMIENTO_PATH)
    resultado = load_json(RESULTADO_PATH)
    if not MODEL_PATH.exists():
        raise RuntimeError(f"No existe modelo visual: {MODEL_PATH}")
    if not TIMESTAMP_PATH.exists():
        raise RuntimeError("No existe radar_timestamp.txt")

    radar_timestamp = intel.get("radar_timestamp") or TIMESTAMP_PATH.read_text(encoding="utf-8").strip()
    radar_dt = parse_ts(radar_timestamp)
    if radar_dt is None:
        raise RuntimeError("Timestamp del radar inválido")
    file_dt = parse_ts(TIMESTAMP_PATH.read_text(encoding="utf-8").strip())
    if file_dt != radar_dt:
        raise RuntimeError("Timestamp de inteligencia y radar_timestamp.txt desincronizados")
    tracking_dt = parse_ts(seguimiento.get("radar_timestamp"))
    if tracking_dt is not None and tracking_dt != radar_dt:
        raise RuntimeError("Tracking y radar están desincronizados")

    payload = joblib.load(MODEL_PATH)
    models, calibrations = validate_bundle(payload)
    width, height = get_dimensions(resultado)
    track, history = principal_track(estado, seguimiento, radar_dt)
    image_index = trainer.build_image_index()
    vector, selected, anchor = build_final_vector(
        track, history, radar_dt, image_index, payload, width, height
    )

    predictions = {}
    calibration_report = {}
    calibrated_count = 0
    for horizon in trainer.HORIZONS:
        predictions[str(horizon)] = {}
        for target in trainer.TARGETS:
            key = f"{horizon}m_{target}"
            raw = positive_probability(models[key], vector)
            calibrated = apply_calibration(raw, calibrations[key])
            predictions[str(horizon)][target] = None if calibrated is None else round(calibrated, 6)
            enabled = calibrated is not None
            calibrated_count += int(enabled)
            calibration_report[key] = {
                "aplicada": enabled,
                "motivo": calibrations[key].get("reason"),
                "muestras_calibracion": calibrations[key].get("calibration_samples"),
                "muestras_test": calibrations[key].get("test_samples"),
                "brier_test_sin_calibrar": calibrations[key].get("brier_raw_test"),
                "brier_test_calibrado": calibrations[key].get("brier_calibrated_test"),
            }

    expected_count = len(trainer.HORIZONS) * len(trainer.TARGETS)
    all_calibrated = calibrated_count == expected_count
    state_name = "CALIBRADO" if all_calibrated else ("CALIBRADO_PARCIAL" if calibrated_count else "DIAGNOSTICO_SIN_CALIBRAR")
    frame_final = selected[-1][0]
    frame_age = (radar_dt - frame_final).total_seconds() / 60.0
    if frame_age < 0 or frame_age > 15.0:
        raise RuntimeError(f"La imagen radar más reciente está desactualizada: {frame_age:.1f} min")

    track_id = track.get("track_id", track.get("id"))
    intel["ia_visual_temporal"] = {
        "disponible": True,
        "estado_modelo": state_name,
        "probabilidades_calibradas": all_calibrated,
        "probabilidades_calibradas_parcialmente": calibrated_count > 0 and not all_calibrated,
        "modelo": payload["model_version"],
        "dataset_sha256": payload.get("dataset_sha256"),
        "modelos_aprobados": payload.get("approved_models"),
        "radar_timestamp": iso(radar_dt),
        "track_id": track_id,
        "features": int(vector.shape[0]),
        "frames_utilizados": len(selected),
        "frame_inicial": iso(selected[0][0]),
        "frame_final": iso(frame_final),
        "edad_frame_final_min": round(frame_age, 2),
        "secuencia_compensada_por_movimiento": True,
        "ancla_causal_px": {"x": round(float(anchor[0]), 3), "y": round(float(anchor[1]), 3)},
        "calibracion_por_modelo": calibration_report,
        "eventos_validacion": payload.get("event_split"),
        "metricas_validacion_independiente": payload.get("validation_metrics"),
        "probabilidades": predictions,
    }
    atomic_json(INTEL_PATH, intel)
    print(json.dumps({
        "ok": True,
        "estado_modelo": state_name,
        "track_id": track_id,
        "frames": len(selected),
        "modelos_calibrados": calibrated_count,
        "modelos_totales": expected_count,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
