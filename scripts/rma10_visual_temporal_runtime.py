#!/usr/bin/env python3
"""
ClimaAR - Runtime RMA10 Visual/Temporal AI V6.

Carga el modelo visual/temporal entrenado y genera las 12 probabilidades
operativas para el track principal actual, usando exactamente la misma
representación de features del entrenamiento.

No reemplaza la lógica heurística/fusión existente. Agrega el resultado
bajo `ia_visual_temporal` en inteligencia_tormenta.json.
"""

from __future__ import annotations

import json
import sys
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

BASE = Path(__file__).resolve().parents[1]
SCRIPTS = BASE / "scripts"

MODEL_PATH = BASE / "data/models/rma10_visual_temporal_model.joblib"
INTEL_PATH = BASE / "data/ia/inteligencia_tormenta.json"
ESTADO_PATH = BASE / "data/radar/seguimiento/estado.json"
SEGUIMIENTO_PATH = BASE / "data/radar/seguimiento/seguimiento_temporal.json"
TIMESTAMP_PATH = BASE / "data/radar/radar_timestamp.txt"

sys.path.insert(0, str(SCRIPTS))
import entrenar_modelo_visual_temporal_rma10 as trainer  # noqa: E402


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_ts(value):
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def principal_track(estado, seguimiento):
    principal_id = seguimiento.get("track_principal_candidato_id")
    if principal_id is None:
        raise RuntimeError("Tracking no definió track_principal_candidato_id.")

    principal_text = str(principal_id)
    for track in estado.get("tracks", []):
        if not isinstance(track, dict):
            continue
        track_id = track.get("track_id", track.get("id"))
        if track_id is not None and str(track_id) == principal_text:
            if track.get("activo") is False:
                raise RuntimeError("El track principal no está activo.")
            if track.get("presente_en_ultimo_frame") is False:
                raise RuntimeError("El track principal no está presente en el último frame.")
            return track

    raise RuntimeError(
        f"El track principal {principal_id} no existe entre los tracks activos."
    )


def track_history_for_trainer(track):
    history = []
    for item in track.get("historial", []):
        if not isinstance(item, dict):
            continue
        ts = parse_ts(item.get("timestamp", item.get("frame_utc")))
        if ts is None:
            continue
        try:
            x = float(item["x"])
            y = float(item["y"])
        except Exception:
            continue
        history.append((ts, x, y))
    history.sort(key=lambda x: x[0])
    if not history:
        raise RuntimeError("El track principal no tiene historial utilizable.")
    return history


def build_runtime_feature(track, radar_dt, image_index):
    history = track_history_for_trainer(track)
    image_times = [ts.timestamp() for ts, _ in image_index]

    end = bisect_right(image_times, radar_dt.timestamp()) - 1
    if end < 0:
        raise RuntimeError("No hay imágenes RMA10 anteriores al radar actual.")

    window_start = radar_dt.timestamp() - trainer.MAX_SEQUENCE_WINDOW_MIN * 60.0
    selected = []
    for idx in range(end, -1, -1):
        ts, path = image_index[idx]
        if ts.timestamp() < window_start:
            break
        selected.append((ts, path))
        if len(selected) == trainer.SEQUENCE_LEN:
            break
    selected.reverse()

    if len(selected) < trainer.MIN_SEQUENCE_LEN:
        raise RuntimeError(
            f"Solo hay {len(selected)} frames RMA10 válidos; "
            f"se requieren {trainer.MIN_SEQUENCE_LEN}."
        )

    gaps = [
        (selected[i][0] - selected[i - 1][0]).total_seconds() / 60.0
        for i in range(1, len(selected))
    ]
    if any(gap <= 0 or gap > trainer.MAX_IMAGE_GAP_MIN for gap in gaps):
        raise RuntimeError("La secuencia RMA10 actual tiene gaps inválidos.")

    anchor = trainer.causal_position(history, radar_dt)
    if anchor is None:
        raise RuntimeError("No se pudo obtener posición causal del track principal.")

    frames_32 = [
        np.zeros((32, 32), dtype=np.float32)
        for _ in range(trainer.SEQUENCE_LEN)
    ]
    frames_16 = [
        np.zeros((16, 16), dtype=np.float32)
        for _ in range(trainer.SEQUENCE_LEN)
    ]
    valid_mask = [False] * trainer.SEQUENCE_LEN
    elapsed_min = [trainer.MAX_SEQUENCE_WINDOW_MIN] * trainer.SEQUENCE_LEN
    feature_gaps = [trainer.MAX_IMAGE_GAP_MIN] * (trainer.SEQUENCE_LEN - 1)

    start = trainer.SEQUENCE_LEN - len(selected)

    for index, (image_ts, path) in enumerate(selected):
        dbz = trainer.load_dbz(str(path.resolve()))
        patch = trainer.extract_patch(
            dbz,
            anchor[0],
            anchor[1],
            trainer.PATCH_SIZE,
        )
        slot = start + index
        frames_32[slot] = trainer.resize_field(patch, 32)
        frames_16[slot] = trainer.resize_field(patch, 16)
        valid_mask[slot] = True
        elapsed_min[slot] = max(
            0.0,
            (radar_dt - image_ts).total_seconds() / 60.0,
        )

    for index, gap in enumerate(gaps):
        feature_gaps[start + index] = gap

    vector = trainer.feature_vector(
        {"32": frames_32, "16": frames_16},
        valid_mask,
        elapsed_min,
        feature_gaps,
    )

    return vector, selected, anchor


def positive_probability(model, vector):
    probabilities = model.predict_proba(vector.reshape(1, -1))[0]
    classes = list(model.classes_)
    if 1 in classes:
        return float(probabilities[classes.index(1)])
    return 0.0


def main():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"No existe el modelo V6: {MODEL_PATH}")
    if not INTEL_PATH.exists():
        raise RuntimeError(
            "No existe inteligencia_tormenta.json. "
            "Ejecutá primero tormenta_intelligence_timestamp.py."
        )

    intel = load_json(INTEL_PATH)
    estado = load_json(ESTADO_PATH)
    seguimiento = load_json(SEGUIMIENTO_PATH)

    radar_ts = parse_ts(
        intel.get("radar_timestamp")
        or TIMESTAMP_PATH.read_text(encoding="utf-8").strip()
    )
    if radar_ts is None:
        raise RuntimeError("No se pudo determinar el timestamp del radar.")

    intel_ts = parse_ts(intel.get("radar_timestamp"))
    if intel_ts is not None and intel_ts != radar_ts:
        raise RuntimeError("inteligencia_tormenta.json está desincronizado.")

    tracking_ts = parse_ts(seguimiento.get("radar_timestamp"))
    if tracking_ts is not None and tracking_ts != radar_ts:
        raise RuntimeError("Tracking y radar están desincronizados; no se publica V6.")

    payload = joblib.load(MODEL_PATH)
    if not isinstance(payload, dict):
        raise RuntimeError("Payload del modelo V6 inválido.")

    version = payload.get("version")
    models = payload.get("models")
    feature_config = payload.get("feature_config", {})
    expected_features = int(feature_config.get("feature_count", 0))

    if not isinstance(models, dict) or len(models) != 12:
        raise RuntimeError("El modelo V6 no contiene los 12 modelos esperados.")

    image_index = trainer.build_image_index()
    track = principal_track(estado, seguimiento)
    vector, selected, anchor = build_runtime_feature(
        track, radar_ts, image_index
    )

    if expected_features and vector.shape[0] != expected_features:
        raise RuntimeError(
            f"Features incompatibles: runtime={vector.shape[0]} "
            f"modelo={expected_features}."
        )

    predictions = {}
    for horizon in trainer.HORIZONS:
        predictions[str(horizon)] = {}
        for target in trainer.TARGETS:
            key = (target, horizon)
            model = models.get(key)
            if model is None:
                raise RuntimeError(f"Falta modelo V6: {key}")
            predictions[str(horizon)][target] = round(
                positive_probability(model, vector),
                6,
            )

    track_id = track.get("track_id", track.get("id"))
    intel["ia_visual_temporal"] = {
        "disponible": True,
        "modelo": version,
        "radar_timestamp": iso(radar_ts),
        "track_id": track_id,
        "features": int(vector.shape[0]),
        "frames_utilizados": len(selected),
        "frame_inicial": iso(selected[0][0]),
        "frame_final": iso(selected[-1][0]),
        "ancla_causal_px": {
            "x": round(float(anchor[0]), 3),
            "y": round(float(anchor[1]), 3),
        },
        "probabilidades": predictions,
    }

    atomic_json(INTEL_PATH, intel)

    print("=== RMA10 VISUAL/TEMPORAL V6 OPERATIVO ===")
    print(f"Modelo: {version}")
    print(f"Track principal: {track_id}")
    print(f"Features: {vector.shape[0]}")
    print(f"Frames: {len(selected)}")
    for horizon in trainer.HORIZONS:
        row = predictions[str(horizon)]
        print(
            f"{horizon} min | "
            f"strengthened={row['strengthened']:.4f} | "
            f"weakened={row['weakened']:.4f} | "
            f"toward_bahia={row['toward_bahia']:.4f}"
        )


if __name__ == "__main__":
    main()
