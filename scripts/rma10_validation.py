"""Validation utilities shared by the causal RMA10 trainers.

All splits are grouped by track_id and ordered by first observation time so
rows from the same tracked echo cannot leak across train/calibration/test.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score


_REQUIRED_EVENT_COLUMNS = {"track_id", "frame_utc"}
_FEATURE_COLUMNS = {
    "past_intensity_delta",
    "past_area_ratio",
    "motion_valid",
    "motion_bearing_deg",
    "bearing_to_bahia_deg",
}


def _event_table(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    missing = _REQUIRED_EVENT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas para split cronológico: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset vacío: no se pueden crear splits")
    tracks = df["track_id"].astype(str).str.strip()
    timestamps = pd.to_datetime(df["frame_utc"], utc=True, errors="coerce")
    if tracks.eq("").any() or timestamps.isna().any():
        raise ValueError("track_id/frame_utc contienen valores vacíos o inválidos")
    work = pd.DataFrame({"track_id": tracks.to_numpy(), "timestamp": timestamps.to_numpy()})
    events = (
        work.groupby("track_id", sort=False)["timestamp"]
        .min()
        .sort_values(kind="stable")
        .rename("first_timestamp")
        .reset_index()
    )
    event_order = {track: i for i, track in enumerate(events["track_id"].tolist())}
    event_ids = tracks.map(event_order).to_numpy(dtype=int)
    return events, event_ids


def _masks(
    event_ids: np.ndarray,
    train_events: set[int],
    cal_events: set[int],
    test_events: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.isin(event_ids, list(train_events)),
        np.isin(event_ids, list(cal_events)),
        np.isin(event_ids, list(test_events)),
    )


def chronological_event_split(df: pd.DataFrame) -> dict[str, Any]:
    """Chronological 60/20/20 split with whole tracks assigned to one partition."""
    events, event_ids = _event_table(df)
    count = len(events)
    if count < 5:
        raise ValueError(
            f"Se requieren al menos 5 tracks/eventos independientes; encontrados={count}"
        )
    train_end = max(1, int(math.floor(count * 0.60)))
    cal_end = max(train_end + 1, int(math.floor(count * 0.80)))
    cal_end = min(cal_end, count - 1)
    train_set = set(range(0, train_end))
    cal_set = set(range(train_end, cal_end))
    test_set = set(range(cal_end, count))
    train_mask, calibration_mask, test_mask = _masks(
        event_ids, train_set, cal_set, test_set
    )
    if not train_mask.any() or not calibration_mask.any() or not test_mask.any():
        raise ValueError("Split cronológico produjo una partición vacía")
    return {
        "train_mask": train_mask,
        "calibration_mask": calibration_mask,
        "test_mask": test_mask,
        "event_by_track": dict(zip(events["track_id"], range(count))),
        "event_count": count,
        "train_events": len(train_set),
        "calibration_events": len(cal_set),
        "test_events": len(test_set),
        "train_rows": int(train_mask.sum()),
        "calibration_rows": int(calibration_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "first_event_utc": events.iloc[0]["first_timestamp"].isoformat(),
        "last_event_utc": events.iloc[-1]["first_timestamp"].isoformat(),
    }


def rolling_origin_event_splits(
    df: pd.DataFrame, n_last_events: int = 4, min_rows: int = 100
) -> list[dict[str, Any]]:
    """Expanding-window folds: past tracks=train, prior track=calibration, next=test."""
    if n_last_events < 1 or min_rows < 1:
        raise ValueError("n_last_events y min_rows deben ser positivos")
    events, event_ids = _event_table(df)
    count = len(events)
    first_test = max(2, count - int(n_last_events))
    folds: list[dict[str, Any]] = []
    for test_event in range(first_test, count):
        calibration_event = test_event - 1
        train_set = set(range(0, calibration_event))
        cal_set = {calibration_event}
        test_set = {test_event}
        train_mask, calibration_mask, test_mask = _masks(
            event_ids, train_set, cal_set, test_set
        )
        if min(
            int(train_mask.sum()),
            int(calibration_mask.sum()),
            int(test_mask.sum()),
        ) < min_rows:
            continue
        folds.append(
            {
                "train_mask": train_mask,
                "calibration_mask": calibration_mask,
                "test_mask": test_mask,
                "test_event": str(events.iloc[test_event]["track_id"]),
                "calibration_event": str(events.iloc[calibration_event]["track_id"]),
                "train_events": len(train_set),
                "train_samples": int(train_mask.sum()),
                "calibration_samples": int(calibration_mask.sum()),
                "test_samples": int(test_mask.sum()),
            }
        )
    return folds


def classification_metrics(y_true: Any, probability: Any) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    if y.size != p.size or y.size == 0:
        raise ValueError("y_true y probability deben tener igual longitud no vacía")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Los targets deben ser binarios (0/1)")
    if not np.isfinite(p).all():
        raise ValueError("Las probabilidades contienen NaN o infinitos")
    p = np.clip(p, 0.0, 1.0)
    predicted = (p >= 0.5).astype(int)
    auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None
    return {
        "samples": int(y.size),
        "positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "roc_auc": auc,
        "brier": float(brier_score_loss(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
    }


def persistence_baseline(features: Any, target: str) -> np.ndarray:
    """Causal persistence score from the latest observed trend/direction only."""
    if target not in {"strengthened", "weakened", "toward_bahia"}:
        raise ValueError(f"Target de persistencia desconocido: {target}")
    if not isinstance(features, pd.DataFrame):
        raise TypeError("persistence_baseline requiere un DataFrame con features causales")
    missing = _FEATURE_COLUMNS - set(features.columns)
    if missing:
        raise ValueError(f"Faltan features para baseline de persistencia: {sorted(missing)}")
    intensity = (
        pd.to_numeric(features["past_intensity_delta"], errors="coerce")
        .fillna(0)
        .to_numpy(float)
    )
    area = (
        pd.to_numeric(features["past_area_ratio"], errors="coerce")
        .fillna(1)
        .to_numpy(float)
    )
    valid = (
        pd.to_numeric(features["motion_valid"], errors="coerce")
        .fillna(0)
        .to_numpy(float)
        > 0.5
    )
    if target == "strengthened":
        return np.where(
            (intensity >= 1.0) & (area >= 0.80),
            0.80,
            np.where(intensity > 0, 0.60, 0.20),
        )
    if target == "weakened":
        return np.where(
            (intensity <= -1.0) & (area <= 1.25),
            0.80,
            np.where(intensity < 0, 0.60, 0.20),
        )
    motion = (
        pd.to_numeric(features["motion_bearing_deg"], errors="coerce")
        .fillna(0)
        .to_numpy(float)
    )
    bearing = (
        pd.to_numeric(features["bearing_to_bahia_deg"], errors="coerce")
        .fillna(0)
        .to_numpy(float)
    )
    delta = np.abs((motion - bearing + 180.0) % 360.0 - 180.0)
    directional_score = np.clip(1.0 - delta / 90.0, 0.0, 1.0)
    return np.where(valid, directional_score, 0.20)


def fit_platt_calibration(
    raw_cal: Any,
    y_cal: Any,
    raw_test: Any,
    y_test: Any,
    min_samples: int = 100,
) -> dict[str, Any]:
    """Fit Platt scaling on calibration data only.

    raw_test and y_test remain in the signature for backwards compatibility with
    existing trainers, but are deliberately not inspected here. The held-out
    test partition must only be used by the trainer for final reporting and
    model-vs-persistence evaluation, never to fit or enable calibration.
    """
    cal_p = np.asarray(raw_cal, dtype=float).reshape(-1)
    cal_y = np.asarray(y_cal, dtype=int).reshape(-1)

    try:
        test_samples = int(np.asarray(raw_test).reshape(-1).size)
    except (TypeError, ValueError):
        test_samples = 0

    result: dict[str, Any] = {
        "enabled": False,
        "approved": False,
        "method": "PLATT_LOGIT",
        "reason": None,
        "calibration_samples": int(cal_y.size),
        "test_samples": test_samples,
        "a": None,
        "b": None,
        "brier_raw_test": None,
        "brier_calibrated_test": None,
    }

    if cal_p.size != cal_y.size or not cal_y.size:
        result["reason"] = "INVALID_CALIBRATION_ARRAY_LENGTHS"
        return result
    if (
        not np.isfinite(cal_p).all()
        or not np.isin(cal_y, [0, 1]).all()
        or (cal_p < 0.0).any()
        or (cal_p > 1.0).any()
    ):
        result["reason"] = "NONFINITE_OR_INVALID_CALIBRATION_DATA"
        return result
    if cal_y.size < min_samples or np.unique(cal_y).size < 2:
        result["reason"] = "INSUFFICIENT_CALIBRATION_SAMPLES_OR_CLASSES"
        return result

    eps = 1e-6
    clipped = np.clip(cal_p, eps, 1.0 - eps)
    cal_logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    try:
        model.fit(cal_logits, cal_y)
    except (ValueError, FloatingPointError) as exc:
        result["reason"] = f"PLATT_FIT_FAILED: {type(exc).__name__}"
        return result

    result.update(
        {
            "a": float(model.coef_[0, 0]),
            "b": float(model.intercept_[0]),
            "enabled": True,
            "approved": False,
            "reason": None,
        }
    )
    return result
