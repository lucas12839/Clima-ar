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


def _track_table(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Return whole-track time bounds; tracks remain independent identities."""
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
    table = work.groupby("track_id", sort=False)["timestamp"].agg(first_timestamp="min", last_timestamp="max")
    table = table.sort_values(["first_timestamp", "last_timestamp"], kind="stable")
    return table, tracks.to_numpy(), timestamps


def _masks_for_tracks(track_values: np.ndarray, groups: tuple[set[str], set[str], set[str]]):
    return tuple(np.isin(track_values, list(group)) for group in groups)


def _counts_by_horizon(df: pd.DataFrame, masks: tuple[np.ndarray, ...], min_rows: int) -> bool:
    if "horizonte_min" not in df.columns:
        return all(int(mask.sum()) >= min_rows for mask in masks)
    horizons = pd.to_numeric(df["horizonte_min"], errors="coerce").to_numpy()
    required = sorted(pd.Series(horizons).dropna().unique().tolist())
    return all(int((mask & (horizons == h)).sum()) >= min_rows for mask in masks for h in required)


def chronological_event_split(df: pd.DataFrame) -> dict[str, Any]:
    """Chronological 60/20/20 split; whole tracks crossing a boundary are excluded."""
    table, track_values, timestamps = _track_table(df)
    starts = table["first_timestamp"].sort_values().reset_index(drop=True)
    if len(table) < 5:
        raise ValueError(f"Se requieren al menos 5 tracks independientes; encontrados={len(table)}")
    start_boundary = starts.iloc[max(1, int(len(starts) * .60))]
    test_boundary = starts.iloc[max(2, int(len(starts) * .80))]
    train_ids = set(table.index[table["last_timestamp"] < start_boundary])
    cal_ids = set(table.index[(table["first_timestamp"] >= start_boundary) & (table["last_timestamp"] < test_boundary)])
    test_ids = set(table.index[table["first_timestamp"] >= test_boundary])
    masks = _masks_for_tracks(track_values, (train_ids, cal_ids, test_ids))
    if any(not m.any() for m in masks):
        raise ValueError("Split cronológico produjo una partición vacía tras excluir tracks que cruzan límites")
    # Strict temporal separation is checked on observation timestamps, not track start alone.
    max_train = timestamps[masks[0]].max(); min_cal = timestamps[masks[1]].min()
    max_cal = timestamps[masks[1]].max(); min_test = timestamps[masks[2]].min()
    if not (max_train < min_cal and max_cal < min_test):
        raise RuntimeError("Fuga temporal detectada entre particiones")
    return {
        "train_mask": masks[0], "calibration_mask": masks[1], "test_mask": masks[2],
        "event_by_track": {track: i for i, track in enumerate(table.index)},
        "event_count": len(table), "train_events": len(train_ids),
        "calibration_events": len(cal_ids), "test_events": len(test_ids),
        "train_rows": int(masks[0].sum()), "calibration_rows": int(masks[1].sum()),
        "test_rows": int(masks[2].sum()),
        "first_event_utc": table.iloc[0]["first_timestamp"].isoformat(),
        "last_event_utc": table.iloc[-1]["first_timestamp"].isoformat(),
    }


def rolling_origin_event_splits(
    df: pd.DataFrame, n_last_events: int = 4, min_rows: int = 100
) -> list[dict[str, Any]]:
    """Expanding temporal windows; each full track belongs to one split only.

    Calibration and test use consecutive two-day windows. Tracks crossing either
    boundary are excluded rather than split. No cells are merged into episodes.
    """
    if n_last_events < 1 or min_rows < 1:
        raise ValueError("n_last_events y min_rows deben ser positivos")
    table, track_values, timestamps = _track_table(df)
    if len(table) < 5:
        return []
    first_day = table["first_timestamp"].min().ceil("D")
    last_day = table["last_timestamp"].max().floor("D")
    candidates = pd.date_range(first_day + pd.Timedelta(days=2), last_day - pd.Timedelta(days=4), freq="D")
    folds: list[dict[str, Any]] = []
    for cal_start in candidates:
        test_start = cal_start + pd.Timedelta(days=2)
        test_end = test_start + pd.Timedelta(days=2)
        train_ids = set(table.index[table["last_timestamp"] < cal_start])
        cal_ids = set(table.index[(table["first_timestamp"] >= cal_start) & (table["last_timestamp"] < test_start)])
        test_ids = set(table.index[(table["first_timestamp"] >= test_start) & (table["last_timestamp"] < test_end)])
        masks = _masks_for_tracks(track_values, (train_ids, cal_ids, test_ids))
        if not _counts_by_horizon(df, masks, min_rows):
            continue
        max_train = timestamps[masks[0]].max(); min_cal = timestamps[masks[1]].min()
        max_cal = timestamps[masks[1]].max(); min_test = timestamps[masks[2]].min()
        if not (max_train < min_cal and max_cal < min_test):
            continue
        folds.append({
            "train_mask": masks[0], "calibration_mask": masks[1], "test_mask": masks[2],
            "test_event": f"{test_start.isoformat()}..{test_end.isoformat()}",
            "calibration_event": f"{cal_start.isoformat()}..{test_start.isoformat()}",
            "train_events": len(train_ids), "train_samples": int(masks[0].sum()),
            "calibration_samples": int(masks[1].sum()), "test_samples": int(masks[2].sum()),
            "train_max_utc": max_train.isoformat(), "calibration_min_utc": min_cal.isoformat(),
            "calibration_max_utc": max_cal.isoformat(), "test_min_utc": min_test.isoformat(),
        })
    return folds[-max(2, int(n_last_events)):]


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
