#!/usr/bin/env python3
"""Train V1 RMA10 using causal features and chronological storm-event validation."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rma10_features import V1_FEATURES
from scripts.rma10_validation import (
    classification_metrics,
    chronological_event_split,
    fit_platt_calibration,
    persistence_baseline,
    rolling_origin_event_splits,
)

ROOT = REPO_ROOT
DATASET = ROOT / "data/radar/rma10_dataset_historico.csv"
SUMMARY = ROOT / "data/radar/rma10_dataset_resumen.json"
MODEL_DIR = ROOT / "data/models"
MODEL_PATH = MODEL_DIR / "rma10_model.joblib"
METRICS_PATH = MODEL_DIR / "rma10_model_metrics.json"

MODEL_VERSION = "rma10-v1-causal-3-holdout-isolated"
FEATURES = list(V1_FEATURES)
HORIZONS = (5, 10, 20, 30)
TARGETS = ("strengthened", "weakened", "toward_bahia")
MIN_CALIBRATION_SAMPLES = 100
MIN_TEST_SAMPLES = 100


def make_model(n_estimators=300):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive_probability(model, matrix):
    probabilities = model.predict_proba(matrix)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(matrix), dtype=float)
    return probabilities[:, classes.index(1)]


def calibrated_probability(raw, calibration):
    if not calibration.get("enabled"):
        return None
    eps = 1e-6
    raw = np.clip(np.asarray(raw, dtype=float), eps, 1.0 - eps)
    logits = np.log(raw / (1.0 - raw))
    z = np.clip(float(calibration["a"]) * logits + float(calibration["b"]), -60, 60)
    return 1.0 / (1.0 + np.exp(-z))



def evaluate_rolling_origin(df, horizon, target, folds):
    subset = df[df["horizonte_min"].astype(int) == horizon]
    results = []
    for fold in folds:
        indices = subset.index.to_numpy()
        train = subset.loc[fold["train_mask"][indices]]
        calibration_set = subset.loc[fold["calibration_mask"][indices]]
        test = subset.loc[fold["test_mask"][indices]]
        if min(len(train), len(calibration_set), len(test)) < MIN_CALIBRATION_SAMPLES:
            continue
        y_train = train[target].astype(int).to_numpy()
        y_cal = calibration_set[target].astype(int).to_numpy()
        y_test = test[target].astype(int).to_numpy()
        if len(np.unique(y_train)) < 2 or len(np.unique(y_cal)) < 2 or len(np.unique(y_test)) < 2:
            continue
        model = make_model(n_estimators=100)
        model.fit(train[FEATURES].astype(float), y_train)
        raw_cal = positive_probability(model, calibration_set[FEATURES].astype(float))
        raw_test = positive_probability(model, test[FEATURES].astype(float))
        calibration = fit_platt_calibration(raw_cal, y_cal, raw_test, y_test, min_samples=MIN_CALIBRATION_SAMPLES)
        raw_metrics = classification_metrics(y_test, raw_test)
        baseline_metrics = classification_metrics(y_test, persistence_baseline(test, target))
        beats = bool(
            raw_metrics["roc_auc"] is not None
            and raw_metrics["roc_auc"] > 0.55
            and raw_metrics["brier"] < baseline_metrics["brier"]
            and raw_metrics["balanced_accuracy"] is not None
            and baseline_metrics["balanced_accuracy"] is not None
            and raw_metrics["balanced_accuracy"] >= baseline_metrics["balanced_accuracy"]
        )
        results.append({
            "test_event": fold["test_event"],
            "calibration_event": fold["calibration_event"],
            "train_samples": int(len(train)),
            "calibration_samples": int(len(calibration_set)),
            "test_samples": int(len(test)),
            "model_raw": raw_metrics,
            "persistence_baseline": baseline_metrics,
            "calibration_enabled": bool(calibration.get("enabled")),
            "beats_persistence": beats,
            "approved": bool(beats and calibration.get("enabled")),
        })
    return results

def train_one(df, split, horizon: int, target: str, rolling_folds):
    mask_horizon = df["horizonte_min"].astype(int) == horizon
    subset = df.loc[mask_horizon].copy()
    train = subset.loc[split["train_mask"][subset.index.to_numpy()]]
    calibration_set = subset.loc[split["calibration_mask"][subset.index.to_numpy()]]
    test = subset.loc[split["test_mask"][subset.index.to_numpy()]]

    if train.empty or calibration_set.empty or test.empty:
        raise RuntimeError(f"Split vacío para {horizon}m/{target}")

    X_train = train[FEATURES].astype(float)
    y_train = train[target].astype(int)
    X_cal = calibration_set[FEATURES].astype(float)
    y_cal = calibration_set[target].astype(int)
    X_test = test[FEATURES].astype(float)
    y_test = test[target].astype(int)

    if not np.isfinite(X_train.to_numpy()).all() or not np.isfinite(X_cal.to_numpy()).all() or not np.isfinite(X_test.to_numpy()).all():
        raise RuntimeError(f"Features no finitas para {horizon}m/{target}")

    if y_train.nunique() >= 2:
        model = make_model()
        model.fit(X_train, y_train)
    else:
        model = DummyClassifier(strategy="prior")
        model.fit(X_train, y_train)

    raw_cal = positive_probability(model, X_cal)
    raw_test = positive_probability(model, X_test)
    y_test_array = y_test.to_numpy()

    calibration = fit_platt_calibration(
        raw_cal,
        y_cal.to_numpy(),
        raw_test,
        y_test_array,
        min_samples=MIN_CALIBRATION_SAMPLES,
    )

    baseline = persistence_baseline(X_test, target)
    baseline_metrics = classification_metrics(y_test_array, baseline)
    raw_metrics = classification_metrics(y_test_array, raw_test)

    calibrated_metrics = None
    if calibration.get("a") is not None and calibration.get("b") is not None:
        calibrated = calibrated_probability(raw_test, calibration)
        if calibrated is not None:
            calibrated_metrics = classification_metrics(y_test_array, calibrated)

    beats_baseline = bool(
        raw_metrics["samples"] >= MIN_TEST_SAMPLES
        and raw_metrics["roc_auc"] is not None
        and raw_metrics["roc_auc"] > 0.55
        and raw_metrics["brier"] < baseline_metrics["brier"]
        and raw_metrics["balanced_accuracy"] is not None
        and baseline_metrics["balanced_accuracy"] is not None
        and raw_metrics["balanced_accuracy"] >= baseline_metrics["balanced_accuracy"]
    )
    # The final holdout is reporting-only. It must never select or disable
    # calibration/model approval; that decision comes from rolling-origin folds.
    rolling_metrics = evaluate_rolling_origin(df, horizon, target, rolling_folds)
    rolling_passes = sum(bool(item["approved"]) for item in rolling_metrics)
    rolling_ok = len(rolling_metrics) >= 2 and rolling_passes == len(rolling_metrics)
    if calibration.get("enabled") and not rolling_ok:
        calibration["enabled"] = False
        calibration["reason"] = "ROLLING_ORIGIN_VALIDATION_NOT_PASSED"
    calibration["rolling_origin_folds"] = len(rolling_metrics)
    calibration["rolling_origin_passes"] = rolling_passes
    calibration["approved"] = bool(calibration.get("enabled") and rolling_ok)
    if not calibration.get("enabled"):
        calibrated_metrics = None

    metrics = {
        "horizon_min": horizon,
        "target": target,
        "train_samples": int(len(train)),
        "calibration_samples": int(len(calibration_set)),
        "test_samples": int(len(test)),
        "train_positive": int(y_train.sum()),
        "calibration_positive": int(y_cal.sum()),
        "test_positive": int(y_test.sum()),
        "baseline_persistence": baseline_metrics,
        "model_raw": raw_metrics,
        "model_calibrated": calibrated_metrics,
        "beats_persistence": beats_baseline,
        "calibration_enabled": bool(calibration.get("enabled")),
        "calibration_reason": calibration.get("reason"),
        "rolling_origin": rolling_metrics,
        "rolling_origin_passes": rolling_passes,
        "rolling_origin_folds": len(rolling_metrics),
    }
    return model, calibration, metrics


def main():
    if not DATASET.exists() or not SUMMARY.exists():
        raise SystemExit("Falta dataset RMA10 o su resumen. Reconstruir antes de entrenar.")

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "rma10-causal-v2" or summary.get("migration_version") != "rma10-causal-migration-2":
        raise SystemExit("Dataset RMA10 sin migración causal v2. Migrar antes de entrenar.")
    if summary.get("features_causales") != FEATURES:
        raise SystemExit("El esquema de features del dataset no coincide con V1.")
    if float(summary.get("radio_operativo_km", 0)) != 240.0:
        raise SystemExit("El dataset no conserva el radio operativo RMA10 de 240 km.")

    df = pd.read_csv(DATASET)
    required = set(FEATURES) | {"horizonte_min", "track_id", "frame_utc", *TARGETS, "label_valid"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Dataset incompleto: {sorted(missing)}")
    df = df[df["label_valid"].astype(bool)].copy()
    df["track_id"] = df["track_id"].astype(str).str.strip()
    df["horizonte_min"] = pd.to_numeric(df["horizonte_min"], errors="coerce")
    df = df.dropna(subset=["track_id", "horizonte_min", "frame_utc", *FEATURES, *TARGETS])
    df = df.sort_values(["frame_utc", "track_id", "horizonte_min"]).reset_index(drop=True)

    split = chronological_event_split(df)
    rolling_folds = rolling_origin_event_splits(df, n_last_events=4, min_rows=MIN_CALIBRATION_SAMPLES)
    if len(rolling_folds) < 2:
        raise SystemExit(f"Se requieren al menos 2 folds rolling-origin; encontrados={len(rolling_folds)}")
    models = {}
    calibrations = {}
    validation_metrics = {}

    for horizon in HORIZONS:
        for target in TARGETS:
            key = f"{horizon}m_{target}"
            model, calibration, metrics = train_one(df, split, horizon, target, rolling_folds)
            models[key] = model
            calibrations[key] = calibration
            validation_metrics[key] = metrics
            print(
                f"{key}: test={metrics['test_samples']} "
                f"AUC={metrics['model_raw']['roc_auc']} "
                f"Brier={metrics['model_raw']['brier']:.4f} "
                f"persistencia={metrics['baseline_persistence']['brier']:.4f} "
                f"calibrado={metrics['calibration_enabled']}"
            )

    dataset_hash = sha256_file(DATASET)
    trained_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    bundle = {
        "model_version": MODEL_VERSION,
        "features_version": "rma10-causal-features-v2",
        "features": FEATURES,
        "horizons_min": list(HORIZONS),
        "targets": list(TARGETS),
        "models": models,
        "calibration": calibrations,
        "validation_metrics": validation_metrics,
        "event_split": split,
        "dataset_sha256": dataset_hash,
        "trained_at_utc": trained_at,
        "approved_models": sum(bool(item.get("approved")) for item in calibrations.values()),
        "total_models": len(models),
    }
    # Boolean masks and track mapping are audit-only; keep a compact serializable split summary.
    bundle["event_split"] = {
        key: value for key, value in split.items()
        if key not in {"train_mask", "calibration_mask", "test_mask", "event_by_track"}
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)

    metrics_payload = {
        "model_version": MODEL_VERSION,
        "features": FEATURES,
        "dataset_sha256": dataset_hash,
        "trained_at_utc": trained_at,
        "event_split": bundle["event_split"],
        "approved_models": bundle["approved_models"],
        "total_models": bundle["total_models"],
        "models": validation_metrics,
    }
    METRICS_PATH.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "ok": True,
        "model": str(MODEL_PATH),
        "approved_models": bundle["approved_models"],
        "total_models": bundle["total_models"],
        "event_split": bundle["event_split"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
