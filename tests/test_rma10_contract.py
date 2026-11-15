"""Contrato entrenador <-> predictor RMA10 y aislamiento del holdout.

Se ejecuta antes de entrenar: si falla, no se gasta el entrenamiento completo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import entrenar_modelo_rma10 as trainer  # noqa: E402
from scripts import prediccion_rma10_ml as predictor  # noqa: E402
from scripts.rma10_validation import (  # noqa: E402
    chronological_event_split,
    rolling_origin_event_splits,
)

EXPECTED_VERSION = "rma10-v1-causal-3-holdout-isolated"


def toy_bundle(version=EXPECTED_VERSION):
    keys = [f"{h}m_{t}" for h in predictor.HORIZONS for t in predictor.TARGETS]
    return {
        "model_version": version,
        "features": list(predictor.FEATURES),
        "horizons_min": list(predictor.HORIZONS),
        "targets": list(predictor.TARGETS),
        "models": {k: object() for k in keys},
        "calibration": {k: {"enabled": False} for k in keys},
    }


def test_versions_match_and_are_expected():
    assert trainer.MODEL_VERSION == predictor.MODEL_VERSION == EXPECTED_VERSION


def test_contract_constants_match():
    assert trainer.FEATURES == predictor.FEATURES
    assert tuple(trainer.HORIZONS) == tuple(predictor.HORIZONS)
    assert tuple(trainer.TARGETS) == tuple(predictor.TARGETS)


def test_predictor_accepts_trainer_bundle_shape():
    models, calibration = predictor.validate_bundle(toy_bundle(trainer.MODEL_VERSION))
    assert len(models) == len(calibration) == 12


@pytest.mark.parametrize("old", ["rma10-v1", "rma10-v1-causal-2"])
def test_predictor_rejects_old_versions(old):
    with pytest.raises(RuntimeError, match="incompatible"):
        predictor.validate_bundle(toy_bundle(old))


def synthetic_events(n_events=10, rows_per_event=30):
    rows = []
    for e in range(n_events):
        for r in range(rows_per_event):
            rows.append({
                "track_id": f"t{e}",
                "frame_utc": pd.Timestamp("2026-01-01", tz="UTC")
                + pd.Timedelta(hours=e * 3, minutes=r * 5),
                "horizonte_min": 5,
            })
    return pd.DataFrame(rows)


def test_holdout_isolated_by_event():
    df = synthetic_events()
    split = chronological_event_split(df)
    tracks = df["track_id"].to_numpy()
    sets = {
        name: set(tracks[split[f"{name}_mask"]])
        for name in ("train", "calibration", "test")
    }
    assert not (sets["train"] & sets["calibration"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["calibration"] & sets["test"])
    assert (split["train_mask"] | split["calibration_mask"] | split["test_mask"]).all()


def test_rolling_folds_never_train_on_future():
    df = synthetic_events()
    first_seen = df.groupby("track_id")["frame_utc"].min()
    for fold in rolling_origin_event_splits(df, n_last_events=4, min_rows=10):
        test_t = first_seen[fold["test_event"]]
        train_tracks = set(df["track_id"].to_numpy()[fold["train_mask"]])
        assert all(first_seen[t] < test_t for t in train_tracks)
        assert fold["test_event"] not in train_tracks
