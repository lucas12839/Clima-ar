#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(".")
DATASET = ROOT / "data" / "radar" / "rma10_dataset_historico.csv"
MODEL_DIR = ROOT / "data" / "models"
MODEL_PATH = MODEL_DIR / "rma10_model.joblib"
METRICS_PATH = MODEL_DIR / "rma10_model_metrics.json"

HORIZONS = (5, 10, 20, 30)

TARGETS = (
    "strengthened",
    "weakened",
    "toward_bahia",
)

FEATURES = (
    "intensity",
    "area_px",
    "width_px",
    "height_px",
    "lat",
    "lon",
    "distance_km",
    "bearing_to_bahia_deg",
    "displacement_px",
    "speed_px_min",
    "motion_bearing_deg",
)


def make_model():
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def safe_auc(y_true, proba):
    if y_true.nunique() < 2:
        return None

    return float(
        roc_auc_score(
            y_true,
            proba,
        )
    )


def train_one(
    df,
    horizon,
    target,
    groups_train,
    groups_test,
):
    subset = df[
        df["horizonte_min"] == horizon
    ].copy()

    subset = subset.dropna(
        subset=list(FEATURES) + [target, "track_id"]
    )

    if subset.empty:
        raise RuntimeError(
            f"No hay datos para horizonte={horizon}, target={target}"
        )

    train = subset[
        subset["track_id"].isin(groups_train)
    ]

    test = subset[
        subset["track_id"].isin(groups_test)
    ]

    if len(train) < 20 or len(test) < 10:
        raise RuntimeError(
            f"Datos insuficientes para "
            f"{horizon}m/{target}: "
            f"train={len(train)}, test={len(test)}"
        )

    y_train = train[target].astype(int)
    y_test = test[target].astype(int)

    if y_train.nunique() < 2:
        raise RuntimeError(
            f"El entrenamiento {horizon}m/{target} "
            f"tiene una sola clase."
        )

    model = make_model()

    model.fit(
        train[list(FEATURES)],
        y_train,
    )

    pred = model.predict(
        test[list(FEATURES)]
    )

    proba = model.predict_proba(
        test[list(FEATURES)]
    )[:, 1]

    metrics = {
        "samples_train": int(len(train)),
        "samples_test": int(len(test)),
        "positive_train": int(y_train.sum()),
        "positive_test": int(y_test.sum()),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_test,
                pred,
            )
        ),
        "roc_auc": safe_auc(
            y_test,
            proba,
        ),
        "classification_report": classification_report(
            y_test,
            pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    return model, metrics


def main():

    if not DATASET.exists():
        raise SystemExit(
            f"No existe el dataset RMA10: {DATASET}"
        )

    df = pd.read_csv(DATASET)

    required = (
        set(FEATURES)
        | {"horizonte_min", "track_id"}
        | set(TARGETS)
    )

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise SystemExit(
            f"Faltan columnas obligatorias: {missing}"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df["track_id"] = df["track_id"].astype(str)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            groups=df["track_id"],
        )
    )

    groups_train = set(
        df.iloc[train_idx]["track_id"]
    )

    groups_test = set(
        df.iloc[test_idx]["track_id"]
    )

    models = {}

    metrics = {
        "model_version": "rma10-v1",
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat().replace(
            "+00:00",
            "Z",
        ),
        "dataset": str(DATASET),
        "dataset_rows": int(len(df)),
        "tracks_total": int(
            df["track_id"].nunique()
        ),
        "tracks_train": int(
            len(groups_train)
        ),
        "tracks_test": int(
            len(groups_test)
        ),
        "features": list(FEATURES),
        "horizons_min": list(HORIZONS),
        "targets": list(TARGETS),
        "models": {},
    }

    for horizon in HORIZONS:

        for target in TARGETS:

            key = f"{horizon}m_{target}"

            model, result = train_one(
                df,
                horizon,
                target,
                groups_train,
                groups_test,
            )

            models[key] = model
            metrics["models"][key] = result

    bundle = {
        "model_version": "rma10-v1",
        "features": list(FEATURES),
        "horizons_min": list(HORIZONS),
        "targets": list(TARGETS),
        "models": models,
    }

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        bundle,
        MODEL_PATH,
        compress=3,
    )

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "ok": True,
                "model_path": str(
                    MODEL_PATH
                ),
                "metrics_path": str(
                    METRICS_PATH
                ),
                "dataset_rows": len(df),
                "tracks_total": int(
                    df["track_id"].nunique()
                ),
                "models_trained": len(
                    models
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
