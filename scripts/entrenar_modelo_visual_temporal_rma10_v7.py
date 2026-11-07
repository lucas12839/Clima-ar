#!/usr/bin/env python3
"""
ClimaAR — RMA10 Visual/Temporal AI V7

Objetivo:
- Mantener V1 como referencia.
- Aprender sobre el mismo split por track_id que V1.
- Combinar la representación visual/temporal RMA10 con las variables
  cinemáticas ya demostradas por V1.
- Reducir la dimensionalidad visual con PCA entrenada SOLO con train.
- Añadir descriptores compactos de estructura, intensidad y evolución.
- Evaluar V7 y V1 sobre exactamente los mismos casos.
- No toca el modelo operativo existente.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from entrenar_modelo_visual_temporal_rma10 import (
    build_histories,
    build_image_index,
    build_sequence_samples,
    HORIZONS,
    TARGETS,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/radar/rma10_dataset_historico.csv"

MODEL_PATH = ROOT / "data/models/rma10_visual_temporal_v7_model.joblib"
METRICS_PATH = ROOT / "data/models/rma10_visual_temporal_v7_model_metrics.json"

MODEL_VERSION = "rma10-visual-temporal-v7-radar-ensemble"
RANDOM_STATE = 42
PCA_COMPONENTS = 64

V1_FEATURES = (
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


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def frame_summary(frame):
    a = np.asarray(frame, dtype=np.float32)

    return np.asarray(
        [
            float(a.mean()),
            float(a.std()),
            float(a.min()),
            float(a.max()),
            float(np.percentile(a, 90)),
            float(np.percentile(a, 95)),
            float(np.mean(a > 20)),
            float(np.mean(a > 40)),
        ],
        dtype=np.float32,
    )


def compact_visual_features(X):
    """
    Extrae información interpretable de los 14349 features V6.
    No utiliza ninguna etiqueta.
    """

    X = np.asarray(X, dtype=np.float32)

    # Layout exacto de V6.
    o_abs32 = 0
    o_abs16 = 6144
    o_diff32 = 7680
    o_diff16 = 12800
    o_stats = 14080
    o_shapes = 14152
    o_valid = 14332
    o_elapsed = 14338
    o_gaps = 14344

    out = []

    for row in X:
        f = []

        abs32 = row[o_abs32:o_abs16].reshape(6, 32, 32)
        abs16 = row[o_abs16:o_diff32].reshape(6, 16, 16)
        diff32 = row[o_diff32:o_diff16].reshape(5, 32, 32)
        diff16 = row[o_diff16:o_stats].reshape(5, 16, 16)

        # Resumen de cada frame.
        for frame in abs32:
            f.extend(frame_summary(frame))

        for frame in abs16:
            f.extend(frame_summary(frame))

        # Evolución entre frames.
        for frame in diff32:
            f.extend(
                [
                    float(frame.mean()),
                    float(frame.std()),
                    float(np.mean(frame > 2.0)),
                    float(np.mean(frame < -2.0)),
                ]
            )

        for frame in diff16:
            f.extend(
                [
                    float(frame.mean()),
                    float(frame.std()),
                    float(np.mean(frame > 2.0)),
                    float(np.mean(frame < -2.0)),
                ]
            )

        # Tendencias temporales.
        means = np.asarray(
            [frame.mean() for frame in abs16],
            dtype=np.float32,
        )

        p95 = np.asarray(
            [np.percentile(frame, 95) for frame in abs16],
            dtype=np.float32,
        )

        high40 = np.asarray(
            [np.mean(frame > 40) for frame in abs16],
            dtype=np.float32,
        )

        f.extend(
            [
                float(means[-1] - means[0]),
                float(p95[-1] - p95[0]),
                float(high40[-1] - high40[0]),
                float(np.polyfit(np.arange(6), means, 1)[0]),
                float(np.polyfit(np.arange(6), p95, 1)[0]),
                float(np.polyfit(np.arange(6), high40, 1)[0]),
            ]
        )

        # Descriptores originales de V6.
        f.extend(row[o_stats:o_shapes].tolist())
        f.extend(row[o_shapes:o_valid].tolist())

        # valid + elapsed + gaps.
        f.extend(row[o_valid : o_gaps + 5].tolist())

        out.append(f)

    return np.asarray(out, dtype=np.float32)


def build_v1_features(df, metadata):
    keyed = {}

    for row in df.itertuples(index=False):
        track = str(getattr(row, "track_id")).strip()
        ts = str(getattr(row, "frame_utc")).strip()

        keyed[(track, ts)] = [
            float(getattr(row, feature))
            for feature in V1_FEATURES
        ]

    values = []
    missing = 0

    for row in metadata.itertuples(index=False):
        key = (
            str(row.track_id).strip(),
            str(row.frame_utc).strip(),
        )

        item = keyed.get(key)

        if item is None or not np.all(np.isfinite(item)):
            missing += 1
            values.append([0.0] * len(V1_FEATURES))
        else:
            values.append(item)

    return np.asarray(values, dtype=np.float32), missing


def make_models():

    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=10,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    et = ExtraTreesClassifier(
        n_estimators=600,
        max_depth=12,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=43,
        n_jobs=-1,
    )

    hgb = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.035,
        max_leaf_nodes=15,
        min_samples_leaf=8,
        l2_regularization=2.0,
        random_state=44,
    )

    scaler = StandardScaler()

    lr = LogisticRegression(
        max_iter=2000,
        C=0.25,
        class_weight="balanced",
        solver="liblinear",
        random_state=45,
    )

    return rf, et, hgb, scaler, lr


def main():

    print("=== RMA10 VISUAL/TEMPORAL AI V7 ===")
    print("Version:", MODEL_VERSION)

    df = pd.read_csv(DATASET)

    df["track_id"] = df["track_id"].astype(str)

    df = df[
        df["horizonte_min"].isin(HORIZONS)
    ].copy()

    required = set(V1_FEATURES) | {
        "frame_utc",
        "track_id",
        "x",
        "y",
        "horizonte_min",
        *TARGETS,
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Faltan columnas: {missing}"
        )

    histories = build_histories(df)

    image_index = build_image_index()

    X_visual, metadata, counters, sequence_counts = build_sequence_samples(
        df,
        histories,
        image_index,
    )

    if len(X_visual) < 300:
        raise RuntimeError(
            f"Solo hay {len(X_visual)} secuencias visuales."
        )

    X_compact = compact_visual_features(
        X_visual
    )

    X_v1, missing_v1 = build_v1_features(
        df,
        metadata,
    )

    print(
        "Secuencias visuales:",
        len(X_visual),
    )

    print(
        "Features V6 originales:",
        X_visual.shape[1],
    )

    print(
        "Features visuales compactas:",
        X_compact.shape[1],
    )

    print(
        "Features V1:",
        X_v1.shape[1],
    )

    print(
        "Asociaciones V1 faltantes:",
        missing_v1,
    )

    # Asociar cada secuencia visual
    # con su fila correspondiente.
    lookup = {}

    for i, row in metadata.iterrows():

        lookup[
            (
                str(row["track_id"]).strip(),
                str(row["frame_utc"]).strip(),
            )
        ] = int(i)

    # MISMO split de V1.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    train_idx_df, test_idx_df = next(
        splitter.split(
            df,
            groups=df["track_id"],
        )
    )

    train_tracks = set(
        df.iloc[train_idx_df]["track_id"].astype(str)
    )

    test_tracks = set(
        df.iloc[test_idx_df]["track_id"].astype(str)
    )

    models = {}

    metrics = {
        "version": MODEL_VERSION,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "sequences_valid": int(
            len(X_visual)
        ),
        "features_originales": int(
            X_visual.shape[1]
        ),
        "features_compactas": int(
            X_compact.shape[1]
        ),
        "features_v1": list(V1_FEATURES),
        "pca_components": PCA_COMPONENTS,
        "split": (
            "GroupShuffleSplit "
            "test_size=0.25 "
            "random_state=42 "
            "por track_id"
        ),
        "models": {},
    }

    for horizon in HORIZONS:

        horizon_rows = df[
            df["horizonte_min"] == horizon
        ]

        associations = []

        for row in horizon_rows.itertuples(
            index=False
        ):

            key = (
                str(row.track_id).strip(),
                str(row.frame_utc).strip(),
            )

            idx = lookup.get(key)

            if idx is not None:
                associations.append(
                    (idx, row)
                )

        if not associations:
            continue

        idx_all = np.asarray(
            [
                x[0]
                for x in associations
            ],
            dtype=np.int64,
        )

        rows_all = [
            x[1]
            for x in associations
        ]

        groups_all = np.asarray(
            [
                str(r.track_id)
                for r in rows_all
            ],
            dtype=object,
        )

        train_mask = np.isin(
            groups_all,
            list(train_tracks),
        )

        test_mask = np.isin(
            groups_all,
            list(test_tracks),
        )

        train_ids = idx_all[
            train_mask
        ]

        test_ids = idx_all[
            test_mask
        ]

        if (
            len(train_ids) < 50
            or len(test_ids) < 20
        ):
            raise RuntimeError(
                f"Datos insuficientes V7 "
                f"{horizon}m: "
                f"train={len(train_ids)} "
                f"test={len(test_ids)}"
            )

        # PCA SOLO sobre train.
        pca = PCA(
            n_components=min(
                PCA_COMPONENTS,
                len(train_ids) - 1,
            ),
            svd_solver="randomized",
            random_state=RANDOM_STATE,
        )

        pca.fit(
            X_visual[train_ids]
        )

        visual_train = pca.transform(
            X_visual[train_ids]
        )

        visual_test = pca.transform(
            X_visual[test_ids]
        )

        # Combinación:
        # visual compacto + V1 + PCA visual.
        Z_train = np.hstack(
            [
                X_compact[train_ids],
                X_v1[train_ids],
                visual_train,
            ]
        )

        Z_test = np.hstack(
            [
                X_compact[test_ids],
                X_v1[test_ids],
                visual_test,
            ]
        )

        for target in TARGETS:

            key = f"{target}_{horizon}"

            y = np.asarray(
                [
                    int(
                        getattr(
                            r,
                            target,
                        )
                    )
                    for r in rows_all
                ],
                dtype=np.int8,
            )

            y_train = y[
                train_mask
            ]

            y_test = y[
                test_mask
            ]

            if (
                len(np.unique(y_train)) < 2
                or len(np.unique(y_test)) < 2
            ):
                metrics["models"][key] = {
                    "status": "invalid_classes"
                }

                continue

            rf, et, hgb, scaler, lr = make_models()

            rf.fit(
                Z_train,
                y_train,
            )

            et.fit(
                Z_train,
                y_train,
            )

            hgb.fit(
                Z_train,
                y_train,
            )

            Ztr_lr = scaler.fit_transform(
                Z_train
            )

            Zte_lr = scaler.transform(
                Z_test
            )

            lr.fit(
                Ztr_lr,
                y_train,
            )

            p_rf = rf.predict_proba(
                Z_test
            )[:, 1]

            p_et = et.predict_proba(
                Z_test
            )[:, 1]

            p_hgb = hgb.predict_proba(
                Z_test
            )[:, 1]

            p_lr = lr.predict_proba(
                Zte_lr
            )[:, 1]

            # Ensemble.
            p = (
                0.20 * p_rf
                + 0.30 * p_et
                + 0.30 * p_hgb
                + 0.20 * p_lr
            )

            pred = (
                p >= 0.5
            ).astype(np.int8)

            result = {
                "status": "ok",
                "samples_train": int(
                    len(y_train)
                ),
                "samples_test": int(
                    len(y_test)
                ),
                "positive_train": int(
                    y_train.sum()
                ),
                "positive_test": int(
                    y_test.sum()
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(
                        y_test,
                        pred,
                    )
                ),
                "auc": safe_auc(
                    y_test,
                    p,
                ),
                "brier": float(
                    brier_score_loss(
                        y_test,
                        p,
                    )
                ),
            }

            metrics["models"][key] = result

            models[key] = {
                "horizon": horizon,
                "target": target,
                "pca": pca,
                "rf": rf,
                "et": et,
                "hgb": hgb,
                "scaler": scaler,
                "lr": lr,
                "compact_size": int(
                    X_compact.shape[1]
                ),
                "v1_size": int(
                    X_v1.shape[1]
                ),
            }

            print(
                f"{key}: "
                f"train={len(y_train)} "
                f"test={len(y_test)} "
                f"AUC={result['auc']:.4f} "
                f"bal_acc="
                f"{result['balanced_accuracy']:.4f} "
                f"Brier="
                f"{result['brier']:.4f}"
            )

    if len(models) != 12:
        raise RuntimeError(
            f"Solo se entrenaron "
            f"{len(models)}/12 modelos V7."
        )

    bundle = {
        "model_version": MODEL_VERSION,
        "v1_features": list(
            V1_FEATURES
        ),
        "pca_components": PCA_COMPONENTS,
        "compact_feature_size": int(
            X_compact.shape[1]
        ),
        "models": models,
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_model = MODEL_PATH.with_suffix(
        ".joblib.tmp"
    )

    joblib.dump(
        bundle,
        tmp_model,
        compress=3,
    )

    tmp_model.replace(
        MODEL_PATH
    )

    tmp_metrics = METRICS_PATH.with_suffix(
        ".json.tmp"
    )

    tmp_metrics.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    tmp_metrics.replace(
        METRICS_PATH
    )

    print("=== V7 FINALIZADO ===")
    print("Modelo:", MODEL_PATH)
    print("Métricas:", METRICS_PATH)


if __name__ == "__main__":
    main()
