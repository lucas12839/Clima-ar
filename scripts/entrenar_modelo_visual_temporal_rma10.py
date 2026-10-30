#!/usr/bin/env python3
"""Entrenamiento visual/temporal RMA10: 6 frames -> 12 modelos."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/radar/rma10_dataset_historico.csv"
RADAR_DIR = ROOT / "data/radar/historico"
MODEL_PATH = ROOT / "data/models/rma10_visual_temporal_model.joblib"
METRICS_PATH = ROOT / "data/models/rma10_visual_temporal_model_metrics.json"

SEQUENCE_LEN = 6
PATCH_SIZE = 64
DOWNSAMPLE = 4
MAX_GAP_MIN = 20
DBZ_MIN, DBZ_MAX = -15.0, 75.0
PALETTE_FIRST, PALETTE_LAST = 64, 207


def parse_dt(value) -> datetime:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def frame_name(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S.png")


def load_rma10_field_uncached(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode != "P":
            raise ValueError(f"RMA10 inválido: {path.name}: modo {im.mode}")

        palette = im.getpalette()
        if palette is None or len(palette) < 208 * 3:
            raise ValueError(f"Paleta RMA10 inválida: {path.name}")

        idx = np.asarray(im, dtype=np.uint8)

        transparency = im.info.get("transparency")
        if isinstance(transparency, bytes):
            transparency = transparency[0] if transparency else None

        mask = (idx >= PALETTE_FIRST) & (idx <= PALETTE_LAST)

        if transparency is not None:
            mask &= idx != int(transparency)

        field = np.zeros(idx.shape, dtype=np.float32)

        field[mask] = (
            DBZ_MIN
            + (
                (idx[mask].astype(np.float32) - PALETTE_FIRST)
                / (PALETTE_LAST - PALETTE_FIRST)
            )
            * (DBZ_MAX - DBZ_MIN)
        )

        return np.clip(
            field / (DBZ_MAX - DBZ_MIN),
            0.0,
            1.0,
        )


@lru_cache(maxsize=8)
def load_rma10_field(path_str: str) -> np.ndarray:
    """Caché limitada para evitar consumir RAM excesivamente."""
    return load_rma10_field_uncached(Path(path_str))


def crop_resize(
    field: np.ndarray,
    x: float,
    y: float,
) -> np.ndarray:
    size = PATCH_SIZE
    half = size // 2

    cx = int(round(float(x)))
    cy = int(round(float(y)))

    x0 = cx - half
    y0 = cy - half
    x1 = cx + half
    y1 = cy + half

    patch = np.zeros(
        (size, size),
        dtype=np.float32,
    )

    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(field.shape[1], x1)
    sy1 = min(field.shape[0], y1)

    if sx1 > sx0 and sy1 > sy0:
        patch[
            sy0 - y0:sy1 - y0,
            sx0 - x0:sx1 - x0,
        ] = field[
            sy0:sy1,
            sx0:sx1,
        ]

    if DOWNSAMPLE > 1:
        patch = np.asarray(
            Image.fromarray(
                patch,
                mode="F",
            ).resize(
                (
                    size // DOWNSAMPLE,
                    size // DOWNSAMPLE,
                ),
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )

    return patch


def feature_vector(
    patches: list[np.ndarray],
) -> np.ndarray:
    arr = np.stack(patches).astype(np.float32)

    parts = [
        frame.ravel()
        for frame in arr
    ]

    parts += [
        (arr[i] - arr[i - 1]).ravel()
        for i in range(1, len(arr))
    ]

    stats = []

    for frame in arr:
        stats.extend(
            [
                np.mean(frame),
                np.std(frame),
                np.percentile(frame, 75),
                np.percentile(frame, 90),
                np.percentile(frame, 95),
                np.percentile(frame, 99),
                np.max(frame),
                np.count_nonzero(frame > 0.55) / frame.size,
                np.count_nonzero(frame > 0.70) / frame.size,
                np.count_nonzero(frame > 0.85) / frame.size,
            ]
        )

    means = np.array(
        [np.mean(frame) for frame in arr],
        dtype=np.float32,
    )

    p95 = np.array(
        [np.percentile(frame, 95) for frame in arr],
        dtype=np.float32,
    )

    parts += [
        np.asarray(
            stats,
            dtype=np.float32,
        ),
        np.diff(means),
        np.diff(p95),
        np.asarray(
            [
                means[-1] - means[0],
                p95[-1] - p95[0],
            ],
            dtype=np.float32,
        ),
    ]

    return np.concatenate(parts).astype(np.float32)


def load_dataset() -> pd.DataFrame:
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    df = pd.read_csv(DATASET)

    required = {
        "track_id",
        "frame_utc",
        "x",
        "y",
        "horizon_min",
        "strengthened",
        "weakened",
        "toward_bahia",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Faltan columnas: {missing}"
        )

    df = df.copy()

    df["dt"] = [
        parse_dt(value)
        for value in df["frame_utc"]
    ]

    df = df.drop_duplicates(
        [
            "track_id",
            "frame_utc",
            "horizon_min",
        ]
    )

    return (
        df.sort_values(
            [
                "track_id",
                "dt",
                "horizon_min",
            ]
        )
        .reset_index(drop=True)
    )


def build_histories(
    df: pd.DataFrame,
) -> dict:
    observations = (
        df[
            [
                "track_id",
                "frame_utc",
                "dt",
                "x",
                "y",
            ]
        ]
        .drop_duplicates(
            [
                "track_id",
                "frame_utc",
            ]
        )
        .sort_values(
            [
                "track_id",
                "dt",
            ]
        )
    )

    histories = {}

    for track_id, group in observations.groupby(
        "track_id"
    ):
        histories[track_id] = [
            {
                "frame_utc": str(row.frame_utc),
                "dt": row.dt,
                "x": float(row.x),
                "y": float(row.y),
            }
            for row in group.itertuples(
                index=False
            )
        ]

    return histories


def build_samples(
    df: pd.DataFrame,
    histories: dict,
) -> tuple[np.ndarray, pd.DataFrame]:

    X = []
    metadata = []

    positions = {
        track_id: {
            item["dt"]: i
            for i, item in enumerate(history)
        }
        for track_id, history in histories.items()
    }

    skipped = 0

    for number, row in enumerate(
        df.itertuples(index=False),
        start=1,
    ):
        history = histories.get(
            row.track_id
        )

        position = positions.get(
            row.track_id,
            {},
        ).get(row.dt)

        if (
            not history
            or position is None
            or position < SEQUENCE_LEN - 1
        ):
            skipped += 1
            continue

        sequence = history[
            position - SEQUENCE_LEN + 1:
            position + 1
        ]

        gaps = [
            (
                sequence[i]["dt"]
                - sequence[i - 1]["dt"]
            ).total_seconds() / 60.0
            for i in range(1, len(sequence))
        ]

        if any(
            gap <= 0
            or gap > MAX_GAP_MIN
            for gap in gaps
        ):
            skipped += 1
            continue

        try:
            patches = []

            for item in sequence:
                path = (
                    RADAR_DIR
                    / frame_name(item["dt"])
                )

                if not path.exists():
                    raise FileNotFoundError(path)

                field = load_rma10_field(
                    str(path)
                )

                patches.append(
                    crop_resize(
                        field,
                        item["x"],
                        item["y"],
                    )
                )

            X.append(
                feature_vector(patches)
            )

            metadata.append(
                {
                    "track_id": row.track_id,
                    "frame_utc": str(
                        row.frame_utc
                    ),
                    "horizon_min": int(
                        row.horizon_min
                    ),
                    "strengthened": int(
                        row.strengthened
                    ),
                    "weakened": int(
                        row.weakened
                    ),
                    "toward_bahia": int(
                        row.toward_bahia
                    ),
                }
            )

        except (
            OSError,
            ValueError,
        ):
            skipped += 1

        if number % 500 == 0:
            print(
                f"[secuencias] "
                f"{number}/{len(df)} | "
                f"válidas={len(X)} | "
                f"omitidas={skipped}"
            )

    if not X:
        raise RuntimeError(
            "No se pudo construir ninguna "
            "secuencia visual válida."
        )

    return (
        np.stack(X),
        pd.DataFrame(metadata),
    )


def train_one(
    X: np.ndarray,
    meta: pd.DataFrame,
    target: str,
    horizon: int,
):
    subset = meta[
        meta.horizon_min == horizon
    ]

    if subset.empty:
        return None, {
            "status": "insufficient_data",
            "target": target,
            "horizon_min": horizon,
            "samples": 0,
        }

    indices = subset.index.to_numpy()

    y = subset[
        target
    ].astype(int).to_numpy()

    groups = subset[
        "track_id"
    ].to_numpy()

    if len(np.unique(y)) < 2:
        return None, {
            "status": "insufficient_classes",
            "target": target,
            "horizon_min": horizon,
            "samples": len(y),
            "classes": sorted(
                np.unique(y).tolist()
            ),
        }

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(
            X[indices],
            y,
            groups,
        )
    )

    if len(
        np.unique(y[train_idx])
    ) < 2:
        return None, {
            "status": "invalid_train_split",
            "target": target,
            "horizon_min": horizon,
            "samples": len(y),
        }

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X[indices][train_idx],
        y[train_idx],
    )

    if len(model.classes_) != 2:
        return None, {
            "status": "invalid_model_classes",
            "target": target,
            "horizon_min": horizon,
            "samples": len(y),
        }

    positive_position = int(
        np.where(
            model.classes_ == 1
        )[0][0]
    )

    probabilities = (
        model.predict_proba(
            X[indices][test_idx]
        )[:, positive_position]
    )

    predictions = model.predict(
        X[indices][test_idx]
    )

    y_test = y[test_idx]

    metrics = {
        "status": "ok",
        "target": target,
        "horizon_min": horizon,
        "samples": int(len(y)),
        "train_samples": int(
            len(train_idx)
        ),
        "test_samples": int(
            len(test_idx)
        ),
        "positive_rate": float(
            np.mean(y)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_test,
                predictions,
            )
        ),
        "brier": float(
            brier_score_loss(
                y_test,
                probabilities,
            )
        ),
        "auc": (
            float(
                roc_auc_score(
                    y_test,
                    probabilities,
                )
            )
            if len(np.unique(y_test)) == 2
            else None
        ),
    }

    return model, metrics


def main():
    print(
        "=== RMA10 VISUAL/TEMPORAL AI ==="
    )

    df = load_dataset()

    print(
        f"Dataset: {len(df)} filas | "
        f"tracks: {df.track_id.nunique()}"
    )

    histories = build_histories(df)

    print(
        f"Historias: {len(histories)}"
    )

    X, metadata = build_samples(
        df,
        histories,
    )

    print(
        f"Secuencias válidas: {len(X)} | "
        f"features: {X.shape[1]}"
    )

    models = {}

    metrics = {
        "version":
            "rma10-visual-temporal-v1",
        "sequence_len":
            SEQUENCE_LEN,
        "patch_size":
            PATCH_SIZE,
        "downsample":
            DOWNSAMPLE,
        "max_gap_min":
            MAX_GAP_MIN,
        "input":
            "RMA10 normalized dBZ temporal patches",
        "split":
            "GroupShuffleSplit by track_id",
        "models": {},
    }

    for horizon in (
        5,
        10,
        20,
        30,
    ):
        for target in (
            "strengthened",
            "weakened",
            "toward_bahia",
        ):
            print(
                f"Entrenando "
                f"{target} "
                f"{horizon} min..."
            )

            model, result = train_one(
                X,
                metadata,
                target,
                horizon,
            )

            key = (
                f"{target}_{horizon}m"
            )

            metrics[
                "models"
            ][key] = result

            if model is not None:
                models[key] = model

                print(
                    f"OK | "
                    f"n={result['samples']} | "
                    f"bal_acc="
                    f"{result['balanced_accuracy']:.3f} | "
                    f"AUC="
                    f"{result['auc']}"
                )
            else:
                print(
                    "NO ENTRENADO | "
                    f"{result['status']}"
                )

    if not models:
        raise RuntimeError(
            "No se entrenó ningún modelo."
        )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "version":
                "rma10-visual-temporal-v1",
            "models":
                models,
            "sequence_len":
                SEQUENCE_LEN,
            "patch_size":
                PATCH_SIZE,
            "downsample":
                DOWNSAMPLE,
            "max_gap_min":
                MAX_GAP_MIN,
            "dbz_min":
                DBZ_MIN,
            "dbz_max":
                DBZ_MAX,
            "palette_first":
                PALETTE_FIRST,
            "palette_last":
                PALETTE_LAST,
        },
        MODEL_PATH,
    )

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "=== FINALIZADO ==="
    )

    print(
        f"Modelo: {MODEL_PATH}"
    )

    print(
        f"Métricas: {METRICS_PATH}"
    )

    print(
        f"Modelos entrenados: "
        f"{len(models)}/12"
    )


if __name__ == "__main__":
    main()
