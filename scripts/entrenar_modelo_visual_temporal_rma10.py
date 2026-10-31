#!/usr/bin/env python3
"""
ClimaAR — RMA10 Visual/Temporal AI — MODELO DEFINITIVO

Integra:
- Variables V1.
- Secuencia visual causal de hasta 6 radares.
- Características multiescala.
- Diferencias temporales.
- PCA visual.
- Ensemble RF + ExtraTrees + HistGradientBoosting + LogisticRegression.

No utiliza información futura.
El modelo operativo final se guarda en:
data/models/rma10_visual_temporal_model.joblib
"""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data/radar/rma10_dataset_historico.csv"
RADAR_DIR = ROOT / "data/radar/historico"

MODEL_DIR = ROOT / "data/models"
MODEL_PATH = MODEL_DIR / "rma10_visual_temporal_model.joblib"
METRICS_PATH = MODEL_DIR / "rma10_visual_temporal_model_metrics.json"

MODEL_VERSION = "rma10-visual-temporal-final-ensemble-1"

HORIZONS = (5, 10, 20, 30)

TARGETS = (
    "strengthened",
    "weakened",
    "toward_bahia",
)

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

SEQUENCE_LEN = 6
MIN_SEQUENCE_LEN = 3

MAX_IMAGE_GAP_MIN = 45.0
MAX_SEQUENCE_WINDOW_MIN = 120.0

PATCH_SIZE = 64
PCA_COMPONENTS = 64

DBZ_MIN = -15.0
DBZ_MAX = 75.0

PALETTE_START = 64
PALETTE_END = 207

RANDOM_STATE = 42

_TIMESTAMP_RE = re.compile(r"^(\d{14})$")


def canonical_track_id(value) -> str:
    if value is None or pd.isna(value):
        return ""

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        if float(value).is_integer():
            return str(int(value))

    text = str(value).strip()

    try:
        numeric = float(text)
        if np.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass

    return text


def parse_timestamp(value):
    if value is None:
        return None

    dt = pd.to_datetime(
        str(value).strip(),
        utc=True,
        errors="coerce",
    )

    if pd.isna(dt):
        return None

    return dt.to_pydatetime()


def parse_timestamp_from_name(path: Path):
    match = _TIMESTAMP_RE.fullmatch(path.stem)

    if not match:
        return None

    try:
        dt = datetime.strptime(
            match.group(1),
            "%Y%m%d%H%M%S",
        )
    except ValueError:
        return None

    return dt.replace(tzinfo=timezone.utc)


def safe_auc(y_true, probability):
    if len(np.unique(y_true)) < 2:
        return None

    return float(
        roc_auc_score(
            y_true,
            probability,
        )
    )


def build_image_index():

    by_timestamp = {}

    rejected = 0
    unreadable = 0
    duplicated = 0

    for path in sorted(
        RADAR_DIR.rglob("*.png")
    ):

        timestamp = parse_timestamp_from_name(
            path
        )

        if timestamp is None:
            rejected += 1
            continue

        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            unreadable += 1
            continue

        key = timestamp.timestamp()

        if key in by_timestamp:
            duplicated += 1

            old = by_timestamp[key]

            if path.stat().st_size > old.stat().st_size:
                by_timestamp[key] = path
        else:
            by_timestamp[key] = path

    index = sorted(
        [
            (
                datetime.fromtimestamp(
                    timestamp,
                    tz=timezone.utc,
                ),
                path,
            )
            for timestamp, path in by_timestamp.items()
        ],
        key=lambda item: item[0],
    )

    if not index:
        raise RuntimeError(
            "No se encontraron imágenes RMA10 válidas."
        )

    print(
        f"Imágenes indexadas: {len(index)}"
        f" | rechazadas: {rejected}"
        f" | ilegibles: {unreadable}"
        f" | duplicadas: {duplicated}"
    )

    print(
        "Rango imágenes: "
        f"{index[0][0].isoformat()} -> "
        f"{index[-1][0].isoformat()}"
    )

    return index


@lru_cache(maxsize=128)
def load_dbz(path_string):

    with Image.open(path_string) as image:
        image = image.convert("P")

        palette = image.getpalette()

        if palette is None:
            raise ValueError(
                "Imagen RMA10 sin paleta."
            )

        indices = np.asarray(
            image,
            dtype=np.uint8,
        )

    dbz = np.full(
        indices.shape,
        np.nan,
        dtype=np.float32,
    )

    valid = (
        (indices >= PALETTE_START)
        & (indices <= PALETTE_END)
    )

    if np.any(valid):

        values = indices[valid].astype(
            np.float32
        )

        dbz[valid] = (
            DBZ_MIN
            + (
                (
                    values
                    - PALETTE_START
                )
                / float(
                    PALETTE_END
                    - PALETTE_START
                )
            )
            * (
                DBZ_MAX
                - DBZ_MIN
            )
        )

    return dbz


def extract_patch(
    dbz,
    x,
    y,
    size,
):

    height, width = dbz.shape

    cx = int(round(float(x)))
    cy = int(round(float(y)))

    half = size // 2

    x0 = cx - half
    y0 = cy - half

    x1 = x0 + size
    y1 = y0 + size

    patch = np.full(
        (size, size),
        DBZ_MIN,
        dtype=np.float32,
    )

    sx0 = max(0, x0)
    sy0 = max(0, y0)

    sx1 = min(width, x1)
    sy1 = min(height, y1)

    if sx1 <= sx0 or sy1 <= sy0:
        return patch

    source = dbz[
        sy0:sy1,
        sx0:sx1,
    ]

    finite = np.isfinite(source)

    if np.any(finite):
        fill = float(
            np.nanmedian(source)
        )
    else:
        fill = DBZ_MIN

    dx0 = sx0 - x0
    dy0 = sy0 - y0

    patch[
        dy0:dy0 + source.shape[0],
        dx0:dx0 + source.shape[1],
    ] = np.where(
        finite,
        source,
        fill,
    )

    return patch


def resize_field(
    field,
    size,
):

    image = Image.fromarray(
        field,
        mode="F",
    )

    resized = image.resize(
        (size, size),
        Image.Resampling.BILINEAR,
    )

    return np.asarray(
        resized,
        dtype=np.float32,
    )


def frame_stats(frame):

    values = frame.ravel().astype(
        np.float32
    )

    percentiles = np.percentile(
        values,
        [50, 75, 90, 95, 99],
    )

    return np.asarray(
        [
            float(values.mean()),
            float(values.std()),
            float(values.min()),
            float(values.max()),
            *percentiles.tolist(),
            float(np.mean(values > 20.0)),
            float(np.mean(values > 40.0)),
            float(np.mean(values > 50.0)),
        ],
        dtype=np.float32,
    )


def shape_features(frame):

    features = []

    for threshold in (
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
    ):

        mask = frame >= threshold

        count = int(mask.sum())

        if count == 0:
            features.extend([0.0] * 6)
            continue

        yy, xx = np.nonzero(mask)

        weights = (
            frame[mask]
            - threshold
            + 1.0
        )

        x_mean = float(xx.mean())
        y_mean = float(yy.mean())

        spread = float(
            np.sqrt(
                (
                    (xx - x_mean) ** 2
                    + (yy - y_mean) ** 2
                ).mean()
            )
            / frame.shape[0]
        )

        features.extend(
            [
                count / float(frame.size),
                x_mean / frame.shape[1],
                y_mean / frame.shape[0],
                spread,
                float(weights.mean()),
                float(frame[mask].max()),
            ]
        )

    return np.asarray(
        features,
        dtype=np.float32,
    )


def feature_vector(
    frames,
    valid_mask,
    elapsed,
    gaps,
):

    abs32 = []
    abs16 = []

    stats = []
    shapes = []

    diff32 = []
    diff16 = []

    for i in range(SEQUENCE_LEN):

        if valid_mask[i]:

            abs32.append(
                frames["32"][i].ravel()
            )

            abs16.append(
                frames["16"][i].ravel()
            )

            stats.append(
                frame_stats(
                    frames["16"][i]
                )
            )

            shapes.append(
                shape_features(
                    frames["16"][i]
                )
            )

        else:

            abs32.append(
                np.zeros(
                    32 * 32,
                    dtype=np.float32,
                )
            )

            abs16.append(
                np.zeros(
                    16 * 16,
                    dtype=np.float32,
                )
            )

            stats.append(
                np.zeros(
                    12,
                    dtype=np.float32,
                )
            )

            shapes.append(
                np.zeros(
                    30,
                    dtype=np.float32,
                )
            )

    for i in range(
        1,
        SEQUENCE_LEN,
    ):

        if (
            valid_mask[i]
            and valid_mask[i - 1]
        ):

            diff32.append(
                (
                    frames["32"][i]
                    - frames["32"][i - 1]
                ).ravel()
            )

            diff16.append(
                (
                    frames["16"][i]
                    - frames["16"][i - 1]
                ).ravel()
            )

        else:

            diff32.append(
                np.zeros(
                    32 * 32,
                    dtype=np.float32,
                )
            )

            diff16.append(
                np.zeros(
                    16 * 16,
                    dtype=np.float32,
                )
            )

    return np.concatenate(
        [
            np.concatenate(abs32),
            np.concatenate(abs16),
            np.concatenate(diff32),
            np.concatenate(diff16),
            np.concatenate(stats),
            np.concatenate(shapes),
            np.asarray(
                valid_mask,
                dtype=np.float32,
            ),
            np.asarray(
                elapsed,
                dtype=np.float32,
            ),
            np.asarray(
                gaps,
                dtype=np.float32,
            ),
        ]
    ).astype(
        np.float32
    )


def build_histories(df):

    histories = {}

    clean = df.drop_duplicates(
        subset=[
            "track_id",
            "frame_utc",
        ]
    )

    for track_id, group in clean.groupby(
        "track_id",
        sort=False,
    ):

        observations = []

        for row in group.itertuples(
            index=False
        ):

            timestamp = parse_timestamp(
                getattr(
                    row,
                    "frame_utc",
                )
            )

            if timestamp is None:
                continue

            try:
                x = float(
                    getattr(row, "x")
                )

                y = float(
                    getattr(row, "y")
                )

            except Exception:
                continue

            observations.append(
                (
                    timestamp,
                    x,
                    y,
                )
            )

        observations.sort(
            key=lambda item: item[0]
        )

        if observations:

            histories[
                canonical_track_id(
                    track_id
                )
            ] = observations

    return histories


def causal_position(
    history,
    timestamp,
):

    timestamps = [
        item[0]
        for item in history
    ]

    index = bisect_right(
        timestamps,
        timestamp,
    ) - 1

    if index < 0:
        return None

    if index == 0:
        return (
            history[0][1],
            history[0][2],
        )

    previous = history[
        index - 1
    ]

    current = history[
        index
    ]

    dt = (
        current[0]
        - previous[0]
    ).total_seconds() / 60.0

    if dt <= 0:
        return (
            current[1],
            current[2],
        )

    elapsed = (
        timestamp
        - current[0]
    ).total_seconds() / 60.0

    elapsed = max(
        0.0,
        min(
            elapsed,
            MAX_IMAGE_GAP_MIN,
        ),
    )

    vx = (
        current[1]
        - previous[1]
    ) / dt

    vy = (
        current[2]
        - previous[2]
    ) / dt

    return (
        current[1] + vx * elapsed,
        current[2] + vy * elapsed,
    )


def find_visual_sequence(
    timestamp,
    position,
    image_index,
):

    if position is None:
        return None

    x, y = position

    timestamps = [
        item[0]
        for item in image_index
    ]

    index = bisect_right(
        timestamps,
        timestamp,
    ) - 1

    if index < 0:
        return None

    selected = []

    current_index = index

    while (
        current_index >= 0
        and len(selected)
        < SEQUENCE_LEN
    ):

        frame_time, path = image_index[
            current_index
        ]

        delta = (
            timestamp
            - frame_time
        ).total_seconds() / 60.0

        if delta < 0:
            current_index -= 1
            continue

        if delta > MAX_SEQUENCE_WINDOW_MIN:
            break

        selected.append(
            (
                frame_time,
                path,
            )
        )

        current_index -= 1

    selected.reverse()

    if len(selected) < MIN_SEQUENCE_LEN:
        return None

    frames32 = []
    frames16 = []

    elapsed = []
    gaps = []

    last_time = None

    for frame_time, path in selected:

        dbz = load_dbz(
            str(path)
        )

        patch = extract_patch(
            dbz,
            x,
            y,
            PATCH_SIZE,
        )

        frames32.append(
            resize_field(
                patch,
                32,
            )
        )

        frames16.append(
            resize_field(
                patch,
                16,
            )
        )

        elapsed.append(
            (
                timestamp
                - frame_time
            ).total_seconds()
            / 60.0
        )

        if last_time is None:
            gaps.append(0.0)
        else:
            gaps.append(
                (
                    frame_time
                    - last_time
                ).total_seconds()
                / 60.0
            )

        last_time = frame_time

    while len(frames32) < SEQUENCE_LEN:

        frames32.insert(
            0,
            np.zeros(
                (32, 32),
                dtype=np.float32,
            ),
        )

        frames16.insert(
            0,
            np.zeros(
                (16, 16),
                dtype=np.float32,
            ),
        )

        elapsed.insert(
            0,
            0.0,
        )

        gaps.insert(
            0,
            0.0,
        )

    valid_mask = [
        0.0
        if i < SEQUENCE_LEN - len(selected)
        else 1.0
        for i in range(
            SEQUENCE_LEN
        )
    ]

    return (
        {
            "32": frames32,
            "16": frames16,
        },
        valid_mask,
        elapsed,
        gaps,
    )


def build_visual_dataset(
    df,
    histories,
    image_index,
):

    visual_rows = []

    for row_index, row in df.iterrows():

        track_id = canonical_track_id(
            row["track_id"]
        )

        timestamp = parse_timestamp(
            row["frame_utc"]
        )

        if timestamp is None:
            continue

        history = histories.get(
            track_id
        )

        if not history:
            continue

        position = causal_position(
            history,
            timestamp,
        )

        sequence = find_visual_sequence(
            timestamp,
            position,
            image_index,
        )

        if sequence is None:
            continue

        frames, mask, elapsed, gaps = sequence

        features = feature_vector(
            frames,
            mask,
            elapsed,
            gaps,
        )

        visual_rows.append(
            (
                row_index,
                features,
            )
        )

    print(
        f"Secuencias visuales válidas: "
        f"{len(visual_rows)}"
    )

    return visual_rows


def make_ensemble():

    return [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=700,
                max_depth=14,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            0.20,
        ),
        (
            "extra",
            ExtraTreesClassifier(
                n_estimators=700,
                max_depth=16,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            0.30,
        ),
        (
            "hist",
            HistGradientBoostingClassifier(
                max_iter=350,
                learning_rate=0.045,
                max_leaf_nodes=31,
                min_samples_leaf=12,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            ),
            0.30,
        ),
        (
            "logistic",
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            C=0.7,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            0.20,
        ),
    ]


def main():

    print(
        "=== CLIMAAR RMA10 — MODELO DEFINITIVO ==="
    )

    if not DATASET.exists():
        raise SystemExit(
            f"No existe dataset: {DATASET}"
        )

    if not RADAR_DIR.exists():
        raise SystemExit(
            f"No existe radar: {RADAR_DIR}"
        )

    df = pd.read_csv(
        DATASET
    )

    required = (
        set(V1_FEATURES)
        | set(TARGETS)
        | {
            "track_id",
            "horizonte_min",
            "frame_utc",
            "x",
            "y",
        }
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

    df["track_id"] = df[
        "track_id"
    ].map(
        canonical_track_id
    )

    print(
        f"Dataset: {len(df)} filas"
        f" | tracks: "
        f"{df['track_id'].nunique()}"
    )

    image_index = build_image_index()

    histories = build_histories(
        df
    )

    print(
        f"Historias construidas: "
        f"{len(histories)}"
    )

    visual_rows = build_visual_dataset(
        df,
        histories,
        image_index,
    )

    visual_map = {
        index: features
        for index, features
        in visual_rows
    }

    visual_dim = (
        len(next(iter(visual_map.values())))
        if visual_map
        else 0
    )

    print(
        f"Features visuales: {visual_dim}"
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            groups=df["track_id"],
        )
    )

    train_indices = set(
        df.index[train_idx]
    )

    test_indices = set(
        df.index[test_idx]
    )

    print(
        f"Tracks train: "
        f"{df.iloc[train_idx]['track_id'].nunique()}"
        f" | test: "
        f"{df.iloc[test_idx]['track_id'].nunique()}"
    )

    visual_train = [
        visual_map[index]
        for index in train_indices
        if index in visual_map
    ]

    if not visual_train:
        raise RuntimeError(
            "No hay secuencias visuales "
            "válidas en entrenamiento."
        )

    visual_train_matrix = np.vstack(
        visual_train
    )

    components = min(
        PCA_COMPONENTS,
        visual_train_matrix.shape[0],
        visual_train_matrix.shape[1],
    )

    pca = PCA(
        n_components=components,
        random_state=RANDOM_STATE,
    )

    pca.fit(
        visual_train_matrix
    )

    print(
        f"PCA visual: {components} componentes"
    )

    base_matrix = df[
        list(V1_FEATURES)
    ].astype(
        float
    ).to_numpy(
        dtype=np.float32
    )

    index_positions = {
        index: position
        for position, index
        in enumerate(df.index)
    }

    final_rows = []

    for index in df.index:

        base = base_matrix[
            index_positions[index]
        ]

        if index in visual_map:

            visual = pca.transform(
                visual_map[index].reshape(
                    1,
                    -1,
                )
            )[0]

            mask = np.array(
                [1.0],
                dtype=np.float32,
            )

        else:

            visual = np.zeros(
                components,
                dtype=np.float32,
            )

            mask = np.array(
                [0.0],
                dtype=np.float32,
            )

        final_rows.append(
            np.concatenate(
                [
                    base,
                    visual,
                    mask,
                ]
            )
        )

    X = np.vstack(
        final_rows
    ).astype(
        np.float32
    )

    print(
        f"Features finales: {X.shape[1]}"
    )

    models = {}

    metrics = {
        "model_version": MODEL_VERSION,
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset_rows": int(len(df)),
        "tracks_total": int(
            df["track_id"].nunique()
        ),
        "tracks_train": int(
            df.iloc[train_idx][
                "track_id"
            ].nunique()
        ),
        "tracks_test": int(
            df.iloc[test_idx][
                "track_id"
            ].nunique()
        ),
        "v1_features": list(
            V1_FEATURES
        ),
        "visual_raw_features": int(
            visual_dim
        ),
        "pca_features": int(
            components
        ),
        "final_features": int(
            X.shape[1]
        ),
        "horizons": list(
            HORIZONS
        ),
        "targets": list(
            TARGETS
        ),
        "models": {},
    }

    for horizon in HORIZONS:

        for target in TARGETS:

            key = f"{horizon}m_{target}"

            mask = (
                df["horizonte_min"]
                == horizon
            ).to_numpy()

            mask &= (
                df[target]
                .notna()
                .to_numpy()
            )

            mask &= (
                df["track_id"]
                != ""
            ).to_numpy()

            positions = np.flatnonzero(
                mask
            )

            train_positions = [
                pos
                for pos in positions
                if df.index[pos]
                in train_indices
            ]

            test_positions = [
                pos
                for pos in positions
                if df.index[pos]
                in test_indices
            ]

            if (
                len(train_positions) < 20
                or len(test_positions) < 10
            ):
                raise RuntimeError(
                    f"Datos insuficientes "
                    f"{key}: "
                    f"train={len(train_positions)} "
                    f"test={len(test_positions)}"
                )

            X_train = X[
                train_positions
            ]

            X_test = X[
                test_positions
            ]

            y_train = (
                df.iloc[
                    train_positions
                ][target]
                .astype(int)
                .to_numpy()
            )

            y_test = (
                df.iloc[
                    test_positions
                ][target]
                .astype(int)
                .to_numpy()
            )

            if len(np.unique(y_train)) < 2:
                raise RuntimeError(
                    f"Una sola clase en "
                    f"entrenamiento: {key}"
                )

            trained_models = []

            probabilities = []

            for (
                name,
                model,
                weight,
            ) in make_ensemble():

                model.fit(
                    X_train,
                    y_train,
                )

                probability = model.predict_proba(
                    X_test
                )[:, 1]

                trained_models.append(
                    (
                        name,
                        model,
                        weight,
                    )
                )

                probabilities.append(
                    probability * weight
                )

            total_weight = sum(
                item[2]
                for item in trained_models
            )

            probability = (
                np.sum(
                    probabilities,
                    axis=0,
                )
                / total_weight
            )

            prediction = (
                probability >= 0.5
            ).astype(int)

            auc = safe_auc(
                y_test,
                probability,
            )

            brier = float(
                brier_score_loss(
                    y_test,
                    probability,
                )
            )

            balanced = float(
                balanced_accuracy_score(
                    y_test,
                    prediction,
                )
            )

            models[key] = trained_models

            metrics["models"][key] = {
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
                "roc_auc": auc,
                "brier": brier,
                "balanced_accuracy": balanced,
            }

            print(
                f"{key}: "
                f"AUC={auc:.4f} "
                f"bal_acc={balanced:.4f} "
                f"Brier={brier:.4f}"
            )

    bundle = {
        "model_version": MODEL_VERSION,
        "features_v1": list(V1_FEATURES),
        "visual_raw_features": int(
            visual_dim
        ),
        "pca_components": int(
            components
        ),
        "sequence_len": SEQUENCE_LEN,
        "min_sequence_len": MIN_SEQUENCE_LEN,
        "horizons_min": list(
            HORIZONS
        ),
        "targets": list(
            TARGETS
        ),
        "pca": pca,
        "models": models,
        "config": {
            "patch_size": PATCH_SIZE,
            "max_image_gap_min":
                MAX_IMAGE_GAP_MIN,
            "max_sequence_window_min":
                MAX_SEQUENCE_WINDOW_MIN,
            "dbz_min": DBZ_MIN,
            "dbz_max": DBZ_MAX,
            "palette_start":
                PALETTE_START,
            "palette_end":
                PALETTE_END,
        },
    }

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_model = MODEL_PATH.with_suffix(
        ".tmp.joblib"
    )

    joblib.dump(
        bundle,
        tmp_model,
        compress=3,
    )

    tmp_model.replace(
        MODEL_PATH
    )

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=== MODELO DEFINITIVO ENTRENADO ==="
    )
    print(
        f"Modelo: {MODEL_PATH}"
    )
    print(
        f"Métricas: {METRICS_PATH}"
    )
    print(
        "Modelos entrenados: "
        f"{len(models)}/12"
    )


if __name__ == "__main__":
    main()
