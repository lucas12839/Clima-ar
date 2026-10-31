#!/usr/bin/env python3
"""
ClimaAR — RMA10 Visual/Temporal AI V6

Mejoras sobre V5:
- Mantiene el anclaje espacial causal único de V5.
- Secuencias de imágenes RMA10 reales, sin repetir frames.
- Representación multiescala 32x32 + 16x16.
- Diferencias temporales multiescala.
- Descriptores visuales de estructura/intensidad por umbrales.
- Ensemble RF + ExtraTrees + HistGradientBoosting.
- Validación separada por track_id.
- Escritura atómica del modelo y métricas.
- No modifica el decoder operativo ni la lógica RGB histórica.
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
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data/radar/rma10_dataset_historico.csv"
RADAR_DIR = ROOT / "data/radar/historico"

MODEL_PATH = ROOT / "data/models/rma10_visual_temporal_model.joblib"
METRICS_PATH = ROOT / "data/models/rma10_visual_temporal_model_metrics.json"

MODEL_VERSION = "rma10-visual-temporal-v6-multiscale-ensemble-final"

SEQUENCE_LEN = 6
MIN_SEQUENCE_LEN = 3
MAX_SEQUENCE_WINDOW_MIN = 120.0
MAX_IMAGE_GAP_MIN = 45.0

PATCH_SIZE = 64
DOWNSAMPLE_SIZES = (32, 16)

HORIZONS = (5, 10, 20, 30)
TARGETS = ("strengthened", "weakened", "toward_bahia")

RANDOM_STATE = 42

DBZ_MIN = -15.0
DBZ_MAX = 75.0
PALETTE_START = 64
PALETTE_END = 207

_TIMESTAMP_RE = re.compile(r"^(\d{14})$")


def parse_timestamp_from_name(path: Path) -> datetime | None:
    match = _TIMESTAMP_RE.fullmatch(path.stem)

    if not match:
        return None

    try:
        dt = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None

    if not 2020 <= dt.year <= 2035:
        return None

    return dt.replace(tzinfo=timezone.utc)


def parse_dataset_timestamp(value) -> datetime | None:
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


def decode_rma10_dbz(image: Image.Image) -> np.ndarray:
    image = image.convert("P")

    if image.getpalette() is None:
        raise ValueError("Imagen RMA10 sin paleta")

    indices = np.asarray(image, dtype=np.uint8)

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
        values = indices[valid].astype(np.float32)

        dbz[valid] = (
            DBZ_MIN
            + (
                (values - PALETTE_START)
                / float(PALETTE_END - PALETTE_START)
            )
            * (DBZ_MAX - DBZ_MIN)
        )

    return dbz


def parse_bool_label(value) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value != 0)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "si", "sí"}:
        return 1
    if text in {"false", "0", "no", "n"}:
        return 0
    raise ValueError(f"Etiqueta booleana inválida: {value!r}")


def build_image_index():
    by_timestamp = {}

    rejected_name = 0
    unreadable = 0
    duplicate_timestamp = 0

    for path in sorted(RADAR_DIR.rglob("*.png")):
        timestamp = parse_timestamp_from_name(path)

        if timestamp is None:
            rejected_name += 1
            continue

        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            unreadable += 1
            continue

        key = timestamp.timestamp()

        if key in by_timestamp:
            duplicate_timestamp += 1

            old = by_timestamp[key]

            if path.stat().st_size > old.stat().st_size:
                by_timestamp[key] = path

        else:
            by_timestamp[key] = path

    index = sorted(
        (
            (
                datetime.fromtimestamp(
                    timestamp,
                    tz=timezone.utc,
                ),
                path,
            )
            for timestamp, path in by_timestamp.items()
        ),
        key=lambda item: item[0],
    )

    if not index:
        raise RuntimeError(
            "No se encontraron imágenes RMA10 válidas."
        )

    print(
        f"Imágenes indexadas: {len(index)}"
        f" | rechazadas: {rejected_name}"
        f" | ilegibles: {unreadable}"
        f" | duplicadas: {duplicate_timestamp}"
    )

    print(
        f"Rango imágenes: "
        f"{index[0][0].isoformat()} -> "
        f"{index[-1][0].isoformat()}"
    )

    return index


@lru_cache(maxsize=96)
def load_dbz(path_string: str) -> np.ndarray:
    with Image.open(path_string) as image:
        return decode_rma10_dbz(image)


def extract_patch(
    dbz: np.ndarray,
    x: float,
    y: float,
    size: int,
) -> np.ndarray:
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

    source = dbz[sy0:sy1, sx0:sx1]

    finite = np.isfinite(source)

    fill = (
        float(np.nanmedian(source))
        if np.any(finite)
        else DBZ_MIN
    )

    dx0 = sx0 - x0
    dy0 = sy0 - y0

    patch[
        dy0 : dy0 + source.shape[0],
        dx0 : dx0 + source.shape[1],
    ] = np.where(
        finite,
        source,
        fill,
    )

    return patch


def resize_field(
    field: np.ndarray,
    size: int,
) -> np.ndarray:
    resized = Image.fromarray(
        field,
        mode="F",
    ).resize(
        (size, size),
        Image.Resampling.BILINEAR,
    )

    return np.asarray(
        resized,
        dtype=np.float32,
    )


def frame_stats(frame: np.ndarray) -> np.ndarray:
    values = frame.ravel().astype(np.float32)

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


def shape_features(frame: np.ndarray) -> np.ndarray:
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
    elapsed_min,
    gaps_min,
) -> np.ndarray:
    absolute_32 = np.concatenate(
        [
            frame.ravel()
            for frame in frames["32"]
        ]
    )

    absolute_16 = np.concatenate(
        [
            frame.ravel()
            for frame in frames["16"]
        ]
    )

    diffs_32 = []
    diffs_16 = []

    stats = []
    shapes = []

    for index in range(SEQUENCE_LEN):
        if valid_mask[index]:
            stats.append(
                frame_stats(
                    frames["16"][index]
                )
            )

            shapes.append(
                shape_features(
                    frames["16"][index]
                )
            )
        else:
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

    for index in range(1, SEQUENCE_LEN):
        if (
            valid_mask[index]
            and valid_mask[index - 1]
        ):
            diffs_32.append(
                (
                    frames["32"][index]
                    - frames["32"][index - 1]
                ).ravel()
            )

            diffs_16.append(
                (
                    frames["16"][index]
                    - frames["16"][index - 1]
                ).ravel()
            )
        else:
            diffs_32.append(
                np.zeros(
                    32 * 32,
                    dtype=np.float32,
                )
            )

            diffs_16.append(
                np.zeros(
                    16 * 16,
                    dtype=np.float32,
                )
            )

    return np.concatenate(
        [
            absolute_32,
            absolute_16,
            np.concatenate(diffs_32),
            np.concatenate(diffs_16),
            np.concatenate(stats),
            np.concatenate(shapes),
            np.asarray(
                valid_mask,
                dtype=np.float32,
            ),
            np.asarray(
                elapsed_min,
                dtype=np.float32,
            ),
            np.asarray(
                gaps_min,
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


def build_histories(df: pd.DataFrame):
    required = {
        "frame_utc",
        "track_id",
        "x",
        "y",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing)}"
        )

    histories = {}

    clean = df.drop_duplicates(
        subset=["track_id", "frame_utc"]
    )

    for track_id, group in clean.groupby(
        "track_id",
        sort=False,
    ):
        observations = []

        for row in group.itertuples(index=False):
            timestamp = parse_dataset_timestamp(
                getattr(row, "frame_utc")
            )

            if timestamp is None:
                continue

            try:
                x = float(getattr(row, "x"))
                y = float(getattr(row, "y"))
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
            histories[track_id] = observations

    return histories


def causal_position(
    history,
    timestamp,
):
    timestamps = [
        item[0]
        for item in history
    ]

    index = np.searchsorted(
        timestamps,
        timestamp,
        side="right",
    ) - 1

    if index < 0:
        return None

    if index == 0:
        return (
            history[0][1],
            history[0][2],
        )

    previous = history[index - 1]
    current = history[index]

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


def select_real_images(
    target_timestamp,
    image_index,
    image_times,
):
    if not image_index:
        return []

    end = bisect_right(
        image_times,
        target_timestamp.timestamp(),
    ) - 1

    if end < 0:
        return []

    window_start = (
        target_timestamp.timestamp()
        - MAX_SEQUENCE_WINDOW_MIN * 60.0
    )

    selected = []

    for index in range(
        end,
        -1,
        -1,
    ):
        timestamp, path = image_index[index]

        if timestamp.timestamp() < window_start:
            break

        selected.append(
            (
                timestamp,
                path,
            )
        )

        if len(selected) == SEQUENCE_LEN:
            break

    selected.reverse()

    return selected


def build_sequence_samples(
    df,
    histories,
    image_index,
):
    image_times = [
        timestamp.timestamp()
        for timestamp, _ in image_index
    ]

    features = []
    metadata = []

    counters = {
        "rows": 0,
        "short_history": 0,
        "bad_gap": 0,
        "missing_images": 0,
        "valid": 0,
    }

    sequence_length_counts = {
        str(length): 0
        for length in range(
            MIN_SEQUENCE_LEN,
            SEQUENCE_LEN + 1,
        )
    }

    for row in df.itertuples(
        index=False
    ):
        counters["rows"] += 1

        track_id = getattr(
            row,
            "track_id",
        )

        history = histories.get(
            track_id
        )

        target_timestamp = parse_dataset_timestamp(
            getattr(row, "frame_utc")
        )

        if not history or target_timestamp is None:
            continue

        selected = select_real_images(
            target_timestamp,
            image_index,
            image_times,
        )

        if len(selected) < MIN_SEQUENCE_LEN:
            counters["short_history"] += 1
            continue

        gaps = [
            (
                selected[index][0]
                - selected[index - 1][0]
            ).total_seconds() / 60.0
            for index in range(
                1,
                len(selected),
            )
        ]

        if any(
            gap <= 0
            or gap > MAX_IMAGE_GAP_MIN
            for gap in gaps
        ):
            counters["bad_gap"] += 1
            continue

        # Una única referencia espacial causal:
        # todos los frames miran el mismo punto.
        # El desplazamiento del núcleo queda visible.
        anchor = causal_position(
            history,
            target_timestamp,
        )

        if anchor is None:
            counters["missing_images"] += 1
            continue

        frames_32 = [
            np.zeros(
                (32, 32),
                dtype=np.float32,
            )
            for _ in range(SEQUENCE_LEN)
        ]

        frames_16 = [
            np.zeros(
                (16, 16),
                dtype=np.float32,
            )
            for _ in range(SEQUENCE_LEN)
        ]

        valid_mask = [
            False
        ] * SEQUENCE_LEN

        elapsed_min = [
            MAX_SEQUENCE_WINDOW_MIN
        ] * SEQUENCE_LEN

        feature_gaps = [
            MAX_IMAGE_GAP_MIN
        ] * (SEQUENCE_LEN - 1)

        start = (
            SEQUENCE_LEN
            - len(selected)
        )

        ok = True

        for index, (
            image_timestamp,
            path,
        ) in enumerate(selected):
            if not ok:
                break

            try:
                dbz = load_dbz(
                    str(path.resolve())
                )

                patch = extract_patch(
                    dbz,
                    anchor[0],
                    anchor[1],
                    PATCH_SIZE,
                )

                slot = start + index

                frames_32[slot] = resize_field(
                    patch,
                    32,
                )

                frames_16[slot] = resize_field(
                    patch,
                    16,
                )

                valid_mask[slot] = True

                elapsed_min[slot] = max(
                    0.0,
                    (
                        target_timestamp
                        - image_timestamp
                    ).total_seconds() / 60.0,
                )

            except Exception:
                ok = False

        if not ok:
            counters["missing_images"] += 1
            continue

        for index, gap in enumerate(gaps):
            feature_gaps[
                start + index
            ] = gap

        vector = feature_vector(
            {
                "32": frames_32,
                "16": frames_16,
            },
            valid_mask,
            elapsed_min,
            feature_gaps,
        )

        features.append(vector)

        metadata.append(
            {
                "track_id": track_id,
                "frame_utc": target_timestamp.isoformat(),
            }
        )

        counters["valid"] += 1

        sequence_length_counts[
            str(len(selected))
        ] += 1

    for key, value in counters.items():
        print(
            f"{key}: {value}"
        )

    print(
        "Longitudes de secuencia: "
        f"{sequence_length_counts}"
    )

    return (
        np.asarray(
            features,
            dtype=np.float32,
        ),
        pd.DataFrame(metadata),
        counters,
        sequence_length_counts,
    )


def make_model():
    random_forest = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    extra_trees = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=43,
        n_jobs=-1,
    )

    histogram_gradient_boosting = (
        HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=44,
        )
    )

    return VotingClassifier(
        estimators=[
            ("rf", random_forest),
            ("et", extra_trees),
            (
                "hgb",
                histogram_gradient_boosting,
            ),
        ],
        voting="soft",
        weights=[1, 1, 1],
    )


def train_one(
    X,
    y,
    groups,
    target,
    horizon,
):
    result = {
        "target": target,
        "horizon_min": horizon,
        "status": None,
        "samples": int(len(y)),
    }

    if (
        len(y) < 40
        or len(np.unique(y)) < 2
        or len(np.unique(groups)) < 2
    ):
        result["status"] = (
            "insufficient_data"
        )
        return None, result

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    try:
        train_index, test_index = next(
            splitter.split(
                X,
                y,
                groups,
            )
        )
    except Exception:
        result["status"] = (
            "invalid_split"
        )
        return None, result

    if (
        len(np.unique(y[train_index])) < 2
        or len(np.unique(y[test_index])) < 2
    ):
        result["status"] = (
            "invalid_class_split"
        )
        return None, result

    model = make_model()

    model.fit(
        X[train_index],
        y[train_index],
    )

    probabilities = model.predict_proba(
        X[test_index]
    )

    positive_index = list(
        model.classes_
    ).index(1)

    positive_probability = probabilities[
        :,
        positive_index,
    ]

    predictions = (
        positive_probability >= 0.5
    ).astype(np.int8)

    result["status"] = "ok"

    result["balanced_accuracy"] = float(
        balanced_accuracy_score(
            y[test_index],
            predictions,
        )
    )

    result["auc"] = float(
        roc_auc_score(
            y[test_index],
            positive_probability,
        )
    )

    result["brier"] = float(
        brier_score_loss(
            y[test_index],
            positive_probability,
        )
    )

    result["test_samples"] = int(
        len(test_index)
    )

    return model, result


def atomic_joblib_dump(
    payload,
    destination: Path,
):
    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    joblib.dump(
        payload,
        temporary,
    )

    temporary.replace(
        destination
    )


def atomic_text_write(
    text: str,
    destination: Path,
):
    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    temporary.write_text(
        text,
        encoding="utf-8",
    )

    temporary.replace(
        destination
    )


def main():
    print(
        "=== RMA10 VISUAL/TEMPORAL AI V6 ==="
    )

    print(
        f"Version: {MODEL_VERSION}"
    )

    if not DATASET.exists():
        raise FileNotFoundError(
            DATASET
        )

    if not RADAR_DIR.exists():
        raise FileNotFoundError(
            RADAR_DIR
        )

    df = pd.read_csv(
        DATASET
    )

    required = {
        "frame_utc",
        "track_id",
        "x",
        "y",
        "horizonte_min",
    } | set(TARGETS)

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Faltan columnas: "
            f"{sorted(missing)}"
        )

    print(
        f"Dataset: {len(df)} filas | "
        f"tracks: {df['track_id'].nunique()}"
    )

    histories = build_histories(
        df
    )

    print(
        f"Historias construidas: "
        f"{len(histories)}"
    )

    image_index = build_image_index()

    (
        X,
        metadata,
        counters,
        sequence_counts,
    ) = build_sequence_samples(
        df,
        histories,
        image_index,
    )

    print(
        f"Secuencias visuales válidas: "
        f"{len(X)}"
    )

    if len(X) < 300:
        raise RuntimeError(
            f"Solo se construyeron "
            f"{len(X)} secuencias. "
            "Se aborta para no "
            "reemplazar el modelo."
        )

    if X.ndim != 2:
        raise RuntimeError(
            f"Forma de features inválida: "
            f"{X.shape}"
        )

    print(
        f"Features por secuencia: "
        f"{X.shape[1]}"
    )

    lookup = {
        (
            item["track_id"],
            item["frame_utc"],
        ): index
        for index, item
        in metadata.iterrows()
    }

    models = {}

    metrics = {
        "version": MODEL_VERSION,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset_rows": int(len(df)),
        "tracks_total": int(
            df["track_id"].nunique()
        ),
        "sequences_valid": int(
            len(X)
        ),
        "features_per_sequence": int(
            X.shape[1]
        ),
        "counters": counters,
        "sequence_length_counts":
            sequence_counts,
        "models": {},
    }

    trained = 0

    for horizon in HORIZONS:
        horizon_df = df[
            df["horizonte_min"]
            == horizon
        ]

        for target in TARGETS:
            print(
                f"Entrenando "
                f"{target} {horizon} min..."
            )

            indices = []
            labels = []

            for row in horizon_df.itertuples(
                index=False
            ):
                key = (
                    getattr(
                        row,
                        "track_id",
                    ),
                    str(
                        getattr(
                            row,
                            "frame_utc",
                        )
                    ),
                )

                index = lookup.get(
                    key
                )

                if index is None:
                    continue

                value = getattr(
                    row,
                    target,
                )

                if pd.isna(value):
                    continue

                indices.append(index)
                labels.append(
                    int(bool(value))
                )

            if not indices:
                metrics["models"][
                    f"{target}_{horizon}"
                ] = {
                    "target": target,
                    "horizon_min": horizon,
                    "status":
                        "insufficient_data",
                    "samples": 0,
                }

                continue

            index_array = np.asarray(
                indices,
                dtype=np.int64,
            )

            y = np.asarray(
                labels,
                dtype=np.int8,
            )

            groups = metadata.iloc[
                index_array
            ]["track_id"].to_numpy()

            model, result = train_one(
                X[index_array],
                y,
                groups,
                target,
                horizon,
            )

            metrics["models"][
                f"{target}_{horizon}"
            ] = result

            if model is None:
                print(
                    "NO ENTRENADO | "
                    f"status={result['status']} "
                    f"| samples={result['samples']}"
                )
                continue

            models[
                (target, horizon)
            ] = model

            trained += 1

            print(
                "OK | "
                f"samples={result['samples']} | "
                f"bal_acc="
                f"{result['balanced_accuracy']:.4f} | "
                f"AUC="
                f"{result['auc']:.4f}"
            )

    if trained != 12:
        raise RuntimeError(
            f"Solo {trained}/12 modelos entrenados. "
            "Se aborta para no reemplazar el modelo."
        )

    payload = {
        "version": MODEL_VERSION,
        "models": models,
        "feature_config": {
            "sequence_len": SEQUENCE_LEN,
            "min_sequence_len":
                MIN_SEQUENCE_LEN,
            "max_sequence_window_min":
                MAX_SEQUENCE_WINDOW_MIN,
            "max_image_gap_min":
                MAX_IMAGE_GAP_MIN,
            "patch_size": PATCH_SIZE,
            "downsample_sizes":
                DOWNSAMPLE_SIZES,
            "feature_count": int(
                X.shape[1]
            ),
            "dbz_min": DBZ_MIN,
            "dbz_max": DBZ_MAX,
            "palette_start":
                PALETTE_START,
            "palette_end":
                PALETTE_END,
        },
        "targets": list(TARGETS),
        "horizons": list(HORIZONS),
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    atomic_joblib_dump(
        payload,
        MODEL_PATH,
    )

    atomic_text_write(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        METRICS_PATH,
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
        f"{trained}/12"
    )

    print(
        f"Features: {X.shape[1]}"
    )


if __name__ == "__main__":
    main()
