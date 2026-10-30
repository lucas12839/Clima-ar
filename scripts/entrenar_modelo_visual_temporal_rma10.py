#!/usr/bin/env python3
"""
Entrenamiento visual/temporal RMA10.

Objetivo:
- aprender directamente de secuencias temporales RMA10;
- seguir el mismo track/núcleo;
- usar ventanas deslizantes de hasta 6 imágenes;
- aprovechar tracks con 3, 4, 5 o 6+ observaciones;
- predecir fortalecimiento, debilitamiento y movimiento hacia Bahía Blanca;
- entrenar por horizonte 5/10/20/30 min;
- separar train/test por track_id para evitar fuga temporal.

No modifica Storm Engine, GOES ni fusion_multimodal.
"""

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
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data/radar/rma10_dataset_historico.csv"
RADAR_DIR = ROOT / "data/radar/historico"

MODEL_PATH = ROOT / "data/models/rma10_visual_temporal_model.joblib"
METRICS_PATH = ROOT / "data/models/rma10_visual_temporal_model_metrics.json"

SEQUENCE_LEN = 6
MIN_SEQUENCE_LEN = 3

PATCH_SIZE = 64
DOWNSAMPLE = 4

MAX_GAP_MIN = 20

DBZ_MIN = -15.0
DBZ_MAX = 75.0

PALETTE_FIRST = 64
PALETTE_LAST = 207


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


def validate_rma10_palette(im: Image.Image) -> None:
    if im.mode != "P":
        raise ValueError(
            f"Imagen RMA10 inválida: modo {im.mode}, se esperaba P"
        )

    palette = im.getpalette()

    if palette is None or len(palette) < 208 * 3:
        raise ValueError("Paleta RMA10 ausente o incompleta")


@lru_cache(maxsize=16)
def load_rma10_field(path_string: str) -> np.ndarray:
    path = Path(path_string)

    with Image.open(path) as im:
        validate_rma10_palette(im)

        indices = np.asarray(im, dtype=np.uint8)

        transparency = im.info.get("transparency")

        if isinstance(transparency, bytes):
            transparency = transparency[0] if transparency else None

        field = np.zeros(
            indices.shape,
            dtype=np.float32,
        )

        mask = (
            (indices >= PALETTE_FIRST)
            & (indices <= PALETTE_LAST)
        )

        if transparency is not None:
            mask &= indices != int(transparency)

        field[mask] = (
            DBZ_MIN
            + (
                (
                    indices[mask].astype(np.float32)
                    - PALETTE_FIRST
                )
                / (PALETTE_LAST - PALETTE_FIRST)
            )
            * (DBZ_MAX - DBZ_MIN)
        )

        field /= DBZ_MAX - DBZ_MIN

        return np.clip(
            field,
            0.0,
            1.0,
        )


def crop_resize(
    field: np.ndarray,
    x: float,
    y: float,
) -> np.ndarray:

    h, w = field.shape

    half = PATCH_SIZE // 2

    cx = int(round(float(x)))
    cy = int(round(float(y)))

    x0 = cx - half
    y0 = cy - half

    x1 = x0 + PATCH_SIZE
    y1 = y0 + PATCH_SIZE

    patch = np.zeros(
        (PATCH_SIZE, PATCH_SIZE),
        dtype=np.float32,
    )

    sx0 = max(0, x0)
    sy0 = max(0, y0)

    sx1 = min(w, x1)
    sy1 = min(h, y1)

    if sx1 > sx0 and sy1 > sy0:

        dx0 = sx0 - x0
        dy0 = sy0 - y0

        dx1 = dx0 + (sx1 - sx0)
        dy1 = dy0 + (sy1 - sy0)

        patch[
            dy0:dy1,
            dx0:dx1,
        ] = field[
            sy0:sy1,
            sx0:sx1,
        ]

    if DOWNSAMPLE > 1:

        small = Image.fromarray(
            patch,
            mode="F",
        ).resize(
            (
                PATCH_SIZE // DOWNSAMPLE,
                PATCH_SIZE // DOWNSAMPLE,
            ),
            resample=Image.Resampling.BILINEAR,
        )

        patch = np.asarray(
            small,
            dtype=np.float32,
        )

    return patch


def feature_vector(
    patches: list[np.ndarray],
) -> np.ndarray:

    arr = np.stack(patches).astype(
        np.float32
    )

    features = []

    # Campo visual absoluto.
    for frame in arr:
        features.append(
            frame.ravel()
        )

    # Evolución frame a frame.
    for i in range(1, len(arr)):
        features.append(
            (arr[i] - arr[i - 1]).ravel()
        )

    # Estadísticas de cada frame.
    for frame in arr:

        features.extend(
            [
                np.mean(frame),
                np.std(frame),
                np.percentile(frame, 75),
                np.percentile(frame, 90),
                np.percentile(frame, 95),
                np.percentile(frame, 99),
                np.max(frame),
                np.count_nonzero(
                    frame > 0.55
                ) / frame.size,
                np.count_nonzero(
                    frame > 0.70
                ) / frame.size,
                np.count_nonzero(
                    frame > 0.85
                ) / frame.size,
            ]
        )

    means = np.array(
        [
            np.mean(frame)
            for frame in arr
        ],
        dtype=np.float32,
    )

    p95s = np.array(
        [
            np.percentile(frame, 95)
            for frame in arr
        ],
        dtype=np.float32,
    )

    features.extend(
        [
            np.diff(means),
            np.diff(p95s),
            means[-1] - means[0],
            p95s[-1] - p95s[0],
        ]
    )

    return np.concatenate(
        [
            np.asarray(
                value,
                dtype=np.float32,
            ).ravel()
            for value in features
        ]
    )


def load_dataset() -> pd.DataFrame:

    if not DATASET.exists():
        raise FileNotFoundError(
            f"No existe dataset: {DATASET}"
        )

    df = pd.read_csv(DATASET)

    required = {
        "track_id",
        "frame_utc",
        "x",
        "y",
        "horizonte_min",
        "strengthened",
        "weakened",
        "toward_bahia",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Faltan columnas del dataset: {missing}"
        )

    df = df.copy()

    df["dt"] = [
        parse_dt(v)
        for v in df["frame_utc"]
    ]

    df = df.drop_duplicates(
        subset=[
            "track_id",
            "frame_utc",
            "horizonte_min",
        ]
    )

    return df


def build_histories(
    df: pd.DataFrame,
) -> dict:

    histories = {}

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
            subset=[
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

    for track_id, group in observations.groupby(
        "track_id"
    ):

        rows = []

        for row in group.itertuples(
            index=False
        ):

            rows.append(
                {
                    "frame_utc": str(
                        row.frame_utc
                    ),
                    "dt": row.dt,
                    "x": float(row.x),
                    "y": float(row.y),
                }
            )

        histories[track_id] = rows

    return histories


def make_sequence_samples(
    df: pd.DataFrame,
    histories: dict,
) -> tuple[np.ndarray, pd.DataFrame]:

    X = []
    metadata = []

    index_by_track_dt = {}

    for track_id, history in histories.items():

        index_by_track_dt[track_id] = {
            item["dt"]: i
            for i, item in enumerate(history)
        }

    total_rows = len(df)

    skipped = 0
    valid = 0

    for n, row in enumerate(
        df.itertuples(index=False),
        start=1,
    ):

        track_id = row.track_id

        horizon = int(
            row.horizonte_min
        )

        history = histories.get(
            track_id
        )

        if not history:
            skipped += 1
            continue

        current_dt = row.dt

        pos = (
            index_by_track_dt
            .get(track_id, {})
            .get(current_dt)
        )

        if pos is None:
            skipped += 1
            continue

        # Tomamos hasta 6 frames anteriores,
        # incluyendo el actual.
        start = max(
            0,
            pos - SEQUENCE_LEN + 1,
        )

        seq = history[start : pos + 1]

        if len(seq) < MIN_SEQUENCE_LEN:
            skipped += 1
            continue

        gaps = [
            (
                seq[i]["dt"]
                - seq[i - 1]["dt"]
            ).total_seconds()
            / 60.0
            for i in range(1, len(seq))
        ]

        if any(
            gap <= 0
            or gap > MAX_GAP_MIN
            for gap in gaps
        ):
            skipped += 1
            continue

        # Si hay menos de 6 frames,
        # repetimos el primer frame para mantener
        # una representación de tamaño fijo.
        while len(seq) < SEQUENCE_LEN:
            seq.insert(
                0,
                seq[0],
            )

        patches = []

        try:

            for item in seq:

                path = (
                    RADAR_DIR
                    / frame_name(
                        item["dt"]
                    )
                )

                if not path.exists():
                    raise FileNotFoundError(
                        path
                    )

                field = load_rma10_field(
                    str(path)
                )

                patch = crop_resize(
                    field,
                    item["x"],
                    item["y"],
                )

                patches.append(patch)

            vector = feature_vector(
                patches
            )

            X.append(vector)

            metadata.append(
                {
                    "track_id": track_id,
                    "frame_utc": str(
                        row.frame_utc
                    ),
                    "horizonte_min": horizon,
                    "strengthened": int(
                        row.strengthened
                    ),
                    "weakened": int(
                        row.weakened
                    ),
                    "toward_bahia": int(
                        row.toward_bahia
                    ),
                    "sequence_frames": min(
                        pos + 1,
                        SEQUENCE_LEN,
                    ),
                }
            )

            valid += 1

        except (
            FileNotFoundError,
            ValueError,
        ):
            skipped += 1

        if n % 500 == 0:

            print(
                f"[secuencias] "
                f"{n}/{total_rows} "
                f"| válidas={valid} "
                f"| omitidas={skipped}"
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
) -> dict:

    subset = meta[
        meta["horizonte_min"]
        == horizon
    ].copy()

    if subset.empty:

        return {
            "status": "insufficient_data",
            "target": target,
            "horizonte_min": horizon,
            "samples": 0,
        }

    indices = subset.index.to_numpy()

    Xh = X[indices]

    y = (
        subset[target]
        .astype(int)
        .to_numpy()
    )

    groups = subset[
        "track_id"
    ].to_numpy()

    if len(np.unique(y)) < 2:

        return {
            "status": "insufficient_classes",
            "target": target,
            "horizonte_min": horizon,
            "samples": len(y),
            "classes": sorted(
                np.unique(y).tolist()
            ),
        }

    if len(np.unique(groups)) < 2:

        return {
            "status": "insufficient_groups",
            "target": target,
            "horizonte_min": horizon,
            "samples": len(y),
            "groups": int(
                len(np.unique(groups))
            ),
        }

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(
            Xh,
            y,
            groups=groups,
        )
    )

    y_train = y[train_idx]
    y_test = y[test_idx]

    if len(np.unique(y_train)) < 2:

        return {
            "status": "invalid_train_split",
            "target": target,
            "horizonte_min": horizon,
            "samples": len(y),
        }

    if len(np.unique(y_test)) < 2:

        return {
            "status": "invalid_test_split",
            "target": target,
            "horizonte_min": horizon,
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
        Xh[train_idx],
        y_train,
    )

    probabilities = model.predict_proba(
        Xh[test_idx]
    )

    if len(model.classes_) != 2:

        return {
            "status": "invalid_model_classes",
            "target": target,
            "horizonte_min": horizon,
            "samples": len(y),
        }

    positive_class_pos = int(
        np.where(
            model.classes_ == 1
        )[0][0]
    )

    prob = probabilities[
        :,
        positive_class_pos,
    ]

    pred = model.predict(
        Xh[test_idx]
    )

    metrics = {
        "status": "ok",
        "target": target,
        "horizonte_min": horizon,
        "samples": int(len(y)),
        "groups": int(
            len(np.unique(groups))
        ),
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
                pred,
            )
        ),
        "brier": float(
            brier_score_loss(
                y_test,
                prob,
            )
        ),
        "auc": float(
            roc_auc_score(
                y_test,
                prob,
            )
        ),
    }

    return {
        **metrics,
        "_model": model,
    }


def main() -> None:

    print(
        "=== RMA10 VISUAL/TEMPORAL AI ==="
    )

    print(
        f"Dataset: {DATASET}"
    )

    print(
        f"Radar:   {RADAR_DIR}"
    )

    df = load_dataset()

    print(
        f"Filas dataset: {len(df)} "
        f"| tracks: "
        f"{df['track_id'].nunique()}"
    )

    histories = build_histories(
        df
    )

    print(
        f"Historias construidas: "
        f"{len(histories)}"
    )

    X, meta = make_sequence_samples(
        df,
        histories,
    )

    print(
        f"Secuencias visuales válidas: "
        f"{len(X)}"
    )

    print(
        f"Features por secuencia: "
        f"{X.shape[1]}"
    )

    print(
        f"Tracks utilizados: "
        f"{meta['track_id'].nunique()}"
    )

    targets = [
        "strengthened",
        "weakened",
        "toward_bahia",
    ]

    horizons = [
        5,
        10,
        20,
        30,
    ]

    models = {}

    metrics = {
        "version":
            "rma10-visual-temporal-v2",
        "sequence_len":
            SEQUENCE_LEN,
        "min_sequence_len":
            MIN_SEQUENCE_LEN,
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

    for horizon in horizons:

        for target in targets:

            print(
                f"\nEntrenando "
                f"{target} "
                f"{horizon} min..."
            )

            result = train_one(
                X,
                meta,
                target,
                horizon,
            )

            key = (
                f"{target}_{horizon}m"
            )

            model = result.pop(
                "_model",
                None,
            )

            metrics[
                "models"
            ][key] = result

            if model is not None:

                models[key] = model

                print(
                    f"OK | "
                    f"samples="
                    f"{result['samples']} "
                    f"| groups="
                    f"{result['groups']} "
                    f"| bal_acc="
                    f"{result['balanced_accuracy']:.3f} "
                    f"| AUC="
                    f"{result['auc']:.3f}"
                )

            else:

                print(
                    f"NO ENTRENADO | "
                    f"status="
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
                "rma10-visual-temporal-v2",
            "models":
                models,
            "sequence_len":
                SEQUENCE_LEN,
            "min_sequence_len":
                MIN_SEQUENCE_LEN,
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

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n=== FINALIZADO ==="
    )

    print(
        f"Modelo:   {MODEL_PATH}"
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
