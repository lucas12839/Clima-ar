#!/usr/bin/env python3
"""
RMA10 VISUAL/TEMPORAL AI - V3

Entrena directamente sobre secuencias temporales de imágenes RMA10.

Objetivos:
- aprender evolución visual del núcleo;
- usar hasta 6 imágenes temporales reales;
- aceptar secuencias de 3 a 6 observaciones;
- predecir fortalecimiento, debilitamiento y movimiento hacia Bahía Blanca;
- horizontes 5/10/20/30 min;
- separar train/test por track_id;
- no modificar Storm Engine, GOES ni fusion_multimodal.

La secuencia usa:
  - imágenes absolutas en dBZ;
  - diferencias temporales;
  - estadísticas por frame;
  - evolución de media y p95.

Las secuencias incompletas se completan repitiendo
la primera imagen REAL disponible. No se inventan imágenes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import re

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

MODEL_VERSION = "rma10-visual-temporal-v3"

SEQUENCE_LEN = 6
MIN_SEQUENCE_LEN = 3

MAX_GAP_MIN = 20.0
MAX_IMAGE_MATCH_MIN = 10.0

PATCH_SIZE = 64
DOWNSAMPLE_SIZE = 16

HORIZONS = [5, 10, 20, 30]
TARGETS = [
    "strengthened",
    "weakened",
    "toward_bahia",
]

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# RMA10 dBZ palette
# ---------------------------------------------------------------------------

DBZ_MIN = -15.0
DBZ_MAX = 75.0

PALETTE_START = 64
PALETTE_END = 207


def decode_rma10_dbz(image: Image.Image) -> np.ndarray:
    """
    Decodifica la paleta operacional RMA10 a dBZ.

    Los índices de color válidos del producto RMA10 están
    entre 64 y 207.
    """
    img = image.convert("P")

    palette = img.getpalette()
    if palette is None:
        raise ValueError("Imagen RMA10 sin paleta")

    indices = np.asarray(img, dtype=np.uint8)

    dbz = np.full(indices.shape, np.nan, dtype=np.float32)

    valid = (
        (indices >= PALETTE_START)
        & (indices <= PALETTE_END)
    )

    if np.any(valid):
        values = indices[valid].astype(np.float32)

        dbz_values = (
            DBZ_MIN
            + (
                (values - PALETTE_START)
                / float(PALETTE_END - PALETTE_START)
            )
            * (DBZ_MAX - DBZ_MIN)
        )

        dbz[valid] = dbz_values

    return dbz


# ---------------------------------------------------------------------------
# Timestamp handling
# ---------------------------------------------------------------------------

_TIMESTAMP_RE = re.compile(r"(?<!\d)(\d{14})(?!\d)")


def parse_timestamp_from_name(path: Path) -> datetime | None:
    """
    Extrae YYYYMMDDHHMMSS desde cualquier parte del nombre.
    """
    match = _TIMESTAMP_RE.search(path.name)

    if not match:
        return None

    try:
        dt = datetime.strptime(
            match.group(1),
            "%Y%m%d%H%M%S",
        )

        return dt.replace(tzinfo=timezone.utc)

    except ValueError:
        return None


def parse_dataset_timestamp(value) -> datetime | None:
    if value is None:
        return None

    try:
        text = str(value).strip()

        dt = pd.to_datetime(
            text,
            utc=True,
            errors="coerce",
        )

        if pd.isna(dt):
            return None

        return dt.to_pydatetime()

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Image index
# ---------------------------------------------------------------------------

def build_image_index():
    """
    Indexa TODAS las imágenes PNG del histórico.

    Se usa rglob para no depender de que el histórico esté
    necesariamente en un único directorio plano.
    """

    index = []

    for path in RADAR_DIR.rglob("*.png"):
        timestamp = parse_timestamp_from_name(path)

        if timestamp is None:
            continue

        index.append(
            (
                timestamp,
                path,
            )
        )

    index.sort(key=lambda item: item[0])

    print(f"Imágenes indexadas: {len(index)}")

    if index:
        print(
            "Rango imágenes:",
            index[0][0].isoformat(),
            "->",
            index[-1][0].isoformat(),
        )

    return index


def resolve_image(
    target_timestamp: datetime,
    image_index,
):
    """
    Busca la imagen temporalmente más cercana.

    Solo acepta una imagen si está dentro de
    MAX_IMAGE_MATCH_MIN.
    """

    if not image_index:
        return None

    target_ts = target_timestamp.timestamp()

    best_path = None
    best_delta = float("inf")

    # El corpus histórico no es gigantesco; mantenerlo simple
    # evita dependencias adicionales.
    for timestamp, path in image_index:
        delta = abs(timestamp.timestamp() - target_ts)

        if delta < best_delta:
            best_delta = delta
            best_path = path

    if best_path is None:
        return None

    if best_delta > MAX_IMAGE_MATCH_MIN * 60.0:
        return None

    return best_path


# ---------------------------------------------------------------------------
# Cached radar decoding
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def load_dbz(path_string: str) -> np.ndarray:
    path = Path(path_string)

    with Image.open(path) as image:
        return decode_rma10_dbz(image)


# ---------------------------------------------------------------------------
# Spatial extraction
# ---------------------------------------------------------------------------

def extract_patch(
    dbz: np.ndarray,
    x: float,
    y: float,
) -> np.ndarray:
    """
    Extrae un parche centrado en x/y y lo reduce a 16x16.
    """

    height, width = dbz.shape

    cx = int(round(float(x)))
    cy = int(round(float(y)))

    half = PATCH_SIZE // 2

    x0 = cx - half
    y0 = cy - half
    x1 = x0 + PATCH_SIZE
    y1 = y0 + PATCH_SIZE

    patch = np.full(
        (PATCH_SIZE, PATCH_SIZE),
        np.nan,
        dtype=np.float32,
    )

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(width, x1)
    src_y1 = min(height, y1)

    if src_x1 > src_x0 and src_y1 > src_y0:

        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        patch[
            dst_y0:dst_y1,
            dst_x0:dst_x1,
        ] = dbz[
            src_y0:src_y1,
            src_x0:src_x1,
        ]

    # NaN -> valor neutro para el modelo.
    finite = np.isfinite(patch)

    if np.any(finite):
        fill = float(np.nanmedian(patch))
    else:
        fill = DBZ_MIN

    patch[~finite] = fill

    # Reducción 64 -> 16.
    small = Image.fromarray(
        patch.astype(np.float32),
        mode="F",
    ).resize(
        (DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE),
        Image.Resampling.BILINEAR,
    )

    return np.asarray(
        small,
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def frame_stats(frame: np.ndarray) -> np.ndarray:
    values = frame.astype(np.float32).ravel()

    return np.asarray(
        [
            float(np.mean(values)),
            float(np.std(values)),
            float(np.min(values)),
            float(np.max(values)),
            float(np.percentile(values, 50)),
            float(np.percentile(values, 75)),
            float(np.percentile(values, 90)),
            float(np.percentile(values, 95)),
            float(np.percentile(values, 99)),
            float(np.mean(values > 40.0)),
        ],
        dtype=np.float32,
    )


def feature_vector(frames: list[np.ndarray]) -> np.ndarray:
    """
    Vector:
      6 frames absolutas       = 1536
      5 diferencias            = 1280
      estadísticas 6 frames   =   60
      evolución media/p95     =   12
      --------------------------------
                                2888
    """

    absolute = [
        frame.ravel()
        for frame in frames
    ]

    diffs = [
        frames[i] - frames[i - 1]
        for i in range(1, len(frames))
    ]

    stats = [
        frame_stats(frame)
        for frame in frames
    ]

    means = np.asarray(
        [
            float(np.mean(frame))
            for frame in frames
        ],
        dtype=np.float32,
    )

    p95 = np.asarray(
        [
            float(np.percentile(frame, 95))
            for frame in frames
        ],
        dtype=np.float32,
    )

    mean_diff = np.diff(means)
    p95_diff = np.diff(p95)

    vector = np.concatenate(
        [
            np.concatenate(absolute),
            np.concatenate(
                [
                    diff.ravel()
                    for diff in diffs
                ]
            ),
            np.concatenate(stats),
            mean_diff,
            p95_diff,
        ]
    )

    return vector.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset histories
# ---------------------------------------------------------------------------

def build_histories(df: pd.DataFrame):
    """
    Construye una historia única por track.

    Se eliminan duplicados de track/frame porque el CSV
    contiene varias filas por horizonte.
    """

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
    ).copy()

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
                {
                    "timestamp": timestamp,
                    "x": x,
                    "y": y,
                }
            )

        observations.sort(
            key=lambda item: item["timestamp"]
        )

        if observations:
            histories[track_id] = observations

    return histories


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------

def build_sequence_samples(
    df: pd.DataFrame,
    histories,
    image_index,
):
    X = []
    metadata = []

    counters = {
        "rows": 0,
        "no_history": 0,
        "short_history": 0,
        "bad_gap": 0,
        "missing_images": 0,
        "valid": 0,
    }

    for row in df.itertuples(index=False):

        counters["rows"] += 1

        track_id = getattr(row, "track_id")

        history = histories.get(track_id)

        if not history:
            counters["no_history"] += 1
            continue

        current_timestamp = parse_dataset_timestamp(
            getattr(row, "frame_utc")
        )

        if current_timestamp is None:
            counters["no_history"] += 1
            continue

        # Todas las observaciones hasta el frame actual.
        previous = [
            item
            for item in history
            if item["timestamp"] <= current_timestamp
        ]

        if len(previous) < MIN_SEQUENCE_LEN:
            counters["short_history"] += 1
            continue

        # Usamos como máximo las últimas 6 observaciones reales.
        sequence = previous[-SEQUENCE_LEN:]

        # Validación temporal.
        bad_gap = False

        for i in range(1, len(sequence)):
            gap = (
                sequence[i]["timestamp"]
                - sequence[i - 1]["timestamp"]
            ).total_seconds() / 60.0

            if gap <= 0 or gap > MAX_GAP_MIN:
                bad_gap = True
                break

        if bad_gap:
            counters["bad_gap"] += 1
            continue

        # Resolver imágenes.
        resolved = []

        for observation in sequence:

            path = resolve_image(
                observation["timestamp"],
                image_index,
            )

            if path is None:
                resolved = []
                break

            resolved.append(
                (
                    observation,
                    path,
                )
            )

        if not resolved:
            counters["missing_images"] += 1
            continue

        # Si tenemos menos de 6 frames, repetimos el primer
        # frame REAL para mantener una entrada de tamaño fijo.
        while len(resolved) < SEQUENCE_LEN:
            resolved.insert(
                0,
                resolved[0],
            )

        frames = []

        try:
            for observation, path in resolved:

                dbz = load_dbz(
                    str(path.resolve())
                )

                frame = extract_patch(
                    dbz,
                    observation["x"],
                    observation["y"],
                )

                frames.append(frame)

        except Exception:
            counters["missing_images"] += 1
            continue

        if len(frames) != SEQUENCE_LEN:
            counters["missing_images"] += 1
            continue

        X.append(
            feature_vector(frames)
        )

        metadata.append(
            {
                "track_id": track_id,
                "frame_utc": str(
                    getattr(row, "frame_utc")
                ),
            }
        )

        counters["valid"] += 1

    print(
        "rows:",
        counters["rows"],
    )
    print(
        "no_history:",
        counters["no_history"],
    )
    print(
        "short_history:",
        counters["short_history"],
    )
    print(
        "bad_gap:",
        counters["bad_gap"],
    )
    print(
        "missing_images:",
        counters["missing_images"],
    )
    print(
        "valid:",
        counters["valid"],
    )

    return (
        np.asarray(X, dtype=np.float32),
        pd.DataFrame(metadata),
        counters,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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

    if len(y) < 6:
        result["status"] = "insufficient_data"
        return None, result

    unique_classes = np.unique(y)

    if len(unique_classes) < 2:
        result["status"] = "insufficient_classes"
        return None, result

    unique_groups = np.unique(groups)

    if len(unique_groups) < 2:
        result["status"] = "insufficient_groups"
        return None, result

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    try:
        train_idx, test_idx = next(
            splitter.split(
                X,
                y,
                groups,
            )
        )
    except Exception:
        result["status"] = "invalid_split"
        return None, result

    if (
        len(train_idx) == 0
        or len(test_idx) == 0
    ):
        result["status"] = "invalid_split"
        return None, result

    if len(np.unique(y[train_idx])) < 2:
        result["status"] = "invalid_train_split"
        return None, result

    if len(np.unique(y[test_idx])) < 1:
        result["status"] = "invalid_test_split"
        return None, result

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X[train_idx],
        y[train_idx],
    )

    predictions = model.predict(
        X[test_idx]
    )

    probabilities = model.predict_proba(
        X[test_idx]
    )

    classes = model.classes_

    positive_probability = None

    if len(classes) == 2:
        positive_index = list(classes).index(1)

        positive_probability = probabilities[
            :,
            positive_index,
        ]

    bal_acc = balanced_accuracy_score(
        y[test_idx],
        predictions,
    )

    result["status"] = "ok"
    result["balanced_accuracy"] = float(
        bal_acc
    )

    if positive_probability is not None:

        try:
            if len(np.unique(y[test_idx])) == 2:
                result["auc"] = float(
                    roc_auc_score(
                        y[test_idx],
                        positive_probability,
                    )
                )

                result["brier"] = float(
                    brier_score_loss(
                        y[test_idx],
                        positive_probability,
                    )
                )
            else:
                result["auc"] = None
                result["brier"] = None

        except Exception:
            result["auc"] = None
            result["brier"] = None

    else:
        result["auc"] = None
        result["brier"] = None

    return model, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print("=== RMA10 VISUAL/TEMPORAL AI ===")
    print(f"Version: {MODEL_VERSION}")

    if not DATASET.exists():
        raise FileNotFoundError(
            f"No existe dataset: {DATASET}"
        )

    if not RADAR_DIR.exists():
        raise FileNotFoundError(
            f"No existe histórico radar: {RADAR_DIR}"
        )

    df = pd.read_csv(DATASET)

    print(
        f"Dataset: {DATASET}"
    )

    print(
        f"Radar:   {RADAR_DIR}"
    )

    print(
        f"Filas dataset: {len(df)} | "
        f"tracks: {df['track_id'].nunique()}"
    )

    histories = build_histories(df)

    print(
        f"Historias construidas: {len(histories)}"
    )

    image_index = build_image_index()

    if not image_index:
        raise RuntimeError(
            "No se encontraron imágenes RMA10 con timestamp."
        )

    X, metadata, counters = (
        build_sequence_samples(
            df,
            histories,
            image_index,
        )
    )

    print(
        f"Secuencias visuales válidas: {len(X)}"
    )

    if len(X) == 0:
        raise RuntimeError(
            "No hay secuencias visuales válidas."
        )

    print(
        f"Features por secuencia: {X.shape[1]}"
    )

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
        "sequences_valid": int(len(X)),
        "features_per_sequence": int(
            X.shape[1]
        ),
        "counters": counters,
        "models": {},
    }

    trained = 0

    for horizon in HORIZONS:

        horizon_df = df[
            df["horizonte_min"] == horizon
        ].copy()

        if horizon_df.empty:
            for target in TARGETS:
                print(
                    f"Entrenando {target} "
                    f"{horizon} min..."
                )

                metrics["models"][
                    f"{target}_{horizon}"
                ] = {
                    "target": target,
                    "horizon_min": horizon,
                    "status": "no_horizon_data",
                }

            continue

        # Metadata identifica cada secuencia por track/frame.
        lookup = {}

        for idx, item in metadata.iterrows():
            lookup[
                (
                    item["track_id"],
                    str(item["frame_utc"]),
                )
            ] = idx

        for target in TARGETS:

            print(
                f"Entrenando {target} "
                f"{horizon} min..."
            )

            indices = []
            labels = []

            for row in horizon_df.itertuples(
                index=False
            ):

                key = (
                    getattr(row, "track_id"),
                    str(
                        getattr(
                            row,
                            "frame_utc",
                        )
                    ),
                )

                idx = lookup.get(key)

                if idx is None:
                    continue

                value = getattr(
                    row,
                    target,
                )

                if pd.isna(value):
                    continue

                indices.append(idx)
                labels.append(
                    int(bool(value))
                )

            if not indices:
                result = {
                    "target": target,
                    "horizon_min": horizon,
                    "status": "insufficient_data",
                    "samples": 0,
                }

                print(
                    "NO ENTRENADO | "
                    "status=insufficient_data"
                )

                metrics["models"][
                    f"{target}_{horizon}"
                ] = result

                continue

            indices = np.asarray(
                indices,
                dtype=np.int64,
            )

            y = np.asarray(
                labels,
                dtype=np.int8,
            )

            X_target = X[indices]

            groups = metadata.iloc[
                indices
            ]["track_id"].to_numpy()

            model, result = train_one(
                X_target,
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
                    f"status={result['status']}"
                )

                continue

            models[
                (
                    target,
                    horizon,
                )
            ] = model

            trained += 1

            print(
                "OK | "
                f"samples={result['samples']} | "
                f"bal_acc={result.get('balanced_accuracy')} | "
                f"AUC={result.get('auc')}"
            )

    payload = {
        "version": MODEL_VERSION,
        "models": models,
        "feature_config": {
            "sequence_len": SEQUENCE_LEN,
            "min_sequence_len": MIN_SEQUENCE_LEN,
            "patch_size": PATCH_SIZE,
            "downsample_size": DOWNSAMPLE_SIZE,
            "max_gap_min": MAX_GAP_MIN,
            "max_image_match_min": MAX_IMAGE_MATCH_MIN,
            "dbz_min": DBZ_MIN,
            "dbz_max": DBZ_MAX,
        },
        "targets": TARGETS,
        "horizons": HORIZONS,
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        payload,
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

    print("=== FINALIZADO ===")
    print(
        f"Modelo:   {MODEL_PATH}"
    )
    print(
        f"Métricas: {METRICS_PATH}"
    )
    print(
        f"Modelos entrenados: "
        f"{trained}/12"
    )


if __name__ == "__main__":
    main()
