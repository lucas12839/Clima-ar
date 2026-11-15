#!/usr/bin/env python3
"""
ClimaAR — RMA10 Visual/Temporal AI — MODELO DEFINITIVO

Integra:
- Las mismas 15 variables causales que V1.
- Secuencia visual causal de hasta 6 radares, compensada por movimiento.
- Características multiescala y diferencias temporales.
- PCA ajustado exclusivamente con eventos de entrenamiento.
- Ensemble ExtraTrees + LogisticRegression.
- Calibración y evaluación cronológica contra persistencia.

No utiliza información futura y no publica probabilidades sin calibración aprobada.
El modelo operativo final se guarda en:
data/models/rma10_visual_temporal_model.joblib
"""

from __future__ import annotations

import json
import re
import sys
import hashlib
from bisect import bisect_right
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.decodificador_rma10_dbz import validar_paleta
from scripts.rma10_features import V1_FEATURES
from scripts.rma10_validation import (
    classification_metrics,
    chronological_event_split,
    fit_platt_calibration,
    persistence_baseline,
    rolling_origin_event_splits,
)

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data/radar/rma10_dataset_historico.csv"
RADAR_DIR = ROOT / "data/radar/historico"

MODEL_DIR = ROOT / "data/models"
MODEL_PATH = MODEL_DIR / "rma10_visual_temporal_model.joblib"
METRICS_PATH = MODEL_DIR / "rma10_visual_temporal_model_metrics.json"

MODEL_VERSION = "rma10-visual-temporal-causal-4-fold-pca"

HORIZONS = (5, 10, 20, 30)

TARGETS = (
    "strengthened",
    "weakened",
    "toward_bahia",
)

V1_FEATURES = tuple(V1_FEATURES)

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

    # Reject syntactically valid but impossible legacy radar years (e.g. 1791).
    if not 2000 <= dt.year <= 2100:
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
            with Image.open(path) as image:
                validar_paleta(image)
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
        validar_paleta(image)
        transparency = image.info.get("transparency")
        indices = np.asarray(image, dtype=np.uint8)

    dbz = np.full(
        indices.shape,
        np.nan,
        dtype=np.float32,
    )

    valid = (
        (indices >= PALETTE_START)
        & (indices <= PALETTE_END)
        & (indices != transparency)
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


def feature_vector(frames, valid_mask, elapsed, gaps):
    absolute = []
    stats = []
    shapes = []
    differences = []

    for index in range(SEQUENCE_LEN):
        if valid_mask[index]:
            frame = frames["16"][index]
            absolute.append(frame.ravel())
            stats.append(frame_stats(frame))
            shapes.append(shape_features(frame))
        else:
            absolute.append(np.zeros(16 * 16, dtype=np.float32))
            stats.append(np.zeros(12, dtype=np.float32))
            shapes.append(np.zeros(24, dtype=np.float32))

    for index in range(1, SEQUENCE_LEN):
        if valid_mask[index] and valid_mask[index - 1]:
            differences.append((frames["16"][index] - frames["16"][index - 1]).ravel())
        else:
            differences.append(np.zeros(16 * 16, dtype=np.float32))

    return np.concatenate([
        np.concatenate(absolute),
        np.concatenate(differences),
        np.concatenate(stats),
        np.concatenate(shapes),
        np.asarray(valid_mask, dtype=np.float32),
        np.asarray(elapsed, dtype=np.float32),
        np.asarray(gaps, dtype=np.float32),
    ]).astype(np.float32)

def build_histories(df):
    histories = {}
    columns = [
        "track_id", "frame_utc", "x", "y", "intensity",
        "area_px", "width_px", "height_px",
    ]
    clean = df.drop_duplicates(subset=["track_id", "frame_utc"])
    for track_id, group in clean.groupby("track_id", sort=False):
        observations = []
        for row in group[columns].itertuples(index=False, name=None):
            track_value, frame_utc, x, y, intensity, area, width, height = row
            timestamp = parse_timestamp(frame_utc)
            if timestamp is None:
                continue
            try:
                values = [float(x), float(y), float(intensity), float(area), float(width), float(height)]
            except (TypeError, ValueError):
                continue
            if not np.isfinite(values).all():
                continue
            observations.append({
                "timestamp": timestamp,
                "x": values[0],
                "y": values[1],
                "intensidad": values[2],
                "pixeles": values[3],
                "ancho": values[4],
                "alto": values[5],
            })
        observations.sort(key=lambda item: item["timestamp"])
        if observations:
            histories[canonical_track_id(track_id)] = observations
    return histories


def causal_position(history, timestamp):
    if not history:
        return None
    timestamps = [item["timestamp"] for item in history]
    index = bisect_right(timestamps, timestamp) - 1
    if index < 0:
        return None
    current = history[index]
    if index == 0:
        return current["x"], current["y"]
    previous = history[index - 1]
    dt = (current["timestamp"] - previous["timestamp"]).total_seconds() / 60.0
    if dt <= 0 or dt > MAX_IMAGE_GAP_MIN:
        return current["x"], current["y"]
    elapsed = (timestamp - current["timestamp"]).total_seconds() / 60.0
    elapsed = max(0.0, min(elapsed, MAX_IMAGE_GAP_MIN))
    vx = (current["x"] - previous["x"]) / dt
    vy = (current["y"] - previous["y"]) / dt
    return current["x"] + vx * elapsed, current["y"] + vy * elapsed


def find_visual_sequence(timestamp, history, image_index, image_times=None):
    """Motion-compensated, causal image sequence centered on each past track position."""
    if not history or not image_index:
        return None
    if image_times is None:
        image_times = [item[0] for item in image_index]
    index = bisect_right(image_times, timestamp) - 1
    if index < 0:
        return None

    selected = []
    later_time = None
    current_index = index
    while current_index >= 0 and len(selected) < SEQUENCE_LEN:
        frame_time, path = image_index[current_index]
        age = (timestamp - frame_time).total_seconds() / 60.0
        if age < 0:
            current_index -= 1
            continue
        if age > MAX_SEQUENCE_WINDOW_MIN:
            break
        position = causal_position(history, frame_time)
        if position is not None:
            if later_time is not None:
                gap = (later_time - frame_time).total_seconds() / 60.0
                if gap > MAX_IMAGE_GAP_MIN:
                    break
            selected.append((frame_time, path, position))
            later_time = frame_time
        current_index -= 1

    selected.reverse()
    if len(selected) < MIN_SEQUENCE_LEN:
        return None
    if (timestamp - selected[-1][0]).total_seconds() / 60.0 > 15.0:
        return None

    frames16, elapsed, gaps = [], [], []
    last_time = None
    for frame_time, path, position in selected:
        dbz = load_dbz(str(path))
        patch = extract_patch(dbz, position[0], position[1], PATCH_SIZE)
        frames16.append(resize_field(patch, 16))
        elapsed.append((timestamp - frame_time).total_seconds() / 60.0)
        gaps.append(0.0 if last_time is None else (frame_time - last_time).total_seconds() / 60.0)
        last_time = frame_time

    valid_count = len(frames16)
    while len(frames16) < SEQUENCE_LEN:
        frames16.insert(0, np.zeros((16, 16), dtype=np.float32))
        elapsed.insert(0, 0.0)
        gaps.insert(0, 0.0)

    valid_mask = [0.0] * (SEQUENCE_LEN - valid_count) + [1.0] * valid_count
    return {"16": frames16}, valid_mask, elapsed, gaps, selected

def iter_visual_features(df, histories, image_index, allowed_indices=None, image_times=None):
    """Compute one sequence per unique track/frame and share it across horizons."""
    if image_times is None:
        image_times = [item[0] for item in image_index]
    groups = df.groupby(["track_id", "frame_utc"], sort=False).indices
    for (_track_key, _frame_key), indices in groups.items():
        row_indices = [int(index) for index in indices]
        if allowed_indices is not None:
            row_indices = [index for index in row_indices if index in allowed_indices]
            if not row_indices:
                continue
        row = df.iloc[row_indices[0]]
        track_id = canonical_track_id(row["track_id"])
        timestamp = parse_timestamp(row["frame_utc"])
        if timestamp is None:
            continue
        history = histories.get(track_id)
        if not history:
            continue
        sequence = find_visual_sequence(timestamp, history, image_index, image_times)
        if sequence is None:
            continue
        frames, mask, elapsed, gaps, _selected = sequence
        yield row_indices, feature_vector(frames, mask, elapsed, gaps)

def build_fold_feature_matrix(df, visual_groups, raw_matrix, train_mask):
    """Build a fold-specific feature matrix with PCA fitted on fold-train groups only."""
    train_mask = np.asarray(train_mask, dtype=bool)
    group_train = []
    group_states = []
    for row_indices in visual_groups:
        states = np.unique(train_mask[np.asarray(row_indices, dtype=int)])
        if states.size != 1:
            raise RuntimeError(
                "Un grupo visual cruza el límite temporal del fold; "
                "no se permite mezclar eventos entre particiones."
            )
        group_train.append(bool(states[0]))
        group_states.append(states[0])

    group_train = np.asarray(group_train, dtype=bool)
    train_count = int(group_train.sum())
    visual_dim = int(raw_matrix.shape[1])
    if train_count < MIN_SEQUENCE_LEN * 20:
        raise RuntimeError(
            f"Secuencias visuales insuficientes para PCA fold-local: {train_count}"
        )

    n_components = min(PCA_COMPONENTS, train_count, visual_dim)
    fold_pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=RANDOM_STATE,
    )
    fold_pca.fit(raw_matrix[group_train])
    transformed = fold_pca.transform(raw_matrix).astype(np.float32)

    matrix = np.zeros(
        (len(df), len(V1_FEATURES) + n_components + 1),
        dtype=np.float32,
    )
    matrix[:, :len(V1_FEATURES)] = (
        df[list(V1_FEATURES)].astype(float).to_numpy(dtype=np.float32)
    )
    for group_index, row_indices in enumerate(visual_groups):
        for row_index in row_indices:
            matrix[row_index, len(V1_FEATURES):len(V1_FEATURES) + n_components] = transformed[group_index]
            matrix[row_index, -1] = 1.0
    return matrix, fold_pca


def make_ensemble(n_estimators=60, include_logistic=True):
    ensemble = [
        (
            "extra",
            ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=14,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            1.0 if not include_logistic else 0.70,
        ),
    ]
    if include_logistic:
        ensemble.append((
            "logistic",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=1500,
                    C=0.7,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )),
            ]),
            0.30,
        ))
    return ensemble

def ensemble_probability(trained_models, matrix):
    weighted = []
    total_weight = 0.0
    for name, model, weight in trained_models:
        classes = list(model.classes_)
        if 1 in classes:
            values = model.predict_proba(matrix)[:, classes.index(1)]
        else:
            values = np.zeros(len(matrix), dtype=float)
        weighted.append(values * weight)
        total_weight += weight
    return np.sum(weighted, axis=0) / max(total_weight, 1e-9)


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    print("=== CLIMAAR RMA10 — VISUAL/TEMPORAL CAUSAL 3 ===")
    if not DATASET.exists() or not RADAR_DIR.exists():
        raise SystemExit("Falta dataset o histórico de imágenes RMA10")
    summary_path = ROOT / "data/radar/rma10_dataset_resumen.json"
    if not summary_path.exists():
        raise SystemExit("Falta resumen del dataset RMA10")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "rma10-causal-v2" or summary.get("migration_version") != "rma10-causal-migration-2":
        raise SystemExit("Dataset sin migración causal v2; migrar antes de entrenar")
    if summary.get("features_causales") != list(V1_FEATURES):
        raise SystemExit("Features causales del dataset no coinciden con el modelo visual")
    if float(summary.get("radio_operativo_km", 0)) != 240.0:
        raise SystemExit("El radio operativo RMA10 debe permanecer en 240 km")

    df = pd.read_csv(DATASET)
    required = set(V1_FEATURES) | set(TARGETS) | {
        "track_id", "horizonte_min", "frame_utc", "x", "y", "label_valid"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Faltan columnas obligatorias: {missing}")
    df["track_id"] = df["track_id"].map(canonical_track_id)
    df["horizonte_min"] = pd.to_numeric(df["horizonte_min"], errors="coerce")
    df = df[df["label_valid"].astype(bool)].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["track_id", "horizonte_min", "frame_utc", *V1_FEATURES, *TARGETS])
    df = df[df["track_id"] != ""].copy()
    df = df.sort_values(["frame_utc", "track_id", "horizonte_min"]).reset_index(drop=True)
    df["horizonte_min"] = df["horizonte_min"].astype(int)

    split = chronological_event_split(df)
    train_indices = set(np.flatnonzero(split["train_mask"]).tolist())
    calibration_indices = set(np.flatnonzero(split["calibration_mask"]).tolist())
    test_indices = set(np.flatnonzero(split["test_mask"]).tolist())
    split_summary = {
        key: value for key, value in split.items()
        if key not in {"train_mask", "calibration_mask", "test_mask", "event_by_track"}
    }
    rolling_folds = rolling_origin_event_splits(df, n_last_events=4, min_rows=100)
    if len(rolling_folds) < 2:
        raise RuntimeError(f"Se requieren 2 folds rolling-origin; encontrados={len(rolling_folds)}")
    print(f"Eventos: {split['event_count']} | train={split['train_rows']} | calibration={split['calibration_rows']} | test={split['test_rows']} | rolling={len(rolling_folds)}")

    image_index = build_image_index()
    image_times = [item[0] for item in image_index]
    histories = build_histories(df)
    print(f"Historias: {len(histories)} | imágenes válidas: {len(image_index)}")

    # Una sola extracción por track/frame; las cuatro filas de horizonte
    # comparten exactamente la misma secuencia visual.
    visual_groups = []
    raw_features = []
    train_group_mask = []
    for row_indices, features in iter_visual_features(
        df, histories, image_index, image_times=image_times
    ):
        visual_groups.append(row_indices)
        raw_features.append(features)
        train_group_mask.append(any(index in train_indices for index in row_indices))

    if not raw_features:
        raise RuntimeError("No se generaron secuencias visuales")
    # Every visual sequence (track/frame) must belong to exactly one temporal
    # partition. Otherwise PCA could ingest validation/test observations.
    for row_indices in visual_groups:
        row_indices_array = np.asarray(row_indices, dtype=int)
        memberships = np.column_stack([
            split["train_mask"][row_indices_array],
            split["calibration_mask"][row_indices_array],
            split["test_mask"][row_indices_array],
        ])
        if not np.all(memberships == memberships[0]):
            raise RuntimeError(
                "Un grupo visual cruza train/calibration/test; "
                "se aborta para impedir leakage temporal."
            )

    raw_matrix = np.asarray(raw_features, dtype=np.float32)
    train_group_mask = np.asarray(train_group_mask, dtype=bool)
    train_visual_count = int(train_group_mask.sum())
    if train_visual_count < MIN_SEQUENCE_LEN * 20:
        raise RuntimeError(f"Secuencias visuales insuficientes en train: {train_visual_count}")

    visual_dim = int(raw_matrix.shape[1])
    components = min(PCA_COMPONENTS, train_visual_count, visual_dim)
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=RANDOM_STATE,
    )
    pca.fit(raw_matrix[train_group_mask])
    visual_pca = pca.transform(raw_matrix).astype(np.float32)

    X = np.zeros((len(df), len(V1_FEATURES) + components + 1), dtype=np.float32)
    X[:, :len(V1_FEATURES)] = df[list(V1_FEATURES)].astype(float).to_numpy(dtype=np.float32)
    for group_index, row_indices in enumerate(visual_groups):
        for row_index in row_indices:
            X[row_index, len(V1_FEATURES):len(V1_FEATURES) + components] = visual_pca[group_index]
            X[row_index, -1] = 1.0

    visual_valid = X[:, -1] > 0.5
    models = {}
    calibrations = {}
    validation_metrics = {}
    for horizon in HORIZONS:
        for target in TARGETS:
            key = f"{horizon}m_{target}"
            horizon_mask = (df["horizonte_min"].to_numpy() == horizon) & visual_valid
            positions = np.flatnonzero(horizon_mask)
            train_pos = np.asarray([p for p in positions if p in train_indices], dtype=int)
            calibration_pos = np.asarray([p for p in positions if p in calibration_indices], dtype=int)
            test_pos = np.asarray([p for p in positions if p in test_indices], dtype=int)
            if len(train_pos) < 100 or len(calibration_pos) < 100 or len(test_pos) < 100:
                raise RuntimeError(
                    f"Secuencias insuficientes {key}: train={len(train_pos)} "
                    f"cal={len(calibration_pos)} test={len(test_pos)}"
                )

            y_train = df.iloc[train_pos][target].astype(int).to_numpy()
            y_cal = df.iloc[calibration_pos][target].astype(int).to_numpy()
            y_test = df.iloc[test_pos][target].astype(int).to_numpy()
            if len(np.unique(y_train)) < 2:
                raise RuntimeError(f"Una sola clase en entrenamiento visual: {key}")

            if target == "toward_bahia":
                # El objetivo direccional no superó persistencia en los folds
                # independientes; se conserva un prior diagnóstico sin probabilidad publicable.
                prior = DummyClassifier(strategy="prior")
                prior.fit(X[train_pos], y_train)
                models[key] = [("prior_no_aprobado", prior, 1.0)]
                calibration = {
                    "enabled": False,
                    "approved": False,
                    "method": "DISABLED_BY_ROLLING_ORIGIN_GATE",
                    "reason": "TOWARD_BAHIA_DID_NOT_BEAT_PERSISTENCE_IN_INDEPENDENT_EVENTS",
                    "calibration_samples": int(len(calibration_pos)),
                    "test_samples": int(len(test_pos)),
                    "rolling_origin_folds": 0,
                    "rolling_origin_passes": 0,
                }
                calibrations[key] = calibration
                baseline_metrics = classification_metrics(
                    y_test, persistence_baseline(df.iloc[test_pos], target)
                )
                prior_probability = ensemble_probability(models[key], X[test_pos])
                validation_metrics[key] = {
                    "train_samples": int(len(train_pos)),
                    "calibration_samples": int(len(calibration_pos)),
                    "test_samples": int(len(test_pos)),
                    "train_positive": int(y_train.sum()),
                    "calibration_positive": int(y_cal.sum()),
                    "test_positive": int(y_test.sum()),
                    "model_raw": classification_metrics(y_test, prior_probability),
                    "model_calibrated": None,
                    "persistence_baseline": baseline_metrics,
                    "beats_persistence": False,
                    "calibration_enabled": False,
                    "calibration_reason": calibration["reason"],
                    "rolling_origin": [],
                    "rolling_origin_passes": 0,
                    "rolling_origin_folds": 0,
                }
                print(f"{key}: no aprobado; se conserva prior sin probabilidad")
                continue

            trained_models = []
            for name, model, weight in make_ensemble():
                model.fit(X[train_pos], y_train)
                trained_models.append((name, model, weight))

            raw_cal = ensemble_probability(trained_models, X[calibration_pos])
            raw_test = ensemble_probability(trained_models, X[test_pos])
            calibration = fit_platt_calibration(
                raw_cal, y_cal, raw_test, y_test, min_samples=100
            )
            raw_metrics = classification_metrics(y_test, raw_test)
            baseline_probability = persistence_baseline(df.iloc[test_pos], target)
            baseline_metrics = classification_metrics(y_test, baseline_probability)
            calibrated_metrics = None
            if calibration.get("a") is not None and calibration.get("b") is not None:
                eps = 1e-6
                clipped = np.clip(raw_test, eps, 1 - eps)
                logits = np.log(clipped / (1 - clipped))
                z = np.clip(float(calibration["a"]) * logits + float(calibration["b"]), -60, 60)
                calibrated_probability = 1.0 / (1.0 + np.exp(-z))
                calibrated_metrics = classification_metrics(y_test, calibrated_probability)

            beats_baseline = bool(
                raw_metrics["samples"] >= 100
                and raw_metrics["roc_auc"] is not None
                and raw_metrics["roc_auc"] > 0.55
                and raw_metrics["brier"] < baseline_metrics["brier"]
                and raw_metrics["balanced_accuracy"] is not None
                and baseline_metrics["balanced_accuracy"] is not None
                and raw_metrics["balanced_accuracy"] >= baseline_metrics["balanced_accuracy"]
            )
            # The final holdout is diagnostic only. Model approval is decided
            # exclusively by independent rolling-origin folds.

            rolling_results = []
            for fold in rolling_folds:
                fold_train = np.asarray([p for p in positions if fold["train_mask"][p]], dtype=int)
                fold_cal = np.asarray([p for p in positions if fold["calibration_mask"][p]], dtype=int)
                fold_test = np.asarray([p for p in positions if fold["test_mask"][p]], dtype=int)
                if min(len(fold_train), len(fold_cal), len(fold_test)) < 100:
                    continue
                y_fold_train = df.iloc[fold_train][target].astype(int).to_numpy()
                y_fold_cal = df.iloc[fold_cal][target].astype(int).to_numpy()
                y_fold_test = df.iloc[fold_test][target].astype(int).to_numpy()
                if min(len(np.unique(y_fold_train)), len(np.unique(y_fold_cal)), len(np.unique(y_fold_test))) < 2:
                    continue
                fold_X, _fold_pca = build_fold_feature_matrix(
                    df,
                    visual_groups,
                    raw_matrix,
                    fold["train_mask"],
                )
                fold_models = []
                for name, model, weight in make_ensemble(n_estimators=20, include_logistic=True):
                    model.fit(fold_X[fold_train], y_fold_train)
                    fold_models.append((name, model, weight))
                fold_raw_cal = ensemble_probability(fold_models, fold_X[fold_cal])
                fold_raw_test = ensemble_probability(fold_models, fold_X[fold_test])
                fold_calibration = fit_platt_calibration(
                    fold_raw_cal, y_fold_cal, fold_raw_test, y_fold_test, min_samples=100
                )
                fold_raw_metrics = classification_metrics(y_fold_test, fold_raw_test)
                fold_baseline_metrics = classification_metrics(
                    y_fold_test, persistence_baseline(df.iloc[fold_test], target)
                )
                fold_beats = bool(
                    fold_raw_metrics["roc_auc"] is not None
                    and fold_raw_metrics["roc_auc"] > 0.55
                    and fold_raw_metrics["brier"] < fold_baseline_metrics["brier"]
                    and fold_raw_metrics["balanced_accuracy"] is not None
                    and fold_baseline_metrics["balanced_accuracy"] is not None
                    and fold_raw_metrics["balanced_accuracy"] >= fold_baseline_metrics["balanced_accuracy"]
                )
                rolling_results.append({
                    "test_event": fold["test_event"],
                    "calibration_event": fold["calibration_event"],
                    "train_samples": int(len(fold_train)),
                    "calibration_samples": int(len(fold_cal)),
                    "test_samples": int(len(fold_test)),
                    "model_raw": fold_raw_metrics,
                    "persistence_baseline": fold_baseline_metrics,
                    "calibration_enabled": bool(fold_calibration.get("enabled")),
                    "beats_persistence": fold_beats,
                    "approved": bool(fold_beats and fold_calibration.get("enabled")),
                    "rolling_model": "EXTRATREES_20_PLUS_LOGISTIC_PROXY",
                })
            rolling_passes = sum(bool(item["approved"]) for item in rolling_results)
            rolling_ok = len(rolling_results) >= 2 and rolling_passes == len(rolling_results)
            if calibration.get("enabled") and not rolling_ok:
                calibration["enabled"] = False
                calibration["reason"] = "ROLLING_ORIGIN_VALIDATION_NOT_PASSED"
            calibration["rolling_origin_folds"] = len(rolling_results)
            calibration["rolling_origin_passes"] = rolling_passes
            calibration["approved"] = bool(rolling_ok and calibration.get("enabled"))
            if not calibration.get("enabled"):
                calibrated_metrics = None

            models[key] = trained_models
            calibrations[key] = calibration
            validation_metrics[key] = {
                "train_samples": int(len(train_pos)),
                "calibration_samples": int(len(calibration_pos)),
                "test_samples": int(len(test_pos)),
                "train_positive": int(y_train.sum()),
                "calibration_positive": int(y_cal.sum()),
                "test_positive": int(y_test.sum()),
                "model_raw": raw_metrics,
                "model_calibrated": calibrated_metrics,
                "persistence_baseline": baseline_metrics,
                "beats_persistence": beats_baseline,
                "calibration_enabled": bool(calibration.get("enabled")),
                "calibration_reason": calibration.get("reason"),
                "rolling_origin": rolling_results,
                "rolling_origin_passes": rolling_passes,
                "rolling_origin_folds": len(rolling_results),
            }
            print(
                f"{key}: AUC={raw_metrics['roc_auc']} "
                f"Brier={raw_metrics['brier']:.4f} "
                f"persistencia={baseline_metrics['brier']:.4f} "
                f"calibrado={calibration.get('enabled')}"
            )

    dataset_hash = sha256_file(DATASET)
    trained_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    approved_models = sum(bool(item.get("approved")) for item in calibrations.values())
    bundle = {
        "model_version": MODEL_VERSION,
        "features_v1": list(V1_FEATURES),
        "features_version": "rma10-causal-features-v2",
        "visual_raw_features": visual_dim,
        "pca_components": components,
        "sequence_len": SEQUENCE_LEN,
        "min_sequence_len": MIN_SEQUENCE_LEN,
        "horizons_min": list(HORIZONS),
        "targets": list(TARGETS),
        "pca": pca,
        "models": models,
        "calibration": calibrations,
        "validation_metrics": validation_metrics,
        "event_split": split_summary,
        "dataset_sha256": dataset_hash,
        "trained_at_utc": trained_at,
        "approved_models": approved_models,
        "total_models": len(models),
        "config": {
            "patch_size": PATCH_SIZE,
            "max_image_gap_min": MAX_IMAGE_GAP_MIN,
            "max_sequence_window_min": MAX_SEQUENCE_WINDOW_MIN,
            "max_current_image_age_min": 15.0,
            "dbz_min": DBZ_MIN,
            "dbz_max": DBZ_MAX,
            "palette_start": PALETTE_START,
            "palette_end": PALETTE_END,
            "motion_compensated": True,
        },
    }
    metrics = {
        "model_version": MODEL_VERSION,
        "generated_utc": trained_at,
        "dataset_rows": int(len(df)),
        "dataset_sha256": dataset_hash,
        "tracks_total": int(df["track_id"].nunique()),
        "images_validas": int(len(image_index)),
        "secuencias_train": int(train_visual_count),
        "secuencias_validas_total": int(visual_valid.sum()),
        "v1_features": list(V1_FEATURES),
        "visual_raw_features": visual_dim,
        "pca_features": components,
        "final_features": int(X.shape[1]),
        "event_split": split_summary,
        "approved_models": approved_models,
        "total_models": len(models),
        "models": validation_metrics,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_model = MODEL_PATH.with_suffix(".tmp.joblib")
    joblib.dump(bundle, tmp_model, compress=3)
    tmp_model.replace(MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Modelo visual causal generado: {MODEL_PATH}")
    print(f"Modelos aprobados: {approved_models}/{len(models)}")


if __name__ == "__main__":
    main()
