#!/usr/bin/env python3
"""
RMA10 VISUAL/TEMPORAL AI - V5
Secuencias causales construidas únicamente con imágenes RMA10 reales.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/radar/rma10_dataset_historico.csv"
RADAR_DIR = ROOT / "data/radar/historico"
MODEL_PATH = ROOT / "data/models/rma10_visual_temporal_model.joblib"
METRICS_PATH = ROOT / "data/models/rma10_visual_temporal_model_metrics.json"

MODEL_VERSION = "rma10-visual-temporal-v5-causal-real-sequences"
SEQUENCE_LEN = 6
MIN_SEQUENCE_LEN = 3
MAX_SEQUENCE_WINDOW_MIN = 120.0
MAX_IMAGE_GAP_MIN = 45.0

PATCH_SIZE = 64
DOWNSAMPLE_SIZE = 16
HORIZONS = [5, 10, 20, 30]
TARGETS = ["strengthened", "weakened", "toward_bahia"]
RANDOM_STATE = 42

DBZ_MIN = -15.0
DBZ_MAX = 75.0
PALETTE_START = 64
PALETTE_END = 207
_TIMESTAMP_RE = re.compile(r"^(\d{14})$")


def decode_rma10_dbz(image: Image.Image) -> np.ndarray:
    img = image.convert("P")
    if img.getpalette() is None:
        raise ValueError("Imagen RMA10 sin paleta")
    indices = np.asarray(img, dtype=np.uint8)
    dbz = np.full(indices.shape, np.nan, dtype=np.float32)
    valid = (indices >= PALETTE_START) & (indices <= PALETTE_END)
    if np.any(valid):
        values = indices[valid].astype(np.float32)
        dbz[valid] = DBZ_MIN + (
            (values - PALETTE_START) / float(PALETTE_END - PALETTE_START)
        ) * (DBZ_MAX - DBZ_MIN)
    return dbz


def parse_timestamp_from_name(path: Path) -> datetime | None:
    m = _TIMESTAMP_RE.fullmatch(path.stem)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    if not 2020 <= dt.year <= 2035:
        return None
    return dt.replace(tzinfo=timezone.utc)


def parse_dataset_timestamp(value) -> datetime | None:
    if value is None:
        return None
    dt = pd.to_datetime(str(value).strip(), utc=True, errors="coerce")
    return None if pd.isna(dt) else dt.to_pydatetime()


def build_image_index():
    by_timestamp = {}
    rejected_name = unreadable = duplicate_timestamp = 0

    for path in sorted(RADAR_DIR.rglob("*.png")):
        ts = parse_timestamp_from_name(path)
        if ts is None:
            rejected_name += 1
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            unreadable += 1
            continue

        key = ts.timestamp()
        if key in by_timestamp:
            duplicate_timestamp += 1
            old = by_timestamp[key]
            if path.stat().st_size > old.stat().st_size:
                by_timestamp[key] = path
        else:
            by_timestamp[key] = path

    index = sorted(
        [(datetime.fromtimestamp(ts, tz=timezone.utc), path)
         for ts, path in by_timestamp.items()],
        key=lambda item: item[0],
    )

    print(f"Imágenes indexadas: {len(index)}")
    print(f"Nombres rechazados: {rejected_name}")
    print(f"Imágenes ilegibles: {unreadable}")
    print(f"Timestamps duplicados: {duplicate_timestamp}")
    if index:
        print(f"Rango imágenes: {index[0][0].isoformat()} -> {index[-1][0].isoformat()}")
    return index


@lru_cache(maxsize=64)
def load_dbz(path_string: str) -> np.ndarray:
    with Image.open(path_string) as image:
        return decode_rma10_dbz(image)


def extract_patch(dbz: np.ndarray, x: float, y: float) -> np.ndarray:
    height, width = dbz.shape
    cx, cy = int(round(float(x))), int(round(float(y)))
    half = PATCH_SIZE // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + PATCH_SIZE, y0 + PATCH_SIZE

    patch = np.full((PATCH_SIZE, PATCH_SIZE), DBZ_MIN, dtype=np.float32)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(width, x1), min(height, y1)

    if sx1 > sx0 and sy1 > sy0:
        dx0, dy0 = sx0 - x0, sy0 - y0
        dx1, dy1 = dx0 + sx1 - sx0, dy0 + sy1 - sy0
        source = dbz[sy0:sy1, sx0:sx1]
        finite = np.isfinite(source)
        fill = float(np.nanmedian(source)) if np.any(finite) else DBZ_MIN
        patch[dy0:dy1, dx0:dx1] = np.where(finite, source, fill)

    small = Image.fromarray(patch, mode="F").resize(
        (DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(small, dtype=np.float32)


def frame_stats(frame: np.ndarray) -> np.ndarray:
    v = frame.ravel().astype(np.float32)
    return np.asarray([
        np.mean(v), np.std(v), np.min(v), np.max(v),
        np.percentile(v, 50), np.percentile(v, 75),
        np.percentile(v, 90), np.percentile(v, 95),
        np.percentile(v, 99), np.mean(v > 40.0),
    ], dtype=np.float32)


def feature_vector(frames, valid_mask, elapsed_min, gaps_min):
    absolute = np.concatenate([f.ravel() for f in frames])

    diffs = []
    for i in range(1, SEQUENCE_LEN):
        if valid_mask[i] and valid_mask[i - 1]:
            diffs.append(frames[i] - frames[i - 1])
        else:
            diffs.append(np.zeros_like(frames[i]))
    diff_vec = np.concatenate([d.ravel() for d in diffs])

    stats = np.concatenate([
        frame_stats(f) if valid_mask[i] else np.zeros(10, dtype=np.float32)
        for i, f in enumerate(frames)
    ])

    means = np.asarray([
        float(np.mean(f)) if valid_mask[i] else 0.0
        for i, f in enumerate(frames)
    ], dtype=np.float32)

    p95 = np.asarray([
        float(np.percentile(f, 95)) if valid_mask[i] else 0.0
        for i, f in enumerate(frames)
    ], dtype=np.float32)

    mean_diff = np.asarray([
        means[i] - means[i - 1]
        if valid_mask[i] and valid_mask[i - 1] else 0.0
        for i in range(1, SEQUENCE_LEN)
    ], dtype=np.float32)

    p95_diff = np.asarray([
        p95[i] - p95[i - 1]
        if valid_mask[i] and valid_mask[i - 1] else 0.0
        for i in range(1, SEQUENCE_LEN)
    ], dtype=np.float32)

    return np.concatenate([
        absolute, diff_vec, stats, mean_diff, p95_diff,
        np.asarray(valid_mask, dtype=np.float32),
        np.asarray(elapsed_min, dtype=np.float32),
        np.asarray(gaps_min, dtype=np.float32),
    ]).astype(np.float32)


def build_histories(df):
    required = {"frame_utc", "track_id", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    histories = {}
    clean = df.drop_duplicates(subset=["track_id", "frame_utc"])

    for track_id, group in clean.groupby("track_id", sort=False):
        obs = []
        for row in group.itertuples(index=False):
            ts = parse_dataset_timestamp(getattr(row, "frame_utc"))
            if ts is None:
                continue
            try:
                x, y = float(getattr(row, "x")), float(getattr(row, "y"))
            except Exception:
                continue
            obs.append({"timestamp": ts, "x": x, "y": y})
        obs.sort(key=lambda item: item["timestamp"])
        if obs:
            histories[track_id] = obs
    return histories


def causal_position(history, timestamp):
    prior = [o for o in history if o["timestamp"] <= timestamp]
    if not prior:
        return None
    if len(prior) == 1:
        return prior[-1]["x"], prior[-1]["y"]

    a, b = prior[-2], prior[-1]
    dt = (b["timestamp"] - a["timestamp"]).total_seconds() / 60.0
    if dt <= 0:
        return b["x"], b["y"]

    elapsed = (timestamp - b["timestamp"]).total_seconds() / 60.0
    elapsed = max(0.0, min(elapsed, MAX_IMAGE_GAP_MIN))
    vx = (b["x"] - a["x"]) / dt
    vy = (b["y"] - a["y"]) / dt
    return b["x"] + vx * elapsed, b["y"] + vy * elapsed


def select_real_images(target_timestamp, image_index):
    if not image_index:
        return []

    times = [ts.timestamp() for ts, _ in image_index]
    end = bisect_right(times, target_timestamp.timestamp()) - 1
    if end < 0:
        return []

    window_start = target_timestamp.timestamp() - MAX_SEQUENCE_WINDOW_MIN * 60.0
    selected = []

    for i in range(end, -1, -1):
        ts, path = image_index[i]
        if ts.timestamp() < window_start:
            break
        selected.append((ts, path))
        if len(selected) == SEQUENCE_LEN:
            break

    selected.reverse()
    return selected


def build_sequence_samples(df, histories, image_index):
    X, metadata = [], []
    counters = {
        "rows": 0, "no_history": 0, "short_history": 0,
        "bad_gap": 0, "missing_images": 0, "valid": 0,
    }
    sequence_len_counts = {
        str(i): 0 for i in range(MIN_SEQUENCE_LEN, SEQUENCE_LEN + 1)
    }

    for row in df.itertuples(index=False):
        counters["rows"] += 1
        track_id = getattr(row, "track_id")
        history = histories.get(track_id)

        if not history:
            counters["no_history"] += 1
            continue

        target_ts = parse_dataset_timestamp(getattr(row, "frame_utc"))
        if target_ts is None:
            counters["no_history"] += 1
            continue

        selected = select_real_images(target_ts, image_index)
        if len(selected) < MIN_SEQUENCE_LEN:
            counters["short_history"] += 1
            continue

        gaps = [
            (selected[i][0] - selected[i - 1][0]).total_seconds() / 60.0
            for i in range(1, len(selected))
        ]

        if any(g <= 0 or g > MAX_IMAGE_GAP_MIN for g in gaps):
            counters["bad_gap"] += 1
            continue

        frames = [
            np.zeros((DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE), dtype=np.float32)
            for _ in range(SEQUENCE_LEN)
        ]
        valid_mask = [False] * SEQUENCE_LEN
        elapsed = [MAX_SEQUENCE_WINDOW_MIN] * SEQUENCE_LEN
        feature_gaps = [MAX_IMAGE_GAP_MIN] * (SEQUENCE_LEN - 1)

        start = SEQUENCE_LEN - len(selected)

        # Referencia espacial única: todos los frames se recortan
        # sobre la posición causal del núcleo en el instante objetivo.
        # Así el movimiento real queda visible dentro del parche.
        anchor_pos = causal_position(history, target_ts)
        ok = anchor_pos is not None

        for j, (image_ts, path) in enumerate(selected):
            if not ok:
                break

            try:
                dbz = load_dbz(str(path.resolve()))
                frames[start + j] = extract_patch(
                    dbz,
                    anchor_pos[0],
                    anchor_pos[1],
                )
            except Exception:
                ok = False
                break

            valid_mask[start + j] = True
            elapsed[start + j] = max(
                0.0,
                (target_ts - image_ts).total_seconds() / 60.0,
            )

        if not ok:
            counters["missing_images"] += 1
            continue

        for j, gap in enumerate(gaps):
            feature_gaps[start + j] = gap

        X.append(feature_vector(frames, valid_mask, elapsed, feature_gaps))
        metadata.append({
            "track_id": track_id,
            "frame_utc": str(getattr(row, "frame_utc")),
        })

        counters["valid"] += 1
        sequence_len_counts[str(len(selected))] += 1

    for key in counters:
        print(f"{key}: {counters[key]}")
    print(f"Longitudes de secuencia: {sequence_len_counts}")

    return (
        np.asarray(X, dtype=np.float32),
        pd.DataFrame(metadata),
        counters,
        sequence_len_counts,
    )


def train_one(X, y, groups, target, horizon):
    result = {
        "target": target,
        "horizon_min": horizon,
        "status": None,
        "samples": int(len(y)),
    }

    if len(y) < 20:
        result["status"] = "insufficient_data"
        return None, result

    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 2:
        result["status"] = "insufficient_classes_or_groups"
        return None, result

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    try:
        train_idx, test_idx = next(splitter.split(X, y, groups))
    except Exception:
        result["status"] = "invalid_split"
        return None, result

    if (
        len(np.unique(y[train_idx])) < 2
        or len(np.unique(y[test_idx])) < 2
    ):
        result["status"] = "invalid_class_split"
        return None, result

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    prob = model.predict_proba(X[test_idx])
    pos_idx = list(model.classes_).index(1)

    result["status"] = "ok"
    result["balanced_accuracy"] = float(
        balanced_accuracy_score(y[test_idx], pred)
    )
    result["auc"] = float(
        roc_auc_score(y[test_idx], prob[:, pos_idx])
    )
    result["brier"] = float(
        brier_score_loss(y[test_idx], prob[:, pos_idx])
    )
    result["test_samples"] = int(len(test_idx))

    return model, result


def main():
    print("=== RMA10 VISUAL/TEMPORAL AI ===")
    print(f"Version: {MODEL_VERSION}")

    if not DATASET.exists():
        raise FileNotFoundError(DATASET)
    if not RADAR_DIR.exists():
        raise FileNotFoundError(RADAR_DIR)

    df = pd.read_csv(DATASET)

    print(f"Dataset: {DATASET}")
    print(f"Radar:   {RADAR_DIR}")
    print(
        f"Filas dataset: {len(df)} | "
        f"tracks: {df['track_id'].nunique()}"
    )

    histories = build_histories(df)
    print(f"Historias construidas: {len(histories)}")

    image_index = build_image_index()
    if not image_index:
        raise RuntimeError("No se encontraron imágenes RMA10 válidas.")

    X, metadata, counters, seq_counts = build_sequence_samples(
        df, histories, image_index
    )

    print(f"Secuencias visuales válidas: {len(X)}")

    # No pisa el modelo anterior si la construcción de datos falla.
    if len(X) < 300:
        raise RuntimeError(
            f"Solo se construyeron {len(X)} secuencias; "
            "se aborta para no reemplazar el modelo con un entrenamiento pobre."
        )

    print(f"Features por secuencia: {X.shape[1]}")

    models = {}
    metrics = {
        "version": MODEL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(df)),
        "tracks_total": int(df["track_id"].nunique()),
        "sequences_valid": int(len(X)),
        "features_per_sequence": int(X.shape[1]),
        "counters": counters,
        "sequence_length_counts": seq_counts,
        "image_range": {
            "start": image_index[0][0].isoformat(),
            "end": image_index[-1][0].isoformat(),
        },
        "models": {},
    }

    lookup = {
        (item["track_id"], str(item["frame_utc"])): idx
        for idx, item in metadata.iterrows()
    }

    trained = 0

    for horizon in HORIZONS:
        horizon_df = df[df["horizonte_min"] == horizon]

        for target in TARGETS:
            print(f"Entrenando {target} {horizon} min...")

            indices = []
            labels = []

            for row in horizon_df.itertuples(index=False):
                key = (
                    getattr(row, "track_id"),
                    str(getattr(row, "frame_utc")),
                )

                idx = lookup.get(key)
                if idx is None:
                    continue

                value = getattr(row, target)
                if pd.isna(value):
                    continue

                indices.append(idx)
                labels.append(int(bool(value)))

            if not indices:
                result = {
                    "target": target,
                    "horizon_min": horizon,
                    "status": "insufficient_data",
                    "samples": 0,
                }
                metrics["models"][f"{target}_{horizon}"] = result
                print("NO ENTRENADO | status=insufficient_data")
                continue

            idx_arr = np.asarray(indices, dtype=np.int64)
            y = np.asarray(labels, dtype=np.int8)
            groups = metadata.iloc[idx_arr]["track_id"].to_numpy()

            model, result = train_one(
                X[idx_arr],
                y,
                groups,
                target,
                horizon,
            )

            metrics["models"][f"{target}_{horizon}"] = result

            if model is None:
                print(
                    f"NO ENTRENADO | status={result['status']} | "
                    f"samples={result['samples']}"
                )
                continue

            models[(target, horizon)] = model
            trained += 1

            print(
                f"OK | samples={result['samples']} | "
                f"bal_acc={result['balanced_accuracy']:.4f} | "
                f"AUC={result['auc']:.4f}"
            )

    if trained < 9:
        raise RuntimeError(
            f"Solo {trained}/12 modelos entrenados; "
            "se aborta para no reemplazar el modelo visual "
            "con cobertura insuficiente."
        )

    payload = {
        "version": MODEL_VERSION,
        "models": models,
        "feature_config": {
            "sequence_len": SEQUENCE_LEN,
            "min_sequence_len": MIN_SEQUENCE_LEN,
            "max_sequence_window_min": MAX_SEQUENCE_WINDOW_MIN,
            "max_image_gap_min": MAX_IMAGE_GAP_MIN,
            "patch_size": PATCH_SIZE,
            "downsample_size": DOWNSAMPLE_SIZE,
            "feature_count": int(X.shape[1]),
            "dbz_min": DBZ_MIN,
            "dbz_max": DBZ_MAX,
        },
        "targets": TARGETS,
        "horizons": HORIZONS,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(payload, MODEL_PATH)

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=== FINALIZADO ===")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Métricas: {METRICS_PATH}")
    print(f"Modelos entrenados: {trained}/12")


if __name__ == "__main__":
    main()
