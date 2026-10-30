#!/usr/bin/env python3
from __future__ import annotations

import json
import re
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

SEQUENCE_LEN = 6
MIN_SEQUENCE_LEN = 3
MAX_GAP_MIN = 20
MAX_IMAGE_MATCH_MIN = 10

PATCH_SIZE = 64
DOWNSAMPLE = 4

DBZ_MIN = -15.0
DBZ_MAX = 75.0

PALETTE_FIRST = 64
PALETTE_LAST = 207

TARGETS = (
    "strengthened",
    "weakened",
    "toward_bahia",
)

HORIZONS = (5, 10, 20, 30)

TIMESTAMP_RE = re.compile(r"(?<!\d)(\d{14})(?!\d)")


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


def parse_filename_dt(path: Path):
    match = TIMESTAMP_RE.search(path.stem)

    if not match:
        return None

    value = match.group(1)

    try:
        dt = datetime.strptime(
            value,
            "%Y%m%d%H%M%S",
        ).replace(tzinfo=timezone.utc)

        # Evita aceptar nombres con fechas claramente imposibles.
        if not (2000 <= dt.year <= 2100):
            return None

        return dt

    except ValueError:
        return None


def build_image_index():
    """
    Indexa TODAS las imágenes del histórico.

    No supone que el nombre exacto sea necesariamente
    YYYYMMDDHHMMSS.png. Busca cualquier timestamp de
    14 dígitos dentro del nombre.
    """

    indexed = []

    for path in RADAR_DIR.glob("*.png"):
        dt = parse_filename_dt(path)

        if dt is None:
            continue

        indexed.append(
            {
                "dt": dt,
                "path": path,
            }
        )

    indexed.sort(key=lambda item: item["dt"])

    if not indexed:
        raise RuntimeError(
            f"No se encontraron imágenes RMA10 con timestamp válido en {RADAR_DIR}"
        )

    print(
        f"Imágenes indexadas: {len(indexed)}"
    )

    print(
        f"Rango imágenes: "
        f"{indexed[0]['dt'].isoformat()} -> "
        f"{indexed[-1]['dt'].isoformat()}"
    )

    return indexed


def resolve_image(
    image_index,
    target_dt,
):
    """
    Busca la imagen temporalmente más cercana.

    La tolerancia máxima evita asociar una imagen incorrecta.
    """

    best = None
    best_delta = None

    for item in image_index:
        delta = abs(
            (
                item["dt"] - target_dt
            ).total_seconds()
        ) / 60.0

        if (
            best_delta is None
            or delta < best_delta
        ):
            best = item
            best_delta = delta

    if (
        best is None
        or best_delta > MAX_IMAGE_MATCH_MIN
    ):
        return None

    return best


def validate_rma10_palette(im):
    if im.mode != "P":
        raise ValueError(
            f"Imagen RMA10 inválida: modo {im.mode}"
        )

    palette = im.getpalette()

    if (
        palette is None
        or len(palette) < 208 * 3
    ):
        raise ValueError(
            "Paleta RMA10 ausente o incompleta"
        )


@lru_cache(maxsize=32)
def load_rma10_field(path_string):
    path = Path(path_string)

    with Image.open(path) as im:
        validate_rma10_palette(im)

        indices = np.asarray(
            im,
            dtype=np.uint8,
        )

        transparency = im.info.get(
            "transparency"
        )

        if isinstance(
            transparency,
            bytes,
        ):
            transparency = (
                transparency[0]
                if transparency
                else None
            )

        field = np.zeros(
            indices.shape,
            dtype=np.float32,
        )

        mask = (
            (indices >= PALETTE_FIRST)
            & (indices <= PALETTE_LAST)
        )

        if transparency is not None:
            mask &= (
                indices
                != int(transparency)
            )

        field[mask] = (
            DBZ_MIN
            + (
                (
                    indices[mask].astype(
                        np.float32
                    )
                    - PALETTE_FIRST
                )
                / (
                    PALETTE_LAST
                    - PALETTE_FIRST
                )
            )
            * (
                DBZ_MAX
                - DBZ_MIN
            )
        )

        field /= (
            DBZ_MAX - DBZ_MIN
        )

        return np.clip(
            field,
            0.0,
            1.0,
        )


def crop_resize(
    field,
    x,
    y,
):
    h, w = field.shape

    half = PATCH_SIZE // 2

    cx = int(round(float(x)))
    cy = int(round(float(y)))

    x0 = cx - half
    y0 = cy - half

    x1 = x0 + PATCH_SIZE
    y1 = y0 + PATCH_SIZE

    patch = np.zeros(
        (
            PATCH_SIZE,
            PATCH_SIZE,
        ),
        dtype=np.float32,
    )

    sx0 = max(0, x0)
    sy0 = max(0, y0)

    sx1 = min(w, x1)
    sy1 = min(h, y1)

    if (
        sx1 > sx0
        and sy1 > sy0
    ):
        dx0 = sx0 - x0
        dy0 = sy0 - y0

        dx1 = dx0 + (
            sx1 - sx0
        )
        dy1 = dy0 + (
            sy1 - sy0
        )

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
                PATCH_SIZE
                // DOWNSAMPLE,
                PATCH_SIZE
                // DOWNSAMPLE,
            ),
            resample=Image.Resampling.BILINEAR,
        )

        patch = np.asarray(
            small,
            dtype=np.float32,
        )

    return patch


def feature_vector(patches):
    arr = np.stack(
        patches
    ).astype(np.float32)

    features = []

    # Imagen absoluta de cada frame.
    for frame in arr:
        features.append(
            frame.ravel()
        )

    # Cambios temporales.
    for i in range(
        1,
        len(arr),
    ):
        features.append(
            (
                arr[i]
                - arr[i - 1]
            ).ravel()
        )

    # Estadísticas visuales.
    for frame in arr:
        features.extend(
            [
                np.mean(frame),
                np.std(frame),
                np.percentile(
                    frame,
                    75,
                ),
                np.percentile(
                    frame,
                    90,
                ),
                np.percentile(
                    frame,
                    95,
                ),
                np.percentile(
                    frame,
                    99,
                ),
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
            np.percentile(
                frame,
                95,
            )
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


def load_dataset():
    if not DATASET.exists():
        raise FileNotFoundError(
            DATASET
        )

    df = pd.read_csv(
        DATASET
    )

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
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Faltan columnas: {missing}"
        )

    df = df.copy()

    df["dt"] = [
        parse_dt(value)
        for value in df[
            "frame_utc"
        ]
    ]

    df = df.drop_duplicates(
        subset=[
            "track_id",
            "frame_utc",
            "horizonte_min",
        ]
    )

    return df


def build_histories(df):
    """
    Historial temporal real de cada track.
    """

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

    histories = {}

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
                    "x": float(
                        row.x
                    ),
                    "y": float(
                        row.y
                    ),
                }
            )

        histories[
            track_id
        ] = rows

    return histories


def build_sequences(
    df,
    histories,
    image_index,
):
    """
    Construye una muestra por observación etiquetada.

    Para cada observación:
      - toma hasta 6 frames anteriores;
      - exige al menos 3 frames reales;
      - comprueba los gaps temporales;
      - resuelve cada frame contra el índice de imágenes;
      - extrae el parche alrededor del núcleo.
    """

    X = []
    metadata = []

    index_maps = {}

    for track_id, history in histories.items():
        index_maps[
            track_id
        ] = {
            item["dt"]: i
            for i, item in enumerate(
                history
            )
        }

    counters = {
        "rows": len(df),
        "no_history": 0,
        "short_history": 0,
        "bad_gap": 0,
        "missing_images": 0,
        "valid": 0,
    }

    for row in df.itertuples(
        index=False
    ):
        track_id = row.track_id

        history = histories.get(
            track_id
        )

        if not history:
            counters[
                "no_history"
            ] += 1
            continue

        pos = (
            index_maps[
                track_id
            ].get(row.dt)
        )

        if pos is None:
            counters[
                "no_history"
            ] += 1
            continue

        start = max(
            0,
            pos
            - SEQUENCE_LEN
            + 1,
        )

        seq = history[
            start : pos + 1
        ]

        if len(seq) < MIN_SEQUENCE_LEN:
            counters[
                "short_history"
            ] += 1
            continue

        gaps = [
            (
                seq[i]["dt"]
                - seq[i - 1]["dt"]
            ).total_seconds()
            / 60.0
            for i in range(
                1,
                len(seq),
            )
        ]

        if any(
            gap <= 0
            or gap > MAX_GAP_MIN
            for gap in gaps
        ):
            counters[
                "bad_gap"
            ] += 1
            continue

        resolved = []

        for item in seq:
            match = resolve_image(
                image_index,
                item["dt"],
            )

            if match is None:
                resolved = []
                break

            resolved.append(
                (
                    item,
                    match,
                )
            )

        if len(resolved) < MIN_SEQUENCE_LEN:
            counters[
                "missing_images"
            ] += 1
            continue

        # Relleno controlado únicamente para mantener
        # dimensión fija. Nunca crea información nueva:
        # repite el primer frame REAL.
        while len(resolved) < SEQUENCE_LEN:
            resolved.insert(
                0,
                resolved[0],
            )

        patches = []

        try:
            for item, match in resolved:
                field = load_rma10_field(
                    str(
                        match["path"]
                    )
                )

                patch = crop_resize(
                    field,
                    item["x"],
                    item["y"],
                )

                patches.append(
                    patch
                )

            vector = feature_vector(
                patches
            )

        except (
            OSError,
            ValueError,
        ):
            counters[
                "missing_images"
            ] += 1
            continue

        X.append(vector)

        metadata.append(
            {
                "track_id": track_id,
                "frame_utc": str(
                    row.frame_utc
                ),
                "horizonte_min": int(
                    row.horizonte_min
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
                "sequence_real_frames": len(
                    seq
                ),
            }
        )

        counters[
            "valid"
        ] += 1

    print(
        "=== RESUMEN SECUENCIAS ==="
    )

    for key, value in counters.items():
        print(
            f"{key}: {value}"
        )

    if not X:
        raise RuntimeError(
            "No se pudo construir ninguna secuencia visual."
        )

    return (
        np.stack(X),
        pd.DataFrame(metadata),
    )


def train_one(
    X,
    meta,
    target,
    horizon,
):
    subset = meta[
        meta[
            "horizonte_min"
        ]
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

    unique_classes = np.unique(y)
    unique_groups = np.unique(
        groups
    )

    if len(unique_classes) < 2:
        return {
            "status": "insufficient_classes",
            "target": target,
            "horizonte_min": horizon,
            "samples": len(y),
            "classes": unique_classes.tolist(),
        }

    if len(unique_groups) < 4:
        return {
            "status": "insufficient_groups",
            "target": target,
            "horizonte_min": horizon,
            "samples": len(y),
            "groups": len(
                unique_groups
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

    y_train = y[
        train_idx
    ]

    y_test = y[
        test_idx
    ]

    if len(
        np.unique(y_train)
    ) < 2:
        return {
            "status": "invalid_train_split",
            "target": target,
            "horizonte_min": horizon,
            "samples": len(y),
        }

    if len(
        np.unique(y_test)
    ) < 2:
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

    predictions = model.predict(
        Xh[test_idx]
    )

    probabilities = model.predict_proba(
        Xh[test_idx]
    )[:, 1]

    bal_acc = balanced_accuracy_score(
        y_test,
        predictions,
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    brier = brier_score_loss(
        y_test,
        probabilities,
    )

    return {
        "status": "trained",
        "target": target,
        "horizonte_min": horizon,
        "samples": len(y),
        "train_samples": len(
            train_idx
        ),
        "test_samples": len(
            test_idx
        ),
        "groups": len(
            unique_groups
        ),
        "balanced_accuracy": float(
            bal_acc
        ),
        "auc": float(auc),
        "brier": float(brier),
        "model": model,
    }


def main():
    print(
        "=== RMA10 VISUAL/TEMPORAL AI v3 ==="
    )

    df = load_dataset()

    print(
        f"Dataset: {DATASET}"
    )

    print(
        f"Radar:   {RADAR_DIR}"
    )

    print(
        f"Filas dataset: {len(df)} "
        f"| tracks: {df['track_id'].nunique()}"
    )

    image_index = build_image_index()

    histories = build_histories(
        df
    )

    print(
        f"Historias construidas: "
        f"{len(histories)}"
    )

    X, meta = build_sequences(
        df,
        histories,
        image_index,
    )

    print(
        f"Secuencias visuales válidas: "
        f"{len(meta)}"
    )

    print(
        f"Features por secuencia: "
        f"{X.shape[1]}"
    )

    models = {}
    metrics = []

    for horizon in HORIZONS:
        for target in TARGETS:
            print(
                f"Entrenando "
                f"{target} "
                f"{horizon} min..."
            )

            result = train_one(
                X,
                meta,
                target,
                horizon,
            )

            clean = {
                key: value
                for key, value in result.items()
                if key != "model"
            }

            metrics.append(
                clean
            )

            if (
                result["status"]
                == "trained"
            ):
                key = (
                    target,
                    horizon,
                )

                models[key] = result[
                    "model"
                ]

                print(
                    "OK | "
                    f"samples={result['samples']} "
                    f"| bal_acc="
                    f"{result['balanced_accuracy']:.3f} "
                    f"| AUC="
                    f"{result['auc']:.3f}"
                )
            else:
                print(
                    "NO ENTRENADO | "
                    f"status={result['status']}"
                )

    if not models:
        raise RuntimeError(
            "No se pudo entrenar ningún modelo."
        )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "version": "rma10-visual-temporal-v3",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "sequence_len": SEQUENCE_LEN,
        "min_sequence_len": MIN_SEQUENCE_LEN,
        "max_gap_min": MAX_GAP_MIN,
        "max_image_match_min": MAX_IMAGE_MATCH_MIN,
        "patch_size": PATCH_SIZE,
        "downsample": DOWNSAMPLE,
        "models": models,
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    metrics_payload = {
        "version": "rma10-visual-temporal-v3",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset_rows": len(df),
        "tracks": int(
            df["track_id"].nunique()
        ),
        "image_count": len(
            image_index
        ),
        "sequence_samples": len(
            meta
        ),
        "features": int(
            X.shape[1]
        ),
        "models_trained": len(
            models
        ),
        "models_total": 12,
        "metrics": metrics,
    }

    METRICS_PATH.write_text(
        json.dumps(
            metrics_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=== FINALIZADO ==="
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
