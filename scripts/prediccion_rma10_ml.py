#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(".")

MODEL_PATH = ROOT / "data/models/rma10_model.joblib"
STATE_PATH = ROOT / "data/radar/seguimiento/estado.json"
RESULTADO_PATH = ROOT / "data/radar/resultado_procesamiento.json"
DATASET_PATH = ROOT / "data/radar/rma10_dataset_historico.csv"
OUTPUT_PATH = ROOT / "data/radar/seguimiento/prediccion_ml_rma10.json"
RADAR_TIMESTAMP_PATH = ROOT / "data/radar/radar_timestamp.txt"

RADAR_DIAMETRO_KM = 240.0

RADAR_LAT = -38.7342
RADAR_LON = -62.1654

BAHIA_LAT = -38.7196
BAHIA_LON = -62.2724

FEATURES = [
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
]

HORIZONS = (5, 10, 20, 30)

TARGETS = (
    "strengthened",
    "weakened",
    "toward_bahia",
)

CALIBRATION_MIN_SAMPLES = 20
CALIBRATION_EPS = 1e-6


def load_json(path):
    if not path.exists():
        raise SystemExit(
            f"No existe: {path}"
        )

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        raise SystemExit(
            f"JSON inválido: {path}: {exc}"
        )


def number(value, default=0.0):
    try:
        value = float(value)

        if math.isfinite(value):
            return value

    except (
        TypeError,
        ValueError,
    ):
        pass

    return default


def distance_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    radius = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(
        lat2 - lat1
    )

    dl = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    return (
        2
        * radius
        * math.asin(
            min(
                1.0,
                math.sqrt(a)
            )
        )
    )


def bearing(
    lat1,
    lon1,
    lat2,
    lon2,
):
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dl = math.radians(
        lon2 - lon1
    )

    y = (
        math.sin(dl)
        * math.cos(p2)
    )

    x = (
        math.cos(p1)
        * math.sin(p2)
        - math.sin(p1)
        * math.cos(p2)
        * math.cos(dl)
    )

    return (
        math.degrees(
            math.atan2(
                y,
                x
            )
        )
        + 360.0
    ) % 360.0


def xy_to_geo(
    x,
    y,
    width,
    height,
):
    cx = width / 2.0
    cy = height / 2.0

    dx = (
        x - cx
    ) / max(
        width / 2.0,
        1.0
    ) * (
        RADAR_DIAMETRO_KM / 2.0
    )

    dy = (
        cy - y
    ) / max(
        height / 2.0,
        1.0
    ) * (
        RADAR_DIAMETRO_KM / 2.0
    )

    km_lat = 110.574

    km_lon = (
        111.320
        * math.cos(
            math.radians(
                RADAR_LAT
            )
        )
    )

    return (
        RADAR_LAT + dy / km_lat,
        RADAR_LON + dx / km_lon,
    )


def get_dimensions(resultado):
    width = number(
        resultado.get(
            "ancho_imagen"
        ),
        0
    )

    height = number(
        resultado.get(
            "alto_imagen"
        ),
        0
    )

    if width <= 0:
        width = number(
            resultado.get(
                "width"
            ),
            0
        )

    if height <= 0:
        height = number(
            resultado.get(
                "height"
            ),
            0
        )

    if width <= 0:
        width = 512.0

    if height <= 0:
        height = 512.0

    return width, height


def get_current_position(track):
    history = track.get(
        "historial",
        []
    )

    if (
        isinstance(
            history,
            list
        )
        and history
    ):
        current = history[-1]

        if isinstance(
            current,
            dict
        ):
            if (
                current.get("x")
                is not None
                and current.get("y")
                is not None
            ):
                return (
                    number(
                        current.get(
                            "x"
                        )
                    ),
                    number(
                        current.get(
                            "y"
                        )
                    ),
                )

    if (
        track.get("x")
        is not None
        and track.get("y")
        is not None
    ):
        return (
            number(
                track.get("x")
            ),
            number(
                track.get("y")
            ),
        )

    return None


def get_motion(track):
    history = track.get(
        "historial",
        []
    )

    if (
        not isinstance(
            history,
            list
        )
        or len(history) < 2
    ):
        return (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    previous = history[-2]
    current = history[-1]

    if (
        not isinstance(
            previous,
            dict
        )
        or not isinstance(
            current,
            dict
        )
    ):
        return (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    x1 = number(
        previous.get("x")
    )

    y1 = number(
        previous.get("y")
    )

    x2 = number(
        current.get("x")
    )

    y2 = number(
        current.get("y")
    )

    dx = x2 - x1
    dy = y2 - y1

    displacement = math.hypot(
        dx,
        dy
    )

    dt_minutes = 0.0

    try:
        t1 = datetime.fromisoformat(
            str(
                previous.get(
                    "timestamp"
                )
            ).replace(
                "Z",
                "+00:00"
            )
        )

        t2 = datetime.fromisoformat(
            str(
                current.get(
                    "timestamp"
                )
            ).replace(
                "Z",
                "+00:00"
            )
        )

        dt_minutes = (
            t2 - t1
        ).total_seconds() / 60.0

    except Exception:
        pass

    if dt_minutes > 0:
        speed = (
            displacement
            / dt_minutes
        )
    else:
        speed = 0.0

    if displacement > 0:
        direction = (
            math.degrees(
                math.atan2(
                    dx,
                    -dy
                )
            )
            + 360.0
        ) % 360.0
    else:
        direction = 0.0

    return (
        dx,
        dy,
        displacement,
        speed,
        direction,
    )


def read_radar_timestamp():
    if not RADAR_TIMESTAMP_PATH.exists():
        return None
    try:
        value = RADAR_TIMESTAMP_PATH.read_text(encoding="utf-8").strip()
        return value or None
    except Exception:
        return None


def get_probability(
    model,
    row,
):
    probabilities = (
        model.predict_proba(
            row
        )[0]
    )

    classes = list(
        model.classes_
    )

    if 1 not in classes:
        return 0.0

    index = classes.index(1)

    return float(
        probabilities[index]
    )


def logit(probability):
    p = min(
        1.0 - CALIBRATION_EPS,
        max(
            CALIBRATION_EPS,
            float(probability)
        )
    )

    return math.log(
        p / (1.0 - p)
    )


def sigmoid(value):
    value = max(
        -60.0,
        min(60.0, float(value))
    )

    return 1.0 / (
        1.0 + math.exp(-value)
    )


def expected_calibration_error(
    y_true,
    probabilities,
    bins=10,
):
    y = np.asarray(
        y_true,
        dtype=float
    )

    p = np.asarray(
        probabilities,
        dtype=float
    )

    if len(y) == 0:
        return None

    error = 0.0

    for lower in np.linspace(
        0.0,
        1.0,
        bins,
        endpoint=False,
    ):
        upper = (
            lower
            + 1.0 / bins
        )

        if upper >= 1.0:
            mask = (
                (p >= lower)
                & (p <= upper)
            )
        else:
            mask = (
                (p >= lower)
                & (p < upper)
            )

        if not np.any(mask):
            continue

        error += (
            float(mask.mean())
            * abs(
                float(p[mask].mean())
                -
                float(y[mask].mean())
            )
        )

    return round(
        error,
        6
    )


def recreate_calibration_groups(df):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )

    _, test_idx = next(
        splitter.split(
            df,
            groups=df["track_id"],
        )
    )

    original_test_groups = set(
        df.iloc[test_idx]["track_id"]
    )

    test_df = df[
        df["track_id"].isin(
            original_test_groups
        )
    ].copy()

    splitter_calibration = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=123,
    )

    calibration_idx, evaluation_idx = next(
        splitter_calibration.split(
            test_df,
            groups=test_df["track_id"],
        )
    )

    calibration_groups = set(
        test_df.iloc[
            calibration_idx
        ]["track_id"]
    )

    evaluation_groups = set(
        test_df.iloc[
            evaluation_idx
        ]["track_id"]
    )

    return (
        calibration_groups,
        evaluation_groups,
    )


def build_probability_calibration(
    models,
):
    if not DATASET_PATH.exists():
        raise RuntimeError(
            "No existe el dataset RMA10 para calibración."
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    required = (
        set(FEATURES)
        | {
            "horizonte_min",
            "track_id",
        }
        | set(TARGETS)
    )

    missing = sorted(
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Faltan columnas para calibración: {missing}"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    ).copy()

    df["track_id"] = (
        df["track_id"].astype(str)
    )

    (
        calibration_groups,
        evaluation_groups,
    ) = recreate_calibration_groups(
        df
    )

    calibration = {}
    metrics = {}

    for horizon in HORIZONS:
        subset = df[
            df["horizonte_min"]
            == horizon
        ].copy()

        subset = subset.dropna(
            subset=(
                list(FEATURES)
                + list(TARGETS)
                + ["track_id"]
            )
        )

        calibration_subset = subset[
            subset["track_id"].isin(
                calibration_groups
            )
        ]

        evaluation_subset = subset[
            subset["track_id"].isin(
                evaluation_groups
            )
        ]

        for target in TARGETS:
            key = (
                f"{horizon}m_"
                f"{target}"
            )

            model = models.get(
                key
            )

            if model is None:
                continue

            y = calibration_subset[
                target
            ].astype(int).to_numpy()

            y_evaluation = evaluation_subset[
                target
            ].astype(int).to_numpy()

            if (
                len(y)
                < CALIBRATION_MIN_SAMPLES
                or len(np.unique(y)) < 2
            ):
                calibration[key] = {
                    "enabled": False,
                    "reason": "CALIBRATION_INSUFFICIENT_DATA",
                    "samples": int(len(y)),
                }
                metrics[key] = dict(
                    calibration[key]
                )
                continue

            raw = model.predict_proba(
                calibration_subset[list(FEATURES)]
            )[:, 1]

            raw_evaluation = model.predict_proba(
                evaluation_subset[list(FEATURES)]
            )[:, 1]

            z = np.array([
                logit(value)
                for value in raw
            ]).reshape(-1, 1)

            calibrator = LogisticRegression(
                C=1e6,
                solver="lbfgs",
                random_state=42,
            )

            calibrator.fit(
                z,
                y,
            )

            a = float(
                calibrator.coef_[0][0]
            )

            b = float(
                calibrator.intercept_[0]
            )

            calibrated = np.array([
                sigmoid(
                    a * logit(value)
                    + b
                )
                for value in raw_evaluation
            ])

            brier_raw = float(
                np.mean(
                    (
                        raw_evaluation
                        - y_evaluation
                    ) ** 2
                )
            )

            brier_calibrated = float(
                np.mean(
                    (
                        calibrated
                        - y_evaluation
                    ) ** 2
                )
            )

            ece_raw = expected_calibration_error(y_evaluation, raw_evaluation)
            ece_calibrated = expected_calibration_error(y_evaluation, calibrated)

            use_calibrated = (
                len(y_evaluation) >= CALIBRATION_MIN_SAMPLES
                and ece_raw is not None
                and ece_calibrated is not None
                and brier_calibrated <= brier_raw
                and ece_calibrated <= ece_raw
            )

            calibration[key] = {
                "enabled": bool(use_calibrated),
                "method": "PLATT_SCALING_LOGIT",
                "samples": int(len(y)),
                "positive_samples": int(y.sum()),
                "a": round(a, 10),
                "b": round(b, 10),
                "selection": "CALIBRATED" if use_calibrated else "RAW",
            }

            metrics[key] = {
                "samples": int(len(y)),
                "positive": int(
                    y_evaluation.sum()
                ),
                "calibration_samples": int(
                    len(y)
                ),
                "evaluation_samples": int(
                    len(y_evaluation)
                ),
                "brier_raw": round(
                    brier_raw,
                    6
                ),
                "brier_calibrated": round(
                    brier_calibrated,
                    6
                ),
                "ece_raw": ece_raw,
                "ece_calibrated": ece_calibrated,
                "selection": "CALIBRATED" if use_calibrated else "RAW",
            }

    return {
        "schema_version": 1,
        "method": "PLATT_SCALING_LOGIT",
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat().replace(
            "+00:00",
            "Z",
        ),
        "calibration_tracks": len(
            calibration_groups
        ),
        "evaluation_tracks": len(
            evaluation_groups
        ),
        "minimum_samples": CALIBRATION_MIN_SAMPLES,
        "models": calibration,
        "metrics": metrics,
    }


def apply_probability_calibration(
    probability,
    calibration,
):
    if not calibration:
        return probability

    if not calibration.get(
        "enabled",
        False,
    ):
        return probability

    a = number(
        calibration.get("a"),
        1.0,
    )

    b = number(
        calibration.get("b"),
        0.0,
    )

    return sigmoid(
        a * logit(probability)
        + b
    )


def main():

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"No existe el modelo: "
            f"{MODEL_PATH}"
        )

    state = load_json(
        STATE_PATH
    )

    resultado = load_json(
        RESULTADO_PATH
    )

    bundle = joblib.load(
        MODEL_PATH
    )

    models = bundle.get(
        "models",
        {}
    )

    if not models:
        raise SystemExit(
            "El modelo no contiene modelos entrenados."
        )

    probability_calibration = (
        build_probability_calibration(
            models
        )
    )

    width, height = (
        get_dimensions(
            resultado
        )
    )

    tracks = state.get(
        "tracks",
        []
    )

    if not isinstance(
        tracks,
        list
    ):
        tracks = []

    predictions = []

    calibrated_values = 0

    for track in tracks:

        if not isinstance(
            track,
            dict
        ):
            continue

        if track.get(
            "activo"
        ) is False:
            continue

        if track.get(
            "presente_en_ultimo_frame"
        ) is False:
            continue

        position = (
            get_current_position(
                track
            )
        )

        if position is None:
            continue

        x, y = position

        (
            dx,
            dy,
            displacement,
            speed,
            motion_direction,
        ) = get_motion(
            track
        )

        lat, lon = xy_to_geo(
            x,
            y,
            width,
            height
        )

        distance = distance_km(
            lat,
            lon,
            BAHIA_LAT,
            BAHIA_LON
        )

        target_bearing = bearing(
            lat,
            lon,
            BAHIA_LAT,
            BAHIA_LON
        )

        intensity = number(
            track.get(
                "intensidad",
                track.get(
                    "intensity",
                    0
                )
            )
        )

        area = number(
            track.get(
                "pixeles",
                track.get(
                    "area_px",
                    0
                )
            )
        )

        track_width = number(
            track.get(
                "ancho",
                track.get(
                    "width_px",
                    0
                )
            )
        )

        track_height = number(
            track.get(
                "alto",
                track.get(
                    "height_px",
                    0
                )
            )
        )

        row = pd.DataFrame(
            [
                {
                    "intensity":
                        intensity,

                    "area_px":
                        area,

                    "width_px":
                        track_width,

                    "height_px":
                        track_height,

                    "lat":
                        lat,

                    "lon":
                        lon,

                    "distance_km":
                        distance,

                    "bearing_to_bahia_deg":
                        target_bearing,

                    "displacement_px":
                        displacement,

                    "speed_px_min":
                        speed,

                    "motion_bearing_deg":
                        motion_direction,
                }
            ],
            columns=FEATURES
        )

        horizons = {}

        for horizon in HORIZONS:

            targets = {}

            for target in TARGETS:

                key = (
                    f"{horizon}m_"
                    f"{target}"
                )

                model = models.get(
                    key
                )

                if model is None:
                    targets[
                        target
                    ] = None
                    continue

                raw_probability = (
                    get_probability(
                        model,
                        row
                    )
                )

                calibrated_probability = (
                    apply_probability_calibration(
                        raw_probability,
                        probability_calibration[
                            "models"
                        ].get(
                            key,
                            {},
                        ),
                    )
                )

                if calibrated_probability != raw_probability:
                    calibrated_values += 1

                targets[
                    target
                ] = round(
                    calibrated_probability
                    * 100.0,
                    1
                )

            horizons[
                str(horizon)
            ] = targets

        predictions.append(
            {
                "track_id":
                    track.get(
                        "track_id",
                        track.get(
                            "id"
                        )
                    ),

                "lat":
                    round(
                        lat,
                        6
                    ),

                "lon":
                    round(
                        lon,
                        6
                    ),

                "distance_km":
                    round(
                        distance,
                        2
                    ),

                "bearing_to_bahia_deg":
                    round(
                        target_bearing,
                        1
                    ),

                "displacement_px":
                    round(
                        displacement,
                        2
                    ),

                "speed_px_min":
                    round(
                        speed,
                        4
                    ),

                "motion_bearing_deg":
                    round(
                        motion_direction,
                        1
                    ),

                "horizontes":
                    horizons,
            }
        )

    timestamp_path = ROOT / "data/radar/radar_timestamp.txt"

    if not timestamp_path.exists():
        raise SystemExit(
            "No existe un timestamp real de RMA10. "
            "No se generará una predicción ML sin frame radar válido."
        )

    timestamp = timestamp_path.read_text(
        encoding="utf-8"
    ).strip()

    if not timestamp:
        raise SystemExit(
            "El timestamp real de RMA10 está vacío. "
            "No se generará una predicción ML sin frame radar válido."
        )

    output = {
        "ok": True,

        "modelo":
            bundle.get(
                "model_version",
                "rma10-v1"
            ),

        "generado_utc":
            datetime.now(
                timezone.utc
            ).isoformat().replace(
                "+00:00",
                "Z"
            ),

        "frame_radar":
            timestamp,

        "tracks_analizados":
            len(predictions),

        "horizontes_min":
            list(HORIZONS),

        "calibracion_probabilistica": {
            "aplicada": any(
                bool(model_info.get("enabled"))
                for model_info in probability_calibration["models"].values()
                if isinstance(model_info, dict)
            ),
            "tracks_calibracion": probability_calibration[
                "calibration_tracks"
            ],
            "tracks_evaluacion": probability_calibration[
                "evaluation_tracks"
            ],
            "minimo_muestras": CALIBRATION_MIN_SAMPLES,
            "valores_calibrados": calibrated_values,
            "metricas": probability_calibration[
                "metrics"
            ],
        },

        "predicciones":
            predictions,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "ok": True,
                "tracks_analizados":
                    len(predictions),
                "output":
                    str(
                        OUTPUT_PATH
                    ),
                "calibracion_probabilistica": True,
                "tracks_calibracion": probability_calibration[
                    "calibration_tracks"
                ],
                "valores_calibrados": calibrated_values,
            },
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
