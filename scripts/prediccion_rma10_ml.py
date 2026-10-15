#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from PIL import Image

ROOT = Path(".")
MODEL_PATH = ROOT / "data/models/rma10_model.joblib"
STATE_PATH = ROOT / "data/radar/seguimiento/estado.json"
IMAGE_PATH = ROOT / "data/radar/actual.png"
OUTPUT_PATH = ROOT / "data/radar/seguimiento/prediccion_ml_rma10.json"

RADAR_LAT = -38.7342
RADAR_LON = -62.1654
BAHIA_LAT = -38.7196
BAHIA_LON = -62.2724
RANGE_KM = 240.0

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


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def distance_km(lat1, lon1, lat2, lon2):
    radius = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

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
            min(1.0, math.sqrt(a))
        )
    )


def bearing(lat1, lon1, lat2, lon2):
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)

    y = math.sin(dl) * math.cos(p2)

    x = (
        math.cos(p1) * math.sin(p2)
        - math.sin(p1)
        * math.cos(p2)
        * math.cos(dl)
    )

    return (
        math.degrees(
            math.atan2(y, x)
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
        (x - cx)
        / max(width / 2.0, 1.0)
        * RANGE_KM
    )

    dy = (
        (cy - y)
        / max(height / 2.0, 1.0)
        * RANGE_KM
    )

    km_lat = 110.574

    km_lon = (
        111.320
        * math.cos(
            math.radians(RADAR_LAT)
        )
    )

    return (
        RADAR_LAT + dy / km_lat,
        RADAR_LON + dx / km_lon,
    )


def last_motion(track):
    history = track.get(
        "historial",
        [],
    )

    if len(history) < 2:
        return 0.0, 0.0, None

    a = history[-2]
    b = history[-1]

    try:
        x1 = float(a["x"])
        y1 = float(a["y"])

        x2 = float(b["x"])
        y2 = float(b["y"])
    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return 0.0, 0.0, None

    try:
        ta = datetime.fromisoformat(
            str(
                a["timestamp"]
            ).replace(
                "Z",
                "+00:00",
            )
        )

        tb = datetime.fromisoformat(
            str(
                b["timestamp"]
            ).replace(
                "Z",
                "+00:00",
            )
        )

        dt = (
            tb - ta
        ).total_seconds() / 60.0

    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        dt = 0.0

    if dt <= 0:
        return 0.0, 0.0, None

    dx = x2 - x1
    dy = y2 - y1

    displacement = math.hypot(
        dx,
        dy,
    )

    speed = displacement / dt

    motion_bearing = None

    if displacement > 0.001:
        motion_bearing = (
            math.degrees(
                math.atan2(
                    dx,
                    -dy,
                )
            )
            + 360.0
        ) % 360.0

    return (
        displacement,
        speed,
        motion_bearing,
    )


def probability(model, row):
    values = model.predict_proba(
        row
    )[0]

    classes = list(
        model.classes_
    )

    if 1 in classes:
        return float(
            values[
                classes.index(1)
            ]
        )

    return 0.0


def main():

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"No existe el modelo: {MODEL_PATH}"
        )

    if not STATE_PATH.exists():
        raise SystemExit(
            f"No existe el estado: {STATE_PATH}"
        )

    if not IMAGE_PATH.exists():
        raise SystemExit(
            f"No existe el radar actual: {IMAGE_PATH}"
        )

    bundle = joblib.load(
        MODEL_PATH
    )

    state = load_json(
        STATE_PATH
    )

    with Image.open(
        IMAGE_PATH
    ) as image:
        width, height = image.size

    models = bundle["models"]

    tracks = state.get(
        "tracks",
        [],
    )

    timestamp = state.get(
        "ultimo_radar_timestamp"
    )

    predictions = []

    for track in tracks:

        if not track.get(
            "activo",
            False,
        ):
            continue

        if not track.get(
            "presente_en_ultimo_frame",
            False,
        ):
            continue

        try:
            x = float(
                track["x"]
            )

            y = float(
                track["y"]
            )

            area = float(
                track.get(
                    "pixeles",
                    0,
                )
            )

            track_width = float(
                track.get(
                    "ancho",
                    0,
                )
            )

            track_height = float(
                track.get(
                    "alto",
                    0,
                )
            )

            intensity = float(
                track.get(
                    "intensidad",
                    0,
                )
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        lat, lon = xy_to_geo(
            x,
            y,
            width,
            height,
        )

        dist = distance_km(
            lat,
            lon,
            BAHIA_LAT,
            BAHIA_LON,
        )

        target_bearing = bearing(
            lat,
            lon,
            BAHIA_LAT,
            BAHIA_LON,
        )

        (
            displacement,
            speed,
            motion_bearing,
        ) = last_motion(
            track
        )

        motion_bearing_value = (
            motion_bearing
            if motion_bearing is not None
            else 0.0
        )

        row = pd.DataFrame(
            [
                {
                    "intensity": intensity,
                    "area_px": area,
                    "width_px": track_width,
                    "height_px": track_height,
                    "lat": lat,
                    "lon": lon,
                    "distance_km": dist,
                    "bearing_to_bahia_deg":
                        target_bearing,
                    "displacement_px":
                        displacement,
                    "speed_px_min":
                        speed,
                    "motion_bearing_deg":
                        motion_bearing_value,
                }
            ],
            columns=FEATURES,
        )

        horizon_predictions = {}

        for horizon in HORIZONS:

            target_predictions = {}

            for target in TARGETS:

                key = (
                    f"{horizon}m_{target}"
                )

                model = models[key]

                p = probability(
                    model,
                    row,
                )

                target_predictions[
                    target
                ] = round(
                    p * 100.0,
                    1,
                )

            horizon_predictions[
                str(horizon)
            ] = target_predictions

        predictions.append(
            {
                "track_id":
                    track.get(
                        "track_id"
                    ),

                "x": round(
                    x,
                    2,
                ),

                "y": round(
                    y,
                    2,
                ),

                "lat": round(
                    lat,
                    6,
                ),

                "lon": round(
                    lon,
                    6,
                ),

                "distance_km":
                    round(
                        dist,
                        2,
                    ),

                "bearing_to_bahia_deg":
                    round(
                        target_bearing,
                        1,
                    ),

                "displacement_px":
                    round(
                        displacement,
                        2,
                    ),

                "speed_px_min":
                    round(
                        speed,
                        3,
                    ),

                "motion_bearing_deg":
                    (
                        round(
                            motion_bearing,
                            1,
                        )
                        if motion_bearing
                        is not None
                        else None
                    ),

                "horizontes":
                    horizon_predictions,
            }
        )

    output = {
        "modelo":
            bundle.get(
                "model_version",
                "rma10-v1",
            ),

        "fuente":
            "SINARAME RMA10",

        "generado_utc":
            datetime.now(
                timezone.utc
            ).isoformat().replace(
                "+00:00",
                "Z",
            ),

        "frame_radar_utc":
            timestamp,

        "tracks_analizados":
            len(predictions),

        "horizontes_min":
            list(HORIZONS),

        "predicciones":
            predictions,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "ok": True,
                "tracks_analizados":
                    len(predictions),
                "output":
                    str(OUTPUT_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
