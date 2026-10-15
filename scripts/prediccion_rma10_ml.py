#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(".")

MODEL_PATH = ROOT / "data/models/rma10_model.joblib"
STATE_PATH = ROOT / "data/radar/seguimiento/estado.json"
RESULTADO_PATH = ROOT / "data/radar/resultado_procesamiento.json"

OUTPUT_PATH = (
    ROOT
    / "data/radar/seguimiento/prediccion_ml_rma10.json"
)

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

                probability = (
                    get_probability(
                        model,
                        row
                    )
                )

                targets[
                    target
                ] = round(
                    probability
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

    timestamp = (
        resultado.get(
            "imagen_procesada"
        )
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
            },
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
