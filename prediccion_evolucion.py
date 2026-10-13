#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


BASE = Path("data/radar/seguimiento")

ESTADO = BASE / "estado.json"
INPUT = BASE / "seguimiento_temporal.json"
OUTPUT = BASE / "prediccion_evolucion.json"
PENDING = BASE / "predicciones_pendientes.json"


HORIZONTES_MIN = (5, 10, 20, 30)

PASO_HUECO_MIN = 5
TOLERANCIA_HUECO_MIN = 12

MAX_PENDIENTES = 5000

MIN_MATCH_PIXELS = 12.0
MAX_MATCH_EXTRA_UNCERTAINTY = 1.75


def load_json(path, default):
    if not path.exists():
        return default

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
    ):
        return default


def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_ts(value):
    if not value:
        return None

    try:
        text = str(value).strip()

        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        dt = datetime.fromisoformat(text)

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt.astimezone(
            timezone.utc
        )

    except (
        TypeError,
        ValueError,
    ):
        return None


def iso(dt):
    return (
        dt.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def canonical_ts(value):
    dt = parse_ts(value)

    if dt is None:
        return None

    return iso(dt)


def clamp(
    value,
    lo=0.0,
    hi=100.0,
):
    try:
        return max(
            lo,
            min(
                hi,
                float(value),
            ),
        )
    except (
        TypeError,
        ValueError,
    ):
        return lo


def track_id(track):
    value = track.get(
        "track_id",
        track.get("id"),
    )

    if value is None:
        return None

    return str(value)


def sector(degrees):
    if degrees is None:
        return None

    names = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSO",
        "SO",
        "OSO",
        "O",
        "ONO",
        "NO",
        "NNO",
    ]

    return names[
        int(
            (degrees % 360)
            / 22.5
            + 0.5
        )
        % 16
    ]


def angle_diff(a, b):
    if a is None or b is None:
        return None

    return abs(
        (a - b + 180)
        % 360
        - 180
    )


def valid_history(track):
    history = []

    for item in track.get(
        "historial",
        [],
    ):

        if not isinstance(
            item,
            dict,
        ):
            continue

        if (
            item.get("x") is None
            or item.get("y") is None
        ):
            continue

        ts = parse_ts(
            item.get("timestamp")
        )

        if ts is None:
            continue

        item = dict(item)

        item["_dt"] = ts

        history.append(item)

    history.sort(
        key=lambda x: x["_dt"]
    )

    return history


def clean_point(item):
    result = dict(item)

    result.pop(
        "_dt",
        None,
    )

    return result


def recent_history(
    track,
    limit=8,
):
    return valid_history(track)[
        -limit:
    ]


def movement_samples(
    history,
    max_interval=60.0,
):
    result = []

    for a, b in zip(
        history,
        history[1:],
    ):

        ta = a["_dt"]
        tb = b["_dt"]

        dt = (
            tb - ta
        ).total_seconds() / 60.0

        if dt < 2:
            continue

        if dt > max_interval:
            continue

        dx = (
            float(b["x"])
            - float(a["x"])
        )

        dy = (
            float(b["y"])
            - float(a["y"])
        )

        speed = (
            math.hypot(
                dx,
                dy,
            )
            / dt
        )

        bearing = (
            math.degrees(
                math.atan2(
                    dx,
                    -dy,
                )
            )
            % 360
            if speed > 0.001
            else None
        )

        result.append(
            {
                "dt": dt,
                "vx": dx / dt,
                "vy": dy / dt,
                "speed": speed,
                "bearing": bearing,
            }
        )

    return result


def estimate_motion(
    track,
    history=None,
):
    if history is None:
        history = recent_history(
            track
        )

    movements = movement_samples(
        history
    )

    if not history:
        return None

    if not movements:
        return None

    weights = list(
        range(
            1,
            len(movements) + 1,
        )
    )

    total = sum(weights)

    vx = sum(
        m["vx"] * w
        for m, w in zip(
            movements,
            weights,
        )
    ) / total

    vy = sum(
        m["vy"] * w
        for m, w in zip(
            movements,
            weights,
        )
    ) / total

    speed = math.hypot(
        vx,
        vy,
    )

    bearing = (
        math.degrees(
            math.atan2(
                vx,
                -vy,
            )
        )
        % 360
        if speed > 0.001
        else None
    )

    acceleration = track.get(
        "aceleracion_pixeles_min2"
    )

    if (
        acceleration is None
        and len(movements) >= 2
    ):

        a = movements[-2]
        b = movements[-1]

        avg_dt = max(
            (
                a["dt"]
                + b["dt"]
            )
            / 2.0,
            1.0,
        )

        acceleration = (
            b["speed"]
            - a["speed"]
        ) / avg_dt

    turns = []

    for i in range(
        1,
        len(movements),
    ):

        value = angle_diff(
            movements[i]["bearing"],
            movements[i - 1][
                "bearing"
            ],
        )

        if value is not None:
            turns.append(value)

    consistency = clamp(
        100.0
        - (
            mean(turns)
            * 0.25
            if turns
            else 0.0
        )
    )

    movement_confidence = track.get(
        "calidad_movimiento",
        track.get(
            "confianza_movimiento",
            track.get(
                "confianza",
                50,
            ),
        ),
    )

    try:
        movement_confidence = float(
            movement_confidence
        )
    except (
        TypeError,
        ValueError,
    ):
        movement_confidence = 50.0

    confidence = clamp(
        0.60
        * movement_confidence
        + 0.25
        * consistency
        + 15.0
        * min(
            len(movements),
            3,
        )
        / 3.0
    )

    area_changes = []

    for item in history[1:]:

        value = item.get(
            "cambio_area_porcentaje"
        )

        if value is None:
            continue

        try:
            area_changes.append(
                float(value)
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

    area_trend = (
        mean(
            area_changes[-4:]
        )
        if area_changes
        else None
    )

    intensification = 50.0

    if area_trend is not None:
        intensification += max(
            -25.0,
            min(
                25.0,
                area_trend * 0.8,
            ),
        )

    if acceleration is not None:
        intensification += max(
            -15.0,
            min(
                15.0,
                float(acceleration)
                * 300.0,
            ),
        )

    intensification = clamp(
        intensification
    )

    return {
        "vx": vx,
        "vy": vy,
        "speed": speed,
        "bearing": bearing,
        "acceleration": acceleration,
        "confidence": confidence,
        "area_trend": area_trend,
        "intensification": intensification,
        "weakening": (
            100.0
            - intensification
        ),
        "history_points": len(
            history
        ),
        "last": clean_point(
            history[-1]
        ),
    }


def project(
    motion,
    x0,
    y0,
    horizon,
):
    vx = motion["vx"]
    vy = motion["vy"]

    speed = motion["speed"]

    acceleration = (
        motion["acceleration"]
    )

    vx_future = vx
    vy_future = vy

    if (
        acceleration is not None
        and abs(
            float(acceleration)
        ) < 0.15
        and speed > 0.01
    ):

        ux = vx / speed
        uy = vy / speed

        delta_v = (
            float(acceleration)
            * horizon
        )

        delta_v = max(
            -speed * 0.5,
            min(
                speed * 0.5,
                delta_v,
            ),
        )

        vx_future += (
            ux * delta_v
        )

        vy_future += (
            uy * delta_v
        )

    x_future = (
        float(x0)
        + vx_future * horizon
    )

    y_future = (
        float(y0)
        + vy_future * horizon
    )

    future_speed = math.hypot(
        vx_future,
        vy_future,
    )

    future_bearing = (
        math.degrees(
            math.atan2(
                vx_future,
                -vy_future,
            )
        )
        % 360
        if future_speed > 0.001
        else None
    )

    base_error = max(
        2.0,
        (
            100.0
            - motion["confidence"]
        )
        * 0.12,
    )

    uncertainty = (
        base_error
        + horizon
        * max(
            0.10,
            0.35
            - motion["confidence"]
            / 400.0,
        )
    )

    if horizon > 60:
        uncertainty += (
            horizon - 60.0
        ) * 0.75

    confidence = (
        motion["confidence"]
        - max(
            0.0,
            horizon - 5.0,
        )
        * 1.2
    )

    if horizon > 60:
        confidence -= (
            horizon - 60.0
        ) * 0.12

    return {
        "x_estimado": round(
            x_future,
            2,
        ),
        "y_estimado": round(
            y_future,
            2,
        ),
        "velocidad_estimada_pixeles_min": round(
            future_speed,
            4,
        ),
        "rumbo_estimado_grados": (
            round(
                future_bearing,
                1,
            )
            if future_bearing
            is not None
            else None
        ),
        "direccion_estimada": sector(
            future_bearing
        ),
        "incertidumbre_pixeles": round(
            uncertainty,
            2,
        ),
        "confianza_prediccion": round(
            clamp(confidence),
            1,
        ),
    }


def validate_point(
    prediction,
    actual,
    modo,
):
    dx = (
        float(actual["x"])
        - float(
            prediction[
                "x_estimado"
            ]
        )
    )

    dy = (
        float(actual["y"])
        - float(
            prediction[
                "y_estimado"
            ]
        )
    )

    return {
        "estado": "VALIDADA",
        "modo": modo,
        "timestamp_real": canonical_ts(
            actual["timestamp"]
        ),
        "x_real": round(
            float(actual["x"]),
            2,
        ),
        "y_real": round(
            float(actual["y"]),
            2,
        ),
        "track_id_real": (
            str(
                actual.get(
                    "track_id"
                )
            )
            if actual.get(
                "track_id"
            )
            is not None
            else None
        ),
        "error_pixeles": round(
            math.hypot(
                dx,
                dy,
            ),
            2,
        ),
        "error_x_pixeles": round(
            dx,
            2,
        ),
        "error_y_pixeles": round(
            dy,
            2,
        ),
    }


def all_observations_at(
    tracks,
    timestamp,
):
    target = canonical_ts(
        timestamp
    )

    result = []

    for track in tracks:

        tid = track_id(
            track
        )

        for obs in valid_history(
            track
        ):

            if (
                canonical_ts(
                    obs["timestamp"]
                )
                != target
            ):
                continue

            item = clean_point(
                obs
            )

            item["track_id"] = tid

            result.append(
                item
            )

    return result


def cross_track_match(
    prediction,
    observations,
):
    if not observations:
        return None

    px = float(
        prediction[
            "x_estimado"
        ]
    )

    py = float(
        prediction[
            "y_estimado"
        ]
    )

    uncertainty = float(
        prediction.get(
            "incertidumbre_pixeles",
            0,
        )
    )

    max_distance = max(
        MIN_MATCH_PIXELS,
        uncertainty
        * MAX_MATCH_EXTRA_UNCERTAINTY,
    )

    best = None

    for obs in observations:

        try:
            distance = math.hypot(
                float(obs["x"])
                - px,
                float(obs["y"])
                - py,
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if (
            distance
            <= max_distance
        ):

            if (
                best is None
                or distance
                < best[0]
            ):
                best = (
                    distance,
                    obs,
                )

    if best is None:
        return None

    return best[1]


def build_observation_index(
    tracks,
):
    result = {}

    for track in tracks:

        tid = track_id(
            track
        )

        for obs in valid_history(
            track
        ):

            ts = canonical_ts(
                obs["timestamp"]
            )

            if ts is None:
                continue

            item = clean_point(
                obs
            )

            item["track_id"] = tid

            result[
                (
                    tid,
                    ts,
                )
            ] = item

    return result


def validate_pending(
    current_ts,
    tracks,
    pending,
):
    now = parse_ts(
        current_ts
    )

    index = (
        build_observation_index(
            tracks
        )
    )

    remaining = []
    validated = []

    for prediction in pending:

        target = canonical_ts(
            prediction.get(
                "timestamp_objetivo"
            )
        )

        if (
            not target
            or not now
        ):
            remaining.append(
                prediction
            )
            continue

        target_dt = parse_ts(
            target
        )

        if (
            target_dt is None
            or target_dt > now
        ):
            remaining.append(
                prediction
            )
            continue

        tid = str(
            prediction.get(
                "track_id"
            )
        )

        actual = index.get(
            (
                tid,
                target,
            )
        )

        if actual is None:
            actual = (
                cross_track_match(
                    prediction,
                    all_observations_at(
                        tracks,
                        target,
                    ),
                )
            )

        if actual is None:
            remaining.append(
                prediction
            )
            continue

        prediction[
            "validacion"
        ] = validate_point(
            prediction,
            actual,
            "TRACK_REASIGNADO_POR_PROXIMIDAD",
        )

        validated.append(
            prediction
        )

    return (
        remaining,
        validated,
    )


def make_gap_predictions(
    track,
    motion,
    start_ts,
    end_ts,
):
    start = parse_ts(
        start_ts
    )

    end = parse_ts(
        end_ts
    )

    if (
        not start
        or not end
    ):
        return []

    gap = (
        end - start
    ).total_seconds() / 60.0

    if (
        gap
        <= TOLERANCIA_HUECO_MIN
    ):
        return []

    origin = motion[
        "last"
    ]

    x0 = float(
        origin["x"]
    )

    y0 = float(
        origin["y"]
    )

    tid = track_id(
        track
    )

    result = []

    horizon = (
        PASO_HUECO_MIN
    )

    while horizon < gap:

        target = (
            start
            + (
                end - start
            )
            * (
                horizon / gap
            )
        )

        result.append(
            {
                "track_id": tid,
                "timestamp_origen": canonical_ts(
                    start_ts
                ),
                "timestamp_objetivo": iso(
                    target
                ),
                "horizonte_desde_origen_minutos": horizon,
                "tipo": "ESTIMACION_HUECO",
                **project(
                    motion,
                    x0,
                    y0,
                    horizon,
                ),
            }
        )

        horizon += (
            PASO_HUECO_MIN
        )

    result.append(
        {
            "track_id": tid,
            "timestamp_origen": canonical_ts(
                start_ts
            ),
            "timestamp_objetivo": canonical_ts(
                end_ts
            ),
            "horizonte_desde_origen_minutos": round(
                gap,
                2,
            ),
            "tipo": "PREDICCION_CIERRE_HUECO",
            **project(
                motion,
                x0,
                y0,
                gap,
            ),
        }
    )

    return result


def predict_track(track):
    history = recent_history(
        track,
        8,
    )

    motion = estimate_motion(
        track,
        history,
    )

    if motion is None:

        return {
            "track_id": track_id(
                track
            ),
            "pronostico_disponible": False,
            "motivo": (
                "SIN_MOVIMIENTO_VALIDO"
            ),
        }

    current = motion[
        "last"
    ]

    forecasts = []

    for horizon in (
        HORIZONTES_MIN
    ):

        forecasts.append(
            {
                "horizonte_minutos": horizon,
                **project(
                    motion,
                    current["x"],
                    current["y"],
                    horizon,
                ),
            }
        )

    return {
        "track_id": track_id(
            track
        ),
        "activo": bool(
            track.get(
                "activo",
                True,
            )
        ),
        "pronostico_disponible": True,
        "posicion_actual": {
            "x": round(
                float(
                    current["x"]
                ),
                2,
            ),
            "y": round(
                float(
                    current["y"]
                ),
                2,
            ),
        },
        "velocidad_actual_pixeles_min": round(
            motion["speed"],
            4,
        ),
        "rumbo_actual_grados": (
            round(
                motion["bearing"],
                1,
            )
            if motion[
                "bearing"
            ]
            is not None
            else None
        ),
        "direccion_actual": sector(
            motion["bearing"]
        ),
        "aceleracion_pixeles_min2": (
            round(
                float(
                    motion[
                        "acceleration"
                    ]
                ),
                5,
            )
            if motion[
                "acceleration"
            ]
            is not None
            else None
        ),
        "tendencia_area_porcentaje": (
            round(
                motion[
                    "area_trend"
                ],
                2,
            )
            if motion[
                "area_trend"
            ]
            is not None
            else None
        ),
        "probabilidad_intensificacion": round(
            motion[
                "intensification"
            ],
            1,
        ),
        "probabilidad_debilitamiento": round(
            motion[
                "weakening"
            ],
            1,
        ),
        "confianza_prediccion": round(
            motion[
                "confidence"
            ],
            1,
        ),
        "puntos_historial_utilizados": (
            motion[
                "history_points"
            ]
        ),
        "horizontes": forecasts,
        "huecos_detectados": 0,
        "estimaciones_hueco": [],
    }


def get_tracks(
    estado,
    temporal,
):
    tracks = estado.get(
        "tracks",
        [],
    )

    if not isinstance(
        tracks,
        list,
    ):
        tracks = []

    tracks = [
        dict(t)
        for t in tracks
        if isinstance(
            t,
            dict,
        )
    ]

    if not tracks:

        fallback = temporal.get(
            "tracks",
            [],
        )

        if isinstance(
            fallback,
            list,
        ):
            tracks = [
                dict(t)
                for t in fallback
                if isinstance(
                    t,
                    dict,
                )
            ]

    return tracks


def previous_radar_timestamp(
    tracks,
    current_ts,
):
    current = parse_ts(
        current_ts
    )

    if current is None:
        return None

    timestamps = []

    for track in tracks:

        for obs in valid_history(
            track
        ):

            if (
                obs["_dt"]
                < current
            ):
                timestamps.append(
                    obs["_dt"]
                )

    if not timestamps:
        return None

    return max(
        timestamps
    )


def tracks_ending_at(
    tracks,
    timestamp,
):
    target = canonical_ts(
        timestamp
    )

    result = []

    for track in tracks:

        history = valid_history(
            track
        )

        if not history:
            continue

        last = history[-1]

        if (
            canonical_ts(
                last["timestamp"]
            )
            == target
        ):
            result.append(
                track
            )

    return result


def current_gap(
    tracks,
    radar_ts,
):
    previous = (
        previous_radar_timestamp(
            tracks,
            radar_ts,
        )
    )

    if previous is None:
        return (
            [],
            0.0,
            None,
        )

    current = parse_ts(
        radar_ts
    )

    gap = (
        current - previous
    ).total_seconds() / 60.0

    if (
        gap
        <= TOLERANCIA_HUECO_MIN
    ):
        return (
            [],
            gap,
            previous,
        )

    now_observations = (
        all_observations_at(
            tracks,
            radar_ts,
        )
    )

    result = []

    previous_tracks = (
        tracks_ending_at(
            tracks,
            previous,
        )
    )

    for track in previous_tracks:

        history = valid_history(
            track
        )

        if len(history) < 2:
            continue

        motion = estimate_motion(
            track,
            history[-8:],
        )

        if motion is None:
            continue

        points = make_gap_predictions(
            track,
            motion,
            iso(previous),
            radar_ts,
        )

        for point in points:

            if (
                point["tipo"]
                != "PREDICCION_CIERRE_HUECO"
            ):
                continue

            actual = (
                cross_track_match(
                    point,
                    now_observations,
                )
            )

            if actual is not None:

                point[
                    "validacion"
                ] = validate_point(
                    point,
                    actual,
                    "TRACK_REASIGNADO_POR_PROXIMIDAD",
                )

        result.extend(
            points
        )

    return (
        result,
        gap,
        previous,
    )


def main():

    estado = load_json(
        ESTADO,
        {},
    )

    temporal = load_json(
        INPUT,
        {},
    )

    radar_ts = canonical_ts(
        estado.get(
            "radar_timestamp"
        )
        or estado.get(
            "ultimo_radar_timestamp"
        )
        or temporal.get(
            "radar_timestamp"
        )
    )

    if radar_ts is None:
        raise SystemExit(
            "ERROR: no se pudo determinar radar_timestamp"
        )

    tracks = get_tracks(
        estado,
        temporal,
    )

    try:

        intervalo = float(
            temporal.get(
                "intervalo_minutos",
                0,
            )
            or 0
        )

    except (
        TypeError,
        ValueError,
    ):

        intervalo = 0.0

    pending_data = load_json(
        PENDING,
        {},
    )

    if isinstance(
        pending_data,
        dict,
    ):
        pending = (
            pending_data.get(
                "pendientes",
                [],
            )
        )
    else:
        pending = []

    if not isinstance(
        pending,
        list,
    ):
        pending = []

    (
        pending,
        validated_now,
    ) = validate_pending(
        radar_ts,
        tracks,
        pending,
    )

    predictions = []

    for track in tracks:

        if len(
            valid_history(track)
        ) < 2:
            continue

        predictions.append(
            predict_track(
                track
            )
        )

    (
        gap_predictions,
        gap_minutes,
        previous,
    ) = current_gap(
        tracks,
        radar_ts,
    )

    existing = {
        (
            str(
                item.get(
                    "track_id"
                )
            ),
            canonical_ts(
                item.get(
                    "timestamp_objetivo"
                )
            ),
        )
        for item in pending
    }

    for point in (
        gap_predictions
    ):

        validation = point.get(
            "validacion",
            {},
        )

        if (
            validation.get(
                "estado"
            )
            == "VALIDADA"
        ):
            continue

        key = (
            str(
                point.get(
                    "track_id"
                )
            ),
            canonical_ts(
                point.get(
                    "timestamp_objetivo"
                )
            ),
        )

        if key in existing:
            continue

        pending.append(
            point
        )

        existing.add(
            key
        )

    pending = pending[
        -MAX_PENDIENTES:
    ]

    available = [
        item
        for item in predictions
        if item.get(
            "pronostico_disponible"
        )
    ]

    errors = []

    for item in (
        validated_now
    ):

        value = item.get(
            "validacion",
            {},
        ).get(
            "error_pixeles"
        )

        if value is None:
            continue

        try:
            errors.append(
                float(value)
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

    for item in (
        gap_predictions
    ):

        value = item.get(
            "validacion",
            {},
        ).get(
            "error_pixeles"
        )

        if value is None:
            continue

        try:
            errors.append(
                float(value)
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

    validated_gaps = sum(
        1
        for item in gap_predictions
        if item.get(
            "validacion",
            {},
        ).get(
            "estado"
        )
        == "VALIDADA"
    )

    summary = {
        "nucleos_con_prediccion": len(
            available
        ),
        "nucleos_sin_prediccion": (
            len(predictions)
            - len(available)
        ),
        "promedio_intensificacion": (
            round(
                mean(
                    item[
                        "probabilidad_intensificacion"
                    ]
                    for item in available
                ),
                1,
            )
            if available
            else None
        ),
        "promedio_debilitamiento": (
            round(
                mean(
                    item[
                        "probabilidad_debilitamiento"
                    ]
                    for item in available
                ),
                1,
            )
            if available
            else None
        ),
        "promedio_confianza": (
            round(
                mean(
                    item[
                        "confianza_prediccion"
                    ]
                    for item in available
                ),
                1,
            )
            if available
            else None
        ),
        "huecos_estimados": len(
            gap_predictions
        ),
        "hueco_actual_minutos": round(
            gap_minutes,
            2,
        ),
        "radar_anterior_timestamp": (
            canonical_ts(
                previous
            )
            if previous
            else None
        ),
        "predicciones_validadas_en_este_frame": len(
            validated_now
        ),
        "cierres_hueco_validados": (
            validated_gaps
        ),
        "error_medio_pixeles_validado": (
            round(
                mean(errors),
                2,
            )
            if errors
            else None
        ),
    }

    result = {
        "schema_version": 5,
        "radar_timestamp": radar_ts,
        "intervalo_minutos": intervalo,
        "tipo": (
            "PREDICCION_MOVIMIENTO_CON_HUECOS"
        ),
        "fuente_tracks": str(
            ESTADO
        ),
        "prediccion_global_disponible": bool(
            available
        ),
        "nucleos": predictions,
        "huecos": gap_predictions,
        "resumen": summary,
        "validaciones": (
            validated_now[-100:]
        ),
        "nota": (
            "Las posiciones estimadas son "
            "extrapolaciones. Nunca se etiquetan "
            "como radar observado. Los huecos "
            "largos se reconstruyen desde el "
            "ultimo radar real anterior y se "
            "validan contra el radar real del "
            "timestamp exacto aunque cambie "
            "el track_id."
        ),
    }

    save_json(
        OUTPUT,
        result,
    )

    save_json(
        PENDING,
        {
            "schema_version": 4,
            "actualizado": radar_ts,
            "pendientes": pending,
        },
    )

    previous_text = (
        canonical_ts(previous)
        if previous
        else "None"
    )

    print(
        f"OK: {OUTPUT}"
    )

    print(
        f"Tracks analizados: "
        f"{len(tracks)}"
    )

    print(
        f"Predicciones: "
        f"{len(predictions)}"
    )

    print(
        f"Radar anterior: "
        f"{previous_text}"
    )

    print(
        f"Hueco actual: "
        f"{round(gap_minutes, 2)} "
        f"minutos"
    )

    print(
        f"Estimaciones de hueco: "
        f"{len(gap_predictions)}"
    )

    print(
        f"Cierres de hueco validados: "
        f"{validated_gaps}"
    )

    print(
        f"Predicciones validadas: "
        f"{len(validated_now)}"
    )

    print(
        f"Pendientes: "
        f"{len(pending)}"
    )


if __name__ == "__main__":
    main()
