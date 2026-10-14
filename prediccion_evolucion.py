#!/usr/bin/env python3

"""
ClimaAR - Predicción de evolución radar.

Modelo híbrido:

- Movimiento reciente ponderado.
- Regresión lineal de trayectoria.
- Estabilidad de rumbo.
- Estabilidad de velocidad.
- Aceleración.
- Evolución del área.
- Evolución de intensidad.
- Incertidumbre adaptativa según horizonte.
- Validación contra observaciones reales.
- Validación por proximidad si cambia el track_id.
- Relleno de huecos radar.
- Intervalo real tomado del histórico de radar.

No modifica:
- radar
- ecos estáticos
- detección de núcleos
- seguimiento temporal
"""

from __future__ import annotations

import json
import math

from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median


# ============================================================
# RUTAS
# ============================================================

BASE = Path("data/radar/seguimiento")

ESTADO = BASE / "estado.json"

INPUT = BASE / "seguimiento_temporal.json"

OUTPUT = BASE / "prediccion_evolucion.json"

PENDING = BASE / "predicciones_pendientes.json"

RADAR_HISTORY = Path("data/radar/historico")


# ============================================================
# CONFIGURACIÓN
# ============================================================

HORIZONTES_MIN = (
    5,
    10,
    20,
    30,
)

PASO_HUECO_MIN = 5

TOLERANCIA_HUECO_MIN = 12

MAX_PENDIENTES = 500

MIN_MATCH_PIXELS = 12.0

MAX_MATCH_EXTRA_UNCERTAINTY = 1.75

MAX_HISTORY = 8


# ============================================================
# UTILIDADES
# ============================================================

def load_json(
    path: Path,
    default,
):
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


def save_json(
    path: Path,
    data,
):
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
        text = str(
            value
        ).strip()

        if text.endswith("Z"):
            text = (
                text[:-1]
                + "+00:00"
            )

        dt = datetime.fromisoformat(
            text
        )

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
        dt.astimezone(
            timezone.utc
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


def canonical_ts(value):
    dt = parse_ts(
        value
    )

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


def number(
    value,
    default=0.0,
):
    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def track_id(track):
    value = track.get(
        "track_id",
        track.get(
            "id"
        ),
    )

    if value is None:
        return None

    return str(value)


# ============================================================
# DIRECCIÓN
# ============================================================

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

    index = int(
        (
            degrees % 360
        )
        / 22.5
        + 0.5
    ) % 16

    return names[index]


def angle_diff(
    a,
    b,
):

    if (
        a is None
        or b is None
    ):
        return None

    return abs(
        (
            a
            - b
            + 180
        )
        % 360
        - 180
    )


# ============================================================
# HISTORIAL
# ============================================================

def valid_history(track):

    history = []

    raw_history = track.get(
        "historial",
        [],
    )

    if not isinstance(
        raw_history,
        list,
    ):
        return history

    for item in raw_history:

        if not isinstance(
            item,
            dict,
        ):
            continue

        if (
            item.get("x")
            is None
            or item.get("y")
            is None
        ):
            continue

        ts = parse_ts(
            item.get(
                "timestamp"
            )
        )

        if ts is None:
            continue

        history.append(
            dict(item)
        )

    history.sort(
        key=lambda item:
            parse_ts(
                item[
                    "timestamp"
                ]
            )
    )

    return history


def recent_history(
    track,
    limit=MAX_HISTORY,
):

    return valid_history(
        track
    )[-limit:]


# ============================================================
# RADAR REAL
# ============================================================

def latest_radar_timestamp():

    if not RADAR_HISTORY.exists():
        return None

    timestamps = []

    for path in RADAR_HISTORY.glob(
        "*.png"
    ):

        name = path.name

        if (
            len(name) != 18
            or not name.endswith(
                ".png"
            )
        ):
            continue

        stamp = name[:-4]

        if not stamp.isdigit():
            continue

        try:

            dt = datetime.strptime(
                stamp,
                "%Y%m%d%H%M%S",
            ).replace(
                tzinfo=timezone.utc
            )

            timestamps.append(
                dt
            )

        except ValueError:
            continue

    if not timestamps:
        return None

    return max(
        timestamps
    )


def previous_radar_timestamp(
    tracks,
    current_ts,
):

    current = parse_ts(
        current_ts
    )

    if (
        current is None
        or not RADAR_HISTORY.exists()
    ):
        return None

    timestamps = []

    for path in RADAR_HISTORY.glob(
        "*.png"
    ):

        name = path.name

        if (
            len(name) != 18
            or not name.endswith(
                ".png"
            )
        ):
            continue

        stamp = name[:-4]

        if not stamp.isdigit():
            continue

        try:

            dt = datetime.strptime(
                stamp,
                "%Y%m%d%H%M%S",
            ).replace(
                tzinfo=timezone.utc
            )

        except ValueError:
            continue

        if dt < current:
            timestamps.append(
                dt
            )

    if not timestamps:
        return None

    return max(
        timestamps
    )


def real_radar_interval_minutes(
    tracks,
    current_ts,
):

    current = parse_ts(
        current_ts
    )

    previous = (
        previous_radar_timestamp(
            tracks,
            current_ts,
        )
    )

    if (
        current is None
        or previous is None
    ):
        return 0.0

    minutes = (
        current
        - previous
    ).total_seconds() / 60.0

    return round(
        max(
            0.0,
            minutes,
        ),
        2,
    )


# ============================================================
# MOVIMIENTO
# ============================================================

def movement_samples(
    history,
    max_interval=60.0,
):

    result = []

    for a, b in zip(
        history,
        history[1:],
    ):

        ta = parse_ts(
            a.get(
                "timestamp"
            )
        )

        tb = parse_ts(
            b.get(
                "timestamp"
            )
        )

        if (
            ta is None
            or tb is None
        ):
            continue

        dt = (
            tb - ta
        ).total_seconds() / 60.0

        if (
            dt < 2
            or dt > max_interval
        ):
            continue

        dx = (
            number(
                b["x"]
            )
            - number(
                a["x"]
            )
        )

        dy = (
            number(
                b["y"]
            )
            - number(
                a["y"]
            )
        )

        speed = (
            math.hypot(
                dx,
                dy,
            )
            / dt
        )

        bearing = None

        if speed > 0.001:

            bearing = (
                math.degrees(
                    math.atan2(
                        dx,
                        -dy,
                    )
                )
                % 360
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


def weighted_mean(
    values,
    weights,
):

    if not values:
        return 0.0

    total = sum(
        weights
    )

    if total <= 0:
        return mean(
            values
        )

    return (
        sum(
            value * weight
            for value, weight
            in zip(
                values,
                weights,
            )
        )
        / total
    )


def linear_regression_velocity(
    history,
):

    if len(history) < 3:
        return None

    first_ts = parse_ts(
        history[0][
            "timestamp"
        ]
    )

    times = []

    xs = []

    ys = []

    for item in history:

        dt = (
            parse_ts(
                item[
                    "timestamp"
                ]
            )
            - first_ts
        ).total_seconds() / 60.0

        times.append(
            dt
        )

        xs.append(
            number(
                item["x"]
            )
        )

        ys.append(
            number(
                item["y"]
            )
        )

    t_mean = mean(
        times
    )

    denominator = sum(
        (
            t - t_mean
        ) ** 2
        for t in times
    )

    if denominator <= 1e-9:
        return None

    x_mean = mean(
        xs
    )

    y_mean = mean(
        ys
    )

    vx = (
        sum(
            (
                t - t_mean
            )
            * (
                x - x_mean
            )
            for t, x
            in zip(
                times,
                xs,
            )
        )
        / denominator
    )

    vy = (
        sum(
            (
                t - t_mean
            )
            * (
                y - y_mean
            )
            for t, y
            in zip(
                times,
                ys,
            )
        )
        / denominator
    )

    return (
        vx,
        vy,
    )


# ============================================================
# ESTIMACIÓN DEL MOVIMIENTO
# ============================================================

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

    if (
        not history
        or not movements
    ):
        return None

    weights = list(
        range(
            1,
            len(movements)
            + 1,
        )
    )

    recent_vx = weighted_mean(
        [
            item["vx"]
            for item in movements
        ],
        weights,
    )

    recent_vy = weighted_mean(
        [
            item["vy"]
            for item in movements
        ],
        weights,
    )

    regression = (
        linear_regression_velocity(
            history[
                -min(
                    6,
                    len(history),
                ):
            ]
        )
    )

    if regression is not None:

        reg_vx, reg_vy = (
            regression
        )

        vx = (
            0.60
            * recent_vx
            + 0.40
            * reg_vx
        )

        vy = (
            0.60
            * recent_vy
            + 0.40
            * reg_vy
        )

    else:

        vx = recent_vx

        vy = recent_vy

    speed = math.hypot(
        vx,
        vy,
    )

    bearing = None

    if speed > 0.001:

        bearing = (
            math.degrees(
                math.atan2(
                    vx,
                    -vy,
                )
            )
            % 360
        )

    # --------------------------------------------------------
    # ACELERACIÓN
    # --------------------------------------------------------

    acceleration_samples = []

    for a, b in zip(
        movements,
        movements[1:],
    ):

        dt = max(
            (
                a["dt"]
                + b["dt"]
            )
            / 2.0,
            1.0,
        )

        acceleration_samples.append(
            (
                b["speed"]
                - a["speed"]
            )
            / dt
        )

    if acceleration_samples:

        acceleration = median(
            acceleration_samples
        )

    else:

        acceleration = number(
            track.get(
                "aceleracion_pixeles_min2",
                0.0,
            )
        )

    # --------------------------------------------------------
    # ESTABILIDAD DE RUMBO
    # --------------------------------------------------------

    turns = []

    for index in range(
        1,
        len(movements),
    ):

        value = angle_diff(
            movements[index][
                "bearing"
            ],
            movements[
                index - 1
            ][
                "bearing"
            ],
        )

        if value is not None:
            turns.append(
                value
            )

    if turns:

        direction_stability = clamp(
            100.0
            - (
                mean(turns)
                * 0.70
            )
        )

    else:

        direction_stability = 100.0

    # --------------------------------------------------------
    # ESTABILIDAD DE VELOCIDAD
    # --------------------------------------------------------

    speeds = [
        item["speed"]
        for item in movements
    ]

    speed_mean = (
        mean(speeds)
        if speeds
        else 0.0
    )

    if speed_mean > 0.01:

        variance = mean(
            (
                speed
                - speed_mean
            ) ** 2
            for speed in speeds
        )

        std = math.sqrt(
            variance
        )

        coefficient = (
            std
            / speed_mean
        )

        speed_stability = clamp(
            100.0
            - coefficient
            * 100.0
        )

    else:

        speed_stability = 0.0

    # --------------------------------------------------------
    # ERROR DE REGRESIÓN
    # --------------------------------------------------------

    regression_residual = 0.0

    if regression is not None:

        reg_vx, reg_vy = (
            regression
        )

        first_ts = parse_ts(
            history[0][
                "timestamp"
            ]
        )

        x0 = number(
            history[0]["x"]
        )

        y0 = number(
            history[0]["y"]
        )

        residuals = []

        for item in history:

            t = (
                parse_ts(
                    item[
                        "timestamp"
                    ]
                )
                - first_ts
            ).total_seconds() / 60.0

            predicted_x = (
                x0
                + reg_vx * t
            )

            predicted_y = (
                y0
                + reg_vy * t
            )

            residuals.append(
                math.hypot(
                    number(
                        item["x"]
                    )
                    - predicted_x,
                    number(
                        item["y"]
                    )
                    - predicted_y,
                )
            )

        if residuals:

            regression_residual = (
                mean(
                    residuals
                )
            )

    # --------------------------------------------------------
    # CONFIANZA DEL MOVIMIENTO
    # --------------------------------------------------------

    track_confidence = track.get(
        "confianza_movimiento",
        track.get(
            "confianza",
            50.0,
        ),
    )

    try:

        track_confidence = float(
            track_confidence
        )

    except (
        TypeError,
        ValueError,
    ):

        track_confidence = 50.0

    # Si viene normalizada 0-1,
    # convertir a porcentaje.

    if (
        0.0
        <= track_confidence
        <= 1.0
    ):

        track_confidence *= 100.0

    history_factor = min(
        100.0,
        55.0
        + len(movements)
        * 7.5,
    )

    residual_score = clamp(
        100.0
        - regression_residual
        * 2.0
    )

    confidence = clamp(
        0.30
        * track_confidence
        + 0.25
        * direction_stability
        + 0.20
        * speed_stability
        + 0.15
        * history_factor
        + 0.10
        * residual_score
    )

    # --------------------------------------------------------
    # EVOLUCIÓN DEL ÁREA
    # --------------------------------------------------------

    area_values = []

    for item in history:

        value = item.get(
            "cambio_area_porcentaje"
        )

        if value is None:
            continue

        try:

            area_values.append(
                float(value)
            )

        except (
            TypeError,
            ValueError,
        ):

            pass

    if area_values:

        area_trend = mean(
            area_values[-4:]
        )

    else:

        area_trend = None

    # --------------------------------------------------------
    # EVOLUCIÓN DE INTENSIDAD
    # --------------------------------------------------------

    intensity_values = []

    for item in history:

        value = item.get(
            "intensidad"
        )

        if value is None:
            continue

        try:

            intensity_values.append(
                float(value)
            )

        except (
            TypeError,
            ValueError,
        ):

            pass

    intensity_trend = None

    if len(
        intensity_values
    ) >= 2:

        intensity_trend = (
            intensity_values[-1]
            - intensity_values[-2]
        )

    # --------------------------------------------------------
    # INTENSIFICACIÓN
    # --------------------------------------------------------

    intensification = 50.0

    if area_trend is not None:

        intensification += clamp(
            area_trend * 0.8,
            -30.0,
            30.0,
        )

    if intensity_trend is not None:

        intensification += clamp(
            intensity_trend * 1.5,
            -20.0,
            20.0,
        )

    intensification += clamp(
        acceleration * 250.0,
        -15.0,
        15.0,
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
        "intensity_trend": intensity_trend,
        "intensification": intensification,
        "weakening":
            100.0
            - intensification,
        "history_points":
            len(history),
        "direction_stability":
            direction_stability,
        "speed_stability":
            speed_stability,
        "regression_residual":
            regression_residual,
        "last": history[-1],
    }


# ============================================================
# PROYECCIÓN
# ============================================================

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

    if (
        speed > 0.01
        and acceleration is not None
    ):

        ux = vx / speed

        uy = vy / speed

        delta_v = (
            float(
                acceleration
            )
            * horizon
        )

        delta_v = max(
            -speed * 0.40,
            min(
                speed * 0.40,
                delta_v,
            ),
        )

        vx_future = (
            vx
            + ux * delta_v
        )

        vy_future = (
            vy
            + uy * delta_v
        )

    else:

        vx_future = vx

        vy_future = vy

    x_future = (
        number(x0)
        + vx_future
        * horizon
    )

    y_future = (
        number(y0)
        + vy_future
        * horizon
    )

    future_speed = math.hypot(
        vx_future,
        vy_future,
    )

    future_bearing = None

    if future_speed > 0.001:

        future_bearing = (
            math.degrees(
                math.atan2(
                    vx_future,
                    -vy_future,
                )
            )
            % 360
        )

    base_error = max(
        2.0,
        (
            100.0
            - motion["confidence"]
        )
        * 0.08,
    )

    uncertainty = (
        base_error
        + horizon
        * max(
            0.08,
            0.28
            - motion["confidence"]
            / 500.0,
        )
    )

    return {
        "x_estimado":
            round(
                x_future,
                2,
            ),

        "y_estimado":
            round(
                y_future,
                2,
            ),

        "velocidad_estimada_pixeles_min":
            round(
                future_speed,
                4,
            ),

        "rumbo_estimado_grados":
            (
                round(
                    future_bearing,
                    1,
                )
                if future_bearing
                is not None
                else None
            ),

        "direccion_estimada":
            sector(
                future_bearing
            ),

        "incertidumbre_pixeles":
            round(
                uncertainty,
                2,
            ),
    }


# ============================================================
# CONFIANZA SEGÚN HORIZONTE
# ============================================================

def confidence_for_horizon(
    motion,
    horizon,
):

    penalties = {
        5: 0.0,
        10: 2.0,
        20: 7.0,
        30: 14.0,
    }

    penalty = penalties.get(
        horizon,
        max(
            0.0,
            horizon - 5.0,
        ) * 0.7,
    )

    return round(
        clamp(
            motion["confidence"]
            - penalty
        ),
        1,
    )


# ============================================================
# VALIDACIÓN
# ============================================================

def validate_point(
    prediction,
    actual,
    modo="MISMO_TRACK",
):

    dx = (
        number(
            actual["x"]
        )
        - number(
            prediction[
                "x_estimado"
            ]
        )
    )

    dy = (
        number(
            actual["y"]
        )
        - number(
            prediction[
                "y_estimado"
            ]
        )
    )

    return {
        "estado":
            "VALIDADA",

        "modo":
            modo,

        "timestamp_real":
            canonical_ts(
                actual[
                    "timestamp"
                ]
            ),

        "x_real":
            round(
                number(
                    actual["x"]
                ),
                2,
            ),

        "y_real":
            round(
                number(
                    actual["y"]
                ),
                2,
            ),

        "track_id_real":
            (
                str(
                    actual[
                        "track_id"
                    ]
                )
                if actual.get(
                    "track_id"
                )
                is not None
                else None
            ),

        "error_pixeles":
            round(
                math.hypot(
                    dx,
                    dy,
                ),
                2,
            ),

        "error_x_pixeles":
            round(
                dx,
                2,
            ),

        "error_y_pixeles":
            round(
                dy,
                2,
            ),
    }


def build_observation_index(
    tracks,
):

    observations = {}

    for track in tracks:

        tid = track_id(
            track
        )

        for obs in valid_history(
            track
        ):

            ts = canonical_ts(
                obs.get(
                    "timestamp"
                )
            )

            if ts is None:
                continue

            item = dict(
                obs
            )

            item[
                "track_id"
            ] = tid

            observations[
                (
                    tid,
                    ts,
                )
            ] = item

    return observations


def all_observations_at(
    tracks,
    timestamp,
):

    target = canonical_ts(
        timestamp
    )

    result = []

    if target is None:
        return result

    for track in tracks:

        tid = track_id(
            track
        )

        for obs in valid_history(
            track
        ):

            if (
                canonical_ts(
                    obs.get(
                        "timestamp"
                    )
                )
                != target
            ):
                continue

            item = dict(
                obs
            )

            item[
                "track_id"
            ] = tid

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

    px = number(
        prediction[
            "x_estimado"
        ]
    )

    py = number(
        prediction[
            "y_estimado"
        ]
    )

    uncertainty = number(
        prediction.get(
            "incertidumbre_pixeles",
            0.0,
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
                number(
                    obs["x"]
                )
                - px,
                number(
                    obs["y"]
                )
                - py,
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

        if (
            distance
            > max_distance
        ):
            continue

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


def validate_pending(
    current_ts,
    tracks,
    pending,
):

    now = parse_ts(
        current_ts
    )

    observations = (
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

        actual = observations.get(
            (
                tid,
                target,
            )
        )

        if actual is not None:

            prediction[
                "validacion"
            ] = validate_point(
                prediction,
                actual,
                "MISMO_TRACK",
            )

            validated.append(
                prediction
            )

            continue

        candidates = (
            all_observations_at(
                tracks,
                target,
            )
        )

        actual = (
            cross_track_match(
                prediction,
                candidates,
            )
        )

        if actual is not None:

            prediction[
                "validacion"
            ] = validate_point(
                prediction,
                actual,
                (
                    "TRACK_REASIGNADO_"
                    "POR_PROXIMIDAD"
                ),
            )

            validated.append(
                prediction
            )

            continue

        remaining.append(
            prediction
        )

    return (
        remaining,
        validated,
    )


# ============================================================
# HUECOS
# ============================================================

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
        start is None
        or end is None
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

    x0 = number(
        origin["x"]
    )

    y0 = number(
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

        target_dt = (
            start.timestamp()
            + horizon * 60.0
        )

        projection = project(
            motion,
            x0,
            y0,
            horizon,
        )

        result.append(
            {
                "track_id":
                    tid,

                "timestamp_origen":
                    canonical_ts(
                        start_ts
                    ),

                "timestamp_objetivo":
                    iso(
                        datetime.fromtimestamp(
                            target_dt,
                            tz=timezone.utc,
                        )
                    ),

                "horizonte_desde_origen_minutos":
                    horizon,

                "tipo":
                    "ESTIMACION_HUECO",

                **projection,

                "confianza_prediccion":
                    confidence_for_horizon(
                        motion,
                        horizon,
                    ),
            }
        )

        horizon += (
            PASO_HUECO_MIN
        )

    projection = project(
        motion,
        x0,
        y0,
        gap,
    )

    result.append(
        {
            "track_id":
                tid,

            "timestamp_origen":
                canonical_ts(
                    start_ts
                ),

            "timestamp_objetivo":
                canonical_ts(
                    end_ts
                ),

            "horizonte_desde_origen_minutos":
                round(
                    gap,
                    2,
                ),

            "tipo":
                "PREDICCION_CIERRE_HUECO",

            **projection,

            "confianza_prediccion":
                confidence_for_horizon(
                    motion,
                    gap,
                ),
        }
    )

    return result


# ============================================================
# PREDICCIÓN DE CADA TRACK
# ============================================================

def predict_track(track):

    tid = track_id(
        track
    )

    history = recent_history(
        track,
        MAX_HISTORY,
    )

    motion = estimate_motion(
        track,
        history,
    )

    if motion is None:

        return (
            {
                "track_id":
                    tid,

                "pronostico_disponible":
                    False,

                "motivo":
                    "SIN_MOVIMIENTO_VALIDO",
            },
            [],
        )

    current = motion[
        "last"
    ]

    forecasts = []

    for horizon in HORIZONTES_MIN:

        forecasts.append(
            {
                "horizonte_minutos":
                    horizon,

                **project(
                    motion,
                    current["x"],
                    current["y"],
                    horizon,
                ),

                "confianza_prediccion":
                    confidence_for_horizon(
                        motion,
                        horizon,
                    ),
            }
        )

    # --------------------------------------------------------
    # HUECOS
    # --------------------------------------------------------

    gap_predictions = []

    full_history = valid_history(
        track
    )

    for a, b in zip(
        full_history,
        full_history[1:],
    ):

        ta = parse_ts(
            a["timestamp"]
        )

        tb = parse_ts(
            b["timestamp"]
        )

        if (
            ta is None
            or tb is None
        ):
            continue

        gap = (
            tb - ta
        ).total_seconds() / 60.0

        if (
            gap
            <= TOLERANCIA_HUECO_MIN
        ):
            continue

        history_before = [
            item
            for item in full_history
            if (
                parse_ts(
                    item[
                        "timestamp"
                    ]
                )
                <= ta
            )
        ][-MAX_HISTORY:]

        before_motion = (
            estimate_motion(
                track,
                history_before,
            )
        )

        if before_motion is None:
            continue

        points = (
            make_gap_predictions(
                track,
                before_motion,
                a["timestamp"],
                b["timestamp"],
            )
        )

        for point in points:

            if (
                point["tipo"]
                != "PREDICCION_CIERRE_HUECO"
            ):
                continue

            point[
                "validacion"
            ] = validate_point(
                point,
                {
                    **b,
                    "track_id":
                        tid,
                },
                "CIERRE_HUECO",
            )

        gap_predictions.extend(
            points
        )

    return (
        {
            "track_id":
                tid,

            "activo":
                bool(
                    track.get(
                        "activo",
                        True,
                    )
                ),

            "pronostico_disponible":
                True,

            "posicion_actual":
                {
                    "x":
                        round(
                            number(
                                current["x"]
                            ),
                            2,
                        ),

                    "y":
                        round(
                            number(
                                current["y"]
                            ),
                            2,
                        ),
                },

            "velocidad_actual_pixeles_min":
                round(
                    motion["speed"],
                    4,
                ),

            "rumbo_actual_grados":
                (
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

            "direccion_actual":
                sector(
                    motion[
                        "bearing"
                    ]
                ),

            "aceleracion_pixeles_min2":
                (
                    round(
                        number(
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

            "tendencia_area_porcentaje":
                (
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

            "probabilidad_intensificacion":
                round(
                    motion[
                        "intensification"
                    ],
                    1,
                ),

            "probabilidad_debilitamiento":
                round(
                    motion[
                        "weakening"
                    ],
                    1,
                ),

            "confianza_prediccion":
                round(
                    motion[
                        "confidence"
                    ],
                    1,
                ),

            "estabilidad_direccion":
                round(
                    motion[
                        "direction_stability"
                    ],
                    1,
                ),

            "estabilidad_velocidad":
                round(
                    motion[
                        "speed_stability"
                    ],
                    1,
                ),

            "error_regresion_pixeles":
                round(
                    motion[
                        "regression_residual"
                    ],
                    2,
                ),

            "puntos_historial_utilizados":
                motion[
                    "history_points"
                ],

            "horizontes":
                forecasts,

            "huecos_detectados":
                len(
                    gap_predictions
                ),

            "estimaciones_hueco":
                gap_predictions,
        },
        gap_predictions,
    )


# ============================================================
# TRACKS
# ============================================================

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
        dict(track)
        for track in tracks
        if isinstance(
            track,
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
                dict(track)
                for track in fallback
                if isinstance(
                    track,
                    dict,
                )
            ]

    return tracks


# ============================================================
# MAIN
# ============================================================

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

        latest = (
            latest_radar_timestamp()
        )

        if latest is not None:

            radar_ts = iso(
                latest
            )

    if radar_ts is None:

        raise RuntimeError(
            "No se pudo determinar "
            "el timestamp del radar."
        )

    tracks = get_tracks(
        estado,
        temporal,
    )

    # --------------------------------------------------------
    # INTERVALO REAL
    # --------------------------------------------------------

    intervalo = (
        real_radar_interval_minutes(
            tracks,
            radar_ts,
        )
    )

    # --------------------------------------------------------
    # PENDIENTES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PREDICCIONES
    # --------------------------------------------------------

    predictions = []

    new_pending = []

    existing_keys = {
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

    gap_count = 0

    for track in tracks:

        if (
            len(
                valid_history(
                    track
                )
            )
            < 2
        ):
            continue

        (
            prediction,
            gap_predictions,
        ) = predict_track(
            track
        )

        predictions.append(
            prediction
        )

        # ----------------------------------------------------
        # HUECOS
        # ----------------------------------------------------

        for point in gap_predictions:

            gap_count += 1

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

            if key in existing_keys:
                continue

            new_pending.append(
                point
            )

            existing_keys.add(
                key
            )

        # ----------------------------------------------------
        # PREDICCIONES NORMALES
        # ----------------------------------------------------

        if not prediction.get(
            "pronostico_disponible"
        ):
            continue

        origin_dt = parse_ts(
            radar_ts
        )

        if origin_dt is None:
            continue

        for forecast in prediction.get(
            "horizontes",
            [],
        ):

            horizon = number(
                forecast.get(
                    "horizonte_minutos"
                )
            )

            target_dt = (
                origin_dt
                + (
                    datetime.min
                    - datetime.min
                )
            )

            target_dt = (
                datetime.fromtimestamp(
                    origin_dt.timestamp()
                    + horizon
                    * 60.0,
                    tz=timezone.utc,
                )
            )

            target_iso = iso(
                target_dt
            )

            key = (
                str(
                    prediction.get(
                        "track_id"
                    )
                ),
                target_iso,
            )

            if key in existing_keys:
                continue

            new_pending.append(
                {
                    "track_id":
                        str(
                            prediction.get(
                                "track_id"
                            )
                        ),

                    "timestamp_origen":
                        radar_ts,

                    "timestamp_objetivo":
                        target_iso,

                    "horizonte_desde_origen_minutos":
                        horizon,

                    "tipo":
                        "PREDICCION_NORMAL",

                    "x_estimado":
                        forecast.get(
                            "x_estimado"
                        ),

                    "y_estimado":
                        forecast.get(
                            "y_estimado"
                        ),

                    "velocidad_estimada_pixeles_min":
                        forecast.get(
                            "velocidad_estimada_pixeles_min"
                        ),

                    "rumbo_estimado_grados":
                        forecast.get(
                            "rumbo_estimado_grados"
                        ),

                    "direccion_estimada":
                        forecast.get(
                            "direccion_estimada"
                        ),

                    "incertidumbre_pixeles":
                        forecast.get(
                            "incertidumbre_pixeles"
                        ),

                    "confianza_prediccion":
                        forecast.get(
                            "confianza_prediccion"
                        ),
                }
            )

            existing_keys.add(
                key
            )

    pending.extend(
        new_pending
    )

    pending = pending[
        -MAX_PENDIENTES:
    ]

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    available = [
        item
        for item in predictions
        if item.get(
            "pronostico_disponible"
        )
    ]

    errors = []

    for item in validated_now:

        value = (
            item.get(
                "validacion",
                {},
            ).get(
                "error_pixeles"
            )
        )

        if value is not None:

            errors.append(
                number(
                    value
                )
            )

    for prediction in predictions:

        for point in prediction.get(
            "estimaciones_hueco",
            [],
        ):

            value = (
                point.get(
                    "validacion",
                    {},
                ).get(
                    "error_pixeles"
                )
            )

            if value is not None:

                errors.append(
                    number(
                        value
                    )
                )

    summary = {
        "nucleos_con_prediccion":
            len(
                available
            ),

        "nucleos_sin_prediccion":
            len(
                predictions
            )
            - len(
                available
            ),

        "promedio_intensificacion":
            (
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

        "promedio_debilitamiento":
            (
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

        "promedio_confianza":
            (
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

        "huecos_estimados":
            gap_count,

        "predicciones_validadas_en_este_frame":
            len(
                validated_now
            ),

        "error_medio_pixeles_validado":
            (
                round(
                    mean(
                        errors
                    ),
                    2,
                )
                if errors
                else None
            ),
    }

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    result = {
        "schema_version":
            4,

        "radar_timestamp":
            radar_ts,

        "intervalo_minutos":
            intervalo,

        "tipo":
            "PREDICCION_MOVIMIENTO_CON_HUECOS",

        "fuente_tracks":
            str(
                ESTADO
            ),

        "prediccion_global_disponible":
            bool(
                available
            ),

        "nucleos":
            predictions,

        "resumen":
            summary,

        "validaciones":
            validated_now[
                -100:
            ],

        "nota":
            (
                "Predicción híbrida basada "
                "en movimiento reciente "
                "ponderado, regresión lineal, "
                "estabilidad de rumbo, "
                "estabilidad de velocidad, "
                "aceleración y evolución "
                "del núcleo. "
                "Las posiciones estimadas "
                "nunca se etiquetan como "
                "radar observado."
            ),
    }

    save_json(
        OUTPUT,
        result,
    )

    save_json(
        PENDING,
        {
            "schema_version":
                3,

            "actualizado":
                radar_ts,

            "pendientes":
                pending,
        },
    )

    print(
        f"OK: {OUTPUT}"
    )

    print(
        "Tracks analizados: "
        f"{len(tracks)}"
    )

    print(
        "Predicciones: "
        f"{len(predictions)}"
    )

    print(
        "Huecos estimados: "
        f"{gap_count}"
    )

    print(
        "Predicciones validadas: "
        f"{len(validated_now)}"
    )

    print(
        "Pendientes: "
        f"{len(pending)}"
    )


if __name__ == "__main__":
    main()
