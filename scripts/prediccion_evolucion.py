#!/usr/bin/env python3
"""
ClimaAR - Predicción de evolución y validación temporal.

Diseño:
- Genera pronósticos 5/10/20/30 min desde cada radar aceptado.
- Guarda esos pronósticos como pendientes futuras.
- Valida pendientes contra observaciones reales del historial.
- Si el radar no tiene exactamente el timestamp objetivo,
  interpola la posición real entre los dos frames que rodean
  el objetivo.
- Permite reasignación espacial cuando el track original cambió.
- Las predicciones históricas RELLENO_HUECO no entran en pendientes.
- Elimina pendientes antiguas heredadas del sistema anterior.
- Mantiene las estimaciones de hueco únicamente como análisis.
- Expone campos compatibles con el resumen actual del workflow.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone, timedelta
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

MAX_PENDIENTES = 1000
MAX_PENDING_AGE_MIN = 90

MIN_MATCH_PIXELS = 12.0
MAX_MATCH_EXTRA_UNCERTAINTY = 1.75

MAX_INTERPOLATION_GAP_MIN = 30.0


def load_json(path: Path, default):
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


def save_json(path: Path, data):
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
        dt.astimezone(timezone.utc)
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
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
            (degrees % 360) / 22.5
            + 0.5
        ) % 16
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
            item.get(
                "timestamp"
            )
        )

        if ts is None:
            continue

        value = dict(item)

        value["_dt"] = ts

        history.append(
            value
        )

    history.sort(
        key=lambda x: x["_dt"]
    )

    return history


def clean_observation(item):
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

        if (
            dt < 2
            or dt > max_interval
        ):
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
    history = (
        history
        if history is not None
        else recent_history(track)
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
            len(movements) + 1,
        )
    )

    total_weight = sum(
        weights
    )

    vx = sum(
        item["vx"] * weight
        for item, weight in zip(
            movements,
            weights,
        )
    ) / total_weight

    vy = sum(
        item["vy"] * weight
        for item, weight in zip(
            movements,
            weights,
        )
    ) / total_weight

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

    acceleration = None

    if len(movements) >= 2:

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
            movements[
                i - 1
            ]["bearing"],
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

    confidence_source = dict(
        track
    )

    confidence_source.update(
        clean_observation(
            history[-1]
        )
    )

    movement_confidence = 50.0

    for key in (
        "calidad_movimiento",
        "confianza_movimiento",
        "confianza",
    ):

        value = confidence_source.get(
            key
        )

        if value is None:
            continue

        try:
            movement_confidence = float(
                value
            )
            break

        except (
            TypeError,
            ValueError,
        ):
            pass

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

        try:
            if value is not None:
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
                area_trend
                * 0.8,
            ),
        )

    if acceleration is not None:

        intensification += max(
            -15.0,
            min(
                15.0,
                acceleration
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
        "last": history[-1],
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
    acceleration = motion[
        "acceleration"
    ]

    vx_future = vx
    vy_future = vy

    if (
        acceleration is not None
        and abs(
            float(
                acceleration
            )
        ) < 0.15
        and speed > 0.01
    ):

        ux = vx / speed
        uy = vy / speed

        delta_v = max(
            -speed * 0.5,
            min(
                speed * 0.5,
                float(
                    acceleration
                )
                * horizon,
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
        + vx_future
        * horizon
    )

    y_future = (
        float(y0)
        + vy_future
        * horizon
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
    }


def confidence_for_horizon(
    motion,
    horizon,
):
    return round(
        clamp(
            motion["confidence"]
            - max(
                0.0,
                horizon - 5.0,
            )
            * 1.2
        ),
        1,
    )


def build_observation_index(
    tracks
):
    observations = {}

    for track in tracks:

        tid = track_id(
            track
        )

        for obs in valid_history(
            track
        ):

            ts = iso(
                obs["_dt"]
            )

            item = clean_observation(
                obs
            )

            item["track_id"] = tid

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

    if not target:
        return []

    result = []

    for track in tracks:

        tid = track_id(
            track
        )

        for obs in valid_history(
            track
        ):

            if (
                iso(obs["_dt"])
                != target
            ):
                continue

            item = clean_observation(
                obs
            )

            item["track_id"] = tid

            result.append(
                item
            )

    return result


def nearest_observation_at(
    tracks,
    timestamp,
    prediction,
):
    observations = all_observations_at(
        tracks,
        timestamp,
    )

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
            and (
                best is None
                or distance
                < best[0]
            )
        ):

            best = (
                distance,
                obs,
            )

    if best is None:
        return None

    return best[1]


def interpolate_track_observation(
    track,
    target_dt,
):
    history = valid_history(
        track
    )

    if len(history) < 2:
        return None

    before = None
    after = None

    for obs in history:

        if obs["_dt"] <= target_dt:
            before = obs

        if obs["_dt"] >= target_dt:

            after = obs

            break

    if (
        before is None
        or after is None
    ):
        return None

    if (
        before["_dt"]
        == after["_dt"]
    ):

        return clean_observation(
            before
        )

    gap = (
        after["_dt"]
        - before["_dt"]
    ).total_seconds() / 60.0

    if (
        gap <= 0
        or gap
        > MAX_INTERPOLATION_GAP_MIN
    ):
        return None

    alpha = (
        target_dt
        - before["_dt"]
    ).total_seconds() / (
        after["_dt"]
        - before["_dt"]
    ).total_seconds()

    x = (
        float(before["x"])
        + (
            float(after["x"])
            - float(before["x"])
        )
        * alpha
    )

    y = (
        float(before["y"])
        + (
            float(after["y"])
            - float(before["y"])
        )
        * alpha
    )

    return {
        "x": x,
        "y": y,
        "timestamp": iso(
            target_dt
        ),
        "track_id": track_id(
            track
        ),
        "interpolado": True,
        "timestamp_anterior": iso(
            before["_dt"]
        ),
        "timestamp_siguiente": iso(
            after["_dt"]
        ),
    }


def make_validation(
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
        "timestamp_objetivo": canonical_ts(
            prediction[
                "timestamp_objetivo"
            ]
        ),
        "timestamp_real": canonical_ts(
            actual["timestamp"]
        ),
        "track_id_predicho": str(
            prediction.get(
                "track_id"
            )
        ),
        "track_id_real": (
            str(
                actual.get(
                    "track_id"
                )
            )
            if actual.get(
                "track_id"
            ) is not None
            else None
        ),
        "x_real": round(
            float(actual["x"]),
            2,
        ),
        "y_real": round(
            float(actual["y"]),
            2,
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
        "interpolado": bool(
            actual.get(
                "interpolado",
                False,
            )
        ),
    }


def validate_prediction(
    prediction,
    tracks,
):
    target_dt = parse_ts(
        prediction.get(
            "timestamp_objetivo"
        )
    )

    if target_dt is None:
        return None

    original_tid = str(
        prediction.get(
            "track_id"
        )
    )

    # ============================================================
    # 1. MISMO TRACK + INTERPOLACIÓN
    # ============================================================

    for track in tracks:

        if (
            track_id(track)
            != original_tid
        ):
            continue

        actual = interpolate_track_observation(
            track,
            target_dt,
        )

        if actual is not None:

            return make_validation(
                prediction,
                actual,
                "MISMO_TRACK_INTERPOLADO",
            )

    # ============================================================
    # 2. MISMO TIMESTAMP REAL
    # ============================================================

    exact_timestamp = iso(
        target_dt
    )

    actual = nearest_observation_at(
        tracks,
        exact_timestamp,
        prediction,
    )

    if actual is not None:

        return make_validation(
            prediction,
            actual,
            (
                "MISMO_TRACK_EXACTO"
                if str(
                    actual.get(
                        "track_id"
                    )
                )
                == original_tid
                else
                "TRACK_REASIGNADO_EXACTO"
            ),
        )

    # ============================================================
    # 3. REASIGNACIÓN + INTERPOLACIÓN
    # ============================================================

    candidates = []

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
            0.0,
        )
    )

    max_distance = max(
        MIN_MATCH_PIXELS,
        uncertainty
        * MAX_MATCH_EXTRA_UNCERTAINTY,
    )

    for track in tracks:

        actual = interpolate_track_observation(
            track,
            target_dt,
        )

        if actual is None:
            continue

        try:

            distance = math.hypot(
                float(actual["x"])
                - px,
                float(actual["y"])
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

            candidates.append(
                (
                    distance,
                    actual,
                )
            )

    if candidates:

        candidates.sort(
            key=lambda item:
            item[0]
        )

        return make_validation(
            prediction,
            candidates[0][1],
            "TRACK_REASIGNADO_INTERPOLADO",
        )

    return None


def validate_pending(
    current_ts,
    tracks,
    pending,
):
    now = parse_ts(
        current_ts
    )

    if now is None:
        return (
            pending,
            [],
            [],
            0,
        )

    remaining = []
    validated = []
    expired = []

    legacy_removed = 0

    for prediction in pending:

        tipo = str(
            prediction.get(
                "tipo",
                "",
            )
        ).upper()

        # ========================================================
        # ELIMINAR DEFINITIVAMENTE LAS PENDIENTES VIEJAS
        # DE TIPO RELLENO_HUECO.
        # ========================================================

        if (
            tipo
            == "RELLENO_HUECO"
        ):

            legacy_removed += 1

            continue

        target_dt = parse_ts(
            prediction.get(
                "timestamp_objetivo"
            )
        )

        if target_dt is None:

            legacy_removed += 1

            continue

        # Todavía no llegó el momento
        # de validar esta predicción.

        if target_dt > now:

            remaining.append(
                prediction
            )

            continue

        validation = validate_prediction(
            prediction,
            tracks,
        )

        if validation is not None:

            prediction[
                "validacion"
            ] = validation

            prediction[
                "estado_validacion"
            ] = "VALIDADA"

            validated.append(
                prediction
            )

            continue

        age = (
            now - target_dt
        ).total_seconds() / 60.0

        if (
            age
            > MAX_PENDING_AGE_MIN
        ):

            prediction[
                "estado_validacion"
            ] = (
                "EXPIRADA_SIN_OBSERVACION"
            )

            prediction[
                "motivo"
            ] = (
                "No fue posible "
                "encontrar una "
                "observación real "
                "compatible dentro "
                "de la ventana."
            )

            expired.append(
                prediction
            )

        else:

            remaining.append(
                prediction
            )

    return (
        remaining,
        validated,
        expired,
        legacy_removed,
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

    horizon = PASO_HUECO_MIN

    while horizon < gap:

        target_dt = (
            start
            + timedelta(
                minutes=horizon
            )
        )

        projection = project(
            motion,
            x0,
            y0,
            horizon,
        )

        result.append(
            {
                "track_id": tid,
                "timestamp_origen": iso(
                    start
                ),
                "timestamp_objetivo": iso(
                    target_dt
                ),
                "horizonte_minutos": horizon,
                "tipo": "RELLENO_HUECO",
                **projection,
                "confianza": confidence_for_horizon(
                    motion,
                    horizon,
                ),
                "intensificacion_probable": round(
                    motion[
                        "intensification"
                    ],
                    1,
                ),
                "debilitamiento_probable": round(
                    motion[
                        "weakening"
                    ],
                    1,
                ),
                "validacion": None,
            }
        )

        horizon += (
            PASO_HUECO_MIN
        )

    return result


def forecast_track(
    track,
    current_dt,
):
    history = recent_history(
        track
    )

    motion = estimate_motion(
        track,
        history,
    )

    if motion is None:
        return None

    x0 = float(
        motion["last"]["x"]
    )

    y0 = float(
        motion["last"]["y"]
    )

    tid = track_id(
        track
    )

    forecasts = []

    for horizon in HORIZONTES_MIN:

        target_dt = (
            current_dt
            + timedelta(
                minutes=horizon
            )
        )

        forecasts.append(
            {
                "track_id": tid,
                "timestamp_origen": iso(
                    current_dt
                ),
                "timestamp_objetivo": iso(
                    target_dt
                ),
                "horizonte_minutos": horizon,
                "tipo": "PREDICCION",
                **project(
                    motion,
                    x0,
                    y0,
                    horizon,
                ),
                "confianza": confidence_for_horizon(
                    motion,
                    horizon,
                ),
                "intensificacion_probable": round(
                    motion[
                        "intensification"
                    ],
                    1,
                ),
                "debilitamiento_probable": round(
                    motion[
                        "weakening"
                    ],
                    1,
                ),
                "validacion": None,
            }
        )

    return (
        motion,
        forecasts,
    )


def process():

    estado = load_json(
        ESTADO,
        {},
    )

    seguimiento = load_json(
        INPUT,
        {},
    )

    tracks = []

    if isinstance(
        estado,
        dict,
    ):

        tracks = estado.get(
            "tracks",
            estado.get(
                "nucleos",
                [],
            ),
        )

    if (
        not tracks
        and isinstance(
            seguimiento,
            dict,
        )
    ):

        tracks = seguimiento.get(
            "tracks",
            seguimiento.get(
                "nucleos",
                [],
            ),
        )

    if not isinstance(
        tracks,
        list,
    ):

        tracks = []

    current_ts = None

    for source in (
        seguimiento,
        estado,
    ):

        if not isinstance(
            source,
            dict,
        ):
            continue

        current_ts = (
            current_ts
            or source.get(
                "radar_timestamp"
            )
            or source.get(
                "ultimo_radar_timestamp"
            )
            or source.get(
                "timestamp"
            )
        )

    timestamp_file = (
        BASE.parent
        / "radar_timestamp.txt"
    )

    if timestamp_file.exists():

        value = (
            timestamp_file
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )

        if value:
            current_ts = value

    current_dt = parse_ts(
        current_ts
    )

    if current_dt is None:

        print(
            "ERROR: timestamp radar inválido."
        )

        save_json(
            OUTPUT,
            {
                "prediccion_global_disponible": False,
                "error": (
                    "TIMESTAMP_RADAR_INVALIDO"
                ),
            },
        )

        return

    # ============================================================
    # CARGAR PENDIENTES
    # ============================================================

    old_pending = load_json(
        PENDING,
        [],
    )

    if isinstance(
        old_pending,
        dict,
    ):

        old_pending = old_pending.get(
            "predicciones",
            old_pending.get(
                "pendientes",
                [],
            ),
        )

    if not isinstance(
        old_pending,
        list,
    ):

        old_pending = []

    (
        pending,
        validations,
        expired,
        legacy_removed,
    ) = validate_pending(
        current_dt,
        tracks,
        old_pending,
    )

    nuclei_output = []

    new_predictions = []

    # ============================================================
    # GENERAR PREDICCIONES
    # ============================================================

    for track in tracks:

        result = forecast_track(
            track,
            current_dt,
        )

        if result is None:
            continue

        motion, forecasts = result

        nucleus_output = {
            "track_id": track_id(
                track
            ),
            "historial_puntos": motion[
                "history_points"
            ],
            "movimiento": {
                "vx": round(
                    motion["vx"],
                    4,
                ),
                "vy": round(
                    motion["vy"],
                    4,
                ),
                "velocidad_pixeles_min": round(
                    motion["speed"],
                    4,
                ),
                "rumbo_grados": (
                    round(
                        motion["bearing"],
                        1,
                    )
                    if motion[
                        "bearing"
                    ] is not None
                    else None
                ),
                "direccion": sector(
                    motion["bearing"]
                ),
                "aceleracion_pixeles_min2": (
                    round(
                        motion[
                            "acceleration"
                        ],
                        6,
                    )
                    if motion[
                        "acceleration"
                    ] is not None
                    else None
                ),
                "confianza": round(
                    motion[
                        "confidence"
                    ],
                    1,
                ),
            },
            "predicciones": forecasts,
            "estimaciones_hueco": [],
        }

        nuclei_output.append(
            nucleus_output
        )

        # ========================================================
        # LAS PREDICCIONES NORMALES SÍ SE GUARDAN
        # COMO PENDIENTES.
        # ========================================================

        new_predictions.extend(
            forecasts
        )

        # ========================================================
        # RECONSTRUCCIÓN DE HUECOS HISTÓRICOS
        # ========================================================

        history = valid_history(
            track
        )

        if len(history) < 2:
            continue

        for a, b in zip(
            history,
            history[1:],
        ):

            ta = a["_dt"]
            tb = b["_dt"]

            gap = (
                tb - ta
            ).total_seconds() / 60.0

            if (
                gap
                <= TOLERANCIA_HUECO_MIN
            ):
                continue

            # Solo información anterior al hueco.
            history_before = [
                item
                for item in history
                if item["_dt"] <= ta
            ]

            motion_before = estimate_motion(
                track,
                history_before,
            )

            if motion_before is None:
                continue

            estimates = make_gap_predictions(
                track,
                motion_before,
                ta,
                tb,
            )

            if not estimates:
                continue

            nucleus_output[
                "estimaciones_hueco"
            ].append(
                {
                    "inicio": iso(
                        ta
                    ),
                    "fin": iso(
                        tb
                    ),
                    "duracion_minutos": round(
                        gap,
                        1,
                    ),
                    "estimaciones": estimates,
                }
            )

            # NO agregar estimates a pendientes.
            #
            # Son reconstrucciones HISTÓRICAS.
            # No esperamos que aparezca un radar futuro
            # con esos timestamps.

    # ============================================================
    # UNIFICAR PENDIENTES
    # ============================================================

    merged = (
        pending
        + new_predictions
    )

    unique = {}

    for prediction in merged:

        tipo = str(
            prediction.get(
                "tipo",
                "",
            )
        ).upper()

        # Nunca permitir que vuelva
        # una pendiente histórica.

        if (
            tipo
            == "RELLENO_HUECO"
        ):
            continue

        target = canonical_ts(
            prediction.get(
                "timestamp_objetivo"
            )
        )

        if target is None:
            continue

        key = (
            str(
                prediction.get(
                    "track_id"
                )
            ),
            target,
            int(
                prediction.get(
                    "horizonte_minutos",
                    0,
                )
            ),
        )

        unique[key] = prediction

    pending = list(
        unique.values()
    )

    pending.sort(
        key=lambda item:
        (
            parse_ts(
                item.get(
                    "timestamp_objetivo"
                )
            )
            or datetime.min.replace(
                tzinfo=timezone.utc
            )
        )
    )

    pending = pending[
        -MAX_PENDIENTES:
    ]

    save_json(
        PENDING,
        pending,
    )

    # ============================================================
    # ESTADÍSTICAS
    # ============================================================

    total_forecasts = sum(
        len(
            nucleus.get(
                "predicciones",
                [],
            )
        )
        for nucleus in nuclei_output
    )

    total_gap_predictions = sum(
        len(
            gap.get(
                "estimaciones",
                [],
            )
        )
        for nucleus in nuclei_output
        for gap in nucleus.get(
            "estimaciones_hueco",
            [],
        )
    )

    # ============================================================
    # COMPATIBILIDAD CON EL WORKFLOW ACTUAL
    #
    # El resumen actual busca:
    # d.get("predicciones", [])
    #
    # Por eso exponemos las predicciones también
    # en el nivel superior.
    # ============================================================

    all_predictions = [
        prediction
        for nucleus in nuclei_output
        for prediction in nucleus.get(
            "predicciones",
            [],
        )
    ]

    output = {
        "radar_timestamp": iso(
            current_dt
        ),
        "prediccion_global_disponible": bool(
            nuclei_output
        ),
        "nucleos": nuclei_output,

        "predicciones": all_predictions,

        "validaciones": validations,
        "validaciones_count": len(
            validations
        ),

        "predicciones_generadas": (
            total_forecasts
        ),

        "predicciones_hueco_generadas": (
            total_gap_predictions
        ),

        "predicciones_pendientes": len(
            pending
        ),

        "predicciones_expiradas": len(
            expired
        ),

        "pendientes_legacy_eliminadas": (
            legacy_removed
        ),

        "expiradas": expired,
    }

    save_json(
        OUTPUT,
        output,
    )

    # ============================================================
    # SALIDA DE CONSOLA
    # ============================================================

    print(
        "Predicción evolución disponible:",
        bool(nuclei_output),
    )

    print(
        "Núcleos con predicción:",
        len(
            nuclei_output
        ),
    )

    print(
        "Predicciones 5/10/20/30:",
        total_forecasts,
    )

    print(
        "Predicciones de hueco:",
        total_gap_predictions,
    )

    print(
        "Validaciones nuevas:",
        len(
            validations
        ),
    )

    print(
        "Pendientes vigentes:",
        len(
            pending
        ),
    )

    print(
        "Pendientes expiradas:",
        len(
            expired
        ),
    )

    print(
        "Pendientes legacy eliminadas:",
        legacy_removed,
    )


def main():
    process()


if __name__ == "__main__":
    main()
