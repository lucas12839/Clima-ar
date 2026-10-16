#!/usr/bin/env python3
"""
ClimaAR - Predicción de evolución y validación temporal.

- Predicciones normales: 5/10/20/30 minutos.
- Reconstrucción de huecos históricos sin contaminación temporal.
- Las estimaciones de hueco NO se agregan a pendientes de validación.
- Las pendientes antiguas expiran.
- Nunca inventa una observación radar.
"""

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

MAX_PENDIENTES = 200
MAX_PENDING_AGE_MIN = 45

MIN_MATCH_PIXELS = 12.0
MAX_MATCH_EXTRA_UNCERTAINTY = 1.75


def load_json(path: Path, default):
    if not path.exists():
        return default

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
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
            (degrees % 360) / 22.5
            + 0.5
        ) % 16
    ]


def angle_diff(a, b):
    if a is None or b is None:
        return None

    return abs(
        (a - b + 180) % 360 - 180
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

        history.append(
            dict(item)
        )

    history.sort(
        key=lambda x: parse_ts(
            x["timestamp"]
        )
    )

    return history


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
        ta = parse_ts(
            a.get("timestamp")
        )

        tb = parse_ts(
            b.get("timestamp")
        )

        if not ta or not tb:
            continue

        dt = (
            tb - ta
        ).total_seconds() / 60.0

        if dt < 2 or dt > max_interval:
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
            math.hypot(dx, dy)
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
    """
    Calcula el movimiento únicamente
    con el historial suministrado.

    Al reconstruir un hueco histórico,
    no utiliza información posterior
    al comienzo de ese hueco.
    """

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

    total_weight = sum(weights)

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
            movements[i - 1]["bearing"],
        )

        if value is not None:
            turns.append(value)

    consistency = clamp(
        100.0
        - (
            mean(turns) * 0.25
            if turns
            else 0.0
        )
    )

    confidence_source = dict(track)

    confidence_source.update(
        history[-1]
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
                area_trend * 0.8,
            ),
        )

    if acceleration is not None:
        intensification += max(
            -15.0,
            min(
                15.0,
                acceleration * 300.0,
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
            if future_bearing is not None
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


def validate_point(
    prediction,
    actual,
    modo="MISMO_TRACK",
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
            ) is not None
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


def build_observation_index(
    tracks
):
    observations = {}

    for track in tracks:
        tid = track_id(track)

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

            item = dict(obs)

            item["track_id"] = tid

            observations[
                (tid, ts)
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
        tid = track_id(track)

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

            item = dict(obs)

            item["track_id"] = tid

            result.append(item)

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

        if distance > max_distance:
            continue

        if (
            best is None
            or distance < best[0]
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
    expired = []

    if now is None:
        return (
            pending,
            validated,
            expired,
        )

    for prediction in pending:

        target = canonical_ts(
            prediction.get(
                "timestamp_objetivo"
            )
        )

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

        if actual is None:
            actual = cross_track_match(
                prediction,
                all_observations_at(
                    tracks,
                    target,
                ),
            )

        if actual is not None:

            actual_tid = str(
                actual.get(
                    "track_id"
                )
            )

            prediction[
                "validacion"
            ] = validate_point(
                prediction,
                actual,
                (
                    "MISMO_TRACK"
                    if actual_tid == tid
                    else
                    "TRACK_REASIGNADO_POR_PROXIMIDAD"
                ),
            )

            validated.append(
                prediction
            )

            continue

        age = (
            now - target_dt
        ).total_seconds() / 60.0

        if age > MAX_PENDING_AGE_MIN:

            prediction[
                "estado_validacion"
            ] = (
                "EXPIRADA_SIN_OBSERVACION_EXACTA"
            )

            prediction[
                "motivo"
            ] = (
                "No apareció una "
                "observación radar "
                "con el timestamp "
                "exacto dentro de "
                "la ventana de "
                "validación."
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
    )


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

    if not start or not end:
        return []

    gap = (
        end - start
    ).total_seconds() / 60.0

    if gap <= TOLERANCIA_HUECO_MIN:
        return []

    origin = motion["last"]

    x0 = float(
        origin["x"]
    )

    y0 = float(
        origin["y"]
    )

    tid = track_id(track)

    result = []

    horizon = PASO_HUECO_MIN

    while horizon < gap:

        target_dt = datetime.fromtimestamp(
            start.timestamp()
            + horizon * 60.0,
            tz=timezone.utc,
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

        horizon += PASO_HUECO_MIN

    return result


def forecast_track(track):
    history = recent_history(
        track
    )

    motion = estimate_motion(
        track,
        history,
    )

    if motion is None:
        return None

    forecasts = []

    x0 = float(
        motion["last"]["x"]
    )

    y0 = float(
        motion["last"]["y"]
    )

    for horizon in HORIZONTES_MIN:

        forecasts.append(
            {
                "horizonte_minutos": horizon,
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

    else:
        tracks = []

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
    ) = validate_pending(
        current_ts,
        tracks,
        old_pending,
    )

    nuclei_output = []

    for track in tracks:

        result = forecast_track(
            track
        )

        if result is None:
            continue

        motion, forecasts = result

        tid = track_id(
            track
        )

        nucleus_output = {
            "track_id": tid,
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

        history = valid_history(
            track
        )

        if len(history) < 2:
            continue

        for a, b in zip(
            history,
            history[1:],
        ):

            ta = parse_ts(
                a["timestamp"]
            )

            tb = parse_ts(
                b["timestamp"]
            )

            if not ta or not tb:
                continue

            gap = (
                tb - ta
            ).total_seconds() / 60.0

            if gap <= TOLERANCIA_HUECO_MIN:
                continue

            # IMPORTANTE:
            # Para reconstruir un hueco solamente se utiliza
            # información disponible antes del comienzo del hueco.
            history_before = [
                item
                for item in history
                if (
                    parse_ts(
                        item[
                            "timestamp"
                        ]
                    )
                    <= ta
                )
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
                    "inicio": iso(ta),
                    "fin": iso(tb),
                    "duracion_minutos": round(
                        gap,
                        1,
                    ),
                    "estimaciones": estimates,
                }
            )

            # ====================================================
            # IMPORTANTE:
            #
            # NO agregar estimates a predicciones_pendientes.
            #
            # Estas predicciones reconstruyen un hueco HISTÓRICO.
            # El timestamp objetivo ya quedó en el pasado y no
            # debemos esperar un radar futuro con ese timestamp.
            # ====================================================

    # ============================================================
    # PENDIENTES
    #
    # Solo permanecen las pendientes que ya existían y que todavía
    # pueden ser validadas.
    # ============================================================

    unique = {}

    for prediction in pending:

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
            str(
                prediction.get(
                    "tipo",
                    "",
                )
            ),
        )

        unique[key] = prediction

    pending = list(
        unique.values()
    )

    pending.sort(
        key=lambda item:
        parse_ts(
            item.get(
                "timestamp_objetivo"
            )
        )
        or datetime.min.replace(
            tzinfo=timezone.utc
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
    # SALIDA
    # ============================================================

    output = {
        "radar_timestamp": canonical_ts(
            current_ts
        ),
        "prediccion_global_disponible": bool(
            nuclei_output
        ),
        "nucleos": nuclei_output,
        "validaciones": validations,
        "validaciones_count": len(
            validations
        ),
        "predicciones_generadas": total_forecasts,
        "predicciones_hueco_generadas": (
            total_gap_predictions
        ),
        "predicciones_pendientes": len(
            pending
        ),
        "predicciones_expiradas": len(
            expired
        ),
        "expiradas": expired,
    }

    save_json(
        OUTPUT,
        output,
    )

    print(
        "Predicción evolución disponible:",
        bool(nuclei_output),
    )

    print(
        "Núcleos con predicción:",
        len(nuclei_output),
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
        len(validations),
    )

    print(
        "Pendientes vigentes:",
        len(pending),
    )

    print(
        "Pendientes expiradas:",
        len(expired),
    )


def main():
    process()


if __name__ == "__main__":
    main()
