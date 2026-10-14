#!/usr/bin/env python3
"""
ClimaAR - Predicción de evolución y relleno de huecos radar.

- Usa data/radar/seguimiento/estado.json como fuente de tracks.
- Usa el seguimiento temporal actual sin modificarlo.
- Genera pronósticos a 5/10/20/30 minutos.
- Genera estimaciones cada 5 minutos dentro de huecos > 12 min.
- Tolera huecos > 30 min aunque seguimiento_temporal.py haya creado
  un nuevo track después del hueco.
- Valida contra observaciones reales del timestamp exacto.
- Si el track_id cambió después de un hueco, busca una observación
  real cercana al punto previsto en ese mismo timestamp.
- Aumenta la incertidumbre con el horizonte.
- Nunca etiqueta una estimación como radar observado.
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
MAX_PENDIENTES = 500

MIN_MATCH_PIXELS = 12.0
MAX_MATCH_EXTRA_UNCERTAINTY = 1.75


def load_json(path: Path, default):
    if not path.exists():
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return default


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
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
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.astimezone(timezone.utc)

    except (TypeError, ValueError):
        return None


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_ts(value):
    dt = parse_ts(value)
    return iso(dt) if dt else None


def clamp(value, lo=0.0, hi=100.0):
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return lo


def track_id(track):
    value = track.get("track_id", track.get("id"))
    return str(value) if value is not None else None


def sector(degrees):
    if degrees is None:
        return None

    names = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSO", "SO", "OSO",
        "O", "ONO", "NO", "NNO",
    ]

    return names[int((degrees % 360) / 22.5 + 0.5) % 16]


def angle_diff(a, b):
    if a is None or b is None:
        return None

    return abs((a - b + 180) % 360 - 180)


def valid_history(track):
    history = []

    for item in track.get("historial", []):
        if not isinstance(item, dict):
            continue

        if item.get("x") is None or item.get("y") is None:
            continue

        ts = parse_ts(item.get("timestamp"))

        if ts is None:
            continue

        history.append(dict(item))

    history.sort(
        key=lambda x: parse_ts(x["timestamp"])
    )

    return history


def recent_history(track, limit=8):
    return valid_history(track)[-limit:]


def previous_radar_timestamp(tracks, current_ts):
    """
    Busca el último timestamp real observado antes del radar actual.
    No utiliza intervalo_minutos de seguimiento_temporal.json.
    """

    current = parse_ts(current_ts)

    if current is None:
        return None

    timestamps = []

    for track in tracks:
        for item in valid_history(track):
            ts = parse_ts(item.get("timestamp"))

            if ts is None:
                continue

            if ts < current:
                timestamps.append(ts)

    if not timestamps:
        return None

    return max(timestamps)


def real_radar_interval_minutes(tracks, current_ts):
    """
    Calcula el intervalo real entre el radar actual y el radar
    observado inmediatamente anterior.
    """

    current = parse_ts(current_ts)

    if current is None:
        return 0.0

    previous = previous_radar_timestamp(
        tracks,
        current_ts,
    )

    if previous is None:
        return 0.0

    minutes = (
        current - previous
    ).total_seconds() / 60.0

    if minutes < 0:
        return 0.0

    return round(minutes, 2)


def movement_samples(history, max_interval=60.0):
    result = []

    for a, b in zip(history, history[1:]):
        ta = parse_ts(a.get("timestamp"))
        tb = parse_ts(b.get("timestamp"))

        if not ta or not tb:
            continue

        dt = (
            tb - ta
        ).total_seconds() / 60.0

        if dt < 2 or dt > max_interval:
            continue

        dx = float(b["x"]) - float(a["x"])
        dy = float(b["y"]) - float(a["y"])

        speed = math.hypot(dx, dy) / dt

        bearing = (
            math.degrees(
                math.atan2(dx, -dy)
            ) % 360
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


def estimate_motion(track, history=None):
    history = (
        history
        if history is not None
        else recent_history(track)
    )

    movements = movement_samples(history)

    if not history or not movements:
        return None

    weights = list(
        range(1, len(movements) + 1)
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

    speed = math.hypot(vx, vy)

    bearing = (
        math.degrees(
            math.atan2(vx, -vy)
        ) % 360
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
            (a["dt"] + b["dt"]) / 2.0,
            1.0,
        )

        acceleration = (
            b["speed"] - a["speed"]
        ) / avg_dt

    turns = []

    for i in range(1, len(movements)):
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

    movement_confidence = track.get(
        "calidad_movimiento",
        track.get(
            "confianza_movimiento",
            track.get("confianza", 50),
        ),
    )

    try:
        movement_confidence = float(
            movement_confidence
        )
    except (TypeError, ValueError):
        movement_confidence = 50.0

    confidence = clamp(
        0.60 * movement_confidence
        + 0.25 * consistency
        + 15.0
        * min(len(movements), 3)
        / 3.0
    )

    area_changes = []

    for item in history[1:]:
        value = item.get(
            "cambio_area_porcentaje"
        )

        if value is not None:
            try:
                area_changes.append(
                    float(value)
                )
            except (TypeError, ValueError):
                pass

    area_trend = (
        mean(area_changes[-4:])
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
                float(acceleration) * 300.0,
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
        "weakening": 100.0 - intensification,
        "history_points": len(history),
        "last": history[-1],
    }


def project(motion, x0, y0, horizon):
    vx = motion["vx"]
    vy = motion["vy"]
    speed = motion["speed"]
    acceleration = motion["acceleration"]

    vx_future = vx
    vy_future = vy

    if (
        acceleration is not None
        and abs(float(acceleration)) < 0.15
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

        vx_future += ux * delta_v
        vy_future += uy * delta_v

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
        ) % 360
        if future_speed > 0.001
        else None
    )

    base_error = max(
        2.0,
        (100.0 - motion["confidence"])
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
            prediction["x_estimado"]
        )
    )

    dy = (
        float(actual["y"])
        - float(
            prediction["y_estimado"]
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
            str(actual.get("track_id"))
            if actual.get("track_id")
            is not None
            else None
        ),
        "error_pixeles": round(
            math.hypot(dx, dy),
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


def build_observation_index(tracks):
    observations = {}

    for track in tracks:
        tid = track_id(track)

        for obs in valid_history(track):
            ts = canonical_ts(
                obs.get("timestamp")
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
    target = canonical_ts(timestamp)
    result = []

    if target is None:
        return result

    for track in tracks:
        tid = track_id(track)

        for obs in valid_history(track):
            if (
                canonical_ts(
                    obs.get("timestamp")
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
        prediction["x_estimado"]
    )

    py = float(
        prediction["y_estimado"]
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
                float(obs["x"]) - px,
                float(obs["y"]) - py,
            )
        except (TypeError, ValueError):
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
    now = parse_ts(current_ts)

    observations = build_observation_index(
        tracks
    )

    remaining = []
    validated = []

    for prediction in pending:
        target = canonical_ts(
            prediction.get(
                "timestamp_objetivo"
            )
        )

        if not target or not now:
            remaining.append(
                prediction
            )
            continue

        target_dt = parse_ts(target)

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
            (tid, target)
        )

        if actual is not None:
            prediction["validacion"] = (
                validate_point(
                    prediction,
                    actual,
                    "MISMO_TRACK",
                )
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

        actual = cross_track_match(
            prediction,
            candidates,
        )

        if actual is not None:
            prediction["validacion"] = (
                validate_point(
                    prediction,
                    actual,
                    "TRACK_REASIGNADO_POR_PROXIMIDAD",
                )
            )

            validated.append(
                prediction
            )

            continue

        remaining.append(
            prediction
        )

    return remaining, validated


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
    start = parse_ts(start_ts)
    end = parse_ts(end_ts)

    if not start or not end:
        return []

    gap = (
        end - start
    ).total_seconds() / 60.0

    if gap <= TOLERANCIA_HUECO_MIN:
        return []

    origin = motion["last"]

    x0 = float(origin["x"])
    y0 = float(origin["y"])

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
                "timestamp_origen": canonical_ts(
                    start_ts
                ),
                "timestamp_objetivo": iso(
                    target_dt
                ),
                "horizonte_desde_origen_minutos": horizon,
                "tipo": "ESTIMACION_HUECO",
                **projection,
                "confianza_prediccion":
                    confidence_for_horizon(
                        motion,
                        horizon,
                    ),
            }
        )

        horizon += PASO_HUECO_MIN

    projection = project(
        motion,
        x0,
        y0,
        gap,
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
            "horizonte_desde_origen_minutos":
                round(gap, 2),
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


def predict_track(track):
    tid = track_id(track)

    history = recent_history(
        track,
        8,
    )

    motion = estimate_motion(
        track,
        history,
    )

    if motion is None:
        return (
            {
                "track_id": tid,
                "pronostico_disponible": False,
                "motivo":
                    "SIN_MOVIMIENTO_VALIDO",
            },
            [],
        )

    current = motion["last"]

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

        if not ta or not tb:
            continue

        gap = (
            tb - ta
        ).total_seconds() / 60.0

        if gap <= TOLERANCIA_HUECO_MIN:
            continue

        history_before = [
            item
            for item in full_history
            if parse_ts(
                item["timestamp"]
            ) <= ta
        ][-8:]

        before_motion = estimate_motion(
            track,
            history_before,
        )

        if before_motion is None:
            continue

        points = make_gap_predictions(
            track,
            before_motion,
            a["timestamp"],
            b["timestamp"],
        )

        for point in points:
            if (
                point["tipo"]
                != "PREDICCION_CIERRE_HUECO"
            ):
                continue

            point["validacion"] = (
                validate_point(
                    point,
                    {
                        **b,
                        "track_id": tid,
                    },
                    "CIERRE_HUECO",
                )
            )

        gap_predictions.extend(
            points
        )

    return (
        {
            "track_id": tid,
            "activo": bool(
                track.get(
                    "activo",
                    True,
                )
            ),
            "pronostico_disponible": True,
            "posicion_actual": {
                "x": round(
                    float(current["x"]),
                    2,
                ),
                "y": round(
                    float(current["y"]),
                    2,
                ),
            },
            "velocidad_actual_pixeles_min":
                round(
                    motion["speed"],
                    4,
                ),
            "rumbo_actual_grados": (
                round(
                    motion["bearing"],
                    1,
                )
                if motion["bearing"]
                is not None
                else None
            ),
            "direccion_actual":
                sector(
                    motion["bearing"]
                ),
            "aceleracion_pixeles_min2": (
                round(
                    float(
                        motion["acceleration"]
                    ),
                    5,
                )
                if motion["acceleration"]
                is not None
                else None
            ),
            "tendencia_area_porcentaje": (
                round(
                    motion["area_trend"],
                    2,
                )
                if motion["area_trend"]
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
                    motion["weakening"],
                    1,
                ),
            "confianza_prediccion":
                round(
                    motion["confidence"],
                    1,
                ),
            "puntos_historial_utilizados":
                motion["history_points"],
            "horizontes": forecasts,
            "huecos_detectados":
                len(gap_predictions),
            "estimaciones_hueco":
                gap_predictions,
        },
        gap_predictions,
    )


def get_tracks(
    estado,
    temporal,
):
    tracks = estado.get(
        "tracks",
        []
    )

    if not isinstance(
        tracks,
        list,
    ):
        tracks = []

    tracks = [
        dict(t)
        for t in tracks
        if isinstance(t, dict)
    ]

    if not tracks:
        fallback = temporal.get(
            "tracks",
            []
        )

        if isinstance(
            fallback,
            list,
        ):
            tracks = [
                dict(t)
                for t in fallback
                if isinstance(t, dict)
            ]

    return tracks


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
        or temporal.get(
            "radar_timestamp"
        )
    )

    tracks = get_tracks(
        estado,
        temporal,
    )

    # CORRECCIÓN:
    # El intervalo se calcula exclusivamente usando
    # timestamps reales de radar presentes en estado.json.
    # No se usa el intervalo_minutos de seguimiento_temporal.json,
    # porque puede quedar arrastrado de un estado anterior.
    intervalo = real_radar_interval_minutes(
        tracks,
        radar_ts,
    )

    pending_data = load_json(
        PENDING,
        {},
    )

    pending = (
        pending_data.get(
            "pendientes",
            [],
        )
        if isinstance(
            pending_data,
            dict,
        )
        else []
    )

    if not isinstance(
        pending,
        list,
    ):
        pending = []

    pending, validated_now = (
        validate_pending(
            radar_ts,
            tracks,
            pending,
        )
    )

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
        if len(
            valid_history(track)
        ) < 2:
            continue

        prediction, gap_predictions = (
            predict_track(track)
        )

        predictions.append(
            prediction
        )

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

    pending.extend(
        new_pending
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

    for item in validated_now:
        value = item.get(
            "validacion",
            {},
        ).get(
            "error_pixeles"
        )

        if value is not None:
            try:
                errors.append(
                    float(value)
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

    for prediction in predictions:
        for point in prediction.get(
            "estimaciones_hueco",
            [],
        ):
            value = point.get(
                "validacion",
                {},
            ).get(
                "error_pixeles"
            )

            if value is not None:
                try:
                    errors.append(
                        float(value)
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

    summary = {
        "nucleos_con_prediccion":
            len(available),

        "nucleos_sin_prediccion":
            len(predictions)
            - len(available),

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

        "huecos_estimados":
            gap_count,

        "predicciones_validadas_en_este_frame":
            len(validated_now),

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
        "schema_version": 4,
        "radar_timestamp":
            radar_ts,

        "intervalo_minutos":
            intervalo,

        "tipo":
            "PREDICCION_MOVIMIENTO_CON_HUECOS",

        "fuente_tracks":
            str(ESTADO),

        "prediccion_global_disponible":
            bool(available),

        "nucleos":
            predictions,

        "resumen":
            summary,

        "validaciones":
            validated_now[-100:],

        "nota": (
            "Las posiciones estimadas son "
            "extrapolaciones del seguimiento "
            "temporal. Nunca se etiquetan como "
            "radar observado. Los huecos mayores "
            "de 30 minutos pueden cambiar de "
            "track_id; por eso la validación de "
            "cierre también puede realizarse por "
            "proximidad en el timestamp real exacto."
        ),
    }

    save_json(
        OUTPUT,
        result,
    )

    save_json(
        PENDING,
        {
            "schema_version": 3,
            "actualizado": radar_ts,
            "pendientes": pending,
        },
    )

    print(
        f"OK: {OUTPUT}"
    )

    print(
        f"Tracks analizados: {len(tracks)}"
    )

    print(
        f"Predicciones: {len(predictions)}"
    )

    print(
        f"Huecos estimados: {gap_count}"
    )

    print(
        "Predicciones validadas: "
        f"{len(validated_now)}"
    )

    print(
        f"Pendientes: {len(pending)}"
    )


if __name__ == "__main__":
    main()
