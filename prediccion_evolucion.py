#!/usr/bin/env python3

"""
ClimaAR - Predicción de evolución radar.

Modelo híbrido y conservador:
- Usa el histórico REAL del radar como fuente de tiempo.
- Combina velocidad reciente y regresión de trayectoria.
- Calcula estabilidad de rumbo y velocidad.
- Estima aceleración solamente cuando hay evidencia suficiente.
- Proyecta con cinemática correcta (velocidad media durante el horizonte).
- Calcula tendencia de área/intensidad desde el historial real.
- Estima incertidumbre con backtesting interno del propio track.
- Valida predicciones contra observaciones posteriores.
- Evita validar por proximidad cuando la coincidencia es ambigua.
- Mantiene compatibilidad con schema_version 4.

No modifica:
- captura del radar
- máscara de ecos estáticos
- detección de núcleos
- seguimiento temporal
"""

from __future__ import annotations

import json
import math

from datetime import datetime, timedelta, timezone
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
CALIBRATION = BASE / "calibracion_prediccion.json"

RADAR_HISTORY = Path("data/radar/historico")


# ============================================================
# CONFIGURACIÓN
# ============================================================

HORIZONTES_MIN = (5, 10, 20, 30)

PASO_HUECO_MIN = 5
TOLERANCIA_HUECO_MIN = 12

MAX_PENDIENTES = 500
MAX_CALIBRATION = 1000
MAX_HISTORY = 10

MIN_MATCH_PIXELS = 10.0
MAX_MATCH_EXTRA_UNCERTAINTY = 1.35
MIN_MATCH_MARGIN = 2.0

MAX_SEGMENT_INTERVAL = 60.0

# No dejamos que una aceleración estimada cambie demasiado
# la velocidad durante un único horizonte.
MAX_SPEED_CHANGE_RATIO = 0.25


# ============================================================
# UTILIDADES
# ============================================================

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
    return (
        dt.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def canonical_ts(value):
    dt = parse_ts(value)
    return iso(dt) if dt is not None else None


def clamp(value, lo=0.0, hi=100.0):
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return lo


def number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def track_id(track):
    value = track.get("track_id", track.get("id"))
    return str(value) if value is not None else None


# ============================================================
# DIRECCIÓN
# ============================================================

def sector(degrees):
    if degrees is None:
        return None

    names = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSO", "SO", "OSO",
        "O", "ONO", "NO", "NNO",
    ]

    index = int((degrees % 360) / 22.5 + 0.5) % 16
    return names[index]


def angle_diff(a, b):
    if a is None or b is None:
        return None

    return abs((a - b + 180) % 360 - 180)


# ============================================================
# HISTORIAL
# ============================================================

def valid_history(track):
    history = []

    raw_history = track.get("historial", [])

    if not isinstance(raw_history, list):
        return history

    for item in raw_history:
        if not isinstance(item, dict):
            continue

        if item.get("x") is None or item.get("y") is None:
            continue

        ts = parse_ts(item.get("timestamp"))

        if ts is None:
            continue

        history.append(dict(item))

    history.sort(
        key=lambda item: parse_ts(item["timestamp"])
    )

    return history


def recent_history(track, limit=MAX_HISTORY):
    return valid_history(track)[-limit:]


# ============================================================
# RADAR REAL
# ============================================================

def radar_timestamps():
    if not RADAR_HISTORY.exists():
        return []

    result = []

    for path in RADAR_HISTORY.glob("*.png"):
        name = path.name

        if len(name) != 18 or not name.endswith(".png"):
            continue

        stamp = name[:-4]

        if not stamp.isdigit():
            continue

        try:
            dt = datetime.strptime(
                stamp,
                "%Y%m%d%H%M%S",
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        result.append(dt)

    return sorted(set(result))


def latest_radar_timestamp():
    timestamps = radar_timestamps()
    return max(timestamps) if timestamps else None


def previous_radar_timestamp(current_ts):
    current = parse_ts(current_ts)

    if current is None:
        return None

    previous = [
        item
        for item in radar_timestamps()
        if item < current
    ]

    return max(previous) if previous else None


def real_radar_interval_minutes(current_ts):
    current = parse_ts(current_ts)
    previous = previous_radar_timestamp(current_ts)

    if current is None or previous is None:
        return 0.0

    return round(
        max(
            0.0,
            (current - previous).total_seconds() / 60.0,
        ),
        2,
    )


# ============================================================
# MOVIMIENTO
# ============================================================

def movement_samples(history, max_interval=MAX_SEGMENT_INTERVAL):
    result = []

    for a, b in zip(history, history[1:]):
        ta = parse_ts(a.get("timestamp"))
        tb = parse_ts(b.get("timestamp"))

        if ta is None or tb is None:
            continue

        dt = (tb - ta).total_seconds() / 60.0

        if dt < 2.0 or dt > max_interval:
            continue

        dx = number(b["x"]) - number(a["x"])
        dy = number(b["y"]) - number(a["y"])

        speed = math.hypot(dx, dy) / dt

        bearing = None

        if speed > 0.001:
            bearing = (
                math.degrees(math.atan2(dx, -dy)) % 360
            )

        result.append(
            {
                "dt": dt,
                "vx": dx / dt,
                "vy": dy / dt,
                "speed": speed,
                "bearing": bearing,
                "timestamp": b["timestamp"],
            }
        )

    return result


def weighted_mean(values, weights):
    if not values:
        return 0.0

    total = sum(weights)

    if total <= 0:
        return mean(values)

    return sum(
        value * weight
        for value, weight in zip(values, weights)
    ) / total


def linear_regression(values_x, values_y):
    if len(values_x) < 2 or len(values_x) != len(values_y):
        return None

    x_mean = mean(values_x)
    y_mean = mean(values_y)

    denominator = sum(
        (x - x_mean) ** 2
        for x in values_x
    )

    if denominator <= 1e-9:
        return None

    slope = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(values_x, values_y)
    ) / denominator

    intercept = y_mean - slope * x_mean

    return slope, intercept


def regression_velocity(history):
    if len(history) < 3:
        return None

    base = parse_ts(history[0]["timestamp"])

    if base is None:
        return None

    times = []
    xs = []
    ys = []

    for item in history:
        ts = parse_ts(item["timestamp"])

        if ts is None:
            continue

        t = (ts - base).total_seconds() / 60.0

        times.append(t)
        xs.append(number(item["x"]))
        ys.append(number(item["y"]))

    rx = linear_regression(times, xs)
    ry = linear_regression(times, ys)

    if rx is None or ry is None:
        return None

    return rx[0], ry[0]


def speed_regression(movements):
    if len(movements) < 3:
        return None

    times = []
    speeds = []

    elapsed = 0.0

    for movement in movements:
        elapsed += movement["dt"]
        times.append(elapsed)
        speeds.append(movement["speed"])

    result = linear_regression(times, speeds)

    if result is None:
        return None

    slope, intercept = result

    predicted = [
        slope * t + intercept
        for t in times
    ]

    mean_speed = mean(speeds)

    ss_tot = sum(
        (value - mean_speed) ** 2
        for value in speeds
    )

    ss_res = sum(
        (actual - estimate) ** 2
        for actual, estimate in zip(speeds, predicted)
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 1e-9
        else 0.0
    )

    return slope, r2


# ============================================================
# BACKTEST INTERNO
# ============================================================

def simple_velocity(history):
    movements = movement_samples(history)

    if not movements:
        return None

    recent = movements[-min(4, len(movements)):]

    weights = list(range(1, len(recent) + 1))

    vx = weighted_mean(
        [item["vx"] for item in recent],
        weights,
    )

    vy = weighted_mean(
        [item["vy"] for item in recent],
        weights,
    )

    return vx, vy


def internal_backtest(history):
    """
    Predice cada punto histórico usando solamente los puntos anteriores.
    Esto permite estimar cuánto error produce el track antes de confiar
    demasiado en una proyección futura.
    """

    errors = []

    if len(history) < 4:
        return {
            "count": 0,
            "median": None,
            "p75": None,
            "mean": None,
        }

    for index in range(3, len(history)):
        previous = history[:index]

        velocity = simple_velocity(previous)

        if velocity is None:
            continue

        current = previous[-1]
        actual = history[index]

        current_ts = parse_ts(current["timestamp"])
        actual_ts = parse_ts(actual["timestamp"])

        if current_ts is None or actual_ts is None:
            continue

        horizon = (
            actual_ts - current_ts
        ).total_seconds() / 60.0

        if horizon <= 0 or horizon > MAX_SEGMENT_INTERVAL:
            continue

        vx, vy = velocity

        predicted_x = number(current["x"]) + vx * horizon
        predicted_y = number(current["y"]) + vy * horizon

        error = math.hypot(
            number(actual["x"]) - predicted_x,
            number(actual["y"]) - predicted_y,
        )

        errors.append(error)

    if not errors:
        return {
            "count": 0,
            "median": None,
            "p75": None,
            "mean": None,
        }

    ordered = sorted(errors)

    p75_index = min(
        len(ordered) - 1,
        max(0, math.ceil(len(ordered) * 0.75) - 1),
    )

    return {
        "count": len(errors),
        "median": round(median(errors), 2),
        "p75": round(ordered[p75_index], 2),
        "mean": round(mean(errors), 2),
    }


# ============================================================
# ESTIMACIÓN DEL MOVIMIENTO
# ============================================================

def history_area_trend(history):
    values = []

    for item in history:
        value = item.get("pixeles")

        if value is None:
            continue

        values.append(number(value))

    if len(values) < 2:
        return None

    changes = []

    for previous, current in zip(values, values[1:]):
        if previous <= 0:
            continue

        changes.append(
            (current - previous)
            / previous
            * 100.0
        )

    if not changes:
        return None

    return mean(changes[-4:])


def history_intensity_trend(history):
    values = []

    for item in history:
        value = item.get("intensidad")

        if value is None:
            continue

        values.append(number(value))

    if len(values) < 2:
        return None

    return mean(
        [
            current - previous
            for previous, current
            in zip(values[-5:], values[-4:])
        ]
    )


def estimate_motion(track, history=None):
    if history is None:
        history = recent_history(track)

    movements = movement_samples(history)

    if not history or not movements:
        return None

    recent = movements[-min(5, len(movements)):]

    weights = list(range(1, len(recent) + 1))

    recent_vx = weighted_mean(
        [item["vx"] for item in recent],
        weights,
    )

    recent_vy = weighted_mean(
        [item["vy"] for item in recent],
        weights,
    )

    regression = regression_velocity(
        history[-min(7, len(history)):]
    )

    turns = []

    for previous, current in zip(
        movements,
        movements[1:],
    ):
        value = angle_diff(
            current["bearing"],
            previous["bearing"],
        )

        if value is not None:
            turns.append(value)

    mean_turn = mean(turns) if turns else 0.0

    direction_stability = clamp(
        100.0 - mean_turn * 0.85
    )

    speeds = [item["speed"] for item in recent]

    speed_mean = mean(speeds) if speeds else 0.0

    if speed_mean > 0.01:
        variance = mean(
            (value - speed_mean) ** 2
            for value in speeds
        )
        speed_cv = math.sqrt(variance) / speed_mean
        speed_stability = clamp(
            100.0 - speed_cv * 100.0
        )
    else:
        speed_cv = 1.0
        speed_stability = 0.0

    # Si el rumbo es estable, la regresión de trayectoria es más útil.
    # Si el rumbo está cambiando, damos más peso al movimiento reciente.
    if regression is not None:
        reg_vx, reg_vy = regression

        if direction_stability >= 75.0:
            reg_weight = 0.60
        elif direction_stability >= 50.0:
            reg_weight = 0.45
        else:
            reg_weight = 0.30

        vx = (
            recent_vx * (1.0 - reg_weight)
            + reg_vx * reg_weight
        )

        vy = (
            recent_vy * (1.0 - reg_weight)
            + reg_vy * reg_weight
        )
    else:
        vx = recent_vx
        vy = recent_vy

    speed = math.hypot(vx, vy)

    bearing = None

    if speed > 0.001:
        bearing = math.degrees(
            math.atan2(vx, -vy)
        ) % 360

    # --------------------------------------------------------
    # ACELERACIÓN: solamente si la tendencia de velocidad
    # tiene evidencia suficiente. Se reduce deliberadamente
    # para evitar amplificar ruido radar.
    # --------------------------------------------------------

    speed_fit = speed_regression(recent)

    acceleration = 0.0
    speed_trend_r2 = 0.0

    if speed_fit is not None:
        slope, r2 = speed_fit
        speed_trend_r2 = max(0.0, r2)

        if r2 >= 0.35:
            acceleration = slope * 0.35

    # Si el rumbo es inestable, no extrapolamos aceleración.
    if direction_stability < 55.0:
        acceleration = 0.0

    # --------------------------------------------------------
    # BACKTEST
    # --------------------------------------------------------

    backtest = internal_backtest(history)

    backtest_median = backtest["median"]
    backtest_p75 = backtest["p75"]

    if backtest_median is None:
        backtest_score = 55.0
    else:
        backtest_score = clamp(
            100.0
            - backtest_median * 5.0
        )

    history_factor = clamp(
        45.0 + len(movements) * 8.0
    )

    track_confidence = number(
        track.get(
            "confianza_movimiento",
            track.get("confianza", 0.5),
        ),
        0.5,
    )

    if 0.0 <= track_confidence <= 1.0:
        track_confidence *= 100.0

    track_confidence = clamp(track_confidence)

    confidence = clamp(
        0.18 * track_confidence
        + 0.22 * direction_stability
        + 0.18 * speed_stability
        + 0.17 * history_factor
        + 0.25 * backtest_score
    )

    # La confianza no puede ser muy alta si el propio backtest
    # muestra errores grandes.
    if backtest_p75 is not None:
        if backtest_p75 > 18:
            confidence = min(confidence, 55.0)
        elif backtest_p75 > 12:
            confidence = min(confidence, 65.0)
        elif backtest_p75 > 8:
            confidence = min(confidence, 78.0)

    area_trend = history_area_trend(history)
    intensity_trend = history_intensity_trend(history)

    intensification = 50.0

    if area_trend is not None:
        intensification += clamp(
            area_trend * 0.8,
            -25.0,
            25.0,
        )

    if intensity_trend is not None:
        intensification += clamp(
            intensity_trend * 1.2,
            -15.0,
            15.0,
        )

    if acceleration > 0:
        intensification += clamp(
            acceleration * 150.0,
            0.0,
            10.0,
        )
    elif acceleration < 0:
        intensification += clamp(
            acceleration * 150.0,
            -10.0,
            0.0,
        )

    intensification = clamp(intensification)

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
        "weakening": 100.0 - intensification,
        "history_points": len(history),
        "direction_stability": direction_stability,
        "speed_stability": speed_stability,
        "speed_cv": speed_cv,
        "speed_trend_r2": speed_trend_r2,
        "mean_turn_degrees": mean_turn,
        "backtest_count": backtest["count"],
        "backtest_median_error": backtest_median,
        "backtest_p75_error": backtest_p75,
        "backtest_mean_error": backtest["mean"],
        "last": history[-1],
    }


# ============================================================
# PROYECCIÓN
# ============================================================

def project(motion, x0, y0, horizon):
    vx = number(motion["vx"])
    vy = number(motion["vy"])
    speed = max(0.0, number(motion["speed"]))
    acceleration = number(
        motion.get("acceleration", 0.0)
    )

    if speed > 0.001:
        ux = vx / speed
        uy = vy / speed
    else:
        ux = 0.0
        uy = 0.0

    # Cambio de velocidad durante el horizonte.
    delta_speed = acceleration * horizon

    max_delta = speed * MAX_SPEED_CHANGE_RATIO

    if max_delta > 0:
        delta_speed = max(
            -max_delta,
            min(max_delta, delta_speed),
        )
    else:
        delta_speed = 0.0

    future_speed = max(
        0.0,
        speed + delta_speed,
    )

    # Cinemática correcta:
    # distancia = v0*t + 1/2*a*t²
    # equivalente a velocidad media * tiempo.
    average_speed = (
        speed + future_speed
    ) / 2.0

    distance = average_speed * horizon

    x_future = number(x0) + ux * distance
    y_future = number(y0) + uy * distance

    future_bearing = None

    if future_speed > 0.001:
        future_bearing = (
            math.degrees(
                math.atan2(
                    ux * future_speed,
                    -uy * future_speed,
                )
            )
            % 360
        )

    # Incertidumbre basada en el comportamiento histórico del track.
    base_error = max(
        3.0,
        number(
            motion.get(
                "backtest_p75_error",
                0.0,
            ),
            0.0,
        ),
    )

    if base_error <= 0:
        base_error = max(
            3.0,
            number(
                motion.get(
                    "backtest_median_error",
                    3.0,
                ),
                3.0,
            ) * 1.5,
        )

    heading_noise = math.radians(
        min(
            45.0,
            number(
                motion.get(
                    "mean_turn_degrees",
                    0.0,
                )
            ),
        )
    )

    directional_error = (
        speed
        * horizon
        * math.sin(heading_noise)
    )

    speed_error = (
        speed
        * horizon
        * number(
            motion.get(
                "speed_cv",
                0.0,
            )
        )
        * 0.35
    )

    uncertainty = (
        base_error
        + directional_error * 0.35
        + speed_error
        + horizon * 0.10
    )

    return {
        "x_estimado": round(x_future, 2),
        "y_estimado": round(y_future, 2),
        "velocidad_estimada_pixeles_min": round(
            future_speed,
            4,
        ),
        "rumbo_estimado_grados": (
            round(future_bearing, 1)
            if future_bearing is not None
            else None
        ),
        "direccion_estimada": sector(
            future_bearing
        ),
        "incertidumbre_pixeles": round(
            max(3.0, uncertainty),
            2,
        ),
    }


# ============================================================
# CONFIANZA POR HORIZONTE
# ============================================================

def confidence_for_horizon(motion, horizon):
    confidence = number(
        motion.get("confidence", 0.0)
    )

    penalty = max(
        0.0,
        horizon - 5.0,
    ) * 0.55

    # Si el backtest es malo, el horizonte largo se penaliza más.
    p75 = motion.get("backtest_p75_error")

    if p75 is not None:
        if p75 > 12:
            penalty += horizon * 0.20
        elif p75 > 8:
            penalty += horizon * 0.10

    return round(
        clamp(
            confidence - penalty
        ),
        1,
    )


# ============================================================
# VALIDACIÓN
# ============================================================

def validate_point(prediction, actual, modo="MISMO_TRACK"):
    dx = (
        number(actual["x"])
        - number(prediction["x_estimado"])
    )

    dy = (
        number(actual["y"])
        - number(prediction["y_estimado"])
    )

    return {
        "estado": "VALIDADA",
        "modo": modo,
        "timestamp_real": canonical_ts(
            actual["timestamp"]
        ),
        "x_real": round(
            number(actual["x"]),
            2,
        ),
        "y_real": round(
            number(actual["y"]),
            2,
        ),
        "track_id_real": (
            str(actual["track_id"])
            if actual.get("track_id") is not None
            else None
        ),
        "error_pixeles": round(
            math.hypot(dx, dy),
            2,
        ),
        "error_x_pixeles": round(dx, 2),
        "error_y_pixeles": round(dy, 2),
    }


def build_observation_index(tracks):
    observations = {}

    for track in tracks:
        tid = track_id(track)

        for obs in valid_history(track):
            ts = canonical_ts(obs.get("timestamp"))

            if ts is None:
                continue

            item = dict(obs)
            item["track_id"] = tid

            observations[(tid, ts)] = item

    return observations


def all_observations_at(tracks, timestamp):
    target = canonical_ts(timestamp)

    if target is None:
        return []

    result = []

    for track in tracks:
        tid = track_id(track)

        for obs in valid_history(track):
            if canonical_ts(obs.get("timestamp")) != target:
                continue

            item = dict(obs)
            item["track_id"] = tid
            result.append(item)

    return result


def cross_track_match(prediction, observations):
    if not observations:
        return None

    px = number(prediction["x_estimado"])
    py = number(prediction["y_estimado"])

    uncertainty = max(
        3.0,
        number(
            prediction.get(
                "incertidumbre_pixeles",
                0.0,
            )
        ),
    )

    max_distance = max(
        MIN_MATCH_PIXELS,
        uncertainty * MAX_MATCH_EXTRA_UNCERTAINTY,
    )

    candidates = []

    for obs in observations:
        distance = math.hypot(
            number(obs["x"]) - px,
            number(obs["y"]) - py,
        )

        if distance <= max_distance:
            candidates.append(
                (distance, obs)
            )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])

    best_distance, best_obs = candidates[0]

    if len(candidates) > 1:
        second_distance = candidates[1][0]

        if (
            second_distance - best_distance
            < MIN_MATCH_MARGIN
        ):
            return None

    return best_obs


def load_calibration():
    data = load_json(
        CALIBRATION,
        {},
    )

    if not isinstance(data, dict):
        data = {}

    records = data.get("registros", [])

    if not isinstance(records, list):
        records = []

    clean = []

    for item in records:
        if not isinstance(item, dict):
            continue

        try:
            horizon = float(item.get("horizonte_minutos"))
            error_x = float(item.get("error_x_pixeles"))
            error_y = float(item.get("error_y_pixeles"))
            error = float(item.get("error_pixeles"))
        except (TypeError, ValueError):
            continue

        if horizon <= 0 or error < 0:
            continue

        clean.append({
            "track_id": str(item.get("track_id")),
            "timestamp_objetivo": canonical_ts(
                item.get("timestamp_objetivo")
            ),
            "horizonte_minutos": horizon,
            "error_x_pixeles": error_x,
            "error_y_pixeles": error_y,
            "error_pixeles": error,
        })

    return clean[-MAX_CALIBRATION:]


def update_calibration(records, validated):
    existing = {
        (
            item.get("track_id"),
            item.get("timestamp_objetivo"),
        )
        for item in records
    }

    for prediction in validated:
        validation = prediction.get("validacion", {})

        if validation.get("estado") != "VALIDADA":
            continue

        target = canonical_ts(
            prediction.get("timestamp_objetivo")
        )

        if target is None:
            continue

        horizon = number(
            prediction.get(
                "horizonte_desde_origen_minutos",
                0,
            )
        )

        if horizon <= 0:
            continue

        key = (
            str(prediction.get("track_id")),
            target,
        )

        if key in existing:
            continue

        records.append({
            "track_id": str(prediction.get("track_id")),
            "timestamp_objetivo": target,
            "horizonte_minutos": horizon,
            "error_x_pixeles": number(
                validation.get("error_x_pixeles")
            ),
            "error_y_pixeles": number(
                validation.get("error_y_pixeles")
            ),
            "error_pixeles": number(
                validation.get("error_pixeles")
            ),
        })

        existing.add(key)

    return records[-MAX_CALIBRATION:]


def calibration_for_horizon(records, horizon):
    values = []

    nearest = min(
        HORIZONTES_MIN,
        key=lambda value: abs(value - horizon),
    )

    for item in records:
        item_horizon = number(
            item.get("horizonte_minutos"),
            -1,
        )

        if abs(item_horizon - nearest) > 1.0:
            continue

        values.append(item)

    # No corregimos con muestras escasas.
    if len(values) < 8:
        return {
            "muestras": len(values),
            "bias_x": 0.0,
            "bias_y": 0.0,
        }

    return {
        "muestras": len(values),
        "bias_x": round(
            median(
                number(item.get("error_x_pixeles"))
                for item in values
            ),
            2,
        ),
        "bias_y": round(
            median(
                number(item.get("error_y_pixeles"))
                for item in values
            ),
            2,
        ),
    }


def apply_calibration(projection, calibration):
    if not calibration:
        return projection

    corrected = dict(projection)

    corrected["x_estimado"] = round(
        number(projection.get("x_estimado"))
        + number(calibration.get("bias_x")),
        2,
    )

    corrected["y_estimado"] = round(
        number(projection.get("y_estimado"))
        + number(calibration.get("bias_y")),
        2,
    )

    corrected["calibracion_muestras"] = int(
        number(calibration.get("muestras"))
    )
    corrected["correccion_bias_x_pixeles"] = round(
        number(calibration.get("bias_x")),
        2,
    )
    corrected["correccion_bias_y_pixeles"] = round(
        number(calibration.get("bias_y")),
        2,
    )

    return corrected


def calibration_summary(records):
    result = {}

    for horizon in HORIZONTES_MIN:
        calibration = calibration_for_horizon(
            records,
            horizon,
        )

        result[str(horizon)] = calibration

    return result


def validate_pending(current_ts, tracks, pending):
    now = parse_ts(current_ts)

    observations = build_observation_index(tracks)

    remaining = []
    validated = []

    for prediction in pending:
        target = canonical_ts(
            prediction.get("timestamp_objetivo")
        )

        if not target or now is None:
            remaining.append(prediction)
            continue

        target_dt = parse_ts(target)

        if target_dt is None or target_dt > now:
            remaining.append(prediction)
            continue

        tid = str(
            prediction.get("track_id")
        )

        actual = observations.get(
            (tid, target)
        )

        if actual is not None:
            prediction["validacion"] = validate_point(
                prediction,
                actual,
                "MISMO_TRACK",
            )
            validated.append(prediction)
            continue

        candidates = all_observations_at(
            tracks,
            target,
        )

        actual = cross_track_match(
            prediction,
            candidates,
        )

        if actual is not None:
            prediction["validacion"] = validate_point(
                prediction,
                actual,
                "TRACK_REASIGNADO_POR_PROXIMIDAD",
            )
            validated.append(prediction)
            continue

        # No descartamos silenciosamente una predicción pasada.
        # Si no hubo observación utilizable, queda pendiente para
        # evitar false positives en la métrica.
        remaining.append(prediction)

    return remaining, validated


# ============================================================
# HUECOS
# ============================================================

def make_gap_predictions(track, motion, start_ts, end_ts):
    start = parse_ts(start_ts)
    end = parse_ts(end_ts)

    if start is None or end is None:
        return []

    gap = (end - start).total_seconds() / 60.0

    if gap <= TOLERANCIA_HUECO_MIN:
        return []

    origin = motion["last"]

    x0 = number(origin["x"])
    y0 = number(origin["y"])

    tid = track_id(track)

    result = []

    horizon = PASO_HUECO_MIN

    while horizon < gap:
        target_dt = start + (
            end - start
        ) * (horizon / gap)

        projection = project(
            motion,
            x0,
            y0,
            horizon,
        )

        result.append(
            {
                "track_id": tid,
                "timestamp_origen": canonical_ts(start_ts),
                "timestamp_objetivo": iso(target_dt),
                "horizonte_desde_origen_minutos": horizon,
                "tipo": "ESTIMACION_HUECO",
                **projection,
                "confianza_prediccion": confidence_for_horizon(
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
            "timestamp_origen": canonical_ts(start_ts),
            "timestamp_objetivo": canonical_ts(end_ts),
            "horizonte_desde_origen_minutos": round(
                gap,
                2,
            ),
            "tipo": "PREDICCION_CIERRE_HUECO",
            **projection,
            "confianza_prediccion": confidence_for_horizon(
                motion,
                gap,
            ),
        }
    )

    return result


# ============================================================
# PREDICCIÓN DE CADA TRACK
# ============================================================

def predict_track(track, calibration_records=None):
    tid = track_id(track)

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
                "track_id": tid,
                "pronostico_disponible": False,
                "motivo": "SIN_MOVIMIENTO_VALIDO",
            },
            [],
        )

    current = motion["last"]

    forecasts = []

    if calibration_records is None:
        calibration_records = []

    for horizon in HORIZONTES_MIN:
        projection = project(
            motion,
            current["x"],
            current["y"],
            horizon,
        )

        calibration = calibration_for_horizon(
            calibration_records,
            horizon,
        )

        projection = apply_calibration(
            projection,
            calibration,
        )

        forecasts.append(
            {
                "horizonte_minutos": horizon,
                **projection,
                "confianza_prediccion": confidence_for_horizon(
                    motion,
                    horizon,
                ),
            }
        )

    # --------------------------------------------------------
    # HUECOS
    # --------------------------------------------------------

    gap_predictions = []

    full_history = valid_history(track)

    for a, b in zip(
        full_history,
        full_history[1:],
    ):
        ta = parse_ts(a["timestamp"])
        tb = parse_ts(b["timestamp"])

        if ta is None or tb is None:
            continue

        gap = (tb - ta).total_seconds() / 60.0

        if gap <= TOLERANCIA_HUECO_MIN:
            continue

        history_before = [
            item
            for item in full_history
            if parse_ts(item["timestamp"]) <= ta
        ][-MAX_HISTORY:]

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
            if point["tipo"] != "PREDICCION_CIERRE_HUECO":
                continue

            point["validacion"] = validate_point(
                point,
                {
                    **b,
                    "track_id": tid,
                },
                "CIERRE_HUECO",
            )

        gap_predictions.extend(points)

    return (
        {
            "track_id": tid,
            "activo": bool(
                track.get("activo", True)
            ),
            "pronostico_disponible": True,
            "posicion_actual": {
                "x": round(
                    number(current["x"]),
                    2,
                ),
                "y": round(
                    number(current["y"]),
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
                if motion["bearing"] is not None
                else None
            ),
            "direccion_actual": sector(
                motion["bearing"]
            ),
            "aceleracion_pixeles_min2": round(
                motion["acceleration"],
                5,
            ),
            "tendencia_area_porcentaje": (
                round(
                    motion["area_trend"],
                    2,
                )
                if motion["area_trend"] is not None
                else None
            ),
            "tendencia_intensidad": (
                round(
                    motion["intensity_trend"],
                    2,
                )
                if motion["intensity_trend"] is not None
                else None
            ),
            "probabilidad_intensificacion": round(
                motion["intensification"],
                1,
            ),
            "probabilidad_debilitamiento": round(
                motion["weakening"],
                1,
            ),
            "confianza_prediccion": round(
                motion["confidence"],
                1,
            ),
            "estabilidad_direccion": round(
                motion["direction_stability"],
                1,
            ),
            "estabilidad_velocidad": round(
                motion["speed_stability"],
                1,
            ),
            "error_regresion_pixeles": (
                round(
                    motion["backtest_mean_error"],
                    2,
                )
                if motion["backtest_mean_error"] is not None
                else None
            ),
            "backtest": {
                "muestras": motion["backtest_count"],
                "error_mediano_pixeles": motion[
                    "backtest_median_error"
                ],
                "error_p75_pixeles": motion[
                    "backtest_p75_error"
                ],
                "error_medio_pixeles": motion[
                    "backtest_mean_error"
                ],
            },
            "puntos_historial_utilizados": motion[
                "history_points"
            ],
            "horizontes": forecasts,
            "huecos_detectados": len(
                gap_predictions
            ),
            "estimaciones_hueco": gap_predictions,
        },
        gap_predictions,
    )


# ============================================================
# TRACKS
# ============================================================

def get_tracks(estado, temporal):
    tracks = estado.get("tracks", [])

    if not isinstance(tracks, list):
        tracks = []

    tracks = [
        dict(track)
        for track in tracks
        if isinstance(track, dict)
    ]

    if not tracks:
        fallback = temporal.get("tracks", [])

        if isinstance(fallback, list):
            tracks = [
                dict(track)
                for track in fallback
                if isinstance(track, dict)
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

    # El histórico del radar es la fuente de verdad.
    latest = latest_radar_timestamp()

    radar_ts = iso(latest) if latest is not None else None

    if radar_ts is None:
        radar_ts = canonical_ts(
            estado.get("radar_timestamp")
            or estado.get("ultimo_radar_timestamp")
            or temporal.get("radar_timestamp")
        )

    if radar_ts is None:
        raise RuntimeError(
            "No se pudo determinar el timestamp real del radar."
        )

    tracks = get_tracks(
        estado,
        temporal,
    )

    intervalo = real_radar_interval_minutes(
        radar_ts
    )

    # --------------------------------------------------------
    # PENDIENTES
    # --------------------------------------------------------

    pending_data = load_json(
        PENDING,
        {},
    )

    if isinstance(pending_data, dict):
        pending = pending_data.get(
            "pendientes",
            [],
        )
    else:
        pending = []

    if not isinstance(pending, list):
        pending = []

    calibration_records = load_calibration()

    pending, validated_now = validate_pending(
        radar_ts,
        tracks,
        pending,
    )

    # Aprendemos únicamente de predicciones ya verificadas antes
    # de generar las nuevas. Así evitamos contaminar la predicción
    # actual con datos que todavía no fueron observados.
    calibration_records = update_calibration(
        calibration_records,
        validated_now,
    )

    # --------------------------------------------------------
    # PREDICCIONES
    # --------------------------------------------------------

    predictions = []
    new_pending = []

    existing_keys = {
        (
            str(item.get("track_id")),
            canonical_ts(
                item.get("timestamp_objetivo")
            ),
        )
        for item in pending
        if isinstance(item, dict)
    }

    gap_count = 0
    gap_validated_now = []

    for track in tracks:
        if len(valid_history(track)) < 2:
            continue

        prediction, gap_predictions = predict_track(
            track,
            calibration_records,
        )

        predictions.append(prediction)

        # ----------------------------------------------------
        # HUECOS
        # ----------------------------------------------------

        for point in gap_predictions:
            gap_count += 1

            validation = point.get(
                "validacion",
                {},
            )

            if validation.get("estado") == "VALIDADA":
                gap_validated_now.append(point)
                continue

            key = (
                str(point.get("track_id")),
                canonical_ts(
                    point.get("timestamp_objetivo")
                ),
            )

            if key in existing_keys:
                continue

            new_pending.append(point)
            existing_keys.add(key)

        # ----------------------------------------------------
        # PREDICCIONES NORMALES
        # ----------------------------------------------------

        if not prediction.get(
            "pronostico_disponible"
        ):
            continue

        origin_dt = parse_ts(radar_ts)

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

            target_dt = origin_dt + timedelta(minutes=horizon)

            target_iso = iso(target_dt)

            key = (
                str(prediction.get("track_id")),
                target_iso,
            )

            if key in existing_keys:
                continue

            new_pending.append(
                {
                    "track_id": str(
                        prediction.get(
                            "track_id"
                        )
                    ),
                    "timestamp_origen": radar_ts,
                    "timestamp_objetivo": target_iso,
                    "horizonte_desde_origen_minutos": horizon,
                    "tipo": "PREDICCION_NORMAL",
                    "x_estimado": forecast.get(
                        "x_estimado"
                    ),
                    "y_estimado": forecast.get(
                        "y_estimado"
                    ),
                    "velocidad_estimada_pixeles_min": forecast.get(
                        "velocidad_estimada_pixeles_min"
                    ),
                    "rumbo_estimado_grados": forecast.get(
                        "rumbo_estimado_grados"
                    ),
                    "direccion_estimada": forecast.get(
                        "direccion_estimada"
                    ),
                    "incertidumbre_pixeles": forecast.get(
                        "incertidumbre_pixeles"
                    ),
                    "confianza_prediccion": forecast.get(
                        "confianza_prediccion"
                    ),
                }
            )

            existing_keys.add(key)

    pending.extend(new_pending)
    pending = pending[-MAX_PENDIENTES:]

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    available = [
        item
        for item in predictions
        if item.get("pronostico_disponible")
    ]

    validation_records = (
        list(validated_now)
        + list(gap_validated_now)
    )

    calibration_records = update_calibration(
        calibration_records,
        gap_validated_now,
    )

    errors = []

    for item in validation_records:
        value = (
            item.get(
                "validacion",
                {},
            ).get(
                "error_pixeles"
            )
        )

        if value is not None:
            errors.append(number(value))

    # Métricas por horizonte. Se mantienen los cuatro horizontes
    # principales aunque todavía no tengan muestras.
    by_horizon = {
        int(horizon): []
        for horizon in HORIZONTES_MIN
    }

    for item in validation_records:
        horizon = number(
            item.get(
                "horizonte_desde_origen_minutos",
                0,
            )
        )

        error = item.get(
            "validacion",
            {},
        ).get(
            "error_pixeles"
        )

        if error is None:
            continue

        nearest = min(
            HORIZONTES_MIN,
            key=lambda value: abs(value - horizon),
        )

        if abs(nearest - horizon) > 1.0:
            continue

        by_horizon[int(nearest)].append(
            number(error)
        )

    horizon_metrics = {}

    for horizon in HORIZONTES_MIN:
        values = by_horizon[int(horizon)]

        if not values:
            horizon_metrics[str(horizon)] = {
                "muestras": 0,
                "error_medio_pixeles": None,
                "error_mediano_pixeles": None,
                "dentro_5_pixeles": None,
                "dentro_10_pixeles": None,
                "dentro_15_pixeles": None,
            }
            continue

        horizon_metrics[str(horizon)] = {
            "muestras": len(values),
            "error_medio_pixeles": round(
                mean(values),
                2,
            ),
            "error_mediano_pixeles": round(
                median(values),
                2,
            ),
            "dentro_5_pixeles": round(
                sum(value <= 5 for value in values)
                / len(values)
                * 100.0,
                1,
            ),
            "dentro_10_pixeles": round(
                sum(value <= 10 for value in values)
                / len(values)
                * 100.0,
                1,
            ),
            "dentro_15_pixeles": round(
                sum(value <= 15 for value in values)
                / len(values)
                * 100.0,
                1,
            ),
        }

    summary = {
        "nucleos_con_prediccion": len(
            available
        ),
        "nucleos_sin_prediccion": len(
            predictions
        ) - len(available),
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
        "huecos_estimados": gap_count,
        "predicciones_validadas_en_este_frame": len(
            validation_records
        ),
        "predicciones_validadas_normales": len(
            validated_now
        ),
        "predicciones_validadas_huecos": len(
            gap_validated_now
        ),
        "error_medio_pixeles_validado": (
            round(
                mean(errors),
                2,
            )
            if errors
            else None
        ),
        "metricas_por_horizonte": horizon_metrics,
    }

    result = {
        "schema_version": 4,
        "radar_timestamp": radar_ts,
        "intervalo_minutos": intervalo,
        "tipo": "PREDICCION_MOVIMIENTO_CON_HUECOS",
        "fuente_tracks": str(ESTADO),
        "prediccion_global_disponible": bool(
            available
        ),
        "nucleos": predictions,
        "resumen": summary,
        "calibracion": calibration_summary(
            calibration_records
        ),
        "validaciones": validated_now[-100:],
        "nota": (
            "Predicción híbrida calibrada con "
            "movimiento reciente, regresión de trayectoria, "
            "estabilidad de rumbo y velocidad, aceleración "
            "conservadora, backtesting interno e incertidumbre "
            "adaptativa. Las posiciones estimadas nunca se "
            "etiquetan como radar observado."
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

    save_json(
        CALIBRATION,
        {
            "schema_version": 1,
            "actualizado": radar_ts,
            "registros": calibration_records[-MAX_CALIBRATION:],
            "resumen": calibration_summary(
                calibration_records
            ),
        },
    )

    print(f"OK: {OUTPUT}")
    print(f"Tracks analizados: {len(tracks)}")
    print(f"Predicciones: {len(predictions)}")
    print(f"Huecos estimados: {gap_count}")
    print(
        "Predicciones validadas:",
        len(validated_now),
    )
    print(
        "Pendientes:",
        len(pending),
    )

    print(
        "===== MÉTRICAS POR HORIZONTE ====="
    )

    for horizon in HORIZONTES_MIN:
        metrics = horizon_metrics[str(horizon)]

        print(
            f"{horizon} min | "
            f"muestras={metrics['muestras']} | "
            f"error_medio={metrics['error_medio_pixeles']} px | "
            f"<=5px={metrics['dentro_5_pixeles']}% | "
            f"<=10px={metrics['dentro_10_pixeles']}% | "
            f"<=15px={metrics['dentro_15_pixeles']}%"
        )

    if errors:
        print(
            "Error medio validado:",
            round(mean(errors), 2),
        )

    print("===== CALIBRACIÓN =====")

    for horizon in HORIZONTES_MIN:
        metrics = calibration_for_horizon(
            calibration_records,
            horizon,
        )

        print(
            f"{horizon} min | "
            f"muestras={metrics['muestras']} | "
            f"bias_x={metrics['bias_x']} px | "
            f"bias_y={metrics['bias_y']} px"
        )


if __name__ == "__main__":
    main()
