from __future__ import annotations

import importlib
import json
import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]

RADAR_FILE = (
    BASE
    / "data"
    / "radar"
    / "radar_nowcast.json"
)


def parse_timestamp(value: str) -> datetime:
    value = value.strip()

    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    dt = datetime.fromisoformat(value)

    if dt.tzinfo is None:
        dt = dt.replace(
            tzinfo=timezone.utc
        )

    return dt.astimezone(timezone.utc)


def load_radar_timestamp() -> datetime:
    if not RADAR_FILE.exists():
        raise FileNotFoundError(
            f"No existe {RADAR_FILE}"
        )

    with RADAR_FILE.open(
        "r",
        encoding="utf-8",
    ) as fh:
        data = json.load(fh)

    nowcast = data.get("nowcast")

    if not isinstance(nowcast, dict):
        raise RuntimeError(
            "radar_nowcast.json no contiene "
            "un bloque nowcast válido."
        )

    timestamp = nowcast.get(
        "frame_actual_utc"
    )

    if not timestamp:
        raise RuntimeError(
            "No existe nowcast.frame_actual_utc."
        )

    return parse_timestamp(timestamp)


def parse_row_time(value):
    if not value:
        return None

    try:
        return parse_timestamp(str(value))
    except Exception:
        return None


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def temporal_interpolation(rows, target):
    """Interpola una serie horaria al instante exacto solicitado."""
    valid = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        dt = parse_row_time(row.get("tiempo_utc"))
        if dt is not None:
            valid.append((dt, row))

    if not valid:
        return None

    valid.sort(key=lambda item: item[0])

    for dt, row in valid:
        if dt == target:
            result = dict(row)
            result["tiempo_utc"] = target.isoformat()
            result["interpolacion_temporal"] = False
            return result

    before = None
    after = None

    for dt, row in valid:
        if dt < target:
            before = (dt, row)
        elif dt > target:
            after = (dt, row)
            break

    if before is None:
        result = dict(valid[0][1])
        result["tiempo_utc"] = target.isoformat()
        result["interpolacion_temporal"] = False
        return result

    if after is None:
        result = dict(valid[-1][1])
        result["tiempo_utc"] = target.isoformat()
        result["interpolacion_temporal"] = False
        return result

    t0, r0 = before
    t1, r1 = after
    total = (t1 - t0).total_seconds()
    alpha = (
        (target - t0).total_seconds() / total
        if total > 0
        else 0.0
    )

    result = dict(r0)

    keys = set(r0) | set(r1)

    for key in keys:
        a = finite(r0.get(key))
        b = finite(r1.get(key))

        if a is not None and b is not None:
            result[key] = a + (b - a) * alpha
        elif key in r1 and key not in r0:
            result[key] = r1[key]

    result["tiempo_utc"] = target.isoformat()
    result["interpolacion_temporal"] = True
    result["factor_temporal"] = round(alpha, 4)
    result["tiempo_anterior_utc"] = t0.isoformat()
    result["tiempo_siguiente_utc"] = t1.isoformat()

    return result


def distance_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
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

    return 2 * r * math.asin(
        min(1.0, math.sqrt(a))
    )


def environment_along_projection(points, projection, radar_timestamp):
    """
    Obtiene el ambiente meteorológico en la posición Y hora de cada
    proyección. Hace interpolación temporal y espacial sobre la grilla ECMWF.
    """
    if not isinstance(projection, dict):
        return None

    projected_lat = finite(
        projection.get("latitud")
    )
    projected_lon = finite(
        projection.get("longitud")
    )
    minutes = finite(
        projection.get("minutos")
    )

    if projected_lat is None or projected_lon is None:
        return None

    if minutes is None:
        minutes = 0.0

    target = radar_timestamp + timedelta(
        minutes=minutes
    )

    candidates = []

    for point in points:
        if not isinstance(point, dict):
            continue

        plat = finite(point.get("latitud"))
        plon = finite(point.get("longitud"))
        rows = point.get("datos", [])

        if plat is None or plon is None or not isinstance(rows, list):
            continue

        row = temporal_interpolation(rows, target)
        if row is None:
            continue

        distance = distance_km(
            projected_lat,
            projected_lon,
            plat,
            plon,
        )

        candidates.append(
            (distance, point, row)
        )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    candidates = candidates[:4]

    # Si la trayectoria cae exactamente sobre un punto de la grilla,
    # no hace falta ponderar otros puntos.
    if candidates[0][0] < 0.001:
        result = dict(candidates[0][2])
        result["latitud"] = projected_lat
        result["longitud"] = projected_lon
        result["punto"] = candidates[0][1].get(
            "punto",
            "GRILLA",
        )
        result["interpolacion_espacial"] = False
        result["distancia_punto_mas_cercano_km"] = 0.0
        return result

    weights = []
    for distance, point, row in candidates:
        weight = 1.0 / max(distance, 0.001) ** 2
        weights.append(
            (weight, point, row, distance)
        )

    weight_sum = sum(item[0] for item in weights)

    nearest_row = candidates[0][2]
    result = dict(nearest_row)

    numeric_keys = set()
    for _, _, row, _ in weights:
        numeric_keys.update(row.keys())

    for key in numeric_keys:
        values = []
        for weight, _, row, _ in weights:
            value = finite(row.get(key))
            if value is not None:
                values.append((weight, value))

        if values:
            denominator = sum(weight for weight, _ in values)
            if denominator > 0:
                result[key] = sum(
                    weight * value
                    for weight, value in values
                ) / denominator

    result["latitud"] = projected_lat
    result["longitud"] = projected_lon
    result["punto"] = "INTERPOLADO_GRILLA_ECMWF"
    result["interpolacion_espacial"] = True
    result["puntos_utilizados"] = [
        point.get("punto")
        for _, point, _, _ in weights
    ]
    result["distancias_puntos_km"] = {
        point.get("punto", "GRILLA"): round(
            distance,
            3,
        )
        for _, point, _, distance in weights
    }

    return result


def main() -> None:
    radar_timestamp = load_radar_timestamp()

    print(
        "Timestamp real del radar:",
        radar_timestamp.isoformat(),
    )

    sys.path.insert(
        0,
        str(BASE / "scripts"),
    )

    engine = importlib.import_module(
        "tormenta_intelligence"
    )

    original_now_dt = getattr(
        engine,
        "now_dt",
        None,
    )

    original_environment_function = getattr(
        engine,
        "environment_along_projection",
        None,
    )

    if original_now_dt is None:
        raise RuntimeError(
            "tormenta_intelligence.py no expone "
            "la función now_dt()."
        )

    if original_environment_function is None:
        raise RuntimeError(
            "tormenta_intelligence.py no expone "
            "environment_along_projection()."
        )

    try:
        engine.now_dt = (
            lambda: radar_timestamp
        )

        engine.environment_along_projection = (
            lambda points, projection: (
                environment_along_projection(
                    points,
                    projection,
                    radar_timestamp,
                )
            )
        )

        print(
            "Storm Engine ejecutándose con "
            "el timestamp del frame Radar."
        )

        print(
            "Meteorología de trayectoria: "
            "interpolación temporal + espacial ECMWF."
        )

        result = engine.main()

    finally:
        engine.now_dt = original_now_dt
        engine.environment_along_projection = (
            original_environment_function
        )

    print(
        "Storm Engine finalizado correctamente."
    )

    return result


if __name__ == "__main__":
    main()
