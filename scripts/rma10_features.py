"""Shared causal features and geometry for the RMA10 operational chain."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Iterable

RADAR_LAT = -38.7342
RADAR_LON = -62.1654
BAHIA_LAT = -38.7196
BAHIA_LON = -62.2724
RADAR_RADIUS_KM = 240.0

# Velocidad física máxima de un sistema convectivo en la región. Un
# desplazamiento mayor entre frames es un salto del centroide (celda que se
# divide, se une o cambia de forma), no movimiento real.
VELOCIDAD_MAX_KMH = 120.0


def velocidad_max_px_min(width: int = 512) -> float:
    km_por_px = RADAR_RADIUS_KM / max((width - 1) / 2.0, 1.0)
    return VELOCIDAD_MAX_KMH / 60.0 / km_por_px
MIN_MOTION_INTERVAL_MIN = 2.0
MAX_MOTION_INTERVAL_MIN = 45.0

V1_FEATURES = (
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
    "motion_valid",
    "motion_interval_min",
    "past_intensity_delta",
    "past_area_ratio",
)


def parse_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif value is None:
        return None
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def pixel_to_latlon(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    if width <= 1 or height <= 1:
        raise ValueError("Dimensiones RMA10 inválidas")
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    east_km = (finite(x, cx) - cx) / max(cx, 1.0) * RADAR_RADIUS_KM
    north_km = (cy - finite(y, cy)) / max(cy, 1.0) * RADAR_RADIUS_KM
    km_lat = 110.574
    km_lon = 111.320 * math.cos(math.radians(RADAR_LAT))
    return RADAR_LAT + north_km / km_lat, RADAR_LON + east_km / km_lon


def latlon_to_pixel(lat: float, lon: float, width: int, height: int) -> tuple[float, float]:
    if width <= 1 or height <= 1:
        raise ValueError("Dimensiones RMA10 inválidas")
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    km_lat = 110.574
    km_lon = 111.320 * math.cos(math.radians(RADAR_LAT))
    east_km = (finite(lon, RADAR_LON) - RADAR_LON) * km_lon
    north_km = (finite(lat, RADAR_LAT) - RADAR_LAT) * km_lat
    x = cx + east_km / RADAR_RADIUS_KM * max(cx, 1.0)
    y = cy - north_km / RADAR_RADIUS_KM * max(cy, 1.0)
    return x, y


def bahia_pixel_position(width: int = 512, height: int = 512) -> tuple[float, float]:
    return latlon_to_pixel(BAHIA_LAT, BAHIA_LON, width, height)


def causal_motion(history: Iterable[dict[str, Any]], timestamp: Any) -> dict[str, float]:
    """Derive motion only from the two latest observations at or before T0."""
    target = parse_utc(timestamp)
    observations = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        dt = parse_utc(item.get("timestamp", item.get("frame_utc", item.get("dt"))))
        if dt is None or (target is not None and dt > target):
            continue
        try:
            x, y = float(item["x"]), float(item["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        observations.append((dt, item, x, y))
    observations.sort(key=lambda row: row[0])
    deduped = {}
    for row in observations:
        deduped[row[0]] = row
    observations = list(deduped.values())
    invalid = {
        "dx_px": 0.0, "dy_px": 0.0, "displacement_px": 0.0,
        "speed_px_min": 0.0, "motion_bearing_deg": 0.0,
        "motion_valid": 0.0, "motion_interval_min": 0.0,
        "past_intensity_delta": 0.0, "past_area_ratio": 0.0,
    }
    if len(observations) < 2:
        return invalid
    previous, current = observations[-2], observations[-1]
    interval = (current[0] - previous[0]).total_seconds() / 60.0
    if not (MIN_MOTION_INTERVAL_MIN <= interval <= MAX_MOTION_INTERVAL_MIN):
        return invalid
    dx, dy = current[2] - previous[2], current[3] - previous[3]
    displacement = math.hypot(dx, dy)
    previous_item, current_item = previous[1], current[1]
    previous_intensity = finite(previous_item.get("intensidad", previous_item.get("intensity")), 0.0)
    current_intensity = finite(current_item.get("intensidad", current_item.get("intensity")), 0.0)
    previous_area = finite(previous_item.get("pixeles", previous_item.get("area_px")), 0.0)
    current_area = finite(current_item.get("pixeles", current_item.get("area_px")), 0.0)
    return {
        "dx_px": dx,
        "dy_px": dy,
        "displacement_px": displacement,
        "speed_px_min": displacement / interval,
        "motion_bearing_deg": ((math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0) if displacement > 0 else 0.0,
        "motion_valid": 1.0,
        "motion_interval_min": interval,
        "past_intensity_delta": current_intensity - previous_intensity,
        "past_area_ratio": current_area / max(previous_area, 1.0),
    }


def build_v1_feature_dict(current: dict[str, Any], history: Iterable[dict[str, Any]], width: int, height: int) -> dict[str, float]:
    x, y = finite(current.get("x")), finite(current.get("y"))
    lat, lon = pixel_to_latlon(x, y, width, height)
    motion = causal_motion(history, current.get("timestamp", current.get("frame_utc", current.get("dt"))))
    return {
        "intensity": finite(current.get("intensidad", current.get("intensity"))),
        "area_px": finite(current.get("pixeles", current.get("area_px"))),
        "width_px": finite(current.get("ancho", current.get("width_px"))),
        "height_px": finite(current.get("alto", current.get("height_px"))),
        "lat": lat,
        "lon": lon,
        "distance_km": distance_km(lat, lon, BAHIA_LAT, BAHIA_LON),
        "bearing_to_bahia_deg": bearing_deg(lat, lon, BAHIA_LAT, BAHIA_LON),
        **{key: motion[key] for key in (
            "displacement_px", "speed_px_min", "motion_bearing_deg",
            "motion_valid", "motion_interval_min", "past_intensity_delta", "past_area_ratio",
        )},
    }


def build_v1_feature_vector(current: dict[str, Any], history: Iterable[dict[str, Any]], width: int, height: int):
    values = build_v1_feature_dict(current, history, width, height)
    return [float(values[name]) for name in V1_FEATURES]
