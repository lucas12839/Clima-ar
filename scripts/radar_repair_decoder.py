"""
ClimaAR - Radar Repair Decoder 8.0

No convierte la paleta RGB de RainViewer a dBZ.
Corrige además el problema por el cual este decoder sobrescribía
radar_nowcast.json y declaraba "sin_precipitacion" con confianza
100% aunque el frame fuera viejo.

Mantiene detección, núcleos, movimiento y fallback.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from scripts import radar_rainviewer as rv
except ImportError:
    import radar_rainviewer as rv


BASE = Path(__file__).resolve().parents[1]
RADAR_DIR = BASE / "data" / "radar"
HISTORY_DIR = RADAR_DIR / "historico"
NOWCAST_FILE = RADAR_DIR / "radar_nowcast.json"

MAX_FRAMES = 13
MAX_AGE_MINUTES = 35.0
FALLBACK_MAX_MINUTES = 90.0

DETECTION_RADIUS_KM = 150.0
PRIORITY_RADIUS_KM = 100.0

VERSION = "8.0-radar-safe"


def now_utc():
    return datetime.now(timezone.utc)


def iso_timestamp(timestamp):
    return datetime.fromtimestamp(
        int(timestamp),
        timezone.utc,
    ).isoformat()


def parse_timestamp_from_name(path):
    stem = path.stem
    candidates = [stem]

    if stem.startswith("radar_"):
        candidates.append(stem[6:])

    for value in candidates:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    return None


def load_history():
    if not HISTORY_DIR.exists():
        return []

    found = {}

    for path in HISTORY_DIR.glob("*.png"):
        if not path.is_file() or path.stat().st_size <= 100:
            continue

        timestamp = parse_timestamp_from_name(path)

        if timestamp is not None:
            found[timestamp] = path

    ordered = sorted(found.items(), key=lambda item: item[0])
    return ordered[-MAX_FRAMES:]


def read_image(path):
    with Image.open(path) as source:
        return source.convert("RGBA").copy()


def rgb_for_cv(image):
    rgba = np.asarray(image)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def analyze(image, timestamp):
    bgr = rgb_for_cv(image)
    result = rv.analyze_frame(bgr)

    return {
        "timestamp": int(timestamp),
        "frame_utc": iso_timestamp(timestamp),
        "area_px": int(result.get("area_px", 0)),
        "coverage": float(result.get("coverage", 0.0)),
        "nuclei": result.get("nuclei", []),
        "priority_nuclei": result.get("priority_nuclei", []),
        "centroid_x": float(result.get("centroid_x", 0.0)),
        "centroid_y": float(result.get("centroid_y", 0.0)),
        "centroid_distance_km": float(
            result.get("centroid_distance_km", 0.0)
        ),
        "intense_pixels": int(
            result.get("intense_pixels", 0)
        ),
    }


def direction_degrees(x, y):
    if abs(x) < 0.05 and abs(y) < 0.05:
        return None

    return round(
        math.degrees(math.atan2(x, -y)) % 360.0,
        1,
    )


def speed_kmh(pixel_speed, timestamp_a, timestamp_b):
    delta_seconds = max(
        1.0,
        float(timestamp_b - timestamp_a),
    )

    meters_per_pixel = rv.radar_geometry()

    return round(
        pixel_speed
        * meters_per_pixel
        / delta_seconds
        * 3.6,
        1,
    )


def movement_toward_center(analysis, movement_x, movement_y):
    dx = (512.0 / 2.0) - analysis["centroid_x"]
    dy = (512.0 / 2.0) - analysis["centroid_y"]

    distance = math.hypot(dx, dy)
    movement = math.hypot(movement_x, movement_y)

    if distance < 1.0 or movement < 0.05:
        return False

    return (dx * movement_x + dy * movement_y) > 0


def build_projection(analysis, movement_x, movement_y, minutes):
    projected_x = (
        analysis["centroid_x"]
        + movement_x * minutes / 10.0
    )
    projected_y = (
        analysis["centroid_y"]
        + movement_y * minutes / 10.0
    )

    projected_x = max(0.0, min(511.0, projected_x))
    projected_y = max(0.0, min(511.0, projected_y))

    distance = rv.distance_from_center_km(
        projected_x,
        projected_y,
    )

    return {
        "minutos": int(minutes),
        "x": round(projected_x, 2),
        "y": round(projected_y, 2),
        "distancia_km": round(distance, 1),
    }


def history_path_for_timestamp(results, timestamp):
    for item in results:
        if item["timestamp"] == timestamp:
            return item["path"]

    raise FileNotFoundError(
        f"No se encontró el frame {timestamp}"
    )


def build_nowcast(results):
    latest = results[-1]
    previous = results[-2] if len(results) >= 2 else None

    age_minutes = max(
        0.0,
        (
            now_utc().timestamp()
            - latest["timestamp"]
        ) / 60.0,
    )

    movement_x = 0.0
    movement_y = 0.0
    pixel_speed = 0.0
    movement_confidence = 0.0

    if previous is not None:
        previous_image = read_image(
            history_path_for_timestamp(
                results,
                previous["timestamp"],
            )
        )

        latest_image = read_image(
            history_path_for_timestamp(
                results,
                latest["timestamp"],
            )
        )

        (
            movement_x,
            movement_y,
            pixel_speed,
        ) = rv.estimate_motion(
            rgb_for_cv(previous_image),
            rgb_for_cv(latest_image),
        )

        interval_minutes = max(
            1.0,
            (
                latest["timestamp"]
                - previous["timestamp"]
            ) / 60.0,
        )

        if interval_minutes <= 30:
            movement_confidence = min(
                1.0,
                max(0.0, pixel_speed / 2.0),
            )

    direction = rv.direction_from_vector(
        movement_x,
        movement_y,
    )

    direction_deg = direction_degrees(
        movement_x,
        movement_y,
    )

    speed = (
        speed_kmh(
            pixel_speed,
            previous["timestamp"],
            latest["timestamp"],
        )
        if previous is not None
        else 0.0
    )

    active = (
        latest["area_px"] > 0
        and len(latest["nuclei"]) > 0
    )

    change_area_pct = 0.0
    strengthening = "sin_datos"

    if previous is not None and previous["area_px"] > 0:
        change_area_pct = round(
            (
                latest["area_px"]
                - previous["area_px"]
            )
            / previous["area_px"]
            * 100.0,
            1,
        )

        if change_area_pct >= 15:
            strengthening = "fortaleciendose"
        elif change_area_pct <= -15:
            strengthening = "debilitandose"
        else:
            strengthening = "estable"

    toward = movement_toward_center(
        latest,
        movement_x,
        movement_y,
    )

    distance = (
        latest["centroid_distance_km"]
        if active
        else None
    )

    eta = None

    if toward and speed >= 5 and distance is not None:
        eta = round(
            distance / speed * 60.0,
            1,
        )

    projections = {}

    if active and movement_confidence > 0:
        for minutes in (15, 30, 60):
            projections[str(minutes)] = build_projection(
                latest,
                movement_x,
                movement_y,
                minutes,
            )

    stale = age_minutes > MAX_AGE_MINUTES

    if stale:
        estado = "radar_desactualizado"
        confidence = 0
    elif not active:
        estado = "sin_precipitacion"
        confidence = 100
    else:
        estado = "actividad_radar"
        confidence = round(
            movement_confidence * 100.0,
            1,
        )

    fallback = {
        "activo": stale,
        "limitado": age_minutes > FALLBACK_MAX_MINUTES,
        "edad_minutos": round(age_minutes, 1),
        "ultima_observacion_utc": latest["frame_utc"],
        "ultima_actividad": active,
        "ultima_area_px": latest["area_px"],
        "ultima_cobertura": round(
            latest["coverage"],
            6,
        ),
        "ultima_distancia_km": (
            round(distance, 1)
            if distance is not None
            else None
        ),
        "movimiento_estimado": {
            "x": round(movement_x, 4),
            "y": round(movement_y, 4),
            "velocidad_kmh": speed,
            "direccion": direction,
            "direccion_grados": direction_deg,
            "hacia_bahia_blanca": toward,
            "confianza": round(
                movement_confidence * 100.0,
                1,
            ),
        },
        "proyecciones": projections,
        "nota": (
            "Se conserva la última observación y su movimiento "
            "estimado. No se presenta como radar actual hasta "
            "recuperar un frame fresco."
        ),
    }

    return {
        "estado": estado,
        "actividad": active and not stale,
        "actividad_ultima_observada": active,
        "confianza": confidence,
        "movimiento": {
            "direccion": direction,
            "velocidad_kmh": speed,
            "confianza": round(
                movement_confidence * 100.0,
                1,
            ),
        },
        "actualizado": now_utc().isoformat(),
        "frame_actual": latest["timestamp"],
        "frame_actual_utc": latest["frame_utc"],
        "frame_timestamp": latest["timestamp"],
        "frame_edad_minutos": round(
            age_minutes,
            1,
        ),
        "frame_es_viejo": stale,
        "datos_vigentes": not stale,
        "frames_analizados": len(results),
        "tiles_ok": 1,
        "area_px": latest["area_px"],
        "distancia_km": distance,
        "fortalecimiento": strengthening,
        "cambio_area_pct": change_area_pct,
        "velocidad_kmh": speed,
        "direccion": direction,
        "direccion_grados": direction_deg,
        "movimiento_hacia_bahia": toward,
        "eta_minutos": eta,
        "proyecciones": projections,
        "confianza_movimiento": round(
            movement_confidence,
            3,
        ),
        "nucleos": latest["nuclei"],
        "priority_nuclei_100km": (
            latest["priority_nuclei"]
        ),
        "detection_radius_km": (
            DETECTION_RADIUS_KM
        ),
        "priority_radius_km": (
            PRIORITY_RADIUS_KM
        ),
        "fallback": fallback,
    }


def main():
    files = load_history()

    if not files:
        raise RuntimeError(
            f"No hay frames en {HISTORY_DIR}"
        )

    print(
        f"CLIMAAR DECODER {VERSION} - "
        f"Frames: {len(files)}"
    )

    results = []

    for timestamp, path in files:
        image = read_image(path)
        result = analyze(
            image,
            timestamp,
        )
        result["path"] = path
        results.append(result)

        print(
            f"{path.name} | "
            f"area={result['area_px']} | "
            f"cobertura={result['coverage']:.6f} | "
            f"nucleos={len(result['nuclei'])}"
        )

    results.sort(
        key=lambda item: item["timestamp"]
    )

    nowcast = build_nowcast(results)
    latest = results[-1]

    output = {
        "version": VERSION,
        "app": "ClimaAR",
        "ubicacion": {
            "latitud": getattr(
                rv,
                "LAT",
                -38.71,
            ),
            "longitud": getattr(
                rv,
                "LON",
                -62.26,
            ),
            "ciudad": "Bahia Blanca",
        },
        "fuente": (
            "RainViewer -> RMA10 Bahia Blanca"
        ),
        "frame_actual": latest["timestamp"],
        "frame_actual_utc": latest["frame_utc"],
        "frame_edad_minutos": nowcast[
            "frame_edad_minutos"
        ],
        "nowcast": nowcast,
        "historial": {
            "frames_procesados": len(results),
            "frames_disponibles": len(files),
        },
        "diagnostico": {
            "cobertura_pct": round(
                latest["coverage"] * 100.0,
                4,
            ),
            "sin_precipitacion": (
                not latest["area_px"]
                and not nowcast[
                    "frame_es_viejo"
                ]
            ),
            "radar_desactualizado": (
                nowcast["frame_es_viejo"]
            ),
            "datos_vigentes": (
                nowcast["datos_vigentes"]
            ),
        },
        "detection_radius_km": (
            DETECTION_RADIUS_KM
        ),
        "priority_radius_km": (
            PRIORITY_RADIUS_KM
        ),
    }

    NOWCAST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    NOWCAST_FILE.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    if nowcast["frame_es_viejo"]:
        print(
            "RADAR DESACTUALIZADO: no se declara "
            "sin_precipitacion actual. Fallback preparado."
        )
    else:
        print(
            "RADAR VIGENTE: análisis actual disponible."
        )


if __name__ == "__main__":
    main()
