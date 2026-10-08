import csv
import json
import math
import os
from datetime import datetime, timezone

import cv2
import numpy as np
import requests

# ============================================================
# CLIMAAR - RADAR RAINVIEWER
# v11.0 - TRACKING DE CELULAS + MOVIMIENTO ROBUSTO
# ============================================================
# Objetivo de esta version:
#   - no estimar el movimiento usando toda la imagen;
#   - detectar celulas/objetos de precipitacion;
#   - separar movimiento global de movimiento de cada celula;
#   - asociar celulas entre frames;
#   - calcular direccion a partir de tracks persistentes;
#   - entregar una confianza de movimiento/direccion medible;
#   - conservar los nombres de archivos y campos existentes para
#     mantener compatibilidad con el resto de ClimaAR.
# ============================================================

LAT = -38.71
LON = -62.26

ZOOM = 7
SIZE = 512

DETECTION_RADIUS_KM = 150.0
PRIORITY_RADIUS_KM = 100.0

# Se usan varios frames para que la direccion no dependa de un solo
# intervalo ruidoso. RainViewer puede devolver intervalos irregulares.
FRAMES = 8
MAX_HISTORY = 144

# Parametros del detector/tracker.
MIN_CELL_AREA_PX = 8
MIN_TRACK_AREA_PX = 10
MAX_CELLS = 40
MATCH_MAX_DISTANCE_PX = 75.0
MAX_TIME_GAP_MIN = 25.0
MIN_TRACK_POINTS = 2

# La confianza de direccion se considera fuerte cuando el track tiene
# persistencia, desplazamiento suficiente y buena concordancia.
MIN_DIRECTION_DISPLACEMENT_PX = 2.0
STRONG_DIRECTION_CONFIDENCE = 0.85

RADAR_DIR = "data/radar"
HISTORY_DIR = os.path.join(RADAR_DIR, "historico")

ACTUAL_PATH = os.path.join(RADAR_DIR, "actual.png")
NOWCAST_PATH = os.path.join(RADAR_DIR, "radar_nowcast.json")
STATUS_PATH = os.path.join(RADAR_DIR, "status.json")
FEATURES_PATH = os.path.join(RADAR_DIR, "radar_features_rainviewer.csv")

LAST_VALID_IMAGE_PATH = os.path.join(
    RADAR_DIR,
    "last_valid_radar.png",
)

LAST_VALID_NOWCAST_PATH = os.path.join(
    RADAR_DIR,
    "last_valid_nowcast.json",
)

RAINVIEWER_API = "https://api.rainviewer.com/public/weather-maps.json"

CSV_FIELDS = [
    "frame_time",
    "frame_utc",
    "source",
    "latitud",
    "longitud",
    "area_px",
    "distance_km",
    "dbz_max",
    "dbz_mean",
    "dbz_p90",
    "dbz_pixels",
    "tiles_ok",
    "nucleos",
    "centroide_x",
    "centroide_y",
    "pixeles_intensos",
    "cobertura_precipitacion",
    "movimiento_x",
    "movimiento_y",
    "direccion",
    "velocidad_pixeles_frame",
]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ClimaAR/1.0"})


# ============================================================
# UTILIDADES
# ============================================================

def ensure_dirs():
    os.makedirs(RADAR_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)


def iso_from_timestamp(ts):
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json_write(path, data):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def atomic_copy_file(source, destination):
    temporary = destination + ".tmp"
    with open(source, "rb") as src, open(temporary, "wb") as dst:
        dst.write(src.read())
    os.replace(temporary, destination)


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================
# RAINVIEWER API
# ============================================================

def download_json(url):
    response = SESSION.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def get_radar_frames():
    data = download_json(RAINVIEWER_API)
    host = data.get("host", "").rstrip("/")
    radar = data.get("radar", {})
    past = radar.get("past", [])

    if not host:
        raise RuntimeError("RainViewer no devolvio host")
    if not past:
        raise RuntimeError("RainViewer no devolvio frames")

    frames = []
    for item in past:
        if not item.get("time") or not item.get("path"):
            continue
        frames.append({
            "time": int(item["time"]),
            "path": item["path"],
        })

    frames.sort(key=lambda item: item["time"])

    if not frames:
        raise RuntimeError("No existen frames validos")

    return host, frames


def build_image_url(host, path):
    return (
        f"{host}{path}/{SIZE}/{ZOOM}/"
        f"{LAT}/{LON}/2/0_0.png"
    )


def build_coverage_url(host):
    return (
        f"{host}/v2/coverage/0/{SIZE}/{ZOOM}/"
        f"{LAT}/{LON}/0/0_0.png"
    )


# ============================================================
# DESCARGA Y DIAGNOSTICO
# ============================================================

def download_image(url):
    response = SESSION.get(url, timeout=40)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if len(response.content) < 512:
        raise RuntimeError("Respuesta de radar demasiado pequena")

    if "image" not in content_type and not response.content.startswith(b"\x89PNG"):
        raise RuntimeError("RainViewer no devolvio PNG")

    data = np.frombuffer(response.content, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise RuntimeError("OpenCV no pudo decodificar imagen")
    if image.ndim not in (2, 3):
        raise RuntimeError("Formato de imagen inesperado")

    height, width = image.shape[:2]
    if width < 64 or height < 64:
        raise RuntimeError("Imagen demasiado pequena")

    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        rgb_max = int(max(
            image[:, :, 0].max(),
            image[:, :, 1].max(),
            image[:, :, 2].max(),
        ))
        alpha_max = int(alpha.max())
        visible_pixels = int(np.count_nonzero(alpha >= 20))
        fully_transparent = alpha_max == 0
        print(
            "[RADAR] "
            f"imagen={width}x{height} rgb_max={rgb_max} "
            f"alpha_max={alpha_max} pixeles_visibles={visible_pixels} "
            f"transparente={fully_transparent}"
        )
    else:
        max_value = int(image.max())
        nonzero = int(np.count_nonzero(image))
        fully_transparent = False
        print(
            "[RADAR] "
            f"imagen={width}x{height} max={max_value} "
            f"pixeles_no_negros={nonzero}"
        )

    return image, fully_transparent


def download_coverage_status(host):
    url = build_coverage_url(host)
    try:
        response = SESSION.get(url, timeout=30)
        response.raise_for_status()
        data = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

        if image is None:
            return {
                "ok": False,
                "cobertura_disponible": None,
                "error": "No se pudo decodificar mascara",
            }

        if image.ndim == 3 and image.shape[2] == 4:
            alpha = image[:, :, 3]
            available = int(np.count_nonzero(alpha >= 20))
            return {
                "ok": True,
                "cobertura_disponible": available > 0,
                "pixeles_cobertura": available,
            }

        nonzero = int(np.count_nonzero(image))
        return {
            "ok": True,
            "cobertura_disponible": nonzero > 0,
            "pixeles_cobertura": nonzero,
        }
    except Exception as exc:
        print(f"[WARN] No se pudo consultar cobertura RainViewer: {exc}")
        return {
            "ok": False,
            "cobertura_disponible": None,
            "error": str(exc),
        }


# ============================================================
# GUARDADO
# ============================================================

def save_png(image, path):
    temporary = path + ".tmp.png"
    ok = cv2.imwrite(temporary, image)
    if not ok:
        raise RuntimeError("No se pudo guardar imagen")
    os.replace(temporary, path)


# ============================================================
# GEOMETRIA
# ============================================================

def meters_per_pixel():
    return (
        156543.03392804097
        / (2 ** ZOOM)
        * math.cos(math.radians(LAT))
    )


def distance_from_center_km(x, y, width, height):
    scale = meters_per_pixel()
    cx = width / 2.0
    cy = height / 2.0
    dx = float(x) - cx
    dy = float(y) - cy
    return math.hypot(dx, dy) * scale / 1000.0


def monitoring_circle(width, height):
    scale = meters_per_pixel()
    radius_px = DETECTION_RADIUS_KM * 1000.0 / scale
    yy, xx = np.ogrid[:height, :width]
    cx = width / 2.0
    cy = height / 2.0
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px ** 2


# ============================================================
# REPRESENTACION DEL RADAR
# ============================================================

def split_image(image):
    if image.ndim == 3 and image.shape[2] == 4:
        b, g, r, alpha = cv2.split(image)
    elif image.ndim == 3:
        b, g, r = cv2.split(image)
        alpha = np.full(b.shape, 255, dtype=np.uint8)
    else:
        b = image
        g = image
        r = image
        alpha = np.full(image.shape, 255, dtype=np.uint8)
    return b, g, r, alpha


def radar_strength(image):
    """Genera una intensidad relativa 0..1 sin inventar dBZ."""
    b, g, r, alpha = split_image(image)
    bgr = cv2.merge([b, g, r])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    value = hsv[:, :, 2].astype(np.float32) / 255.0
    alpha_f = alpha.astype(np.float32) / 255.0

    # RainViewer es una representacion RGB. Se usa saturacion + valor
    # solamente como indice relativo para comparar pixels dentro del
    # mismo producto, nunca como conversion a dBZ.
    strength = np.maximum(saturation * 0.70, value * 0.30)
    strength *= alpha_f
    return strength


def precipitation_mask(image):
    b, g, r, alpha = split_image(image)
    bgr = cv2.merge([b, g, r])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Umbral bajo, pero evitando pixels casi negros/grises que suelen ser
    # fondo o ruido de la tile.
    colorful = saturation >= 10
    visible = value >= 12
    opaque = alpha >= 10

    mask = (colorful & visible & opaque).astype(np.uint8) * 255

    # Elimina puntos aislados y cierra pequeños huecos sin borrar celulas.
    open_kernel = np.ones((2, 2), np.uint8)
    close_kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    monitor = monitoring_circle(mask.shape[1], mask.shape[0])
    mask[~monitor] = 0
    return mask


def component_overlap(mask_a, mask_b, dx, dy):
    """Solapamiento aproximado tras desplazar mask_b por dx/dy."""
    h, w = mask_a.shape
    dx = int(round(dx))
    dy = int(round(dy))

    x0a = max(0, dx)
    x1a = min(w, w + dx)
    y0a = max(0, dy)
    y1a = min(h, h + dy)

    x0b = max(0, -dx)
    x1b = min(w, w - dx)
    y0b = max(0, -dy)
    y1b = min(h, h - dy)

    if x0a >= x1a or y0a >= y1a:
        return 0.0

    a = mask_a[y0a:y1a, x0a:x1a] > 0
    b = mask_b[y0b:y1b, x0b:x1b] > 0
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 0.0


# ============================================================
# DETECCION DE CELULAS
# ============================================================

def analyze_frame(image):
    mask = precipitation_mask(image)
    strength = radar_strength(image)
    height, width = mask.shape

    area = int(cv2.countNonZero(mask))
    monitor = monitoring_circle(width, height)
    monitored_pixels = int(np.count_nonzero(monitor))
    coverage = area / monitored_pixels if monitored_pixels else 0.0

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cells = []
    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < MIN_CELL_AREA_PX:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"]:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx = x + cw / 2.0
            cy = y + ch / 2.0

        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        ys, xs = np.where(contour_mask > 0)
        mean_strength = float(np.mean(strength[ys, xs])) if len(xs) else 0.0
        p90_strength = float(np.percentile(strength[ys, xs], 90)) if len(xs) else 0.0
        max_strength = float(np.max(strength[ys, xs])) if len(xs) else 0.0

        distance = distance_from_center_km(cx, cy, width, height)
        cells.append({
            "x": round(float(cx), 2),
            "y": round(float(cy), 2),
            "area_px": int(round(contour_area)),
            "width_px": int(cw),
            "height_px": int(ch),
            "distance_km": round(distance, 1),
            "prioridad_100km": distance <= PRIORITY_RADIUS_KM,
            "mean_strength": round(mean_strength, 4),
            "p90_strength": round(p90_strength, 4),
            "max_strength": round(max_strength, 4),
        })

    cells.sort(
        key=lambda item: (
            item["area_px"] * (0.5 + item["mean_strength"]),
            item["max_strength"],
        ),
        reverse=True,
    )
    cells = cells[:MAX_CELLS]

    priority_cells = [c for c in cells if c["prioridad_100km"]]

    ys, xs = np.where(mask > 0)
    if len(xs):
        centroid_x = float(np.mean(xs))
        centroid_y = float(np.mean(ys))
        centroid_distance = distance_from_center_km(
            centroid_x, centroid_y, width, height
        )
    else:
        centroid_x = 0.0
        centroid_y = 0.0
        centroid_distance = 0.0

    intense = (
        (strength >= 0.70) & (mask > 0)
    )

    return {
        "mask": mask,
        "strength": strength,
        "area_px": area,
        "coverage": coverage,
        "cells": cells,
        # Alias conservado para compatibilidad con el JSON anterior.
        "nuclei": cells,
        "priority_cells": priority_cells,
        "priority_nuclei": priority_cells,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "centroid_distance_km": round(centroid_distance, 1),
        "intense_pixels": int(np.count_nonzero(intense)),
    }


# ============================================================
# MOVIMIENTO GLOBAL ROBUSTO
# ============================================================

def estimate_global_shift(previous_mask, current_mask):
    """Calcula desplazamiento global de precipitacion, no del fondo."""
    if previous_mask is None or current_mask is None:
        return 0.0, 0.0, 0.0

    a = (previous_mask > 0).astype(np.float32)
    b = (current_mask > 0).astype(np.float32)

    if np.count_nonzero(a) < MIN_TRACK_AREA_PX or np.count_nonzero(b) < MIN_TRACK_AREA_PX:
        return 0.0, 0.0, 0.0

    # Suavizado para que la correlacion no sea dominada por un solo pixel.
    a = cv2.GaussianBlur(a, (0, 0), 1.2)
    b = cv2.GaussianBlur(b, (0, 0), 1.2)

    try:
        shift, response = cv2.phaseCorrelate(a, b)
        dx = float(shift[0])
        dy = float(shift[1])
        response = clamp01(response)

        # Evita desplazamientos absurdos producidos por mascaras muy distintas.
        if math.hypot(dx, dy) > MATCH_MAX_DISTANCE_PX:
            return 0.0, 0.0, 0.0

        return dx, dy, response
    except cv2.error:
        return 0.0, 0.0, 0.0


def match_cells(previous_cells, current_cells, global_dx, global_dy):
    """Asocia objetos por posicion predicha + tamano + intensidad."""
    if not previous_cells or not current_cells:
        return [], list(range(len(current_cells)))

    candidates = []
    for pi, prev in enumerate(previous_cells):
        predicted_x = prev["x"] + global_dx
        predicted_y = prev["y"] + global_dy

        for ci, cur in enumerate(current_cells):
            dx = cur["x"] - predicted_x
            dy = cur["y"] - predicted_y
            distance = math.hypot(dx, dy)
            if distance > MATCH_MAX_DISTANCE_PX:
                continue

            prev_area = max(float(prev["area_px"]), 1.0)
            cur_area = max(float(cur["area_px"]), 1.0)
            area_ratio = min(prev_area, cur_area) / max(prev_area, cur_area)

            prev_strength = safe_float(prev.get("mean_strength"))
            cur_strength = safe_float(cur.get("mean_strength"))
            strength_delta = abs(prev_strength - cur_strength)

            # Score menor = mejor.
            score = (
                distance / MATCH_MAX_DISTANCE_PX * 0.62
                + (1.0 - area_ratio) * 0.23
                + min(strength_delta, 1.0) * 0.15
            )
            candidates.append((score, pi, ci, distance, area_ratio))

    candidates.sort(key=lambda item: item[0])
    used_prev = set()
    used_cur = set()
    matches = []

    for score, pi, ci, distance, area_ratio in candidates:
        if pi in used_prev or ci in used_cur:
            continue
        used_prev.add(pi)
        used_cur.add(ci)
        matches.append({
            "previous_index": pi,
            "current_index": ci,
            "score": float(score),
            "distance_px": float(distance),
            "area_ratio": float(area_ratio),
        })

    unmatched_current = [
        i for i in range(len(current_cells)) if i not in used_cur
    ]
    return matches, unmatched_current


# ============================================================
# TRACKING DE CELULAS
# ============================================================

def update_tracks(tracks, previous_cells, current_cells, timestamp, global_dx, global_dy, dt_minutes):
    matches, unmatched_current = match_cells(
        previous_cells,
        current_cells,
        global_dx,
        global_dy,
    )

    # Primero actualiza tracks existentes mediante la asociacion anterior.
    prev_to_track = {}
    for track_id, track in tracks.items():
        if track.get("last_index") is not None:
            prev_to_track[track["last_index"]] = track_id

    matched_track_ids = set()
    new_track_ids = []

    for match in matches:
        pi = match["previous_index"]
        ci = match["current_index"]
        track_id = prev_to_track.get(pi)

        if track_id is None:
            continue

        cell = current_cells[ci]
        track = tracks[track_id]
        old = track["last_cell"]

        dx = float(cell["x"] - old["x"])
        dy = float(cell["y"] - old["y"])
        distance = math.hypot(dx, dy)

        if dt_minutes > 0:
            vx = dx / dt_minutes
            vy = dy / dt_minutes
        else:
            vx = dx
            vy = dy

        track["points"].append({
            "time": int(timestamp),
            "x": float(cell["x"]),
            "y": float(cell["y"]),
            "dx": dx,
            "dy": dy,
            "vx": vx,
            "vy": vy,
            "match_score": match["score"],
            "match_distance_px": match["distance_px"],
            "area_ratio": match["area_ratio"],
        })
        track["last_cell"] = cell
        track["last_index"] = ci
        track["last_time"] = int(timestamp)
        track["matched_frames"] += 1
        track["last_match_score"] = match["score"]
        matched_track_ids.add(track_id)

    # Crea tracks para celulas que aparecieron nuevas.
    next_id = max(tracks.keys(), default=0) + 1
    for ci in unmatched_current:
        cell = current_cells[ci]
        tracks[next_id] = {
            "id": next_id,
            "last_index": ci,
            "last_time": int(timestamp),
            "last_cell": cell,
            "matched_frames": 1,
            "last_match_score": 1.0,
            "points": [{
                "time": int(timestamp),
                "x": float(cell["x"]),
                "y": float(cell["y"]),
                "dx": 0.0,
                "dy": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "match_score": 1.0,
                "match_distance_px": 0.0,
                "area_ratio": 0.0,
            }],
        }
        new_track_ids.append(next_id)
        next_id += 1

    # Un track solo se mantiene si no se quedo demasiado atrasado.
    cutoff = int(timestamp - MAX_TIME_GAP_MIN * 60)
    stale = [
        track_id
        for track_id, track in tracks.items()
        if track["last_time"] < cutoff
    ]
    for track_id in stale:
        del tracks[track_id]

    return tracks


def circular_direction(dx, dy):
    if abs(dx) < 0.05 and abs(dy) < 0.05:
        return "sin_movimiento"

    # Coordenadas imagen: x+ = este, y+ = sur.
    angle = math.degrees(math.atan2(dx, -dy))
    if angle < 0:
        angle += 360

    directions = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    index = int((angle + 22.5) // 45) % 8
    return directions[index]


def aggregate_track_motion(tracks):
    """Elige/combina tracks persistentes, priorizando los mas confiables."""
    candidates = []

    for track in tracks.values():
        points = track.get("points", [])
        if len(points) < MIN_TRACK_POINTS:
            continue

        recent = points[-5:]
        vectors = [p for p in recent if p["dx"] or p["dy"]]
        if not vectors:
            continue

        dxs = np.array([p["dx"] for p in vectors], dtype=np.float32)
        dys = np.array([p["dy"] for p in vectors], dtype=np.float32)
        scores = np.array([
            max(0.0, 1.0 - p["match_score"]) for p in vectors
        ], dtype=np.float32)

        dx = float(np.median(dxs))
        dy = float(np.median(dys))
        displacement = float(math.hypot(dx, dy))
        consistency = 1.0

        if len(vectors) >= 2:
            angles = np.array(
                [math.atan2(p["dy"], p["dx"]) for p in vectors],
                dtype=np.float32,
            )
            # Magnitud de la media vectorial: 1 = muy consistente.
            consistency = float(math.hypot(
                np.mean(np.cos(angles)),
                np.mean(np.sin(angles)),
            ))

        mean_match = float(np.mean(scores)) if len(scores) else 0.0
        persistence = clamp01(len(points) / 5.0)
        displacement_score = clamp01(displacement / 5.0)
        match_quality = clamp01(mean_match)

        confidence = (
            0.30 * persistence
            + 0.30 * consistency
            + 0.25 * match_quality
            + 0.15 * displacement_score
        )

        if displacement < MIN_DIRECTION_DISPLACEMENT_PX:
            confidence *= 0.55

        # Peso por tamano de la ultima celula.
        area_weight = math.sqrt(max(track["last_cell"]["area_px"], 1))
        priority_weight = 1.25 if track["last_cell"].get("prioridad_100km") else 1.0
        weight = area_weight * priority_weight * max(confidence, 0.05)

        candidates.append({
            "track_id": track["id"],
            "dx": dx,
            "dy": dy,
            "displacement_px": displacement,
            "consistency": consistency,
            "persistence": persistence,
            "match_quality": match_quality,
            "confidence": clamp01(confidence),
            "weight": weight,
            "area_px": track["last_cell"]["area_px"],
            "distance_km": track["last_cell"]["distance_km"],
            "points": len(points),
        })

    if not candidates:
        return {
            "movement_x": 0.0,
            "movement_y": 0.0,
            "speed": 0.0,
            "direction": "sin_datos",
            "direction_confidence": 0.0,
            "movement_confidence": 0.0,
            "tracking_quality": 0.0,
            "active_tracks": 0,
            "primary_track_id": None,
            "tracks": [],
        }

    candidates.sort(
        key=lambda item: (
            item["confidence"] * item["weight"],
            item["area_px"],
        ),
        reverse=True,
    )

    # Se combinan como maximo las 5 mejores celulas. Esto evita que una
    # nube gigante domine toda la solucion si otra celula cercana tiene
    # un movimiento claramente distinto.
    selected = candidates[:5]
    total_weight = sum(c["weight"] for c in selected) or 1.0

    movement_x = sum(c["dx"] * c["weight"] for c in selected) / total_weight
    movement_y = sum(c["dy"] * c["weight"] for c in selected) / total_weight
    speed = math.hypot(movement_x, movement_y)

    direction = circular_direction(movement_x, movement_y)

    # Confianza global basada en la mejor celula + acuerdo entre las
    # celulas seleccionadas.
    best_conf = max(c["confidence"] for c in selected)
    mean_conf = float(np.mean([c["confidence"] for c in selected]))

    if len(selected) >= 2:
        dirs = np.array([
            math.atan2(c["dy"], c["dx"])
            for c in selected
            if math.hypot(c["dx"], c["dy"]) >= MIN_DIRECTION_DISPLACEMENT_PX
        ], dtype=np.float32)
        if len(dirs):
            agreement = float(math.hypot(
                np.mean(np.cos(dirs)),
                np.mean(np.sin(dirs)),
            ))
        else:
            agreement = 0.0
    else:
        agreement = selected[0]["consistency"]

    tracking_quality = clamp01(
        0.35 * best_conf
        + 0.30 * mean_conf
        + 0.35 * agreement
    )

    return {
        "movement_x": round(float(movement_x), 4),
        "movement_y": round(float(movement_y), 4),
        "speed": round(float(speed), 4),
        "direction": direction,
        "direction_confidence": round(float(tracking_quality), 4),
        "movement_confidence": round(float(tracking_quality), 4),
        "tracking_quality": round(float(tracking_quality), 4),
        "active_tracks": len(candidates),
        "primary_track_id": selected[0]["track_id"],
        "tracks": selected,
    }


# Compatibilidad con el nombre de la funcion anterior.
def estimate_motion(previous, current):
    if previous is None or current is None:
        return 0.0, 0.0, 0.0
    previous_mask = precipitation_mask(previous)
    current_mask = precipitation_mask(current)
    return estimate_global_shift(previous_mask, current_mask)


def movement_direction(movement_x, movement_y):
    return circular_direction(movement_x, movement_y)


# ============================================================
# HISTORIAL / CSV
# ============================================================

def cleanup_history():
    files = []
    for name in os.listdir(HISTORY_DIR):
        if not name.endswith(".png"):
            continue
        path = os.path.join(HISTORY_DIR, name)
        if os.path.isfile(path):
            files.append(path)

    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for old in files[MAX_HISTORY:]:
        try:
            os.remove(old)
        except OSError:
            pass


def append_features(row):
    exists = os.path.exists(FEATURES_PATH)
    with open(FEATURES_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# ESTADO DE FUENTE
# ============================================================

def source_health_from_observations(observations, coverage_info):
    if not observations:
        return {
            "estado_fuente": "sin_datos",
            "radar_con_datos": False,
            "motivo": "No hubo frames descargables",
        }

    transparent_count = sum(
        1 for item in observations if item["fully_transparent"]
    )
    visible_count = len(observations) - transparent_count

    # No basta con que el PNG exista: debe tener pixels utilizables en
    # al menos un frame. Asi evitamos tratar un tile vacio como lluvia.
    useful_count = sum(
        1 for item in observations
        if item["analysis"]["area_px"] >= MIN_CELL_AREA_PX
    )

    if useful_count > 0:
        return {
            "estado_fuente": "datos_recibidos",
            "radar_con_datos": True,
            "motivo": (
                f"{useful_count}/{len(observations)} frames con "
                "pixels radar utilizables"
            ),
        }

    if coverage_info.get("cobertura_disponible") is False:
        return {
            "estado_fuente": "sin_cobertura",
            "radar_con_datos": False,
            "motivo": "La mascara indica ausencia de cobertura",
        }

    return {
        "estado_fuente": "fuente_sin_datos",
        "radar_con_datos": False,
        "motivo": (
            f"{visible_count} frames visibles pero sin ecos suficientes "
            "para confirmar precipitacion"
        ),
    }


# ============================================================
# FALLBACK
# ============================================================

def load_last_valid_nowcast():
    if not os.path.exists(LAST_VALID_NOWCAST_PATH):
        return None
    try:
        with open(LAST_VALID_NOWCAST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] No se pudo cargar ultima observacion valida: {exc}")
        return None


def activate_radar_fallback(current_nowcast, current_status):
    last_valid = load_last_valid_nowcast()

    current_nowcast["fallback"] = {
        "activo": True,
        "motivo": (
            "Fuente radar sin datos confirmados; se conserva la "
            "ultima observacion valida."
        ),
        "ultima_observacion_valida": last_valid is not None,
    }
    current_status["fallback_activo"] = True
    current_status["ultima_observacion_valida"] = last_valid is not None

    if last_valid is not None:
        current_nowcast["ultima_observacion_valida"] = last_valid

    if os.path.exists(LAST_VALID_IMAGE_PATH):
        try:
            atomic_copy_file(LAST_VALID_IMAGE_PATH, ACTUAL_PATH)
            current_status["imagen_actual"] = "ultima_imagen_radar_valida"
        except Exception as exc:
            print(f"[WARN] No se pudo restaurar ultima imagen valida: {exc}")

    return current_nowcast, current_status


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def main():
    ensure_dirs()

    print("[CLIMAAR] Iniciando radar RainViewer v11.0")
    print(
        f"[CLIMAAR] Centro={LAT},{LON} "
        f"radio={DETECTION_RADIUS_KM}km prioridad={PRIORITY_RADIUS_KM}km"
    )
    print(
        f"[TRACKING] frames={FRAMES} min_cell={MIN_CELL_AREA_PX}px "
        f"max_match={MATCH_MAX_DISTANCE_PX}px"
    )

    host, frames = get_radar_frames()
    selected = frames[-FRAMES:]
    coverage_info = download_coverage_status(host)

    print(
        "[RADAR] Cobertura: "
        f"{coverage_info.get('cobertura_disponible')}"
    )

    observations = []
    pending_rows = []
    tracks = {}
    previous_cells = []
    previous_mask = None
    previous_timestamp = None
    successful_frames = 0

    for item in selected:
        timestamp = item["time"]
        frame_utc = iso_from_timestamp(timestamp)
        url = build_image_url(host, item["path"])

        print(f"[RADAR] Descargando: {frame_utc}")

        try:
            image, fully_transparent = download_image(url)
            analysis = analyze_frame(image)

            dt_minutes = 0.0
            if previous_timestamp is not None:
                dt_minutes = max(0.0, (timestamp - previous_timestamp) / 60.0)

            global_dx, global_dy, global_response = estimate_global_shift(
                previous_mask,
                analysis["mask"],
            )

            tracks = update_tracks(
                tracks,
                previous_cells,
                analysis["cells"],
                timestamp,
                global_dx,
                global_dy,
                dt_minutes,
            )

            tracking = aggregate_track_motion(tracks)

            history_path = os.path.join(HISTORY_DIR, f"{timestamp}.png")
            save_png(image, history_path)

            observation = {
                "time": timestamp,
                "utc": frame_utc,
                "image": image,
                "analysis": analysis,
                "movement_x": tracking["movement_x"],
                "movement_y": tracking["movement_y"],
                "speed": tracking["speed"],
                "direction": tracking["direction"],
                "direction_confidence": tracking["direction_confidence"],
                "movement_confidence": tracking["movement_confidence"],
                "tracking_quality": tracking["tracking_quality"],
                "tracking": tracking,
                "global_shift_x": global_dx,
                "global_shift_y": global_dy,
                "global_response": global_response,
                "fully_transparent": fully_transparent,
            }
            observations.append(observation)

            row = {
                "frame_time": timestamp,
                "frame_utc": frame_utc,
                "source": "RainViewer",
                "latitud": LAT,
                "longitud": LON,
                "area_px": analysis["area_px"],
                "distance_km": analysis["centroid_distance_km"],
                "dbz_max": "",
                "dbz_mean": "",
                "dbz_p90": "",
                "dbz_pixels": 0,
                "tiles_ok": 1,
                "nucleos": len(analysis["cells"]),
                "centroide_x": round(analysis["centroid_x"], 2),
                "centroide_y": round(analysis["centroid_y"], 2),
                "pixeles_intensos": analysis["intense_pixels"],
                "cobertura_precipitacion": round(analysis["coverage"], 6),
                "movimiento_x": tracking["movement_x"],
                "movimiento_y": tracking["movement_y"],
                "direccion": tracking["direction"],
                "velocidad_pixeles_frame": tracking["speed"],
            }
            pending_rows.append(row)

            previous_cells = analysis["cells"]
            previous_mask = analysis["mask"]
            previous_timestamp = timestamp
            successful_frames += 1

            print(
                "[TRACKING] "
                f"celulas={len(analysis['cells'])} "
                f"mov=({tracking['movement_x']:.2f},"
                f"{tracking['movement_y']:.2f}) "
                f"dir={tracking['direction']} "
                f"conf={tracking['direction_confidence']:.2f} "
                f"tracks={tracking['active_tracks']}"
            )

        except Exception as exc:
            print(f"[WARN] Frame rechazado: {exc}")

    if not observations:
        raise RuntimeError("No hubo ningun frame descargable")

    latest = observations[-1]
    latest_image = latest["image"]
    latest_analysis = latest["analysis"]
    latest_tracking = latest["tracking"]
    latest_time = latest["time"]

    age_seconds = datetime.now(timezone.utc).timestamp() - latest_time
    age_minutes = age_seconds / 60.0

    precipitation_area = latest_analysis["area_px"]
    cells = latest_analysis["cells"]
    priority_cells = latest_analysis["priority_cells"]

    source_health = source_health_from_observations(
        observations,
        coverage_info,
    )

    # ========================================================
    # ESTADO
    # ========================================================
    if age_minutes > 35:
        estado = "radar_desactualizado"
        datos_vigentes = False
        confianza = 0.0
    elif source_health["estado_fuente"] == "sin_cobertura":
        estado = "radar_sin_cobertura"
        datos_vigentes = False
        confianza = 0.0
    elif source_health["estado_fuente"] == "fuente_sin_datos":
        estado = "radar_sin_datos"
        datos_vigentes = False
        confianza = 0.0
    elif precipitation_area > 0:
        estado = "precipitacion_detectada"
        datos_vigentes = True
        base = 0.50 + min(0.25, len(cells) * 0.025)
        confianza = min(
            1.0,
            base + 0.25 * latest_tracking["tracking_quality"],
        )
    else:
        estado = "sin_precipitacion"
        datos_vigentes = True
        confianza = 0.50

    if source_health["radar_con_datos"]:
        for row in pending_rows:
            append_features(row)
    else:
        print(
            "[RADAR] No se agregan filas al CSV: "
            "fuente radar sin datos confirmados."
        )

    # ========================================================
    # NOWCAST
    # ========================================================
    nowcast = {
        "version": "11.0-cell-tracking",
        "generated_utc": utc_now(),
        "source": "RainViewer",
        "radar_frame_utc": latest["utc"],
        "radar_frame_timestamp": latest_time,
        "edad_minutos": round(age_minutes, 1),
        "frame_es_viejo": age_minutes > 35,
        "datos_vigentes": datos_vigentes,
        "estado": estado,
        "confianza": round(confianza, 3),
        "fuente": {
            "estado": source_health["estado_fuente"],
            "radar_con_datos": source_health["radar_con_datos"],
            "motivo": source_health["motivo"],
            "cobertura": coverage_info,
        },
        "centro": {
            "latitud": LAT,
            "longitud": LON,
        },
        "deteccion": {
            "radio_km": DETECTION_RADIUS_KM,
            "prioridad_km": PRIORITY_RADIUS_KM,
            "cobertura_precipitacion": round(
                latest_analysis["coverage"], 6
            ),
            "area_px": latest_analysis["area_px"],
            "nucleos": cells,
            "nucleos_prioridad": priority_cells,
            "pixeles_intensos": latest_analysis["intense_pixels"],
        },
        "movimiento": {
            "x": latest_tracking["movement_x"],
            "y": latest_tracking["movement_y"],
            "velocidad_pixeles_frame": latest_tracking["speed"],
            "direccion": latest_tracking["direction"],
            "direccion_confianza": latest_tracking["direction_confidence"],
            "movimiento_confianza": latest_tracking["movement_confidence"],
            "tracking_quality": latest_tracking["tracking_quality"],
            "active_tracks": latest_tracking["active_tracks"],
            "primary_track_id": latest_tracking["primary_track_id"],
            "tracks": latest_tracking["tracks"],
            "metodo": "tracking_celulas + correlacion_global",
        },
        "historico_frames_validos": successful_frames,
        "frames_transparentes": sum(
            1 for item in observations if item["fully_transparent"]
        ),
        "dbz_disponible": False,
        "nota_dbz": (
            "RainViewer RGB no se convierte artificialmente a dBZ."
        ),
    }

    # ========================================================
    # STATUS
    # ========================================================
    status = {
        "version": "11.0-cell-tracking",
        "updated_utc": utc_now(),
        "source": "RainViewer",
        "estado": estado,
        "datos_vigentes": datos_vigentes,
        "frame_es_viejo": age_minutes > 35,
        "edad_minutos": round(age_minutes, 1),
        "radio_deteccion_km": DETECTION_RADIUS_KM,
        "radio_prioridad_km": PRIORITY_RADIUS_KM,
        "nucleos": len(cells),
        "nucleos_prioridad": len(priority_cells),
        "cobertura_precipitacion": round(
            latest_analysis["coverage"], 6
        ),
        "frames_validos": successful_frames,
        "frames_transparentes": sum(
            1 for item in observations if item["fully_transparent"]
        ),
        "fuente": source_health,
        "cobertura_radar": coverage_info,
        "dbz_disponible": False,
        "fallback_activo": False,
        "ultima_observacion_valida": False,
        "tracking": {
            "version": "cell-tracking-v1",
            "direccion": latest_tracking["direction"],
            "direccion_confianza": latest_tracking["direction_confidence"],
            "movimiento_confianza": latest_tracking["movement_confidence"],
            "tracking_quality": latest_tracking["tracking_quality"],
            "active_tracks": latest_tracking["active_tracks"],
            "primary_track_id": latest_tracking["primary_track_id"],
        },
    }

    # ========================================================
    # FUENTE SANA / FALLBACK
    # ========================================================
    if source_health["radar_con_datos"]:
        save_png(latest_image, ACTUAL_PATH)
        save_png(latest_image, LAST_VALID_IMAGE_PATH)
        atomic_json_write(LAST_VALID_NOWCAST_PATH, nowcast)
    elif estado in (
        "radar_sin_datos",
        "radar_sin_cobertura",
        "radar_desactualizado",
    ):
        nowcast, status = activate_radar_fallback(
            nowcast,
            status,
        )
    else:
        save_png(latest_image, ACTUAL_PATH)

    atomic_json_write(NOWCAST_PATH, nowcast)
    atomic_json_write(STATUS_PATH, status)
    cleanup_history()

    print("")
    print("============================================")
    print(" CLIMAAR RADAR FINALIZADO - v11.0")
    print("============================================")
    print(f"Frame: {latest['utc']}")
    print(f"Edad: {age_minutes:.1f} min")
    print(f"Estado: {estado}")
    print(f"Fuente: {source_health['estado_fuente']}")
    print(f"Precipitacion: {precipitation_area} px")
    print(f"Nucleos: {len(cells)}")
    print(f"Nucleos prioridad: {len(priority_cells)}")
    print(f"Frames validos: {successful_frames}/{len(selected)}")
    print(
        "Frames transparentes: "
        f"{sum(1 for item in observations if item['fully_transparent'])}"
    )
    print(
        "Direccion: "
        f"{latest_tracking['direction']} "
        f"confianza={latest_tracking['direction_confidence']:.2f}"
    )
    print(
        "Movimiento: "
        f"x={latest_tracking['movement_x']:.2f} "
        f"y={latest_tracking['movement_y']:.2f} "
        f"velocidad={latest_tracking['speed']:.2f}"
    )
    print(
        "Tracking quality: "
        f"{latest_tracking['tracking_quality']:.2f}"
    )
    print(
        "Cobertura radar: "
        f"{coverage_info.get('cobertura_disponible')}"
    )
    print("dBZ: no disponible (no se inventa)")
    print("============================================")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise
