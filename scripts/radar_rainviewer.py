import os
import json
import math
import csv
from io import BytesIO
from datetime import datetime, timezone

import requests
from PIL import Image
import numpy as np
import cv2

# ============================================================
# CLIMAAR - RADAR RAINVIEWER V9.0
# Robust RainViewer tiles + coverage diagnostics + dBZ
# No nowcast forecast from RainViewer: only past radar frames.
# ============================================================

LAT = -38.71
LON = -62.26

ZOOM = 7
TILE_SIZE = 512
GRID_RADIUS = 1

DATA_DIR = "data/radar"
HISTORY_DIR = os.path.join(DATA_DIR, "historico")
FEATURES_CSV = os.path.join(DATA_DIR, "radar_features_rainviewer.csv")
NOWCAST_FILE = os.path.join(DATA_DIR, "radar_nowcast.json")
ACTUAL_FILE = os.path.join(DATA_DIR, "actual.png")

API_URL = "https://api.rainviewer.com/public/weather-maps.json"

HEADERS = {
    "User-Agent": "ClimaAR/9.0",
    "Referer": "https://www.rainviewer.com/",
    "Accept": "image/png,image/*;q=0.8,*/*;q=0.5",
}

MAX_HISTORY_FRAMES = 144
MAX_ANALYSIS_FRAMES = 13
TRACK_FRAMES = 6
MIN_COMPONENT_AREA = 20
MAX_COMPONENTS = 12
MAX_TRACK_SPEED_KMH = 180.0
PROJECTION_MINUTES = (15, 30, 45, 60, 90)
PALETTE_DISTANCE = 36.0

# Universal Blue, scheme 2. RainViewer color table:
# dBZ 10..95 inclusive = 86 entries.
DBZ_HEX = [
    "cec08796", "d2c48ba0", "d6c88faa", "dacc93b4", "ded097be",
    "88ddeeff", "6cd1ebff", "51c5e8ff", "36bae5ff", "1baee2ff",
    "00a3e0ff", "009ad5ff", "0091caff", "0088bfff", "007fb4ff",
    "0077aaff", "0070a3ff", "00699cff", "006295ff", "005b8eff",
    "005588ff", "005180ff", "004e78ff", "004a70ff", "004768ff",
    "ffee00ff", "ffe000ff", "ffd200ff", "ffc500ff", "ffb700ff",
    "ffaa00ff", "ff9f00ff", "ff9500ff", "ff8b00ff", "ff8100ff",
    "ff4400ff", "f23600ff", "e62800ff", "d91b00ff", "cd0d00ff",
    "c10000ff", "a80000ff", "8f0000ff", "760000ff", "5d0000ff",
    "ffaaffff", "ff9fffff", "ff95ffff", "ff8bffff", "ff81ffff",
    "ff77ffff", "ff6cffff", "ff62ffff", "ff58ffff", "ff4effff",
] + ["ffffffff"] * 31

DBZ_VALUES = np.arange(10, 96, dtype=np.float32)
DBZ_RGB = np.array(
    [[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in DBZ_HEX],
    dtype=np.float32,
)

if len(DBZ_HEX) != len(DBZ_VALUES):
    raise RuntimeError("Paleta Universal Blue inválida.")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)


# ============================================================
# GEOGRAPHY
# ============================================================

def latlon_to_tile(lat, lon, zoom):
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return int(math.floor(x)), int(math.floor(y))


def pixels_per_km(lat, zoom, tile_size):
    world_km = 40075.016
    km_x = world_km * math.cos(math.radians(lat)) / ((2 ** zoom) * tile_size)
    km_y = world_km / ((2 ** zoom) * tile_size)
    return km_x, km_y


def frame_datetime(ts):
    return datetime.fromtimestamp(int(ts), timezone.utc).isoformat()


def direction_name(degrees):
    names = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    return names[int((degrees + 22.5) / 45) % 8]


def bahia_pixel_position():
    n = float(2 ** ZOOM)
    lat_rad = math.radians(LAT)
    world_x = (LON + 180.0) / 360.0 * n * TILE_SIZE
    world_y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * TILE_SIZE
    tile_x = math.floor(world_x / TILE_SIZE)
    tile_y = math.floor(world_y / TILE_SIZE)
    local_x = world_x - tile_x * TILE_SIZE
    local_y = world_y - tile_y * TILE_SIZE
    return float(GRID_RADIUS * TILE_SIZE + local_x), float(GRID_RADIUS * TILE_SIZE + local_y)


def distance_point_to_bahia(cx, cy):
    bx, by = bahia_pixel_position()
    km_x, km_y = pixels_per_km(LAT, ZOOM, TILE_SIZE)
    return math.hypot((float(cx) - bx) * km_x, (float(cy) - by) * km_y)


# ============================================================
# RAINVIEWER API
# ============================================================

def fetch_rainviewer():
    response = requests.get(API_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()
    data = response.json()

    host = data.get("host", "https://tilecache.rainviewer.com").rstrip("/")
    radar = data.get("radar", {})
    frames = radar.get("past", []) or []

    if not frames:
        raise RuntimeError("RainViewer no devolvió frames históricos.")

    # Use only frames with the fields required by the tile API.
    frames = [f for f in frames if f.get("time") and f.get("path")]
    frames = sorted(frames, key=lambda x: int(x["time"]))
    return host, frames


# ============================================================
# TILE DOWNLOAD
# ============================================================

def _download_png(url):
    last_error = None

    for attempt in range(1, 4):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            status = r.status_code
            content_type = r.headers.get("content-type", "")
            size = len(r.content)

            if status != 200:
                last_error = f"HTTP {status}, bytes={size}, type={content_type}"
                continue

            if size < 100:
                last_error = f"HTTP 200, bytes={size}, type={content_type}"
                continue

            tile = Image.open(BytesIO(r.content)).convert("RGBA")
            arr = np.asarray(tile)
            alpha_pixels = int(np.count_nonzero(arr[:, :, 3] > 0))
            rgb_pixels = int(np.count_nonzero(np.any(arr[:, :, :3] > 3, axis=2)))

            return tile, {
                "ok": True,
                "status": status,
                "bytes": size,
                "content_type": content_type,
                "attempt": attempt,
                "alpha_pixels": alpha_pixels,
                "rgb_pixels": rgb_pixels,
            }

        except Exception as exc:
            last_error = repr(exc)

    return None, {
        "ok": False,
        "status": None,
        "bytes": 0,
        "content_type": "",
        "attempt": 3,
        "alpha_pixels": 0,
        "rgb_pixels": 0,
        "error": last_error,
    }


def radar_tile_urls(host, path, tx, ty):
    # Official Weather Maps API form:
    # {host}{path}/{size}/{z}/{x}/{y}/{color}/{options}.png
    primary = f"{host}{path}/{TILE_SIZE}/{ZOOM}/{tx}/{ty}/2/0_0.png"

    # Coordinate form is useful as a diagnostic/fallback for the center tile.
    # It should represent the same radar frame around the target point.
    coordinate = f"{host}{path}/{TILE_SIZE}/{ZOOM}/{LAT:.6f}/{LON:.6f}/2/0_0.png"
    return primary, coordinate


def coverage_tile_url(host, tx, ty):
    # Official RainViewer coverage endpoint.
    return f"{host}/v2/coverage/0/{TILE_SIZE}/{ZOOM}/{tx}/{ty}/0/0_0.png"


def download_frame(host, frame, xt, yt):
    path = frame["path"]
    timestamp = int(frame["time"])

    image = Image.new("RGBA", (TILE_SIZE * 3, TILE_SIZE * 3), (0, 0, 0, 0))

    diagnostics = []
    successful_tiles = 0
    coverage_nonblack = 0
    coverage_transparent = 0

    for dx in range(-GRID_RADIUS, GRID_RADIUS + 1):
        for dy in range(-GRID_RADIUS, GRID_RADIUS + 1):
            tx = xt + dx
            ty = yt + dy

            primary_url, coordinate_url = radar_tile_urls(host, path, tx, ty)
            tile, info = _download_png(primary_url)

            # If the x/y tile is blank, do not silently call it a radar error.
            # Record a coordinate request for the center tile so the run proves
            # whether the API can return a coordinate-centered image.
            coordinate_info = None
            if dx == 0 and dy == 0:
                _, coordinate_info = _download_png(coordinate_url)

            cov_tile, cov_info = _download_png(coverage_tile_url(host, tx, ty))
            if cov_tile is not None:
                cov = np.asarray(cov_tile)
                cov_rgb = cov[:, :, :3]
                cov_alpha = cov[:, :, 3]
                if np.any(cov_rgb > 3):
                    coverage_nonblack += 1
                if np.any(cov_alpha > 0):
                    coverage_transparent += 1

            info.update({
                "dx": dx,
                "dy": dy,
                "x": tx,
                "y": ty,
                "url": primary_url,
                "coordinate_url": coordinate_url if dx == 0 and dy == 0 else None,
                "coordinate_diagnostic": coordinate_info,
                "coverage": cov_info,
            })
            diagnostics.append(info)

            if tile is None:
                continue

            image.alpha_composite(
                tile,
                ((dx + GRID_RADIUS) * TILE_SIZE, (dy + GRID_RADIUS) * TILE_SIZE),
            )
            successful_tiles += 1

    return (
        image,
        successful_tiles,
        timestamp,
        path,
        diagnostics,
        coverage_nonblack,
        coverage_transparent,
    )


# ============================================================
# dBZ DECODING
# ============================================================

def rgba_to_dbz(image):
    arr = np.asarray(image).astype(np.float32)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    flat = rgb.reshape(-1, 3)
    distances = ((flat[:, None, :] - DBZ_RGB[None, :, :]) ** 2).sum(axis=2)
    indices = np.argmin(distances, axis=1)
    nearest_distance = np.sqrt(np.min(distances, axis=1)).reshape(rgb.shape[:2])

    dbz = DBZ_VALUES[indices].reshape(rgb.shape[:2]).astype(np.float32)

    rgb_nonblack = np.any(rgb > 3, axis=2)
    valid = (alpha > 0) & rgb_nonblack & (nearest_distance <= PALETTE_DISTANCE)
    dbz[~valid] = np.nan
    return dbz


def decode_diagnostics(image, dbz):
    arr = np.asarray(image)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    nontransparent = alpha > 0
    nonblack = np.any(rgb > 3, axis=2)
    finite = np.isfinite(dbz)

    return {
        "modo_imagen": image.mode,
        "ancho": int(image.width),
        "alto": int(image.height),
        "pixeles_totales": int(image.width * image.height),
        "pixeles_alpha": int(np.count_nonzero(nontransparent)),
        "pixeles_no_transparentes": int(np.count_nonzero(nontransparent)),
        "pixeles_rgb_no_negros": int(np.count_nonzero(nonblack)),
        "pixeles_paleta_detectados": int(np.count_nonzero(finite)),
        "pixeles_precipitacion": int(np.count_nonzero(finite)),
        "alpha_max": int(alpha.max()) if alpha.size else 0,
        "rgb_max": int(rgb.max()) if rgb.size else 0,
    }


def dbz_mask(dbz, threshold=10.0):
    return np.where(np.isfinite(dbz) & (dbz >= threshold), 255, 0).astype(np.uint8)


def dbz_metrics(dbz):
    valid = dbz[np.isfinite(dbz)]
    precip = valid[valid >= 10]

    if precip.size == 0:
        return {"dbz_max": None, "dbz_mean": None, "dbz_p90": None, "dbz_pixels": 0}

    return {
        "dbz_max": float(np.max(precip)),
        "dbz_mean": round(float(np.mean(precip)), 1),
        "dbz_p90": round(float(np.percentile(precip, 90)), 1),
        "dbz_pixels": int(precip.size),
    }


def intensity_label(dbz):
    if dbz is None:
        return "sin_precipitacion"
    if dbz < 20:
        return "debil"
    if dbz < 30:
        return "moderada"
    if dbz < 40:
        return "fuerte"
    if dbz < 50:
        return "muy_fuerte"
    if dbz < 60:
        return "severa"
    if dbz < 65:
        return "muy_severa"
    return "extrema"


# ============================================================
# COMPONENTS / TRACKING
# ============================================================

def components_from_mask(mask, dbz):
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    components = []

    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_AREA:
            continue

        ys, xs = np.where(labels == i)
        values = dbz[ys, xs]
        values = values[np.isfinite(values)]

        max_dbz = float(np.max(values)) if values.size else None
        mean_dbz = float(np.mean(values)) if values.size else None

        components.append({
            "area_px": area,
            "centroid_x": float(centers[i][0]),
            "centroid_y": float(centers[i][1]),
            "max_dbz": max_dbz,
            "mean_dbz": mean_dbz,
            "intensidad": intensity_label(max_dbz),
            "distance_km": round(
                distance_point_to_bahia(centers[i][0], centers[i][1]), 2
            ),
            "bbox": [
                int(stats[i, cv2.CC_STAT_LEFT]),
                int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]),
                int(stats[i, cv2.CC_STAT_HEIGHT]),
            ],
        })

    components.sort(
        key=lambda c: (
            c["max_dbz"] if c["max_dbz"] is not None else -999,
            c["area_px"],
        ),
        reverse=True,
    )
    return components[:MAX_COMPONENTS]


def component_speed_and_direction(old_comp, new_comp, seconds):
    if not old_comp or not new_comp or seconds <= 0:
        return None, None, None

    dx = new_comp["centroid_x"] - old_comp["centroid_x"]
    dy = new_comp["centroid_y"] - old_comp["centroid_y"]

    km_x, km_y = pixels_per_km(LAT, ZOOM, TILE_SIZE)
    east_km = dx * km_x
    south_km = dy * km_y

    distance_km = math.hypot(east_km, south_km)
    speed_kmh = distance_km / seconds * 3600.0

    if speed_kmh > MAX_TRACK_SPEED_KMH:
        return None, None, None

    # Meteorological direction: where the cell is moving toward.
    east = east_km
    north = -south_km
    degrees = (math.degrees(math.atan2(east, north)) + 360.0) % 360.0

    return speed_kmh, degrees, direction_name(degrees)


def track_components(results):
    if len(results) < 2:
        return None, None, None, 0.0

    current = results[-1]
    previous = results[-2]

    if not current.get("components") or not previous.get("components"):
        return None, None, None, 0.0

    best = None
    for new_comp in current["components"]:
        for old_comp in previous["components"]:
            dist = math.hypot(
                new_comp["centroid_x"] - old_comp["centroid_x"],
                new_comp["centroid_y"] - old_comp["centroid_y"],
            )
            if dist > 250:
                continue

            seconds = max(
                1,
                int(current["timestamp"]) - int(previous["timestamp"]),
            )
            speed, degrees, direction = component_speed_and_direction(
                old_comp, new_comp, seconds
            )
            if speed is None:
                continue

            score = dist + abs(
                (new_comp.get("max_dbz") or 0) - (old_comp.get("max_dbz") or 0)
            )
            if best is None or score < best[0]:
                best = (score, new_comp, speed, degrees, direction)

    if best is None:
        return None, None, None, 0.0

    _, comp, speed, degrees, direction = best

    confidence = 1.0
    if len(results) >= 3:
        confidence = 0.75

    return speed, degrees, direction, confidence


def projection(comp, speed_kmh, direction_degrees, minutes):
    if comp is None or speed_kmh is None or direction_degrees is None:
        return None

    distance_km = speed_kmh * minutes / 60.0
    rad = math.radians(direction_degrees)

    km_x, km_y = pixels_per_km(LAT, ZOOM, TILE_SIZE)
    dx = (math.sin(rad) * distance_km) / km_x
    dy = (-math.cos(rad) * distance_km) / km_y

    x = comp["centroid_x"] + dx
    y = comp["centroid_y"] + dy

    bx, by = bahia_pixel_position()
    dist = math.hypot((x - bx) * km_x, (y - by) * km_y)

    return {
        "minutos": minutes,
        "x": round(float(x), 2),
        "y": round(float(y), 2),
        "distancia_km_bahia": round(float(dist), 2),
    }


# ============================================================
# FRAME ANALYSIS
# ============================================================

def analyze_frame(image, timestamp, path, tiles_ok, tile_diagnostics,
                  coverage_nonblack, coverage_transparent):
    dbz = rgba_to_dbz(image)
    diagnostics = decode_diagnostics(image, dbz)
    metrics = dbz_metrics(dbz)

    mask = dbz_mask(dbz)
    components = components_from_mask(mask, dbz)

    nearest_distance = None
    if components:
        nearest_distance = min(c["distance_km"] for c in components)

    diagnostics.update({
        "tiles_ok": int(tiles_ok),
        "tiles_total": 9,
        "coverage_tiles_nonblack": int(coverage_nonblack),
        "coverage_tiles_transparent": int(coverage_transparent),
        "tile_diagnostics": tile_diagnostics,
    })

    return {
        "timestamp": int(timestamp),
        "frame_utc": frame_datetime(timestamp),
        "path": path,
        "tiles_ok": int(tiles_ok),
        "components": components,
        "area": int(metrics["dbz_pixels"]),
        "distance_km": nearest_distance,
        **metrics,
        **diagnostics,
    }


def analyze_sequence(results):
    if not results:
        return {
            "actividad": False,
            "area_px": 0,
            "distancia_km": None,
            "dbz_max": None,
            "dbz_mean": None,
            "dbz_p90": None,
            "dbz_pixels": 0,
            "intensidad": "sin_precipitacion",
            "fortalecimiento": "sin_datos",
            "cambio_area_pct": None,
            "velocidad_kmh": None,
            "direccion": None,
            "direccion_grados": None,
            "movimiento_hacia_bahia": False,
            "eta_minutos": None,
            "proyecciones": {},
            "proyeccion_30_min": None,
            "proyeccion_60_min": None,
            "confianza_movimiento": 0.0,
            "nucleos": [],
            "nucleo_principal_id": None,
            "estado": "sin_datos",
        }

    current = results[-1]

    activity = bool(current.get("dbz_pixels", 0) > 0 and current.get("components"))
    area = int(current.get("area", 0))
    dbz_max = current.get("dbz_max")
    dbz_mean = current.get("dbz_mean")
    dbz_p90 = current.get("dbz_p90")
    distance_km = current.get("distance_km")

    strengthening = "sin_datos"
    change_area_pct = None

    if len(results) >= 2:
        previous = results[-2]
        prev_area = int(previous.get("area", 0))
        if prev_area > 0:
            change_area_pct = (area - prev_area) / prev_area * 100.0
            if change_area_pct > 10:
                strengthening = "fortaleciendo"
            elif change_area_pct < -10:
                strengthening = "debilitando"
            else:
                strengthening = "estable"

    speed, degrees, direction, confidence = track_components(
        results[-TRACK_FRAMES:]
    )

    main_component = current["components"][0] if current.get("components") else None
    projections = {}
    if main_component and speed is not None and degrees is not None:
        for minutes in PROJECTION_MINUTES:
            projections[str(minutes)] = projection(
                main_component, speed, degrees, minutes
            )

    movement_toward_bahia = False
    eta = None

    if main_component and speed is not None and degrees is not None:
        bx, by = bahia_pixel_position()
        dx = bx - main_component["centroid_x"]
        dy = by - main_component["centroid_y"]

        km_x, km_y = pixels_per_km(LAT, ZOOM, TILE_SIZE)
        target_east = dx * km_x
        target_north = -dy * km_y
        target_angle = (
            math.degrees(math.atan2(target_east, target_north)) + 360
        ) % 360

        angular_diff = abs((degrees - target_angle + 180) % 360 - 180)
        movement_toward_bahia = angular_diff <= 45.0

        if movement_toward_bahia and distance_km is not None and speed > 1:
            eta = round(distance_km / speed * 60.0, 1)

    state = "precipitacion_detectada" if activity else "sin_precipitacion_detectada"

    return {
        "actividad": activity,
        "area_px": area,
        "distancia_km": distance_km,
        "dbz_max": dbz_max,
        "dbz_mean": dbz_mean,
        "dbz_p90": dbz_p90,
        "dbz_pixels": int(current.get("dbz_pixels", 0)),
        "intensidad": intensity_label(dbz_max),
        "fortalecimiento": strengthening,
        "cambio_area_pct": (
            round(change_area_pct, 1) if change_area_pct is not None else None
        ),
        "velocidad_kmh": round(speed, 1) if speed is not None else None,
        "direccion": direction,
        "direccion_grados": round(degrees, 1) if degrees is not None else None,
        "movimiento_hacia_bahia": movement_toward_bahia,
        "eta_minutos": eta,
        "proyecciones": projections,
        "proyeccion_30_min": projections.get("30"),
        "proyeccion_60_min": projections.get("60"),
        "confianza_movimiento": round(confidence, 2),
        "nucleos": current.get("components", []),
        "nucleo_principal_id": 0 if main_component else None,
        "estado": state,
    }


# ============================================================
# HISTORY / OUTPUT
# ============================================================

def save_history(image, timestamp):
    path = os.path.join(HISTORY_DIR, f"{int(timestamp)}.png")
    image.save(path, format="PNG")
    return path


def trim_history():
    files = []
    for name in os.listdir(HISTORY_DIR):
        if not name.endswith(".png"):
            continue
        try:
            files.append((int(name[:-4]), name))
        except ValueError:
            continue

    files.sort()
    for _, name in files[:-MAX_HISTORY_FRAMES]:
        try:
            os.remove(os.path.join(HISTORY_DIR, name))
        except OSError:
            pass


def append_features(results):
    rows = []
    for r in results:
        rows.append({
            "timestamp": r["timestamp"],
            "frame_utc": r["frame_utc"],
            "tiles_ok": r["tiles_ok"],
            "area_px": r["area"],
            "dbz_max": r["dbz_max"],
            "dbz_mean": r["dbz_mean"],
            "dbz_p90": r["dbz_p90"],
            "dbz_pixels": r["dbz_pixels"],
            "distance_km": r["distance_km"],
            "components": len(r["components"]),
        })

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    existing = os.path.exists(FEATURES_CSV)

    with open(FEATURES_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not existing:
            writer.writeheader()
        writer.writerows(rows)


def make_output(host, frames_analyzed, results, sequence, xt, yt):
    current = results[-1] if results else {}
    coverage_state = "desconocida"

    if current:
        nonblack = current.get("coverage_tiles_nonblack", 0)
        transparent = current.get("coverage_tiles_transparent", 0)
        if nonblack > 0:
            coverage_state = "con_cobertura_radar"
        elif transparent == 0:
            coverage_state = "sin_cobertura_radar_detectada"

    return {
        "fuente": "RainViewer",
        "actualizado": datetime.now(timezone.utc).isoformat(),
        "latitud": LAT,
        "longitud": LON,
        "frames_analizados": len(frames_analyzed),
        "frame_actual": current.get("timestamp"),
        "frame_actual_utc": current.get("frame_utc"),
        "tiles_ok": current.get("tiles_ok", 0),
        "actividad": sequence["actividad"],
        "area_px": sequence["area_px"],
        "distancia_km": sequence["distancia_km"],
        "dbz_max": sequence["dbz_max"],
        "dbz_mean": sequence["dbz_mean"],
        "dbz_p90": sequence["dbz_p90"],
        "dbz_pixels": sequence["dbz_pixels"],
        "intensidad": sequence["intensidad"],
        "fortalecimiento": sequence["fortalecimiento"],
        "cambio_area_pct": sequence["cambio_area_pct"],
        "velocidad_kmh": sequence["velocidad_kmh"],
        "direccion": sequence["direccion"],
        "direccion_grados": sequence["direccion_grados"],
        "movimiento_hacia_bahia": sequence["movimiento_hacia_bahia"],
        "eta_minutos": sequence["eta_minutos"],
        "proyecciones": sequence["proyecciones"],
        "proyeccion_30_min": sequence["proyeccion_30_min"],
        "proyeccion_60_min": sequence["proyeccion_60_min"],
        "confianza_movimiento": sequence["confianza_movimiento"],
        "nucleos": sequence["nucleos"],
        "nucleo_principal_id": sequence["nucleo_principal_id"],
        "cobertura_radar": coverage_state,
        "tile_x": xt,
        "tile_y": yt,
        "estado": sequence["estado"],
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 64)
    print("CLIMAAR RADAR RAINVIEWER V9.0")
    print("Tiles oficiales + cobertura + diagnóstico dBZ")
    print("=" * 64)

    host, frames = fetch_rainviewer()
    xt, yt = latlon_to_tile(LAT, LON, ZOOM)

    print(f"Host: {host}")
    print(f"Tile Bahía Blanca: x={xt}, y={yt}")
    print(f"Frames disponibles: {len(frames)}")
    print(f"Paleta Universal Blue: {len(DBZ_HEX)} colores")
    print(f"Zoom: {ZOOM} | tamaño tile: {TILE_SIZE}")
    print()

    selected = frames[-MAX_ANALYSIS_FRAMES:]
    results = []

    for index, frame in enumerate(selected, 1):
        print("=" * 48)
        print(f"Procesando frame {index}/{len(selected)}")
        print(f"timestamp: {frame['time']}")

        (
            image,
            tiles_ok,
            timestamp,
            path,
            diagnostics,
            coverage_nonblack,
            coverage_transparent,
        ) = download_frame(host, frame, xt, yt)

        dbz = rgba_to_dbz(image)
        diag = decode_diagnostics(image, dbz)
        metrics = dbz_metrics(dbz)

        print(f"tiles OK: {tiles_ok}/9")

        for item in diagnostics:
            print(
                f"  tile x={item['x']} y={item['y']} "
                f"HTTP={item.get('status')} bytes={item.get('bytes')}"
            )
            if item.get("coordinate_diagnostic") is not None:
                c = item["coordinate_diagnostic"]
                print(
                    f"    center-coordinate: HTTP={c.get('status')} "
                    f"bytes={c.get('bytes')} alpha={c.get('alpha_pixels')} "
                    f"rgb={c.get('rgb_pixels')}"
                )

        print(
            f"  cobertura: tiles_no_negros={coverage_nonblack} "
            f"tiles_transparentes={coverage_transparent}"
        )
        print(
            f"alpha={diag['pixeles_alpha']} "
            f"rgb={diag['pixeles_rgb_no_negros']} "
            f"paleta={diag['pixeles_paleta_detectados']} "
            f"precip={diag['pixeles_precipitacion']} "
            f"dBZmax={metrics['dbz_max']}"
        )

        result = analyze_frame(
            image,
            timestamp,
            path,
            tiles_ok,
            diagnostics,
            coverage_nonblack,
            coverage_transparent,
        )
        results.append(result)

        save_history(image, timestamp)

        # Keep the newest frame available to the rest of ClimaAR.
        image.save(ACTUAL_FILE, format="PNG")

    trim_history()
    append_features(results)

    sequence = analyze_sequence(results)
    output = make_output(host, selected, results, sequence, xt, yt)

    with open(NOWCAST_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 64)
    print("RESULTADO NOWCAST / RADAR")
    print("=" * 64)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print("=" * 64)


if __name__ == "__main__":
    main()
