#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from scripts.rma10_features import V1_FEATURES, build_v1_feature_dict, pixel_to_latlon

ROOT = Path(".")
HIST = ROOT / "data" / "radar" / "historico"
OUT = ROOT / "data" / "radar" / "rma10_dataset_historico.csv"
SUMMARY = ROOT / "data" / "radar" / "rma10_dataset_resumen.json"

TIMESTAMP_RE = re.compile(r"^(\d{14})\.png$")
RADAR_LAT = -38.7342
RADAR_LON = -62.1654
BAHIA_LAT = -38.7196
BAHIA_LON = -62.2724
RANGE_KM = 240.0

MAX_MISSED = 2
HORIZONS = (5, 10, 20, 30)

MAX_LABEL_ERROR_MIN = {
    5: 5.0,
    10: 7.5,
    20: 10.0,
    30: 15.0,
}

MIN_AREA_RATIO = 0.80
MIN_SPEED_PX_MIN = 0.05
TOWARD_BAHIA_MAX_ANGLE_DEG = 45.0

NIVELES = {
    0: "FONDO",
    1: "MUY_DEBIL",
    2: "DEBIL",
    3: "MODERADA",
    4: "FUERTE",
    5: "MUY_FUERTE",
    6: "SEVERA",
}


def clasificar_color(r, g, b):
    if max(r, g, b) < 25:
        return 0
    if b > g and g > r * 1.5:
        return 1
    if g > r * 1.35 and g >= b * 0.95:
        return 2
    if r > 100 and g > 100 and b < 100 and abs(r - g) < 100:
        return 3
    if r > 130 and 60 < g < 180 and b < 90 and r > g * 1.15:
        return 4
    if r > 130 and r > g * 1.35 and r > b * 1.35:
        return 5
    if r > 100 and b > 100 and r > g * 1.3:
        return 6
    return 0


def detect(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    pix = im.load()
    cls = [[0] * w for _ in range(h)]

    for y in range(h):
        row = cls[y]
        for x in range(w):
            r, g, b = pix[x, y]
            row[x] = clasificar_color(r, g, b)

    seen = bytearray(w * h)
    out = []

    neigh = (
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),             (1, 0),
        (-1, 1),  (0, 1),    (1, 1),
    )

    for y0 in range(h):
        for x0 in range(w):
            idx = y0 * w + x0

            if seen[idx] or cls[y0][x0] <= 0:
                seen[idx] = 1
                continue

            q = deque([(x0, y0)])
            seen[idx] = 1
            pts = []

            while q:
                x, y = q.popleft()
                pts.append((x, y))

                for dx, dy in neigh:
                    nx, ny = x + dx, y + dy

                    if 0 <= nx < w and 0 <= ny < h:
                        ni = ny * w + nx

                        if not seen[ni] and cls[ny][nx] > 0:
                            seen[ni] = 1
                            q.append((nx, ny))

            if len(pts) < 10:
                continue

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]

            intensity = max(cls[y][x] for x, y in pts)

            out.append({
                "x": sum(xs) / len(xs),
                "y": sum(ys) / len(ys),
                "pixeles": len(pts),
                "ancho": max(xs) - min(xs) + 1,
                "alto": max(ys) - min(ys) + 1,
                "intensidad": intensity,
            })

    return w, h, out


def ts_from(path):
    match = TIMESTAMP_RE.match(path.name)

    if not match:
        return None

    try:
        dt = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None

    if not 2000 <= dt.year <= 2100:
        return None

    return dt.replace(tzinfo=timezone.utc)


def xy_to_geo(x, y, w, h):
    return pixel_to_latlon(x, y, w, h)


def distance_km(lat1, lon1, lat2, lon2):
    r = 6371.0
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

    return 2 * r * math.asin(min(1, math.sqrt(a)))


def bearing(lat1, lon1, lat2, lon2):
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)

    y = math.sin(dl) * math.cos(p2)
    x = (
        math.cos(p1) * math.sin(p2)
        - math.sin(p1) * math.cos(p2) * math.cos(dl)
    )

    return (math.degrees(math.atan2(y, x)) + 360) % 360


def angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def build_tracks(frames):
    tracks = []
    next_id = 1

    for frame in frames:
        nuclei = frame["nuclei"]
        dt = frame["dt"]

        for nucleus in nuclei:
            nucleus["track_id"] = None

        candidates = []

        for track in tracks:
            if not track["active"] or track["missed"] > MAX_MISSED:
                continue

            last = track["obs"][-1]

            elapsed = (dt - last["dt"]).total_seconds() / 60.0

            if elapsed <= 0 or elapsed > 60:
                continue

            if len(track["obs"]) >= 2:
                previous = track["obs"][-2]

                previous_elapsed = (
                    last["dt"] - previous["dt"]
                ).total_seconds() / 60.0

                vx = (
                    (last["x"] - previous["x"]) / previous_elapsed
                    if previous_elapsed > 0 else 0.0
                )
                vy = (
                    (last["y"] - previous["y"]) / previous_elapsed
                    if previous_elapsed > 0 else 0.0
                )

                predicted_x = last["x"] + vx * elapsed
                predicted_y = last["y"] + vy * elapsed
            else:
                predicted_x = last["x"]
                predicted_y = last["y"]

            max_pixels = max(20.0, min(180.0, 20.0 + elapsed * 4.0))

            for index, nucleus in enumerate(nuclei):
                distance = math.hypot(
                    nucleus["x"] - predicted_x,
                    nucleus["y"] - predicted_y,
                )

                if distance <= max_pixels:
                    candidates.append((distance, index, track))

        used_nuclei = set()
        used_tracks = set()

        for distance, index, track in sorted(
            candidates,
            key=lambda item: item[0],
        ):
            if index in used_nuclei or track["id"] in used_tracks:
                continue

            nucleus = nuclei[index]
            nucleus["track_id"] = track["id"]

            track["obs"].append({
                "dt": dt,
                **nucleus,
            })

            track["missed"] = 0
            used_nuclei.add(index)
            used_tracks.add(track["id"])

        for track in tracks:
            if track["active"] and track["id"] not in used_tracks:
                track["missed"] += 1

                if track["missed"] > MAX_MISSED:
                    track["active"] = False

        for nucleus in nuclei:
            if nucleus["track_id"] is None:
                track = {
                    "id": next_id,
                    "active": True,
                    "missed": 0,
                    "obs": [{
                        "dt": dt,
                        **nucleus,
                    }],
                }

                tracks.append(track)
                nucleus["track_id"] = next_id
                next_id += 1

        frame["tracks"] = {
            nucleus["track_id"]: nucleus
            for nucleus in nuclei
        }

    return tracks


def nearest_future(obs, target, max_error):
    best = None
    best_error = None

    for observation in obs:
        if observation["dt"] < target:
            continue

        error = abs(
            (observation["dt"] - target).total_seconds() / 60.0
        )

        if error <= max_error and (
            best_error is None or error < best_error
        ):
            best = observation
            best_error = error

    return best


def interpolate_future_5min(current, observations, target):
    """
    Estima T+5 entre T0 y el siguiente frame real.
    Solo se utiliza cuando el siguiente frame está
    entre 8 y 15 minutos después de T0.
    """
    next_observation = None

    for observation in observations:
        if observation["dt"] <= current["dt"]:
            continue

        next_observation = observation
        break

    if next_observation is None:
        return None

    interval_min = (
        next_observation["dt"] - current["dt"]
    ).total_seconds() / 60.0

    if interval_min < 8.0 or interval_min > 15.0:
        return None

    elapsed_min = (target - current["dt"]).total_seconds() / 60.0

    if elapsed_min <= 0.0 or elapsed_min >= interval_min:
        return None

    alpha = elapsed_min / interval_min
    future = dict(next_observation)

    future["dt"] = target
    future["x"] = current["x"] + alpha * (
        next_observation["x"] - current["x"]
    )
    future["y"] = current["y"] + alpha * (
        next_observation["y"] - current["y"]
    )
    future["intensidad"] = current["intensidad"] + alpha * (
        next_observation["intensidad"] - current["intensidad"]
    )
    future["pixeles"] = current["pixeles"] + alpha * (
        next_observation["pixeles"] - current["pixeles"]
    )

    return future


def label_evolution(current, future):
    intensity_delta = future["intensidad"] - current["intensidad"]
    area_ratio = future["pixeles"] / max(current["pixeles"], 1)

    strengthened = int(
        (
            intensity_delta >= 1
            and area_ratio >= MIN_AREA_RATIO
        )
        or intensity_delta >= 2
    )

    weakened = int(
        (
            intensity_delta <= -1
            and area_ratio <= 1.25
        )
        or intensity_delta <= -2
    )

    return intensity_delta, area_ratio, strengthened, weakened


def main():
    files = []

    for path in HIST.glob("*.png"):
        dt = ts_from(path)

        if dt is not None:
            files.append((dt, path))

    files.sort()

    valid = []
    skipped = 0

    for dt, path in files:
        try:
            image = Image.open(path).convert("RGB")

            if len(image.getcolors(maxcolors=10_000_000) or []) < 5:
                skipped += 1
                continue

            width, height, nuclei = detect(path)

            valid.append({
                "dt": dt,
                "path": path.name,
                "w": width,
                "h": height,
                "nuclei": nuclei,
            })

        except Exception:
            skipped += 1

    if len(valid) < 3:
        raise SystemExit("No hay suficientes frames RMA10 validos.")

    frame_by_dt = {frame["dt"]: frame for frame in valid}
    tracks = build_tracks(valid)
    rows = []

    for track in tracks:
        observations = track["obs"]

        for observation in observations:
            frame = frame_by_dt.get(observation["dt"])

            if frame is None:
                continue

            history = [
                {
                    "dt": item["dt"],
                    "x": item["x"],
                    "y": item["y"],
                    "intensidad": item["intensidad"],
                    "pixeles": item["pixeles"],
                    "ancho": item["ancho"],
                    "alto": item["alto"],
                }
                for item in observations
                if item["dt"] <= observation["dt"]
            ]

            current_features = build_v1_feature_dict(
                {**observation, "timestamp": observation["dt"]},
                history,
                frame["w"],
                frame["h"],
            )

            lat, lon = xy_to_geo(
                observation["x"],
                observation["y"],
                frame["w"],
                frame["h"],
            )

            distance = distance_km(lat, lon, BAHIA_LAT, BAHIA_LON)
            bearing_to_bahia = bearing(
                lat,
                lon,
                BAHIA_LAT,
                BAHIA_LON,
            )

            for horizon in HORIZONS:
                target = observation["dt"] + timedelta(minutes=horizon)

                if horizon == 5:
                    future = interpolate_future_5min(
                        observation,
                        observations,
                        target,
                    )
                else:
                    future = nearest_future(
                        observations,
                        target,
                        4.0,
                    )

                if future is None:
                    continue

                elapsed = (
                    future["dt"] - observation["dt"]
                ).total_seconds() / 60.0

                if elapsed <= 0:
                    continue

                dx = future["x"] - observation["x"]
                dy = future["y"] - observation["y"]
                displacement = math.hypot(dx, dy)
                speed = displacement / elapsed

                motion_bearing = (
                    math.degrees(math.atan2(dx, -dy)) + 360
                ) % 360

                future_lat, future_lon = xy_to_geo(
                    future["x"],
                    future["y"],
                    frame["w"],
                    frame["h"],
                )

                future_distance = distance_km(
                    future_lat,
                    future_lon,
                    BAHIA_LAT,
                    BAHIA_LON,
                )

                (
                    intensity_delta,
                    area_ratio,
                    strengthened,
                    weakened,
                ) = label_evolution(observation, future)

                toward_bahia = int(
                    angle_diff(
                        motion_bearing,
                        bearing_to_bahia,
                    ) <= TOWARD_BAHIA_MAX_ANGLE_DEG
                    and speed > MIN_SPEED_PX_MIN
                )

                rows.append({
                    "frame_utc": (
                        observation["dt"].isoformat().replace("+00:00", "Z")
                    ),
                    "track_id": track["id"],
                    "horizonte_min": horizon,
                    "x": round(observation["x"], 3),
                    "y": round(observation["y"], 3),
                    "area_px": observation["pixeles"],
                    "width_px": observation["ancho"],
                    "height_px": observation["alto"],
                    "intensity": observation["intensidad"],
                    "lat": round(current_features["lat"], 6),
                    "lon": round(current_features["lon"], 6),
                    "distance_km": round(current_features["distance_km"], 3),
                    "bearing_to_bahia_deg": round(
                        current_features["bearing_to_bahia_deg"], 2
                    ),
                    "future_intensity": future["intensidad"],
                    "intensity_delta": intensity_delta,
                    "strengthened": strengthened,
                    "weakened": weakened,
                    "future_x": round(future["x"], 3),
                    "future_y": round(future["y"], 3),
                    "displacement_px": round(
                        current_features["displacement_px"], 3
                    ),
                    "speed_px_min": round(
                        current_features["speed_px_min"], 4
                    ),
                    "motion_bearing_deg": round(
                        current_features["motion_bearing_deg"], 2
                    ),
                    "motion_valid": int(current_features["motion_valid"]),
                    "motion_interval_min": round(
                        current_features["motion_interval_min"], 3
                    ),
                    "past_intensity_delta": round(
                        current_features["past_intensity_delta"], 3
                    ),
                    "past_area_ratio": round(
                        current_features["past_area_ratio"], 4
                    ),
                    "future_displacement_px": round(displacement, 3),
                    "future_speed_px_min": round(speed, 4),
                    "future_motion_bearing_deg": round(motion_bearing, 2),
                    "future_timestamp_utc": (
                        future["dt"].isoformat().replace("+00:00", "Z")
                    ),
                    "label_elapsed_min": round(elapsed, 3),
                    "label_error_min": round(abs(elapsed - horizon), 3),
                    "label_source": (
                        "INTERPOLATED_5M" if horizon == 5 else "OBSERVED"
                    ),
                    "toward_bahia": toward_bahia,
                    "future_distance_km": round(future_distance, 3),
                    "label_valid": 1,
                    "source": "SMN RMA10",
                    "future_area_ratio": round(area_ratio, 4),
                })

    if not rows:
        raise SystemExit("No se pudieron construir registros de entrenamiento.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())

    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": "rma10-causal-v2",
        "migration_version": "rma10-causal-migration-2",
        "features_version": "rma10-causal-features-v2",
        "radio_operativo_km": 240.0,
        "features_causales": list(V1_FEATURES),
        "tracking_source": "HISTORICAL_LEGACY_TRACKER",
        "frames_encontrados": len(files),
        "frames_validos": len(valid),
        "frames_descartados": skipped,
        "registros_entrenamiento": len(rows),
        "horizontes": list(HORIZONS),
        "tracks": len({row["track_id"] for row in rows}),
        "strengthened": sum(row["strengthened"] for row in rows),
        "weakened": sum(row["weakened"] for row in rows),
        "toward_bahia": sum(row["toward_bahia"] for row in rows),
        "generated_utc": (
            datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        ),
    }

    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
