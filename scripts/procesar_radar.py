#!/usr/bin/env python3
"""Decode the official RMA10 ZH_MAX palette and append radar features."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(".")
PALETTE_PATH = ROOT / "config/rma10_palette.json"
IMAGE_PATH = ROOT / "data/radar/latest.png"
OUT = ROOT / "data/radar/radar_features.csv"

THRESHOLDS = [20, 30, 40, 45, 50, 55, 60]


def load_palette() -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(PALETTE_PATH.read_text(encoding="utf-8"))
    dbz = np.arange(data["min_dbz"], data["max_dbz"] + 1, data["step_dbz"])
    colors = np.asarray(data["colors_rgb"], dtype=np.float32)
    if len(colors) != len(dbz):
        raise RuntimeError("La tabla de colores RMA10 no coincide con el rango dBZ.")
    return dbz.astype(np.float32), colors


def crop_radar(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    left, top, right, bottom = 0.01, 0.113, 0.82, 0.748
    x0, x1 = int(w * left), int(w * right)
    y0, y1 = int(h * top), int(h * bottom)
    return arr[y0:y1, x0:x1], (x0, y0)


def decode(arr: np.ndarray, dbz_values: np.ndarray, palette: np.ndarray):
    rgb = arr[..., :3].astype(np.float32)
    flat = rgb.reshape(-1, 3)

    # The radar field contains black map background, white labels/borders and
    # other overlays. Only accept pixels with a sufficiently close match to
    # the calibrated SMN colorbar.
    distances = np.empty((flat.shape[0], palette.shape[0]), dtype=np.float32)
    for start in range(0, flat.shape[0], 100_000):
        block = flat[start:start + 100_000]
        distances[start:start + 100_000] = np.sum(
            (block[:, None, :] - palette[None, :, :]) ** 2, axis=2
        )

    idx = np.argmin(distances, axis=1)
    best = np.sqrt(np.min(distances, axis=1))

    # 30 RGB units is deliberately conservative to reject labels/borders.
    valid = best <= 30.0

    values = dbz_values[idx]
    values[~valid] = np.nan

    return values.reshape(arr.shape[:2]), best.reshape(arr.shape[:2])


def centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return math.nan, math.nan
    return float(xs.mean()), float(ys.mean())


def process() -> dict:
    if not IMAGE_PATH.exists():
        raise SystemExit("No existe data/radar/latest.png")

    dbz_values, palette = load_palette()
    image = Image.open(IMAGE_PATH).convert("RGB")
    full = np.asarray(image)
    radar, (x0, y0) = crop_radar(full)

    values, distances = decode(radar, dbz_values, palette)

    valid = np.isfinite(values)
    if not valid.any():
        raise RuntimeError("No se pudieron decodificar píxeles del RMA10.")

    valid_values = values[valid]
    row = {
        "frame_time": str(int(datetime.now(timezone.utc).timestamp())),
        "frame_utc": datetime.now(timezone.utc).isoformat(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "SMN RMA10 ZH_MAX",
        "image_path": str(IMAGE_PATH),
        "width": int(full.shape[1]),
        "height": int(full.shape[0]),
        "crop_x0": x0,
        "crop_y0": y0,
        "valid_pixels": int(valid.sum()),
        "mean_dbz": round(float(np.mean(valid_values)), 3),
        "max_dbz": round(float(np.max(valid_values)), 3),
    }

    for threshold in THRESHOLDS:
        mask = valid & (values >= threshold)
        row[f"pixels_ge_{threshold}dbz"] = int(mask.sum())
        cx, cy = centroid(mask)
        row[f"centroid_x_ge_{threshold}dbz"] = "" if math.isnan(cx) else round(cx, 3)
        row[f"centroid_y_ge_{threshold}dbz"] = "" if math.isnan(cy) else round(cy, 3)

    # Use UTC capture time from the file metadata if available; the workflow
    # calls this immediately after capture, so current UTC is the fallback.
    fields = list(row.keys())
    OUT.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if OUT.exists():
        with OUT.open("r", encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))

    # Avoid duplicate frame rows if the workflow is manually retried.
    frame_time = row["frame_time"]
    existing = [r for r in existing if r.get("frame_time") != frame_time]
    existing.append(row)
    existing.sort(key=lambda r: int(float(r["frame_time"])))

    # Preserve all known columns from older rows while ensuring the new schema exists.
    all_fields = fields[:]
    for old in existing:
        for key in old.keys():
            if key not in all_fields:
                all_fields.append(key)

    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(existing)

    print(json.dumps(row, indent=2))
    return row


if __name__ == "__main__":
    process()
