import csv
import json
import math
import os
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import requests


LAT = -38.71
LON = -62.26

ZOOM = 7
SIZE = 512

FRAMES = 6
MAX_HISTORY = 144

RADAR_DIR = "data/radar"
HISTORY_DIR = os.path.join(RADAR_DIR, "historico")

ACTUAL_PATH = os.path.join(RADAR_DIR, "actual.png")
NOWCAST_PATH = os.path.join(RADAR_DIR, "radar_nowcast.json")
STATUS_PATH = os.path.join(RADAR_DIR, "status.json")

FEATURES_PATH = os.path.join(
    RADAR_DIR,
    "radar_features_rainviewer.csv"
)

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

RAINVIEWER_API = "https://api.rainviewer.com/public/weather-maps.json"

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "ClimaAR/1.0",
        "Accept": "application/json,image/png,image/*",
    }
)


def ensure_dirs():
    os.makedirs(RADAR_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)


def utc_now():
    return datetime.now(timezone.utc)


def iso_from_timestamp(ts):
    return datetime.fromtimestamp(
        int(ts),
        tz=timezone.utc
    ).isoformat()


def download_json(url):
    r = SESSION.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def get_radar_frames():
    data = download_json(RAINVIEWER_API)

    host = data.get("host", "").rstrip("/")
    radar = data.get("radar", {})

    past = radar.get("past", [])

    if not past:
        raise RuntimeError("RainViewer no devolvió frames históricos")

    frames = []

    for item in past:
        ts = item.get("time")
        path = item.get("path")

        if not ts or not path:
            continue

        frames.append(
            {
                "time": int(ts),
                "path": path,
            }
        )

    frames.sort(key=lambda x: x["time"])

    if not frames:
        raise RuntimeError("No hay frames válidos de RainViewer")

    return host, frames


def build_image_url(host, path):
    return (
        f"{host}{path}/"
        f"{SIZE}/{ZOOM}/{LAT}/{LON}/"
        f"2/1_1.png"
    )


def download_image(url):
    r = SESSION.get(url, timeout=40)
    r.raise_for_status()

    data = np.frombuffer(r.content, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError("No se pudo decodificar la imagen")

    return image


def precipitation_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    b, g, r = cv2.split(image)

    # Detecta colores típicos de precipitación de RainViewer.
    colorful = (
        (np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])) > 25
    )

    bright = (
        (r > 70)
        | (g > 70)
        | (b > 70)
    )

    mask = colorful & bright

    mask = (mask.astype(np.uint8) * 255)

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


def analyze_frame(image):
    mask = precipitation_mask(image)

    area = int(cv2.countNonZero(mask))

    h, w = mask.shape

    coverage = (
        float(area) / float(w * h)
        if w and h
        else 0.0
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    nuclei = []

    for contour in contours:
        contour_area = cv2.contourArea(contour)

        if contour_area < 20:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)

        moments = cv2.moments(contour)

        if moments["m00"] != 0:
            cx = moments["m
