from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import requests
from PIL import Image


# ============================================================
# CLIMAAR - MOTOR OPERATIVO
#
# RainViewer + Nowcast RainViewer + SAZB + modelo historico
#
# El modelo historico es una firma atmosferica de contexto.
# NO se presenta como probabilidad meteorologica calibrada.
# ============================================================

LAT = -38.71
LON = -62.26

ZOOM = 7
TILE_SIZE = 512
GRID_RADIUS = 1

DATA_DIR = Path("data/radar")
MODEL_DIR = Path("modelo")

RADAR_IMAGE = DATA_DIR / "actual.png"
STATUS_FILE = DATA_DIR / "status.json"
NOWCAST_FILE = DATA_DIR / "radar_nowcast.json"

SAZB_FILE = Path("data/sazb/status.json")

MODEL_FILE = MODEL_DIR / "climaar_modelo_historico.joblib"
FEATURES_FILE = MODEL_DIR / "climaar_features.json"
METRICS_FILE = MODEL_DIR / "climaar_modelo_historico_metricas.json"

RAINVIEWER_API = (
    "https://api.rainviewer.com/public/weather-maps.json"
)

OPEN_METEO_API = (
    "https://api.open-meteo.com/v1/forecast"
)


# ============================================================
# UTILIDADES
# ============================================================

def numero(valor, default=None):
    try:
        x = float(valor)

        if math.isfinite(x):
            return x

    except (TypeError, ValueError):
        pass

    return default


def ahora_utc():
    return datetime.now(timezone.utc).isoformat()


def guardar_status(data):
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    STATUS_FILE.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


def latlon_to_tile(lat, lon, zoom):
    n = 2.0 ** zoom

    x = int(
        (lon + 180.0)
        / 360.0
        * n
    )

    lat_rad = math.radians(lat)

    y = int(
        (
            1.0
            - math.asinh(
                math.tan(lat_rad)
            ) / math.pi
        )
        / 2.0
        * n
    )

    return x, y


# ============================================================
# RADAR RAINVIEWER
# ============================================================

def descargar_radar():

    print("=" * 70)
    print("CLIMAAR - RADAR RAINVIEWER")
    print("=" * 70)

    response = requests.get(
        RAINVIEWER_API,
        timeout=30,
        headers={
            "User-Agent": "ClimaAR/5.0.1"
        }
    )

    response.raise_for_status()

    data = response.json()

    host = (
        data.get(
            "host",
            "https://tilecache.rainviewer.com"
        )
        .rstrip("/")
    )

    frames = (
        data
        .get("radar", {})
        .get("past", [])
    )

    if not frames:
        raise RuntimeError(
            "RainViewer no devolvio frames radar."
        )

    frame = frames[-1]

    path = frame.get("path")
    timestamp = frame.get("time")

    if not path or not timestamp:
        raise RuntimeError(
            "Frame radar invalido."
        )

    xt, yt = latlon_to_tile(
        LAT,
        LON,
        ZOOM
    )

    grid = GRID_RADIUS * 2 + 1

    image_size = TILE_SIZE * grid

    image = Image.new(
        "RGBA",
        (image_size, image_size),
        (0, 0, 0, 0)
    )

    tiles_ok = 0

    for dx in range(
        -GRID_RADIUS,
        GRID_RADIUS + 1
    ):

        for dy in range(
            -GRID_RADIUS,
            GRID_RADIUS + 1
        ):

            tile_x = xt + dx
            tile_y = yt + dy

            url = (
                f"{host}"
                f"{path}"
                f"/{TILE_SIZE}"
                f"/{ZOOM}"
                f"/{tile_x}"
                f"/{tile_y}"
                f"/2/1_1.png"
            )

            try:

                r = requests.get(
                    url,
                    timeout=20
                )

                if (
                    r.status_code != 200
                    or len(r.content) < 100
                ):
                    continue

                tile = (
                    Image
                    .open(
                        BytesIO(r.content)
                    )
                    .convert("RGBA")
                )

                image.paste(
                    tile,
                    (
                        (dx + GRID_RADIUS)
                        * TILE_SIZE,
                        (dy + GRID_RADIUS)
                        * TILE_SIZE
                    ),
                    tile
                )

                tiles_ok += 1

            except Exception as exc:

                print(
                    f"Error tile {tile_x}/{tile_y}: {exc}"
                )

    if tiles_ok == 0:
        raise RuntimeError(
            "No se pudo descargar ninguna tesela radar."
        )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image.save(
        RADAR_IMAGE
    )

    print(
        f"Radar guardado: {RADAR_IMAGE}"
    )

    print(
        f"Tiles correctas: {tiles_ok}/9"
    )

    return {
        "timestamp": int(timestamp),
        "tiles_ok": tiles_ok,
        "frame_path": path
    }


# ============================================================
# ANALISIS RADAR
# ============================================================

def analizar_radar():

    image = cv2.imread(
        str(RADAR_IMAGE),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise RuntimeError(
            "No se pudo leer actual.png."
        )

    if (
        image.ndim == 3
        and image.shape[2] >= 4
    ):

        alpha = image[:, :, 3]

    else:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        alpha = gray

    mask = (
        alpha > 15
    ).astype(
        np.uint8
    ) * 255

    area = int(
        np.count_nonzero(mask)
    )

    if area == 0:

        return {
            "actividad": False,
            "area_px": 0,
            "distancia_aprox_km": None,
            "estado": "sin_precipitacion_detectada"
        }

    ys, xs = np.where(
        mask > 0
    )

    h, w = mask.shape

    cx = w / 2.0
    cy = h / 2.0

    distance_px = float(
        np.min(
            np.sqrt(
                (xs - cx) ** 2
                +
                (ys - cy) ** 2
            )
        )
    )

    distance_km = round(
        distance_px * 0.85,
        1
    )

    return {
        "actividad": True,
        "area_px": area,
        "distancia_aprox_km": distance_km,
        "estado": "precipitacion_detectada"
    }


# ============================================================
# NOWCAST RAINVIEWER
# ============================================================

def cargar_nowcast_radar():

    if not NOWCAST_FILE.exists():

        return {
            "disponible": False,
            "motivo":
                "Todavia no existe radar_nowcast.json."
        }

    try:

        raw = json.loads(
            NOWCAST_FILE.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            raw,
            dict
        ):

            return {
                "disponible": False,
                "motivo":
                    "radar_nowcast.json no contiene un objeto JSON."
            }

        nowcast = raw.get(
            "nowcast",
            {}
        )

        historial = raw.get(
            "historial",
            {}
        )

        if not isinstance(
            nowcast,
            dict
        ):
            nowcast = {}

        if not isinstance(
            historial,
            dict
        ):
            historial = {}

        return {

            "disponible": True,

            "version":
                raw.get("version"),

            "fuente":
                raw.get(
                    "fuente",
                    "RainViewer"
                ),

            "actualizado":
                nowcast.get(
                    "actualizado"
                ),

            "frame_actual_utc":
                nowcast.get(
                    "frame_actual_utc"
                ),

            "frames_analizados":
                nowcast.get(
                    "frames_analizados"
                ),

            "tiles
