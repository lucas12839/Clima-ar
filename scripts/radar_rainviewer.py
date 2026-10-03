import requests
import os
import json
import math

from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from datetime import datetime, timezone


# ==============================
# CONFIGURACIÓN BAHÍA BLANCA
# ==============================

LAT = -38.0055
LON = -62.0
ZOOM = 7

DATA_DIR = "data/radar"

os.makedirs(DATA_DIR, exist_ok=True)


# ==============================
# CONVERSIÓN LAT/LON -> TILE
# ==============================

def latlon_to_tile(lat, lon, zoom):
    n = 2.0 ** zoom

    xtile = (
        (lon + 180.0)
        / 360.0
        * n
    )

    lat_rad = math.radians(lat)

    ytile = (
        1.0
        - math.asinh(
            math.tan(lat_rad)
        ) / math.pi
    ) / 2.0 * n

    return int(xtile), int(ytile)


# ==============================
# OBTENER FRAMES DE RAINVIEWER
# ==============================

def get_radar_frames():

    url = (
        "https://api.rainviewer.com/"
        "public/weather-maps.json"
    )

    response = requests.get(
        url,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    frames = data["radar"]["past"]

    # Últimos 6 frames
    return frames[-6:]


# ==============================
# DESCARGAR TILE
# ==============================

def download_tile(url):

    response = requests.get(
        url,
        timeout=20
    )

    if response.status_code == 200:

        return Image.open(
            BytesIO(response.content)
        ).convert("RGBA")

    return None


# ==============================
# INICIO
# ==============================

frames = get_radar_frames()

print(
    f"Frames encontrados: {len(frames)}"
)

images_for_analysis = []

last_image_path = None


# ==============================
# PROCESAR FRAMES
# ==============================

for frame in frames:

    timestamp = frame["time"]

    xtile, ytile = latlon_to_tile(
        LAT,
        LON,
        ZOOM
    )

    tiles = []

    # Área 3x3 alrededor de Bahía Blanca
    for dx in [-1, 0, 1]:

        for dy in [-1, 0, 1]:

            url = (
                "https://tilecache.rainviewer.com/"
                f"v2/radar/{timestamp}/512/"
                f"{ZOOM}/{xtile + dx}/"
                f"{ytile + dy}/2/1_1.png"
            )

            image = download_tile(url)

            if image is not None:

                tiles.append(
                    (dx, dy, image)
                )

    if not tiles:
        print(
            f"No se pudieron descargar tiles "
            f"para {timestamp}"
        )
        continue


    # ==========================
    # UNIR TILES
    # ==========================

    big = Image.new(
        "RGBA",
        (512 * 3, 512 * 3),
        (0, 0, 0, 0)
    )

    for dx, dy, image in tiles:

        big.paste(
            image,
            (
                (dx + 1) * 512,
                (dy + 1) * 512
            )
        )


    # ==========================
    # GUARDAR FRAME
    # ==========================

    path = (
        f"{DATA_DIR}/radar_{timestamp}.png"
    )

    big.save(path)

    last_image_path = path


    # ==========================
    # PREPARAR ANÁLISIS
    # ==========================

    rgba = np.array(big)

    gray = cv2.cvtColor(
        rgba,
        cv2.COLOR_RGBA2GRAY
    )

    _, threshold = cv2.threshold(
        gray,
        20,
        255,
        cv2.THRESH_BINARY
    )

    images_for_analysis.append(
        threshold
    )


# ==============================
# ESTADO INICIAL
# ==============================

status = {

    "actualizado":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "ultimo_radar":
        frames[-1]["time"]
        if frames
        else None,

    "distancia_km": None,

    "eta_minutos": None,

    "fortalecimiento":
        "sin datos",

    "estado":
        "sin lluvia cercana"
}


# ==============================
# ANÁLISIS
# ==============================

if len(images_for_analysis) >= 2:

    previous = images_for_analysis[-2]

    current = images_for_analysis[-1]


    # ==========================
    # ÁREA DE LLUVIA
    # ==========================

    area_previous = np.count_nonzero(
        previous
    )

    area_current = np.count_nonzero(
        current
    )


    if area_current > area_previous * 1.10:

        status["fortalecimiento"] = (
            "fortaleciendose"
        )

    elif area_current < area_previous * 0.90:

        status["fortalecimiento"] = (
            "debilitandose"
        )

    else:

        status["fortalecimiento"] = (
            "estable"
        )


    # ==========================
    # MOVIMIENTO
    # ==========================

    flow = cv2.calcOpticalFlowFarneback(
        previous,
        current,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )


    rain_pixels = np.where(
        current > 0
    )


    if np.any(current > 0):

        mean_flow = np.mean(
            flow[rain_pixels],
            axis=0
        )

    else:

        mean_flow = np.array(
            [0.0, 0.0]
        )


    # ==========================
    # VELOCIDAD ESTIMADA
    # ==========================

    velocity_px = (
        np.linalg.norm(mean_flow)
        * 6
    )

    velocity_kmh = (
        velocity_px * 2.5
    )


    # ==========================
    # DISTANCIA A BAHÍA BLANCA
    # ==========================

    height, width = current.shape

    center_y = height // 2
    center_x = width // 2


    ys, xs = np.where(
        current > 0
    )


    if len(xs) > 0:

        distances = np.sqrt(
            (xs - center_x) ** 2
            +
            (ys - center_y) ** 2
        )

        minimum_distance_px = (
            np.min(distances)
        )


        minimum_distance_km = (
            minimum_distance_px
            * 2.5
            / 3
        )


        status["distancia_km"] = round(
            float(
                minimum_distance_km
            ),
            1
        )


        # ==========================
        # ETA
        # ==========================

        if velocity_kmh > 5:

            eta_hours = (
                minimum_distance_km
                / velocity_kmh
            )

            status["eta_minutos"] = int(
                eta_hours * 60
            )

            status["estado"] = (
                "tormenta acercandose"
            )

        else:

            status["estado"] = (
                "tormenta estacionaria"
            )

    else:

        status["estado"] = (
            "cielo despejado en radar"
        )


# ==============================
# GUARDAR STATUS
# ==============================

status_path = (
    f"{DATA_DIR}/status.json"
)

with open(
    status_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        status,
        file,
        indent=2,
        ensure_ascii=False
    )


# ==============================
# GUARDAR RADAR ACTUAL
# ==============================

if last_image_path:

    Image.open(
        last_image_path
    ).save(
        f"{DATA_DIR}/actual.png"
    )


# ==============================
# MOSTRAR RESULTADO
# ==============================

print(
    json.dumps(
        status,
        indent=2,
        ensure_ascii=False
    )
  )
