import requests
import os
import json
import math

from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from datetime import datetime, timezone


# ==========================================
# CONFIGURACIÓN CLIMAAR - BAHÍA BLANCA
# ==========================================

LAT = -38.0055
LON = -62.0
ZOOM = 7

DATA_DIR = "data/radar"

os.makedirs(DATA_DIR, exist_ok=True)


# ==========================================
# API RAINVIEWER
# ==========================================

API_URL = (
    "https://api.rainviewer.com/"
    "public/weather-maps.json"
)


# ==========================================
# CONVERTIR LAT/LON A TILE
# ==========================================

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


# ==========================================
# OBTENER FRAMES
# ==========================================

def get_radar_data():

    response = requests.get(
        API_URL,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if "radar" not in data:
        raise RuntimeError(
            "RainViewer no devolvió datos de radar."
        )

    host = data.get("host")

    if not host:
        raise RuntimeError(
            "RainViewer no devolvió host."
        )

    frames = data["radar"].get(
        "past",
        []
    )

    if not frames:
        raise RuntimeError(
            "RainViewer no devolvió frames."
        )

    return host.rstrip("/"), frames


# ==========================================
# DESCARGAR TILE
# ==========================================

def download_tile(url):

    try:

        response = requests.get(
            url,
            timeout=30
        )

        response.raise_for_status()

        return Image.open(
            BytesIO(response.content)
        ).convert("RGBA")

    except Exception as error:

        print(
            f"Error descargando tile: {error}"
        )

        return None


# ==========================================
# OBTENER DATOS
# ==========================================

host, frames = get_radar_data()

print(
    f"Frames encontrados: {len(frames)}"
)

print(
    f"Host RainViewer: {host}"
)


# ==========================================
# TOMAR LOS ÚLTIMOS 6 FRAMES
# ==========================================

frames = frames[-6:]


images_for_analysis = []

last_image_path = None

last_timestamp = None


# ==========================================
# CALCULAR TILE CENTRAL
# ==========================================

xtile, ytile = latlon_to_tile(
    LAT,
    LON,
    ZOOM
)


# ==========================================
# PROCESAR CADA FRAME
# ==========================================

for frame in frames:

    timestamp = frame["time"]

    frame_path = frame.get("path")

    if not frame_path:

        print(
            f"Frame {timestamp} sin path."
        )

        continue


    print(
        f"Procesando frame {timestamp}"
    )

    print(
        f"Path: {frame_path}"
    )


    # ======================================
    # DESCARGAR 3x3 TILES
    # ======================================

    tiles = []


    for dx in [-1, 0, 1]:

        for dy in [-1, 0, 1]:

            url = (
                host
                + frame_path
                + f"/512/{ZOOM}/"
                f"{xtile + dx}/"
                f"{ytile + dy}/"
                "2/1_1.png"
            )


            print(
                f"Tile: {url}"
            )


            image = download_tile(url)


            if image is not None:

                tiles.append(
                    (
                        dx,
                        dy,
                        image
                    )
                )


    # ======================================
    # SI NO HAY TILES
    # ======================================

    if not tiles:

        print(
            f"NO SE PUDIERON DESCARGAR "
            f"TILES PARA {timestamp}"
        )

        continue


    print(
        f"TILES DESCARGADOS: {len(tiles)}"
    )


    # ======================================
    # UNIR TILES
    # ======================================

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


    # ======================================
    # GUARDAR FRAME
    # ======================================

    path = (
        f"{DATA_DIR}/"
        f"radar_{timestamp}.png"
    )


    big.save(path)


    last_image_path = path

    last_timestamp = timestamp


    # ======================================
    # CONVERTIR PARA ANÁLISIS
    # ======================================

    rgba = np.array(big)


    # Usamos el canal alfa para detectar
    # zonas donde RainViewer realmente
    # tiene datos de radar.

    alpha = rgba[:, :, 3]


    # También analizamos luminosidad/color.

    gray = cv2.cvtColor(
        rgba,
        cv2.COLOR_RGBA2GRAY
    )


    # Máscara combinada.

    mask = np.logical_and(
        alpha > 20,
        gray > 20
    )


    threshold = (
        mask.astype(np.uint8)
        * 255
    )


    images_for_analysis.append(
        threshold
    )


# ==========================================
# ESTADO INICIAL
# ==========================================

status = {

    "actualizado":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "ultimo_radar":
        last_timestamp,

    "distancia_km":
        None,

    "eta_minutos":
        None,

    "fortalecimiento":
        "sin datos",

    "estado":
        "sin lluvia cercana"
}


# ==========================================
# ANALIZAR MOVIMIENTO
# ==========================================

if len(images_for_analysis) >= 2:

    previous = (
        images_for_analysis[-2]
    )

    current = (
        images_for_analysis[-1]
    )


    # ======================================
    # ÁREA DE PRECIPITACIÓN
    # ======================================

    area_previous = np.count_nonzero(
        previous
    )

    area_current = np.count_nonzero(
        current
    )


    print(
        f"Área anterior: {area_previous}"
    )

    print(
        f"Área actual: {area_current}"
    )


    if area_previous > 0:

        ratio = (
            area_current
            / area_previous
        )

    else:

        ratio = 1.0


    if ratio > 1.10:

        status["fortalecimiento"] = (
            "fortaleciendose"
        )

    elif ratio < 0.90:

        status["fortalecimiento"] = (
            "debilitandose"
        )

    else:

        status["fortalecimiento"] = (
            "estable"
        )


    # ======================================
    # MOVIMIENTO ÓPTICO
    # ======================================

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


    print(
        f"Movimiento medio: {mean_flow}"
    )


    # ======================================
    # VELOCIDAD APROXIMADA
    # ======================================

    velocity_px_hour = (
        np.linalg.norm(mean_flow)
        * 6
    )


    # Aproximación para zoom 7.
    PIXEL_KM = 2.5


    velocity_kmh = (
        velocity_px_hour
        * PIXEL_KM
    )


    print(
        f"Velocidad estimada: "
        f"{velocity_kmh:.2f} km/h"
    )


    # ======================================
    # DISTANCIA A BAHÍA BLANCA
    # ======================================

    height, width = (
        current.shape
    )


    center_y = (
        height // 2
    )

    center_x = (
        width // 2
    )


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

            * PIXEL_KM

            / 3

        )


        status["distancia_km"] = round(
            float(
                minimum_distance_km
            ),
            1
        )


        # ==================================
        # ETA
        # ==================================

        if velocity_kmh > 5:

            eta_hours = (

                minimum_distance_km

                / velocity_kmh

            )


            status["eta_minutos"] = max(

                1,

                int(
                    eta_hours * 60
                )

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


# ==========================================
# GUARDAR STATUS
# ==========================================

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


# ==========================================
# GUARDAR RADAR ACTUAL
# ==========================================

if last_image_path:

    Image.open(
        last_image_path
    ).save(
        f"{DATA_DIR}/actual.png"
    )


# ==========================================
# RESULTADO
# ==========================================

print("")
print("======================================")
print("CLIMAAR RADAR FINALIZADO")
print("======================================")

print(
    json.dumps(
        status,
        indent=2,
        ensure_ascii=False
    )
)

print("======================================")
