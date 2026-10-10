import os
import math
import json
from io import BytesIO

import requests
from PIL import Image


# ============================================================
# CLIMAAR - PRUEBA RADAR RAINBOW
# ============================================================

API_URL = "https://api.rainbow.ai/tiles/v1"

# Bahía Blanca
LAT = -38.71
LON = -62.26

# Rainbow radar admite zoom 0-7
ZOOM = 7

# 3x3 tiles alrededor de Bahía Blanca
GRID_RADIUS = 1

# Salidas
OUTPUT_DIR = "data/radar"
OUTPUT_FILE = f"{OUTPUT_DIR}/rainbow_bahia_blanca.png"
STATUS_FILE = f"{OUTPUT_DIR}/rainbow_status.json"

TOKEN = os.getenv("RAINBOW_API_TOKEN")


# ============================================================
# CONVERSIÓN LAT/LON -> XYZ
# ============================================================

def latlon_to_tile(lat, lon, zoom):
    """
    Convierte coordenadas geográficas a tile XYZ Web Mercator.
    """

    n = 2 ** zoom

    x = int((lon + 180.0) / 360.0 * n)

    lat_rad = math.radians(lat)

    y = int(
        (
            1.0
            - math.asinh(math.tan(lat_rad)) / math.pi
        )
        / 2.0
        * n
    )

    return x, y


# ============================================================
# TOKEN
# ============================================================

def comprobar_token():

    if not TOKEN:

        print("ERROR: no existe RAINBOW_API_TOKEN")

        return False

    print("Token Rainbow detectado.")

    return True


# ============================================================
# SNAPSHOT
# ============================================================

def obtener_snapshot():

    url = f"{API_URL}/snapshot"

    headers = {
        "Ocp-Apim-Subscription-Key": TOKEN
    }

    params = {
        "layer": "radars"
    }

    print()
    print("Consultando snapshot Rainbow...")

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30
    )

    print("HTTP snapshot:", response.status_code)

    if response.status_code != 200:

        print(response.text[:1000])

        return None

    try:

        data = response.json()

    except Exception:

        print("ERROR: Rainbow no devolvió JSON.")

        print(response.text[:1000])

        return None

    snapshot = data.get("snapshot")

    if snapshot is None:

        print("ERROR: no se encontró snapshot.")

        print(data)

        return None

    print("Snapshot:", snapshot)

    return int(snapshot)


# ============================================================
# DESCARGAR TILE
# ============================================================

def descargar_tile(snapshot, x, y):

    url = (
        f"{API_URL}/radars/"
        f"{snapshot}/"
        f"{ZOOM}/"
        f"{x}/"
        f"{y}"
    )

    headers = {
        "Ocp-Apim-Subscription-Key": TOKEN
    }

    params = {
        "color": 0,
        "coverage": 1,
        "use_precip_type": 0
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=30
        )

    except Exception as e:

        print(
            f"ERROR tile {x}/{y}: {e}"
        )

        return None, {
            "x": x,
            "y": y,
            "status": "request_error",
            "error": str(e)
        }

    print(
        f"Tile {x}/{y} -> HTTP {response.status_code}"
    )

    if response.status_code != 200:

        return None, {
            "x": x,
            "y": y,
            "status": response.status_code,
            "size": len(response.content)
        }

    content_type = (
        response.headers
        .get("content-type", "")
        .lower()
    )

    if "image/png" not in content_type:

        print(
            f"Tile {x}/{y}: respuesta no PNG"
        )

        return None, {
            "x": x,
            "y": y,
            "status": "not_png",
            "content_type": content_type
        }

    try:

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGBA")

        # Cargar realmente los píxeles
        image.load()

    except Exception as e:

        print(
            f"ERROR procesando tile {x}/{y}: {e}"
        )

        return None, {
            "x": x,
            "y": y,
            "status": "invalid_image",
            "error": str(e)
        }

    # Analizamos transparencia
    alpha = image.getchannel("A")

    alpha_extrema = alpha.getextrema()

    # Cantidad de píxeles con alpha > 0
    alpha_data = sum(
        1
        for value in alpha.getdata()
        if value > 0
    )

    total_pixels = image.width * image.height

    coverage_percent = (
        alpha_data / total_pixels
    ) * 100.0

    print(
        f"   Cobertura visible: "
        f"{coverage_percent:.2f}%"
    )

    return image, {
        "x": x,
        "y": y,
        "status": 200,
        "width": image.width,
        "height": image.height,
        "coverage_percent": round(
            coverage_percent,
            2
        ),
        "alpha_min": alpha_extrema[0],
        "alpha_max": alpha_extrema[1]
    }


# ============================================================
# ARMAR MOSAICO 3x3
# ============================================================

def descargar_mosaico(snapshot):

    center_x, center_y = latlon_to_tile(
        LAT,
        LON,
        ZOOM
    )

    print()
    print("========================================")
    print("CENTRO RADAR")
    print("========================================")
    print("Latitud:", LAT)
    print("Longitud:", LON)
    print("Zoom:", ZOOM)
    print("Tile central:", center_x, center_y)

    tile_size = 256

    grid_size = (
        GRID_RADIUS * 2
    ) + 1

    mosaic = Image.new(
        "RGBA",
        (
            grid_size * tile_size,
            grid_size * tile_size
        ),
        (0, 0, 0, 0)
    )

    results = []

    valid_tiles = 0
    tiles_with_data = 0

    for dy in range(
        -GRID_RADIUS,
        GRID_RADIUS + 1
    ):

        for dx in range(
            -GRID_RADIUS,
            GRID_RADIUS + 1
        ):

            x = center_x + dx
            y = center_y + dy

            image, info = descargar_tile(
                snapshot,
                x,
                y
            )

            results.append(info)

            if image is None:
                continue

            valid_tiles += 1

            coverage = info.get(
                "coverage_percent",
                0
            )

            if coverage > 0:
                tiles_with_data += 1

            paste_x = (
                (dx + GRID_RADIUS)
                * tile_size
            )

            paste_y = (
                (dy + GRID_RADIUS)
                * tile_size
            )

            mosaic.paste(
                image,
                (
                    paste_x,
                    paste_y
                )
            )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    mosaic.save(
        OUTPUT_FILE,
        "PNG"
    )

    status = {
        "provider": "Rainbow",
        "latitude": LAT,
        "longitude": LON,
        "zoom": ZOOM,
        "center_tile": {
            "x": center_x,
            "y": center_y
        },
        "snapshot": snapshot,
        "tiles_requested": grid_size * grid_size,
        "tiles_valid": valid_tiles,
        "tiles_with_data": tiles_with_data,
        "tiles": results,
        "image": OUTPUT_FILE
    }

    with open(
        STATUS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            status,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("========================================")
    print("RESULTADO RAINBOW")
    print("========================================")
    print(
        "Tiles solicitados:",
        grid_size * grid_size
    )
    print(
        "Tiles válidos:",
        valid_tiles
    )
    print(
        "Tiles con datos:",
        tiles_with_data
    )
    print()
    print("Imagen:", OUTPUT_FILE)
    print("Estado:", STATUS_FILE)

    return status


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print("CLIMAAR - RAINBOW RADAR TEST")
    print("========================================")

    if not comprobar_token():
        return

    snapshot = obtener_snapshot()

    if snapshot is None:
        return

    descargar_mosaico(snapshot)


if __name__ == "__main__":
    main()
