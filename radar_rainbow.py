import os
import math
import json
from io import BytesIO

import requests
from PIL import Image


# ============================================================
# CLIMAAR - PRUEBA RADAR RAINBOW + REFLECTIVIDAD dBZ
# ============================================================

API_URL = "https://api.rainbow.ai/tiles/v1"

LAT = -38.71
LON = -62.26

ZOOM = 7
GRID_RADIUS = 1

OUTPUT_DIR = "data/radar"

VISUAL_FILE = f"{OUTPUT_DIR}/rainbow_bahia_blanca.png"
STATUS_FILE = f"{OUTPUT_DIR}/rainbow_status.json"

TOKEN = os.getenv("RAINBOW_API_TOKEN")


# ============================================================
# LAT/LON -> TILE XYZ
# ============================================================

def latlon_to_tile(lat, lon, zoom):

    n = 2 ** zoom

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
# TOKEN
# ============================================================

def comprobar_token():

    if not TOKEN:

        print(
            "ERROR: no existe "
            "RAINBOW_API_TOKEN"
        )

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

    print(
        "HTTP snapshot:",
        response.status_code
    )

    if response.status_code != 200:

        print(response.text[:1000])

        return None

    data = response.json()

    snapshot = data.get("snapshot")

    if snapshot is None:

        print(
            "ERROR: no se encontró snapshot."
        )

        print(data)

        return None

    snapshot = int(snapshot)

    print(
        "Snapshot:",
        snapshot
    )

    return snapshot


# ============================================================
# DESCARGAR TILE VISUAL
# ============================================================

def descargar_tile_visual(
    snapshot,
    x,
    y
):

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
        "coverage": 0,
        "use_precip_type": 0
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=30
        )

        if response.status_code != 200:
            return None

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGBA")

        image.load()

        return image

    except Exception as e:

        print(
            f"Error visual {x}/{y}: {e}"
        )

        return None


# ============================================================
# DESCARGAR TILE RAW dBZ
# ============================================================

def descargar_tile_dbz(
    snapshot,
    x,
    y
):

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

    # IMPORTANTE:
    # Rainbow documenta dbz_u8 para obtener
    # reflectividad cruda.
    params = {
        "color": "dbz_u8",
        "coverage": 0,
        "use_precip_type": 0
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=30
        )

        if response.status_code != 200:

            print(
                f"DBZ {x}/{y}: "
                f"HTTP {response.status_code}"
            )

            return None

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGBA")

        image.load()

        return image

    except Exception as e:

        print(
            f"Error DBZ {x}/{y}: {e}"
        )

        return None


# ============================================================
# ANALIZAR dBZ
# ============================================================

def analizar_dbz(image):

    pixels = image.load()

    total = image.width * image.height

    cobertura = 0

    dbz_values = []

    dbz_10 = 0
    dbz_20 = 0
    dbz_30 = 0
    dbz_40 = 0
    dbz_50 = 0

    for y in range(image.height):

        for x in range(image.width):

            r, g, b, a = pixels[x, y]

            # Alpha 0 = sin cobertura radar
            if a == 0:
                continue

            cobertura += 1

            # Bit alto = nieve
            snow = (r & 128) == 128

            # Rainbow:
            # dbz = (R & 127) - 32
            dbz = (r & 127) - 32

            dbz_values.append(dbz)

            if dbz >= 10:
                dbz_10 += 1

            if dbz >= 20:
                dbz_20 += 1

            if dbz >= 30:
                dbz_30 += 1

            if dbz >= 40:
                dbz_40 += 1

            if dbz >= 50:
                dbz_50 += 1

    if not dbz_values:

        return {
            "coverage_percent": 0,
            "min_dbz": None,
            "max_dbz": None,
            "mean_dbz": None,
            "pixels_dbz_10": 0,
            "pixels_dbz_20": 0,
            "pixels_dbz_30": 0,
            "pixels_dbz_40": 0,
            "pixels_dbz_50": 0
        }

    return {
        "coverage_percent": round(
            cobertura / total * 100,
            2
        ),

        "min_dbz": min(dbz_values),

        "max_dbz": max(dbz_values),

        "mean_dbz": round(
            sum(dbz_values)
            / len(dbz_values),
            2
        ),

        "pixels_dbz_10": dbz_10,

        "pixels_dbz_20": dbz_20,

        "pixels_dbz_30": dbz_30,

        "pixels_dbz_40": dbz_40,

        "pixels_dbz_50": dbz_50
    }


# ============================================================
# PROCESAR MOSAICO
# ============================================================

def procesar(snapshot):

    center_x, center_y = latlon_to_tile(
        LAT,
        LON,
        ZOOM
    )

    print()
    print("========================================")
    print("RADAR RAINBOW")
    print("========================================")
    print("Lat:", LAT)
    print("Lon:", LON)
    print("Zoom:", ZOOM)
    print(
        "Tile central:",
        center_x,
        center_y
    )

    tile_size = 256

    grid_size = (
        GRID_RADIUS * 2
    ) + 1

    visual_mosaic = Image.new(
        "RGBA",
        (
            grid_size * tile_size,
            grid_size * tile_size
        ),
        (0, 0, 0, 0)
    )

    tiles = []

    total_tiles = 0
    valid_visual = 0
    valid_dbz = 0

    global_dbz = []

    for dy in range(
        -GRID_RADIUS,
        GRID_RADIUS + 1
    ):

        for dx in range(
            -GRID_RADIUS,
            GRID_RADIUS + 1
        ):

            total_tiles += 1

            x = center_x + dx
            y = center_y + dy

            print()
            print(
                f"Procesando tile "
                f"{x}/{y}"
            )

            # ------------------------
            # Imagen visual
            # ------------------------

            visual = descargar_tile_visual(
                snapshot,
                x,
                y
            )

            if visual is not None:

                valid_visual += 1

                paste_x = (
                    (dx + GRID_RADIUS)
                    * tile_size
                )

                paste_y = (
                    (dy + GRID_RADIUS)
                    * tile_size
                )

                visual_mosaic.paste(
                    visual,
                    (
                        paste_x,
                        paste_y
                    )
                )

            # ------------------------
            # Reflectividad dBZ
            # ------------------------

            dbz_image = descargar_tile_dbz(
                snapshot,
                x,
                y
            )

            if dbz_image is None:

                tiles.append({
                    "x": x,
                    "y": y,
                    "dbz": None
                })

                continue

            valid_dbz += 1

            analysis = analizar_dbz(
                dbz_image
            )

            analysis["x"] = x
            analysis["y"] = y

            tiles.append(analysis)

            if analysis["max_dbz"] is not None:

                global_dbz.extend(
                    [
                        analysis["max_dbz"],
                        analysis["mean_dbz"]
                    ]
                )

            print(
                "Cobertura:",
                analysis["coverage_percent"],
                "%"
            )

            print(
                "Min dBZ:",
                analysis["min_dbz"]
            )

            print(
                "Max dBZ:",
                analysis["max_dbz"]
            )

            print(
                "Promedio:",
                analysis["mean_dbz"]
            )

            print(
                ">=20 dBZ:",
                analysis["pixels_dbz_20"]
            )

            print(
                ">=30 dBZ:",
                analysis["pixels_dbz_30"]
            )

            print(
                ">=40 dBZ:",
                analysis["pixels_dbz_40"]
            )

            print(
                ">=50 dBZ:",
                analysis["pixels_dbz_50"]
            )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    visual_mosaic.save(
        VISUAL_FILE,
        "PNG"
    )

    # ========================================================
    # RESUMEN
    # ========================================================

    max_dbz_global = None

    if tiles:

        values = [
            t["max_dbz"]
            for t in tiles
            if t.get("max_dbz") is not None
        ]

        if values:

            max_dbz_global = max(values)

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

        "tiles_requested": total_tiles,

        "tiles_visual_ok": valid_visual,

        "tiles_dbz_ok": valid_dbz,

        "global_max_dbz": max_dbz_global,

        "tiles": tiles,

        "visual_image": VISUAL_FILE
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
    print("RESULTADO FINAL")
    print("========================================")

    print(
        "Tiles:",
        total_tiles
    )

    print(
        "Visual OK:",
        valid_visual
    )

    print(
        "dBZ OK:",
        valid_dbz
    )

    print(
        "Máximo dBZ:",
        max_dbz_global
    )

    print()
    print(
        "Imagen:",
        VISUAL_FILE
    )

    print(
        "Estado:",
        STATUS_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "CLIMAAR - RAINBOW RADAR "
        "REFLECTIVIDAD TEST"
    )

    if not comprobar_token():
        return

    snapshot = obtener_snapshot()

    if snapshot is None:
        return

    procesar(snapshot)


if __name__ == "__main__":
    main()
