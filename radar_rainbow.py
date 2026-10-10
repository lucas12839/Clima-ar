import os
import time
import requests

# ============================================================
# CLIMAAR - PRUEBA RADAR RAINBOW
# ============================================================

API_URL = "https://api.rainbow.ai"

# Bahía Blanca
LAT = -38.71
LON = -62.26

# Zoom máximo permitido actualmente para radar
ZOOM = 7

# Tile correspondiente aproximadamente a Bahía Blanca
TILE_X = 41
TILE_Y = 78

OUTPUT_FILE = "data/radar/rainbow_bahia_blanca.png"

TOKEN = os.getenv("RAINBOW_API_TOKEN")


def comprobar_token():
    if not TOKEN:
        print("ERROR: no existe la variable RAINBOW_API_TOKEN")
        print("El código está correcto, pero falta configurar la clave API.")
        return False

    return True


def obtener_snapshot():
    """
    Obtiene el snapshot radar más reciente.
    """

    url = f"{API_URL}/tiles/v1/snapshot"

    headers = {
        "Ocp-Apim-Subscription-Key": TOKEN
    }

    params = {
        "layer": "radars"
    }

    print("Consultando snapshot de Rainbow...")

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30
    )

    print("HTTP snapshot:", response.status_code)

    if response.status_code != 200:
        print("Respuesta:")
        print(response.text[:1000])
        return None

    data = response.json()

    snapshot = data.get("snapshot")

    if not snapshot:
        print("ERROR: Rainbow no devolvió snapshot.")
        print(data)
        return None

    return snapshot


def descargar_radar(snapshot):
    """
    Descarga el tile radar correspondiente a Bahía Blanca.
    """

    url = (
        f"{API_URL}/tiles/v1/radars/"
        f"{snapshot}/{ZOOM}/{TILE_X}/{TILE_Y}"
    )

    headers = {
        "Ocp-Apim-Subscription-Key": TOKEN
    }

    params = {
        "color": 0,
        "coverage": 1,
        "use_precip_type": 0
    }

    print()
    print("Descargando radar...")
    print("Snapshot:", snapshot)
    print("Tile:", TILE_X, TILE_Y)
    print("Zoom:", ZOOM)

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30
    )

    print("HTTP radar:", response.status_code)
    print("Content-Type:", response.headers.get("content-type"))
    print("Tamaño:", len(response.content), "bytes")

    if response.status_code != 200:
        print("ERROR descargando radar:")
        print(response.text[:1000])
        return False

    if not response.content.startswith(b"\x89PNG"):
        print("ERROR: la respuesta no parece ser una imagen PNG.")
        print(response.content[:100])
        return False

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    with open(OUTPUT_FILE, "wb") as f:
        f.write(response.content)

    print()
    print("========================================")
    print("RADAR RAINBOW DESCARGADO CORRECTAMENTE")
    print("========================================")
    print("Archivo:", OUTPUT_FILE)
    print("Snapshot:", snapshot)
    print("Hora snapshot UTC:", time.strftime(
        "%Y-%m-%d %H:%M:%S",
        time.gmtime(snapshot)
    ))

    return True


def main():
    print("========================================")
    print("CLIMAAR - PRUEBA RAINBOW RADAR")
    print("========================================")
    print()

    if not comprobar_token():
        return

    snapshot = obtener_snapshot()

    if snapshot is None:
        return

    descargar_radar(snapshot)


if __name__ == "__main__":
    main()
