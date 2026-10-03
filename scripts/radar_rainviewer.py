import requests
import os
import json
import math
import sys

from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from datetime import datetime, timezone


LAT = -38.0055
LON = -62.0
ZOOM = 7

DATA_DIR = "data/radar"
os.makedirs(DATA_DIR, exist_ok=True)


def latlon_to_tile(lat, lon, zoom):
    n = 2.0 ** zoom

    xtile = (lon + 180.0) / 360.0 * n

    lat_rad = math.radians(lat)

    ytile = (
        1.0
        - math.asinh(math.tan(lat_rad)) / math.pi
    ) / 2.0 * n

    return int(xtile), int(ytile)


print("=== CLIMAAR V4 OFICIAL ===")


# ---------------------------------------------------------
# OBTENER DATOS OFICIALES DE RAINVIEWER
# ---------------------------------------------------------

api_url = "https://api.rainviewer.com/public/weather-maps.json"

try:
    response = requests.get(api_url, timeout=20)
    response.raise_for_status()
    data = response.json()
except Exception as e:
    print(f"ERROR API RAINVIEWER: {e}")
    sys.exit(1)


host = data.get(
    "host",
    "https://tilecache.rainviewer.com"
).rstrip("/")

frames = data.get("radar", {}).get("past", [])[-6:]


print(f"Host: {host}")
print(f"Frames: {len(frames)}")


if not frames:
    print("ERROR CRITICO: RainViewer no devolvio frames.")
    sys.exit(1)


# ---------------------------------------------------------
# UBICACION BAHIA BLANCA
# ---------------------------------------------------------

xt, yt = latlon_to_tile(
    LAT,
    LON,
    ZOOM
)

print(f"Tile central: X={xt} Y={yt}")


results = []


# ---------------------------------------------------------
# DESCARGAR FRAMES
# ---------------------------------------------------------

for frame in frames:

    path = frame["path"]
    timestamp = frame["time"]

    big = Image.new(
        "RGBA",
        (1536, 1536),
        (0, 0, 0, 0)
    )

    ok = 0

    for dx in [-1, 0, 1]:

        for dy in [-1, 0, 1]:

            url = (
                f"{host}"
                f"{path}"
                f"/512/{ZOOM}/"
                f"{xt + dx}/"
                f"{yt + dy}/"
                f"2/1_1.png"
            )

            try:

                resp = requests.get(
                    url,
                    timeout=15
                )

                if (
                    resp.status_code == 200
                    and len(resp.content) > 200
                ):

                    tile = Image.open(
                        BytesIO(resp.content)
                    ).convert("RGBA")

                    tile_array = np.array(tile)

                    # Comprobar que la tesela tenga
                    # al menos algun pixel visible.
                    if np.any(
                        tile_array[:, :, 3] > 0
                    ):

                        big.paste(
                            tile,
                            (
                                (dx + 1) * 512,
                                (dy + 1) * 512
                            ),
                            tile
                        )

                    ok += 1

            except Exception as e:

                print(
                    f"fail {url} -> {e}"
                )

    print(
        f"{timestamp} -> "
        f"{path} "
        f"tiles OK: {ok}"
    )

    # Si este frame no pudo descargar
    # ninguna tesela, se ignora.
    if ok == 0:

        print(
            "ADVERTENCIA: "
            "0 tiles en este frame, lo salteo"
        )

        continue


    # -----------------------------------------------------
    # MASCARA DE RADAR
    # -----------------------------------------------------

    alpha = np.array(big)[:, :, 3]

    _, mask = cv2.threshold(
        alpha,
        15,
        255,
        cv2.THRESH_BINARY
    )


    area = int(
        np.count_nonzero(mask)
    )


    results.append(
        {
            "mask": mask,
            "img": big,
            "area": area,
            "time": timestamp,
            "path": path
        }
    )


# ---------------------------------------------------------
# EXIGIR AL MENOS UN FRAME VALIDO
# ---------------------------------------------------------

if not results:

    print(
        "ERROR CRITICO: "
        "No se descargo ningun radar. "
        "Falla el workflow a proposito."
    )

    sys.exit(1)


# ---------------------------------------------------------
# GUARDAR RADAR ACTUAL
# ---------------------------------------------------------

actual_path = os.path.join(
    DATA_DIR,
    "actual.png"
)

results[-1]["img"].save(
    actual_path
)


print(
    f"Guardado actual.png "
    f"area={results[-1]['area']}"
)


# ---------------------------------------------------------
# STATUS
# ---------------------------------------------------------

status = {

    "actualizado":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "host_usado":
        host,

    "area_px":
        results[-1]["area"],

    "distancia_km":
        None,

    "eta_minutos":
        None,

    "duracion_estimada_min":
        None,

    "fortalecimiento":
        "estable",

    "estado":
        "cielo despejado"
}


# ---------------------------------------------------------
# ANALISIS ENTRE FRAMES
# ---------------------------------------------------------

if len(results) >= 2:

    prev = results[-2]
    curr = results[-1]


    # -----------------------------------------------------
    # FORTALECIMIENTO / DEBILITAMIENTO
    # -----------------------------------------------------

    if (
        curr["area"]
        > prev["area"] * 1.15
    ):

        status["fortalecimiento"] = (
            "fortaleciendose"
        )

    elif (
        curr["area"]
        < prev["area"] * 0.85
    ):

        status["fortalecimiento"] = (
            "debilitandose"
        )


    # -----------------------------------------------------
    # DISTANCIA
    # -----------------------------------------------------

    ys, xs = np.where(
        curr["mask"] > 0
    )


    if len(xs) > 0:

        h, w = curr["mask"].shape

        cy = h // 2
        cx = w // 2


        dists = np.sqrt(
            (xs - cx) ** 2
            +
            (ys - cy) ** 2
        )


        # Aproximacion inicial.
        # Se ajustara posteriormente
        # con georreferenciacion precisa.

        km_per_px = 0.85


        dist_km = (
            np.min(dists)
            * km_per_px
        )


        status["distancia_km"] = round(
            float(dist_km),
            1
        )


        status["estado"] = (
            f"lluvia a "
            f"{status['distancia_km']} km"
        )


        # -------------------------------------------------
        # MOVIMIENTO
        # -------------------------------------------------

        try:

            flow = cv2.calcOpticalFlowFarneback(
                prev["mask"],
                curr["mask"],
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0
            )


            vx = np.mean(
                flow[ys, xs, 0]
            )

            vy = np.mean(
                flow[ys, xs, 1]
            )


            speed = (
                math.sqrt(
                    vx ** 2
                    +
                    vy ** 2
                )
                * km_per_px
                * 6
            )


            if speed > 3:

                status["eta_minutos"] = int(
                    (dist_km / speed)
                    * 60
                )


        except Exception:

            pass


# ---------------------------------------------------------
# GUARDAR STATUS
# ---------------------------------------------------------

status_path = os.path.join(
    DATA_DIR,
    "status.json"
)


with open(
    status_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        status,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    json.dumps(
        status,
        indent=2,
        ensure_ascii=False
    )
)


print("=== V4 OK ===")
