import os
import sys
import json
import math
from io import BytesIO
from datetime import datetime, timezone

import requests
from PIL import Image
import numpy as np
import cv2


LAT = -38.71
LON = -62.26
ZOOM = 7

DATA_DIR = "data/radar"
os.makedirs(DATA_DIR, exist_ok=True)

API_URL = "https://api.rainviewer.com/public/weather-maps.json"

HEADERS = {
    "User-Agent": "ClimaAR/5.0",
    "Referer": "https://www.rainviewer.com/",
    "Accept": "image/png,image/*;q=0.8,*/*;q=0.5",
}

TILE_SIZE = 512
GRID_RADIUS = 1
MIN_TILE_BYTES = 200


def latlon_to_tile(lat, lon, zoom):
    n = 2.0 ** zoom

    xtile = (lon + 180.0) / 360.0 * n

    lat_rad = math.radians(lat)

    ytile = (
        1.0
        - math.asinh(math.tan(lat_rad)) / math.pi
    ) / 2.0 * n

    return int(xtile), int(ytile)


def pixels_per_km(lat, zoom, tile_size):
    world_km = 40075.016

    km_x = (
        world_km
        * math.cos(math.radians(lat))
        / ((2 ** zoom) * tile_size)
    )

    km_y = (
        world_km
        / ((2 ** zoom) * tile_size)
    )

    return km_x, km_y


def fetch_rainviewer():
    try:
        response = requests.get(
            API_URL,
            headers=HEADERS,
            timeout=25,
        )

        response.raise_for_status()
        data = response.json()

    except Exception as exc:
        print(f"ERROR API RAINVIEWER: {exc}")
        sys.exit(1)

    host = data.get(
        "host",
        "https://tilecache.rainviewer.com",
    ).rstrip("/")

    frames = data.get(
        "radar",
        {}
    ).get(
        "past",
        []
    )

    if not frames:
        print(
            "ERROR CRITICO: "
            "RainViewer no devolvio frames."
        )
        sys.exit(1)

    return host, frames


def download_frame(
    host,
    frame,
    xt,
    yt
):
    path = frame["path"]
    timestamp = int(frame["time"])

    size = TILE_SIZE * 3

    image = Image.new(
        "RGBA",
        (size, size),
        (0, 0, 0, 0),
    )

    successful_tiles = 0

    for dx in (-1, 0, 1):

        for dy in (-1, 0, 1):

            url = (
                f"{host}{path}"
                f"/{TILE_SIZE}/{ZOOM}/"
                f"{xt + dx}/{yt + dy}/2/1_1.png"
            )

            try:

                response = requests.get(
                    url,
                    headers=HEADERS,
                    timeout=20,
                )

                if (
                    response.status_code != 200
                    or len(response.content)
                    < MIN_TILE_BYTES
                ):
                    print(
                        f"Tile invalida: "
                        f"HTTP {response.status_code} "
                        f"{url}"
                    )
                    continue

                tile = Image.open(
                    BytesIO(
                        response.content
                    )
                ).convert("RGBA")

                x = (
                    dx + 1
                ) * TILE_SIZE

                y = (
                    dy + 1
                ) * TILE_SIZE

                image.alpha_composite(
                    tile,
                    (x, y)
                )

                successful_tiles += 1

            except Exception as exc:

                print(
                    f"FAIL tile {url} -> {exc}"
                )

    return (
        image,
        successful_tiles,
        timestamp,
        path
    )


def radar_mask(image):

    array = np.asarray(image)

    alpha = array[:, :, 3]

    mask = np.where(
        alpha >= 20,
        255,
        0,
    ).astype(np.uint8)

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return mask


def largest_component(mask):

    count, labels, stats, centers = (
        cv2.connectedComponentsWithStats(
            mask,
            8,
        )
    )

    best = None

    for i in range(
        1,
        count
    ):

        area = int(
            stats[
                i,
                cv2.CC_STAT_AREA
            ]
        )

        if area < 20:
            continue

        candidate = {
            "area_px": area,

            "centroid_x": float(
                centers[i][0]
            ),

            "centroid_y": float(
                centers[i][1]
            ),

            "bbox": [
                int(
                    stats[
                        i,
                        cv2.CC_STAT_LEFT
                    ]
                ),

                int(
                    stats[
                        i,
                        cv2.CC_STAT_TOP
                    ]
                ),

                int(
                    stats[
                        i,
                        cv2.CC_STAT_WIDTH
                    ]
                ),

                int(
                    stats[
                        i,
                        cv2.CC_STAT_HEIGHT
                    ]
                ),
            ],
        }

        if (
            best is None
            or area > best["area_px"]
        ):
            best = candidate

    return best


def distance_to_bahia(mask):

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:
        return None

    height, width = mask.shape

    center_x = width / 2.0
    center_y = height / 2.0

    km_x, km_y = pixels_per_km(
        LAT,
        ZOOM,
        TILE_SIZE,
    )

    dx = (
        xs - center_x
    ) * km_x

    dy = (
        ys - center_y
    ) * km_y

    distances = np.sqrt(
        dx * dx
        + dy * dy
    )

    return float(
        np.min(distances)
    )


def analyze_frame(image):

    mask = radar_mask(
        image
    )

    area = int(
        np.count_nonzero(
            mask
        )
    )

    component = largest_component(
        mask
    )

    distance = distance_to_bahia(
        mask
    )

    return {
        "mask": mask,
        "area": area,
        "component": component,
        "distance_km": distance,
    }


def motion_between(
    previous,
    current
):

    if (
        previous is None
        or current is None
    ):
        return None

    previous_component = (
        previous["component"]
    )

    current_component = (
        current["component"]
    )

    if (
        previous_component is None
        or current_component is None
    ):
        return None

    dx_px = (
        current_component["centroid_x"]
        - previous_component["centroid_x"]
    )

    dy_px = (
        current_component["centroid_y"]
        - previous_component["centroid_y"]
    )

    km_x, km_y = pixels_per_km(
        LAT,
        ZOOM,
        TILE_SIZE,
    )

    east_km = (
        dx_px * km_x
    )

    south_km = (
        dy_px * km_y
    )

    distance_km = math.hypot(
        east_km,
        south_km,
    )

    direction = (
        math.degrees(
            math.atan2(
                east_km,
                -south_km,
            )
        )
        + 360.0
    ) % 360.0

    return {
        "dx_px": round(
            dx_px,
            2,
        ),

        "dy_px": round(
            dy_px,
            2,
        ),

        "movimiento_km": round(
            distance_km,
            2,
        ),

        "direccion_grados": round(
            direction,
            1,
        ),
    }


def direction_name(
    degrees
):

    names = [
        "N",
        "NE",
        "E",
        "SE",
        "S",
        "SO",
        "O",
        "NO",
    ]

    return names[
        int(
            (
                degrees
                + 22.5
            ) // 45
        ) % 8
    ]


def calculate_speed(
    movement,
    previous_timestamp,
    current_timestamp
):

    if movement is None:
        return None

    if previous_timestamp is None:
        return None

    elapsed_minutes = (
        current_timestamp
        - previous_timestamp
    ) / 60.0

    if elapsed_minutes <= 0:
        return None

    speed_kmh = (
        movement["movimiento_km"]
        / (
            elapsed_minutes
            / 60.0
        )
    )

    return {
        "intervalo_minutos": round(
            elapsed_minutes,
            1,
        ),

        "velocidad_kmh": round(
            speed_kmh,
            1,
        ),
    }


def analyze_sequence(
    results
):

    if not results:
        return None

    current = results[-1]

    status = {
        "actualizado":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "fuente":
            "RainViewer",

        "latitud":
            LAT,

        "longitud":
            LON,

        "frames_analizados":
            len(results),

        "area_px":
            current["area"],

        "distancia_km":
            (
                round(
                    current["distance_km"],
                    1,
                )
                if current["distance_km"]
                is not None
                else None
            ),

        "fortalecimiento":
            "sin_datos",

        "velocidad_kmh":
            None,

        "direccion":
            None,

        "direccion_grados":
            None,

        "eta_minutos":
            None,

        "confianza_movimiento":
            0.0,

        "estado":
            "sin_precipitacion",
    }

    if current["area"] > 0:

        status["estado"] = (
            "precipitacion detectada"
        )

    if len(results) < 2:
        return status

    previous = results[-2]

    movement = motion_between(
        previous,
        current,
    )

    speed = calculate_speed(
        movement,
        previous["timestamp"],
        current["timestamp"],
    )

    if speed is not None:

        status[
            "velocidad_kmh"
        ] = speed[
            "velocidad_kmh"
        ]

    if movement is not None:

        status[
            "direccion_grados"
        ] = movement[
            "direccion_grados"
        ]

        status[
            "direccion"
        ] = direction_name(
            movement[
                "direccion_grados"
            ]
        )

    previous_area = (
        previous["area"]
    )

    current_area = (
        current["area"]
    )

    if previous_area > 0:

        change = (
            current_area
            / previous_area
            - 1.0
        ) * 100.0

        if change >= 15:

            status[
                "fortalecimiento"
            ] = "fortaleciendose"

        elif change <= -15:

            status[
                "fortalecimiento"
            ] = "debilitandose"

        else:

            status[
                "fortalecimiento"
            ] = "estable"

    if (
        movement is not None
        and speed is not None
        and current["component"]
        is not None
    ):

        confidence = 0.75

        if current["area"] >= 100:
            confidence += 0.10

        if (
            speed["velocidad_kmh"]
            <= 120
        ):
            confidence += 0.10

        status[
            "confianza_movimiento"
        ] = round(
            min(
                confidence,
                0.95
            ),
            2,
        )

    return status


def save_status(
    status
):

    path = os.path.join(
        DATA_DIR,
        "status.json",
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            status,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main():

    print(
        "=== CLIMAAR RAINVIEWER V5 ==="
    )

    host, frames = (
        fetch_rainviewer()
    )

    xt, yt = latlon_to_tile(
        LAT,
        LON,
        ZOOM,
    )

    print(
        f"Host: {host}"
    )

    print(
        f"Frames disponibles: "
        f"{len(frames)}"
    )

    print(
        f"Tile central: "
        f"X={xt} Y={yt}"
    )

    selected_frames = (
        frames[-12:]
    )

    results = []

    for frame in selected_frames:

        (
            image,
            tiles_ok,
            timestamp,
            path,
        ) = download_frame(
            host,
            frame,
            xt,
            yt,
        )

        print(
            f"{timestamp} -> "
            f"tiles OK: "
            f"{tiles_ok}/9"
        )

        if tiles_ok == 0:

            print(
                "Frame descartado: "
                "0 tiles."
            )

            continue

        analysis = analyze_frame(
            image
        )

        results.append(
            {
                "timestamp":
                    timestamp,

                "path":
                    path,

                "image":
                    image,

                **analysis,
            }
        )

    if not results:

        print(
            "ERROR CRITICO: "
            "No se pudo obtener "
            "ningun frame."
        )

        sys.exit(1)

    current = results[-1]

    actual_path = os.path.join(
        DATA_DIR,
        "actual.png",
    )

    current["image"].save(
        actual_path
    )

    status = analyze_sequence(
        results
    )

    status[
        "timestamp_frame"
    ] = current[
        "timestamp"
    ]

    status[
        "frame_utc"
    ] = datetime.fromtimestamp(
        current["timestamp"],
        timezone.utc,
    ).isoformat()

    status["tiles"] = 9

    save_status(
        status
    )

    print(
        json.dumps(
            status,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "=== V5 OK ==="
    )


if __name__ == "__main__":
    main()
