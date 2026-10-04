import os
import sys
import json
import math
import csv
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
HISTORY_DIR = os.path.join(DATA_DIR, "historico")
FEATURES_CSV = os.path.join(DATA_DIR, "radar_features_rainviewer.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

API_URL = "https://api.rainviewer.com/public/weather-maps.json"

HEADERS = {
    "User-Agent": "ClimaAR/5.1",
    "Referer": "https://www.rainviewer.com/",
    "Accept": "image/png,image/*;q=0.8,*/*;q=0.5",
}

TILE_SIZE = 512
GRID_RADIUS = 1
MIN_TILE_BYTES = 200
MAX_HISTORY_FRAMES = 144


def latlon_to_tile(lat, lon, zoom):
    n = 2.0 ** zoom
    xtile = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)

    ytile = (
        1.0 - math.asinh(math.tan(lat_rad)) / math.pi
    ) / 2.0 * n

    return int(xtile), int(ytile)


def pixels_per_km(lat, zoom, tile_size):
    world_km = 40075.016

    km_x = (
        world_km * math.cos(math.radians(lat))
        / ((2 ** zoom) * tile_size)
    )

    km_y = world_km / ((2 ** zoom) * tile_size)

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


def download_frame(host, frame, xt, yt):
    path = frame["path"]
    timestamp = int(frame["time"])

    size = TILE_SIZE * 3

    image = Image.new(
        "RGBA",
        (size, size),
        (0, 0, 0, 0),
    )

    successful_tiles = 0

    for dx in range(
        -GRID_RADIUS,
        GRID_RADIUS + 1
    ):
        for dy in range(
            -GRID_RADIUS,
            GRID_RADIUS + 1
        ):
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
                    BytesIO(response.content)
                ).convert("RGBA")

                x = (
                    dx + GRID_RADIUS
                ) * TILE_SIZE

                y = (
                    dy + GRID_RADIUS
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
    if (
        movement is None
        or previous_timestamp is None
    ):
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


def strengthening_state(
    previous_area,
    current_area
):
    if (
        previous_area <= 0
        or current_area <= 0
    ):
        return "sin_datos"

    change = (
        current_area
        / previous_area
        - 1.0
    ) * 100.0

    if change >= 15:
        return "fortaleciendose"

    if change <= -15:
        return "debilitandose"

    return "estable"


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

        "timestamp_frame":
            current["timestamp"],

        "frame_utc":
            datetime.fromtimestamp(
                current["timestamp"],
                timezone.utc,
            ).isoformat(),

        "tiles":
            current["tiles_ok"],
    }

    if current["area"] > 0:
        status["estado"] = (
            "precipitacion_detectada"
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

    status[
        "fortalecimiento"
    ] = strengthening_state(
        previous["area"],
        current["area"],
    )

    if (
        movement is not None
        and speed is not None
        and current["component"]
        is not None
    ):
        confidence = 0.60

        if current["area"] >= 100:
            confidence += 0.10

        if (
            speed["velocidad_kmh"]
            <= 120
        ):
            confidence += 0.10

        if (
            movement["movimiento_km"]
            >= 1
        ):
            confidence += 0.10

        if len(results) >= 4:
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


def save_features(
    results
):
    rows = []

    for item in results:
        component = item[
            "component"
        ]

        row = {
            "frame_time":
                item["timestamp"],

            "frame_utc":
                datetime.fromtimestamp(
                    item["timestamp"],
                    timezone.utc,
                ).isoformat(),

            "source":
                "RainViewer",

            "latitud":
                LAT,

            "longitud":
                LON,

            "area_px":
                item["area"],

            "distance_km":
                (
                    round(
                        item["distance_km"],
                        2,
                    )
                    if item["distance_km"]
                    is not None
                    else ""
                ),

            "tiles_ok":
                item["tiles_ok"],

            "centroid_x":
                (
                    round(
                        component[
                            "centroid_x"
                        ],
                        2,
                    )
                    if component
                    is not None
                    else ""
                ),

            "centroid_y":
                (
                    round(
                        component[
                            "centroid_y"
                        ],
                        2,
                    )
                    if component
                    is not None
                    else ""
                ),

            "bbox_x":
                (
                    component[
                        "bbox"
                    ][0]
                    if component
                    is not None
                    else ""
                ),

            "bbox_y":
                (
                    component[
                        "bbox"
                    ][1]
                    if component
                    is not None
                    else ""
                ),

            "bbox_width":
                (
                    component[
                        "bbox"
                    ][2]
                    if component
                    is not None
                    else ""
                ),

            "bbox_height":
                (
                    component[
                        "bbox"
                    ][3]
                    if component
                    is not None
                    else ""
                ),
        }

        rows.append(
            row
        )

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    existing = []

    if os.path.exists(
        FEATURES_CSV
    ):
        try:
            with open(
                FEATURES_CSV,
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                existing = list(
                    csv.DictReader(
                        file
                    )
                )

        except Exception as exc:
            print(
                f"AVISO: no se pudo leer "
                f"{FEATURES_CSV}: {exc}"
            )

    by_timestamp = {}

    for row in existing:
        timestamp = row.get(
            "frame_time"
        )

        if timestamp:
            by_timestamp[
                str(timestamp)
            ] = row

    for row in rows:
        by_timestamp[
            str(row["frame_time"])
        ] = row

    combined = list(
        by_timestamp.values()
    )

    combined.sort(
        key=lambda row: int(
            float(
                row["frame_time"]
            )
        )
    )

    with open(
        FEATURES_CSV,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            combined
        )

    print(
        f"Features guardadas: "
        f"{len(combined)} frames"
    )


def save_history_frame(
    item
):
    timestamp = item[
        "timestamp"
    ]

    filename = os.path.join(
        HISTORY_DIR,
        f"radar_{timestamp}.png",
    )

    if not os.path.exists(
        filename
    ):
        item["image"].save(
            filename,
            optimize=True,
        )

        print(
            f"Historico guardado: "
            f"{filename}"
        )

    else:
        print(
            f"Historico ya existe: "
            f"{filename}"
        )


def cleanup_history():
    files = []

    for filename in os.listdir(
        HISTORY_DIR
    ):
        if not filename.startswith(
            "radar_"
        ):
            continue

        if not filename.endswith(
            ".png"
        ):
            continue

        path = os.path.join(
            HISTORY_DIR,
            filename,
        )

        try:
            timestamp = int(
                filename[
                    len("radar_"):-4
                ]
            )

        except ValueError:
            continue

        files.append(
            (
                timestamp,
                path
            )
        )

    files.sort(
        key=lambda item: item[0]
    )

    while len(files) > MAX_HISTORY_FRAMES:
        _, path = files.pop(0)

        try:
            os.remove(
                path
            )

            print(
                f"Historico eliminado: "
                f"{path}"
            )

        except OSError as exc:
            print(
                f"AVISO: no se pudo eliminar "
                f"{path}: {exc}"
            )


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
        "=== CLIMAAR RAINVIEWER V5.1 ==="
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

        item = {
            "timestamp":
                timestamp,

            "path":
                path,

            "image":
                image,

            "tiles_ok":
                tiles_ok,

            **analysis,
        }

        results.append(
            item
        )

        save_history_frame(
            item
        )

    if not results:

        print(
            "ERROR CRITICO: "
            "No se pudo obtener "
            "ningun frame."
        )

        sys.exit(1)

    results.sort(
        key=lambda item:
            item["timestamp"]
    )

    current = results[-1]

    actual_path = os.path.join(
        DATA_DIR,
        "actual.png",
    )

    current["image"].save(
        actual_path,
        optimize=True,
    )

    save_features(
        results
    )

    cleanup_history()

    status = analyze_sequence(
        results
    )

    if status is None:

        print(
            "ERROR CRITICO: "
            "No se pudo generar "
            "status."
        )

        sys.exit(1)

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
        "=== CLIMAAR RAINVIEWER V5.1 OK ==="
    )


if __name__ == "__main__":
    main()
