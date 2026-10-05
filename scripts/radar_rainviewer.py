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


# ============================================================
# CLIMAAR - RADAR RAINVIEWER + NOWCAST
# Version 6.1
# ============================================================

LAT = -38.71
LON = -62.26

ZOOM = 7

DATA_DIR = "data/radar"
HISTORY_DIR = os.path.join(DATA_DIR, "historico")

FEATURES_CSV = os.path.join(
    DATA_DIR,
    "radar_features_rainviewer.csv"
)

NOWCAST_FILE = os.path.join(
    DATA_DIR,
    "radar_nowcast.json"
)

ACTUAL_FILE = os.path.join(
    DATA_DIR,
    "actual.png"
)

API_URL = (
    "https://api.rainviewer.com/public/weather-maps.json"
)

HEADERS = {
    "User-Agent": "ClimaAR/6.1",
    "Referer": "https://www.rainviewer.com/",
    "Accept": "image/png,image/*;q=0.8,*/*;q=0.5",
}

TILE_SIZE = 512
GRID_RADIUS = 1

MIN_TILE_BYTES = 200

MAX_HISTORY_FRAMES = 144

# Cantidad máxima de frames usados para calcular movimiento.
TRACK_FRAMES = 6

# Mínimo de movimiento para considerar que hay desplazamiento real.
MIN_MOVEMENT_KM = 1.0

# Ventanas de proyección del nowcast.
PROJECTION_MINUTES = (15, 30, 45, 60, 90)


# ============================================================
# DIRECTORIOS
# ============================================================

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)


# ============================================================
# UTILIDADES
# ============================================================

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


def pixels_per_km(lat, zoom, tile_size):

    world_km = 40075.016

    km_x = (
        world_km
        * math.cos(
            math.radians(lat)
        )
        / (
            (2 ** zoom)
            * tile_size
        )
    )

    km_y = (
        world_km
        / (
            (2 ** zoom)
            * tile_size
        )
    )

    return km_x, km_y


def frame_datetime(timestamp):

    return datetime.fromtimestamp(
        int(timestamp),
        timezone.utc
    ).isoformat()


# ============================================================
# RAINVIEWER API
# ============================================================

def fetch_rainviewer():

    try:

        response = requests.get(
            API_URL,
            headers=HEADERS,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

    except Exception as exc:

        print(
            f"ERROR API RAINVIEWER: {exc}"
        )

        sys.exit(1)

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

        print(
            "ERROR CRITICO: "
            "RainViewer no devolvio frames."
        )

        sys.exit(1)

    return host, frames


# ============================================================
# DESCARGA DE UN FRAME
# ============================================================

def download_frame(
    host,
    frame,
    xt,
    yt
):

    path = frame["path"]

    timestamp = int(
        frame["time"]
    )

    image_size = (
        TILE_SIZE
        * 3
    )

    image = Image.new(
        "RGBA",
        (
            image_size,
            image_size
        ),
        (
            0,
            0,
            0,
            0
        )
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
                f"/{TILE_SIZE}"
                f"/{ZOOM}"
                f"/{xt + dx}"
                f"/{yt + dy}"
                f"/2/1_1.png"
            )

            try:

                response = requests.get(
                    url,
                    headers=HEADERS,
                    timeout=20
                )

                if (
                    response.status_code != 200
                    or len(response.content)
                    < MIN_TILE_BYTES
                ):

                    continue

                tile = (
                    Image
                    .open(
                        BytesIO(
                            response.content
                        )
                    )
                    .convert("RGBA")
                )

                x = (
                    dx
                    + GRID_RADIUS
                ) * TILE_SIZE

                y = (
                    dy
                    + GRID_RADIUS
                ) * TILE_SIZE

                image.alpha_composite(
                    tile,
                    (
                        x,
                        y
                    )
                )

                successful_tiles += 1

            except Exception as exc:

                print(
                    "ERROR TILE:",
                    exc
                )

    return (
        image,
        successful_tiles,
        timestamp,
        path
    )


# ============================================================
# MASCARA RADAR
# ============================================================

def radar_mask(image):

    array = np.asarray(
        image
    )

    alpha = array[:, :, 3]

    mask = np.where(
        alpha >= 20,
        255,
        0
    ).astype(
        np.uint8
    )

    kernel = np.ones(
        (
            3,
            3
        ),
        np.uint8
    )

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


# ============================================================
# COMPONENTE PRINCIPAL
# ============================================================

def largest_component(mask):

    count, labels, stats, centers = (
        cv2.connectedComponentsWithStats(
            mask,
            8
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

            "area_px":
                area,

            "centroid_x":
                float(
                    centers[i][0]
                ),

            "centroid_y":
                float(
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
                )
            ]
        }

        if (
            best is None
            or area > best["area_px"]
        ):

            best = candidate

    return best


# ============================================================
# DISTANCIA A BAHIA BLANCA
# ============================================================

def distance_to_bahia(mask):

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:
        return None

    height, width = (
        mask.shape
    )

    center_x = (
        width / 2.0
    )

    center_y = (
        height / 2.0
    )

    km_x, km_y = (
        pixels_per_km(
            LAT,
            ZOOM,
            TILE_SIZE
        )
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


# ============================================================
# ANALISIS DE FRAME
# ============================================================

def analyze_frame(
    image,
    timestamp,
    tiles_ok,
    path
):

    mask = radar_mask(
        image
    )

    area = int(
        np.count_nonzero(
            mask
        )
    )

    component = (
        largest_component(
            mask
        )
    )

    distance = (
        distance_to_bahia(
            mask
        )
    )

    return {

        "timestamp":
            int(timestamp),

        "frame_utc":
            frame_datetime(
                timestamp
            ),

        "tiles_ok":
            int(tiles_ok),

        "path":
            path,

        "area":
            area,

        "component":
            component,

        "distance_km":
            distance
    }


# ============================================================
# MOVIMIENTO ENTRE DOS FRAMES
# ============================================================

def motion_between(
    previous,
    current
):

    if (
        previous is None
        or current is None
    ):
        return None

    pc = previous.get(
        "component"
    )

    cc = current.get(
        "component"
    )

    if (
        pc is None
        or cc is None
    ):
        return None

    dx_px = (
        cc["centroid_x"]
        - pc["centroid_x"]
    )

    dy_px = (
        cc["centroid_y"]
        - pc["centroid_y"]
    )

    km_x, km_y = (
        pixels_per_km(
            LAT,
            ZOOM,
            TILE_SIZE
        )
    )

    east_km = (
        dx_px * km_x
    )

    south_km = (
        dy_px * km_y
    )

    movement_km = math.hypot(
        east_km,
        south_km
    )

    direction = (
        math.degrees(
            math.atan2(
                east_km,
                -south_km
            )
        )
        + 360.0
    ) % 360.0

    elapsed_minutes = (
        current["timestamp"]
        - previous["timestamp"]
    ) / 60.0

    if elapsed_minutes <= 0:
        return None

    speed_kmh = (
        movement_km
        / (
            elapsed_minutes
            / 60.0
        )
    )

    return {

        "dx_px":
            dx_px,

        "dy_px":
            dy_px,

        "movimiento_km":
            movement_km,

        "direccion_grados":
            direction,

        "velocidad_kmh":
            speed_kmh,

        "intervalo_minutos":
            elapsed_minutes
    }


# ============================================================
# NOMBRE DE DIRECCION
# ============================================================

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
        "NO"
    ]

    index = int(
        (
            degrees
            + 22.5
        ) / 45
    ) % 8

    return names[index]


# ============================================================
# FORTALECIMIENTO
# ============================================================

def strengthening_state(
    previous_area,
    current_area
):

    if (
        previous_area <= 0
        or current_area <= 0
    ):

        return (
            "sin_datos",
            None
        )

    change = (
        current_area
        / previous_area
        - 1.0
    ) * 100.0

    if change >= 15:

        return (
            "fortaleciendose",
            change
        )

    if change <= -15:

        return (
            "debilitandose",
            change
        )

    return (
        "estable",
        change
    )


# ============================================================
# TRACK MULTIFRAME
# ============================================================

def build_track(
    results
):

    valid = []

    for item in results:

        if (
            item.get("component")
            is not None
        ):

            valid.append(
                item
            )

    if len(valid) < 2:

        return {
            "valido": False,
            "motivo":
                "No hay suficientes frames con precipitacion."
        }

    valid = valid[
        -TRACK_FRAMES:
    ]

    movements = []

    for i in range(
        1,
        len(valid)
    ):

        movement = (
            motion_between(
                valid[i - 1],
                valid[i]
            )
        )

        if movement is not None:

            movements.append(
                movement
            )

    if not movements:

        return {
            "valido": False,
            "motivo":
                "No se pudo calcular movimiento."
        }

    speeds = np.array(
        [
            x["velocidad_kmh"]
            for x in movements
        ],
        dtype=float
    )

    directions = np.array(
        [
            x["direccion_grados"]
            for x in movements
        ],
        dtype=float
    )

    distances = np.array(
        [
            x["movimiento_km"]
            for x in movements
        ],
        dtype=float
    )

    median_speed = float(
        np.median(
            speeds
        )
    )

    median_distance = float(
        np.median(
            distances
        )
    )

    # Promedio circular de direcciones.
    radians = np.deg2rad(
        directions
    )

    mean_sin = np.mean(
        np.sin(radians)
    )

    mean_cos = np.mean(
        np.cos(radians)
    )

    mean_direction = (
        math.degrees(
            math.atan2(
                mean_sin,
                mean_cos
            )
        )
        + 360
    ) % 360

    # Consistencia de dirección.
    resultant = math.sqrt(
        mean_sin ** 2
        + mean_cos ** 2
    )

    # Consistencia de velocidad.
    if median_speed > 0:

        speed_cv = (
            float(
                np.std(
                    speeds
                )
            )
            / median_speed
        )

    else:

        speed_cv = 1.0

    speed_consistency = max(
        0.0,
        min(
            1.0,
            1.0
            - speed_cv
        )
    )

    direction_consistency = max(
        0.0,
        min(
            1.0,
            resultant
        )
    )

    sample_score = min(
        1.0,
        len(movements) / 5.0
    )

    confidence = (
        0.45
        * direction_consistency
        + 0.35
        * speed_consistency
        + 0.20
        * sample_score
    )

    return {

        "valido":
            True,

        "frames_validos":
            len(valid),

        "intervalos":
            len(movements),

        "velocidad_kmh":
            round(
                median_speed,
                1
            ),

        "direccion_grados":
            round(
                mean_direction,
                1
            ),

        "direccion":
            direction_name(
                mean_direction
            ),

        "movimiento_mediano_km":
            round(
                median_distance,
                2
            ),

        "confianza":
            round(
                min(
                    confidence,
                    0.95
                ),
                2
            )
    }


# ============================================================
# ¿LA CELULA VA HACIA BAHIA?
# ============================================================

def movement_towards_bahia(
    current,
    track
):

    if (
        not track
        or not track.get(
            "valido",
            False
        )
    ):

        return False

    component = (
        current.get(
            "component"
        )
    )

    if component is None:
        return False

    cx = component[
        "centroid_x"
    ]

    cy = component[
        "centroid_y"
    ]

    width = (
        TILE_SIZE * 3
    )

    height = (
        TILE_SIZE * 3
    )

    center_x = (
        width / 2.0
    )

    center_y = (
        height / 2.0
    )

    dx_to_bahia = (
        center_x - cx
    )

    dy_to_bahia = (
        center_y - cy
    )

    distance_to_target = math.hypot(
        dx_to_bahia,
        dy_to_bahia
    )

    if distance_to_target < 5:
        return True

    target_angle = (
        math.degrees(
            math.atan2(
                dx_to_bahia,
                -dy_to_bahia
            )
        )
        + 360
    ) % 360

    current_angle = track[
        "direccion_grados"
    ]

    diff = abs(
        (
            current_angle
            - target_angle
            + 180
        ) % 360
        - 180
    )

    return diff <= 45


# ============================================================
# ETA
# ============================================================

def calculate_eta(
    current,
    track
):

    if (
        not track
        or not track.get(
            "valido",
            False
        )
    ):

        return None

    speed = track.get(
        "velocidad_kmh"
    )

    if (
        speed is None
        or speed < 5
    ):

        return None

    if not movement_towards_bahia(
        current,
        track
    ):

        return None

    distance = (
        current.get(
            "distance_km"
        )
    )

    if (
        distance is None
        or distance <= 0
    ):

        return None

    eta = (
        distance
        / speed
        * 60.0
    )

    if eta < 0:
        return None

    if eta > 360:
        return None

    return round(
        eta,
        1
    )


# ============================================================
# PROYECCION
# ============================================================

def projection(
    current,
    track,
    minutes
):

    if (
        not track
        or not track.get(
            "valido",
            False
        )
    ):

        return None

    component = (
        current.get(
            "component"
        )
    )

    if component is None:
        return None

    speed = (
        track["velocidad_kmh"]
    )

    direction = math.radians(
        track["direccion_grados"]
    )

    distance_km = (
        speed
        * minutes
        / 60.0
    )

    km_x, km_y = (
        pixels_per_km(
            LAT,
            ZOOM,
            TILE_SIZE
        )
    )

    east_km = (
        math.sin(direction)
        * distance_km
    )

    south_km = (
        -math.cos(direction)
        * distance_km
    )

    projected_x = (
        component["centroid_x"]
        + east_km / km_x
    )

    projected_y = (
        component["centroid_y"]
        + south_km / km_y
    )

    width = (
        TILE_SIZE * 3
    )

    height = (
        TILE_SIZE * 3
    )

    inside_radar = (
        projected_x >= 0
        and projected_x <= width
        and projected_y >= 0
        and projected_y <= height
    )

    return {

        "minutos":
            minutes,

        "distancia_proyectada_km":
            round(
                distance_km,
                1
            ),

        "x":
            round(
                projected_x,
                1
            ),

        "y":
            round(
                projected_y,
                1
            ),

        "dentro_area_radar":
            bool(
                inside_radar
            )
    }


# ============================================================
# ANALISIS COMPLETO
# ============================================================

def analyze_sequence(
    results
):

    if not results:
        return None

    current = results[-1]

    status = {

        "fuente":
            "RainViewer",

        "actualizado":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "latitud":
            LAT,

        "longitud":
            LON,

        "frames_analizados":
            len(results),

        "frame_actual":
            current["timestamp"],

        "frame_actual_utc":
            current["frame_utc"],

        "tiles_ok":
            current["tiles_ok"],

        "actividad":
            current["area"] > 0,

        "area_px":
            current["area"],

        "distancia_km":
            (
                round(
                    current[
                        "distance_km"
                    ],
                    1
                )
                if current[
                    "distance_km"
                ] is not None
                else None
            ),

        "fortalecimiento":
            "sin_datos",

        "cambio_area_pct":
            None,

        "velocidad_kmh":
            None,

        "direccion":
            None,

        "direccion_grados":
            None,

        "movimiento_hacia_bahia":
            False,

        "eta_minutos":
            None,

        "proyecciones":
            {},

        # Compatibilidad con la app actual.
        "proyeccion_30_min":
            None,

        "proyeccion_60_min":
            None,

        "confianza_movimiento":
            0.0,

        "estado":
            (
                "precipitacion_detectada"
                if current["area"] > 0
                else
                "sin_precipitacion_detectada"
            )
    }

    if len(results) < 2:
        return status

    previous = results[-2]

    strengthening, change = (
        strengthening_state(
            previous["area"],
            current["area"]
        )
    )

    status[
        "fortalecimiento"
    ] = strengthening

    if change is not None:

        status[
            "cambio_area_pct"
        ] = round(
            change,
            1
        )

    track = build_track(
        results
    )

    if not track.get(
        "valido",
        False
    ):

        return status

    status[
        "velocidad_kmh"
    ] = track[
        "velocidad_kmh"
    ]

    status[
        "direccion_grados"
    ] = track[
        "direccion_grados"
    ]

    status[
        "direccion"
    ] = track[
        "direccion"
    ]

    status[
        "confianza_movimiento"
    ] = track[
        "confianza"
    ]

    toward = (
        movement_towards_bahia(
            current,
            track
        )
    )

    status[
        "movimiento_hacia_bahia"
    ] = toward

    eta = calculate_eta(
        current,
        track
    )

    status[
        "eta_minutos"
    ] = eta

    for minutes in PROJECTION_MINUTES:

        projected = projection(
            current,
            track,
            minutes
        )

        status[
            "proyecciones"
        ][
            str(minutes)
        ] = projected

    # Mantener las claves que ya consume la app 4.4.0.
    status[
        "proyeccion_30_min"
    ] = status[
        "proyecciones"
    ].get("30")

    status[
        "proyeccion_60_min"
    ] = status[
        "proyecciones"
    ].get("60")

    return status


# ============================================================
# GUARDAR FEATURES
# ============================================================

def save_features(
    results
):

    fieldnames = [

        "frame_time",
        "frame_utc",
        "source",
        "latitud",
        "longitud",
        "area_px",
        "distance_km",
        "tiles_ok",
        "centroid_x",
        "centroid_y",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height"
    ]

    with open(
        FEATURES_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for item in results:

            component = (
                item["component"]
            )

            if component:

                bbox = (
                    component["bbox"]
                )

                centroid_x = round(
                    component[
                        "centroid_x"
                    ],
                    2
                )

                centroid_y = round(
                    component[
                        "centroid_y"
                    ],
                    2
                )

                bbox_x = bbox[0]
                bbox_y = bbox[1]
                bbox_width = bbox[2]
                bbox_height = bbox[3]

            else:

                centroid_x = ""
                centroid_y = ""
                bbox_x = ""
                bbox_y = ""
                bbox_width = ""
                bbox_height = ""

            writer.writerow({

                "frame_time":
                    item["timestamp"],

                "frame_utc":
                    item["frame_utc"],

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
                            item[
                                "distance_km"
                            ],
                            2
                        )
                        if item[
                            "distance_km"
                        ] is not None
                        else ""
                    ),

                "tiles_ok":
                    item["tiles_ok"],

                "centroid_x":
                    centroid_x,

                "centroid_y":
                    centroid_y,

                "bbox_x":
                    bbox_x,

                "bbox_y":
                    bbox_y,

                "bbox_width":
                    bbox_width,

                "bbox_height":
                    bbox_height
            })


# ============================================================
# HISTORICO
# ============================================================

def save_history(
    timestamp,
    image
):

    filename = os.path.join(
        HISTORY_DIR,
        f"radar_{timestamp}.png"
    )

    image.save(
        filename
    )

    files = []

    for name in os.listdir(
        HISTORY_DIR
    ):

        if (
            name.startswith(
                "radar_"
            )
            and name.endswith(
                ".png"
            )
        ):

            path = os.path.join(
                HISTORY_DIR,
                name
            )

            try:

                files.append(
                    (
                        int(
                            name[
                                6:-4
                            ]
                        ),
                        path
                    )
                )

            except ValueError:

                pass

    files.sort(
        key=lambda x: x[0]
    )

    while len(files) > MAX_HISTORY_FRAMES:

        _, old_path = files.pop(
            0
        )

        try:
            os.remove(
                old_path
            )
        except OSError:
            pass


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CLIMAAR - RADAR RAINVIEWER NOWCAST 6.1")
    print("=" * 70)

    host, frames = (
        fetch_rainviewer()
    )

    xt, yt = latlon_to_tile(
        LAT,
        LON,
        ZOOM
    )

    # RainViewer normalmente entrega
    # alrededor de 12 frames pasados.
    # Usamos todos los disponibles.
    selected_frames = frames[
        -12:
    ]

    results = []

    latest_image = None

    for index, frame in enumerate(
        selected_frames,
        start=1
    ):

        print(
            f"Procesando frame "
            f"{index}/{len(selected_frames)}"
        )

        image, tiles_ok, timestamp, path = (
            download_frame(
                host,
                frame,
                xt,
                yt
            )
        )

        if tiles_ok == 0:

            print(
                "Frame sin tiles validas."
            )

            continue

        analysis = analyze_frame(
            image,
            timestamp,
            tiles_ok,
            path
        )

        results.append(
            analysis
        )

        latest_image = image

    if not results:

        print(
            "ERROR: no se pudo procesar "
            "ningun frame."
        )

        sys.exit(1)

    results.sort(
        key=lambda x:
        x["timestamp"]
    )

    current = results[-1]

    if latest_image is not None:

        latest_image.save(
            ACTUAL_FILE
        )

        save_history(
            current["timestamp"],
            latest_image
        )

    save_features(
        results
    )

    nowcast = analyze_sequence(
        results
    )

    output = {

        "version":
            "6.1",

        "app":
            "ClimaAR",

        "ubicacion": {

            "latitud":
                LAT,

            "longitud":
                LON,

            "ciudad":
                "Bahia Blanca"
        },

        "fuente":
            "RainViewer",

        "nowcast":
            nowcast,

        "historial": {

            "frames_procesados":
                len(results),

            "frames_disponibles":
                len(selected_frames),

            "max_history_frames":
                MAX_HISTORY_FRAMES
        },

        "nota": (
            "El nowcast se calcula "
            "por seguimiento de movimiento "
            "de la precipitacion radar. "
            "Las proyecciones son extrapolaciones "
            "de la trayectoria observada y no "
            "constituyen una garantia de trayectoria futura. "
            "La ETA y las proyecciones "
            "deben interpretarse con su confianza."
        )
    }

    with open(
        NOWCAST_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 70)
    print("NOWCAST")
    print("=" * 70)

    print(
        json.dumps(
            nowcast,
            ensure_ascii=False,
            indent=2
        )
    )

    print()
    print(
        f"Frames procesados: "
        f"{len(results)}"
    )

    print(
        f"Actual: "
        f"{ACTUAL_FILE}"
    )

    print(
        f"Nowcast: "
        f"{NOWCAST_FILE}"
    )

    print(
        f"Features: "
        f"{FEATURES_CSV}"
    )


if __name__ == "__main__":
    main()
