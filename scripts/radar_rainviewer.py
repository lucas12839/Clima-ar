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
# Version 7.0
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
    "User-Agent": "ClimaAR/7.0",
    "Referer": "https://www.rainviewer.com/",
    "Accept": "image/png,image/*;q=0.8,*/*;q=0.5",
}

TILE_SIZE = 512
GRID_RADIUS = 1

MIN_TILE_BYTES = 200

MAX_HISTORY_FRAMES = 144

# Cantidad máxima de frames utilizados
# para calcular la trayectoria.
TRACK_FRAMES = 6

# Área mínima para considerar un núcleo.
MIN_COMPONENT_AREA = 20

# Máximo de núcleos simultáneos.
MAX_COMPONENTS = 12

# Velocidad máxima razonable para mantener
# la identidad de un núcleo entre frames.
MAX_TRACK_SPEED_KMH = 180.0

# Ventanas de proyección.
PROJECTION_MINUTES = (
    15,
    30,
    45,
    60,
    90
)


# ============================================================
# DIRECTORIOS
# ============================================================

os.makedirs(
    DATA_DIR,
    exist_ok=True
)

os.makedirs(
    HISTORY_DIR,
    exist_ok=True
)


# ============================================================
# UTILIDADES
# ============================================================

def latlon_to_tile(
    lat,
    lon,
    zoom
):

    n = 2.0 ** zoom

    x = (
        (lon + 180.0)
        / 360.0
        * n
    )

    lat_rad = math.radians(
        lat
    )

    y = (
        1.0
        -
        math.asinh(
            math.tan(
                lat_rad
            )
        )
        / math.pi
    ) / 2.0 * n

    return (
        int(x),
        int(y)
    )


def pixels_per_km(
    lat,
    zoom,
    tile_size
):

    world_km = 40075.016

    km_x = (
        world_km
        *
        math.cos(
            math.radians(lat)
        )
        /
        (
            (2 ** zoom)
            *
            tile_size
        )
    )

    km_y = (
        world_km
        /
        (
            (2 ** zoom)
            *
            tile_size
        )
    )

    return (
        km_x,
        km_y
    )


def frame_datetime(
    timestamp
):

    return datetime.fromtimestamp(
        int(timestamp),
        timezone.utc
    ).isoformat()


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
        )
        / 45
    ) % 8

    return names[index]


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

    return (
        host,
        frames
    )


# ============================================================
# DESCARGA DE FRAME
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
        TILE_SIZE * 3
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
                    or
                    len(response.content)
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

def radar_mask(
    image
):

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
# DETECCION DE MULTIPLES NUCLEOS
# ============================================================

def components_from_mask(
    mask
):

    count, labels, stats, centers = (
        cv2.connectedComponentsWithStats(
            mask,
            8
        )
    )

    components = []

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

        if area < MIN_COMPONENT_AREA:
            continue

        components.append({

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
        })

    components.sort(
        key=lambda c:
        c["area_px"],
        reverse=True
    )

    return components[
        :MAX_COMPONENTS
    ]


def largest_component(
    mask
):

    components = (
        components_from_mask(
            mask
        )
    )

    if not components:
        return None

    return components[0]


# ============================================================
# DISTANCIA A BAHIA
# ============================================================

def distance_point_to_bahia(
    cx,
    cy
):

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

    km_x, km_y = (
        pixels_per_km(
            LAT,
            ZOOM,
            TILE_SIZE
        )
    )

    return math.hypot(
        (
            cx
            - center_x
        ) * km_x,

        (
            cy
            - center_y
        ) * km_y
    )


def distance_to_bahia(
    mask
):

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:
        return None

    width = mask.shape[1]
    height = mask.shape[0]

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
        np.min(
            distances
        )
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

    components = (
        components_from_mask(
            mask
        )
    )

    area = int(
        np.count_nonzero(
            mask
        )
    )

    for comp in components:

        comp[
            "distance_km"
        ] = round(
            distance_point_to_bahia(
                comp[
                    "centroid_x"
                ],
                comp[
                    "centroid_y"
                ]
            ),
            2
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

        "components":
            components,

        "component":
            (
                components[0]
                if components
                else None
            ),

        "distance_km":
            distance_to_bahia(
                mask
            )
    }


# ============================================================
# COSTO DE ASIGNACION DE NUCLEOS
# ============================================================

def component_cost(
    old,
    new,
    elapsed_minutes
):

    km_x, km_y = (
        pixels_per_km(
            LAT,
            ZOOM,
            TILE_SIZE
        )
    )

    dx = (
        new["centroid_x"]
        -
        old["centroid_x"]
    ) * km_x

    dy = (
        new["centroid_y"]
        -
        old["centroid_y"]
    ) * km_y

    distance_km = math.hypot(
        dx,
        dy
    )

    max_distance = max(
        5.0,
        MAX_TRACK_SPEED_KMH
        *
        max(
            elapsed_minutes,
            1.0
        )
        / 60.0
    )

    if (
        distance_km
        > max_distance
    ):

        return None

    area_ratio = (
        max(
            old["area_px"],
            new["area_px"]
        )
        /
        max(
            1,
            min(
                old["area_px"],
                new["area_px"]
            )
        )
    )

    area_penalty = min(
        2.0,
        math.log(
            area_ratio
        )
    )

    return (
        distance_km
        / max_distance
        +
        0.15
        * area_penalty
    )


# ============================================================
# TRACKING DE MULTIPLES NUCLEOS
# ============================================================

def track_components(
    results
):

    tracks = {}

    next_id = 1

    previous = []

    for frame_index, frame in enumerate(
        results
    ):

        components = (
            frame["components"]
        )

        for comp in components:

            comp[
                "track_id"
            ] = None

        if frame_index == 0:

            for comp in components:

                comp[
                    "track_id"
                ] = next_id

                tracks[
                    next_id
                ] = [

                    (
                        frame["timestamp"],
                        comp["centroid_x"],
                        comp["centroid_y"],
                        comp["area_px"],
                        frame_index
                    )
                ]

                next_id += 1

            previous = components

            continue

        elapsed = (
            frame["timestamp"]
            -
            results[
                frame_index - 1
            ]["timestamp"]
        ) / 60.0

        candidates = []

        for pi, old in enumerate(
            previous
        ):

            for ci, new in enumerate(
                components
            ):

                if (
                    new["track_id"]
                    is not None
                ):

                    continue

                cost = (
                    component_cost(
                        old,
                        new,
                        elapsed
                    )
                )

                if cost is not None:

                    candidates.append(
                        (
                            cost,
                            pi,
                            ci
                        )
                    )

        used_old = set()
        used_new = set()

        for _, pi, ci in sorted(
            candidates
        ):

            if (
                pi in used_old
                or
                ci in used_new
            ):

                continue

            old = previous[pi]
            new = components[ci]

            track_id = old.get(
                "track_id"
            )

            if track_id is None:
                continue

            new[
                "track_id"
            ] = track_id

            tracks.setdefault(
                track_id,
                []
            ).append(

                (
                    frame["timestamp"],
                    new["centroid_x"],
                    new["centroid_y"],
                    new["area_px"],
                    frame_index
                )
            )

            used_old.add(pi)
            used_new.add(ci)

        for comp in components:

            if (
                comp["track_id"]
                is None
            ):

                comp[
                    "track_id"
                ] = next_id

                tracks[
                    next_id
                ] = [

                    (
                        frame["timestamp"],
                        comp["centroid_x"],
                        comp["centroid_y"],
                        comp["area_px"],
                        frame_index
                    )
                ]

                next_id += 1

        previous = components

    return tracks


# ============================================================
# MOVIMIENTO
# ============================================================

def movement_between_points(
    previous,
    current
):

    elapsed_minutes = (
        current[0]
        -
        previous[0]
    ) / 60.0

    if elapsed_minutes <= 0:
        return None

    km_x, km_y = (
        pixels_per_km(
            LAT,
            ZOOM,
            TILE_SIZE
        )
    )

    east_km = (
        current[1]
        -
        previous[1]
    ) * km_x

    south_km = (
        current[2]
        -
        previous[2]
    ) * km_y

    distance_km = math.hypot(
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

    speed = (
        distance_km
        /
        (
            elapsed_minutes
            / 60.0
        )
    )

    return {

        "distance_km":
            distance_km,

        "direction_deg":
            direction,

        "speed_kmh":
            speed,

        "elapsed_minutes":
            elapsed_minutes
    }


# ============================================================
# RESUMEN DE TRAYECTORIA
# ============================================================

def summarize_track(
    track_points
):

    if len(track_points) < 2:

        return {

            "valido":
                False,

            "motivo":
                "Insuficientes puntos de trayectoria."
        }

    points = track_points[
        -TRACK_FRAMES:
    ]

    movements = []

    for i in range(
        1,
        len(points)
    ):

        move = (
            movement_between_points(
                points[i - 1],
                points[i]
            )
        )

        if move is not None:

            movements.append(
                move
            )

    if not movements:

        return {

            "valido":
                False,

            "motivo":
                "No se pudo calcular movimiento."
        }

    speeds = np.array(
        [
            m["speed_kmh"]
            for m in movements
        ],
        dtype=float
    )

    distances = np.array(
        [
            m["distance_km"]
            for m in movements
        ],
        dtype=float
    )

    directions = np.deg2rad(
        np.array(
            [
                m["direction_deg"]
                for m in movements
            ],
            dtype=float
        )
    )

    mean_sin = float(
        np.mean(
            np.sin(
                directions
            )
        )
    )

    mean_cos = float(
        np.mean(
            np.cos(
                directions
            )
        )
    )

    mean_direction = (
        math.degrees(
            math.atan2(
                mean_sin,
                mean_cos
            )
        )
        + 360.0
    ) % 360.0

    resultant = math.sqrt(
        mean_sin ** 2
        +
        mean_cos ** 2
    )

    median_speed = float(
        np.median(
            speeds
        )
    )

    speed_cv = (
        float(
            np.std(
                speeds
            )
        )
        /
        median_speed
        if median_speed > 0
        else 1.0
    )

    speed_consistency = max(
        0.0,
        min(
            1.0,
            1.0 - speed_cv
        )
    )

    confidence = min(
        0.95,
        0.45 * resultant
        +
        0.35 * speed_consistency
        +
        0.20
        *
        min(
            1.0,
            len(movements)
            / 5.0
        )
    )

    first_area = points[0][3]
    last_area = points[-1][3]

    if first_area > 0:

        change = (
            last_area
            / first_area
            - 1.0
        ) * 100.0

    else:

        change = None

    if change is None:

        trend = "sin_datos"

    elif change >= 15:

        trend = "fortaleciendose"

    elif change <= -15:

        trend = "debilitandose"

    else:

        trend = "estable"

    return {

        "valido":
            True,

        "frames_validos":
            len(points),

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
                float(
                    np.median(
                        distances
                    )
                ),
                2
            ),

        "confianza":
            round(
                confidence,
                2
            ),

        "fortalecimiento":
            trend,

        "cambio_area_pct":
            (
                round(
                    change,
                    1
                )
                if change is not None
                else None
            )
    }


# ============================================================
# ¿SE DIRIGE A BAHIA?
# ============================================================

def movement_towards_bahia(
    component,
    track
):

    if (
        not track.get(
            "valido"
        )
        or component is None
    ):

        return False

    width = (
        TILE_SIZE * 3
    )

    height = (
        TILE_SIZE * 3
    )

    cx = component[
        "centroid_x"
    ]

    cy = component[
        "centroid_y"
    ]

    dx = (
        width / 2.0
        - cx
    )

    dy = (
        height / 2.0
        - cy
    )

    if math.hypot(
        dx,
        dy
    ) < 5:

        return True

    target_angle = (
        math.degrees(
            math.atan2(
                dx,
                -dy
            )
        )
        + 360.0
    ) % 360.0

    diff = abs(
        (
            track[
                "direccion_grados"
            ]
            -
            target_angle
            +
            180.0
        ) % 360.0
        - 180.0
    )

    return diff <= 45.0


# ============================================================
# PIXEL -> LAT/LON
# ============================================================

def pixel_to_latlon(
    local_x,
    local_y
):

    xt, yt = (
        latlon_to_tile(
            LAT,
            LON,
            ZOOM
        )
    )

    origin_x = (
        xt - GRID_RADIUS
    ) * TILE_SIZE

    origin_y = (
        yt - GRID_RADIUS
    ) * TILE_SIZE

    world_x = (
        origin_x
        + local_x
    )

    world_y = (
        origin_y
        + local_y
    )

    n = float(
        2 ** ZOOM
    )

    lon = (
        world_x
        / TILE_SIZE
        / n
        * 360.0
        - 180.0
    )

    y = (
        world_y
        / TILE_SIZE
        / n
    )

    lat = math.degrees(
        math.atan(
            math.sinh(
                math.pi
                *
                (
                    1.0
                    - 2.0 * y
                )
            )
        )
    )

    return (
        lat,
        lon
    )


# ============================================================
# PROYECCION
# ============================================================

def projection(
    component,
    track,
    minutes
):

    if (
        not track.get(
            "valido"
        )
        or component is None
    ):

        return None

    speed = (
        track[
            "velocidad_kmh"
        ]
    )

    direction = math.radians(
        track[
            "direccion_grados"
        ]
    )

    travel_km = (
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

    projected_x = (
        component[
            "centroid_x"
        ]
        +
        math.sin(
            direction
        )
        * travel_km
        / km_x
    )

    projected_y = (
        component[
            "centroid_y"
        ]
        -
        math.cos(
            direction
        )
        * travel_km
        / km_y
    )

    width = (
        TILE_SIZE * 3
    )

    height = (
        TILE_SIZE * 3
    )

    inside = (
        0 <= projected_x < width
        and
        0 <= projected_y < height
    )

    distance_to_bahia = (
        distance_point_to_bahia(
            projected_x,
            projected_y
        )
    )

    lat, lon = (
        pixel_to_latlon(
            projected_x,
            projected_y
        )
    )

    return {

        "minutos":
            minutes,

        "distancia_proyectada_km":
            round(
                travel_km,
                1
            ),

        "distancia_a_bahia_km":
            round(
                distance_to_bahia,
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

        "latitud":
            round(
                lat,
                5
            ),

        "longitud":
            round(
                lon,
                5
            ),

        "dentro_area_radar":
            bool(
                inside
            )
    }


# ============================================================
# ETA
# ============================================================

def calculate_eta(
    component,
    track
):

    if (
        not track.get(
            "valido"
        )
        or
        track[
            "velocidad_kmh"
        ] < 5
    ):

        return None

    if not movement_towards_bahia(
        component,
        track
    ):

        return None

    distance = (
        distance_point_to_bahia(
            component[
                "centroid_x"
            ],
            component[
                "centroid_y"
            ]
        )
    )

    eta = (
        distance
        /
        track[
            "velocidad_kmh"
        ]
        * 60.0
    )

    if (
        0 <= eta <= 360
    ):

        return round(
            eta,
            1
        )

    return None


# ============================================================
# ANALISIS COMPLETO
# ============================================================

def analyze_sequence(
    results
):

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
            current[
                "timestamp"
            ],

        "frame_actual_utc":
            current[
                "frame_utc"
            ],

        "tiles_ok":
            current[
                "tiles_ok"
            ],

        "actividad":
            current[
                "area"
            ] > 0,

        "area_px":
            current[
                "area"
            ],

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

        # Compatibilidad con app actual.
        "proyeccion_30_min":
            None,

        "proyeccion_60_min":
            None,

        "confianza_movimiento":
            0.0,

        "nucleos":
            [],

        "nucleo_principal_id":
            None,

        "estado":
            (
                "precipitacion_detectada"
                if current[
                    "area"
                ] > 0
                else
                "sin_precipitacion_detectada"
            )
    }

    if (
        len(results) < 2
        or
        not current[
            "components"
        ]
    ):

        return status

    tracks = (
        track_components(
            results
        )
    )

    current_components = (
        current[
            "components"
        ]
    )

    nuclei = []

    for comp in current_components:

        track_id = comp[
            "track_id"
        ]

        points = tracks.get(
            track_id,
            []
        )

        summary = (
            summarize_track(
                points
            )
        )

        toward = (
            movement_towards_bahia(
                comp,
                summary
            )
        )

        eta = (
            calculate_eta(
                comp,
                summary
            )
        )

        projections = {}

        if summary.get(
            "valido"
        ):

            for minutes in (
                PROJECTION_MINUTES
            ):

                projections[
                    str(minutes)
                ] = projection(
                    comp,
                    summary,
                    minutes
                )

        nuclei.append({

            "id":
                track_id,

            "area_px":
                comp[
                    "area_px"
                ],

            "distancia_km":
                round(
                    comp[
                        "distance_km"
                    ],
                    1
                ),

            "velocidad_kmh":
                summary.get(
                    "velocidad_kmh"
                ),

            "direccion":
                summary.get(
                    "direccion"
                ),

            "direccion_grados":
                summary.get(
                    "direccion_grados"
                ),

            "movimiento_hacia_bahia":
                toward,

            "eta_minutos":
                eta,

            "fortalecimiento":
                summary.get(
                    "fortalecimiento",
                    "sin_datos"
                ),

            "cambio_area_pct":
                summary.get(
                    "cambio_area_pct"
                ),

            "confianza_movimiento":
                summary.get(
                    "confianza",
                    0.0
                ),

            "frames_track":
                summary.get(
                    "frames_validos",
                    1
                ),

            "proyecciones":
                projections,

            "bbox":
                comp[
                    "bbox"
                ]
        })

    nuclei.sort(
        key=lambda n: (
            not n[
                "movimiento_hacia_bahia"
            ],
            n[
                "distancia_km"
            ],
            -n[
                "area_px"
            ]
        )
    )

    if not nuclei:

        return status

    principal = nuclei[0]

    status[
        "nucleos"
    ] = nuclei

    status[
        "nucleo_principal_id"
    ] = principal[
        "id"
    ]

    status[
        "area_px"
    ] = principal[
        "area_px"
    ]

    status[
        "distancia_km"
    ] = principal[
        "distancia_km"
    ]

    status[
        "velocidad_kmh"
    ] = principal[
        "velocidad_kmh"
    ]

    status[
        "direccion"
    ] = principal[
        "direccion"
    ]

    status[
        "direccion_grados"
    ] = principal[
        "direccion_grados"
    ]

    status[
        "movimiento_hacia_bahia"
    ] = principal[
        "movimiento_hacia_bahia"
    ]

    status[
        "eta_minutos"
    ] = principal[
        "eta_minutos"
    ]

    status[
        "fortalecimiento"
    ] = principal[
        "fortalecimiento"
    ]

    status[
        "cambio_area_pct"
    ] = principal[
        "cambio_area_pct"
    ]

    status[
        "confianza_movimiento"
    ] = principal[
        "confianza_movimiento"
    ]

    status[
        "proyecciones"
    ] = principal[
        "proyecciones"
    ]

    status[
        "proyeccion_30_min"
    ] = principal[
        "proyecciones"
    ].get(
        "30"
    )

    status[
        "proyeccion_60_min"
    ] = principal[
        "proyecciones"
    ].get(
        "60"
    )

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
        "nucleos"
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

            writer.writerow({

                "frame_time":
                    item[
                        "timestamp"
                    ],

                "frame_utc":
                    item[
                        "frame_utc"
                    ],

                "source":
                    "RainViewer",

                "latitud":
                    LAT,

                "longitud":
                    LON,

                "area_px":
                    item[
                        "area"
                    ],

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
                    item[
                        "tiles_ok"
                    ],

                "nucleos":
                    json.dumps(
                        item[
                            "components"
                        ],
                        ensure_ascii=False
                    )
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
            and
            name.endswith(
                ".png"
            )
        ):

            try:

                files.append(
                    (
                        int(
                            name[
                                6:-4
                            ]
                        ),
                        os.path.join(
                            HISTORY_DIR,
                            name
                        )
                    )
                )

            except ValueError:

                pass

    files.sort()

    while (
        len(files)
        > MAX_HISTORY_FRAMES
    ):

        _, old_path = (
            files.pop(0)
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

    print(
        "=" * 70
    )

    print(
        "CLIMAAR - "
        "RADAR RAINVIEWER NOWCAST 7.0"
    )

    print(
        "=" * 70
    )

    host, frames = (
        fetch_rainviewer()
    )

    xt, yt = (
        latlon_to_tile(
            LAT,
            LON,
            ZOOM
        )
    )

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
            f"{index}/"
            f"{len(selected_frames)}"
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

        results.append(
            analyze_frame(
                image,
                timestamp,
                tiles_ok,
                path
            )
        )

        latest_image = image

    if not results:

        print(
            "ERROR: no se pudo "
            "procesar ningun frame."
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
            current[
                "timestamp"
            ],
            latest_image
        )

    save_features(
        results
    )

    nowcast = (
        analyze_sequence(
            results
        )
    )

    output = {

        "version":
            "7.0",

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
            "El nowcast sigue multiples "
            "nucleos de precipitacion y "
            "extrapola su movimiento observado. "
            "Las proyecciones no garantizan "
            "la trayectoria futura y deben "
            "interpretarse con su confianza."
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

    print(
        json.dumps(
            nowcast,
            ensure_ascii=False,
            indent=2
        )
    )

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
