import os
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
# CLIMAAR - RADAR RAINVIEWER V7.2
# dBZ REAL + SEGUIMIENTO DE NÚCLEOS + NOWCAST
# ============================================================

LAT = -38.71
LON = -62.26

ZOOM = 7
TILE_SIZE = 512
GRID_RADIUS = 1

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
    "User-Agent": "ClimaAR/7.2",
    "Referer": "https://www.rainviewer.com/",
    "Accept": "image/png,image/*;q=0.8,*/*;q=0.5",
}

MAX_HISTORY_FRAMES = 144
TRACK_FRAMES = 6

MIN_COMPONENT_AREA = 20
MAX_COMPONENTS = 12

MAX_TRACK_SPEED_KMH = 180.0

PROJECTION_MINUTES = (
    15,
    30,
    45,
    60,
    90
)


# ============================================================
# TABLA UNIVERSAL BLUE DE RAINVIEWER
#
# Color scheme 2
# dBZ aproximado desde la tabla oficial de RainViewer.
#
# IMPORTANTE:
# RainViewer utiliza colores discretos. Por eso algunos
# valores altos comparten color y no pueden distinguirse
# individualmente solo desde el PNG.
# ============================================================

DBZ_HEX = [
    "cec08796",
    "d2c48ba0",
    "d6c88faa",
    "dacc93b4",
    "ded097be",

    "88ddeeff",
    "6cd1ebff",
    "51c5e8ff",
    "36bae5ff",
    "1baee2ff",

    "00a3e0ff",
    "009ad5ff",
    "0091caff",
    "0088bfff",
    "007fb4ff",

    "0077aaff",
    "0070a3ff",
    "00699cff",
    "006295ff",
    "005b8eff",

    "005588ff",
    "005180ff",
    "004e78ff",
    "004a70ff",
    "004768ff",

    "ffee00ff",
    "ffe000ff",
    "ffd200ff",
    "ffc500ff",
    "ffb700ff",

    "ffaa00ff",
    "ff9f00ff",
    "ff9500ff",
    "ff8b00ff",
    "ff8100ff",

    "ff4400ff",
    "f23600ff",
    "e62800ff",
    "d91b00ff",
    "cd0d00ff",

    "c10000ff",
    "a80000ff",
    "8f0000ff",
    "760000ff",
    "5d0000ff",

    "ffaaffff",
    "ff9fffff",
    "ff95ffff",
    "ff8bffff",
    "ff81ffff",

    "ff77ffff",
    "ff6cffff",
    "ff62ffff",
    "ff58ffff",
    "ff4effff",

    "ffffffff",
    "ffffffff",
    "ffffffff",
    "ffffffff",
    "ffffffff",

    "ffffffff",
    "ffffffff",
    "ffffffff",
    "ffffffff",
    "ffffffff",

    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
    "00ff00ff",

    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
    "00ff00ff",

    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
    "00ff00ff",

    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
    "00ff00ff",

    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
]


DBZ_VALUES = np.arange(
    10,
    96,
    dtype=np.float32
)


DBZ_RGB = np.array(
    [
        [
            int(h[i:i + 2], 16)
            for i in (0, 2, 4)
        ]
        for h in DBZ_HEX
    ],
    dtype=np.float32
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
# CONVERSIÓN LAT/LON → TILE
# ============================================================

def latlon_to_tile(
    lat,
    lon,
    zoom
):

    n = 2.0 ** zoom

    x = (
        (lon + 180.0)
        /
        360.0
        *
        n
    )

    lat_rad = math.radians(lat)

    y = (
        (
            1.0
            -
            math.asinh(
                math.tan(lat_rad)
            )
            /
            math.pi
        )
        /
        2.0
        *
        n
    )

    return (
        int(x),
        int(y)
    )


# ============================================================
# ESCALA PIXEL → KM
# ============================================================

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


# ============================================================
# FECHA DEL FRAME
# ============================================================

def frame_datetime(
    ts
):

    return datetime.fromtimestamp(
        int(ts),
        timezone.utc
    ).isoformat()


# ============================================================
# DIRECCIÓN
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

    return names[
        int(
            (
                degrees
                +
                22.5
            )
            /
            45
        )
        %
        8
    ]


# ============================================================
# OBTENER FRAMES DE RAINVIEWER
# ============================================================

def fetch_rainviewer():

    response = requests.get(
        API_URL,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    host = data.get(
        "host",
        "https://tilecache.rainviewer.com"
    ).rstrip("/")

    frames = (
        data
        .get("radar", {})
        .get("past", [])
    )

    if not frames:

        raise RuntimeError(
            "RainViewer no devolvió frames."
        )

    return (
        host,
        frames
    )


# ============================================================
# DESCARGAR FRAME
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

    image = Image.new(
        "RGBA",
        (
            TILE_SIZE * 3,
            TILE_SIZE * 3
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

            # =================================================
            # RainViewer:
            #
            # 2 = Universal Blue
            # 0 = sin suavizado
            # 0 = configuración nieve
            #
            # El objetivo es conservar los colores originales
            # para poder traducirlos posteriormente a dBZ.
            # =================================================

            url = (
                f"{host}"
                f"{path}/"
                f"{TILE_SIZE}/"
                f"{ZOOM}/"
                f"{xt + dx}/"
                f"{yt + dy}/"
                f"2/0_0.png"
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
                    len(response.content) < 200
                ):
                    continue

                tile = (
                    Image
                    .open(
                        BytesIO(
                            response.content
                        )
                    )
                    .convert(
                        "RGBA"
                    )
                )

                image.alpha_composite(
                    tile,
                    (
                        (
                            dx
                            +
                            GRID_RADIUS
                        )
                        *
                        TILE_SIZE,

                        (
                            dy
                            +
                            GRID_RADIUS
                        )
                        *
                        TILE_SIZE
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
# RGB → dBZ
# ============================================================

def rgba_to_dbz(
    image
):

    array = np.asarray(
        image
    ).astype(
        np.float32
    )

    rgb = array[:, :, :3]

    alpha = array[:, :, 3]

    flat = rgb.reshape(
        -1,
        3
    )

    # Distancia euclídea entre cada pixel y todos los
    # colores conocidos de la tabla.

    distances = (
        (
            flat[:, None, :]
            -
            DBZ_RGB[None, :, :]
        )
        ** 2
    ).sum(
        axis=2
    )

    indices = np.argmin(
        distances,
        axis=1
    )

    nearest_distance = np.sqrt(
        np.min(
            distances,
            axis=1
        )
    ).reshape(
        rgb.shape[:2]
    )

    dbz = (
        DBZ_VALUES[
            indices
        ]
        .reshape(
            rgb.shape[:2]
        )
        .astype(
            np.float32
        )
    )

    # Solo consideramos válidos los píxeles:
    #
    # 1. con alpha
    # 2. suficientemente cercanos a un color oficial

    valid = (
        (alpha > 0)
        &
        (
            nearest_distance
            <=
            8.0
        )
    )

    dbz[
        ~valid
    ] = np.nan

    return dbz


# ============================================================
# MÁSCARA DE PRECIPITACIÓN
# ============================================================

def dbz_mask(
    dbz,
    threshold=10.0
):

    return np.where(
        np.isfinite(dbz)
        &
        (
            dbz
            >=
            threshold
        ),
        255,
        0
    ).astype(
        np.uint8
    )


# ============================================================
# MÉTRICAS dBZ
# ============================================================

def dbz_metrics(
    dbz
):

    valid = dbz[
        np.isfinite(dbz)
    ]

    precip = valid[
        valid >= 10
    ]

    if precip.size == 0:

        return {
            "dbz_max": None,
            "dbz_mean": None,
            "dbz_p90": None,
            "dbz_pixels": 0
        }

    return {

        "dbz_max":
            float(
                np.max(
                    precip
                )
            ),

        "dbz_mean":
            round(
                float(
                    np.mean(
                        precip
                    )
                ),
                1
            ),

        "dbz_p90":
            float(
                np.percentile(
                    precip,
                    90
                )
            ),

        "dbz_pixels":
            int(
                precip.size
            )
    }


# ============================================================
# CLASIFICACIÓN DE INTENSIDAD
# ============================================================

def intensity_label(
    dbz
):

    if dbz is None:
        return "sin_precipitacion"

    if dbz < 20:
        return "debil"

    if dbz < 30:
        return "moderada"

    if dbz < 40:
        return "fuerte"

    if dbz < 50:
        return "muy_fuerte"

    if dbz < 60:
        return "severa"

    if dbz < 65:
        return "muy_severa"

    return "extrema"


# ============================================================
# POSICIÓN EXACTA DE BAHÍA BLANCA DENTRO DEL MOSAICO
# ============================================================

def bahia_pixel_position():

    n = float(
        2 ** ZOOM
    )

    lat_rad = math.radians(
        LAT
    )

    world_x = (
        (LON + 180.0)
        / 360.0
        * n
        * TILE_SIZE
    )

    world_y = (
        (
            1.0
            -
            math.asinh(
                math.tan(
                    lat_rad
                )
            )
            / math.pi
        )
        / 2.0
        * n
        * TILE_SIZE
    )

    tile_x = math.floor(
        world_x / TILE_SIZE
    )

    tile_y = math.floor(
        world_y / TILE_SIZE
    )

    local_x = (
        world_x
        -
        tile_x * TILE_SIZE
    )

    local_y = (
        world_y
        -
        tile_y * TILE_SIZE
    )

    mosaic_x = (
        GRID_RADIUS * TILE_SIZE
        + local_x
    )

    mosaic_y = (
        GRID_RADIUS * TILE_SIZE
        + local_y
    )

    return (
        float(mosaic_x),
        float(mosaic_y)
    )


# ============================================================
# DISTANCIA DE UN PUNTO A BAHÍA BLANCA
# ============================================================

def distance_point_to_bahia(
    cx,
    cy
):

    bahia_x, bahia_y = (
        bahia_pixel_position()
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
            float(cx)
            -
            bahia_x
        )
        *
        km_x,

        (
            float(cy)
            -
            bahia_y
        )
        *
        km_y
    )


# ============================================================
# DISTANCIA MÍNIMA DEL ECO A BAHÍA
# ============================================================

def distance_to_bahia(
    mask
):

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:

        return None

    bahia_x, bahia_y = (
        bahia_pixel_position()
    )

    km_x, km_y = (
        pixels_per_km(
            LAT,
            ZOOM,
            TILE_SIZE
        )
    )

    distances = np.sqrt(
        (
            (
                xs
                -
                bahia_x
            )
            *
            km_x
        )
        ** 2
        +
        (
            (
                ys
                -
                bahia_y
            )
            *
            km_y
        )
        ** 2
    )

    return float(
        np.min(
            distances
        )
    )


# ============================================================
# DETECCIÓN DE NÚCLEOS
# ============================================================

def components_from_mask(
    mask,
    dbz
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

        ys, xs = np.where(
            labels == i
        )

        values = dbz[
            ys,
            xs
        ]

        values = values[
            np.isfinite(values)
        ]

        max_dbz = (
            float(
                np.max(
                    values
                )
            )
            if values.size
            else None
        )

        mean_dbz = (
            float(
                np.mean(
                    values
                )
            )
            if values.size
            else None
        )

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

            "max_dbz":
                max_dbz,

            "mean_dbz":
                mean_dbz,

            "intensidad":
                intensity_label(
                    max_dbz
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
        key=lambda c: (
            c["max_dbz"]
            if c["max_dbz"]
            is not None
            else -999,

            c["area_px"]
        ),
        reverse=True
    )

    return components[
        :MAX_COMPONENTS
    ]


# ============================================================
# ANALIZAR FRAME
# ============================================================

def analyze_frame(
    image,
    timestamp,
    tiles_ok,
    path
):

    dbz = rgba_to_dbz(
        image
    )

    mask = dbz_mask(
        dbz
    )

    components = (
        components_from_mask(
            mask,
            dbz
        )
    )

    metrics = dbz_metrics(
        dbz
    )

    for component in components:

        component[
            "distance_km"
        ] = round(
            distance_point_to_bahia(
                component[
                    "centroid_x"
                ],
                component[
                    "centroid_y"
                ]
            ),
            2
        )

    distance = (
        distance_to_bahia(
            mask
        )
    )

    return {

        "timestamp":
            int(
                timestamp
            ),

        "frame_utc":
            frame_datetime(
                timestamp
            ),

        "tiles_ok":
            int(
                tiles_ok
            ),

        "path":
            path,

        "area":
            int(
                np.count_nonzero(
                    mask
                )
            ),

        "components":
            components,

        "distance_km":
            (
                round(
                    distance,
                    2
                )
                if distance is not None
                else None
            ),

        **metrics
    }


# ============================================================
# MOVIMIENTO ENTRE DOS POSICIONES
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
        +
        360.0
    ) % 360.0

    speed = (
        distance_km
        /
        (
            elapsed_minutes
            /
            60.0
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
# COSTO PARA ASOCIAR NÚCLEOS ENTRE FRAMES
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

    distance_km = math.hypot(

        (
            new["centroid_x"]
            -
            old["centroid_x"]
        )
        *
        km_x,

        (
            new["centroid_y"]
            -
            old["centroid_y"]
        )
        *
        km_y
    )

    max_distance = max(

        5.0,

        MAX_TRACK_SPEED_KMH
        *
        max(
            elapsed_minutes,
            1.0
        )
        /
        60.0
    )

    if distance_km > max_distance:

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
        /
        max_distance
        +
        0.15
        *
        area_penalty
    )


# ============================================================
# SEGUIMIENTO DE NÚCLEOS
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

        for component in components:

            component[
                "track_id"
            ] = None

        if frame_index == 0:

            for component in components:

                component[
                    "track_id"
                ] = next_id

                tracks[
                    next_id
                ] = [

                    (
                        frame[
                            "timestamp"
                        ],

                        component[
                            "centroid_x"
                        ],

                        component[
                            "centroid_y"
                        ],

                        component[
                            "area_px"
                        ],

                        frame_index
                    )
                ]

                next_id += 1

            previous = components

            continue

        elapsed = (
            frame[
                "timestamp"
            ]
            -
            results[
                frame_index - 1
            ][
                "timestamp"
            ]
        ) / 60.0

        candidates = []

        for previous_index, old in enumerate(
            previous
        ):

            for component_index, new in enumerate(
                components
            ):

                cost = component_cost(
                    old,
                    new,
                    elapsed
                )

                if cost is not None:

                    candidates.append(
                        (
                            cost,
                            previous_index,
                            component_index
                        )
                    )

        used_old = set()

        used_new = set()

        for (
            _,
            previous_index,
            component_index
        ) in sorted(
            candidates
        ):

            if (
                previous_index
                in
                used_old
                or
                component_index
                in
                used_new
            ):

                continue

            old = previous[
                previous_index
            ]

            new = components[
                component_index
            ]

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
                    frame[
                        "timestamp"
                    ],

                    new[
                        "centroid_x"
                    ],

                    new[
                        "centroid_y"
                    ],

                    new[
                        "area_px"
                    ],

                    frame_index
                )
            )

            used_old.add(
                previous_index
            )

            used_new.add(
                component_index
            )

        for component in components:

            if (
                component[
                    "track_id"
                ]
                is None
            ):

                component[
                    "track_id"
                ] = next_id

                tracks[
                    next_id
                ] = [

                    (
                        frame[
                            "timestamp"
                        ],

                        component[
                            "centroid_x"
                        ],

                        component[
                            "centroid_y"
                        ],

                        component[
                            "area_px"
                        ],

                        frame_index
                    )
                ]

                next_id += 1

        previous = components

    return tracks


# ============================================================
# RESUMEN DE TRAYECTORIA
# ============================================================

def summarize_track(
    track_points
):

    if len(
        track_points
    ) < 2:

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

        movement = (
            movement_between_points(
                points[
                    i - 1
                ],
                points[
                    i
                ]
            )
        )

        if movement is not None:

            movements.append(
                movement
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
            m[
                "speed_kmh"
            ]
            for m in movements
        ],
        dtype=float
    )

    distances = np.array(
        [
            m[
                "distance_km"
            ]
            for m in movements
        ],
        dtype=float
    )

    directions = np.deg2rad(
        np.array(
            [
                m[
                    "direction_deg"
                ]
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
        +
        360.0
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
            1.0
            -
            speed_cv
        )
    )

    confidence = min(
        0.95,

        0.45
        *
        resultant

        +

        0.35
        *
        speed_consistency

        +

        0.20
        *
        min(
            1.0,
            len(
                movements
            )
            /
            5.0
        )
    )

    first_area = points[
        0
    ][
        3
    ]

    last_area = points[
        -1
    ][
        3
    ]

    change = (

        (
            last_area
            /
            first_area
            -
            1.0
        )
        *
        100.0

        if first_area > 0

        else None
    )

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
# ¿EL NÚCLEO SE MUEVE HACIA BAHÍA?
# ============================================================

def movement_towards_bahia(
    component,
    track
):

    if (
        not track.get(
            "valido"
        )
        or
        component is None
    ):

        return False

    bahia_x, bahia_y = (
        bahia_pixel_position()
    )

    dx = (
        bahia_x
        -
        component[
            "centroid_x"
        ]
    )

    dy = (
        bahia_y
        -
        component[
            "centroid_y"
        ]
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
        +
        360.0
    ) % 360.0

    difference = abs(

        (
            track[
                "direccion_grados"
            ]
            -
            target_angle
            +
            180.0
        )
        % 360.0
        -
        180.0
    )

    return (
        difference
        <=
        45.0
    )


# ============================================================
# PIXEL → LAT/LON
# ============================================================

def pixel_to_latlon(
    local_x,
    local_y
):

    xt, yt = latlon_to_tile(
        LAT,
        LON,
        ZOOM
    )

    world_x = (
        xt
        -
        GRID_RADIUS
    ) * TILE_SIZE + local_x

    world_y = (
        yt
        -
        GRID_RADIUS
    ) * TILE_SIZE + local_y

    n = float(
        2 ** ZOOM
    )

    lon = (
        world_x
        /
        TILE_SIZE
        /
        n
        *
        360.0
        -
        180.0
    )

    y = (
        world_y
        /
        TILE_SIZE
        /
        n
    )

    lat = math.degrees(
        math.atan(
            math.sinh(
                math.pi
                *
                (
                    1.0
                    -
                    2.0 * y
                )
            )
        )
    )

    return (
        lat,
        lon
    )


# ============================================================
# PROYECCIÓN DEL NÚCLEO
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
        or
        component is None
    ):

        return None

    speed = track[
        "velocidad_kmh"
    ]

    direction = math.radians(
        track[
            "direccion_grados"
        ]
    )

    travel_km = (
        speed
        *
        minutes
        /
        60.0
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
        *
        travel_km
        /
        km_x
    )

    projected_y = (
        component[
            "centroid_y"
        ]
        -
        math.cos(
            direction
        )
        *
        travel_km
        /
        km_y
    )

    inside = (

        0
        <=
        projected_x
        <
        TILE_SIZE * 3

        and

        0
        <=
        projected_y
        <
        TILE_SIZE * 3
    )

    distance = (
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
                distance,
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
        *
        60.0
    )

    return (

        round(
            eta,
            1
        )

        if
        0
        <=
        eta
        <=
        360

        else
        None
    )


# ============================================================
# ANÁLISIS DE LA SECUENCIA
# ============================================================

def analyze_sequence(
    results
):

    current = results[
        -1
    ]

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
            current[
                "distance_km"
            ],

        "dbz_max":
            current[
                "dbz_max"
            ],

        "dbz_mean":
            current[
                "dbz_mean"
            ],

        "dbz_p90":
            current[
                "dbz_p90"
            ],

        "dbz_pixels":
            current[
                "dbz_pixels"
            ],

        "intensidad":
            intensity_label(
                current[
                    "dbz_max"
                ]
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
                if
                current[
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

    nuclei = []

    for component in current[
        "components"
    ]:

        track_id = component[
            "track_id"
        ]

        summary = (
            summarize_track(
                tracks.get(
                    track_id,
                    []
                )
            )
        )

        toward = (
            movement_towards_bahia(
                component,
                summary
            )
        )

        eta = (
            calculate_eta(
                component,
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
                    component,
                    summary,
                    minutes
                )

        nuclei.append({

            "id":
                track_id,

            "area_px":
                component[
                    "area_px"
                ],

            "max_dbz":
                component[
                    "max_dbz"
                ],

            "mean_dbz":
                (
                    round(
                        component[
                            "mean_dbz"
                        ],
                        1
                    )

                    if

                    component[
                        "mean_dbz"
                    ] is not None

                    else
                    None
                ),

            "intensidad":
                component[
                    "intensidad"
                ],

            "distancia_km":
                round(
                    component[
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
                component[
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

            -(
                n[
                    "max_dbz"
                ]

                if

                n[
                    "max_dbz"
                ] is not None

                else
                -999
            )
        )
    )

    if not nuclei:

        return status

    principal = nuclei[
        0
    ]

    status.update({

        "nucleos":
            nuclei,

        "nucleo_principal_id":
            principal[
                "id"
            ],

        "area_px":
            principal[
                "area_px"
            ],

        "distancia_km":
            principal[
                "distancia_km"
            ],

        "dbz_max":
            principal[
                "max_dbz"
            ],

        "dbz_mean":
            principal[
                "mean_dbz"
            ],

        "intensidad":
            principal[
                "intensidad"
            ],

        "velocidad_kmh":
            principal[
                "velocidad_kmh"
            ],

        "direccion":
            principal[
                "direccion"
            ],

        "direccion_grados":
            principal[
                "direccion_grados"
            ],

        "movimiento_hacia_bahia":
            principal[
                "movimiento_hacia_bahia"
            ],

        "eta_minutos":
            principal[
                "eta_minutos"
            ],

        "fortalecimiento":
            principal[
                "fortalecimiento"
            ],

        "cambio_area_pct":
            principal[
                "cambio_area_pct"
            ],

        "confianza_movimiento":
            principal[
                "confianza_movimiento"
            ],

        "proyecciones":
            principal[
                "proyecciones"
            ],

        "proyeccion_30_min":
            principal[
                "proyecciones"
            ].get(
                "30"
            ),

        "proyeccion_60_min":
            principal[
                "proyecciones"
            ].get(
                "60"
            )
    })

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
        "dbz_max",
        "dbz_mean",
        "dbz_p90",
        "dbz_pixels",
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
                    item[
                        "distance_km"
                    ],

                "dbz_max":
                    item[
                        "dbz_max"
                    ],

                "dbz_mean":
                    item[
                        "dbz_mean"
                    ],

                "dbz_p90":
                    item[
                        "dbz_p90"
                    ],

                "dbz_pixels":
                    item[
                        "dbz_pixels"
                    ],

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
# GUARDAR HISTÓRICO
# ============================================================

def save_history(
    timestamp,
    image
):

    image.save(
        os.path.join(
            HISTORY_DIR,
            f"radar_{timestamp}.png"
        )
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

    while len(
        files
    ) > MAX_HISTORY_FRAMES:

        _, old_path = (
            files.pop(
                0
            )
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
        "RADAR RAINVIEWER NOWCAST 7.2 + dBZ"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # RainViewer
    # --------------------------------------------------------

    host, frames = (
        fetch_rainviewer()
    )

    # --------------------------------------------------------
    # Tile central de Bahía Blanca
    # --------------------------------------------------------

    xt, yt = (
        latlon_to_tile(
            LAT,
            LON,
            ZOOM
        )
    )

    # --------------------------------------------------------
    # Últimos 12 frames disponibles
    # --------------------------------------------------------

    selected_frames = frames[
        -12:
    ]

    results = []

    latest_image = None

    # --------------------------------------------------------
    # Procesamiento
    # --------------------------------------------------------

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
                "Frame sin tiles válidas."
            )

            continue

        item = analyze_frame(
            image,
            timestamp,
            tiles_ok,
            path
        )

        results.append(
            item
        )

        latest_image = image

        print(
            f"  dBZ max={item['dbz_max']} | "
            f"media={item['dbz_mean']} | "
            f"p90={item['dbz_p90']} | "
            f"pix={item['dbz_pixels']}"
        )

    # --------------------------------------------------------
    # Validación
    # --------------------------------------------------------

    if not results:

        raise RuntimeError(
            "No se pudo procesar ningún frame."
        )

    results.sort(
        key=lambda x:
        x["timestamp"]
    )

    current = results[
        -1
    ]

    # --------------------------------------------------------
    # Imagen actual
    # --------------------------------------------------------

    latest_image.save(
        ACTUAL_FILE
    )

    # --------------------------------------------------------
    # Histórico
    # --------------------------------------------------------

    save_history(
        current[
            "timestamp"
        ],
        latest_image
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    save_features(
        results
    )

    # --------------------------------------------------------
    # NOWCAST
    # --------------------------------------------------------

    nowcast = (
        analyze_sequence(
            results
        )
    )

    # --------------------------------------------------------
    # JSON FINAL
    # --------------------------------------------------------

    output = {

        "version":
            "7.2",

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

        "rainviewer_color_scheme":
            "Universal Blue (2)",

        "rainviewer_smooth":
            0,

        "rainviewer_snow":
            0,

        "dbz_decode":
            "tabla oficial Universal Blue",

        "bahia_pixel_mosaico": {
            "x": round(bahia_pixel_position()[0], 2),
            "y": round(bahia_pixel_position()[1], 2)
        },

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
            "dBZ decodificado desde la tabla oficial "
            "de colores de RainViewer. Los colores "
            "repetidos de la paleta no permiten "
            "distinguir todos los valores individuales; "
            "65-74 y 75-95 comparten colores."
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

    # --------------------------------------------------------
    # Mostrar resultado en Actions
    # --------------------------------------------------------

    print(
        ""
    )

    print(
        "=" * 70
    )

    print(
        "RESULTADO NOWCAST"
    )

    print(
        "=" * 70
    )

    print(
        json.dumps(
            nowcast,
            ensure_ascii=False,
            indent=2
        )
    )

    print(
        "=" * 70
    )

    print(
        "CLIMAAR RADAR V7.1 FINALIZADO"
    )

    print(
        "=" * 70
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    main()
