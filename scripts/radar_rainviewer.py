import csv
import json
import math
import os
from datetime import datetime, timezone

import cv2
import numpy as np
import requests


# ============================================================
# CLIMAAR - RADAR RAINVIEWER
# v10.0 - SOURCE HEALTH + WEAK ECHO DETECTION
# ============================================================

LAT = -38.71
LON = -62.26

ZOOM = 7
SIZE = 512

DETECTION_RADIUS_KM = 150.0
PRIORITY_RADIUS_KM = 100.0

FRAMES = 6
MAX_HISTORY = 144

RADAR_DIR = "data/radar"
HISTORY_DIR = os.path.join(RADAR_DIR, "historico")

ACTUAL_PATH = os.path.join(RADAR_DIR, "actual.png")
NOWCAST_PATH = os.path.join(RADAR_DIR, "radar_nowcast.json")
STATUS_PATH = os.path.join(RADAR_DIR, "status.json")
FEATURES_PATH = os.path.join(
    RADAR_DIR,
    "radar_features_rainviewer.csv"
)

LAST_VALID_IMAGE_PATH = os.path.join(
    RADAR_DIR,
    "last_valid_radar.png"
)

LAST_VALID_NOWCAST_PATH = os.path.join(
    RADAR_DIR,
    "last_valid_nowcast.json"
)

RAINVIEWER_API = (
    "https://api.rainviewer.com/public/weather-maps.json"
)


# ============================================================
# CSV
# ============================================================

CSV_FIELDS = [
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
    "nucleos",
    "centroide_x",
    "centroide_y",
    "pixeles_intensos",
    "cobertura_precipitacion",
    "movimiento_x",
    "movimiento_y",
    "direccion",
    "velocidad_pixeles_frame",
]


SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": "ClimaAR/1.0"
    }
)


# ============================================================
# UTILIDADES
# ============================================================

def ensure_dirs():

    os.makedirs(
        RADAR_DIR,
        exist_ok=True
    )

    os.makedirs(
        HISTORY_DIR,
        exist_ok=True
    )


def iso_from_timestamp(ts):

    return datetime.fromtimestamp(
        int(ts),
        tz=timezone.utc
    ).isoformat()


def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


def atomic_json_write(
    path,
    data
):

    temporary = path + ".tmp"

    with open(
        temporary,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    os.replace(
        temporary,
        path
    )


def atomic_copy_file(
    source,
    destination
):

    temporary = destination + ".tmp"

    with open(
        source,
        "rb"
    ) as src:

        with open(
            temporary,
            "wb"
        ) as dst:

            dst.write(
                src.read()
            )

    os.replace(
        temporary,
        destination
    )


# ============================================================
# RAINVIEWER API
# ============================================================

def download_json(url):

    response = SESSION.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def get_radar_frames():

    data = download_json(
        RAINVIEWER_API
    )

    host = data.get(
        "host",
        ""
    ).rstrip("/")

    radar = data.get(
        "radar",
        {}
    )

    past = radar.get(
        "past",
        []
    )

    if not host:

        raise RuntimeError(
            "RainViewer no devolvio host"
        )

    if not past:

        raise RuntimeError(
            "RainViewer no devolvio frames"
        )

    frames = []

    for item in past:

        if not item.get("time"):
            continue

        if not item.get("path"):
            continue

        frames.append(
            {
                "time": int(
                    item["time"]
                ),
                "path": item["path"]
            }
        )

    frames.sort(
        key=lambda item:
        item["time"]
    )

    if not frames:

        raise RuntimeError(
            "No existen frames validos"
        )

    return host, frames


def build_image_url(
    host,
    path
):

    return (
        f"{host}{path}/"
        f"{SIZE}/{ZOOM}/"
        f"{LAT}/{LON}/"
        f"2/0_0.png"
    )


def build_coverage_url(host):

    return (
        f"{host}/v2/coverage/0/"
        f"{SIZE}/{ZOOM}/"
        f"{LAT}/{LON}/"
        f"0/0_0.png"
    )


# ============================================================
# DESCARGA Y DIAGNOSTICO
# ============================================================

def download_image(url):

    response = SESSION.get(
        url,
        timeout=40
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "content-type",
        ""
    ).lower()

    if len(response.content) < 512:

        raise RuntimeError(
            "Respuesta de radar demasiado pequena"
        )

    if (
        "image" not in content_type
        and not response.content.startswith(
            b"\x89PNG"
        )
    ):

        raise RuntimeError(
            "RainViewer no devolvio PNG"
        )

    data = np.frombuffer(
        response.content,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        data,
        cv2.IMREAD_UNCHANGED
    )

    if image is None:

        raise RuntimeError(
            "OpenCV no pudo decodificar imagen"
        )

    if image.ndim not in (
        2,
        3
    ):

        raise RuntimeError(
            "Formato de imagen inesperado"
        )

    height, width = image.shape[:2]

    if width < 64 or height < 64:

        raise RuntimeError(
            "Imagen demasiado pequena"
        )

    if (
        image.ndim == 3
        and image.shape[2] == 4
    ):

        alpha = image[:, :, 3]

        rgb_max = int(
            max(
                image[:, :, 0].max(),
                image[:, :, 1].max(),
                image[:, :, 2].max()
            )
        )

        alpha_max = int(
            alpha.max()
        )

        visible_pixels = int(
            np.count_nonzero(
                alpha >= 20
            )
        )

        fully_transparent = (
            alpha_max == 0
        )

        print(
            "[RADAR] "
            f"imagen={width}x{height} "
            f"rgb_max={rgb_max} "
            f"alpha_max={alpha_max} "
            f"pixeles_visibles={visible_pixels} "
            f"transparente={fully_transparent}"
        )

    else:

        max_value = int(
            image.max()
        )

        nonzero = int(
            np.count_nonzero(
                image
            )
        )

        fully_transparent = False

        print(
            "[RADAR] "
            f"imagen={width}x{height} "
            f"max={max_value} "
            f"pixeles_no_negros={nonzero}"
        )

    return (
        image,
        fully_transparent
    )


def download_coverage_status(host):

    url = build_coverage_url(
        host
    )

    try:

        response = SESSION.get(
            url,
            timeout=30
        )

        response.raise_for_status()

        data = np.frombuffer(
            response.content,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            data,
            cv2.IMREAD_UNCHANGED
        )

        if image is None:

            return {
                "ok": False,
                "cobertura_disponible": None,
                "error":
                    "No se pudo decodificar mascara"
            }

        if (
            image.ndim == 3
            and image.shape[2] == 4
        ):

            alpha = image[:, :, 3]

            available = int(
                np.count_nonzero(
                    alpha >= 20
                )
            )

            return {
                "ok": True,
                "cobertura_disponible":
                    available > 0,
                "pixeles_cobertura":
                    available
            }

        nonzero = int(
            np.count_nonzero(
                image
            )
        )

        return {
            "ok": True,
            "cobertura_disponible":
                nonzero > 0,
            "pixeles_cobertura":
                nonzero
        }

    except Exception as exc:

        print(
            "[WARN] "
            "No se pudo consultar cobertura "
            f"RainViewer: {exc}"
        )

        return {
            "ok": False,
            "cobertura_disponible": None,
            "error": str(exc)
        }


# ============================================================
# GUARDADO DE IMAGEN
# ============================================================

def save_png(
    image,
    path
):

    temporary = path + ".tmp.png"

    ok = cv2.imwrite(
        temporary,
        image
    )

    if not ok:

        raise RuntimeError(
            "No se pudo guardar imagen"
        )

    os.replace(
        temporary,
        path
    )


# ============================================================
# GEOMETRIA
# ============================================================

def meters_per_pixel():

    return (
        156543.03392804097
        / (2 ** ZOOM)
        * math.cos(
            math.radians(LAT)
        )
    )


def distance_from_center_km(
    x,
    y,
    width,
    height
):

    scale = meters_per_pixel()

    cx = width / 2.0
    cy = height / 2.0

    dx = float(x) - cx
    dy = float(y) - cy

    return (
        math.hypot(
            dx,
            dy
        )
        * scale
        / 1000.0
    )


def monitoring_circle(
    width,
    height
):

    scale = meters_per_pixel()

    radius_px = (
        DETECTION_RADIUS_KM
        * 1000.0
        / scale
    )

    yy, xx = np.ogrid[
        :height,
        :width
    ]

    cx = width / 2.0
    cy = height / 2.0

    return (
        (xx - cx) ** 2
        + (yy - cy) ** 2
        <= radius_px ** 2
    )


# ============================================================
# MASCARA DE PRECIPITACION
# ============================================================

def precipitation_mask(image):

    if (
        image.ndim == 3
        and image.shape[2] == 4
    ):

        b, g, r, alpha = cv2.split(
            image
        )

    elif image.ndim == 3:

        b, g, r = cv2.split(
            image
        )

        alpha = np.full(
            b.shape,
            255,
            dtype=np.uint8
        )

    else:

        b = image
        g = image
        r = image

        alpha = np.full(
            image.shape,
            255,
            dtype=np.uint8
        )

    bgr = cv2.merge(
        [
            b,
            g,
            r
        ]
    )

    hsv = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2HSV
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Umbral bajo para detectar
    # ecos radar debiles.
    colorful = (
        saturation >= 8
    )

    visible = (
        value >= 8
    )

    opaque = (
        alpha >= 10
    )

    mask = (
        colorful
        & visible
        & opaque
    ).astype(
        np.uint8
    ) * 255

    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    monitor = monitoring_circle(
        mask.shape[1],
        mask.shape[0]
    )

    mask[
        ~monitor
    ] = 0

    return mask


# ============================================================
# ANALISIS
# ============================================================

def analyze_frame(image):

    mask = precipitation_mask(
        image
    )

    height, width = mask.shape

    area = int(
        cv2.countNonZero(
            mask
        )
    )

    monitor = monitoring_circle(
        width,
        height
    )

    monitored_pixels = int(
        np.count_nonzero(
            monitor
        )
    )

    coverage = (
        area / monitored_pixels
        if monitored_pixels
        else 0.0
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    nuclei = []

    for contour in contours:

        contour_area = cv2.contourArea(
            contour
        )

        if contour_area < 3:
            continue

        x, y, cw, ch = cv2.boundingRect(
            contour
        )

        moments = cv2.moments(
            contour
        )

        if moments["m00"]:

            cx = (
                moments["m10"]
                / moments["m00"]
            )

            cy = (
                moments["m01"]
                / moments["m00"]
            )

        else:

            cx = (
                x
                + cw / 2.0
            )

            cy = (
                y
                + ch / 2.0
            )

        distance = distance_from_center_km(
            cx,
            cy,
            width,
            height
        )

        nuclei.append(
            {
                "x": round(
                    float(cx),
                    2
                ),
                "y": round(
                    float(cy),
                    2
                ),
                "area_px": int(
                    contour_area
                ),
                "width_px": int(
                    cw
                ),
                "height_px": int(
                    ch
                ),
                "distance_km": round(
                    distance,
                    1
                ),
                "prioridad_100km": (
                    distance
                    <= PRIORITY_RADIUS_KM
                )
            }
        )

    nuclei.sort(
        key=lambda item:
        item["area_px"],
        reverse=True
    )

    nuclei = nuclei[:50]

    priority_nuclei = [
        item
        for item in nuclei
        if item[
            "prioridad_100km"
        ]
    ]

    ys, xs = np.where(
        mask > 0
    )

    if len(xs):

        centroid_x = float(
            np.mean(xs)
        )

        centroid_y = float(
            np.mean(ys)
        )

        centroid_distance = (
            distance_from_center_km(
                centroid_x,
                centroid_y,
                width,
                height
            )
        )

    else:

        centroid_x = 0.0
        centroid_y = 0.0
        centroid_distance = 0.0

    if image.ndim == 3:

        b = image[:, :, 0]
        g = image[:, :, 1]
        r = image[:, :, 2]

    else:

        b = image
        g = image
        r = image

    intense = (
        (
            (r > 180)
            |
            (g > 180)
            |
            (b > 180)
        )
        &
        (mask > 0)
    )

    return {
        "area_px":
            area,

        "coverage":
            coverage,

        "nuclei":
            nuclei,

        "priority_nuclei":
            priority_nuclei,

        "centroid_x":
            centroid_x,

        "centroid_y":
            centroid_y,

        "centroid_distance_km":
            round(
                centroid_distance,
                1
            ),

        "intense_pixels":
            int(
                np.count_nonzero(
                    intense
                )
            )
    }


# ============================================================
# MOVIMIENTO
# ============================================================

def estimate_motion(
    previous,
    current
):

    if (
        previous is None
        or current is None
    ):

        return (
            0.0,
            0.0,
            0.0
        )

    def to_gray(image):

        if image.ndim == 3:

            if image.shape[2] == 4:

                image = image[:, :, :3]

            return cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

        return image

    previous_gray = to_gray(
        previous
    )

    current_gray = to_gray(
        current
    )

    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        0.5,
        3,
        21,
        3,
        5,
        1.2,
        0
    )

    fx = flow[:, :, 0]
    fy = flow[:, :, 1]

    magnitude = np.sqrt(
        fx * fx
        + fy * fy
    )

    valid = (
        magnitude > 0.2
    )

    if not np.any(valid):

        return (
            0.0,
            0.0,
            0.0
        )

    movement_x = float(
        np.median(
            fx[valid]
        )
    )

    movement_y = float(
        np.median(
            fy[valid]
        )
    )

    speed = float(
        np.median(
            magnitude[valid]
        )
    )

    return (
        movement_x,
        movement_y,
        speed
    )


def movement_direction(
    movement_x,
    movement_y
):

    if (
        abs(movement_x) < 0.05
        and abs(movement_y) < 0.05
    ):

        return "sin_movimiento"

    angle = math.degrees(
        math.atan2(
            movement_x,
            -movement_y
        )
    )

    if angle < 0:
        angle += 360

    directions = [
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
        (angle + 22.5)
        // 45
    ) % 8

    return directions[index]


# ============================================================
# HISTORIAL
# ============================================================

def cleanup_history():

    files = []

    for name in os.listdir(
        HISTORY_DIR
    ):

        if not name.endswith(
            ".png"
        ):
            continue

        path = os.path.join(
            HISTORY_DIR,
            name
        )

        if os.path.isfile(path):

            files.append(
                path
            )

    files.sort(
        key=lambda p:
        os.path.getmtime(p),
        reverse=True
    )

    for old in files[
        MAX_HISTORY:
    ]:

        try:

            os.remove(
                old
            )

        except OSError:

            pass


# ============================================================
# CSV
# ============================================================

def append_features(row):

    exists = os.path.exists(
        FEATURES_PATH
    )

    with open(
        FEATURES_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS
        )

        if not exists:

            writer.writeheader()

        writer.writerow(
            row
        )


# ============================================================
# ESTADO DE FUENTE
# ============================================================

def source_health_from_observations(
    observations,
    coverage_info
):

    if not observations:

        return {
            "estado_fuente":
                "sin_datos",

            "radar_con_datos":
                False,

            "motivo":
                "No hubo frames descargables"
        }

    transparent_count = sum(
        1
        for item in observations
        if item[
            "fully_transparent"
        ]
    )

    visible_count = (
        len(observations)
        - transparent_count
    )

    # Si hubo al menos un frame
    # con datos visibles, RainViewer
    # entrego informacion util.
    if visible_count > 0:

        return {
            "estado_fuente":
                "datos_recibidos",

            "radar_con_datos":
                True,

            "motivo":
                "Hay frames con pixeles radar visibles"
        }

    # Sin cobertura geografica.
    if (
        coverage_info.get(
            "cobertura_disponible"
        )
        is False
    ):

        return {
            "estado_fuente":
                "sin_cobertura",

            "radar_con_datos":
                False,

            "motivo":
                "La mascara indica ausencia de cobertura"
        }

    # TODOS transparentes:
    # NO afirmamos que esta seco.
    return {
        "estado_fuente":
            "fuente_sin_datos",

        "radar_con_datos":
            False,

        "motivo":
            (
                "Todos los frames recientes "
                "estan completamente transparentes"
            )
    }


# ============================================================
# FALLBACK
# ============================================================

def load_last_valid_nowcast():

    if not os.path.exists(
        LAST_VALID_NOWCAST_PATH
    ):

        return None

    try:

        with open(
            LAST_VALID_NOWCAST_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as exc:

        print(
            "[WARN] "
            "No se pudo cargar ultima "
            f"observacion valida: {exc}"
        )

        return None


def activate_radar_fallback(
    current_nowcast,
    current_status
):

    last_valid = load_last_valid_nowcast()

    current_nowcast[
        "fallback"
    ] = {

        "activo":
            True,

        "motivo":
            (
                "Fuente radar sin datos "
                "confirmados; se conserva "
                "la ultima observacion valida."
            ),

        "ultima_observacion_valida":
            last_valid is not None
    }

    current_status[
        "fallback_activo"
    ] = True

    current_status[
        "ultima_observacion_valida"
    ] = (
        last_valid is not None
    )

    if last_valid is not None:

        current_nowcast[
            "ultima_observacion_valida"
        ] = last_valid

    if os.path.exists(
        LAST_VALID_IMAGE_PATH
    ):

        try:

            atomic_copy_file(
                LAST_VALID_IMAGE_PATH,
                ACTUAL_PATH
            )

            current_status[
                "imagen_actual"
            ] = (
                "ultima_imagen_radar_valida"
            )

        except Exception as exc:

            print(
                "[WARN] "
                "No se pudo restaurar "
                f"ultima imagen valida: {exc}"
            )

    return (
        current_nowcast,
        current_status
    )


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def main():

    ensure_dirs()

    print(
        "[CLIMAAR] "
        "Iniciando radar RainViewer v10.0"
    )

    print(
        "[CLIMAAR] "
        f"Centro={LAT},{LON} "
        f"radio={DETECTION_RADIUS_KM}km "
        f"prioridad={PRIORITY_RADIUS_KM}km"
    )

    host, frames = get_radar_frames()

    selected = frames[
        -FRAMES:
    ]

    coverage_info = download_coverage_status(
        host
    )

    print(
        "[RADAR] Cobertura: "
        f"{coverage_info.get('cobertura_disponible')}"
    )

    observations = []

    pending_rows = []

    previous_image = None

    successful_frames = 0

    for item in selected:

        timestamp = item[
            "time"
        ]

        frame_utc = iso_from_timestamp(
            timestamp
        )

        url = build_image_url(
            host,
            item["path"]
        )

        print(
            "[RADAR] "
            f"Descargando: {frame_utc}"
        )

        try:

            image, fully_transparent = (
                download_image(
                    url
                )
            )

            analysis = analyze_frame(
                image
            )

            (
                movement_x,
                movement_y,
                speed
            ) = estimate_motion(
                previous_image,
                image
            )

            direction = movement_direction(
                movement_x,
                movement_y
            )

            history_path = os.path.join(
                HISTORY_DIR,
                f"{timestamp}.png"
            )

            save_png(
                image,
                history_path
            )

            previous_image = image

            successful_frames += 1

            observation = {

                "time":
                    timestamp,

                "utc":
                    frame_utc,

                "image":
                    image,

                "analysis":
                    analysis,

                "movement_x":
                    movement_x,

                "movement_y":
                    movement_y,

                "speed":
                    speed,

                "direction":
                    direction,

                "fully_transparent":
                    fully_transparent
            }

            observations.append(
                observation
            )

            row = {

                "frame_time":
                    timestamp,

                "frame_utc":
                    frame_utc,

                "source":
                    "RainViewer",

                "latitud":
                    LAT,

                "longitud":
                    LON,

                "area_px":
                    analysis[
                        "area_px"
                    ],

                "distance_km":
                    analysis[
                        "centroid_distance_km"
                    ],

                "dbz_max":
                    "",

                "dbz_mean":
                    "",

                "dbz_p90":
                    "",

                "dbz_pixels":
                    0,

                "tiles_ok":
                    1,

                "nucleos":
                    len(
                        analysis[
                            "nuclei"
                        ]
                    ),

                "centroide_x":
                    round(
                        analysis[
                            "centroid_x"
                        ],
                        2
                    ),

                "centroide_y":
                    round(
                        analysis[
                            "centroid_y"
                        ],
                        2
                    ),

                "pixeles_intensos":
                    analysis[
                        "intense_pixels"
                    ],

                "cobertura_precipitacion":
                    round(
                        analysis[
                            "coverage"
                        ],
                        6
                    ),

                "movimiento_x":
                    round(
                        movement_x,
                        4
                    ),

                "movimiento_y":
                    round(
                        movement_y,
                        4
                    ),

                "direccion":
                    direction,

                "velocidad_pixeles_frame":
                    round(
                        speed,
                        4
                    )
            }

            pending_rows.append(
                row
            )

        except Exception as exc:

            print(
                "[WARN] "
                f"Frame rechazado: {exc}"
            )

    if not observations:

        raise RuntimeError(
            "No hubo ningun frame descargable"
        )

    latest = observations[
        -1
    ]

    latest_image = latest[
        "image"
    ]

    latest_analysis = latest[
        "analysis"
    ]

    latest_time = latest[
        "time"
    ]

    age_seconds = (
        datetime.now(
            timezone.utc
        ).timestamp()
        - latest_time
    )

    age_minutes = (
        age_seconds
        / 60.0
    )

    precipitation_area = (
        latest_analysis[
            "area_px"
        ]
    )

    nuclei = latest_analysis[
        "nuclei"
    ]

    priority_nuclei = latest_analysis[
        "priority_nuclei"
    ]

    source_health = (
        source_health_from_observations(
            observations,
            coverage_info
        )
    )

    # ========================================================
    # ESTADO
    # ========================================================

    if age_minutes > 35:

        estado = (
            "radar_desactualizado"
        )

        datos_vigentes = False

        confianza = 0.0

    elif (
        source_health[
            "estado_fuente"
        ]
        == "sin_cobertura"
    ):

        estado = (
            "radar_sin_cobertura"
        )

        datos_vigentes = False

        confianza = 0.0

    elif (
        source_health[
            "estado_fuente"
        ]
        == "fuente_sin_datos"
    ):

        estado = (
            "radar_sin_datos"
        )

        datos_vigentes = False

        confianza = 0.0

    elif precipitation_area > 0:

        estado = (
            "precipitacion_detectada"
        )

        datos_vigentes = True

        confianza = min(
            1.0,
            0.50
            + min(
                0.40,
                len(nuclei) * 0.03
            )
        )

    else:

        estado = (
            "sin_precipitacion"
        )

        datos_vigentes = True

        confianza = 0.50

    # ========================================================
    # PUBLICAR FEATURES SOLO SI LA FUENTE ENTREGO DATOS
    # ========================================================

    if source_health[
        "radar_con_datos"
    ]:

        for row in pending_rows:

            append_features(
                row
            )

    else:

        print(
            "[RADAR] "
            "No se agregan filas al CSV: "
            "fuente radar sin datos confirmados."
        )

    # ========================================================
    # NOWCAST
    # ========================================================

    nowcast = {

        "version":
            "10.0-source-health",

        "generated_utc":
            utc_now(),

        "source":
            "RainViewer",

        "radar_frame_utc":
            latest["utc"],

        "radar_frame_timestamp":
            latest_time,

        "edad_minutos":
            round(
                age_minutes,
                1
            ),

        "frame_es_viejo":
            age_minutes > 35,

        "datos_vigentes":
            datos_vigentes,

        "estado":
            estado,

        "confianza":
            round(
                confianza,
                3
            ),

        "fuente": {

            "estado":
                source_health[
                    "estado_fuente"
                ],

            "radar_con_datos":
                source_health[
                    "radar_con_datos"
                ],

            "motivo":
                source_health[
                    "motivo"
                ],

            "cobertura":
                coverage_info
        },

        "centro": {

            "latitud":
                LAT,

            "longitud":
                LON
        },

        "deteccion": {

            "radio_km":
                DETECTION_RADIUS_KM,

            "prioridad_km":
                PRIORITY_RADIUS_KM,

            "cobertura_precipitacion":
                round(
                    latest_analysis[
                        "coverage"
                    ],
                    6
                ),

            "area_px":
                latest_analysis[
                    "area_px"
                ],

            "nucleos":
                nuclei,

            "nucleos_prioridad":
                priority_nuclei,

            "pixeles_intensos":
                latest_analysis[
                    "intense_pixels"
                ]
        },

        "movimiento": {

            "x":
                round(
                    latest[
                        "movement_x"
                    ],
                    4
                ),

            "y":
                round(
                    latest[
                        "movement_y"
                    ],
                    4
                ),

            "velocidad_pixeles_frame":
                round(
                    latest[
                        "speed"
                    ],
                    4
                ),

            "direccion":
                latest[
                    "direction"
                ]
        },

        "historico_frames_validos":
            successful_frames,

        "frames_transparentes":
            sum(
                1
                for item in observations
                if item[
                    "fully_transparent"
                ]
            ),

        "dbz_disponible":
            False,

        "nota_dbz":
            (
                "RainViewer RGB no se "
                "convierte artificialmente "
                "a dBZ."
            )
    }

    # ========================================================
    # STATUS
    # ========================================================

    status = {

        "version":
            "10.0-source-health",

        "updated_utc":
            utc_now(),

        "source":
            "RainViewer",

        "estado":
            estado,

        "datos_vigentes":
            datos_vigentes,

        "frame_es_viejo":
            age_minutes > 35,

        "edad_minutos":
            round(
                age_minutes,
                1
            ),

        "radio_deteccion_km":
            DETECTION_RADIUS_KM,

        "radio_prioridad_km":
            PRIORITY_RADIUS_KM,

        "nucleos":
            len(nuclei),

        "nucleos_prioridad":
            len(priority_nuclei),

        "cobertura_precipitacion":
            round(
                latest_analysis[
                    "coverage"
                ],
                6
            ),

        "frames_validos":
            successful_frames,

        "frames_transparentes":
            sum(
                1
                for item in observations
                if item[
                    "fully_transparent"
                ]
            ),

        "fuente":
            source_health,

        "cobertura_radar":
            coverage_info,

        "dbz_disponible":
            False,

        "fallback_activo":
            False,

        "ultima_observacion_valida":
            False
    }

    # ========================================================
    # FUENTE SANA / FALLBACK
    # ========================================================

    if source_health[
        "radar_con_datos"
    ]:

        save_png(
            latest_image,
            ACTUAL_PATH
        )

        save_png(
            latest_image,
            LAST_VALID_IMAGE_PATH
        )

        atomic_json_write(
            LAST_VALID_NOWCAST_PATH,
            nowcast
        )

    elif (
        estado
        in (
            "radar_sin_datos",
            "radar_sin_cobertura",
            "radar_desactualizado"
        )
    ):

        (
            nowcast,
            status
        ) = activate_radar_fallback(
            nowcast,
            status
        )

    else:

        save_png(
            latest_image,
            ACTUAL_PATH
        )

    atomic_json_write(
        NOWCAST_PATH,
        nowcast
    )

    atomic_json_write(
        STATUS_PATH,
        status
    )

    cleanup_history()

    print("")

    print(
        "============================================"
    )

    print(
        " CLIMAAR RADAR FINALIZADO"
    )

    print(
        "============================================"
    )

    print(
        f"Frame: {latest['utc']}"
    )

    print(
        f"Edad: {age_minutes:.1f} min"
    )

    print(
        f"Estado: {estado}"
    )

    print(
        "Fuente: "
        f"{source_health['estado_fuente']}"
    )

    print(
        f"Precipitacion: "
        f"{precipitation_area} px"
    )

    print(
        f"Nucleos: "
        f"{len(nuclei)}"
    )

    print(
        "Nucleos prioridad: "
        f"{len(priority_nuclei)}"
    )

    print(
        "Frames validos: "
        f"{successful_frames}/{len(selected)}"
    )

    print(
        "Frames transparentes: "
        f"{sum(1 for item in observations if item['fully_transparent'])}"
    )

    print(
        "Cobertura radar: "
        f"{coverage_info.get('cobertura_disponible')}"
    )

    print(
        "dBZ: no disponible "
        "(no se inventa)"
    )

    print(
        "============================================"
    )


# ============================================================
# EJECUCION
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print(
            f"[ERROR] {exc}"
        )

        raise
