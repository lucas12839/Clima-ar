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

RAINVIEWER_API = (
    "https://api.rainviewer.com/public/weather-maps.json"
)

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
    os.makedirs(RADAR_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)


def iso_from_timestamp(ts):
    return datetime.fromtimestamp(
        int(ts),
        tz=timezone.utc
    ).isoformat()


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


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
        key=lambda item: item["time"]
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
    """
    RainViewer permite solicitar una imagen
    centrada directamente por lat/lon.

    Formato oficial:

    /{size}/{z}/{lat}/{lon}/{color}/{smooth}_{snow}.png
    """

    return (
        f"{host}{path}/"
        f"{SIZE}/{ZOOM}/"
        f"{LAT}/{LON}/"
        f"2/0_0.png"
    )


# ============================================================
# DESCARGA Y VALIDACION DE IMAGEN
# ============================================================

def download_image(url):
    response = SESSION.get(
        url,
        timeout=40
    )

    response.raise_for_status()

    content_type = (
        response.headers.get(
            "content-type",
            ""
        ).lower()
    )

    if len(response.content) < 512:
        raise RuntimeError(
            "RainViewer devolvio una respuesta demasiado pequena"
        )

    if (
        "image" not in content_type
        and not response.content.startswith(b"\x89PNG")
    ):
        raise RuntimeError(
            "RainViewer no devolvio una imagen PNG valida"
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
            "OpenCV no pudo decodificar la imagen"
        )

    if image.ndim not in (2, 3):
        raise RuntimeError(
            "Formato de imagen inesperado"
        )

    height, width = image.shape[:2]

    if width < 64 or height < 64:
        raise RuntimeError(
            "Imagen de radar demasiado pequena"
        )

    # --------------------------------------------------------
    # DIAGNOSTICO DE CONTENIDO
    # --------------------------------------------------------

    if image.ndim == 3 and image.shape[2] == 4:

        b, g, r, alpha = cv2.split(
            image
        )

        rgb_max = int(
            max(
                b.max(),
                g.max(),
                r.max()
            )
        )

        alpha_max = int(
            alpha.max()
        )

        alpha_pixels = int(
            np.count_nonzero(
                alpha >= 20
            )
        )

        # Imagen completamente transparente =
        # fuente sin datos utilizables.
        if alpha_max == 0:
            raise RuntimeError(
                "RainViewer devolvio una imagen completamente transparente"
            )

        # Si existe alpha, guardamos RGBA.
        print(
            "[RADAR] "
            f"imagen={width}x{height} "
            f"rgb_max={rgb_max} "
            f"pixeles_visibles={alpha_pixels}"
        )

    else:

        rgb_max = int(
            image.max()
        )

        nonzero = int(
            np.count_nonzero(image)
        )

        print(
            "[RADAR] "
            f"imagen={width}x{height} "
            f"max={rgb_max} "
            f"pixeles_no_negros={nonzero}"
        )

    return image


def save_png(
    image,
    path
):
    temporary = (
        path
        + ".tmp"
    )

    ok = cv2.imwrite(
        temporary,
        image
    )

    if not ok:
        raise RuntimeError(
            "No se pudo guardar la imagen"
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

    dx = (
        float(x)
        - cx
    )

    dy = (
        float(y)
        - cy
    )

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
# DETECCION DE PRECIPITACION
# ============================================================

def precipitation_mask(image):

    if image.ndim == 3 and image.shape[2] == 4:

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
        [b, g, r]
    )

    hsv = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2HSV
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    colorful = (
        saturation >= 12
    )

    visible = (
        value >= 15
    )

    opaque = (
        alpha >= 20
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

def analyze_frame(
    image
):
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

        if contour_area < 8:
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

        distance = (
            distance_from_center_km(
                cx,
                cy,
                width,
                height
            )
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
                "width_px": int(cw),
                "height_px": int(ch),
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
        if item["prioridad_100km"]
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
        "area_px": area,
        "coverage": coverage,
        "nuclei": nuclei,
        "priority_nuclei": priority_nuclei,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "centroid_distance_km": round(
            centroid_distance,
            1
        ),
        "intense_pixels": int(
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

    previous_bgr = (
        previous[:, :, :3]
        if previous.ndim == 3
        else previous
    )

    current_bgr = (
        current[:, :, :3]
        if current.ndim == 3
        else current
    )

    previous_gray = cv2.cvtColor(
        previous_bgr,
        cv2.COLOR_BGR2GRAY
    )

    current_gray = cv2.cvtColor(
        current_bgr,
        cv2.COLOR_BGR2GRAY
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

    speed = math.hypot(
        movement_x,
        movement_y
    )

    return (
        movement_x,
        movement_y,
        speed
    )


def direction_from_vector(
    x,
    y
):

    if (
        abs(x) < 0.05
        and abs(y) < 0.05
    ):
        return "estacionario"

    angle = (
        math.degrees(
            math.atan2(
                x,
                -y
            )
        )
        % 360
    )

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
        (
            angle
            + 22.5
        )
        % 360
        / 45
    )

    return directions[index]


# ============================================================
# CSV
# ============================================================

def load_existing_rows():

    if not os.path.exists(
        FEATURES_PATH
    ):
        return []

    rows = []

    try:

        with open(
            FEATURES_PATH,
            "r",
            encoding="utf-8",
            newline=""
        ) as handle:

            reader = csv.DictReader(
                handle
            )

            for row in reader:

                rows.append(
                    {
                        field:
                        row.get(
                            field,
                            ""
                        )
                        for field
                        in CSV_FIELDS
                    }
                )

    except Exception as exc:

        print(
            "[WARN] Error leyendo CSV:",
            exc
        )

    return rows


def write_features(
    rows
):

    unique = {}

    for row in rows:

        frame_time = str(
            row.get(
                "frame_time",
                ""
            )
        ).strip()

        if not frame_time:
            continue

        clean = {
            field:
            row.get(
                field,
                ""
            )
            for field
            in CSV_FIELDS
        }

        unique[
            frame_time
        ] = clean

    ordered = list(
        unique.values()
    )

    ordered.sort(
        key=lambda row:
        int(
            row["frame_time"]
        )
    )

    ordered = ordered[
        -MAX_HISTORY:
    ]

    temporary = (
        FEATURES_PATH
        + ".tmp"
    )

    with open(
        temporary,
        "w",
        encoding="utf-8",
        newline=""
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS
        )

        writer.writeheader()

        writer.writerows(
            ordered
        )

    os.replace(
        temporary,
        FEATURES_PATH
    )


# ============================================================
# JSON
# ============================================================

def write_json(
    path,
    data
):

    temporary = (
        path
        + ".tmp"
    )

    with open(
        temporary,
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            data,
            handle,
            ensure_ascii=False,
            indent=2
        )

    os.replace(
        temporary,
        path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_dirs()

    print(
        "========================================"
    )

    print(
        "CLIMAAR - RAINVIEWER RADAR"
    )

    print(
        f"Deteccion: {DETECTION_RADIUS_KM} km"
    )

    print(
        f"Prioridad: {PRIORITY_RADIUS_KM} km"
    )

    print(
        "========================================"
    )

    try:

        host, frames = get_radar_frames()

    except Exception as exc:

        print(
            "[ERROR] No se pudo consultar RainViewer:",
            exc
        )

        write_json(
            STATUS_PATH,
            {
                "estado": "error_fuente",
                "datos_vigentes": False,
                "error": str(exc),
                "updated_at": utc_now()
            }
        )

        return 1

    selected = frames[
        -FRAMES:
    ]

    rows = load_existing_rows()

    images = []

    diagnostics = []

    previous_image = None

    latest_valid_image = None

    latest_valid_time = None

    latest_analysis = None

    latest_motion = (
        0.0,
        0.0,
        0.0
    )

    for frame in selected:

        frame_time = int(
            frame["time"]
        )

        frame_utc = iso_from_timestamp(
            frame_time
        )

        url = build_image_url(
            host,
            frame["path"]
        )

        print(
            "[RADAR] Descargando:",
            frame_utc
        )

        try:

            image = download_image(
                url
            )

            analysis = analyze_frame(
                image
            )

            motion = estimate_motion(
                previous_image,
                image
            )

            direction = (
                direction_from_vector(
                    motion[0],
                    motion[1]
                )
            )

            row = {
                "frame_time": frame_time,
                "frame_utc": frame_utc,
                "source": "RainViewer",
                "latitud": LAT,
                "longitud": LON,
                "area_px": analysis["area_px"],
                "distance_km": analysis[
                    "centroid_distance_km"
                ],
                "dbz_max": "",
                "dbz_mean": "",
                "dbz_p90": "",
                "dbz_pixels": "",
                "tiles_ok": 1,
                "nucleos": json.dumps(
                    analysis["nuclei"],
                    ensure_ascii=False,
                    separators=(",", ":")
                ),
                "centroide_x": analysis[
                    "centroid_x"
                ],
                "centroide_y": analysis[
                    "centroid_y"
                ],
                "pixeles_intensos": analysis[
                    "intense_pixels"
                ],
                "cobertura_precipitacion": (
                    analysis["coverage"]
                ),
                "movimiento_x": motion[0],
                "movimiento_y": motion[1],
                "direccion": direction,
                "velocidad_pixeles_frame": motion[2],
            }

            rows.append(
                row
            )

            images.append(
                image
            )

            diagnostics.append(
                {
                    "frame_utc": frame_utc,
                    "valido": True,
                    "area_px": analysis[
                        "area_px"
                    ],
                    "nucleos": len(
                        analysis["nuclei"]
                    ),
                    "prioridad_100km": len(
                        analysis[
                            "priority_nuclei"
                        ]
                    )
                }
            )

            latest_valid_image = image
            latest_valid_time = frame_time
            latest_analysis = analysis
            latest_motion = motion

            previous_image = image

        except Exception as exc:

            print(
                "[WARN] Frame rechazado:",
                exc
            )

            diagnostics.append(
                {
                    "frame_utc": frame_utc,
                    "valido": False,
                    "error": str(exc)
                }
            )

    if not latest_valid_image:

        print(
            "[ERROR] No hubo ningun frame valido."
        )

        write_json(
            STATUS_PATH,
            {
                "estado": "error_fuente",
                "datos_vigentes": False,
                "frame_valido": False,
                "deteccion_radio_km": (
                    DETECTION_RADIUS_KM
                ),
                "prioridad_radio_km": (
                    PRIORITY_RADIUS_KM
                ),
                "diagnosticos": diagnostics,
                "updated_at": utc_now()
            }
        )

        return 1

    # --------------------------------------------------------
    # GUARDAR IMAGEN ACTUAL
    # --------------------------------------------------------

    save_png(
        latest_valid_image,
        ACTUAL_PATH
    )

    # --------------------------------------------------------
    # GUARDAR CSV
    # --------------------------------------------------------

    write_features(
        rows
    )

    # --------------------------------------------------------
    # ESTADO
    # --------------------------------------------------------

    analysis = latest_analysis

    precipitation = (
        analysis["area_px"] > 0
    )

    state = (
        "precipitacion_detectada"
        if precipitation
        else "sin_precipitacion"
    )

    now = datetime.now(
        timezone.utc
    )

    age_minutes = (
        now.timestamp()
        - latest_valid_time
    ) / 60.0

    fresh = (
        age_minutes <= 35
    )

    if not fresh:

        state = (
            "radar_desactualizado"
        )

    nowcast = {

        "version": "9.0-rainviewer-safe",

        "generated_at": utc_now(),

        "source": "RainViewer",

        "frame_actual_utc":
            iso_from_timestamp(
                latest_valid_time
            ),

        "frame_age_minutes":
            round(
                age_minutes,
                1
            ),

        "frame_es_viejo":
            not fresh,

        "datos_vigentes":
            bool(fresh),

        "estado":
            state,

        "deteccion_radio_km":
            DETECTION_RADIUS_KM,

        "prioridad_radio_km":
            PRIORITY_RADIUS_KM,

        "precipitacion_detectada":
            precipitation,

        "cobertura_precipitacion":
            analysis["coverage"],

        "area_px":
            analysis["area_px"],

        "nucleos":
            analysis["nuclei"],

        "nucleos_prioridad_100km":
            analysis[
                "priority_nuclei"
            ],

        "centroide": {

            "x":
                analysis["centroid_x"],

            "y":
                analysis["centroid_y"],

            "distancia_km":
                analysis[
                    "centroid_distance_km"
                ]
        },

        "movimiento": {

            "x":
                latest_motion[0],

            "y":
                latest_motion[1],

            "velocidad_pixeles_frame":
                latest_motion[2],

            "direccion":
                direction_from_vector(
                    latest_motion[0],
                    latest_motion[1]
                )
        },

        "dbz_disponible":
            False,

        "dbz_nota":
            "RainViewer RGB no se convierte artificialmente a dBZ.",

        "diagnosticos":
            diagnostics
    }

    write_json(
        NOWCAST_PATH,
        nowcast
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    write_json(
        STATUS_PATH,
        {
            "estado": state,
            "datos_vigentes": bool(
                fresh
            ),
            "frame_actual_utc":
                iso_from_timestamp(
                    latest_valid_time
                ),
            "frame_age_minutes":
                round(
                    age_minutes,
                    1
                ),
            "deteccion_radio_km":
                DETECTION_RADIUS_KM,
            "prioridad_radio_km":
                PRIORITY_RADIUS_KM,
            "precipitacion_detectada":
                precipitation,
            "nucleos":
                len(
                    analysis["nuclei"]
                ),
            "nucleos_prioridad_100km":
                len(
                    analysis[
                        "priority_nuclei"
                    ]
                ),
            "updated_at":
                utc_now()
        }
    )

    print(
        "----------------------------------------"
    )

    print(
        "[OK] Radar procesado"
    )

    print(
        "[OK] Frame:",
        iso_from_timestamp(
            latest_valid_time
        )
    )

    print(
        "[OK] Edad:",
        round(
            age_minutes,
            1
        ),
        "min"
    )

    print(
        "[OK] Precipitacion:",
        precipitation
    )

    print(
        "[OK] Nucleos:",
        len(
            analysis["nuclei"]
        )
    )

    print(
        "[OK] Prioridad 100 km:",
        len(
            analysis[
                "priority_nuclei"
            ]
        )
    )

    print(
        "[OK] actual.png guardado"
    )

    print(
        "========================================"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
