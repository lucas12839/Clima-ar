import csv
import json
import math
import os
from datetime import datetime, timezone

import cv2
import numpy as np
import requests


LAT = -38.71
LON = -62.26

ZOOM = 7
SIZE = 512

# ClimaAR vigila un radio de 150 km y da prioridad a lo que
# entra en los 100 km alrededor de Bahia Blanca.
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

RAINVIEWER_API = (
    "https://api.rainviewer.com/public/weather-maps.json"
)

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": "ClimaAR/1.0"
    }
)


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

    past = data.get(
        "radar",
        {}
    ).get(
        "past",
        []
    )

    if not host or not past:
        raise RuntimeError(
            "RainViewer no devolvio datos de radar validos"
        )

    frames = [
        {
            "time": int(item["time"]),
            "path": item["path"]
        }
        for item in past
        if item.get("time")
        and item.get("path")
    ]

    frames.sort(
        key=lambda item: item["time"]
    )

    if not frames:
        raise RuntimeError(
            "No hay frames validos de RainViewer"
        )

    return host, frames


def build_image_url(host, path):
    return (
        f"{host}{path}/"
        f"{SIZE}/{ZOOM}/{LAT}/{LON}/"
        f"2/1_1.png"
    )


def download_image(url):
    response = SESSION.get(
        url,
        timeout=40
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
        raise RuntimeError(
            "No se pudo decodificar la imagen del radar"
        )

    return image


def radar_geometry():
    """
    Calcula la escala aproximada de Web Mercator para la latitud
    de Bahia Blanca. A zoom 7 y 512 px, la imagen cubre
    ampliamente el radio operativo de 150 km.
    """
    meters_per_pixel_equator = (
        156543.03392804097 / (2 ** ZOOM)
    )

    meters_per_pixel = (
        meters_per_pixel_equator
        * math.cos(math.radians(LAT))
    )

    return meters_per_pixel


def distance_from_center_km(x, y):
    meters_per_pixel = radar_geometry()

    dx = (
        float(x) - SIZE / 2.0
    )

    dy = (
        float(y) - SIZE / 2.0
    )

    return (
        math.hypot(dx, dy)
        * meters_per_pixel
        / 1000.0
    )


def monitoring_circle():
    meters_per_pixel = radar_geometry()

    radius_px = (
        DETECTION_RADIUS_KM
        * 1000.0
        / meters_per_pixel
    )

    yy, xx = np.ogrid[
        :SIZE,
        :SIZE
    ]

    cx = SIZE / 2.0
    cy = SIZE / 2.0

    return (
        (xx - cx) ** 2
        + (yy - cy) ** 2
        <= radius_px ** 2
    )


def precipitation_mask(image):
    """
    Detecta ecos de precipitacion debiles sin convertirlos a dBZ.

    RainViewer entrega una imagen coloreada, no una medicion dBZ
    calibrada en este endpoint. Por eso esta funcion detecta
    presencia de eco/lluvia por color y transparencia, pero NO
    inventa valores dBZ.
    """
    if image.ndim == 3 and image.shape[2] == 4:
        b, g, r, alpha = cv2.split(
            image
        )
    else:
        b, g, r = cv2.split(
            image
        )

        alpha = np.full(
            b.shape,
            255,
            dtype=np.uint8
        )

    hsv = cv2.cvtColor(
        cv2.merge(
            [b, g, r]
        ),
        cv2.COLOR_BGR2HSV
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Umbral deliberadamente sensible para no perder ecos debiles.
    colorful = saturation >= 18
    visible = value >= 22
    opaque = alpha >= 20

    mask = (
        colorful
        & visible
        & opaque
    ).astype(
        np.uint8
    ) * 255

    # Solo cerramos pequenos huecos.
    # No hacemos MORPH_OPEN porque podria eliminar ecos debiles.
    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    monitor = monitoring_circle()

    mask[
        ~monitor
    ] = 0

    return mask


def analyze_frame(image):
    mask = precipitation_mask(
        image
    )

    area = int(
        cv2.countNonZero(mask)
    )

    height, width = mask.shape

    coverage = (
        area / float(
            width * height
        )
        if width and height
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

        # Sensible para ecos debiles pequenos.
        if contour_area < 8:
            continue

        x, y, cw, ch = cv2.boundingRect(
            contour
        )

        moments = cv2.moments(
            contour
        )

        if moments["m00"] != 0:
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
            cy
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
                    distance <= PRIORITY_RADIUS_KM
                ),
            }
        )

    nuclei.sort(
        key=lambda item: item["area_px"],
        reverse=True
    )

    nuclei = nuclei[:50]

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
                centroid_y
            )
        )
    else:
        centroid_x = 0.0
        centroid_y = 0.0
        centroid_distance = 0.0

    if image.ndim == 3 and image.shape[2] == 4:
        b, g, r, _ = cv2.split(
            image
        )
    else:
        b, g, r = cv2.split(
            image
        )

    intense = (
        (
            (b > 180)
            |
            (g > 180)
            |
            (r > 180)
        )
        &
        (mask > 0)
    )

    priority_nuclei = [
        nucleus
        for nucleus in nuclei
        if nucleus["prioridad_100km"]
    ]

    return {
        "area_px": area,
        "coverage": coverage,
        "nuclei": nuclei,
        "priority_nuclei": priority_nuclei,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "centroid_distance_km": (
            round(
                centroid_distance,
                1
            )
        ),
        "intense_pixels": int(
            np.count_nonzero(
                intense
            )
        ),
    }


def estimate_motion(
    previous,
    current
):
    if (
        previous is None
        or current is None
    ):
        return 0.0, 0.0, 0.0

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
        fx * fx + fy * fy
    )

    valid = magnitude > 0.2

    if not np.any(valid):
        return 0.0, 0.0, 0.0

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
            angle + 22.5
        )
        % 360
        / 45
    )

    return directions[index]


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
                if row:
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
            "[WARN] No se pudo leer "
            f"CSV anterior: {exc}"
        )

    return rows


def normalize_nuclei(
    value
):
    if isinstance(
        value,
        list
    ):
        return value

    if isinstance(
        value,
        str
    ):
        try:
            parsed = json.loads(
                value
            )

            if isinstance(
                parsed,
                list
            ):
                return parsed

        except (
            TypeError,
            ValueError,
            json.JSONDecodeError
        ):
            pass

    return []


def write_features(
    rows
):
    unique = {}

    for row in rows:
        key = str(
            row.get(
                "frame_time",
                ""
            )
        ).strip()

        if not key:
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

        clean["nucleos"] = json.dumps(
            normalize_nuclei(
                clean["nucleos"]
            ),
            ensure_ascii=False,
            separators=(
                ",",
                ":"
            )
        )

        unique[key] = clean

    ordered = list(
        unique.values()
    )

    ordered.sort(
        key=lambda row:
        int(row["frame_time"])
    )

    ordered = ordered[-4320:]

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
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL
        )

        writer.writeheader()

        writer.writerows(
            ordered
        )

    os.replace(
        temporary,
        FEATURES_PATH
    )

    print(
        f"[OK] Features: "
        f"{len(ordered)} filas / "
        f"{len(CSV_FIELDS)} columnas"
    )


def save_json(
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


def save_image(
    image,
    path
):
    if not cv2.imwrite(
        path,
        image
    ):
        raise RuntimeError(
            f"No se pudo guardar {path}"
        )


def clean_old_history():
    files = [
        os.path.join(
            HISTORY_DIR,
            name
        )
        for name in os.listdir(
            HISTORY_DIR
        )
        if os.path.isfile(
            os.path.join(
                HISTORY_DIR,
                name
            )
        )
    ]

    files.sort(
        key=os.path.getmtime
    )

    while len(files) > MAX_HISTORY:
        os.remove(
            files.pop(0)
        )


def main():
    ensure_dirs()

    print(
        "[INFO] Iniciando radar "
        "RainViewer..."
    )

    print(
        "[INFO] Zona de deteccion: "
        f"{DETECTION_RADIUS_KM:.0f} km"
    )

    print(
        "[INFO] Zona prioritaria: "
        f"{PRIORITY_RADIUS_KM:.0f} km"
    )

    host, frames = (
        get_radar_frames()
    )

    selected = frames[-FRAMES:]

    downloaded = []

    for frame in selected:
        url = build_image_url(
            host,
            frame["path"]
        )

        print(
            "[INFO] Descargando "
            f"{iso_from_timestamp(frame['time'])}"
        )

        try:
            image = download_image(
                url
            )

            downloaded.append(
                {
                    "time": frame["time"],
                    "image": image
                }
            )

        except requests.RequestException as exc:
            print(
                "[WARN] Error descargando "
                f"frame: {exc}"
            )

    if not downloaded:
        raise RuntimeError(
            "No se pudo descargar "
            "ningun frame de RainViewer"
        )

    rows = load_existing_rows()

    previous_image = None

    for item in downloaded:
        image = item["image"]

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

        previous_image = image

        distance = (
            analysis[
                "centroid_distance_km"
            ]
            if analysis["area_px"] > 0
            else 0.0
        )

        frame_time = int(
            item["time"]
        )

        rows.append(
            {
                "frame_time":
                    frame_time,

                "frame_utc":
                    iso_from_timestamp(
                        frame_time
                    ),

                "source":
                    "RainViewer",

                "latitud":
                    LAT,

                "longitud":
                    LON,

                "area_px":
                    analysis["area_px"],

                "distance_km":
                    round(
                        distance,
                        3
                    ),

                "dbz_max":
                    "",

                "dbz_mean":
                    "",

                "dbz_p90":
                    "",

                "dbz_pixels":
                    "",

                "tiles_ok":
                    1,

                "nucleos":
                    json.dumps(
                        analysis["nuclei"],
                        ensure_ascii=False,
                        separators=(
                            ",",
                            ":"
                        )
                    ),

                "centroide_x":
                    round(
                        analysis["centroid_x"],
                        3
                    ),

                "centroide_y":
                    round(
                        analysis["centroid_y"],
                        3
                    ),

                "pixeles_intensos":
                    analysis[
                        "intense_pixels"
                    ],

                "cobertura_precipitacion":
                    round(
                        analysis["coverage"],
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
                    direction_from_vector(
                        movement_x,
                        movement_y
                    ),

                "velocidad_pixeles_frame":
                    round(
                        speed,
                        4
                    ),
            }
        )

        save_image(
            image,
            os.path.join(
                HISTORY_DIR,
                f"{frame_time}.png"
            )
        )

    write_features(
        rows
    )

    latest = downloaded[-1]["image"]

    latest_analysis = analyze_frame(
        latest
    )

    save_image(
        latest,
        ACTUAL_PATH
    )

    save_json(
        NOWCAST_PATH,
        {
            "generated_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "source":
                "RainViewer",

            "latitude":
                LAT,

            "longitude":
                LON,

            "detection_radius_km":
                DETECTION_RADIUS_KM,

            "priority_radius_km":
                PRIORITY_RADIUS_KM,

            "frames_analyzed":
                len(downloaded),

            "latest_frame":
                iso_from_timestamp(
                    downloaded[-1]["time"]
                ),

            "precipitation":
                {
                    "area_px":
                        latest_analysis[
                            "area_px"
                        ],

                    "coverage":
                        latest_analysis[
                            "coverage"
                        ],

                    "nuclei":
                        latest_analysis[
                            "nuclei"
                        ],

                    "priority_nuclei_100km":
                        latest_analysis[
                            "priority_nuclei"
                        ],

                    "centroid_x":
                        latest_analysis[
                            "centroid_x"
                        ],

                    "centroid_y":
                        latest_analysis[
                            "centroid_y"
                        ],

                    "centroid_distance_km":
                        latest_analysis[
                            "centroid_distance_km"
                        ],
                },
        }
    )

    save_json(
        STATUS_PATH,
        {
            "ok":
                True,

            "source":
                "RainViewer",

            "updated_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "latest_frame":
                iso_from_timestamp(
                    downloaded[-1]["time"]
                ),

            "frames_downloaded":
                len(downloaded),

            "features_updated":
                True,

            "csv_columns":
                len(CSV_FIELDS),

            "detection_radius_km":
                DETECTION_RADIUS_KM,

            "priority_radius_km":
                PRIORITY_RADIUS_KM,

            "nuclei_detected_latest":
                len(
                    latest_analysis[
                        "nuclei"
                    ]
                ),

            "nuclei_priority_100km_latest":
                len(
                    latest_analysis[
                        "priority_nuclei"
                    ]
                ),
        }
    )

    clean_old_history()

    print(
        "[OK] Radar actualizado "
        "correctamente"
    )


if __name__ == "__main__":
    main()
