#!/usr/bin/env python3

"""
ClimaAR - RainViewer Radar Engine

Obtiene frames reales de RainViewer,
guarda histórico,
actualiza actual.png,
genera features para seguimiento de tormentas
y prepara radar_nowcast.json.

NO inventa dBZ.
Las variables dBZ quedan vacías hasta disponer
de una fuente calibrada.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


# ============================================================
# CONFIGURACIÓN
# ============================================================

LAT = -38.71
LON = -62.26

ZOOM = 7
SIZE = 512

COLOR_SCHEME = 2
SMOOTH = 0
SNOW = 0

FRAMES = 6
MAX_HISTORY = 144

API = "https://api.rainviewer.com/public/weather-maps.json"

BASE = Path(__file__).resolve().parents[1]

RADAR = BASE / "data" / "radar"
HISTORY = RADAR / "historico"

ACTUAL = RADAR / "actual.png"
NOWCAST = RADAR / "radar_nowcast.json"
STATUS = RADAR / "status.json"

FEATURES = RADAR / "radar_features_rainviewer.csv"

TEMP = RADAR / "_frames"


# ============================================================
# DIRECTORIOS
# ============================================================

def prepare_directories():

    RADAR.mkdir(
        parents=True,
        exist_ok=True
    )

    HISTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    TEMP.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# API
# ============================================================

def get_api_data():

    last_error = None

    for attempt in range(3):

        try:

            response = requests.get(
                API,
                timeout=30,
                headers={
                    "Cache-Control": "no-cache",
                    "User-Agent": "ClimaAR/1.0"
                }
            )

            response.raise_for_status()

            data = response.json()

            radar = data.get(
                "radar",
                {}
            )

            frames = radar.get(
                "past",
                []
            )

            if not frames:
                raise RuntimeError(
                    "RainViewer no devolvió frames"
                )

            host = data.get(
                "host",
                ""
            )

            if not host:
                raise RuntimeError(
                    "RainViewer no devolvió host"
                )

            return data, frames

        except Exception as exc:

            last_error = exc

            print(
                f"Intento {attempt + 1}/3: {exc}"
            )

            time.sleep(2)

    raise RuntimeError(
        f"RainViewer error: {last_error}"
    )


# ============================================================
# URL
# ============================================================

def frame_url(
    host,
    frame
):

    path = frame.get(
        "path"
    )

    if not path:
        raise RuntimeError(
            "Frame sin path"
        )

    return (
        f"{host}{path}/"
        f"{SIZE}/{ZOOM}/{LAT}/{LON}/"
        f"{COLOR_SCHEME}/{SMOOTH}_{SNOW}.png"
    )


# ============================================================
# DESCARGA
# ============================================================

def download_frame(
    url,
    timestamp
):

    output = (
        TEMP /
        f"frame_{timestamp}.png"
    )

    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "ClimaAR/1.0"
        }
    )

    response.raise_for_status()

    output.write_bytes(
        response.content
    )

    image = Image.open(
        output
    ).convert("RGBA")

    image.save(
        output,
        format="PNG"
    )

    return output


# ============================================================
# ANÁLISIS DE IMAGEN
# ============================================================

def analyze_image(
    path
):

    image = Image.open(
        path
    ).convert("RGBA")

    rgba = np.array(
        image
    )

    rgb = rgba[:, :, :3]

    maximum = rgb.max(
        axis=2
    )

    minimum = rgb.min(
        axis=2
    )

    intensity = (
        maximum -
        minimum
    )

    mask = (
        (intensity > 25)
        &
        (maximum > 70)
    )

    pixels = int(
        mask.sum()
    )

    total = mask.size

    coverage = (
        pixels / total
        if total
        else 0.0
    )

    strong = (
        (
            (rgb[:, :, 0] > 140)
            |
            (rgb[:, :, 1] > 140)
            |
            (rgb[:, :, 2] > 140)
        )
        &
        mask
    )

    strong_pixels = int(
        strong.sum()
    )

    # --------------------------------------------------------
    # CENTROIDE
    # --------------------------------------------------------

    centroid_x = None
    centroid_y = None

    if pixels > 0:

        ys, xs = np.where(
            mask
        )

        if len(xs) > 0:

            centroid_x = float(
                np.mean(xs)
            )

            centroid_y = float(
                np.mean(ys)
            )

    # --------------------------------------------------------
    # NÚCLEOS
    # --------------------------------------------------------

    nuclei = []

    if cv2 is not None and pixels > 0:

        try:

            binary = (
                mask.astype(
                    np.uint8
                )
                *
                255
            )

            kernel = np.ones(
                (5, 5),
                np.uint8
            )

            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                kernel
            )

            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:

                area = float(
                    cv2.contourArea(
                        contour
                    )
                )

                if area < 20:
                    continue

                moments = cv2.moments(
                    contour
                )

                if (
                    moments["m00"]
                    == 0
                ):
                    continue

                cx = (
                    moments["m10"]
                    /
                    moments["m00"]
                )

                cy = (
                    moments["m01"]
                    /
                    moments["m00"]
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
                        "area_px": round(
                            area,
                            2
                        )
                    }
                )

            nuclei.sort(
                key=lambda x:
                x["area_px"],
                reverse=True
            )

            nuclei = nuclei[:20]

        except Exception as exc:

            print(
                f"Advertencia núcleos: {exc}"
            )

    if pixels == 0:

        activity = (
            "sin_precipitacion"
        )

    elif coverage < 0.01:

        activity = (
            "precipitacion_debil"
        )

    elif coverage < 0.05:

        activity = (
            "precipitacion_moderada"
        )

    else:

        activity = (
            "precipitacion_extensa"
        )

    return {

        "pixels_precipitacion":
            pixels,

        "cobertura_precipitacion":
            round(
                coverage,
                6
            ),

        "pixeles_intensos":
            strong_pixels,

        "actividad":
            activity,

        "centroide_x":
            centroid_x,

        "centroide_y":
            centroid_y,

        "nucleos":
            nuclei
    }


# ============================================================
# MOVIMIENTO
# ============================================================

def estimate_motion(
    previous_path,
    current_path
):

    result = {

        "direccion":
            "desconocida",

        "velocidad_pixeles_frame":
            0.0,

        "dx":
            0.0,

        "dy":
            0.0
    }

    if cv2 is None:
        return result

    try:

        previous = cv2.imread(
            str(previous_path),
            cv2.IMREAD_GRAYSCALE
        )

        current = cv2.imread(
            str(current_path),
            cv2.IMREAD_GRAYSCALE
        )

        if (
            previous is None
            or
            current is None
        ):
            return result

        previous = cv2.GaussianBlur(
            previous,
            (9, 9),
            0
        )

        current = cv2.GaussianBlur(
            current,
            (9, 9),
            0
        )

        flow = cv2.calcOpticalFlowFarneback(

            previous,
            current,
            None,

            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        fx = flow[:, :, 0]
        fy = flow[:, :, 1]

        magnitude = np.sqrt(
            fx ** 2
            +
            fy ** 2
        )

        valid = (
            magnitude > 0.3
        )

        if not np.any(valid):
            return result

        mean_x = float(
            np.median(
                fx[valid]
            )
        )

        mean_y = float(
            np.median(
                fy[valid]
            )
        )

        speed = math.sqrt(
            mean_x ** 2
            +
            mean_y ** 2
        )

        if (
            abs(mean_x)
            >
            abs(mean_y)
        ):

            direction = (
                "este"
                if mean_x > 0
                else "oeste"
            )

        else:

            direction = (
                "sur"
                if mean_y > 0
                else "norte"
            )

        result = {

            "direccion":
                direction,

            "velocidad_pixeles_frame":
                round(
                    speed,
                    3
                ),

            "dx":
                round(
                    mean_x,
                    3
                ),

            "dy":
                round(
                    mean_y,
                    3
                )
        }

    except Exception as exc:

        print(
            f"Advertencia movimiento: {exc}"
        )

    return result


# ============================================================
# DISTANCIA APROXIMADA
# ============================================================

def pixel_distance_km(
    x,
    y
):

    if (
        x is None
        or
        y is None
    ):
        return None

    center_x = SIZE / 2
    center_y = SIZE / 2

    dx = x - center_x
    dy = y - center_y

    # Aproximación para zoom 7.
    # Se usa solamente como variable relativa
    # hasta calibrar georreferenciación exacta.

    km_per_pixel = 0.305

    return round(
        math.sqrt(
            dx * dx
            +
            dy * dy
        )
        *
        km_per_pixel,
        2
    )


# ============================================================
# FEATURES CSV
# ============================================================

def ensure_features_file():

    if FEATURES.exists():
        return

    with FEATURES.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([

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
            "velocidad_pixeles_frame"
        ])


def append_feature(
    timestamp,
    analysis,
    motion,
    tiles_ok=1
):

    ensure_features_file()

    dt = datetime.fromtimestamp(
        int(timestamp),
        tz=timezone.utc
    )

    distance = pixel_distance_km(
        analysis.get(
            "centroide_x"
        ),
        analysis.get(
            "centroide_y"
        )
    )

    row = [

        int(timestamp),

        dt.isoformat(),

        "RainViewer",

        LAT,
        LON,

        analysis.get(
            "pixels_precipitacion",
            0
        ),

        distance,

        "",

        "",

        "",

        0,

        tiles_ok,

        json.dumps(
            analysis.get(
                "nucleos",
                []
            ),
            ensure_ascii=False,
            separators=(
                ",",
                ":"
            )
        ),

        analysis.get(
            "centroide_x"
        ),

        analysis.get(
            "centroide_y"
        ),

        analysis.get(
            "pixeles_intensos",
            0
        ),

        analysis.get(
            "cobertura_precipitacion",
            0
        ),

        motion.get(
            "dx",
            0
        ),

        motion.get(
            "dy",
            0
        ),

        motion.get(
            "direccion",
            "desconocida"
        ),

        motion.get(
            "velocidad_pixeles_frame",
            0
        )
    ]

    existing = set()

    if FEATURES.exists():

        try:

            with FEATURES.open(
                "r",
                encoding="utf-8"
            ) as file:

                reader = csv.DictReader(
                    file
                )

                for item in reader:

                    try:

                        existing.add(
                            int(
                                float(
                                    item[
                                        "frame_time"
                                    ]
                                )
                            )
                        )

                    except Exception:
                        pass

        except Exception:
            pass

    if int(timestamp) in existing:

        return

    with FEATURES.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            row
        )


# ============================================================
# LIMPIAR FEATURES
# ============================================================

def cleanup_features():

    if not FEATURES.exists():
        return

    try:

        with FEATURES.open(
            "r",
            encoding="utf-8"
        ) as file:

            reader = list(
                csv.DictReader(
                    file
                )
            )

        reader.sort(
            key=lambda row:
            int(
                float(
                    row["frame_time"]
                )
            )
        )

        reader = reader[
            -4320:
        ]

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
            "nucleos",
            "centroide_x",
            "centroide_y",
            "pixeles_intensos",
            "cobertura_precipitacion",
            "movimiento_x",
            "movimiento_y",
            "direccion",
            "velocidad_pixeles_frame"
        ]

        with FEATURES.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                reader
            )

    except Exception as exc:

        print(
            f"Advertencia limpieza CSV: {exc}"
        )


# ============================================================
# NOWCAST
# ============================================================

def build_nowcast(
    analyzed,
    motion
):

    latest = (
        analyzed[-1]
        if analyzed
        else {}
    )

    return {

        "version":
            "2.0",

        "generado_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "fuente":
            "RainViewer",

        "centro": {

            "latitud":
                LAT,

            "longitud":
                LON
        },

        "frames_analizados":
            len(analyzed),

        "ultimo_frame":
            latest,

        "movimiento":
            motion,

        "prediccion": {

            "disponible":
                True,

            "metodo":
                "seguimiento_radar",

            "horizonte_minutos":
                30
        }
    }


# ============================================================
# LIMPIAR TEMPORALES
# ============================================================

def cleanup_temp():

    for file in TEMP.glob(
        "frame_*.png"
    ):

        try:
            file.unlink()
        except Exception:
            pass


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 50
    )

    print(
        "CLIMAAR - RAINVIEWER V2"
    )

    print(
        "=" * 50
    )

    prepare_directories()

    try:

        data, frames = (
            get_api_data()
        )

        host = data[
            "host"
        ]

        selected = frames[
            -FRAMES:
        ]

        downloaded = []

        for frame in selected:

            timestamp = int(
                frame.get(
                    "time",
                    time.time()
                )
            )

            url = frame_url(
                host,
                frame
            )

            print(
                f"Frame {timestamp}"
            )

            try:

                path = download_frame(
                    url,
                    timestamp
                )

                downloaded.append(
                    (
                        frame,
                        path
                    )
                )

            except Exception as exc:

                print(
                    f"Error frame {timestamp}: {exc}"
                )

        if not downloaded:

            raise RuntimeError(
                "No se descargó ningún frame"
            )

        analyzed = []

        for frame, path in downloaded:

            timestamp = int(
                frame.get(
                    "time",
                    time.time()
                )
            )

            analysis = analyze_image(
                path
            )

            analyzed.append({

                "time":
                    timestamp,

                "analisis":
                    analysis
            })

        # ----------------------------------------------------
        # MOVIMIENTO
        # ----------------------------------------------------

        motion = {

            "direccion":
                "desconocida",

            "velocidad_pixeles_frame":
                0.0,

            "dx":
                0.0,

            "dy":
                0.0
        }

        if len(downloaded) >= 2:

            motion = estimate_motion(

                downloaded[-2][1],

                downloaded[-1][1]
            )

        # ----------------------------------------------------
        # HISTÓRICO + FEATURES
        # ----------------------------------------------------

        for index, (
            frame,
            path
        ) in enumerate(
            downloaded
        ):

            timestamp = int(
                frame.get(
                    "time",
                    time.time()
                )
            )

            analysis = analyzed[
                index
            ][
                "analisis"
            ]

            frame_motion = motion

            if index > 0:

                frame_motion = (
                    estimate_motion(
                        downloaded[
                            index - 1
                        ][1],
                        path
                    )
                )

            history_path = (
                HISTORY /
                f"radar_{timestamp}.png"
            )

            shutil.copy2(
                path,
                history_path
            )

            append_feature(

                timestamp,

                analysis,

                frame_motion,

                tiles_ok=1
            )

        # ----------------------------------------------------
        # ACTUAL
        # ----------------------------------------------------

        latest_path = downloaded[
            -1
        ][1]

        shutil.copy2(
            latest_path,
            ACTUAL
        )

        # ----------------------------------------------------
        # NOWCAST
        # ----------------------------------------------------

        nowcast = build_nowcast(
            analyzed,
            motion
        )

        NOWCAST.write_text(

            json.dumps(
                nowcast,
                indent=2,
                ensure_ascii=False
            ),

            encoding="utf-8"
        )

        cleanup_history()

        cleanup_features()

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        status = {

            "ok":
                True,

            "fuente":
                "RainViewer",

            "frames":
                len(downloaded),

            "actual":
                str(ACTUAL),

            "features":
                str(FEATURES),

            "nowcast":
                str(NOWCAST),

            "timestamp_utc":
                datetime.now(
                    timezone.utc
                ).isoformat()
        }

        STATUS.write_text(

            json.dumps(
                status,
                indent=2,
                ensure_ascii=False
            ),

            encoding="utf-8"
        )

        cleanup_temp()

        print("")
        print(
            "CLIMAAR RAINVIEWER OK"
        )

        print(
            f"Frames: {len(downloaded)}"
        )

        print(
            f"Features: {FEATURES}"
        )

        print(
            f"Actual: {ACTUAL}"
        )

        print(
            f"Nowcast: {NOWCAST}"
        )

    except Exception as exc:

        print("")
        print(
            "ERROR RAINVIEWER"
        )

        print(
            str(exc)
        )

        if ACTUAL.exists():

            fallback = {

                "ok":
                    False,

                "fuente":
                    "RainViewer",

                "fallback":
                    True,

                "actual_existente":
                    True,

                "error":
                    str(exc),

                "timestamp_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat()
            }

            NOWCAST.write_text(

                json.dumps(
                    fallback,
                    indent=2,
                    ensure_ascii=False
                ),

                encoding="utf-8"
            )

        else:

            raise


if __name__ == "__main__":
    main()
