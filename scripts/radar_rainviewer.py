#!/usr/bin/env python3

"""
ClimaAR - RainViewer ingestion
Obtiene radar actualizado, guarda histórico, genera actual.png
y radar_nowcast.json para el motor de inteligencia.
"""

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

LAT = -38.0055
LON = -62.0

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


# ============================================================
# DIRECTORIOS
# ============================================================

def prepare_directories():
    RADAR.mkdir(parents=True, exist_ok=True)
    HISTORY.mkdir(parents=True, exist_ok=True)


# ============================================================
# RAINVIEWER API
# ============================================================

def get_api_data():
    last_error = None

    for attempt in range(3):
        try:
            response = requests.get(
                API,
                timeout=20,
                headers={
                    "Cache-Control": "no-cache",
                    "User-Agent": "ClimaAR/1.0"
                }
            )

            response.raise_for_status()

            data = response.json()

            frames = data.get("radar", {}).get("past", [])

            if not frames:
                raise RuntimeError("RainViewer no devolvió frames de radar")

            return data, frames

        except Exception as exc:
            last_error = exc
            print(f"RainViewer intento {attempt + 1}/3: {exc}")
            time.sleep(2)

    raise RuntimeError(
        f"No se pudo obtener RainViewer: {last_error}"
    )


# ============================================================
# URL DE IMAGEN
# ============================================================

def frame_url(host, frame):
    path = frame.get("path")

    if not path:
        raise RuntimeError("Frame sin path")

    return (
        f"{host}{path}/"
        f"{SIZE}/{ZOOM}/{LAT}/{LON}/"
        f"{COLOR_SCHEME}/{SMOOTH}_{SNOW}.png"
    )


# ============================================================
# DESCARGAR FRAME
# ============================================================

def download_frame(url):
    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "ClimaAR/1.0"
        }
    )

    response.raise_for_status()

    image_path = RADAR / "_latest_rainviewer.png"

    image_path.write_bytes(response.content)

    image = Image.open(image_path).convert("RGBA")

    image.save(image_path, format="PNG")

    return image_path


# ============================================================
# ANÁLISIS BÁSICO DEL RADAR
# ============================================================

def analyze_image(path):
    image = Image.open(path).convert("RGBA")

    rgba = np.array(image)

    rgb = rgba[:, :, :3]

    # Detectamos píxeles con color significativo.
    intensity = rgb.max(axis=2) - rgb.min(axis=2)

    mask = (
        (intensity > 25)
        & (rgb.max(axis=2) > 70)
    )

    pixels = int(mask.sum())

    total = mask.size

    coverage = pixels / total if total else 0.0

    result = {
        "pixels_precipitacion": pixels,
        "cobertura_precipitacion": round(coverage, 6),
        "actividad": "sin_datos"
    }

    if pixels == 0:
        result["actividad"] = "sin_precipitacion"
    elif coverage < 0.01:
        result["actividad"] = "precipitacion_debil"
    elif coverage < 0.05:
        result["actividad"] = "precipitacion_moderada"
    else:
        result["actividad"] = "precipitacion_extensa"

    # --------------------------------------------------------
    # Detección de zonas intensas aproximada
    # --------------------------------------------------------

    strong = (
        (rgb[:, :, 0] > 140)
        | (rgb[:, :, 1] > 140)
        | (rgb[:, :, 2] > 140)
    ) & mask

    strong_pixels = int(strong.sum())

    result["pixeles_intensos"] = strong_pixels

    if strong_pixels > 0:
        result["actividad_intensa"] = True
    else:
        result["actividad_intensa"] = False

    return result


# ============================================================
# ESTIMACIÓN DE MOVIMIENTO
# ============================================================

def estimate_motion(previous_path, current_path):
    result = {
        "direccion": "desconocida",
        "velocidad_pixeles_frame": 0.0
    }

    if cv2 is None:
        return result

    try:
        prev = cv2.imread(str(previous_path), cv2.IMREAD_GRAYSCALE)
        curr = cv2.imread(str(current_path), cv2.IMREAD_GRAYSCALE)

        if prev is None or curr is None:
            return result

        prev = cv2.GaussianBlur(prev, (9, 9), 0)
        curr = cv2.GaussianBlur(curr, (9, 9), 0)

        flow = cv2.calcOpticalFlowFarneback(
            prev,
            curr,
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

        magnitude = np.sqrt(fx ** 2 + fy ** 2)

        valid = magnitude > 0.3

        if not np.any(valid):
            return result

        mean_x = float(np.mean(fx[valid]))
        mean_y = float(np.mean(fy[valid]))

        speed = math.sqrt(mean_x ** 2 + mean_y ** 2)

        if abs(mean_x) > abs(mean_y):
            direction = "este" if mean_x > 0 else "oeste"
        else:
            direction = "sur" if mean_y > 0 else "norte"

        result["direccion"] = direction
        result["velocidad_pixeles_frame"] = round(speed, 3)

    except Exception as exc:
        print(f"Advertencia movimiento: {exc}")

    return result


# ============================================================
# GUARDAR HISTÓRICO
# ============================================================

def save_history(source_path, timestamp):
    output = HISTORY / f"radar_{timestamp}.png"

    shutil.copy2(source_path, output)

    return output


# ============================================================
# LIMPIAR HISTÓRICO
# ============================================================

def cleanup_history():
    files = sorted(
        HISTORY.glob("radar_*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    for old_file in files[MAX_HISTORY:]:
        try:
            old_file.unlink()
        except Exception:
            pass


# ============================================================
# NOWCAST
# ============================================================

def build_nowcast(frames_info, motion):
    latest = frames_info[-1] if frames_info else {}

    return {
        "version": "1.0",
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "fuente": "RainViewer",
        "centro": {
            "latitud": LAT,
            "longitud": LON
        },
        "frames_analizados": len(frames_info),
        "ultimo_frame": latest,
        "movimiento": motion,
        "prediccion": {
            "disponible": True,
            "metodo": "seguimiento_radar",
            "horizonte_minutos": 30
        }
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("==========================================")
    print(" CLIMAAR - RAINVIEWER")
    print("==========================================")

    prepare_directories()

    try:

        data, frames = get_api_data()

        host = data.get("host", "")

        if not host:
            raise RuntimeError("RainViewer no devolvió host")

        # Tomamos los últimos frames disponibles.
        selected = frames[-FRAMES:]

        downloaded = []

        for frame in selected:

            url = frame_url(host, frame)

            print(f"Descargando: {url}")

            try:
                path = download_frame(url)
                downloaded.append((frame, path))

            except Exception as exc:
                print(f"Error descargando frame: {exc}")

        if not downloaded:
            raise RuntimeError(
                "No se pudo descargar ningún frame de RainViewer"
            )

        # ----------------------------------------------------
        # Analizar frames
        # ----------------------------------------------------

        analyzed = []

        for frame, path in downloaded:

            analysis = analyze_image(path)

            timestamp = frame.get("time")

            analyzed.append({
                "time": timestamp,
                "path": str(path),
                "analisis": analysis
            })

        # ----------------------------------------------------
        # Frame actual
        # ----------------------------------------------------

        latest_frame, latest_path = downloaded[-1]

        timestamp = latest_frame.get(
            "time",
            int(time.time())
        )

        history_file = save_history(
            latest_path,
            timestamp
        )

        # ESTE ES EL ARCHIVO QUE NECESITA EL RESTO DE CLIMAAR.
        shutil.copy2(
            latest_path,
            ACTUAL
        )

        print(f"Radar actual: {ACTUAL}")
        print(f"Histórico: {history_file}")

        # ----------------------------------------------------
        # Movimiento
        # ----------------------------------------------------

        motion = {
            "direccion": "desconocida",
            "velocidad_pixeles_frame": 0.0
        }

        if len(downloaded) >= 2:

            previous_path = downloaded[-2][1]

            motion = estimate_motion(
                previous_path,
                latest_path
            )

        # ----------------------------------------------------
        # Nowcast
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

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        status = {
            "ok": True,
            "fuente": "RainViewer",
            "frames": len(downloaded),
            "actual": str(ACTUAL),
            "nowcast": str(NOWCAST),
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat()
        }

        status_file = RADAR / "status.json"

        status_file.write_text(
            json.dumps(
                status,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        print("")
        print("==========================================")
        print(" CLIMAAR RAINVIEWER OK")
        print("==========================================")
        print(f"Frames: {len(downloaded)}")
        print(f"Actual: {ACTUAL}")
        print(f"Nowcast: {NOWCAST}")
        print(f"Movimiento: {motion}")

    except Exception as exc:

        print("")
        print("==========================================")
        print(" ERROR RAINVIEWER")
        print("==========================================")
        print(str(exc))

        # No borramos el radar anterior.
        # Esto permite que ClimaAR conserve el último
        # dato disponible si RainViewer falla.

        if ACTUAL.exists():

            print(
                "Se conserva el radar anterior como respaldo."
            )

            fallback = {
                "ok": False,
                "fuente": "RainViewer",
                "fallback": True,
                "actual_existente": True,
                "error": str(exc),
                "timestamp_utc": datetime.now(
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
