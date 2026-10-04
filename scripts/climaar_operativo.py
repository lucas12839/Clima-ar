from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import requests
from PIL import Image


# ============================================================
# CLIMAAR - MOTOR OPERATIVO
# Radar + firma atmosférica
# ============================================================

LAT = -38.71
LON = -62.26
ZOOM = 7

DATA_DIR = Path("data/radar")
MODEL_DIR = Path("modelo")

DATA_DIR.mkdir(parents=True, exist_ok=True)

RADAR_IMAGE = DATA_DIR / "actual.png"
STATUS_FILE = DATA_DIR / "status.json"

MODEL_FILE = MODEL_DIR / "climaar_modelo_dia_severo.joblib"
MODEL_SUMMARY = MODEL_DIR / "resumen_modelo.json"
MODEL_THRESHOLD = MODEL_DIR / "umbral_operativo.json"


RAINVIEWER_API = (
    "https://api.rainviewer.com/public/weather-maps.json"
)

SMN_BASE = (
    "https://ssl.smn.gob.ar/dpd/descarga_opendata.php"
    "?file=observaciones/datohorario{}.txt"
)


# ============================================================
# UTILIDADES
# ============================================================

def numero(valor, default=None):
    try:
        resultado = float(valor)

        if math.isfinite(resultado):
            return resultado

    except (TypeError, ValueError):
        pass

    return default


def latlon_to_tile(lat, lon, zoom):
    n = 2.0 ** zoom

    x = int(
        (lon + 180.0)
        / 360.0
        * n
    )

    lat_rad = math.radians(lat)

    y = int(
        (
            1.0
            - math.asinh(math.tan(lat_rad))
            / math.pi
        )
        / 2.0
        * n
    )

    return x, y


# ============================================================
# RADAR
# ============================================================

def descargar_radar():

    print("=" * 70)
    print("CLIMAAR - RADAR")
    print("=" * 70)

    try:

        response = requests.get(
            RAINVIEWER_API,
            timeout=30,
            headers={
                "User-Agent": "ClimaAR/3.0"
            }
        )

        response.raise_for_status()

        data = response.json()

    except Exception as exc:

        raise RuntimeError(
            f"No se pudo consultar RainViewer: {exc}"
        )

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

        raise RuntimeError(
            "RainViewer no devolvio frames radar."
        )

    frame = frames[-1]

    path = frame.get("path")

    timestamp = frame.get("time")

    if not path or not timestamp:

        raise RuntimeError(
            "Frame radar invalido."
        )

    xt, yt = latlon_to_tile(
        LAT,
        LON,
        ZOOM
    )

    image = Image.new(
        "RGBA",
        (1536, 1536),
        (0, 0, 0, 0)
    )

    tiles_ok = 0

    for dx in [-1, 0, 1]:

        for dy in [-1, 0, 1]:

            url = (
                f"{host}"
                f"{path}"
                f"/512/{ZOOM}/"
                f"{xt + dx}/"
                f"{yt + dy}/"
                f"2/1_1.png"
            )

            try:

                r = requests.get(
                    url,
                    timeout=20
                )

                if (
                    r.status_code != 200
                    or len(r.content) < 200
                ):
                    continue

                tile = (
                    Image
                    .open(BytesIO(r.content))
                    .convert("RGBA")
                )

                image.paste(
                    tile,
                    (
                        (dx + 1) * 512,
                        (dy + 1) * 512
                    ),
                    tile
                )

                tiles_ok += 1

            except Exception:
                continue

    if tiles_ok == 0:

        raise RuntimeError(
            "No se pudo descargar ninguna tesela radar."
        )

    image.save(RADAR_IMAGE)

    print(
        f"Radar guardado: {RADAR_IMAGE}"
    )

    print(
        f"Tiles descargadas: {tiles_ok}"
    )

    return {
        "timestamp": int(timestamp),
        "host": host,
        "path": path,
        "tiles_ok": tiles_ok
    }


# ============================================================
# ANALISIS RADAR
# ============================================================

def analizar_radar():

    if not RADAR_IMAGE.exists():

        raise RuntimeError(
            "No existe actual.png."
        )

    image = cv2.imread(
        str(RADAR_IMAGE),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:

        raise RuntimeError(
            "No se pudo leer actual.png."
        )

    if image.shape[2] >= 4:

        alpha = image[:, :, 3]

    else:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        alpha = gray

    mask = (
        alpha > 15
    ).astype(np.uint8) * 255

    area = int(
        np.count_nonzero(mask)
    )

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:

        return {
            "actividad": False,
            "area_px": 0,
            "distancia_aprox_km": None,
            "estado": "sin_precipitacion_detectada"
        }

    h, w = mask.shape

    cx = w / 2.0
    cy = h / 2.0

    distances = np.sqrt(
        (xs - cx) ** 2
        +
        (ys - cy) ** 2
    )

    distance_px = float(
        np.min(distances)
    )

    # Aproximacion operacional.
    # No se presenta como distancia georreferenciada
    # de precision meteorologica.

    km_per_pixel = 0.85

    distance_km = (
        distance_px
        * km_per_pixel
    )

    return {
        "actividad": True,
        "area_px": area,
        "distancia_aprox_km":
            round(distance_km, 1),
        "estado":
            "precipitacion_detectada"
    }


# ============================================================
# DATOS SMN
# ============================================================

def obtener_observaciones_smn():

    ahora = datetime.now(
        timezone.utc
    )

    filas = []

    for offset in range(4):

        fecha = (
            ahora.date()
            - pd.Timedelta(days=offset)
        )

        url = SMN_BASE.format(
            fecha.strftime("%Y%m%d")
        )

        try:

            response = requests.get(
                url,
                timeout=30,
                headers={
                    "User-Agent":
                        "ClimaAR/3.0"
                }
            )

            if response.status_code != 200:
                continue

            texto = response.content.decode(
                "latin-1",
                errors="replace"
            )

            for linea in texto.splitlines():

                partes = linea.strip().split()

                if len(partes) < 8:
                    continue

                estacion = " ".join(
                    partes[7:]
                )

                if (
                    "BAHIA BLANCA"
                    not in estacion.upper()
                ):
                    continue

                filas.append(
                    {
                        "fecha":
                            partes[0],
                        "hora":
                            partes[1],
                        "temperatura":
                            numero(partes[2]),
                        "humedad":
                            numero(partes[3]),
                        "presion":
                            numero(partes[4]),
                        "direccion_viento":
                            numero(partes[5]),
                        "velocidad_viento":
                            numero(partes[6])
                    }
                )

        except Exception:

            continue

    if not filas:

        return None

    df = pd.DataFrame(filas)

    df["fecha_hora"] = pd.to_datetime(
        df["fecha"].astype(str)
        + " "
        + df["hora"].astype(str),
        errors="coerce"
    )

    df = (
        df
        .dropna(subset=["fecha_hora"])
        .sort_values("fecha_hora")
        .drop_duplicates(
            "fecha_hora"
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# FIRMA ATMOSFERICA
# ============================================================

def calcular_firma_atmosferica(df):

    if df is None or df.empty:

        return {
            "disponible": False,
            "motivo":
                "Sin observaciones SMN suficientes."
        }

    columnas = [
        "temperatura",
        "humedad",
        "presion",
        "direccion_viento",
        "velocidad_viento"
    ]

    for columna in columnas:

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "temperatura",
            "humedad",
            "presion"
        ]
    )

    if df.empty:

        return {
            "disponible": False,
            "motivo":
                "Observaciones SMN sin valores numericos."
        }

    temperatura = df[
        "temperatura"
    ]

    humedad = df[
        "humedad"
    ]

    presion = df[
        "presion"
    ]

    viento = df[
        "velocidad_viento"
    ]

    firma = {

        "temperatura_media":
            float(temperatura.mean()),

        "temperatura_min":
            float(temperatura.min()),

        "temperatura_max":
            float(temperatura.max()),

        "humedad_media":
            float(humedad.mean()),

        "humedad_min":
            float(humedad.min()),

        "humedad_max":
            float(humedad.max()),

        "presion_media":
            float(presion.mean()),

        "presion_min":
            float(presion.min()),

        "presion_max":
            float(presion.max()),

        "viento_medio":
            float(viento.mean())
            if viento.notna().any()
            else 0.0,

        "rafaga_max":
            float(viento.max())
            if viento.notna().any()
            else 0.0,

        "observaciones":
            int(len(df))
    }

    return {
        "disponible": True,
        "datos": firma
    }


# ============================================================
# MODELO ENTRENADO
# ============================================================

def cargar_modelo():

    if not MODEL_FILE.exists():

        return None

    try:

        return joblib.load(
            MODEL_FILE
        )

    except Exception as exc:

        print(
            f"Advertencia modelo: {exc}"
        )

        return None


def calcular_score_modelo():

    modelo = cargar_modelo()

    if modelo is None:

        return {
            "disponible": False,
            "motivo":
                "Modelo atmosferico no disponible."
        }

    resumen = {}

    if MODEL_SUMMARY.exists():

        try:

            resumen = json.loads(
                MODEL_SUMMARY.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:
            resumen = {}

    umbral = None

    if MODEL_THRESHOLD.exists():

        try:

            threshold_data = json.loads(
                MODEL_THRESHOLD.read_text(
                    encoding="utf-8"
                )
            )

            umbral = numero(
                threshold_data.get(
                    "umbral_exploratorio"
                )
            )

        except Exception:
            pass

    return {
        "disponible": True,
        "tipo":
            modelo.get(
                "tipo",
                "firma_atmosferica"
            ),
        "version":
            modelo.get(
                "version",
                "desconocida"
            ),
        "umbral_exploratorio":
            umbral,
        "dias_entrenamiento":
            resumen.get(
                "dias_totales"
            ),
        "referencias_smn":
            resumen.get(
                "dias_referencia_smn"
            ),
        "estado":
            "modelo_cargado"
    }


# ============================================================
# ESTADO OPERATIVO
# ============================================================

def generar_estado():

    radar = descargar_radar()

    radar_analisis = analizar_radar()

    observaciones = (
        obtener_observaciones_smn()
    )

    firma = calcular_firma_atmosferica(
        observaciones
    )

    modelo = calcular_score_modelo()

    if (
        radar_analisis["actividad"]
        and firma.get("disponible")
    ):

        estado = (
            "actividad_meteorologica_detectada"
        )

    elif radar_analisis["actividad"]:

        estado = (
            "precipitacion_detectada"
        )

    else:

        estado = "sin_actividad_radar"

    resultado = {

        "version": "3.0",

        "actualizado_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "ubicacion": {
            "ciudad":
                "Bahia Blanca",
            "latitud": LAT,
            "longitud": LON
        },

        "estado": estado,

        "radar": {
            "fuente":
                "RainViewer",
            "timestamp":
                radar["timestamp"],
            "tiles_ok":
                radar["tiles_ok"],
            **radar_analisis
        },

        "atmosfera_smn": firma,

        "modelo_atmosferico":
            modelo,

        "criterios_meteorologicos":
            "Las clasificaciones meteorologicas "
            "operativas deben verificarse contra "
            "criterios oficiales del SMN.",

        "limitaciones": [
            "El modelo atmosferico actual "
            "es exploratorio.",

            "Las cinco referencias SMN "
            "no permiten interpretar el "
            "score como probabilidad calibrada.",

            "La distancia radar es aproximada "
            "hasta completar la georreferenciacion "
            "precisa."
        ]
    }

    STATUS_FILE.write_text(
        json.dumps(
            resultado,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        json.dumps(
            resultado,
            indent=2,
            ensure_ascii=False
        )
    )

    print("=" * 70)
    print("CLIMAAR OPERATIVO: OK")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        generar_estado()

    except Exception as exc:

        print(
            f"ERROR OPERATIVO: {exc}"
        )

        sys.exit(1)
