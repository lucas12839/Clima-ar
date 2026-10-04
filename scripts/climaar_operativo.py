from __future__ import annotations

import json
import math
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
# CLIMAAR 5.0 - MOTOR OPERATIVO
#
# RainViewer + SAZB + modelo historico
#
# IMPORTANTE:
# El modelo historico se usa solamente como contexto
# exploratorio. NO se presenta como probabilidad calibrada.
# ============================================================

LAT = -38.71
LON = -62.26

ZOOM = 7
TILE_SIZE = 512
GRID_RADIUS = 1

DATA_DIR = Path("data/radar")
MODEL_DIR = Path("modelo")

RADAR_IMAGE = DATA_DIR / "actual.png"
STATUS_FILE = DATA_DIR / "status.json"

SAZB_FILE = Path("data/sazb/status.json")

MODEL_FILE = (
    MODEL_DIR /
    "climaar_modelo_historico.joblib"
)

FEATURES_FILE = (
    MODEL_DIR /
    "climaar_features.json"
)

METRICS_FILE = (
    MODEL_DIR /
    "climaar_modelo_historico_metricas.json"
)

RAINVIEWER_API = (
    "https://api.rainviewer.com/public/weather-maps.json"
)

OPEN_METEO_API = (
    "https://api.open-meteo.com/v1/forecast"
)


# ============================================================
# UTILIDADES
# ============================================================

def numero(valor, default=None):

    try:
        x = float(valor)

        if math.isfinite(x):
            return x

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
            - math.asinh(
                math.tan(lat_rad)
            )
            / math.pi
        )
        / 2.0
        * n
    )

    return x, y


# ============================================================
# RADAR RAINVIEWER
# ============================================================

def descargar_radar():

    print("=" * 70)
    print("CLIMAAR - RADAR RAINVIEWER")
    print("=" * 70)

    response = requests.get(
        RAINVIEWER_API,
        timeout=30,
        headers={
            "User-Agent": "ClimaAR/5.0"
        }
    )

    response.raise_for_status()

    data = response.json()

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

    size = (
        TILE_SIZE
        * (GRID_RADIUS * 2 + 1)
    )

    image = Image.new(
        "RGBA",
        (size, size),
        (0, 0, 0, 0)
    )

    tiles_ok = 0

    for dx in range(
        -GRID_RADIUS,
        GRID_RADIUS + 1
    ):

        for dy in range(
            -GRID_RADIUS,
            GRID_RADIUS + 1
        ):

            url = (
                f"{host}"
                f"{path}"
                f"/{TILE_SIZE}"
                f"/{ZOOM}"
                f"/{xt + dx}"
                f"/{yt + dy}"
                f"/2/1_1.png"
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
                    .open(
                        BytesIO(
                            r.content
                        )
                    )
                    .convert("RGBA")
                )

                image.paste(
                    tile,
                    (
                        (dx + GRID_RADIUS)
                        * TILE_SIZE,

                        (dy + GRID_RADIUS)
                        * TILE_SIZE
                    ),
                    tile
                )

                tiles_ok += 1

            except Exception:
                continue

    if tiles_ok == 0:
        raise RuntimeError(
            "No se pudo descargar ninguna "
            "tesela radar."
        )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image.save(
        RADAR_IMAGE
    )

    print(
        f"Radar guardado: {RADAR_IMAGE}"
    )

    print(
        f"Tiles correctas: {tiles_ok}/9"
    )

    return {
        "timestamp": int(timestamp),
        "tiles_ok": tiles_ok,
        "frame_path": path
    }


# ============================================================
# ANALISIS RADAR
# ============================================================

def analizar_radar():

    image = cv2.imread(
        str(RADAR_IMAGE),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise RuntimeError(
            "No se pudo leer actual.png."
        )

    if (
        image.ndim == 3
        and image.shape[2] >= 4
    ):

        alpha = image[:, :, 3]

    else:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        alpha = gray

    mask = (
        alpha > 15
    ).astype(
        np.uint8
    ) * 255

    area = int(
        np.count_nonzero(mask)
    )

    if area == 0:

        return {
            "actividad": False,
            "area_px": 0,
            "distancia_aprox_km": None,
            "estado":
                "sin_precipitacion_detectada"
        }

    ys, xs = np.where(
        mask > 0
    )

    h, w = mask.shape

    cx = w / 2.0
    cy = h / 2.0

    distance_px = float(
        np.min(
            np.sqrt(
                (xs - cx) ** 2
                +
                (ys - cy) ** 2
            )
        )
    )

    # Aproximacion operacional.
    # No es distancia georreferenciada
    # de precision meteorologica.

    distance_km = round(
        distance_px * 0.85,
        1
    )

    return {
        "actividad": True,
        "area_px": area,
        "distancia_aprox_km":
            distance_km,
        "estado":
            "precipitacion_detectada"
    }


# ============================================================
# OBSERVACION SAZB
# ============================================================

def cargar_sazb():

    if not SAZB_FILE.exists():

        return {
            "disponible": False,
            "motivo":
                "No existe "
                "data/sazb/status.json."
        }

    try:

        raw = json.loads(
            SAZB_FILE.read_text(
                encoding="utf-8"
            )
        )

        # Compatibilidad con respuestas
        # que puedan venir envueltas en "content".

        if (
            isinstance(raw, dict)
            and "content" in raw
        ):

            raw = json.loads(
                raw["content"]
            )

        if not raw.get(
            "observacion_valida"
        ):

            return {
                "disponible": False,
                "motivo":
                    "La observacion SAZB "
                    "no es valida."
            }

        return {

            "disponible": True,

            "fuente":
                raw.get("fuente"),

            "estacion":
                raw.get("estacion"),

            "estacion_nombre":
                raw.get(
                    "estacion_nombre"
                ),

            "obsTime_utc":
                raw.get("obsTime_utc"),

            "edad_minutos":
                numero(
                    raw.get(
                        "edad_minutos"
                    )
                ),

            "temperatura_c":
                numero(
                    raw.get(
                        "temperatura_c"
                    )
                ),

            "punto_rocio_c":
                numero(
                    raw.get(
                        "punto_rocio_c"
                    )
                ),

            "viento_kt":
                numero(
                    raw.get(
                        "viento_kt"
                    )
                ),

            "direccion_viento":
                numero(
                    raw.get(
                        "direccion_viento"
                    )
                ),

            "presion_hpa":
                numero(
                    raw.get(
                        "presion_hpa"
                    )
                ),

            "visibilidad_millas":
                raw.get(
                    "visibilidad_millas"
                )
        }

    except Exception as exc:

        return {
            "disponible": False,
            "motivo":
                f"No se pudo leer SAZB: {exc}"
        }


# ============================================================
# MODELO HISTORICO
# ============================================================

def cargar_modelo():

    if not MODEL_FILE.exists():
        return None

    try:

        paquete = joblib.load(
            MODEL_FILE
        )

        if not isinstance(
            paquete,
            dict
        ):
            return None

        if (
            "modelo" not in paquete
            or
            "scaler" not in paquete
        ):
            return None

        return paquete

    except Exception:
        return None


def cargar_features():

    if not FEATURES_FILE.exists():
        return None

    try:

        data = json.loads(
            FEATURES_FILE.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(
            data,
            list
        ):
            return data

    except Exception:
        pass

    return None


# ============================================================
# OPEN-METEO
#
# Se utiliza porque el modelo historico fue entrenado
# con variables meteorologicas equivalentes.
# SAZB sigue siendo la observacion operacional.
# ============================================================

def obtener_openmeteo_horario():

    params = {

        "latitude": LAT,

        "longitude": LON,

        "hourly": ",".join([

            "temperature_2m",

            "relative_humidity_2m",

            "dew_point_2m",

            "pressure_msl",

            "precipitation",

            "wind_speed_10m",

            "wind_direction_10m",

            "wind_gusts_10m"
        ]),

        "past_days": 3,

        "forecast_days": 0,

        "timezone": "UTC"
    }

    response = requests.get(
        OPEN_METEO_API,
        params=params,
        timeout=30,
        headers={
            "User-Agent":
                "ClimaAR/5.0"
        }
    )

    response.raise_for_status()

    data = response.json()

    hourly = data.get(
        "hourly"
    )

    if (
        not hourly
        or "time" not in hourly
    ):

        raise RuntimeError(
            "Open-Meteo no devolvio "
            "datos horarios."
        )

    df = pd.DataFrame(
        hourly
    )

    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce",
        utc=True
    )

    df = (
        df
        .dropna(
            subset=["time"]
        )
        .sort_values("time")
        .reset_index(
            drop=True
        )
    )

    numeric = [

        "temperature_2m",

        "relative_humidity_2m",

        "dew_point_2m",

        "pressure_msl",

        "precipitation",

        "wind_speed_10m",

        "wind_direction_10m",

        "wind_gusts_10m"
    ]

    for col in numeric:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df


# ============================================================
# CONSTRUIR FIRMA DIARIA
# ============================================================

def construir_features_dia(df):

    if df.empty:

        raise RuntimeError(
            "No hay datos meteorologicos."
        )

    df = df.copy()

    df["temp_rocio_dif"] = (
        df["temperature_2m"]
        -
        df["dew_point_2m"]
    )

    rad = np.deg2rad(
        df["wind_direction_10m"]
    )

    df["viento_u"] = (
        df["wind_speed_10m"]
        * np.sin(rad)
    )

    df["viento_v"] = (
        df["wind_speed_10m"]
        * np.cos(rad)
    )

    df["cambio_presion_1h"] = (
        df["pressure_msl"]
        .diff(1)
    )

    df["cambio_presion_3h"] = (
        df["pressure_msl"]
        .diff(3)
    )

    df["cambio_temp_1h"] = (
        df["temperature_2m"]
        .diff(1)
    )

    df["cambio_temp_3h"] = (
        df["temperature_2m"]
        .diff(3)
    )

    df["cambio_humedad_1h"] = (
        df["relative_humidity_2m"]
        .diff(1)
    )

    df["cambio_viento_1h"] = (
        df["wind_speed_10m"]
        .diff(1)
    )

    df["lluvia_3h"] = (
        df["precipitation"]
        .rolling(
            3,
            min_periods=3
        )
        .sum()
    )

    df["lluvia_6h"] = (
        df["precipitation"]
        .rolling(
            6,
            min_periods=6
        )
        .sum()
    )

    df["lluvia_12h"] = (
        df["precipitation"]
        .rolling(
            12,
            min_periods=12
        )
        .sum()
    )

    df["rafaga_max_3h"] = (
        df["wind_gusts_10m"]
        .rolling(
            3,
            min_periods=3
        )
        .max()
    )

    df["rafaga_max_6h"] = (
        df["wind_gusts_10m"]
        .rolling(
            6,
            min_periods=6
        )
        .max()
    )

    df["viento_max_6h"] = (
        df["wind_speed_10m"]
        .rolling(
            6,
            min_periods=6
        )
        .max()
    )

    df["presion_min_6h"] = (
        df["pressure_msl"]
        .rolling(
            6,
            min_periods=6
        )
        .min()
    )

    df["presion_max_6h"] = (
        df["pressure_msl"]
        .rolling(
            6,
            min_periods=6
        )
        .max()
    )

    df["humedad_max_6h"] = (
        df["relative_humidity_2m"]
        .rolling(
            6,
            min_periods=6
        )
        .max()
    )

    df["humedad_min_6h"] = (
        df["relative_humidity_2m"]
        .rolling(
            6,
            min_periods=6
        )
        .min()
    )

    df["fecha"] = (
        df["time"]
        .dt.normalize()
    )

    fecha_actual = df[
        "fecha"
    ].max()

    dia = df[
        df["fecha"]
        ==
        fecha_actual
    ].copy()

    if len(dia) < 12:

        raise RuntimeError(
            "Hay menos de 12 horas disponibles "
            "para evaluar la firma."
        )

    agregaciones = {

        "temperature_2m": [
            "mean",
            "max",
            "min"
        ],

        "relative_humidity_2m": [
            "mean",
            "max",
            "min"
        ],

        "dew_point_2m": [
            "mean",
            "max"
        ],

        "pressure_msl": [
            "mean",
            "min",
            "max"
        ],

        "precipitation": [
            "sum",
            "max"
        ],

        "wind_speed_10m": [
            "mean",
            "max"
        ],

        "wind_gusts_10m": [
            "mean",
            "max"
        ],

        "temp_rocio_dif": [
            "mean",
            "min"
        ],

        "viento_u": [
            "mean",
            "max",
            "min"
        ],

        "viento_v": [
            "mean",
            "max",
            "min"
        ],

        "cambio_presion_1h": [
            "mean",
            "min"
        ],

        "cambio_presion_3h": [
            "mean",
            "min"
        ],

        "cambio_temp_1h": [
            "mean",
            "min",
            "max"
        ],

        "cambio_temp_3h": [
            "mean",
            "min",
            "max"
        ],

        "cambio_humedad_1h": [
            "mean",
            "min",
            "max"
        ],

        "cambio_viento_1h": [
            "mean",
            "max"
        ],

        "lluvia_3h": [
            "max"
        ],

        "lluvia_6h": [
            "max"
        ],

        "lluvia_12h": [
            "max"
        ],

        "rafaga_max_3h": [
            "max"
        ],

        "rafaga_max_6h": [
            "max"
        ],

        "viento_max_6h": [
            "max"
        ],

        "presion_min_6h": [
            "min"
        ],

        "presion_max_6h": [
            "max"
        ],

        "humedad_max_6h": [
            "max"
        ],

        "humedad_min_6h": [
            "min"
        ]
    }

    out = (
        dia
        .agg(agregaciones)
        .to_frame()
        .T
    )

    out.columns = [
        "_".join(
            map(
                str,
                c
            )
        )
        for c in out.columns
    ]

    out["mes"] = int(
        fecha_actual.month
    )

    out["dia_ano"] = int(
        fecha_actual.dayofyear
    )

    return out


# ============================================================
# EVALUAR MODELO HISTORICO
# ============================================================

def evaluar_firma_historica():

    paquete = cargar_modelo()
    features = cargar_features()

    if (
        paquete is None
        or not features
    ):

        return {
            "disponible": False,
            "motivo":
                "Modelo historico o "
                "features no disponibles."
        }

    try:

        df = (
            obtener_openmeteo_horario()
        )

        fila = (
            construir_features_dia(df)
        )

        faltantes = [
            f
            for f in features
            if f not in fila.columns
        ]

        if faltantes:

            return {
                "disponible": False,
                "motivo":
                    "Faltan features: "
                    +
                    ", ".join(
                        faltantes
                    )
            }

        X = (
            fila[features]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

        if X.isna().any().any():

            return {
                "disponible": False,
                "motivo":
                    "La firma actual contiene "
                    "valores faltantes."
            }

        scaler = (
            paquete["scaler"]
        )

        modelo = (
            paquete["modelo"]
        )

        X_scaled = (
            scaler.transform(X)
        )

        probabilidad = float(
            modelo.predict_proba(
                X_scaled
            )[0, 1]
        )

        metricas = {}

        if METRICS_FILE.exists():

            try:

                metricas = json.loads(
                    METRICS_FILE.read_text(
                        encoding="utf-8"
                    )
                )

            except Exception:
                metricas = {}

        fecha_evaluada = (
            df["fecha"]
            .max()
            .date()
        )

        horas = int(
            len(
                df[
                    df["fecha"]
                    ==
                    df["fecha"].max()
                ]
            )
        )

        return {

            "disponible": True,

            "tipo":
                "firma_atmosferica_historica_exploratoria",

            "fecha_evaluada_utc":
                str(
                    fecha_evaluada
                ),

            "horas_disponibles":
                horas,

            "score_exploratorio":
                round(
                    probabilidad,
                    4
                ),

            "umbral_referencia":
                metricas.get(
                    "umbral_referencia"
                ),

            "dias_entrenamiento":
                metricas.get(
                    "dias_totales"
                ),

            "eventos_entrenamiento":
                metricas.get(
                    "eventos_severos_reales"
                ),

            "interpretacion":
                "No es una probabilidad calibrada "
                "ni una alerta. Es una similitud "
                "exploratoria con firmas historicas."
        }

    except Exception as exc:

        return {
            "disponible": False,
            "motivo":
                "No se pudo evaluar la "
                f"firma historica: {exc}"
        }


# ============================================================
# GENERAR ESTADO
# ============================================================

def generar_estado():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    radar = descargar_radar()

    radar_analisis = (
        analizar_radar()
    )

    sazb = cargar_sazb()

    historico = (
        evaluar_firma_historica()
    )

    if radar_analisis[
        "actividad"
    ]:

        estado = (
            "precipitacion_detectada"
        )

    else:

        estado = (
            "sin_actividad_radar"
        )

    resultado = {

        "version": "5.0",

        "actualizado_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "ubicacion": {

            "ciudad":
                "Bahia Blanca",

            "latitud":
                LAT,

            "longitud":
                LON
        },

        "estado":
            estado,

        "radar": {

            "fuente":
                "RainViewer",

            **radar,

            **radar_analisis
        },

        "observacion_sazb":
            sazb,

        "modelo_historico":
            historico,

        "motor_operativo": {

            "radar":
                "RainViewer",

            "observacion":
                "SAZB",

            "contexto_historico":
                "RandomForest historico ClimaAR",

            "nowcast_radar_ia":
                False
        },

        "limitaciones": [

            "El score historico es exploratorio "
            "y no esta calibrado como probabilidad.",

            "El modelo historico fue entrenado "
            "con firmas diarias de eventos catalogados.",

            "La prediccion de movimiento de celulas "
            "y ETA requiere la secuencia temporal "
            "del radar.",

            "La distancia radar actual es aproximada."
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
    print(
        "CLIMAAR OPERATIVO 5.0: OK"
    )
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
