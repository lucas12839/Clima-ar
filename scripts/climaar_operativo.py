from __future__ import annotations

import json
import math
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
# RainViewer + Nowcast + SAZB + modelo historico
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
NOWCAST_FILE = DATA_DIR / "radar_nowcast.json"

SAZB_FILE = Path("data/sazb/status.json")

MODEL_FILE = MODEL_DIR / "climaar_modelo_historico.joblib"
FEATURES_FILE = MODEL_DIR / "climaar_features.json"
METRICS_FILE = MODEL_DIR / "climaar_modelo_historico_metricas.json"

RAINVIEWER_API = "https://api.rainviewer.com/public/weather-maps.json"
OPEN_METEO_API = "https://api.open-meteo.com/v1/forecast"


# ============================================================
# UTILIDADES
# ============================================================

def ahora_utc():
    return datetime.now(timezone.utc).isoformat()


def numero(valor, default=None):
    try:
        x = float(valor)
        if math.isfinite(x):
            return x
    except (TypeError, ValueError):
        pass
    return default


def guardar_status(data):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    STATUS_FILE.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


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
            - math.asinh(math.tan(lat_rad)) / math.pi
        )
        / 2.0
        * n
    )

    return x, y


# ============================================================
# RADAR RAINVIEWER
# ============================================================

def descargar_radar():

    response = requests.get(
        RAINVIEWER_API,
        timeout=30,
        headers={
            "User-Agent": "ClimaAR/5.1.0"
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

    if not path or timestamp is None:
        raise RuntimeError(
            "Frame radar invalido."
        )

    xt, yt = latlon_to_tile(
        LAT,
        LON,
        ZOOM
    )

    grid = GRID_RADIUS * 2 + 1

    image = Image.new(
        "RGBA",
        (
            TILE_SIZE * grid,
            TILE_SIZE * grid
        ),
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

            tile_x = xt + dx
            tile_y = yt + dy

            url = (
                f"{host}{path}"
                f"/{TILE_SIZE}"
                f"/{ZOOM}"
                f"/{tile_x}"
                f"/{tile_y}"
                f"/2/1_1.png"
            )

            try:

                r = requests.get(
                    url,
                    timeout=20
                )

                if (
                    r.status_code != 200
                    or len(r.content) < 100
                ):
                    continue

                tile = (
                    Image
                    .open(
                        BytesIO(r.content)
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

            except Exception as exc:

                print(
                    f"Error tile {tile_x}/{tile_y}: {exc}"
                )

    if tiles_ok == 0:
        raise RuntimeError(
            "No se pudo descargar ninguna tesela radar."
        )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image.save(
        RADAR_IMAGE
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
        alpha = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

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

    return {
        "actividad": True,
        "area_px": area,
        "distancia_aprox_km": round(
            distance_px * 0.85,
            1
        ),
        "estado":
            "precipitacion_detectada"
    }


# ============================================================
# NOWCAST RAINVIEWER
# ============================================================

def cargar_nowcast_radar():

    if not NOWCAST_FILE.exists():

        return {
            "disponible": False,
            "motivo":
                "No existe radar_nowcast.json."
        }

    try:

        raw = json.loads(
            NOWCAST_FILE.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            raw,
            dict
        ):

            return {
                "disponible": False,
                "motivo":
                    "radar_nowcast.json no contiene un objeto JSON."
            }

        nowcast = raw.get(
            "nowcast",
            {}
        )

        historial = raw.get(
            "historial",
            {}
        )

        if not isinstance(
            nowcast,
            dict
        ):
            nowcast = {}

        if not isinstance(
            historial,
            dict
        ):
            historial = {}

        return {

            "disponible": True,

            "version":
                raw.get("version"),

            "fuente":
                raw.get(
                    "fuente",
                    "RainViewer"
                ),

            "actualizado":
                nowcast.get(
                    "actualizado"
                ),

            "frame_actual_utc":
                nowcast.get(
                    "frame_actual_utc"
                ),

            "frames_analizados":
                nowcast.get(
                    "frames_analizados"
                ),

            "tiles_ok":
                nowcast.get(
                    "tiles_ok"
                ),

            "actividad":
                nowcast.get(
                    "actividad"
                ),

            "area_px":
                nowcast.get(
                    "area_px"
                ),

            "distancia_km":
                nowcast.get(
                    "distancia_km"
                ),

            "fortalecimiento":
                nowcast.get(
                    "fortalecimiento"
                ),

            "cambio_area_pct":
                nowcast.get(
                    "cambio_area_pct"
                ),

            "velocidad_kmh":
                nowcast.get(
                    "velocidad_kmh"
                ),

            "direccion":
                nowcast.get(
                    "direccion"
                ),

            "direccion_grados":
                nowcast.get(
                    "direccion_grados"
                ),

            "movimiento_hacia_bahia":
                nowcast.get(
                    "movimiento_hacia_bahia"
                ),

            "eta_minutos":
                nowcast.get(
                    "eta_minutos"
                ),

            "proyeccion_30_min":
                nowcast.get(
                    "proyeccion_30_min"
                ),

            "proyeccion_60_min":
                nowcast.get(
                    "proyeccion_60_min"
                ),

            "confianza_movimiento":
                nowcast.get(
                    "confianza_movimiento"
                ),

            "estado":
                nowcast.get(
                    "estado"
                ),

            "historial": {

                "frames_procesados":
                    historial.get(
                        "frames_procesados"
                    ),

                "frames_disponibles":
                    historial.get(
                        "frames_disponibles"
                    ),

                "max_history_frames":
                    historial.get(
                        "max_history_frames"
                    )
            },

            "nota":
                raw.get("nota")
        }

    except Exception as exc:

        return {
            "disponible": False,
            "motivo":
                f"No se pudo leer radar_nowcast.json: {exc}"
        }


# ============================================================
# SAZB
# ============================================================

def cargar_sazb():

    if not SAZB_FILE.exists():

        return {
            "disponible": False,
            "motivo":
                "No existe data/sazb/status.json."
        }

    try:

        raw = json.loads(
            SAZB_FILE.read_text(
                encoding="utf-8"
            )
        )

        if (
            isinstance(raw, dict)
            and isinstance(
                raw.get("content"),
                str
            )
        ):

            raw = json.loads(
                raw["content"]
            )

        if not raw.get(
            "observacion_valida",
            False
        ):

            return {
                "disponible": False,
                "motivo":
                    "La observacion SAZB no es valida."
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
                raw.get(
                    "obsTime_utc"
                ),

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
                numero(
                    raw.get(
                        "visibilidad_millas"
                    )
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

    except Exception as exc:

        print(
            f"Error cargando modelo: {exc}"
        )

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

        if isinstance(
            data,
            dict
        ):

            for key in (
                "features",
                "columnas",
                "feature_names"
            ):

                value = data.get(
                    key
                )

                if isinstance(
                    value,
                    list
                ):
                    return value

    except Exception as exc:

        print(
            f"Error leyendo features: {exc}"
        )

    return None


# ============================================================
# OPEN-METEO
# ============================================================

def obtener_openmeteo_horario():

    params = {

        "latitude":
            LAT,

        "longitude":
            LON,

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

        "past_days":
            3,

        "forecast_days":
            0,

        "timezone":
            "UTC"
    }

    response = requests.get(
        OPEN_METEO_API,
        params=params,
        timeout=30,
        headers={
            "User-Agent":
                "ClimaAR/5.1.0"
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
            "Open-Meteo no devolvio datos horarios."
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
        .reset_index(drop=True)
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

        if col not in df.columns:
            df[col] = np.nan

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df


# ============================================================
# FEATURES
# ============================================================

def preparar_features_horarias(df):

    df = df.copy()

    df["temp_rocio_dif"] = (
        df["temperature_2m"]
        -
        df["dew_point_2m"]
    )

    direccion = np.deg2rad(
        df["wind_direction_10m"]
    )

    df["viento_u"] = (
        df["wind_speed_10m"]
        *
        np.sin(direccion)
    )

    df["viento_v"] = (
        df["wind_speed_10m"]
        *
        np.cos(direccion)
    )

    df["cambio_presion_1h"] = (
        df["pressure_msl"].diff(1)
    )

    df["cambio_presion_3h"] = (
        df["pressure_msl"].diff(3)
    )

    df["cambio_temp_1h"] = (
        df["temperature_2m"].diff(1)
    )

    df["cambio_temp_3h"] = (
        df["temperature_2m"].diff(3)
    )

    df["cambio_humedad_1h"] = (
        df["relative_humidity_2m"].diff(1)
    )

    df["cambio_viento_1h"] = (
        df["wind_speed_10m"].diff(1)
    )

    for horas in (
        3,
        6,
        12,
        24
    ):

        df[f"lluvia_{horas}h"] = (
            df["precipitation"]
            .rolling(
                horas,
                min_periods=1
            )
            .sum()
        )

        df[f"racha_max_{horas}h"] = (
            df["wind_gusts_10m"]
            .rolling(
                horas,
                min_periods=1
            )
            .max()
        )

        df[f"viento_max_{horas}h"] = (
            df["wind_speed_10m"]
            .rolling(
                horas,
                min_periods=1
            )
            .max()
        )

        df[f"presion_min_{horas}h"] = (
            df["pressure_msl"]
            .rolling(
                horas,
                min_periods=1
            )
            .min()
        )

        df[f"presion_max_{horas}h"] = (
            df["pressure_msl"]
            .rolling(
                horas,
                min_periods=1
            )
            .max()
        )

    return df


def construir_features_dia(df):

    if df.empty:
        return pd.DataFrame()

    df = preparar_features_horarias(
        df
    )

    df["fecha"] = (
        df["time"]
        .dt
        .date
    )

    fecha = df["fecha"].max()

    dia = df[
        df["fecha"] == fecha
    ].copy()

    if len(dia) < 12:
        return pd.DataFrame()

    resultado = {}

    for col in dia.columns:

        if col in (
            "time",
            "fecha"
        ):
            continue

        if not pd.api.types.is_numeric_dtype(
            dia[col]
        ):
            continue

        serie = pd.to_numeric(
            dia[col],
            errors="coerce"
        ).dropna()

        if serie.empty:
            continue

        resultado[
            f"{col}_mean"
        ] = float(
            serie.mean()
        )

        resultado[
            f"{col}_max"
        ] = float(
            serie.max()
        )

        resultado[
            f"{col}_min"
        ] = float(
            serie.min()
        )

        resultado[
            f"{col}_std"
        ] = float(
            serie.std()
            if len(serie) > 1
            else 0.0
        )

    resultado["dia"] = fecha.day
    resultado["mes"] = fecha.month
    resultado["año"] = fecha.year
    resultado["dia_semana"] = (
        pd.Timestamp(fecha).dayofweek
    )

    return pd.DataFrame(
        [resultado]
    )


def construir_vector_modelo(
    features_dia,
    features
):

    if features_dia.empty:
        return None

    if not features:
        return None

    vector = pd.DataFrame(
        0.0,
        index=[0],
        columns=features
    )

    for feature in features:

        if feature in features_dia.columns:

            value = features_dia.iloc[0][
                feature
            ]

            vector.loc[0, feature] = numero(
                value,
                0.0
            )

    return vector


# ============================================================
# EVALUACION MODELO
# ============================================================

def evaluar_modelo_historico(
    df_horario
):

    paquete = cargar_modelo()

    if paquete is None:

        return {
            "disponible": False,
            "motivo":
                "No se encontro un modelo historico valido."
        }

    features = cargar_features()

    if not features:
        features = paquete.get(
            "features"
        )

    if not features:

        return {
            "disponible": False,
            "motivo":
                "No se encontraron features del modelo."
        }

    features_dia = construir_features_dia(
        df_horario
    )

    if features_dia.empty:

        return {
            "disponible": False,
            "motivo":
                "No hay suficientes datos horarios para evaluar el modelo."
        }

    vector = construir_vector_modelo(
        features_dia,
        features
    )

    if vector is None:

        return {
            "disponible": False,
            "motivo":
                "No se pudo construir el vector del modelo."
        }

    try:

        modelo = paquete["modelo"]
        scaler = paquete["scaler"]

        X = scaler.transform(
            vector
        )

        clase = int(
            modelo.predict(X)[0]
        )

        resultado = {

            "disponible": True,

            "modelo":
                paquete.get(
                    "version",
                    "historico"
                ),

            "features":
                len(features),

            "clasificacion":
                clase,

            "firma_historica":
                bool(clase),

            "modelo_calibrado":
                False,

            "advertencia":
                "Firma atmosferica historica exploratoria; no es una probabilidad meteorologica inmediata calibrada."
        }

        if hasattr(
            modelo,
            "predict_proba"
        ):

            resultado[
                "probabilidad_modelo"
            ] = float(
                np.max(
                    modelo.predict_proba(X)[0]
                )
            )

        return resultado

    except Exception as exc:

        return {
            "disponible": False,
            "motivo":
                f"Error evaluando modelo historico: {exc}"
        }


# ============================================================
# METRICAS
# ============================================================

def cargar_metricas():

    if not METRICS_FILE.exists():

        return {
            "disponible": False,
            "motivo":
                "No existe archivo de metricas."
        }

    try:

        return {
            "disponible": True,
            "datos": json.loads(
                METRICS_FILE.read_text(
                    encoding="utf-8"
                )
            )
        }

    except Exception as exc:

        return {
            "disponible": False,
            "motivo":
                f"No se pudieron leer las metricas: {exc}"
        }


# ============================================================
# MAIN
# ============================================================

def main():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    status = {

        "app":
            "ClimaAR",

        "version":
            "5.1.0",

        "ubicacion": {

            "latitud":
                LAT,

            "longitud":
                LON,

            "ciudad":
                "Bahia Blanca"
        },

        "fuentes": {

            "radar":
                "RainViewer",

            "observacion":
                "SAZB",

            "meteorologia":
                "Open-Meteo",

            "modelo_historico":
                "ClimaAR"
        },

        "radar":
            {},

        "nowcast_radar":
            {},

        "nowcast_radar_operativo":
            False,

        "nowcast_radar_ia":
            False,

        "sazb":
            {},

        "openmeteo":
            {},

        "modelo_historico":
            {},

        "metricas_modelo":
            {},

        "modelo_historico_es_calibrado":
            False,

        "estado":
            "iniciando",

        "ultima_actualizacion_utc":
            ahora_utc()
    }

    # --------------------------------------------------------
    # RADAR
    # --------------------------------------------------------

    try:

        radar = descargar_radar()

        analisis = analizar_radar()

        status["radar"] = {

            "disponible":
                True,

            "fuente":
                "RainViewer",

            "timestamp":
                radar["timestamp"],

            "tiles_ok":
                radar["tiles_ok"],

            "frame_path":
                radar["frame_path"],

            "actividad":
                analisis["actividad"],

            "area_px":
                analisis["area_px"],

            "distancia_aprox_km":
                analisis["distancia_aprox_km"],

            "estado":
                analisis["estado"]
        }

    except Exception as exc:

        status["radar"] = {

            "disponible":
                False,

            "fuente":
                "RainViewer",

            "motivo":
                str(exc)
        }

        print(
            f"ERROR RADAR: {exc}"
        )

    # --------------------------------------------------------
    # NOWCAST
    # --------------------------------------------------------

    try:

        nowcast = cargar_nowcast_radar()

        status[
            "nowcast_radar"
        ] = nowcast

        status[
            "nowcast_radar_operativo"
        ] = bool(
            nowcast.get(
                "disponible",
                False
            )
        )

    except Exception as exc:

        status[
            "nowcast_radar"
        ] = {

            "disponible":
                False,

            "motivo":
                str(exc)
        }

    # --------------------------------------------------------
    # SAZB
    # --------------------------------------------------------

    try:

        status[
            "sazb"
        ] = cargar_sazb()

    except Exception as exc:

        status[
            "sazb"
        ] = {

            "disponible":
                False,

            "motivo":
                str(exc)
        }

    # --------------------------------------------------------
    # OPEN-METEO + MODELO
    # --------------------------------------------------------

    try:

        df = obtener_openmeteo_horario()

        status[
            "openmeteo"
        ] = {

            "disponible":
                True,

            "registros":
                int(len(df)),

            "ultimo_registro_utc":
                (
                    df["time"]
                    .max()
                    .isoformat()
                    if not df.empty
                    else None
                )
        }

        status[
            "modelo_historico"
        ] = evaluar_modelo_historico(
            df
        )

    except Exception as exc:

        status[
            "openmeteo"
        ] = {

            "disponible":
                False,

            "motivo":
                str(exc)
        }

        status[
            "modelo_historico"
        ] = {

            "disponible":
                False,

            "motivo":
                f"No se pudo evaluar el modelo: {exc}"
        }

    # --------------------------------------------------------
    # METRICAS
    # --------------------------------------------------------

    try:

        status[
            "metricas_modelo"
        ] = cargar_metricas()

    except Exception as exc:

        status[
            "metricas_modelo"
        ] = {

            "disponible":
                False,

            "motivo":
                str(exc)
        }

    # --------------------------------------------------------
    # ESTADO FINAL
    # --------------------------------------------------------

    status[
        "nowcast_radar_ia"
    ] = False

    status[
        "estado"
    ] = "operativo"

    status[
        "ultima_actualizacion_utc"
    ] = ahora_utc()

    guardar_status(
        status
    )

    print(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
