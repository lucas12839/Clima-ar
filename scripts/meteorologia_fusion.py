import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests


# ============================================================
# CLIMAAR INTELLIGENCE
# MOTOR DE ADQUISICION Y FUSION METEOROLOGICA
# Version 1.0.1
# ============================================================

VERSION = "1.0.1"

LAT = -38.71
LON = -62.26
CIUDAD = "Bahia Blanca"

BASE_DIR = Path(__file__).resolve().parents[1]

RADAR_DIR = BASE_DIR / "data" / "radar"
SAZB_DIR = BASE_DIR / "data" / "sazb"
IA_DIR = BASE_DIR / "data" / "ia"
METEO_DIR = BASE_DIR / "data" / "meteorologia"

NOWCAST_FILE = RADAR_DIR / "radar_nowcast.json"
SAZB_FILE = SAZB_DIR / "status.json"

FUSION_FILE = IA_DIR / "estado_meteorologico.json"
HISTORICO_FILE = IA_DIR / "historial_meteorologico.csv"
AMBIENTE_FILE = METEO_DIR / "ambiente_futuro.json"

ECMWF_API = "https://api.open-meteo.com/v1/ecmwf"

HEADERS = {
    "User-Agent": f"ClimaAR-Intelligence/{VERSION}",
    "Accept": "application/json",
}

REQUEST_TIMEOUT = 30


# ============================================================
# GRILLA METEOROLOGICA
# ============================================================

GRID = [
    {
        "id": "BB_CENTRO",
        "lat": -38.71,
        "lon": -62.26,
    },

    {
        "id": "BB_N",
        "lat": -38.46,
        "lon": -62.26,
    },

    {
        "id": "BB_NE",
        "lat": -38.46,
        "lon": -62.01,
    },

    {
        "id": "BB_E",
        "lat": -38.71,
        "lon": -62.01,
    },

    {
        "id": "BB_SE",
        "lat": -38.96,
        "lon": -62.01,
    },

    {
        "id": "BB_S",
        "lat": -38.96,
        "lon": -62.26,
    },

    {
        "id": "BB_SO",
        "lat": -38.96,
        "lon": -62.51,
    },

    {
        "id": "BB_O",
        "lat": -38.71,
        "lon": -62.51,
    },

    {
        "id": "BB_NO",
        "lat": -38.46,
        "lon": -62.51,
    },
]


# ============================================================
# VARIABLES ECMWF
#
# Los niveles de viento correctos para ECMWF/Open-Meteo
# son 10 m, 100 m y 200 m.
# ============================================================

HOURLY_VARIABLES = [

    # Superficie
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",

    "pressure_msl",
    "surface_pressure",

    # Precipitacion
    "precipitation",
    "rain",
    "showers",

    # Nubes
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",

    # Visibilidad
    "visibility",

    # Viento
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",

    "wind_speed_100m",
    "wind_direction_100m",

    "wind_speed_200m",
    "wind_direction_200m",

    # Conveccion
    "cape",
    "convective_inhibition",

    # Humedad atmosferica
    "vapour_pressure_deficit",
    "boundary_layer_height",
    "total_column_integrated_water_vapour",

    # Nivel de congelacion
    "freezing_level_height",

    # Actividad electrica modelada
    "lightning_density",
]


# ============================================================
# UTILIDADES
# ============================================================

def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def finite(value):
    if value is None:
        return None

    try:
        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except Exception:
        return None


def load_json(path):
    try:

        if not path.exists():
            return None

        with path.open(
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as exc:

        print(
            f"AVISO: no se pudo leer {path}: {exc}"
        )

        return None


def save_json(path, data):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        )

    temp.replace(path)


def safe_get(
    mapping,
    key,
    default=None
):

    if not isinstance(
        mapping,
        dict
    ):
        return default

    return mapping.get(
        key,
        default
    )


# ============================================================
# RADAR V7
# ============================================================

def cargar_radar():

    data = load_json(
        NOWCAST_FILE
    )

    if not isinstance(
        data,
        dict
    ):

        return {
            "disponible": False,
            "estado": "sin_datos",
        }

    nowcast = data.get(
        "nowcast",
        {}
    )

    if not isinstance(
        nowcast,
        dict
    ):
        nowcast = {}

    return {

        "disponible": True,

        "version": data.get(
            "version"
        ),

        "fuente": data.get(
            "fuente"
        ),

        "actualizado": nowcast.get(
            "actualizado"
        ),

        "frame_actual_utc": nowcast.get(
            "frame_actual_utc"
        ),

        "frames_analizados": nowcast.get(
            "frames_analizados"
        ),

        "tiles_ok": nowcast.get(
            "tiles_ok"
        ),

        "actividad": nowcast.get(
            "actividad"
        ),

        "area_px": nowcast.get(
            "area_px"
        ),

        "distancia_km": nowcast.get(
            "distancia_km"
        ),

        "fortalecimiento": nowcast.get(
            "fortalecimiento"
        ),

        "cambio_area_pct": nowcast.get(
            "cambio_area_pct"
        ),

        "velocidad_kmh": nowcast.get(
            "velocidad_kmh"
        ),

        "direccion": nowcast.get(
            "direccion"
        ),

        "direccion_grados": nowcast.get(
            "direccion_grados"
        ),

        "movimiento_hacia_bahia": nowcast.get(
            "movimiento_hacia_bahia"
        ),

        "eta_minutos": nowcast.get(
            "eta_minutos"
        ),

        "proyecciones": nowcast.get(
            "proyecciones",
            {}
        ),

        "confianza_movimiento": nowcast.get(
            "confianza_movimiento"
        ),

        "nucleos": nowcast.get(
            "nucleos",
            []
        ),

        "nucleo_principal_id": nowcast.get(
            "nucleo_principal_id"
        ),

        "estado": nowcast.get(
            "estado"
        ),
    }


# ============================================================
# SAZB
# ============================================================

def cargar_sazb():

    data = load_json(
        SAZB_FILE
    )

    if not isinstance(
        data,
        dict
    ):

        return {
            "disponible": False,
            "estado": "sin_datos",
        }

    return {

        "disponible": bool(
            data.get(
                "observacion_valida",
                False
            )
        ),

        "fuente": data.get(
            "fuente"
        ),

        "estacion": data.get(
            "estacion"
        ),

        "obsTime_utc": data.get(
            "obsTime_utc"
        ),

        "edad_minutos": data.get(
            "edad_minutos"
        ),

        "temperatura_c": data.get(
            "temperatura_c"
        ),

        "punto_rocio_c": data.get(
            "punto_rocio_c"
        ),

        "viento_kt": data.get(
            "viento_kt"
        ),

        "direccion_viento": data.get(
            "direccion_viento"
        ),

        "presion_hpa": data.get(
            "presion_hpa"
        ),

        "visibilidad_millas": data.get(
            "visibilidad_millas"
        ),
    }


# ============================================================
# ECMWF
# ============================================================

def consultar_ecmwf():

    latitudes = ",".join(
        str(
            punto["lat"]
        )
        for punto in GRID
    )

    longitudes = ",".join(
        str(
            punto["lon"]
        )
        for punto in GRID
    )

    params = {

        "latitude": latitudes,

        "longitude": longitudes,

        "hourly": ",".join(
            HOURLY_VARIABLES
        ),

        "forecast_hours": 12,

        "past_hours": 3,

        "timezone": "UTC",

        "temperature_unit": "celsius",

        "wind_speed_unit": "kmh",

        "precipitation_unit": "mm",

        "cell_selection": "land",
    }

    response = requests.get(
        ECMWF_API,
        params=params,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    if isinstance(
        data,
        dict
    ):

        return [
            data
        ]

    if not isinstance(
        data,
        list
    ):

        raise RuntimeError(
            "ECMWF devolvio un formato inesperado."
        )

    return data


# ============================================================
# NORMALIZACION ECMWF
# ============================================================

def normalizar_punto(
    point,
    grid_point
):

    hourly = point.get(
        "hourly",
        {}
    )

    hourly_units = point.get(
        "hourly_units",
        {}
    )

    times = hourly.get(
        "time",
        []
    )

    registros = []

    for index, timestamp in enumerate(
        times
    ):

        registro = {

            "tiempo_utc": timestamp,

            "punto": grid_point["id"],

            "latitud": grid_point["lat"],

            "longitud": grid_point["lon"],
        }

        for variable in HOURLY_VARIABLES:

            values = hourly.get(
                variable,
                []
            )

            if index < len(
                values
            ):

                value = values[
                    index
                ]

            else:

                value = None

            registro[
                variable
            ] = finite(
                value
            )

        registros.append(
            registro
        )

    return {

        "punto": grid_point["id"],

        "latitud": grid_point["lat"],

        "longitud": grid_point["lon"],

        "timezone": point.get(
            "timezone"
        ),

        "elevacion_m": finite(
            point.get(
                "elevation"
            )
        ),

        "unidades": hourly_units,

        "datos": registros,
    }


# ============================================================
# AMBIENTE FUTURO
# ============================================================

def construir_ambiente(
    ecmwf_points
):

    ambiente = []

    for index, point in enumerate(
        ecmwf_points
    ):

        if index >= len(
            GRID
        ):
            break

        ambiente.append(
            normalizar_punto(
                point,
                GRID[index]
            )
        )

    return ambiente


# ============================================================
# DATO ACTUAL DEL CENTRO
# ============================================================

def obtener_actual_centro(
    ambiente
):

    for punto in ambiente:

        if punto["punto"] != "BB_CENTRO":
            continue

        datos = punto.get(
            "datos",
            []
        )

        if not datos:
            return None

        ahora = datetime.now(
            timezone.utc
        )

        mejor = None
        mejor_diferencia = None

        for registro in datos:

            try:

                dt = datetime.fromisoformat(
                    registro[
                        "tiempo_utc"
                    ].replace(
                        "Z",
                        "+00:00"
                    )
                )

            except Exception:
                continue

            diferencia = abs(
                (
                    dt - ahora
                ).total_seconds()
            )

            if (
                mejor_diferencia is None
                or
                diferencia < mejor_diferencia
            ):

                mejor = registro

                mejor_diferencia = diferencia

        return mejor

    return None


# ============================================================
# CALIDAD DE DATOS
# ============================================================

def calcular_calidad(
    radar,
    sazb,
    ambiente
):

    puntos_validos = len(
        [
            punto
            for punto in ambiente
            if punto.get(
                "datos"
            )
        ]
    )

    radar_ok = (
        radar.get(
            "disponible",
            False
        )
        and
        radar.get(
            "tiles_ok"
        ) == 9
    )

    sazb_ok = sazb.get(
        "disponible",
        False
    )

    score = 0

    if radar_ok:
        score += 40

    if sazb_ok:
        score += 20

    if puntos_validos >= 9:

        score += 40

    elif puntos_validos >= 6:

        score += 30

    elif puntos_validos >= 3:

        score += 15

    score = min(
        100,
        score
    )

    if score >= 85:

        nivel = "alto"

    elif score >= 60:

        nivel = "medio"

    else:

        nivel = "bajo"

    return {

        "score": score,

        "nivel": nivel,

        "radar_valido": radar_ok,

        "sazb_valido": sazb_ok,

        "puntos_ecmwf_validos": puntos_validos,
    }


# ============================================================
# HISTORICO
# ============================================================

HISTORICO_FIELDS = [

    "timestamp_utc",

    "radar_actividad",
    "radar_area_px",
    "radar_distancia_km",
    "radar_fortalecimiento",
    "radar_cambio_area_pct",
    "radar_velocidad_kmh",
    "radar_direccion",
    "radar_direccion_grados",
    "radar_movimiento_hacia_bahia",
    "radar_eta_minutos",
    "radar_confianza_movimiento",

    "sazb_temperatura_c",
    "sazb_punto_rocio_c",
    "sazb_viento_kt",
    "sazb_direccion_viento",
    "sazb_presion_hpa",

    "ecmwf_temperatura_2m",
    "ecmwf_punto_rocio_2m",
    "ecmwf_humedad_2m",

    "ecmwf_cape",
    "ecmwf_cin",

    "ecmwf_viento_10m",
    "ecmwf_direccion_10m",
    "ecmwf_rafaga_10m",

    "ecmwf_viento_100m",
    "ecmwf_direccion_100m",

    "ecmwf_viento_200m",
    "ecmwf_direccion_200m",

    "ecmwf_precipitacion",
    "ecmwf_lluvia",
    "ecmwf_chubascos",

    "ecmwf_nubosidad",

    "ecmwf_agua_precipitable",

    "ecmwf_pbl",

    "ecmwf_nivel_congelacion",

    "ecmwf_lightning_density",

    "calidad_score",
    "calidad_nivel",
]


def append_historico(
    radar,
    sazb,
    actual,
    calidad
):

    HISTORICO_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    existe = HISTORICO_FILE.exists()

    row = {

        "timestamp_utc": utc_now(),

        "radar_actividad": radar.get(
            "actividad"
        ),

        "radar_area_px": radar.get(
            "area_px"
        ),

        "radar_distancia_km": radar.get(
            "distancia_km"
        ),

        "radar_fortalecimiento": radar.get(
            "fortalecimiento"
        ),

        "radar_cambio_area_pct": radar.get(
            "cambio_area_pct"
        ),

        "radar_velocidad_kmh": radar.get(
            "velocidad_kmh"
        ),

        "radar_direccion": radar.get(
            "direccion"
        ),

        "radar_direccion_grados": radar.get(
            "direccion_grados"
        ),

        "radar_movimiento_hacia_bahia": radar.get(
            "movimiento_hacia_bahia"
        ),

        "radar_eta_minutos": radar.get(
            "eta_minutos"
        ),

        "radar_confianza_movimiento": radar.get(
            "confianza_movimiento"
        ),

        "sazb_temperatura_c": sazb.get(
            "temperatura_c"
        ),

        "sazb_punto_rocio_c": sazb.get(
            "punto_rocio_c"
        ),

        "sazb_viento_kt": sazb.get(
            "viento_kt"
        ),

        "sazb_direccion_viento": sazb.get(
            "direccion_viento"
        ),

        "sazb_presion_hpa": sazb.get(
            "presion_hpa"
        ),

        "ecmwf_temperatura_2m": safe_get(
            actual,
            "temperature_2m"
        ),

        "ecmwf_punto_rocio_2m": safe_get(
            actual,
            "dew_point_2m"
        ),

        "ecmwf_humedad_2m": safe_get(
            actual,
            "relative_humidity_2m"
        ),

        "ecmwf_cape": safe_get(
            actual,
            "cape"
        ),

        "ecmwf_cin": safe_get(
            actual,
            "convective_inhibition"
        ),

        "ecmwf_viento_10m": safe_get(
            actual,
            "wind_speed_10m"
        ),

        "ecmwf_direccion_10m": safe_get(
            actual,
            "wind_direction_10m"
        ),

        "ecmwf_rafaga_10m": safe_get(
            actual,
            "wind_gusts_10m"
        ),

        "ecmwf_viento_100m": safe_get(
            actual,
            "wind_speed_100m"
        ),

        "ecmwf_direccion_100m": safe_get(
            actual,
            "wind_direction_100m"
        ),

        "ecmwf_viento_200m": safe_get(
            actual,
            "wind_speed_200m"
        ),

        "ecmwf_direccion_200m": safe_get(
            actual,
            "wind_direction_200m"
        ),

        "ecmwf_precipitacion": safe_get(
            actual,
            "precipitation"
        ),

        "ecmwf_lluvia": safe_get(
            actual,
            "rain"
        ),

        "ecmwf_chubascos": safe_get(
            actual,
            "showers"
        ),

        "ecmwf_nubosidad": safe_get(
            actual,
            "cloud_cover"
        ),

        "ecmwf_agua_precipitable": safe_get(
            actual,
            "total_column_integrated_water_vapour"
        ),

        "ecmwf_pbl": safe_get(
            actual,
            "boundary_layer_height"
        ),

        "ecmwf_nivel_congelacion": safe_get(
            actual,
            "freezing_level_height"
        ),

        "ecmwf_lightning_density": safe_get(
            actual,
            "lightning_density"
        ),

        "calidad_score": calidad.get(
            "score"
        ),

        "calidad_nivel": calidad.get(
            "nivel"
        ),
    }

    with HISTORICO_FILE.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=HISTORICO_FIELDS
        )

        if not existe:

            writer.writeheader()

        writer.writerow(
            row
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "============================================"
    )

    print(
        "ClimaAR Intelligence"
    )

    print(
        "Fusion Meteorologica"
    )

    print(
        f"Version {VERSION}"
    )

    print(
        "============================================"
    )

    # --------------------------------------------------------
    # RADAR
    # --------------------------------------------------------

    radar = cargar_radar()

    print(
        "Radar V7:",
        "OK"
        if radar.get(
            "disponible"
        )
        else
        "SIN DATOS"
    )

    # --------------------------------------------------------
    # SAZB
    # --------------------------------------------------------

    sazb = cargar_sazb()

    print(
        "SAZB:",
        "OK"
        if sazb.get(
            "disponible"
        )
        else
        "SIN DATOS"
    )

    # --------------------------------------------------------
    # ECMWF
    # --------------------------------------------------------

    print(
        "Consultando ECMWF..."
    )

    try:

        ecmwf_raw = consultar_ecmwf()

    except Exception as exc:

        print(
            f"ERROR ECMWF: {exc}"
        )

        return 1

    print(
        f"ECMWF: {len(ecmwf_raw)} puntos recibidos"
    )

    # --------------------------------------------------------
    # AMBIENTE
    # --------------------------------------------------------

    ambiente = construir_ambiente(
        ecmwf_raw
    )

    actual = obtener_actual_centro(
        ambiente
    )

    # --------------------------------------------------------
    # CALIDAD
    # --------------------------------------------------------

    calidad = calcular_calidad(
        radar,
        sazb,
        ambiente
    )

    timestamp = utc_now()

    # --------------------------------------------------------
    # SALIDA PRINCIPAL
    # --------------------------------------------------------

    fusion = {

        "version": VERSION,

        "motor": "ClimaAR Intelligence",

        "actualizado_utc": timestamp,

        "ubicacion": {

            "ciudad": CIUDAD,

            "latitud": LAT,

            "longitud": LON,
        },

        "calidad_datos": calidad,

        "radar": radar,

        "observacion_sazb": sazb,

        "ambiente_actual": actual,

        "ambiente_futuro": {

            "modelo": "ECMWF IFS HRES",

            "ventana_horas": 12,

            "puntos": ambiente,
        },

        "estado": "operativo",

        "proxima_etapa": (
            "Motor IA de evolucion, trayectoria, "
            "fortalecimiento, debilitamiento, ETA, "
            "duracion y peligros."
        ),
    }

    save_json(
        FUSION_FILE,
        fusion
    )

    # --------------------------------------------------------
    # AMBIENTE FUTURO SEPARADO
    # --------------------------------------------------------

    save_json(
        AMBIENTE_FILE,
        {

            "version": VERSION,

            "actualizado_utc": timestamp,

            "modelo": "ECMWF IFS HRES",

            "ventana_horas": 12,

            "puntos": ambiente,
        }
    )

    # --------------------------------------------------------
    # HISTORICO
    # --------------------------------------------------------

    append_historico(
        radar,
        sazb,
        actual,
        calidad
    )

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    print(
        "--------------------------------------------"
    )

    print(
        f"Calidad de datos: "
        f"{calidad['score']}/100 "
        f"({calidad['nivel']})"
    )

    print(
        f"Puntos meteorologicos: "
        f"{calidad['puntos_ecmwf_validos']}/9"
    )

    print(
        f"Salida principal:"
    )

    print(
        FUSION_FILE
    )

    print(
        "Ambiente futuro:"
    )

    print(
        AMBIENTE_FILE
    )

    print(
        "Historico:"
    )

    print(
        HISTORICO_FILE
    )

    print(
        "--------------------------------------------"
    )

    print(
        "FUSION METEOROLOGICA OK"
    )

    return 0


if __name__ == "__main__":

    sys.exit(
        main()
    )
