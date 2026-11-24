import csv
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


# ============================================================
# CLIMAAR INTELLIGENCE
# MOTOR DE ADQUISICION Y FUSION METEOROLOGICA
# Version 1.1.0
# ============================================================

VERSION = "1.1.0"

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

# Corto a propósito: si Open-Meteo no responde se usa el último pronóstico
# guardado en vez de alargar el ciclo varios minutos.
REQUEST_TIMEOUT = 20


# ============================================================
# GRILLA METEOROLOGICA
# ============================================================

GRID = [
    {"id": "BB_CENTRO", "lat": -38.71, "lon": -62.26},
    {"id": "BB_N", "lat": -38.46, "lon": -62.26},
    {"id": "BB_NE", "lat": -38.46, "lon": -62.01},
    {"id": "BB_E", "lat": -38.71, "lon": -62.01},
    {"id": "BB_SE", "lat": -38.96, "lon": -62.01},
    {"id": "BB_S", "lat": -38.96, "lon": -62.26},
    {"id": "BB_SO", "lat": -38.96, "lon": -62.51},
    {"id": "BB_O", "lat": -38.71, "lon": -62.51},
    {"id": "BB_NO", "lat": -38.46, "lon": -62.51},
]


# ============================================================
# VARIABLES ECMWF
# ============================================================

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "surface_pressure",
    "precipitation",
    "rain",
    "showers",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "visibility",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_speed_200m",
    "wind_direction_200m",
    "cape",
    "convective_inhibition",
    "vapour_pressure_deficit",
    "boundary_layer_height",
    "total_column_integrated_water_vapour",
    "freezing_level_height",
    "lightning_density",
    # En altura: cizalladura, gradiente térmico y aire seco (ambiente_severo.py).
    # Verificado en Open-Meteo /v1/ecmwf el 19/11/2026.
    "wind_speed_850hPa",
    "wind_direction_850hPa",
    "wind_speed_700hPa",
    "wind_direction_700hPa",
    "wind_speed_500hPa",
    "wind_direction_500hPa",
    "temperature_850hPa",
    "temperature_700hPa",
    "temperature_500hPa",
    "relative_humidity_700hPa",
    "relative_humidity_500hPa",
    # Nivel de congelamiento: /v1/ecmwf no trae freezing_level_height (llega
    # vacío). Se calcula del perfil de temperatura con la altura real de cada
    # nivel (verificado en Open-Meteo /v1/ecmwf el 19/11/2026).
    "temperature_925hPa",
    "temperature_600hPa",
    "geopotential_height_925hPa",
    "geopotential_height_850hPa",
    "geopotential_height_700hPa",
    "geopotential_height_600hPa",
    "geopotential_height_500hPa",
]

NIVELES_PERFIL_HPA = (925, 850, 700, 600, 500)


# ============================================================
# UTILIDADES
# ============================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


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

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as exc:
        print(f"AVISO: no se pudo leer {path}: {exc}")
        return None


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_suffix(path.suffix + ".tmp")

    with temp.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            separators=(",", ":"),  # compacto: 1 MB -> ~0,5 MB por ciclo (auditoría 20/11/2026)
            allow_nan=False,
        )

    temp.replace(path)


def safe_get(mapping, key, default=None):
    if not isinstance(mapping, dict):
        return default

    return mapping.get(key, default)


# ============================================================
# RADAR
# ============================================================

def cargar_radar():
    data = load_json(NOWCAST_FILE)

    if not isinstance(data, dict):
        return {
            "disponible": False,
            "estado": "sin_datos",
        }

    nowcast = data.get("nowcast", {})

    if not isinstance(nowcast, dict):
        nowcast = {}

    return {
        "disponible": True,
        "version": data.get("version"),
        "fuente": data.get("fuente"),
        "actualizado": nowcast.get("actualizado"),
        "frame_actual_utc": nowcast.get("frame_actual_utc"),
        "frames_analizados": nowcast.get("frames_analizados"),
        "tiles_ok": nowcast.get("tiles_ok"),
        "actividad": nowcast.get("actividad"),
        "area_px": nowcast.get("area_px"),
        "distancia_km": nowcast.get("distancia_km"),
        "fortalecimiento": nowcast.get("fortalecimiento"),
        "cambio_area_pct": nowcast.get("cambio_area_pct"),
        "velocidad_kmh": nowcast.get("velocidad_kmh"),
        "direccion": nowcast.get("direccion"),
        "direccion_grados": nowcast.get("direccion_grados"),
        "movimiento_hacia_bahia": nowcast.get(
            "movimiento_hacia_bahia"
        ),
        "eta_minutos": nowcast.get("eta_minutos"),
        "proyecciones": nowcast.get("proyecciones", {}),
        "confianza_movimiento": nowcast.get(
            "confianza_movimiento"
        ),
        "nucleos": nowcast.get("nucleos", []),
        "nucleo_principal_id": nowcast.get(
            "nucleo_principal_id"
        ),
        "estado": nowcast.get("estado"),
    }


# ============================================================
# SAZB
# ============================================================

def cargar_sazb():
    data = load_json(SAZB_FILE)

    if not isinstance(data, dict):
        return {
            "disponible": False,
            "estado": "sin_datos",
        }

    return {
        "disponible": bool(
            data.get("observacion_valida", False)
        ),
        "fuente": data.get("fuente"),
        "estacion": data.get("estacion"),
        "obsTime_utc": data.get("obsTime_utc"),
        "edad_minutos": data.get("edad_minutos"),
        "temperatura_c": data.get("temperatura_c"),
        "punto_rocio_c": data.get("punto_rocio_c"),
        "viento_kt": data.get("viento_kt"),
        "direccion_viento": data.get("direccion_viento"),
        "presion_hpa": data.get("presion_hpa"),
        "visibilidad_millas": data.get("visibilidad_millas"),
    }


# ============================================================
# ECMWF
# ============================================================

def _ecmwf_params(puntos):
    return {
        "latitude": ",".join(
            str(punto["lat"]) for punto in puntos
        ),
        "longitude": ",".join(
            str(punto["lon"]) for punto in puntos
        ),
        "hourly": ",".join(HOURLY_VARIABLES),
        "forecast_hours": 12,
        "past_hours": 3,
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "cell_selection": "land",
    }


def _consultar_ecmwf_grupo(puntos, timeout=REQUEST_TIMEOUT):
    response = requests.get(
        ECMWF_API,
        params=_ecmwf_params(puntos),
        headers=HEADERS,
        timeout=timeout,
    )

    response.raise_for_status()

    data = response.json()

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise RuntimeError(
            "ECMWF devolvio un formato inesperado."
        )

    if len(data) != len(puntos):
        raise RuntimeError(
            "ECMWF devolvio una cantidad inesperada de puntos: "
            f"{len(data)}/{len(puntos)}."
        )

    return data


def consultar_ecmwf():
    try:
        print(
            f"Consultando ECMWF completo: "
            f"{len(GRID)} puntos, timeout={REQUEST_TIMEOUT}s"
        )

        resultado = _consultar_ecmwf_grupo(GRID)

        print(
            f"ECMWF: {len(resultado)} puntos recibidos"
        )

        return resultado

    except Exception as exc:
        print(
            f"AVISO ECMWF consulta completa fallo: {exc}"
        )

    grupos = [
        GRID[0:3],
        GRID[3:6],
        GRID[6:9],
    ]

    resultados = [None] * len(grupos)

    print(
        "ECMWF: usando consulta dividida 3x3"
    )

    with ThreadPoolExecutor(max_workers=3) as executor:
        futuros = {
            executor.submit(
                _consultar_ecmwf_grupo,
                grupo,
                REQUEST_TIMEOUT,
            ): indice
            for indice, grupo in enumerate(grupos)
        }

        for futuro in as_completed(futuros):
            indice = futuros[futuro]

            try:
                resultados[indice] = futuro.result()

                print(
                    f"ECMWF grupo {indice + 1}/3: "
                    f"{len(resultados[indice])} puntos recibidos"
                )

            except Exception as exc:
                print(
                    f"AVISO ECMWF grupo {indice + 1}/3 fallo: "
                    f"{exc}"
                )

    faltan = [i for i, r in enumerate(resultados) if r is None]
    if faltan:
        print(f"ECMWF: un reintento para los grupos {[i + 1 for i in faltan]}")
        with ThreadPoolExecutor(max_workers=3) as executor:
            futuros = {
                executor.submit(_consultar_ecmwf_grupo, grupos[i], REQUEST_TIMEOUT): i
                for i in faltan
            }
            for futuro in as_completed(futuros):
                indice = futuros[futuro]
                try:
                    resultados[indice] = futuro.result()
                except Exception as exc:
                    raise RuntimeError(
                        "ECMWF no pudo obtener el grupo "
                        f"{indice + 1}/3: {exc}"
                    )

    datos = []

    for resultado in resultados:
        datos.extend(resultado)

    if len(datos) != len(GRID):
        raise RuntimeError(
            "ECMWF incompleto: "
            f"{len(datos)}/{len(GRID)} puntos."
        )

    print(
        f"ECMWF: {len(datos)} puntos recibidos "
        "mediante consulta dividida"
    )

    return datos


# ============================================================
# NIVEL DE CONGELAMIENTO (perfil ECMWF)
# ============================================================

def nivel_congelamiento_m(registro, elevacion_m=None):
    """Altura sobre el nivel del mar (m) donde la temperatura cruza 0 °C,
    interpolando entre la superficie y los niveles 925-500 hPa con su altura
    real. None si no hay datos o si el cruce está por encima de 500 hPa."""

    perfil = []
    t2 = finite(registro.get("temperature_2m"))
    if t2 is not None:
        perfil.append(((finite(elevacion_m) or 0.0) + 2.0, t2))
    for nivel in NIVELES_PERFIL_HPA:
        z = finite(registro.get(f"geopotential_height_{nivel}hPa"))
        t = finite(registro.get(f"temperature_{nivel}hPa"))
        if z is None or t is None:
            continue
        if perfil and z <= perfil[0][0]:
            continue  # nivel bajo tierra
        perfil.append((z, t))
    perfil.sort()
    if len(perfil) < 2:
        return None
    if perfil[0][1] <= 0.0:
        return round(perfil[0][0])
    for (z1, t1), (z2, t2_) in zip(perfil, perfil[1:]):
        if t1 > 0.0 >= t2_:
            return round(z1 + (z2 - z1) * t1 / (t1 - t2_))
    return None


# ============================================================
# NORMALIZACION ECMWF
# ============================================================

def normalizar_punto(point, grid_point):
    hourly = point.get("hourly", {})
    hourly_units = point.get("hourly_units", {})
    times = hourly.get("time", [])

    registros = []

    for index, timestamp in enumerate(times):
        registro = {
            "tiempo_utc": timestamp,
            "punto": grid_point["id"],
            "latitud": grid_point["lat"],
            "longitud": grid_point["lon"],
        }

        for variable in HOURLY_VARIABLES:
            values = hourly.get(variable, [])

            if index < len(values):
                value = values[index]
            else:
                value = None

            registro[variable] = finite(value)

        if registro.get("freezing_level_height") is None:
            registro["freezing_level_height"] = nivel_congelamiento_m(
                registro, point.get("elevation")
            )

        registros.append(registro)

    hourly_units = dict(hourly_units or {})
    if not hourly.get("freezing_level_height") or all(
        v is None for v in hourly.get("freezing_level_height", [])
    ):
        hourly_units["freezing_level_height"] = "m (calculado del perfil ECMWF 925-500 hPa)"

    return {
        "punto": grid_point["id"],
        "latitud": grid_point["lat"],
        "longitud": grid_point["lon"],
        "timezone": point.get("timezone"),
        "elevacion_m": finite(point.get("elevation")),
        "unidades": hourly_units,
        "datos": registros,
    }


# ============================================================
# AMBIENTE FUTURO
# ============================================================

def construir_ambiente(ecmwf_points):
    ambiente = []

    for index, point in enumerate(ecmwf_points):
        if index >= len(GRID):
            break

        ambiente.append(
            normalizar_punto(
                point,
                GRID[index],
            )
        )

    return ambiente


# ============================================================
# DATO ACTUAL DEL CENTRO
# ============================================================

def obtener_actual_centro(ambiente):
    for punto in ambiente:
        if punto["punto"] != "BB_CENTRO":
            continue

        datos = punto.get("datos", [])

        if not datos:
            return None

        ahora = datetime.now(timezone.utc)

        mejor = None
        mejor_diferencia = None

        for registro in datos:
            try:
                texto = str(
                    registro["tiempo_utc"]
                ).strip()

                if texto.endswith("Z"):
                    texto = texto[:-1] + "+00:00"

                dt = datetime.fromisoformat(texto)

                if dt.tzinfo is None:
                    dt = dt.replace(
                        tzinfo=timezone.utc
                    )
                else:
                    dt = dt.astimezone(timezone.utc)

            except Exception:
                continue

            diferencia = abs(
                (dt - ahora).total_seconds()
            )

            if (
                mejor_diferencia is None
                or diferencia < mejor_diferencia
            ):
                mejor = registro
                mejor_diferencia = diferencia

        return mejor

    return None


# ============================================================
# FUSION SUPERFICIE ACTUAL: SAZB + ECMWF
# ============================================================

# El informe del aeropuerto (METAR) es horario: con 15 min solo valía ~10 min
# por hora. Misma regla que scripts/estaciones/frescura.py ("sazb").
SAZB_FRESH_MAX_MINUTES = 75


def _parse_utc_timestamp(value):
    if not value:
        return None

    try:
        texto = str(value).strip()

        if texto.endswith("Z"):
            texto = texto[:-1] + "+00:00"

        dt = datetime.fromisoformat(texto)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

        return dt

    except Exception:
        return None


def obtener_edad_sazb_minutos(sazb, now=None):
    # El timestamp de observacion es la fuente de verdad. El campo
    # edad_minutos puede haber quedado cacheado desde una ejecucion anterior.
    obs_time = _parse_utc_timestamp(sazb.get("obsTime_utc"))

    if obs_time is not None:
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)

        return (now - obs_time).total_seconds() / 60.0

    # Sin timestamp verificable no se puede certificar frescura.
    return None


def fusionar_actual_ecmwf_sazb(actual, sazb, now=None):
    """Prioriza SAZB/AWC solo si la observacion tiene 0-75 min (informe horario).

    Una observacion vieja, futura o sin timestamp nunca reemplaza
    los valores actuales de ECMWF.
    """
    resultado = dict(actual)
    edad_minutos = obtener_edad_sazb_minutos(sazb, now=now)
    sazb_disponible = bool(sazb.get("disponible"))
    sazb_fresca = bool(
        sazb_disponible
        and edad_minutos is not None
        and 0.0 <= edad_minutos <= SAZB_FRESH_MAX_MINUTES
    )

    resultado["observacion_sazb_utc"] = sazb.get("obsTime_utc")
    resultado["observacion_sazb_edad_minutos"] = edad_minutos
    resultado["observacion_sazb_fresca"] = sazb_fresca
    resultado["modelo_ecmwf_viento_10m_kmh"] = finite(
        actual.get("wind_speed_10m")
    )
    resultado["modelo_ecmwf_direccion_10m"] = finite(
        actual.get("wind_direction_10m")
    )
    resultado["modelo_ecmwf_rafaga_10m_kmh"] = finite(
        actual.get("wind_gusts_10m")
    )

    if not sazb_fresca:
        resultado["fuente_superficie_actual"] = "ECMWF"
        resultado["estado_observacion_actual"] = (
            "sin_observacion_sazb" if not sazb_disponible
            else "observacion_sazb_vencida"
        )
        resultado["viento_actual_fuente"] = "ECMWF"
        resultado["viento_actual_kmh"] = finite(
            actual.get("wind_speed_10m")
        )
        resultado["direccion_viento_actual"] = finite(
            actual.get("wind_direction_10m")
        )
        return resultado

    temperatura = finite(sazb.get("temperatura_c"))
    if temperatura is not None:
        resultado["temperature_2m"] = temperatura

    punto_rocio = finite(sazb.get("punto_rocio_c"))
    if punto_rocio is not None:
        resultado["dew_point_2m"] = punto_rocio

    presion = finite(sazb.get("presion_hpa"))
    if presion is not None:
        resultado["pressure_msl"] = presion

    viento_kt = finite(sazb.get("viento_kt"))
    viento_kmh = viento_kt * 1.852 if viento_kt is not None else None
    if viento_kmh is not None:
        resultado["wind_speed_10m"] = viento_kmh

    direccion = finite(sazb.get("direccion_viento"))
    if direccion is not None:
        resultado["wind_direction_10m"] = direccion

    resultado["fuente_superficie_actual"] = "SAZB_AWC"
    resultado["estado_observacion_actual"] = "observacion_reciente"
    resultado["viento_actual_fuente"] = "SAZB_AWC"
    resultado["viento_actual_kmh"] = viento_kmh
    resultado["direccion_viento_actual"] = direccion
    return resultado


# ============================================================
# CALIDAD DE DATOS
# ============================================================

def calcular_calidad(radar, sazb, ambiente, now=None):
    """Calcula calidad de fuentes sin confundir disponibilidad con vigencia."""
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    puntos_validos = sum(
        1 for punto in ambiente
        if isinstance(punto, dict) and punto.get("datos")
    )

    tiles_ok = radar.get("tiles_ok")
    radar_ok = (
        radar.get("disponible", False)
        and (tiles_ok is True or tiles_ok == 9)
    )

    sazb_disponible = bool(sazb.get("disponible", False))
    sazb_edad = obtener_edad_sazb_minutos(sazb, now=now)
    sazb_fresco = bool(
        sazb_disponible
        and sazb_edad is not None
        and 0.0 <= sazb_edad <= SAZB_FRESH_MAX_MINUTES
    )

    score = 0
    if radar_ok:
        score += 40
    if sazb_fresco:
        score += 20
    if puntos_validos >= 9:
        score += 40
    elif puntos_validos >= 6:
        score += 30
    elif puntos_validos >= 3:
        score += 15

    score = min(100, score)
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
        "sazb_valido": sazb_fresco,
        "sazb_disponible": sazb_disponible,
        "sazb_fresco": sazb_fresco,
        "sazb_edad_minutos": round(sazb_edad, 1) if sazb_edad is not None else None,
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


def append_historico(radar, sazb, actual, calidad):
    HISTORICO_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existe = HISTORICO_FILE.exists()

    row = {
        "timestamp_utc": utc_now(),
        "radar_actividad": radar.get("actividad"),
        "radar_area_px": radar.get("area_px"),
        "radar_distancia_km": radar.get("distancia_km"),
        "radar_fortalecimiento": radar.get("fortalecimiento"),
        "radar_cambio_area_pct": radar.get("cambio_area_pct"),
        "radar_velocidad_kmh": radar.get("velocidad_kmh"),
        "radar_direccion": radar.get("direccion"),
        "radar_direccion_grados": radar.get(
            "direccion_grados"
        ),
        "radar_movimiento_hacia_bahia": radar.get(
            "movimiento_hacia_bahia"
        ),
        "radar_eta_minutos": radar.get("eta_minutos"),
        "radar_confianza_movimiento": radar.get(
            "confianza_movimiento"
        ),
        "sazb_temperatura_c": sazb.get("temperatura_c"),
        "sazb_punto_rocio_c": sazb.get("punto_rocio_c"),
        "sazb_viento_kt": sazb.get("viento_kt"),
        "sazb_direccion_viento": sazb.get(
            "direccion_viento"
        ),
        "sazb_presion_hpa": sazb.get("presion_hpa"),
        "ecmwf_temperatura_2m": safe_get(
            actual,
            "temperature_2m",
        ),
        "ecmwf_punto_rocio_2m": safe_get(
            actual,
            "dew_point_2m",
        ),
        "ecmwf_humedad_2m": safe_get(
            actual,
            "relative_humidity_2m",
        ),
        "ecmwf_cape": safe_get(
            actual,
            "cape",
        ),
        "ecmwf_cin": safe_get(
            actual,
            "convective_inhibition",
        ),
        "ecmwf_viento_10m": safe_get(
            actual,
            "wind_speed_10m",
        ),
        "ecmwf_direccion_10m": safe_get(
            actual,
            "wind_direction_10m",
        ),
        "ecmwf_rafaga_10m": safe_get(
            actual,
            "wind_gusts_10m",
        ),
        "ecmwf_viento_100m": safe_get(
            actual,
            "wind_speed_100m",
        ),
        "ecmwf_direccion_100m": safe_get(
            actual,
            "wind_direction_100m",
        ),
        "ecmwf_viento_200m": safe_get(
            actual,
            "wind_speed_200m",
        ),
        "ecmwf_direccion_200m": safe_get(
            actual,
            "wind_direction_200m",
        ),
        "ecmwf_precipitacion": safe_get(
            actual,
            "precipitation",
        ),
        "ecmwf_lluvia": safe_get(
            actual,
            "rain",
        ),
        "ecmwf_chubascos": safe_get(
            actual,
            "showers",
        ),
        "ecmwf_nubosidad": safe_get(
            actual,
            "cloud_cover",
        ),
        "ecmwf_agua_precipitable": safe_get(
            actual,
            "total_column_integrated_water_vapour",
        ),
        "ecmwf_pbl": safe_get(
            actual,
            "boundary_layer_height",
        ),
        "ecmwf_nivel_congelacion": safe_get(
            actual,
            "freezing_level_height",
        ),
        "ecmwf_lightning_density": safe_get(
            actual,
            "lightning_density",
        ),
        "calidad_score": calidad.get("score"),
        "calidad_nivel": calidad.get("nivel"),
    }

    with HISTORICO_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=HISTORICO_FIELDS,
        )

        if not existe:
            writer.writeheader()

        writer.writerow(row)

    # Solo los últimos 30 días (antes crecía sin fin).
    try:
        from recorte_historial import recortar_csv
    except ImportError:
        from scripts.recorte_historial import recortar_csv
    recortar_csv(HISTORICO_FILE)


# ============================================================
# MAIN
# ============================================================

# 23/11/2026: 3 h -> 9 h. El pronóstico guardado cubre de 3 h antes a 12 h
# después de bajarlo y se usa la hora que corresponde a "ahora": a las 9 h
# todavía quedan 3 h de pronóstico por delante. Con 3 h, un corte de
# Open-Meteo más largo dejaba sin publicar TODO el ciclo (también el radar).
# Si se usa el respaldo, el control de salud lo avisa.
ECMWF_RESPALDO_MAX_MIN = 540.0

# El pronóstico ECMWF es horario: si el guardado tiene menos de esto, se
# reutiliza sin consultar Open-Meteo (ciclos más rápidos y estables).
ECMWF_CACHE_MIN = 30.0


def cargar_respaldo_ecmwf(motivo):
    """Último ambiente ECMWF guardado, si es reciente. (ambiente, info) o (None, info)."""
    datos = load_json(AMBIENTE_FILE)
    puntos = datos.get("puntos") if isinstance(datos, dict) else None
    texto = str((datos or {}).get("actualizado_utc") or "").strip()
    info = {"usado": False, "motivo": motivo[:200]}
    if not puntos or not texto:
        return None, info
    if texto.endswith("Z"):
        texto = texto[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(texto)
    except ValueError:
        return None, info
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    edad = (datetime.now(timezone.utc) - dt).total_seconds() / 60.0
    info["edad_minutos"] = round(edad, 1)
    info["actualizado_utc"] = (datos or {}).get("actualizado_utc")
    if edad < 0 or edad > ECMWF_RESPALDO_MAX_MIN:
        return None, info
    info["usado"] = True
    return puntos, info


def main():
    print("============================================")
    print("ClimaAR Intelligence")
    print("Fusion Meteorologica")
    print(f"Version {VERSION}")
    print("============================================")

    radar = cargar_radar()

    print(
        "Radar V7:",
        "OK" if radar.get("disponible") else "SIN DATOS",
    )

    principal = radar.get("nucleo_principal_id")
    bahia_principal = radar.get("bahia_blanca", {}).get("nucleo_principal_id")

    # Sin principal solo es válido cuando el radar no tiene actividad.
    if principal is None and radar.get("actividad") is not False:
        raise RuntimeError(
            "ERROR: meteorología recibió un Nowcast sin principal."
        )

    if (
        bahia_principal is not None
        and str(principal) != str(bahia_principal)
    ):
        raise RuntimeError(
            "ERROR: meteorología recibió principales inconsistentes: "
            f"nowcast={principal} bahia={bahia_principal}"
        )

    sazb = cargar_sazb()

    print(
        "SAZB:",
        "OK" if sazb.get("disponible") else "SIN DATOS",
    )

    print("Consultando ECMWF...")

    respaldo_ecmwf = None
    cache_ecmwf = None

    reciente, info_cache = cargar_respaldo_ecmwf("pronóstico reciente")
    if reciente is not None and info_cache.get("edad_minutos", 1e9) < ECMWF_CACHE_MIN:
        ambiente = reciente
        cache_ecmwf = {
            "usado": True,
            "edad_minutos": info_cache["edad_minutos"],
            "actualizado_utc": info_cache.get("actualizado_utc"),
        }
        print(f"ECMWF: se reutiliza el pronóstico guardado ({info_cache['edad_minutos']} min).")

    if cache_ecmwf is None:
        try:
            ecmwf_raw = consultar_ecmwf()

            print(
                f"ECMWF: {len(ecmwf_raw)} puntos recibidos"
            )

            ambiente = construir_ambiente(ecmwf_raw)

        except Exception as exc:
            # Open-Meteo a veces no responde (timeouts). Antes eso cortaba todo
            # el ciclo de Intelligence. Ahora se usa el último pronóstico ECMWF
            # válido (cubre 12 h) si tiene menos de ECMWF_RESPALDO_MAX_MIN.
            print(f"ERROR ECMWF: {exc}")
            ambiente, respaldo_ecmwf = cargar_respaldo_ecmwf(str(exc))
            if ambiente is None:
                print("ERROR: sin pronóstico ECMWF de respaldo utilizable.")
                return 1
            print(
                "AVISO: se usa el pronóstico ECMWF anterior "
                f"({respaldo_ecmwf['edad_minutos']} min)."
            )

    # Modelo en el punto de cada localidad (modelo_localidades.py): se pide
    # solo cuando se renueva el ECMWF; en los ciclos con el guardado se
    # reutiliza el anterior mientras cubra la hora. Nunca corta el ciclo.
    import modelo_localidades as ml
    previo_ml = (load_json(FUSION_FILE) or {}).get("modelo_localidades")
    if cache_ecmwf is None or not ml.vigente(previo_ml):
        modelo_locs = ml.consultar() or (previo_ml if ml.vigente(previo_ml) else None)
    else:
        modelo_locs = previo_ml

    actual_ecmwf = obtener_actual_centro(ambiente)

    if actual_ecmwf is None:
        print(
            "ERROR: no se pudo obtener el punto "
            "meteorologico actual de BB_CENTRO."
        )
        return 1

    actual = fusionar_actual_ecmwf_sazb(
        actual_ecmwf,
        sazb,
    )

    calidad = calcular_calidad(
        radar,
        sazb,
        ambiente,
    )

    timestamp = utc_now()

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
        "estado": "operativo" if respaldo_ecmwf is None else "operativo_con_ecmwf_anterior",
        "ecmwf_respaldo": respaldo_ecmwf or {"usado": False},
        "ecmwf_cache": cache_ecmwf or {"usado": False},
        "modelo_localidades": modelo_locs,
        "proxima_etapa": (
            "Motor IA de evolucion, trayectoria, "
            "fortalecimiento, debilitamiento, ETA, "
            "duracion y peligros."
        ),
    }

    save_json(
        FUSION_FILE,
        fusion,
    )

    # Con el pronóstico de respaldo no se reescribe: conserva su hora real.
    if respaldo_ecmwf is None and cache_ecmwf is None:
        save_json(
            AMBIENTE_FILE,
            {
                "version": VERSION,
                "actualizado_utc": timestamp,
                "modelo": "ECMWF IFS HRES",
                "ventana_horas": 12,
                "puntos": ambiente,
            },
        )

    append_historico(
        radar,
        sazb,
        actual_ecmwf,
        calidad,
    )

    print("--------------------------------------------")
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
        "Superficie actual:",
        actual.get("fuente_superficie_actual"),
        "| estado:",
        actual.get("estado_observacion_actual"),
    )
    print(
        "Viento actual:",
        actual.get("viento_actual_kmh"),
        "km/h | edad SAZB:",
        actual.get("observacion_sazb_edad_minutos"),
        "min",
    )
    print("Salida principal:")
    print(FUSION_FILE)
    print("Ambiente futuro:")
    print(AMBIENTE_FILE)
    print("Historico:")
    print(HISTORICO_FILE)
    print("--------------------------------------------")
    print("FUSION METEOROLOGICA OK")

    return 0


if __name__ == "__main__":
    sys.exit(main())
