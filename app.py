from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

try:
    from PIL import Image
except Exception:
    Image = None


BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# CONFIGURACIÓN DE DATOS REMOTOS
# ============================================================

REMOTE_DATA_ENABLED = os.getenv(
    "CLIMAAR_REMOTE_DATA_ENABLED",
    "0",
).strip().lower() in {"1", "true", "yes", "on"}

REMOTE_CACHE_SECONDS = float(
    os.getenv(
        "CLIMAAR_REMOTE_CACHE_SECONDS",
        "60",
    )
)

REMOTE_HISTORY_SECONDS = float(
    os.getenv(
        "CLIMAAR_REMOTE_HISTORY_SECONDS",
        "300",
    )
)

GITHUB_REPO = os.getenv(
    "CLIMAAR_GITHUB_REPO",
    "lucas12839/Clima-ar",
).strip()

GITHUB_REF = os.getenv(
    "CLIMAAR_GITHUB_REF",
    "main",
).strip()

GITHUB_RAW_BASE = os.getenv(
    "CLIMAAR_DATA_BASE_URL",
    f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_REF}",
).rstrip("/")

GITHUB_API_BASE = os.getenv(
    "CLIMAAR_GITHUB_API_BASE",
    "https://api.github.com",
).rstrip("/")


# Cuando está activo el modo remoto, NUNCA se escriben encima
# de los archivos reales del repositorio.
DATA_BASE_DIR = (
    BASE_DIR / ".climaar_remote_cache"
    if REMOTE_DATA_ENABLED
    else BASE_DIR
)


RADAR_DIR = DATA_BASE_DIR / "data" / "radar"
HISTORY_DIR = RADAR_DIR / "historico"
IA_DIR = DATA_BASE_DIR / "data" / "ia"


NOWCAST_FILE = RADAR_DIR / "radar_nowcast.json"
RESULTADO_FILE = RADAR_DIR / "resultado_procesamiento.json"
DECISION_FILE = IA_DIR / "resultado_final.json"
STORM_FILE = IA_DIR / "inteligencia_tormenta.json"
METEO_FILE = IA_DIR / "estado_meteorologico.json"
PREDICTION_FILE = (
    RADAR_DIR
    / "seguimiento"
    / "prediccion_evolucion.json"
)
ML_PREDICTION_FILE = (
    RADAR_DIR
    / "seguimiento"
    / "prediccion_ml_rma10.json"
)
TIMESTAMP_FILE = RADAR_DIR / "radar_timestamp.txt"
ACTUAL_IMAGE = RADAR_DIR / "actual.png"


# ============================================================
# CONFIGURACIÓN OPERATIVA
# ============================================================

LAT = -38.7196
LON = -62.2724
CITY = "Bahía Blanca"

VERSION = "8.1-rma10-operativo"
FUENTE = "SINARAME RMA10"

MAX_RADAR_AGE_MINUTES = 40.0
MAX_FUTURE_SKEW_MINUTES = 2.0
MAX_LIVE_PREDICTION_MINUTES = 30.0


# ============================================================
# CACHÉ REMOTO
# ============================================================

_REMOTE_LOCK = threading.RLock()

_REMOTE_LAST_SYNC: dict[str, float] = {}

# -inf: la primera consulta siempre va a GitHub (con 0.0, en un equipo recién
# prendido el reloj puede valer menos de 5 min y la lista quedaba vacía).
_REMOTE_HISTORY_CACHE: tuple[
    float,
    list[str],
] = (float("-inf"), [])


def _cors_origins() -> list[str]:
    raw = os.getenv(
        "CLIMAAR_CORS_ORIGINS",
        "*",
    ).strip()

    if not raw:
        return ["*"]

    return [
        item.strip()
        for item in raw.split(",")
        if item.strip()
    ] or ["*"]


CORS_ORIGINS = _cors_origins()
CORS_WILDCARD = CORS_ORIGINS == ["*"]


app = FastAPI(
    title="ClimaAR - Radar Bahía Blanca",
    description=(
        "ClimaAR basado en SINARAME/RMA10 "
        "para radar operativo."
    ),
    version=VERSION,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=(
        False
        if CORS_WILDCARD
        else True
    ),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GITHUB / CACHÉ
# ============================================================

def _remote_relative(path: Path) -> str | None:
    try:
        return path.relative_to(
            DATA_BASE_DIR
        ).as_posix()
    except ValueError:
        return None


def sync_remote_file(path: Path) -> Path:
    """
    Si el modo remoto está desactivado:
        no hace absolutamente nada.

    Si está activado:
        descarga el archivo desde GitHub cuando corresponde,
        guardándolo en una caché local separada.
    """

    if not REMOTE_DATA_ENABLED:
        return path

    relative = _remote_relative(path)

    if not relative:
        return path

    now = time.monotonic()

    with _REMOTE_LOCK:
        last = _REMOTE_LAST_SYNC.get(
            relative,
            0.0,
        )

        if (
            path.is_file()
            and now - last < REMOTE_CACHE_SECONDS
        ):
            return path

        url = (
            f"{GITHUB_RAW_BASE}/"
            f"{quote(relative, safe='/')}"
        )

        # GitHub Raw/CDN puede conservar una copia anterior del archivo.
        # ClimaAR ya controla su propia caché local, por lo que agregamos
        # una clave que cambia cada minuto para forzar una versión fresca
        # de los datos remotos sin modificar Open-Meteo ni otras peticiones.
        cache_bucket = int(time.time() // 60)

        headers = {
            "User-Agent": "ClimaAR/8.1",
            "Accept": "application/octet-stream",
            "Cache-Control": "no-cache",
        }

        try:
            response = requests.get(
                url,
                params={
                    "climaar_cache": cache_bucket,
                },
                timeout=(5, 10),
                headers=headers,
            )

            if response.status_code == 200:
                path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                tmp = path.with_name(
                    path.name + ".tmp"
                )

                tmp.write_bytes(
                    response.content
                )

                os.replace(
                    tmp,
                    path,
                )

        except Exception:
            # Si GitHub falla, se conserva la
            # última copia válida existente.
            pass

        finally:
            _REMOTE_LAST_SYNC[relative] = now

    return path


def remote_history_names() -> list[str]:
    """
    Obtiene la lista de frames históricos disponibles
    en GitHub.

    Se consulta como máximo cada REMOTE_HISTORY_SECONDS.
    """

    global _REMOTE_HISTORY_CACHE

    if not REMOTE_DATA_ENABLED:
        return []

    now = time.monotonic()

    with _REMOTE_LOCK:
        cached_at, cached_names = (
            _REMOTE_HISTORY_CACHE
        )

        if (
            now - cached_at
            < REMOTE_HISTORY_SECONDS
        ):
            return list(cached_names)

        # Árbol de la carpeta (sin tope práctico). La lista de "contents"
        # corta en 1000 archivos y devolvía solo los MÁS VIEJOS: la
        # animación quedaba vacía (24/11/2026).
        url = (
            f"{GITHUB_API_BASE}/repos/"
            f"{quote(GITHUB_REPO, safe='/')}"
            f"/git/trees/{quote(GITHUB_REF, safe='')}:data/radar/historico"
        )

        params: dict[str, str] = {}

        headers = {
            "User-Agent": "ClimaAR/8.1",
            "Accept": (
                "application/vnd.github+json"
            ),
        }

        names: list[str] = []

        try:
            response = requests.get(
                url,
                params=params,
                timeout=(5, 10),
                headers=headers,
            )

            if response.status_code == 200:
                payload = response.json()

                if isinstance(payload, dict):
                    payload = [
                        {"name": item.get("path")}
                        for item in payload.get("tree") or []
                        if isinstance(item, dict)
                    ]

                if isinstance(
                    payload,
                    list,
                ):
                    for item in payload:
                        if not isinstance(
                            item,
                            dict,
                        ):
                            continue

                        name = item.get(
                            "name"
                        )

                        if (
                            isinstance(
                                name,
                                str,
                            )
                            and name.endswith(
                                ".png"
                            )
                            and len(
                                name[:-4]
                            )
                            == 14
                            and name[:-4].isdigit()
                        ):
                            names.append(name)

                names = sorted(
                    set(names)
                )

        except Exception:
            names = list(
                cached_names
            )

        _REMOTE_HISTORY_CACHE = (
            now,
            names,
        )

    _limpiar_cache_historico()

    return list(names)


# En un servidor que no se reinicia (VPS), las imágenes viejas del radar que
# se bajaron para la animación no deben acumularse en el disco.
CACHE_HISTORICO_HORAS = 6.0


def _limpiar_cache_historico() -> None:
    if not REMOTE_DATA_ENABLED:
        return
    limite = time.time() - CACHE_HISTORICO_HORAS * 3600.0
    try:
        for archivo in HISTORY_DIR.glob("*.png"):
            try:
                if archivo.stat().st_mtime < limite:
                    archivo.unlink()
            except OSError:
                continue
    except OSError:
        return


# ============================================================
# LECTURA DE DATOS
# ============================================================

def read_json(
    path: Path,
) -> dict[str, Any] | None:

    path = sync_remote_file(path)

    try:
        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        ValueError,
        TypeError,
    ):
        return None

    return (
        value
        if isinstance(
            value,
            dict,
        )
        else None
    )


def parse_iso(
    value: Any,
) -> datetime | None:

    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        return None

    text = value.strip()

    if text.endswith("Z"):
        text = (
            text[:-1]
            + "+00:00"
        )

    try:
        dt = datetime.fromisoformat(
            text
        )

    except ValueError:
        return None

    if dt.tzinfo is None:
        return None

    return dt.astimezone(
        timezone.utc
    )


# ============================================================
# RADAR
# ============================================================

def radar_timestamp() -> str | None:

    sync_remote_file(
        TIMESTAMP_FILE
    )

    try:
        value = (
            TIMESTAMP_FILE
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )

    except OSError:
        value = ""

    return value or None


def frame_age_minutes(
    timestamp: str | None,
) -> float | None:

    dt = parse_iso(
        timestamp
    )

    if dt is None:
        return None

    age = (
        datetime.now(
            timezone.utc
        )
        - dt
    ).total_seconds() / 60.0

    return round(
        age,
        2,
    )


def radar_status(
    timestamp: str | None = None,
) -> str:

    timestamp = (
        timestamp
        or radar_timestamp()
    )

    if not timestamp:
        return "no_disponible"

    dt = parse_iso(
        timestamp
    )

    if dt is None:
        return "no_disponible"

    age = frame_age_minutes(
        timestamp
    )

    if (
        age is None
        or age < -MAX_FUTURE_SKEW_MINUTES
    ):
        return "no_disponible"

    if (
        age
        > MAX_RADAR_AGE_MINUTES
    ):
        return "retrasado"

    return "actualizado"


def radar_metadata() -> dict[str, Any]:

    timestamp = radar_timestamp()

    age = frame_age_minutes(
        timestamp
    )

    status = radar_status(
        timestamp
    )

    return {
        "frame_actual_utc": timestamp,
        "frame_edad_minutos": age,
        "estado_radar": status,
        "radar_actualizado": (
            status == "actualizado"
        ),
        "radar_retrasado": (
            status == "retrasado"
        ),
        "radar_disponible": (
            status != "no_disponible"
        ),
        "edad_maxima_procesamiento_minutos": (
            MAX_RADAR_AGE_MINUTES
        ),
    }


def png_is_valid(
    path: Path,
) -> tuple[bool, str]:

    sync_remote_file(path)

    if not path.is_file():
        return (
            False,
            "archivo_no_existe",
        )

    try:
        if path.stat().st_size <= 1024:
            return (
                False,
                "archivo_demasiado_pequeno",
            )

    except OSError:
        return (
            False,
            "no_se_pudo_leer_tamanio",
        )

    if Image is None:
        return (
            False,
            "Pillow_no_disponible",
        )

    try:
        with Image.open(path) as image:

            if image.format != "PNG":
                return (
                    False,
                    "formato_no_png",
                )

            image.verify()

        return (
            True,
            "ok",
        )

    except Exception:
        return (
            False,
            "png_invalido",
        )


# ============================================================
# METEOROLOGÍA EXTERNA
# ============================================================

_CURRENT_CACHE: dict[str, Any] = {
    "ts": 0.0,
    "data": None,
}

CURRENT_CACHE_SECONDS = 300.0


def get_openmeteo_data() -> (
    dict[str, Any] | None
):
    """Tiempo actual. Se guarda 5 min: no hace falta pedirlo en cada visita."""

    now = time.time()

    if (
        _CURRENT_CACHE["data"] is not None
        and now - _CURRENT_CACHE["ts"] < CURRENT_CACHE_SECONDS
    ):
        return _CURRENT_CACHE["data"]

    if now - _CURRENT_CACHE.get("fallo", 0.0) < REINTENTO_MODELO_SEG:
        return _CURRENT_CACHE["data"]

    data = _fetch_openmeteo_current()

    if data is not None:
        _CURRENT_CACHE["ts"] = now
        _CURRENT_CACHE["data"] = data
        return data

    _CURRENT_CACHE["fallo"] = now
    return _CURRENT_CACHE["data"]


def _fetch_openmeteo_current() -> (
    dict[str, Any] | None
):

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={LAT}"
        f"&longitude={LON}"
        "&current="
        "temperature_2m,"
        "dewpoint_2m,"
        "pressure_msl,"
        "wind_speed_10m,"
        "wind_direction_10m,"
        "precipitation,"
        "apparent_temperature,"
        "relative_humidity_2m,"
        "wind_gusts_10m,"
        "cloud_cover,"
        "weather_code"
        "&timezone="
        "America/Argentina/"
        "Buenos_Aires"
    )

    try:
        response = requests.get(
            url,
            timeout=10,
        )

        response.raise_for_status()

        payload = response.json()

        current = payload.get(
            "current"
        )

        if not isinstance(
            current,
            dict,
        ):
            return None

        return {
            "temperatura": current.get(
                "temperature_2m"
            ),
            "punto_rocio": current.get(
                "dewpoint_2m"
            ),
            "presion": current.get(
                "pressure_msl"
            ),
            "viento_vel": current.get(
                "wind_speed_10m"
            ),
            "viento_dir": current.get(
                "wind_direction_10m"
            ),
            "precipitacion": current.get(
                "precipitation"
            ),
            "sensacion_termica": current.get(
                "apparent_temperature"
            ),
            "humedad": current.get(
                "relative_humidity_2m"
            ),
            "rafagas": current.get(
                "wind_gusts_10m"
            ),
            "nubosidad": current.get(
                "cloud_cover"
            ),
            "codigo_tiempo": current.get(
                "weather_code"
            ),
            "fuente": "Open-Meteo",
        }

    except Exception:
        return None


_FORECAST_CACHE: dict[str, Any] = {
    "ts": 0.0,
    "data": None,
}

FORECAST_CACHE_SECONDS = 900.0


def get_openmeteo_forecast() -> (
    list[dict[str, Any]] | None
):
    """Próximos 3 días completos (sin hoy) desde Open-Meteo."""

    now = time.time()

    if (
        _FORECAST_CACHE["data"] is not None
        and now - _FORECAST_CACHE["ts"]
        < FORECAST_CACHE_SECONDS
    ):
        return _FORECAST_CACHE["data"]

    if now - _FORECAST_CACHE.get("fallo", 0.0) < REINTENTO_MODELO_SEG:
        return _FORECAST_CACHE["data"]
    # Si algo falla abajo, se marca el intento (se vuelve a probar al minuto).
    _FORECAST_CACHE["fallo"] = now

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={LAT}"
        f"&longitude={LON}"
        "&daily="
        "weather_code,"
        "temperature_2m_max,"
        "temperature_2m_min,"
        "precipitation_probability_max,"
        "precipitation_sum,"
        "wind_speed_10m_max,"
        "wind_gusts_10m_max,"
        "wind_direction_10m_dominant"
        "&forecast_days=4"
        "&timezone="
        "America/Argentina/"
        "Buenos_Aires"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        daily = response.json().get("daily")

        if not isinstance(daily, dict):
            return _FORECAST_CACHE["data"]

        dates = daily.get("time") or []

        def pick(key: str, i: int) -> Any:
            values = daily.get(key)
            return (
                values[i]
                if isinstance(values, list)
                and i < len(values)
                else None
            )

        days = [
            {
                "fecha": dates[i],
                "codigo_tiempo": pick("weather_code", i),
                "minima": pick("temperature_2m_min", i),
                "maxima": pick("temperature_2m_max", i),
                "prob_lluvia": pick(
                    "precipitation_probability_max", i
                ),
                "lluvia_mm": pick("precipitation_sum", i),
                "viento": pick("wind_speed_10m_max", i),
                "rafagas": pick("wind_gusts_10m_max", i),
                "viento_direccion": pick(
                    "wind_direction_10m_dominant", i
                ),
            }
            # 24/11/2026: incluye hoy (posición 0); /pronostico lo separa.
            for i in range(0, min(4, len(dates)))
        ]

        _FORECAST_CACHE["ts"] = now
        _FORECAST_CACHE["data"] = days
        _FORECAST_CACHE["fallo"] = 0.0

        return days

    except Exception:
        return _FORECAST_CACHE["data"]


# ============================================================
# TIEMPO Y PRONÓSTICO POR LOCALIDAD (24/11/2026)
# ============================================================
# Antes la página mostraba el tiempo y el pronóstico de Bahía Blanca para
# cualquier localidad elegida. Ahora el modelo europeo se pide para el punto
# de cada localidad, en UN solo pedido para las 7 (Open-Meteo acepta varias
# coordenadas juntas), guardado 5 min (actual) y 15 min (pronóstico).

_LOCALES_CACHE: dict[str, dict[str, Any]] = {
    "actual": {"ts": 0.0, "data": None, "fallo": 0.0},
    "pronostico": {"ts": 0.0, "data": None, "fallo": 0.0},
}
# Si Open-Meteo no responde, no se lo vuelve a esperar en cada pedido de la
# página (cada intento puede tardar 10-12 s): se reintenta al minuto.
REINTENTO_MODELO_SEG = 60.0


def _puntos_localidades() -> list[dict[str, Any]]:
    cfg = read_json(LOCALIDADES_FILE) or {}
    return [
        l for l in cfg.get("localidades") or []
        if isinstance(l, dict) and l.get("id") and _num(l.get("lat")) is not None
        and _num(l.get("lon")) is not None
    ]


def _openmeteo_varios(params: str) -> list[tuple[str, dict[str, Any]]] | None:
    puntos = _puntos_localidades()
    if not puntos:
        return None
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=" + ",".join(f"{p['lat']:.4f}" for p in puntos)
        + "&longitude=" + ",".join(f"{p['lon']:.4f}" for p in puntos)
        + params + "&timezone=America/Argentina/Buenos_Aires"
    )
    try:
        response = requests.get(url, timeout=12)
        response.raise_for_status()
        datos = response.json()
    except Exception:
        return None
    # Con un solo punto Open-Meteo devuelve un objeto; con varios, una lista
    # en el mismo orden de las coordenadas.
    lista = datos if isinstance(datos, list) else [datos]
    if len(lista) != len(puntos):
        return None
    return [(p["id"], d if isinstance(d, dict) else {}) for p, d in zip(puntos, lista)]


def _cache_locales(clave: str, segundos: float, armar) -> dict[str, Any]:
    c = _LOCALES_CACHE[clave]
    ahora = time.time()
    if c["data"] is not None and ahora - c["ts"] < segundos:
        return c["data"]
    if ahora - c.get("fallo", 0.0) < REINTENTO_MODELO_SEG:
        return c["data"] or {}
    data = armar()
    if data:
        c["ts"], c["data"] = ahora, data
        return data
    c["fallo"] = ahora
    return c["data"] or {}


def get_openmeteo_locales() -> dict[str, dict[str, Any]]:
    """Tiempo actual del modelo europeo en el punto de cada localidad."""

    def armar():
        res = _openmeteo_varios(
            "&current=temperature_2m,dewpoint_2m,pressure_msl,wind_speed_10m,"
            "wind_direction_10m,precipitation,apparent_temperature,relative_humidity_2m,"
            "wind_gusts_10m,cloud_cover,weather_code"
        )
        if not res:
            return None
        salida = {}
        for ident, d in res:
            c = d.get("current")
            if not isinstance(c, dict):
                continue
            salida[ident] = {
                "temperatura": c.get("temperature_2m"), "punto_rocio": c.get("dewpoint_2m"),
                "presion": c.get("pressure_msl"), "viento_vel": c.get("wind_speed_10m"),
                "viento_dir": c.get("wind_direction_10m"), "precipitacion": c.get("precipitation"),
                "sensacion_termica": c.get("apparent_temperature"),
                "humedad": c.get("relative_humidity_2m"), "rafagas": c.get("wind_gusts_10m"),
                "nubosidad": c.get("cloud_cover"), "codigo_tiempo": c.get("weather_code"),
                "fuente": "Open-Meteo",
            }
        return salida or None

    return _cache_locales("actual", CURRENT_CACHE_SECONDS, armar)


def get_openmeteo_forecast_locales() -> dict[str, list[dict[str, Any]]]:
    """Hoy y los próximos 3 días del modelo europeo en cada localidad."""

    def armar():
        res = _openmeteo_varios(
            "&daily=weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max,precipitation_sum,wind_speed_10m_max,"
            "wind_gusts_10m_max,wind_direction_10m_dominant&forecast_days=4"
        )
        if not res:
            return None
        salida = {}
        for ident, d in res:
            daily = d.get("daily")
            if not isinstance(daily, dict):
                continue
            fechas = daily.get("time") or []

            def pick(k, i):
                v = daily.get(k)
                return v[i] if isinstance(v, list) and i < len(v) else None

            salida[ident] = [
                {
                    "fecha": fechas[i], "codigo_tiempo": pick("weather_code", i),
                    "minima": pick("temperature_2m_min", i), "maxima": pick("temperature_2m_max", i),
                    "prob_lluvia": pick("precipitation_probability_max", i),
                    "lluvia_mm": pick("precipitation_sum", i), "viento": pick("wind_speed_10m_max", i),
                    "rafagas": pick("wind_gusts_10m_max", i),
                    "viento_direccion": pick("wind_direction_10m_dominant", i),
                }
                for i in range(min(4, len(fechas)))
            ]
        return salida or None

    return _cache_locales("pronostico", FORECAST_CACHE_SECONDS, armar)


# ============================================================
# NOWCAST / NÚCLEOS
# ============================================================

def radar_threat_fields(
    nc: dict[str, Any],
) -> dict[str, Any]:
    """
    Campos del núcleo principal / trayectoria hacia Bahía Blanca
    tal como los publica el nowcast, para la interfaz.
    """

    bahia = nc.get("bahia_blanca")

    if not isinstance(bahia, dict):
        bahia = {}

    principal = nc.get("nucleo_principal_id")

    if principal is None:
        principal = bahia.get("nucleo_principal_id")

    keys = (
        "distancia_km",
        "eta_minutos",
        "movimiento_hacia_bahia",
        "direccion_grados",
        "velocidad_kmh",
        "fortalecimiento",
        "cambio_area_pct",
        "confianza_movimiento",
        "rumbo_bahia_grados",
        "error_rumbo_bahia_grados",
        "proyecciones",
    )

    fields = {
        key: nc.get(key)
        for key in keys
    }

    if fields["distancia_km"] is None:
        fields["distancia_km"] = bahia.get(
            "distancia_principal_km"
        )

    return {
        "nucleo_principal_id": principal,
        "track_principal_id": nc.get(
            "track_principal_id",
            principal,
        ),
        "bahia_blanca": bahia,
        **fields,
    }


def get_nowcast() -> dict[str, Any]:
    return (
        read_json(
            NOWCAST_FILE
        )
        or {}
    )


def get_nucleos(
    nowcast: dict[str, Any],
) -> list[dict[str, Any]]:

    block = nowcast.get(
        "nowcast"
    )

    if not isinstance(
        block,
        dict,
    ):
        return []

    value = block.get(
        "nucleos"
    )

    return (
        value
        if isinstance(
            value,
            list,
        )
        else []
    )


def max_intensity(
    nowcast: dict[str, Any],
) -> int | None:

    block = nowcast.get(
        "nowcast"
    )

    if not isinstance(
        block,
        dict,
    ):
        return None

    value = block.get(
        "intensidad_nivel_max"
    )

    if isinstance(
        value,
        (int, float),
    ):
        return int(value)

    values = [
        item.get(
            "intensidad"
        )
        for item in get_nucleos(
            nowcast
        )
        if (
            isinstance(
                item,
                dict,
            )
            and isinstance(
                item.get(
                    "intensidad"
                ),
                (int, float),
            )
        )
    ]

    return (
        int(max(values))
        if values
        else None
    )


def intensity_label(
    level: int | None,
) -> str:

    # Es la intensidad del eco radar (escala RMA10 0-6). NO es la
    # clasificación meteorológica ni la severidad de la tormenta.
    if level is None:
        return "Sin datos"

    if level <= 0:
        return "Sin eco"

    return f"Eco radar nivel {level}/6"


# ============================================================
# HISTORIAL
# ============================================================

def load_history_list() -> (
    list[tuple[str, Path]]
):

    names: set[str] = set()

    if HISTORY_DIR.exists():

        for path in HISTORY_DIR.glob(
            "*.png"
        ):

            if (
                len(path.stem)
                == 14
                and path.stem.isdigit()
            ):
                names.add(
                    path.name
                )

    if REMOTE_DATA_ENABLED:
        names.update(
            remote_history_names()
        )

    items = [
        (
            name[:-4],
            HISTORY_DIR / name,
        )
        for name in names
        if (
            name.endswith(
                ".png"
            )
            and len(
                name[:-4]
            )
            == 14
            and name[:-4].isdigit()
        )
    ]

    items.sort(
        key=lambda item: item[0]
    )

    return items[-13:]


# ============================================================
# SINCRONIZACIÓN RADAR / NOWCAST / ML
# ============================================================

def synchronized_frame_status() -> (
    dict[str, Any]
):

    radar_ts = radar_timestamp()

    nowcast = (
        get_nowcast()
        .get(
            "nowcast",
            {},
        )
    )

    ml = (
        read_json(
            ML_PREDICTION_FILE
        )
        or {}
    )

    nowcast_ts = (
        nowcast.get(
            "frame_actual_utc"
        )
        if isinstance(
            nowcast,
            dict,
        )
        else None
    )

    ml_ts = ml.get(
        "frame_radar"
    )

    radar_dt = parse_iso(
        radar_ts
    )

    nowcast_dt = parse_iso(
        nowcast_ts
    )

    ml_dt = parse_iso(
        ml_ts
    )

    equal = bool(
        radar_dt is not None
        and nowcast_dt is not None
        and ml_dt is not None
        and radar_dt
        == nowcast_dt
        == ml_dt
    )

    return {
        "radar": radar_ts,
        "nowcast": nowcast_ts,
        "ml": ml_ts,
        "sincronizados": equal,
    }


# ============================================================
# INTELIGENCIA
# ============================================================

def sanitize_storm_probabilities(storm: Any) -> Any:
    if not isinstance(storm, dict):
        return storm

    sanitized = dict(storm)
    if sanitized.get("probabilidades_calibradas") is True:
        return sanitized

    probabilities = sanitized.get("probabilidades")
    if isinstance(probabilities, dict):
        sanitized["probabilidades"] = {
            key: None
            for key in probabilities
        }

    hazards = sanitized.get("peligros")
    if isinstance(hazards, dict):
        sanitized["peligros"] = {
            key: None
            for key in hazards
        }

    ai = sanitized.get("ia_v8")
    if isinstance(ai, dict) and "probabilidad_impacto" in ai:
        sanitized["ia_v8"] = dict(ai)
        sanitized["ia_v8"]["probabilidad_impacto"] = None

    return sanitized


def sanitize_ml_probabilities(ml: Any) -> Any:
    if not isinstance(ml, dict):
        return ml

    sanitized = dict(ml)
    calibration = sanitized.get("calibracion_probabilistica", {})
    if not isinstance(calibration, dict):
        calibration = {}

    per_model = calibration.get("por_modelo", {})
    metrics = calibration.get("metricas", {})
    if not isinstance(per_model, dict):
        per_model = {}
    if not isinstance(metrics, dict):
        metrics = {}

    predictions = sanitized.get("predicciones")
    if not isinstance(predictions, list):
        return sanitized

    sanitized_predictions = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            sanitized_predictions.append(prediction)
            continue

        clean_prediction = dict(prediction)
        horizons = clean_prediction.get("horizontes")
        if isinstance(horizons, dict):
            clean_horizons = {}
            for horizon, values in horizons.items():
                if not isinstance(values, dict):
                    clean_horizons[horizon] = values
                    continue

                clean_values = dict(values)
                for target in ("strengthened", "weakened", "toward_bahia"):
                    key = f"{horizon}m_{target}"
                    model_calibration = per_model.get(key)
                    if isinstance(model_calibration, dict):
                        calibrated = model_calibration.get("aplicada") is True
                    else:
                        metric = metrics.get(key, {})
                        calibrated = (
                            isinstance(metric, dict)
                            and metric.get("selection") == "CALIBRATED"
                        )
                    if not calibrated:
                        clean_values[target] = None

                clean_horizons[horizon] = clean_values

            clean_prediction["horizontes"] = clean_horizons

        sanitized_predictions.append(clean_prediction)

    sanitized["predicciones"] = sanitized_predictions
    return sanitized


def sanitize_decision_probabilities(decision: Any) -> Any:
    if not isinstance(decision, dict):
        return decision

    sanitized = dict(decision)
    sanitized["inteligencia"] = sanitize_storm_probabilities(
        sanitized.get("inteligencia")
    )

    summary = sanitized.get("resumen_operativo")
    if isinstance(summary, dict):
        summary = dict(summary)
        for key in (
            "probabilidad_tormenta",
            "probabilidad_llegada_bahia",
            "probabilidad_fortalecimiento",
            "probabilidad_debilitamiento",
            "lluvia_fuerte",
            "viento_fuerte",
            "granizo",
            "rayo",
        ):
            if key in summary:
                summary[key] = None
        sanitized["resumen_operativo"] = summary

    return sanitized


def result_payload() -> (
    dict[str, Any]
):

    storm = sanitize_storm_probabilities(
        read_json(STORM_FILE)
    )

    if storm is not None:

        storm = dict(
            storm
        )

        storm[
            "interpretacion_probabilidades"
        ] = (
            "calibradas"
            if storm.get("probabilidades_calibradas") is True
            else "no_publicadas_sin_calibracion"
        )
        storm["tipo_probabilidades"] = (
            "calibradas"
            if storm.get("probabilidades_calibradas") is True
            else "heuristicas_no_calibradas"
        )

        storm["probabilidades_calibradas"] = (
            storm.get("probabilidades_calibradas") is True
        )

    return {
        "radar_nowcast":
            get_nowcast(),

        "resultado_procesamiento":
            read_json(
                RESULTADO_FILE
            ),

        "resultado_final":
            sanitize_decision_probabilities(
                read_json(DECISION_FILE)
            ),

        "inteligencia_tormenta":
            storm,

        "estado_meteorologico":
            read_json(
                METEO_FILE
            ),

        "prediccion_evolucion":
            read_json(
                PREDICTION_FILE
            ),

        "sincronizacion":
            synchronized_frame_status(),

        **radar_metadata(),
    }


# ============================================================
# API PRINCIPAL
# ============================================================

@app.get("/")
def inicio() -> dict[str, Any]:

    return {
        "app": "ClimaAR",

        "estado": (
            "operativo"
            if radar_status()
            == "actualizado"
            else "degradado"
        ),

        "version": VERSION,

        "fuente_radar":
            FUENTE,

        "ciudad":
            CITY,

        "radar":
            "RMA10",

        **radar_metadata(),
    }


@app.get("/health")
def health() -> dict[str, Any]:

    metadata = radar_metadata()

    required = {
        "timestamp":
            TIMESTAMP_FILE.is_file(),

        "radar_nowcast":
            NOWCAST_FILE.is_file(),

        "imagen":
            ACTUAL_IMAGE.is_file(),
    }

    png_ok, png_reason = (
        png_is_valid(
            ACTUAL_IMAGE
        )
    )

    required[
        "imagen_png_valido"
    ] = png_ok

    status = metadata[
        "estado_radar"
    ]

    if (
        status == "actualizado"
        and all(
            required.values()
        )
    ):
        overall = "ok"

    elif status == "retrasado":
        overall = "degraded"

    else:
        overall = "unavailable"

    return {
        "status": overall,
        "app": "ClimaAR",
        "version": VERSION,
        "fuente": FUENTE,
        "radar": "RMA10",
        "ciudad": CITY,
        "lat": LAT,
        "lon": LON,

        **metadata,

        "archivos":
            required,

        "imagen_validacion":
            png_reason,

        "sincronizacion":
            synchronized_frame_status(),

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }


@app.get("/estado")
def estado() -> dict[str, Any]:

    nowcast = get_nowcast()

    nc = nowcast.get(
        "nowcast",
        {},
    )

    if not isinstance(
        nc,
        dict,
    ):
        nc = {}

    meteo = (
        get_openmeteo_data()
    )

    timestamp = (
        radar_timestamp()
    )

    age = (
        frame_age_minutes(
            timestamp
        )
    )

    level = max_intensity(
        nowcast
    )

    nucleos = get_nucleos(
        nowcast
    )

    speed = nc.get(
        "velocidad_kmh"
    )

    direction = nc.get(
        "direccion"
    )

    movement = (
        "Sin movimiento confirmado"
    )

    if (
        isinstance(
            speed,
            (int, float),
        )
        and speed > 0
    ):

        movement = (
            f"{speed:.1f} km/h"
        )

        if direction:
            movement += (
                f" - {direction}"
            )

    return {
        "temperatura":
            meteo.get(
                "temperatura"
            )
            if meteo
            else None,

        "punto_rocio":
            meteo.get(
                "punto_rocio"
            )
            if meteo
            else None,

        "viento":
            meteo.get(
                "viento_vel"
            )
            if meteo
            else None,

        "viento_direccion":
            meteo.get(
                "viento_dir"
            )
            if meteo
            else None,

        "sensacion_termica":
            meteo.get("sensacion_termica")
            if meteo
            else None,

        "humedad":
            meteo.get("humedad")
            if meteo
            else None,

        "rafagas":
            meteo.get("rafagas")
            if meteo
            else None,

        "nubosidad":
            meteo.get("nubosidad")
            if meteo
            else None,

        "codigo_tiempo":
            meteo.get("codigo_tiempo")
            if meteo
            else None,

        "presion":
            meteo.get(
                "presion"
            )
            if meteo
            else None,

        "precipitacion":
            meteo.get(
                "precipitacion"
            )
            if meteo
            else None,

        "meteorologia_externa": (
            "disponible"
            if meteo
            else "no_disponible"
        ),

        "cobertura_radar":
            bool(nucleos),

        "intensidad_nivel":
            level,

        "intensidad":
            intensity_label(
                level
            ),

        "nucleos":
            len(nucleos),

        "actualizado":
            timestamp,

        "frame_edad_minutos":
            age,

        "estado_radar":
            radar_status(
                timestamp
            ),

        "nowcast":
            nc.get(
                "estado",
                nowcast.get(
                    "estado",
                    "sin datos",
                ),
            ),

        "movimiento":
            movement,

        "movimiento_hacia_bahia":
            nc.get(
                "movimiento_hacia_bahia"
            ),

        "eta_minutos":
            nc.get(
                "eta_minutos"
            ),

        "fuente":
            FUENTE,

        "version":
            VERSION,
    }


def radar_geo() -> dict[str, Any]:
    """Georreferenciación real publicada por Intelligence (RMA10)."""

    storm = read_json(STORM_FILE) or {}
    block = storm.get("radar")

    if not isinstance(block, dict):
        return {"bounds": None, "nucleos_geo": {}}

    geo = block.get("georreferenciacion")
    bounds = None

    try:
        if (
            isinstance(geo, dict)
            and geo.get("disponible") is True
            and (geo.get("validacion") or {}).get("estado") is True
        ):
            c = geo["validacion"]["controles_cardinales"]
            bounds = [
                [float(c["sur"]["latitud"]),
                 float(c["oeste"]["longitud"])],
                [float(c["norte"]["latitud"]),
                 float(c["este"]["longitud"])],
            ]
    except (KeyError, TypeError, ValueError):
        bounds = None

    nucleos_geo: dict[str, Any] = {}

    for item in block.get("nucleos_georreferenciados") or []:
        if (
            isinstance(item, dict)
            and isinstance(item.get("georreferencia"), dict)
            and item.get("track_id") is not None
        ):
            nucleos_geo[str(item["track_id"])] = item["georreferencia"]

    return {
        "bounds": bounds,
        "frame": block.get("frame_actual_utc"),
        "centro": (
            [geo.get("radar_latitud"), geo.get("radar_longitud")]
            if isinstance(geo, dict)
            else None
        ),
        "alcance_km": (
            geo.get("alcance_radio_km")
            if isinstance(geo, dict)
            else None
        ),
        "nucleos_geo": nucleos_geo,
    }


_TZ_AR = timezone(timedelta(hours=-3))   # Argentina, sin horario de verano


@app.get("/pronostico")
def pronostico() -> dict[str, Any]:

    days = get_openmeteo_forecast() or []
    hoy_fecha = datetime.now(_TZ_AR).date().isoformat()

    def separar(lista):
        return (next((d for d in lista if d.get("fecha") == hoy_fecha), None),
                [d for d in lista if d.get("fecha") != hoy_fecha][:3])

    hoy, dias = separar(days)
    por_localidad = {}
    for ident, lista in (get_openmeteo_forecast_locales() or {}).items():
        h, ds = separar(lista)
        por_localidad[ident] = {"hoy": h, "dias": ds}

    return {
        "disponible": bool(days),
        "fuente": "Open-Meteo",
        # Hoy aparte (lo usan la portada y la publicación de la mañana);
        # "dias" sigue siendo los próximos 3 días, sin hoy. Esto es Bahía Blanca.
        "hoy": hoy,
        "dias": dias,
        # El mismo pronóstico para el punto de cada localidad (24/11/2026).
        "por_localidad": por_localidad,
    }


@app.get("/radar")
def radar() -> dict[str, Any]:

    nowcast = get_nowcast()

    nc = nowcast.get(
        "nowcast",
        {},
    )

    if not isinstance(
        nc,
        dict,
    ):
        nc = {}

    metadata = (
        radar_metadata()
    )

    geo = radar_geo()

    # Solo se georreferencia si Intelligence analizó el mismo frame.
    geo_vigente = (
        geo.get("frame") is not None
        and geo.get("frame") == nc.get("frame_actual_utc")
    )

    nucleos = []

    for item in get_nucleos(nowcast):
        item = dict(item) if isinstance(item, dict) else item

        if geo_vigente and isinstance(item, dict):
            g = geo["nucleos_geo"].get(
                str(item.get("track_id"))
            )

            if g:
                item["georreferencia"] = g

        nucleos.append(item)

    return {
        "estado":
            "ok"
            if nowcast
            else "sin_datos",

        "bounds":
            geo["bounds"],

        "georreferenciacion": {
            "bounds": geo["bounds"],
            "centro": geo["centro"],
            "alcance_km": geo["alcance_km"],
            "fuente": "inteligencia_tormenta.radar.georreferenciacion",
        },

        "fuente":
            FUENTE,

        "radar":
            "RMA10",

        **metadata,

        "imagen_local":
            "/radar/imagen",

        "frames_historicos":
            len(
                load_history_list()
            ),

        "frames_analizados":
            nc.get(
                "frames_analizados"
            ),

        "frames_analizados_significado":
            (
                "valor generado por "
                "el motor de nowcast; "
                "no asumir que representa "
                "cantidad de imágenes "
                "sin auditar su productor"
            ),

        "nucleos":
            nucleos,

        **radar_threat_fields(
            nc
        ),

        "intensidad_nivel_max":
            max_intensity(
                nowcast
            ),

        "intensidad":
            intensity_label(
                max_intensity(
                    nowcast
                )
            ),

        "sincronizacion":
            synchronized_frame_status(),
    }


@app.get("/radar/imagen")
def radar_imagen():

    valid, reason = (
        png_is_valid(
            ACTUAL_IMAGE
        )
    )

    if not valid:

        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error":
                    "imagen_radar_no_disponible",
                "motivo":
                    reason,
                **radar_metadata(),
            },
        )

    return FileResponse(
        ACTUAL_IMAGE,
        media_type="image/png",
        filename="actual.png",
    )


@app.get("/secuencia")
def secuencia() -> dict[str, Any]:

    frames = (
        load_history_list()
    )

    return {
        "ok": True,

        "cantidad":
            len(frames),

        "frames": [
            {
                "timestamp":
                    timestamp,

                "imagen":
                    (
                        "/radar/historico/"
                        f"{timestamp}"
                    ),
            }

            for timestamp, _
            in frames
        ],

        "frame_actual_utc":
            radar_timestamp(),

        "estado_radar":
            radar_status(),
    }


@app.get(
    "/radar/historico/{timestamp}"
)
def radar_historico(
    timestamp: str,
):

    if (
        len(timestamp) != 14
        or not timestamp.isdigit()
    ):

        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error":
                    "timestamp_invalido",
            },
        )

    path = (
        HISTORY_DIR
        / f"{timestamp}.png"
    )

    valid, reason = (
        png_is_valid(path)
    )

    if not valid:

        return JSONResponse(
            status_code=404,
            content={
                "ok": False,
                "error":
                    "frame_no_disponible",
                "motivo":
                    reason,
                "timestamp":
                    timestamp,
            },
        )

    return FileResponse(
        path,
        media_type="image/png",
        filename=(
            f"{timestamp}.png"
        ),
    )


# ============================================================
# NOWCAST
# ============================================================

@app.get("/nowcast")
def nowcast_endpoint() -> (
    dict[str, Any]
):

    data = get_nowcast()

    return {
        "ok": bool(data),
        "datos": data,
        **radar_metadata(),
    }


# ============================================================
# PREDICCIÓN
# ============================================================

@app.get("/prediccion")
def prediccion() -> (
    dict[str, Any]
):

    prediction = read_json(
        PREDICTION_FILE
    )

    return {
        "ok":
            prediction is not None,

        "datos":
            prediction,

        **radar_metadata(),
    }


@app.get("/prediccion/en-vivo")
def prediccion_en_vivo() -> (
    dict[str, Any]
):

    prediction = read_json(
        PREDICTION_FILE
    )

    radar_ts = (
        radar_timestamp()
    )

    if not isinstance(
        prediction,
        dict,
    ):

        return {
            "ok": False,
            "modo":
                "SIN_PREDICCION",

            "es_observacion_radar":
                True,

            "es_prediccion":
                False,

            "datos":
                None,

            **radar_metadata(),
        }

    # prediccion_evolucion.json solo trae radar_timestamp (el frame para el
    # que se calculó); sin este respaldo el endpoint nunca era válido.
    prediction_ts = (
        prediction.get(
            "frame_prediccion"
        )
        or prediction.get(
            "timestamp"
        )
        or prediction.get(
            "actualizado_utc"
        )
        or prediction.get(
            "radar_timestamp"
        )
    )

    frame_radar = (
        prediction.get(
            "frame_radar"
        )
        or radar_ts
    )

    radar_dt = parse_iso(
        frame_radar
    )

    prediction_dt = parse_iso(
        prediction_ts
    )

    age = None

    if (
        radar_dt is not None
        and prediction_dt is not None
    ):

        age = (
            prediction_dt
            - radar_dt
        ).total_seconds() / 60.0

    valid = (
        radar_dt is not None
        and prediction_dt is not None
        and age is not None
        and 0 <= age
        <= MAX_LIVE_PREDICTION_MINUTES
    )

    if not valid:

        return {
            "ok": False,

            "modo":
                "SIN_PREDICCION_VALIDA",

            "frame_radar":
                frame_radar,

            "frame_prediccion":
                prediction_ts,

            "edad_prediccion_minutos":
                age,

            "ventana_maxima_minutos":
                MAX_LIVE_PREDICTION_MINUTES,

            "es_observacion_radar":
                True,

            "es_prediccion":
                False,

            "nota": (
                "No hay una predicción "
                "entre frames válida "
                "dentro de la ventana "
                "permitida."
            ),

            "datos":
                prediction,

            **radar_metadata(),
        }

    return {
        "ok": True,

        "modo":
            "PREDICCION_ENTRE_FRAMES",

        "frame_radar":
            frame_radar,

        "frame_prediccion":
            prediction_ts,

        "edad_prediccion_minutos":
            round(
                age,
                2,
            ),

        "ventana_maxima_minutos":
            MAX_LIVE_PREDICTION_MINUTES,

        "es_observacion_radar":
            False,

        "es_prediccion":
            True,

        "nota": (
            "Predicción entre frames; "
            "no es una observación "
            "directa del radar."
        ),

        "datos":
            prediction,

        **radar_metadata(),
    }


# ============================================================
# STORM ENGINE
# ============================================================

@app.get("/tormenta")
def tormenta() -> (
    dict[str, Any]
):

    storm = read_json(
        STORM_FILE
    )

    if storm is None:

        return {
            "ok": False,

            "datos":
                None,

            "probabilidades_calibradas":
                False,

            "tipo_probabilidades":
                "no_publicadas",

            "estado": "sin_datos",

            **radar_metadata(),
        }

    storm = sanitize_storm_probabilities(storm)

    return {
        "ok": True,

        "datos":
            storm,

        "probabilidades_calibradas":
            storm.get("probabilidades_calibradas") is True,

        "tipo_probabilidades":
            (
                "calibradas"
                if storm.get("probabilidades_calibradas") is True
                else "heuristicas_no_calibradas"
            ),

        "frame_radar":
            radar_timestamp(),

        **radar_metadata(),
    }


# ============================================================
# MODELO ML
# ============================================================

@app.get("/modelo")
def modelo() -> (
    dict[str, Any]
):

    ml = read_json(
        ML_PREDICTION_FILE
    )

    if ml is None:

        return {
            "ok": False,

            "modelo":
                "rma10-v1",

            "frame_radar":
                radar_timestamp(),

            "horizontes_minutos":
                [
                    5,
                    10,
                    20,
                    30,
                ],

            "tracks_analizados":
                0,

            "predicciones":
                [],

            "calibracion_probabilistica":
                None,

            "calibracion":
                None,

            **radar_metadata(),
        }

    ml = sanitize_ml_probabilities(ml)

    return {
        "ok":
            bool(
                ml.get(
                    "ok",
                    True,
                )
            ),

        "modelo":
            ml.get(
                "modelo",
                "rma10-v1",
            ),

        "frame_radar":
            ml.get(
                "frame_radar"
            ),

        "horizontes_minutos":
            [
                5,
                10,
                20,
                30,
            ],

        "tracks_analizados":
            ml.get(
                "tracks_analizados",
                0,
            ),

        "predicciones":
            ml.get(
                "predicciones",
                [],
            ),

        "calibracion_probabilistica":
            ml.get(
                "calibracion_probabilistica",
                ml.get("calibracion"),
            ),

        "calibracion":
            ml.get(
                "calibracion_probabilistica",
                ml.get("calibracion"),
            ),

        **radar_metadata(),
    }


@app.get("/modelo/nowcast")
def modelo_nowcast() -> (
    dict[str, Any]
):

    ml = read_json(
        ML_PREDICTION_FILE
    )

    ml = sanitize_ml_probabilities(ml)

    return {
        "ok":
            ml is not None,

        "modelo":
            "rma10-v1",

        "frame_radar":
            (
                ml.get(
                    "frame_radar"
                )
                if isinstance(
                    ml,
                    dict,
                )
                else radar_timestamp()
            ),

        "horizontes_minutos":
            [
                5,
                10,
                20,
                30,
            ],

        "datos":
            ml,

        "nota": (
            "Este modelo no debe "
            "interpretarse como un "
            "predictor validado de "
            "tornado, granizo, ráfagas "
            "o rotación."
        ),

        **radar_metadata(),
    }


# ============================================================
# INTELIGENCIA COMPLETA
# ============================================================

@app.get("/inteligencia")
def inteligencia() -> (
    dict[str, Any]
):

    payload = result_payload()

    final = payload.get("resultado_final")
    storm = payload.get("inteligencia_tormenta")

    if not isinstance(final, dict):
        final = {}

    if not isinstance(storm, dict):
        storm = {}

    decision = final.get("motor_decision")

    if not isinstance(decision, dict):
        decision = {}

    severity_block = final.get("severidad_meteorologica")

    severity = (
        severity_block.get("valor")
        if isinstance(severity_block, dict)
        else severity_block
    )

    severity = (
        severity
        or decision.get("severidad_meteorologica")
        or storm.get("severidad")
    )

    if severity == "roja":
        severity = "rojo"

    classification = (
        final.get("clasificacion_meteorologica")
        or decision.get("clasificacion_meteorologica")
        or storm.get("clasificacion_meteorologica")
    )

    threat = (
        final.get("amenaza_bahia")
        or storm.get("amenaza_bahia")
    )

    # Resumen plano y estable para la interfaz; la estructura
    # completa se mantiene intacta debajo.
    return {
        **payload,
        "severidad": severity,
        "clasificacion_meteorologica": classification,
        "amenaza_bahia": threat,
        "estado_operativo": decision.get("estado_operativo"),
        "principal": final.get("principal"),
        "vigencia_radar": (
            storm.get("vigencia_radar")
        ),
        "radar_timestamp": storm.get("radar_timestamp"),
        "radar_age": storm.get("radar_age"),
        "origen_datos": storm.get("origen_datos"),
    }


# ============================================================
# DEBUG
# ============================================================

@app.get("/debug")
def debug() -> (
    dict[str, Any]
):

    return {
        "base_dir":
            str(BASE_DIR),

        "data_base_dir":
            str(DATA_BASE_DIR),

        "remote_data_enabled":
            REMOTE_DATA_ENABLED,

        "remote_cache_seconds":
            REMOTE_CACHE_SECONDS,

        "remote_history_seconds":
            REMOTE_HISTORY_SECONDS,

        "github_repo":
            GITHUB_REPO,

        "github_ref":
            GITHUB_REF,

        "github_raw_base":
            GITHUB_RAW_BASE,

        "radar_dir":
            str(RADAR_DIR),

        "history_dir":
            str(HISTORY_DIR),

        "ia_dir":
            str(IA_DIR),

        "archivos": {
            "nowcast":
                NOWCAST_FILE.is_file(),

            "resultado_procesamiento":
                RESULTADO_FILE.is_file(),

            "decision":
                DECISION_FILE.is_file(),

            "storm":
                STORM_FILE.is_file(),

            "meteo":
                METEO_FILE.is_file(),

            "prediccion":
                PREDICTION_FILE.is_file(),

            "ml":
                ML_PREDICTION_FILE.is_file(),

            "timestamp":
                TIMESTAMP_FILE.is_file(),

            "actual_png":
                ACTUAL_IMAGE.is_file(),
        },

        "sincronizacion":
            synchronized_frame_status(),

        **radar_metadata(),
    }


# ============================================================
# RESUMEN PARA LA PANTALLA GENERAL (interfaz 24/11/2026)
# ============================================================
# Un solo pedido chico con lo que necesita la vista "General":
# tiempo actual, estado por localidad, tormentas sobre el mapa.
# La vista "Meteorólogos" sigue usando /inteligencia y el resto.

SEGUIMIENTO_FILE = RADAR_DIR / "seguimiento" / "estado.json"
EVOLUCION_FILE = RADAR_DIR / "seguimiento" / "prediccion_evolucion.json"
LOCALIDADES_FILE = BASE_DIR / "config" / "localidades.json"

# Geometría del RMA10 (igual que scripts/rma10_features.py).
_RADAR_LAT = -38.7342
_RADAR_LON = -62.1654
_RADAR_RADIUS_KM = 240.0
_RADAR_PX = 512
_KM_LAT = 110.574
_KM_LON = 111.320 * math.cos(math.radians(_RADAR_LAT))

# Puntos de recorrido que se mandan por tormenta (2 h a 10 min).
_MAX_RECORRIDO = 12


def _pixel_a_latlon(x: Any, y: Any) -> list[float] | None:
    try:
        x = float(x)
        y = float(y)
    except (TypeError, ValueError):
        return None

    if not (math.isfinite(x) and math.isfinite(y)):
        return None

    c = (_RADAR_PX - 1) / 2.0
    este = (x - c) / c * _RADAR_RADIUS_KM
    norte = (c - y) / c * _RADAR_RADIUS_KM

    return [
        round(_RADAR_LAT + norte / _KM_LAT, 4),
        round(_RADAR_LON + este / _KM_LON, 4),
    ]


def _km_por_pixel() -> float:
    return _RADAR_RADIUS_KM / ((_RADAR_PX - 1) / 2.0)


def _num(value: Any, digits: int = 1) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if math.isfinite(v) else None


OBS_DIR = IA_DIR / "estaciones" / "observaciones"
# Estaciones de Bahía Blanca para "El tiempo ahora", en orden de preferencia
# (24/11/2026). Centro: en la ciudad, mide cada minuto. Noroeste (CWOP): la
# única que mide ráfagas, humedad y lluvia de la última hora, cada ~10 min.
# Grünbein: respaldo del viento.
_ESTACIONES_BAHIA = (
    ("bcp", "104", "Bahía Blanca Centro"),
    ("cwop", "LU4DRH-11", "Bahía Blanca Noroeste"),
    ("bcp", "1", "Grünbein"),
)
_RUMBOS_TEXTO = {
    "norte": 0, "nornoreste": 22.5, "noreste": 45, "estenoreste": 67.5, "este": 90,
    "estesudeste": 112.5, "sudeste": 135, "sudsudeste": 157.5, "sur": 180,
    "sudsudoeste": 202.5, "sudoeste": 225, "oestesudoeste": 247.5, "oeste": 270,
    "oestenoroeste": 292.5, "noroeste": 315, "nornoroeste": 337.5,
}


def _rumbo_grados(valor: Any) -> float | None:
    if isinstance(valor, (int, float)):
        return float(valor)
    texto = str(valor or "").strip().lower().replace(" ", "").replace("-", "")
    return _RUMBOS_TEXTO.get(texto)


def _viento_zona(ident: str) -> dict[str, Any] | None:
    """El viento que usa la tarjeta de la localidad (máximo de las estaciones
    a 15 km o menos), con la dirección de la estación que lo midió."""
    from scripts.estaciones.frescura import edad_min

    sen = read_json(IA_DIR / "estaciones" / "senales_superficie.json") or {}
    zona = (sen.get("localidades") or {}).get(ident) or {}
    vmax = _num(zona.get("viento_max_kmh"), 0)
    if vmax is None:
        return None
    edad = edad_min(sen.get("generado_utc"))
    if edad is None or edad > 30:
        return None
    raf = _num(zona.get("rafaga_max_kmh"), 0)
    nombres = [str(n) for n in zona.get("nombres") or []]
    direccion, quien = None, None
    for est in (read_json(OBS_DIR / "bcp_actual.json") or {}).get("stations") or []:
        if isinstance(est, dict) and est.get("nombre") in nombres:
            v = est.get("variables") or {}
            if _num(v.get("viento_kmh"), 0) == vmax:
                direccion = _rumbo_grados(v.get("direccion_viento_grados", v.get("direccion_viento")))
                quien = str(est["nombre"]).title().replace("Grumbein", "Grünbein").replace("Bahia", "Bahía")
                break
    return {
        "nombre": f"{quien or 'estaciones'} (zona de 15 km)",
        "edad": edad,
        "viento": vmax,
        "viento_dir": direccion,
        "rafaga": raf if raf is not None and raf >= vmax else None,
    }


def _vientos_trabados() -> set[str]:
    """Estaciones con el anemómetro trabado en 0 (senales_superficie.py)."""
    s = read_json(IA_DIR / "estaciones" / "senales_superficie.json") or {}
    return {str(n).upper() for n in s.get("viento_trabado") or []}


def _sin_viento_cwop(v: dict[str, Any]) -> dict[str, Any]:
    """Las estaciones de aficionados (CWOP) no dan el viento en la página.

    24/11/2026: Bahía Noroeste midió 3 km/h con 33 km/h en el Centro (viento
    del este): está reparada. No se sabe cómo está instalada ninguna estación
    CWOP, y el viento es lo que más depende de eso. La temperatura, la humedad
    y la lluvia sí se usan."""
    return {k: x for k, x in v.items() if not k.startswith(("viento", "rafaga", "direccion_viento"))}


def _mediciones_bahia() -> list[dict[str, Any]]:
    """Mediciones de las estaciones de Bahía, con la edad calculada ahora.

    Solo entran las que están dentro del límite de su red (frescura.py)."""
    from scripts.estaciones.frescura import edad_min, es_fresca

    archivos = {"bcp": read_json(OBS_DIR / "bcp_actual.json") or {},
                "cwop": read_json(OBS_DIR / "cwop_actual.json") or {}}
    trabados = _vientos_trabados()
    salida = []
    for red, ident, nombre in _ESTACIONES_BAHIA:
        for est in archivos[red].get("stations") or []:
            if not isinstance(est, dict):
                continue
            clave = str(est.get("station_id") if red == "bcp" else est.get("indicativo"))
            if clave != ident:
                continue
            hora = est.get("observation_time")
            if not es_fresca(hora, red):
                continue
            v = est.get("variables") if isinstance(est.get("variables"), dict) else {}
            if red == "cwop" or str(est.get("nombre") or "").upper() in trabados:
                v = _sin_viento_cwop(v)
            salida.append({
                "nombre": nombre,
                "edad": edad_min(hora),
                "temperatura": _num(v.get("temperatura_c"), 1),
                "viento": _num(v.get("viento_kmh"), 0),
                "viento_dir": _rumbo_grados(v.get("direccion_viento_grados", v.get("direccion_viento"))),
                "rafaga": _num(v.get("rafaga_15min_kmh"), 0),
                "humedad": _num(v.get("humedad_relativa_pct"), 0),
                "lluvia_1h": _num(v.get("lluvia_1h_mm"), 1),
            })
    return salida


# Estaciones que cuentan como "de la localidad" (24/11/2026): a 8 km o menos.
# Más lejos (o sobre las sierras, como Sierras Grandes para Sierra de la
# Ventana) no representan el pueblo y se usa el modelo europeo en su punto.
ESTACION_LOCAL_MAX_KM = 8.0
AEROPUERTO_SAZB = (-38.71474, -62.16304)
AEROPUERTO_MAX_KM = 12.0


def _dist_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def _mediciones_cerca(lat: float, lon: float) -> list[dict[str, Any]]:
    """Mediciones frescas de las estaciones a 8 km o menos del punto, de la más
    cercana a la más lejana (edad calculada ahora, frescura.py)."""
    from scripts.estaciones.frescura import edad_min, es_fresca

    cat = read_json(IA_DIR / "estaciones" / "catalogo_estaciones.json") or {}
    pos = {}
    for st in cat.get("stations") or []:
        la, lo = _num(st.get("latitude"), 6), _num(st.get("longitude"), 6)
        if la is not None and lo is not None:
            pos[(st.get("network"), str(st.get("collector_id")))] = (la, lo)
    trabados = _vientos_trabados()
    salida = []
    for red in ("bcp", "cwop"):
        for est in (read_json(OBS_DIR / f"{red}_actual.json") or {}).get("stations") or []:
            if not isinstance(est, dict):
                continue
            if red == "bcp":
                coord = pos.get(("bcp", str(est.get("station_id"))))
            else:
                la, lo = _num(est.get("lat"), 6), _num(est.get("lon"), 6)
                coord = (la, lo) if la is not None and lo is not None else None
            if not coord:
                continue
            dist = _dist_km(lat, lon, coord[0], coord[1])
            hora = est.get("observation_time")
            if dist > ESTACION_LOCAL_MAX_KM or not es_fresca(hora, red):
                continue
            v = est.get("variables") if isinstance(est.get("variables"), dict) else {}
            if red == "cwop" or str(est.get("nombre") or "").upper() in trabados:
                v = _sin_viento_cwop(v)
            nombre = str(est.get("nombre") or est.get("indicativo") or "Estación")
            nombre = nombre.replace(" (CWOP)", "").title().replace("Grumbein", "Grünbein").replace("Bahia", "Bahía")
            salida.append({
                "nombre": nombre if dist < 1.5 else f"{nombre}, a {dist:.0f} km",
                "distancia": dist,
                "edad": edad_min(hora),
                "temperatura": _num(v.get("temperatura_c"), 1),
                "viento": _num(v.get("viento_kmh"), 0),
                "viento_dir": _rumbo_grados(v.get("direccion_viento_grados", v.get("direccion_viento"))),
                "rafaga": _num(v.get("rafaga_15min_kmh"), 0),
                "humedad": _num(v.get("humedad_relativa_pct"), 0),
                "lluvia_1h": _num(v.get("lluvia_1h_mm"), 1),
            })
    return sorted(salida, key=lambda m: m["distancia"])


def _tiempo_actual(localidad: dict[str, Any] | None = None) -> dict[str, Any]:
    """El tiempo ahora en una localidad: lo medido primero, el modelo de relleno.

    Sin localidad (o Bahía Blanca) es Bahía, como siempre. Para las demás
    (24/11/2026): estaciones a 8 km o menos y el modelo en su punto.

    24/11/2026: antes se mostraba el modelo (Open-Meteo) aunque las
    estaciones estuvieran al día, y aparecían ráfagas de 47 km/h con viento
    medido de 13 km/h. Ahora cada dato sale de la primera estación de Bahía
    que lo midió hace poco (Centro, Noroeste, Grünbein; el aeropuerto para la
    presión) y el modelo solo completa lo que ninguna mide. Si hay viento
    medido pero ninguna estación midió ráfagas, no se muestra la ráfaga del
    modelo (no se mezcla lo medido con lo estimado)."""

    from scripts.estaciones.frescura import edad_min, es_fresca

    es_bahia = not localidad or localidad.get("id") == "bahia_blanca"
    estado = read_json(METEO_FILE) or {}
    amb = estado.get("ambiente_actual") if isinstance(estado.get("ambiente_actual"), dict) else {}
    if es_bahia:
        modelo = get_openmeteo_data() or {}
    else:
        modelo = (get_openmeteo_locales() or {}).get(localidad["id"]) or {}
        amb = {}     # el ambiente guardado es el de Bahía: no se usa para otro punto
    if not modelo and amb:
        modelo = {
            "temperatura": amb.get("temperature_2m"), "punto_rocio": amb.get("dew_point_2m"),
            "presion": amb.get("pressure_msl"), "viento_vel": amb.get("wind_speed_10m"),
            "viento_dir": amb.get("wind_direction_10m"), "rafagas": amb.get("wind_gusts_10m"),
            "humedad": amb.get("relative_humidity_2m"), "nubosidad": amb.get("cloud_cover"),
            "precipitacion": amb.get("precipitation"),
        }

    obs = estado.get("observacion_sazb")
    obs = obs if isinstance(obs, dict) and obs.get("disponible") else {}
    if not es_fresca(obs.get("obsTime_utc"), "sazb"):
        obs = {}
    if es_bahia:
        medidas = _mediciones_bahia()
    else:
        medidas = _mediciones_cerca(float(localidad["lat"]), float(localidad["lon"]))
        # 24/11/2026: si ninguna estación propia da el viento, se usa el
        # mismo que la tarjeta de la localidad (estaciones a 15 km o menos,
        # senales_superficie.json). Antes arriba salía el modelo y abajo las
        # estaciones: dos números distintos para el mismo lugar.
        if not any(m.get("viento") is not None for m in medidas):
            vz = _viento_zona(localidad["id"])
            if vz:
                medidas.append(vz)
        if _dist_km(float(localidad["lat"]), float(localidad["lon"]), *AEROPUERTO_SAZB) > AEROPUERTO_MAX_KM:
            obs = {}
    if obs:
        kt = _num(obs.get("viento_kt"), 1)
        medidas.append({
            "nombre": "Aeropuerto", "edad": edad_min(obs.get("obsTime_utc")),
            "temperatura": _num(obs.get("temperatura_c"), 1),
            "viento": round(kt * 1.852) if kt is not None else None,
            "viento_dir": _num(obs.get("direccion_viento"), 0),
            "presion": _num(obs.get("presion_hpa"), 0),
        })

    if not medidas and not modelo:
        return {"disponible": False}

    fuentes: dict[str, str] = {}

    def medido(campo: str, clave: str):
        for m in medidas:
            if m.get(clave) is not None:
                fuentes[campo] = f"{m['nombre']} (hace {round(m['edad'] or 0)} min)"
                return m
        return None

    salida: dict[str, Any] = {"disponible": True}
    m = medido("temperatura", "temperatura")
    salida["temperatura"] = m["temperatura"] if m else modelo.get("temperatura")
    # Viento y ráfaga de la MISMA estación (si no, la ráfaga de una puede
    # quedar por debajo del viento de otra). 24/11/2026: manda el orden de
    # las estaciones (Centro primero), no "la que mide los dos": la Noroeste
    # es la única con ráfagas y estaba reparada (3 km/h con 33 en el Centro).
    m = medido("viento", "viento")
    if m:
        etiqueta = f"{m['nombre']} (hace {round(m['edad'] or 0)} min)"
        fuentes["viento"] = etiqueta
        salida["viento_vel"] = m["viento"]
        salida["viento_dir"] = m.get("viento_dir")
        raf = m.get("rafaga")
        # Una ráfaga menor que el viento medio no es una medición válida.
        salida["rafagas"] = raf if raf is not None and (m["viento"] is None or raf >= m["viento"]) else None
        if salida["rafagas"] is not None:
            fuentes["rafagas"] = etiqueta
    else:
        salida["viento_vel"] = modelo.get("viento_vel")
        salida["viento_dir"] = modelo.get("viento_dir")
        salida["rafagas"] = modelo.get("rafagas")
    m = medido("humedad", "humedad")
    salida["humedad"] = m["humedad"] if m else modelo.get("humedad")
    m = medido("presion", "presion")
    salida["presion"] = m["presion"] if m else modelo.get("presion")
    m = medido("precipitacion", "lluvia_1h")
    salida["precipitacion"] = m["lluvia_1h"] if m else modelo.get("precipitacion")

    # La sensación térmica del modelo solo si la temperatura también es del modelo.
    salida["sensacion_termica"] = None if "temperatura" in fuentes else modelo.get("sensacion_termica")
    salida["punto_rocio"] = modelo.get("punto_rocio")
    salida["nubosidad"] = modelo.get("nubosidad")
    salida["codigo_tiempo"] = modelo.get("codigo_tiempo")
    salida["fuentes"] = fuentes
    medidos = sorted(set(fuentes.values()))
    salida["fuente"] = (
        ("medido: " + ", ".join(medidos) if medidos else "")
        + ("; " if medidos and modelo else "")
        + ("cielo y lo que falta: modelo europeo" if modelo else "")
    )
    return salida


def _fenomeno_corto(item: Any) -> dict[str, Any]:
    item = item if isinstance(item, dict) else {}
    return {
        "grado": item.get("grado"),
        "categoria": item.get("categoria"),
        "confirmado": bool(item.get("confirmado")),
        "en_calibracion": bool(item.get("en_calibracion")),
        "fuente": item.get("fuente"),
        # El nivel se mantiene un rato después de la última detección: la
        # portada dice "hace X min" para que no choque con lo de ahora.
        "minutos_desde_deteccion": (
            _num(item.get("minutos_desde_deteccion"), 0)
            if item.get("mantenido") else None
        ),
        # Los números que acompañan al nivel (24/11/2026).
        "mm_h": _num(item.get("mm_h"), 1),
        "mm_1h": _num(item.get("mm_1h"), 1),
        "mm_3h": _num(item.get("mm_3h"), 1),
        "rafaga_kmh": _num(item.get("rafaga_kmh"), 0),
        "viento_kmh": _num(item.get("viento_kmh"), 0),
        "tasa_por_min": _num(item.get("tasa_por_min"), 1),
        "razon": (item.get("razones") or [None])[0],
        "cuando": item.get("cuando"),
    }


def _localidades_resumen() -> list[dict[str, Any]]:
    config = read_json(LOCALIDADES_FILE) or {}
    final = read_json(DECISION_FILE) or {}
    tipo = final.get("tipo_tiempo")
    tipo = tipo if isinstance(tipo, dict) else {}
    por_loc = tipo.get("localidades")
    por_loc = por_loc if isinstance(por_loc, dict) else {}

    salida = []

    for loc in config.get("localidades") or []:
        if not isinstance(loc, dict) or not loc.get("id"):
            continue

        datos = por_loc.get(loc["id"])
        datos = datos if isinstance(datos, dict) else {}
        fen = datos.get("fenomenos")
        fen = fen if isinstance(fen, dict) else {}
        calif = datos.get("calificacion")
        calif = calif if isinstance(calif, dict) else {}
        llegada = datos.get("llegada")

        salida.append({
            "id": loc["id"],
            "nombre": loc.get("nombre"),
            "lat": loc.get("lat"),
            "lon": loc.get("lon"),
            "radio_km": loc.get("radio_km"),
            "con_datos": bool(datos),
            "nivel": calif.get("grado"),
            "etiqueta": calif.get("etiqueta"),
            "resumen_ahora": datos.get("resumen_ahora"),
            # Lo que da el nivel (incluye lo detectado en los últimos minutos).
            "resumen": datos.get("resumen"),
            "resumen_al_llegar": datos.get("resumen_al_llegar"),
            "minutos_hasta_llegada": _num(
                datos.get("minutos_hasta_llegada"), 0
            ),
            # El tiempo ahora en esa localidad (24/11/2026): antes la página
            # mostraba el de Bahía para todas.
            "tiempo_actual": _tiempo_actual(loc),
            "llegada": llegada if isinstance(llegada, dict) else None,
            "sobre_la_localidad": datos.get("sobre_la_localidad"),
            # Lluvia que ya cayó según el radar (mm en 1, 3 y 6 h).
            "acumulado_radar_mm": (
                datos.get("acumulado_radar_mm")
                if isinstance(datos.get("acumulado_radar_mm"), dict) else None
            ),
            # Lo que trae la tormenta que llega, fenómeno por fenómeno.
            "al_llegar": (
                {
                    k: _fenomeno_corto(v)
                    for k, v in datos["al_llegar"].items()
                    if k in ("lluvia", "viento", "granizo", "electrica")
                }
                if isinstance(datos.get("al_llegar"), dict) else None
            ),
            "fenomenos": {
                k: _fenomeno_corto(fen.get(k))
                for k in ("lluvia", "viento", "granizo", "electrica")
            },
        })

    return salida


def _tormentas_mapa() -> list[dict[str, Any]]:
    seguimiento = read_json(SEGUIMIENTO_FILE) or {}
    evolucion = read_json(EVOLUCION_FILE) or {}

    activos = {
        str(t) for t in seguimiento.get("tracks_activos") or []
    }

    pron: dict[str, list[dict[str, Any]]] = {}

    for n in evolucion.get("nucleos") or []:
        if not isinstance(n, dict) or not n.get("pronostico_disponible"):
            continue
        puntos = []
        for h in n.get("horizontes") or []:
            if not isinstance(h, dict):
                continue
            ll = _pixel_a_latlon(h.get("x_estimado"), h.get("y_estimado"))
            if ll:
                puntos.append({
                    "minutos": h.get("horizonte_minutos"),
                    "lat": ll[0],
                    "lon": ll[1],
                    "incertidumbre_km": _num(
                        (h.get("incertidumbre_pixeles") or 0)
                        * _km_por_pixel(),
                        1,
                    ),
                })
        pron[str(n.get("track_id"))] = puntos

    kmpx = _km_por_pixel()
    salida = []

    for t in seguimiento.get("tracks") or []:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("track_id"))
        if tid not in activos or not t.get("presente_en_ultimo_frame"):
            continue

        pos = _pixel_a_latlon(t.get("x"), t.get("y"))
        if not pos:
            continue

        historial_completo = [
            h for h in (t.get("historial") or []) if isinstance(h, dict)
        ]
        historial = historial_completo[-_MAX_RECORRIDO:]
        recorrido = _recorrido_continuo(historial)

        vel = _num(t.get("velocidad_pixeles_min"), 4)

        if seguimiento.get("ultimo_radar_timestamp") == evolucion.get(
            "radar_timestamp"
        ):
            futuro = pron.get(tid, [])
            fuente_futuro = "prediccion_evolucion"
        else:
            # La predicción guardada es de un frame anterior: arrancaría
            # desde otra posición. Se usa el movimiento actual de la celda.
            futuro = _extrapolar(t)
            fuente_futuro = "movimiento_actual"

        salida.append({
            "id": tid,
            "lat": pos[0],
            "lon": pos[1],
            "intensidad": t.get("intensidad"),
            "nivel_intensidad": t.get("nivel_intensidad"),
            "direccion": t.get("direccion"),
            "rumbo_grados": _num(t.get("rumbo_grados"), 0),
            "velocidad_kmh": (
                round(vel * kmpx * 60.0, 0) if vel is not None else None
            ),
            "tendencia_area": t.get("tendencia_area"),
            "confianza_movimiento": _num(t.get("confianza_movimiento"), 2),
            # El centro saltó más de lo físico: puede ser otra tormenta.
            "movimiento_dudoso": bool(t.get("movimiento_dudoso")),
            "tamano_km2": _num(
                (t.get("pixeles") or 0) * kmpx * kmpx, 0
            ),
            "recorrido": recorrido,
            "historia": _historia_celda(historial_completo),
            "pronostico": futuro,
            "pronostico_fuente": fuente_futuro,
        })

    salida.sort(key=lambda x: -(x.get("intensidad") or 0))
    return salida


# ============================================================
# RADAR LIMPIO PARA EL MAPA (solo para mirar)
# ============================================================
# La estructura sigue usando el PNG crudo. Esto arma una versión
# limpia y suave para la gente, una sola vez por imagen.

CLUTTER_PNG = BASE_DIR / "config" / "clutter_rma10.png"
_MAPA_CACHE: dict[str, tuple[float, bytes]] = {}
_MAPA_LOCK = threading.Lock()
_MAPA_MAX = 20


def _imagen_mapa(path: Path) -> bytes | None:
    import radar_visual

    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None

    clave = str(path)
    with _MAPA_LOCK:
        guardada = _MAPA_CACHE.get(clave)
        if guardada and guardada[0] == mtime:
            return guardada[1]

    try:
        datos = radar_visual.imagen_mapa(path, CLUTTER_PNG)
    except Exception:
        datos = None

    if datos is not None:
        with _MAPA_LOCK:
            _MAPA_CACHE[clave] = (mtime, datos)
            while len(_MAPA_CACHE) > _MAPA_MAX:
                _MAPA_CACHE.pop(next(iter(_MAPA_CACHE)))
    return datos


def _respuesta_mapa(path: Path, segundos: int):
    valid, reason = png_is_valid(path)
    if not valid:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "error": "imagen_no_disponible", "motivo": reason},
        )
    datos = _imagen_mapa(path)
    if datos is None:
        # Paleta desconocida: se manda la imagen original antes que nada.
        return FileResponse(path, media_type="image/png")
    return Response(
        content=datos,
        media_type="image/png",
        headers={"Cache-Control": f"public, max-age={segundos}"},
    )


@app.get("/radar/mapa")
def radar_mapa():
    return _respuesta_mapa(ACTUAL_IMAGE, 60)


@app.get("/radar/mapa/{timestamp}")
def radar_mapa_historico(timestamp: str):
    if len(timestamp) != 14 or not timestamp.isdigit():
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "timestamp_invalido"},
        )
    # Una imagen vieja no cambia: el navegador la puede guardar un día.
    return _respuesta_mapa(HISTORY_DIR / f"{timestamp}.png", 86400)


# Salto máximo creíble entre dos puntos del recorrido: 120 km/h
# (tope físico del proyecto) más 5 km de margen por el tamaño de la celda.
_VEL_MAX_KMH = 120.0


def _recorrido_continuo(historial: list[dict[str, Any]]) -> list[list[float]]:
    """Recorrido hacia atrás hasta el primer salto imposible.

    Si el seguimiento heredó el número de otra celda (división o unión),
    el salto no se dibuja como si la tormenta hubiera viajado.
    """

    puntos = []
    for h in historial:
        ll = _pixel_a_latlon(h.get("x"), h.get("y"))
        if ll:
            puntos.append((parse_iso(h.get("timestamp")), ll))

    if not puntos:
        return []

    salida = [puntos[-1][1]]
    for (t_prev, a), (t_sig, b) in zip(
        reversed(puntos[:-1]), reversed(puntos[1:])
    ):
        minutos = 10.0
        if t_prev and t_sig:
            minutos = max((t_sig - t_prev).total_seconds() / 60.0, 1.0)
        dist = math.hypot(
            (a[0] - b[0]) * _KM_LAT, (a[1] - b[1]) * _KM_LON
        )
        if dist > _VEL_MAX_KMH * minutos / 60.0 + 5.0:
            break
        salida.append(a)

    salida.reverse()
    return salida


def _historia_celda(historial: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Historia de una tormenta con lo que guardó el seguimiento (24/11/2026,
    sugerencia de Alex): desde cuándo se la sigue, desde dónde vino, cuántos
    km recorrió y su máximo. Solo la parte continua (la misma regla que el
    recorrido): si heredó el número de otra celda, no se mezcla su pasado."""
    puntos = []
    for h in historial:
        ll = _pixel_a_latlon(h.get("x"), h.get("y"))
        ts = parse_iso(h.get("timestamp"))
        if ll and ts:
            puntos.append({"ts": ts, "ll": ll, "int": _num(h.get("intensidad"), 1),
                           "px": _num(h.get("pixeles"), 1)})
    if len(puntos) < 2:
        return None
    tramo = [puntos[-1]]
    for prev in reversed(puntos[:-1]):
        sig = tramo[-1]
        minutos = max((sig["ts"] - prev["ts"]).total_seconds() / 60.0, 1.0)
        dist = math.hypot((prev["ll"][0] - sig["ll"][0]) * _KM_LAT, (prev["ll"][1] - sig["ll"][1]) * _KM_LON)
        if dist > _VEL_MAX_KMH * minutos / 60.0 + 5.0:
            break
        tramo.append(prev)
    tramo.reverse()
    if len(tramo) < 2:
        return None
    km = sum(math.hypot((a["ll"][0] - b["ll"][0]) * _KM_LAT, (a["ll"][1] - b["ll"][1]) * _KM_LON)
             for a, b in zip(tramo, tramo[1:]))
    con_int = [p for p in tramo if p["int"] is not None]
    maximo = max(con_int, key=lambda p: p["int"]) if con_int else None
    kmpx = _km_por_pixel()
    areas = [p["px"] for p in tramo if p["px"] is not None]
    return {
        "desde_utc": tramo[0]["ts"].isoformat(),
        "minutos": round((tramo[-1]["ts"] - tramo[0]["ts"]).total_seconds() / 60.0),
        "inicio": [round(tramo[0]["ll"][0], 4), round(tramo[0]["ll"][1], 4)],
        "recorrido_km": round(km),
        "intensidad_inicio": tramo[0]["int"],
        "intensidad_max": maximo["int"] if maximo else None,
        "intensidad_max_utc": maximo["ts"].isoformat() if maximo else None,
        "tamano_max_km2": round(max(areas) * kmpx * kmpx) if areas else None,
        "imagenes": len(tramo),
    }


def _extrapolar(t: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        vx = float(t.get("velocidad_x_pixeles_min"))
        vy = float(t.get("velocidad_y_pixeles_min"))
        x = float(t.get("x"))
        y = float(t.get("y"))
    except (TypeError, ValueError):
        return []

    if not all(math.isfinite(v) for v in (vx, vy, x, y)):
        return []

    # Celda quieta o recién aparecida: no hay hacia dónde dibujar.
    if math.hypot(vx, vy) < 0.02:
        return []

    salida = []
    for minutos in (10, 20, 30):
        ll = _pixel_a_latlon(x + vx * minutos, y + vy * minutos)
        if ll:
            salida.append({
                "minutos": minutos,
                "lat": ll[0],
                "lon": ll[1],
                "incertidumbre_km": None,
            })
    return salida


@app.get("/resumen")
def resumen() -> dict[str, Any]:
    """Todo lo que usa la vista General, en un pedido chico."""

    final = read_json(DECISION_FILE) or {}
    tipo = final.get("tipo_tiempo")
    tipo = tipo if isinstance(tipo, dict) else {}
    amb = tipo.get("ambiente")
    amb = amb if isinstance(amb, dict) else {}
    geo = radar_geo()

    return {
        "ciudad": CITY,
        "tiempo_actual": _tiempo_actual(),
        "localidades": _localidades_resumen(),
        "tormentas": _tormentas_mapa(),
        "ambiente": amb.get("etiqueta"),
        "radar": {
            "bounds": geo.get("bounds"),
            "centro": [_RADAR_LAT, _RADAR_LON],
            "alcance_km": _RADAR_RADIUS_KM,
            "imagen": "/radar/mapa",
        },
        "rayos": _rayos_recientes(),
        "precision": _precision_trayectoria(),
        **radar_metadata(),
    }


def _precision_trayectoria() -> dict[str, Any] | None:
    """Error real del seguimiento (mediana, en km) medido por la cadena en cada
    ciclo (salud_cadena.json). 24/11/2026: la página decía un número fijo
    (4,5 km) que ya no era el actual."""
    salud = read_json(IA_DIR / "salud_cadena.json") or {}
    err = (salud.get("trayectoria_5b_error_km") or {}).get("10") or {}
    km, n = _num(err.get("mediana_km"), 1), _num(err.get("muestras"), 0)
    if km is None or not n:
        return None
    return {"minutos": 10, "mediana_km": km, "muestras": int(n)}


def _rayos_recientes() -> dict[str, Any]:
    """Rayos del satélite (GLM) en los últimos 10 y 60 min, para la portada.

    La edad se calcula ahora, al usar el dato (no se confía en una marca
    guardada antes)."""
    datos = read_json(RAYOS_MAPA_FILE) or {}
    lista = datos.get("rayos") if isinstance(datos, dict) else None
    ahora = datetime.now(timezone.utc).timestamp()
    seg = []
    for r in lista or []:
        if isinstance(r, (list, tuple)) and len(r) >= 3:
            t = _num(r[2], 0)
            if t is not None and t <= ahora + 60:
                seg.append(ahora - t)
    actualizado = parse_iso(datos.get("actualizado_utc")) if isinstance(datos, dict) else None
    edad = (
        round((datetime.now(timezone.utc) - actualizado).total_seconds() / 60.0, 1)
        if actualizado else None
    )
    return {
        "ultimos_10_min": sum(1 for x in seg if x <= 600),
        "ultima_hora": sum(1 for x in seg if x <= 3600),
        "edad_datos_minutos": edad,
        # Más de 40 min sin datos nuevos: no se puede decir "sin rayos".
        "disponible": edad is not None and -5 <= edad <= 40,
    }


# ============================================================
# SATÉLITE GOES-19 PARA EL MAPA (vista Meteorólogos, solo para mirar)
# ============================================================
# Nubes del canal 13 sobre los 240 km del radar y los rayos que ya guarda
# Intelligence. Se lee del satélite solo cuando alguien lo pide y se guarda
# un rato: no se agrega nada al repositorio ni a la estructura.

# 24/11/2026: faltaba "data/": en el VPS se pedía goes19/rayos_mapa.json a
# GitHub (no existe) y el mapa del satélite quedaba siempre sin rayos.
RAYOS_MAPA_FILE = DATA_BASE_DIR / "data" / "goes19" / "rayos_mapa.json"
SATELITE_IMAGENES = 6            # última hora (una imagen cada 10 min)
_SAT_LISTA: dict[str, Any] = {"ts": 0.0, "lista": []}
_SAT_LISTA_SEG = 120.0
_SAT_IMG: dict[str, bytes] = {}
_SAT_LOCK = threading.Lock()


def _satelite_lista() -> list[dict[str, Any]]:
    from goes19 import goes19_engine as ge

    ahora = time.time()
    if _SAT_LISTA["lista"] and ahora - _SAT_LISTA["ts"] < _SAT_LISTA_SEG:
        return _SAT_LISTA["lista"]
    prefijo, contiene = ge.PRODUCT_SPECS["abi_c13"]
    referencia = datetime.now(timezone.utc)
    vistos: dict[str, dict[str, Any]] = {}
    for horas in (0, 1):
        for crudo in ge._list_candidate_objects(prefijo, referencia - timedelta(hours=horas)):
            if contiene not in crudo.get("key", ""):
                continue
            meta = ge._enrich_object_metadata(crudo)
            if meta and meta.get("end_time_utc"):
                vistos[meta["key"]] = meta
    lista = sorted(vistos.values(), key=lambda m: m["end_time_utc"])[-SATELITE_IMAGENES:]
    salida = [
        {
            "clave": parse_iso(m["start_time_utc"]).strftime("%Y%m%d%H%M"),
            "hora_utc": m["end_time_utc"],
            "url": m["url"],
        }
        for m in lista
        if parse_iso(m.get("start_time_utc"))
    ]
    if salida:
        _SAT_LISTA["ts"] = ahora
        _SAT_LISTA["lista"] = salida
    return salida or _SAT_LISTA["lista"]


def _satelite_png(item: dict[str, Any]) -> bytes | None:
    import numpy as np

    import satelite_visual
    from goes19.geospatial import prepare_goes_domain_multi

    clave = item["clave"]
    with _SAT_LOCK:
        if clave in _SAT_IMG:
            return _SAT_IMG[clave]
    dominio = prepare_goes_domain_multi(item["url"], ["CMI"], radio_km=_RADAR_RADIUS_KM + 25.0)
    datos = satelite_visual.imagen_nubes(
        dominio["latitude"],
        dominio["longitude"],
        np.asarray(dominio["datos"]["CMI"], np.float32) - 273.15,
    )
    with _SAT_LOCK:
        _SAT_IMG[clave] = datos
        while len(_SAT_IMG) > SATELITE_IMAGENES * 2:
            _SAT_IMG.pop(next(iter(_SAT_IMG)))
    return datos


@app.get("/satelite")
def satelite():
    """Imágenes del satélite de la última hora y los rayos de esa hora."""
    try:
        lista = _satelite_lista()
    except Exception as exc:
        lista = []
        error = f"{type(exc).__name__}"
    else:
        error = None

    rayos = read_json(RAYOS_MAPA_FILE) or {}

    return {
        "imagenes": [
            {"clave": i["clave"], "hora_utc": i["hora_utc"], "imagen": f"/satelite/nubes/{i['clave']}"}
            for i in lista
        ],
        # [lat, lon, segundos UTC] de la última hora (los guarda Intelligence).
        "rayos": {
            "actualizado_utc": rayos.get("actualizado_utc"),
            "lista": rayos.get("rayos") or [],
        },
        "error": error,
    }


@app.get("/satelite/nubes/{clave}")
def satelite_nubes(clave: str):
    if len(clave) != 12 or not clave.isdigit():
        return JSONResponse(status_code=400, content={"ok": False, "error": "clave_invalida"})
    try:
        item = next((i for i in _satelite_lista() if i["clave"] == clave), None)
        if item is None:
            return JSONResponse(status_code=404, content={"ok": False, "error": "imagen_no_disponible"})
        datos = _satelite_png(item)
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"ok": False, "error": "satelite_no_disponible", "motivo": type(exc).__name__},
        )
    # Una imagen ya publicada no cambia.
    return Response(content=datos, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


# ============================================================
# EJECUCIÓN LOCAL
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8000",
            )
        ),
)
