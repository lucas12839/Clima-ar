from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

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

MAX_RADAR_AGE_MINUTES = 45.0
MAX_FUTURE_SKEW_MINUTES = 2.0
MAX_LIVE_PREDICTION_MINUTES = 30.0


# ============================================================
# CACHÉ REMOTO
# ============================================================

_REMOTE_LOCK = threading.RLock()

_REMOTE_LAST_SYNC: dict[str, float] = {}

_REMOTE_HISTORY_CACHE: tuple[
    float,
    list[str],
] = (0.0, [])


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

        headers = {
            "User-Agent": "ClimaAR/8.1",
            "Accept": "application/octet-stream",
        }

        try:
            response = requests.get(
                url,
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

        url = (
            f"{GITHUB_API_BASE}/repos/"
            f"{quote(GITHUB_REPO, safe='/')}"
            "/contents/data/radar/historico"
        )

        params = {
            "ref": GITHUB_REF,
        }

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

        return list(names)


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

def get_openmeteo_data() -> (
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
        "precipitation"
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
            "fuente": "Open-Meteo",
        }

    except Exception:
        return None


# ============================================================
# NOWCAST / NÚCLEOS
# ============================================================

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

    return {
        0: "Sin precipitación",
        1: "Muy débil",
        2: "Débil",
        3: "Moderada",
        4: "Fuerte",
        5: "Muy fuerte",
        6: "Severa",
    }.get(
        level,
        "Sin datos",
    )


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

    return {
        "estado":
            "ok"
            if nowcast
            else "sin_datos",

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
            get_nucleos(
                nowcast
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

    return result_payload()


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
