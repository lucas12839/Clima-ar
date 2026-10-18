from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

try:
    from PIL import Image
except Exception:
    Image = None

BASE_DIR = Path(__file__).resolve().parent
RADAR_DIR = BASE_DIR / "data" / "radar"
HISTORY_DIR = RADAR_DIR / "historico"
IA_DIR = BASE_DIR / "data" / "ia"

NOWCAST_FILE = RADAR_DIR / "radar_nowcast.json"
RESULTADO_FILE = RADAR_DIR / "resultado_procesamiento.json"
DECISION_FILE = IA_DIR / "resultado_final.json"
STORM_FILE = IA_DIR / "inteligencia_tormenta.json"
METEO_FILE = IA_DIR / "estado_meteorologico.json"
PREDICTION_FILE = RADAR_DIR / "seguimiento" / "prediccion_evolucion.json"
ML_PREDICTION_FILE = RADAR_DIR / "seguimiento" / "prediccion_ml_rma10.json"
TIMESTAMP_FILE = RADAR_DIR / "radar_timestamp.txt"
ACTUAL_IMAGE = RADAR_DIR / "actual.png"

LAT = -38.7196
LON = -62.2724
CITY = "Bahía Blanca"
VERSION = "8.1-rma10-operativo"
FUENTE = "SINARAME RMA10"
MAX_RADAR_AGE_MINUTES = 45.0
MAX_FUTURE_SKEW_MINUTES = 2.0
MAX_LIVE_PREDICTION_MINUTES = 30.0


def _cors_origins() -> list[str]:
    raw = os.getenv("CLIMAAR_CORS_ORIGINS", "*").strip()
    if not raw:
        return ["*"]
    return [item.strip() for item in raw.split(",") if item.strip()] or ["*"]


CORS_ORIGINS = _cors_origins()
CORS_WILDCARD = CORS_ORIGINS == ["*"]

app = FastAPI(
    title="ClimaAR - Radar Bahía Blanca",
    description="ClimaAR basado en SINARAME/RMA10 para radar operativo.",
    version=VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False if CORS_WILDCARD else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc)


def radar_timestamp() -> str | None:
    try:
        value = TIMESTAMP_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        value = ""

    if value:
        return value

    return None


def frame_age_minutes(timestamp: str | None) -> float | None:
    dt = parse_iso(timestamp)

    if dt is None:
        return None

    age = (
        datetime.now(timezone.utc) - dt
    ).total_seconds() / 60.0

    return round(age, 2)


def radar_status(timestamp: str | None = None) -> str:
    timestamp = timestamp or radar_timestamp()

    if not timestamp:
        return "no_disponible"

    dt = parse_iso(timestamp)

    if dt is None:
        return "no_disponible"

    age = frame_age_minutes(timestamp)

    if age is None or age < -MAX_FUTURE_SKEW_MINUTES:
        return "no_disponible"

    if age > MAX_RADAR_AGE_MINUTES:
        return "retrasado"

    return "actualizado"


def radar_metadata() -> dict[str, Any]:
    timestamp = radar_timestamp()
    age = frame_age_minutes(timestamp)
    status = radar_status(timestamp)

    return {
        "frame_actual_utc": timestamp,
        "frame_edad_minutos": age,
        "estado_radar": status,
        "radar_actualizado": status == "actualizado",
        "radar_retrasado": status == "retrasado",
        "radar_disponible": status != "no_disponible",
        "edad_maxima_procesamiento_minutos": MAX_RADAR_AGE_MINUTES,
    }


def png_is_valid(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "archivo_no_existe"

    try:
        if path.stat().st_size <= 1024:
            return False, "archivo_demasiado_pequeno"
    except OSError:
        return False, "no_se_pudo_leer_tamanio"

    if Image is None:
        return False, "Pillow_no_disponible"

    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                return False, "formato_no_png"

            image.verify()

        return True, "ok"

    except Exception:
        return False, "png_invalido"


def get_openmeteo_data() -> dict[str, Any] | None:
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={LAT}&longitude={LON}"
        "&current=temperature_2m,dewpoint_2m,pressure_msl,"
        "wind_speed_10m,wind_direction_10m,precipitation"
        "&timezone=America/Argentina/Buenos_Aires"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        payload = response.json()
        current = payload.get("current")

        if not isinstance(current, dict):
            return None

        return {
            "temperatura": current.get("temperature_2m"),
            "punto_rocio": current.get("dewpoint_2m"),
            "presion": current.get("pressure_msl"),
            "viento_vel": current.get("wind_speed_10m"),
            "viento_dir": current.get("wind_direction_10m"),
            "precipitacion": current.get("precipitation"),
            "fuente": "Open-Meteo",
        }

    except Exception:
        return None


def get_nowcast() -> dict[str, Any]:
    return read_json(NOWCAST_FILE) or {}


def get_nucleos(nowcast: dict[str, Any]) -> list[dict[str, Any]]:
    block = nowcast.get("nowcast")

    if not isinstance(block, dict):
        return []

    value = block.get("nucleos")

    return value if isinstance(value, list) else []


def max_intensity(nowcast: dict[str, Any]) -> int | None:
    block = nowcast.get("nowcast")

    if not isinstance(block, dict):
        return None

    value = block.get("intensidad_nivel_max")

    if isinstance(value, (int, float)):
        return int(value)

    values = [
        item.get("intensidad")
        for item in get_nucleos(nowcast)
        if (
            isinstance(item, dict)
            and isinstance(item.get("intensidad"), (int, float))
        )
    ]

    return int(max(values)) if values else None


def intensity_label(level: int | None) -> str:
    return {
        0: "Sin precipitación",
        1: "Muy débil",
        2: "Débil",
        3: "Moderada",
        4: "Fuerte",
        5: "Muy fuerte",
        6: "Severa",
    }.get(level, "Sin datos")


def load_history_list() -> list[tuple[str, Path]]:
    if not HISTORY_DIR.exists():
        return []

    items = []

    for path in HISTORY_DIR.glob("*.png"):
        if len(path.stem) == 14 and path.stem.isdigit():
            items.append((path.stem, path))

    items.sort(key=lambda item: item[0])

    return items[-13:]


def synchronized_frame_status() -> dict[str, Any]:
    radar_ts = radar_timestamp()

    nowcast = get_nowcast().get("nowcast", {})
    ml = read_json(ML_PREDICTION_FILE) or {}

    nowcast_ts = (
        nowcast.get("frame_actual_utc")
        if isinstance(nowcast, dict)
        else None
    )

    ml_ts = ml.get("frame_radar")

    radar_dt = parse_iso(radar_ts)
    nowcast_dt = parse_iso(nowcast_ts)
    ml_dt = parse_iso(ml_ts)

    equal = bool(
        radar_dt is not None
        and nowcast_dt is not None
        and ml_dt is not None
        and radar_dt == nowcast_dt == ml_dt
    )

    return {
        "radar": radar_ts,
        "nowcast": nowcast_ts,
        "ml": ml_ts,
        "sincronizados": equal,
    }


def result_payload() -> dict[str, Any]:
    storm = read_json(STORM_FILE)

    if storm is not None:
        storm = dict(storm)

        storm["interpretacion_probabilidades"] = (
            "heuristicas_no_calibradas"
        )

        storm["probabilidades_calibradas"] = False

        storm["nota"] = (
            "Los campos probabilidad/peligros del Storm Engine son "
            "estimaciones heurísticas; no representan probabilidades "
            "científicas calibradas."
        )

    return {
        "radar_nowcast": get_nowcast(),
        "resultado_procesamiento": read_json(RESULTADO_FILE),
        "resultado_final": read_json(DECISION_FILE),
        "inteligencia_tormenta": storm,
        "estado_meteorologico": read_json(METEO_FILE),
        "prediccion_evolucion": read_json(PREDICTION_FILE),
        "sincronizacion": synchronized_frame_status(),
        **radar_metadata(),
    }


@app.get("/")
def inicio() -> dict[str, Any]:
    return {
        "app": "ClimaAR",
        "estado": (
            "operativo"
            if radar_status() == "actualizado"
            else "degradado"
        ),
        "version": VERSION,
        "fuente_radar": FUENTE,
        "ciudad": CITY,
        "radar": "RMA10",
        **radar_metadata(),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    metadata = radar_metadata()

    required = {
        "timestamp": TIMESTAMP_FILE.is_file(),
        "radar_nowcast": NOWCAST_FILE.is_file(),
        "imagen": ACTUAL_IMAGE.is_file(),
    }

    png_ok, png_reason = png_is_valid(ACTUAL_IMAGE)

    required["imagen_png_valido"] = png_ok

    status = metadata["estado_radar"]

    if status == "actualizado" and all(required.values()):
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
        "archivos": required,
        "imagen_validacion": png_reason,
        "sincronizacion": synchronized_frame_status(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/estado")
def estado() -> dict[str, Any]:
    nowcast = get_nowcast()
    nc = nowcast.get("nowcast", {})

    if not isinstance(nc, dict):
        nc = {}

    meteo = get_openmeteo_data()
    timestamp = radar_timestamp()
    age = frame_age_minutes(timestamp)
    level = max_intensity(nowcast)
    nucleos = get_nucleos(nowcast)

    speed = nc.get("velocidad_kmh")
    direction = nc.get("direccion")

    movement = "Sin movimiento confirmado"

    if isinstance(speed, (int, float)) and speed > 0:
        movement = f"{speed:.1f} km/h"

        if direction:
            movement += f" - {direction}"

    return {
        "temperatura": meteo.get("temperatura") if meteo else None,
        "punto_rocio": meteo.get("punto_rocio") if meteo else None,
        "viento": meteo.get("viento_vel") if meteo else None,
        "viento_direccion": meteo.get("viento_dir") if meteo else None,
        "presion": meteo.get("presion") if meteo else None,
        "precipitacion": meteo.get("precipitacion") if meteo else None,
        "meteorologia_externa": (
            "disponible" if meteo else "no_disponible"
        ),
        "cobertura_radar": bool(nucleos),
        "intensidad_nivel": level,
        "intensidad": intensity_label(level),
        "nucleos": len(nucleos),
        "actualizado": timestamp,
        "frame_edad_minutos": age,
        "estado_radar": radar_status(timestamp),
        "nowcast": nc.get(
            "estado",
            nowcast.get("estado", "sin datos"),
        ),
        "movimiento": movement,
        "movimiento_hacia_bahia": nc.get(
            "movimiento_hacia_bahia"
        ),
        "eta_minutos": nc.get("eta_minutos"),
        "fuente": FUENTE,
        "version": VERSION,
    }


@app.get("/radar")
def radar() -> dict[str, Any]:
    nowcast = get_nowcast()
    nc = nowcast.get("nowcast", {})

    if not isinstance(nc, dict):
        nc = {}

    metadata = radar_metadata()

    return {
        "estado": "ok" if nowcast else "sin_datos",
        "fuente": FUENTE,
        "radar": "RMA10",
        **metadata,
        "imagen_local": "/radar/imagen",
        "frames_historicos": len(load_history_list()),
        "frames_analizados": nc.get("frames_analizados"),
        "frames_analizados_significado": (
            "valor generado por el motor de nowcast; "
            "no asumir que representa cantidad de imágenes "
            "sin auditar su productor"
        ),
        "nucleos": get_nucleos(nowcast),
        "intensidad_nivel_max": max_intensity(nowcast),
        "intensidad": intensity_label(max_intensity(nowcast)),
        "sincronizacion": synchronized_frame_status(),
    }


@app.get("/radar/imagen")
def radar_imagen():
    valid, reason = png_is_valid(ACTUAL_IMAGE)

    if not valid:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error": "imagen_radar_no_disponible",
                "motivo": reason,
                **radar_metadata(),
            },
        )

    return FileResponse(
        ACTUAL_IMAGE,
        media_type="image/png",
        filename="radar.png",
    )


@app.get("/secuencia")
def secuencia() -> dict[str, Any]:
    history = load_history_list()
    frames = []

    for timestamp, path in history:
        valid, _ = png_is_valid(path)

        if not valid:
            continue

        frames.append(
            {
                "timestamp": timestamp,
                "imagen": f"/radar/historico/{timestamp}",
            }
        )

    return {
        "ok": True,
        "cantidad": len(frames),
        "frames": frames,
        "frame_actual_utc": radar_timestamp(),
        "estado_radar": radar_status(),
    }


@app.get("/radar/historico/{timestamp}")
def radar_historico(timestamp: str):
    if not timestamp.isdigit() or len(timestamp) != 14:
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": "timestamp_invalido",
            },
        )

    path = HISTORY_DIR / f"{timestamp}.png"
    valid, reason = png_is_valid(path)

    if not valid:
        return JSONResponse(
            status_code=404,
            content={
                "ok": False,
                "error": "frame_no_disponible",
                "motivo": reason,
            },
        )

    return FileResponse(
        path,
        media_type="image/png",
        filename=f"{timestamp}.png",
    )


@app.get("/nowcast")
def nowcast() -> dict[str, Any]:
    data = get_nowcast()

    if not data:
        return {
            "ok": False,
            "error": "nowcast_no_disponible",
            **radar_metadata(),
        }

    return {
        "ok": True,
        **data,
        **radar_metadata(),
    }


@app.get("/prediccion")
def prediccion() -> dict[str, Any]:
    data = read_json(PREDICTION_FILE)

    if not data:
        return {
            "ok": False,
            "error": "prediccion_no_disponible",
            **radar_metadata(),
        }

    return {
        "ok": True,
        **data,
        **radar_metadata(),
    }


@app.get("/prediccion/en-vivo")
def prediccion_en_vivo() -> dict[str, Any]:
    data = read_json(PREDICTION_FILE)
    radar_ts = radar_timestamp()
    radar_dt = parse_iso(radar_ts)

    if not data:
        return {
            "ok": False,
            "modo": "PREDICCION_ENTRE_FRAMES",
            "error": "prediccion_no_disponible",
            "frame_radar": radar_ts,
            "nota": (
                "No hay predicción entre frames disponible."
            ),
        }

    frame_pred = (
        data.get("frame_actual_utc")
        or data.get("frame_radar")
        or data.get("timestamp")
    )

    pred_dt = parse_iso(frame_pred)
    edad_pred = None

    if radar_dt and pred_dt:
        edad_pred = (
            radar_dt - pred_dt
        ).total_seconds() / 60.0

    if (
        edad_pred is not None
        and (
            edad_pred < 0
            or edad_pred > MAX_LIVE_PREDICTION_MINUTES
        )
    ):
        return {
            "ok": False,
            "modo": "PREDICCION_ENTRE_FRAMES",
            "error": "prediccion_fuera_de_ventana",
            "frame_radar": radar_ts,
            "frame_prediccion": frame_pred,
            "edad_prediccion_minutos": round(
                edad_pred,
                2,
            ),
            "ventana_maxima_minutos": (
                MAX_LIVE_PREDICTION_MINUTES
            ),
            "nota": (
                "Esta salida no representa una nueva "
                "observación radar."
            ),
        }

    return {
        "ok": True,
        "modo": "PREDICCION_ENTRE_FRAMES",
        "frame_radar": radar_ts,
        "frame_prediccion": frame_pred,
        "edad_prediccion_minutos": (
            round(edad_pred, 2)
            if edad_pred is not None
            else None
        ),
        "ventana_maxima_minutos": (
            MAX_LIVE_PREDICTION_MINUTES
        ),
        "es_observacion_radar": False,
        "es_prediccion": True,
        "nota": (
            "Predicción entre frames basada en la "
            "evolución estimada del radar. No representa "
            "una nueva observación radar."
        ),
        "datos": data,
    }


@app.get("/tormenta")
def tormenta() -> dict[str, Any]:
    data = read_json(STORM_FILE)

    if not data:
        return {
            "ok": False,
            "error": "inteligencia_tormenta_no_disponible",
            **radar_metadata(),
        }

    result = dict(data)
    result["probabilidades_calibradas"] = False
    result["tipo_probabilidades"] = "heuristicas_no_calibradas"
    result["advertencia"] = (
        "Las probabilidades del Storm Engine son "
        "estimaciones heurísticas y no están calibradas "
        "como probabilidades meteorológicas verificadas."
    )
    result["frame_radar"] = (
        result.get("frame_radar")
        or radar_timestamp()
    )

    return {
        "ok": True,
        "datos": result,
        **radar_metadata(),
    }


@app.get("/modelo")
def modelo() -> dict[str, Any]:
    data = read_json(ML_PREDICTION_FILE)

    if not data:
        return {
            "ok": False,
            "error": "modelo_ml_no_disponible",
            **radar_metadata(),
        }

    return {
        "ok": bool(data.get("ok")),
        "modelo": data.get("modelo", "rma10-v1"),
        "frame_radar": data.get("frame_radar"),
        "horizontes_minutos": data.get(
            "horizontes_minutos",
            [5, 10, 20, 30],
        ),
        "tracks_analizados": data.get("tracks_analizados"),
        "predicciones": data.get("predicciones", []),
        "calibracion": data.get("calibracion"),
        **radar_metadata(),
    }


@app.get("/modelo/nowcast")
def modelo_nowcast() -> dict[str, Any]:
    ml = read_json(ML_PREDICTION_FILE)

    if not ml:
        return {
            "ok": False,
            "error": "modelo_ml_no_disponible",
            **radar_metadata(),
        }

    predictions = ml.get("predicciones", [])

    if not isinstance(predictions, list):
        predictions = []

    return {
        "ok": bool(ml.get("ok")),
        "modelo": ml.get("modelo", "rma10-v1"),
        "frame_radar": ml.get("frame_radar"),
        "tracks_analizados": ml.get(
            "tracks_analizados",
            len(predictions),
        ),
        "predicciones": predictions,
        "horizontes_minutos": ml.get(
            "horizontes_minutos",
            [5, 10, 20, 30],
        ),
        "nota": (
            "El modelo realiza predicción de evolución "
            "de núcleos/rastros radar. No debe interpretarse "
            "como predictor validado de tornado, granizo, "
            "ráfagas o rotación."
        ),
        **radar_metadata(),
    }


@app.get("/inteligencia")
def inteligencia() -> dict[str, Any]:
    return result_payload()


@app.get("/debug")
def debug() -> dict[str, Any]:
    files = {
        "radar_timestamp": TIMESTAMP_FILE,
        "radar_nowcast": NOWCAST_FILE,
        "resultado_procesamiento": RESULTADO_FILE,
        "decision": DECISION_FILE,
        "storm_engine": STORM_FILE,
        "meteorologia": METEO_FILE,
        "prediccion_evolucion": PREDICTION_FILE,
        "prediccion_ml": ML_PREDICTION_FILE,
        "actual_png": ACTUAL_IMAGE,
    }

    estados = {}

    for name, path in files.items():
        estados[name] = {
            "existe": path.exists(),
            "ruta": str(path.relative_to(BASE_DIR)),
        }

        if path.exists():
            try:
                estados[name]["bytes"] = path.stat().st_size
            except OSError:
                estados[name]["bytes"] = None

    return {
        "app": "ClimaAR",
        "version": VERSION,
        "fuente_radar": FUENTE,
        "radar": radar_metadata(),
        "sincronizacion": synchronized_frame_status(),
        "archivos": estados,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
