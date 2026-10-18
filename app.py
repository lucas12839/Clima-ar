from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

BASE_DIR = Path(__file__).resolve().parent
RADAR_DIR = BASE_DIR / "data" / "radar"
HISTORY_DIR = RADAR_DIR / "historico"

NOWCAST_FILE = RADAR_DIR / "radar_nowcast.json"
RESULTADO_FILE = RADAR_DIR / "resultado_procesamiento.json"
DECISION_FILE = BASE_DIR / "data" / "ia" / "resultado_final.json"
STORM_FILE = BASE_DIR / "data" / "ia" / "inteligencia_tormenta.json"
METEO_FILE = BASE_DIR / "data" / "ia" / "estado_meteorologico.json"
PREDICTION_FILE = RADAR_DIR / "seguimiento" / "prediccion_evolucion.json"
ML_PREDICTION_FILE = RADAR_DIR / "seguimiento" / "prediccion_ml_rma10.json"
TIMESTAMP_FILE = RADAR_DIR / "radar_timestamp.txt"
ACTUAL_IMAGE = RADAR_DIR / "actual.png"

LAT = -38.7196
LON = -62.2724
CITY = "Bahía Blanca"

VERSION = "8.0-rma10"
FUENTE = "SINARAME RMA10"


app = FastAPI(
    title="ClimaAR - Radar Bahía Blanca",
    description="ClimaAR basado exclusivamente en SINARAME/RMA10 para radar operativo.",
    version=VERSION,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        value = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(value, dict):
            return value

    except Exception:
        pass

    return None


def radar_timestamp() -> str | None:

    if TIMESTAMP_FILE.exists():
        try:
            value = TIMESTAMP_FILE.read_text(
                encoding="utf-8"
            ).strip()

            if value:
                return value

        except Exception:
            pass

    nowcast = read_json(NOWCAST_FILE) or {}

    nc = nowcast.get("nowcast", {})

    if not isinstance(nc, dict):
        nc = {}

    return (
        nc.get("frame_actual_utc")
        or nowcast.get("actualizado")
    )


def parse_iso(value: str | None) -> datetime | None:

    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

    except Exception:
        return None


def frame_age_minutes(
    timestamp: str | None,
) -> float | None:

    dt = parse_iso(timestamp)

    if dt is None:
        return None

    return max(
        0.0,
        (
            datetime.now(timezone.utc)
            - dt.astimezone(timezone.utc)
        ).total_seconds()
        / 60.0,
    )


def get_openmeteo_data() -> dict[str, Any] | None:

    try:

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
            "&timezone=America/Argentina/Buenos_Aires"
        )

        response = requests.get(
            url,
            timeout=10,
        )

        response.raise_for_status()

        current = response.json().get(
            "current",
            {},
        )

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
        }

    except Exception:
        return None


def get_nowcast() -> dict[str, Any]:
    return read_json(NOWCAST_FILE) or {}


def get_nucleos(
    nowcast: dict[str, Any],
) -> list[dict[str, Any]]:

    nc = nowcast.get(
        "nowcast",
        {},
    )

    if not isinstance(nc, dict):
        return []

    nucleos = nc.get(
        "nucleos",
        [],
    )

    if isinstance(nucleos, list):
        return nucleos

    return []


def max_intensity(
    nowcast: dict[str, Any],
) -> int | None:

    nc = nowcast.get(
        "nowcast",
        {},
    )

    if isinstance(nc, dict):

        value = nc.get(
            "intensidad_nivel_max"
        )

        if isinstance(value, (int, float)):
            return int(value)

    values = []

    for nucleus in get_nucleos(nowcast):

        if not isinstance(nucleus, dict):
            continue

        value = nucleus.get(
            "intensidad"
        )

        if isinstance(value, (int, float)):
            values.append(value)

    if values:
        return int(max(values))

    return None


def intensity_label(
    level: int | None,
) -> str:

    labels = {
        0: "Sin precipitación",
        1: "Muy débil",
        2: "Débil",
        3: "Moderada",
        4: "Fuerte",
        5: "Muy fuerte",
        6: "Severa",
    }

    return labels.get(
        level,
        "Sin datos",
    )


def load_history_list() -> list[
    tuple[str, Path]
]:

    if not HISTORY_DIR.exists():
        return []

    items = []

    for path in HISTORY_DIR.glob(
        "*.png"
    ):

        stem = path.stem

        if (
            len(stem) == 14
            and stem.isdigit()
        ):
            items.append(
                (
                    stem,
                    path,
                )
            )

    items.sort(
        key=lambda item: item[0]
    )

    return items[-13:]


def result_payload() -> dict[str, Any]:

    return {
        "radar_nowcast": get_nowcast(),
        "resultado_procesamiento": (
            read_json(
                RESULTADO_FILE
            )
        ),
        "resultado_final": (
            read_json(
                DECISION_FILE
            )
        ),
        "inteligencia_tormenta": (
            read_json(
                STORM_FILE
            )
        ),
        "estado_meteorologico": (
            read_json(
                METEO_FILE
            )
        ),
        "prediccion_evolucion": (
            read_json(
                PREDICTION_FILE
            )
        ),
    }


@app.get("/")
def inicio() -> dict[str, Any]:

    return {
        "app": "ClimaAR",
        "estado": "operativo",
        "version": VERSION,
        "fuente_radar": FUENTE,
        "ciudad": CITY,
        "radar": "RMA10",
        "mensaje": (
            "ClimaAR utiliza "
            "SINARAME/RMA10 "
            "como fuente radar "
            "operativa."
        ),
    }


@app.get("/health")
def health() -> dict[str, Any]:

    timestamp = radar_timestamp()

    age = frame_age_minutes(
        timestamp
    )

    return {
        "status": "ok",
        "app": "ClimaAR",
        "version": VERSION,
        "fuente": FUENTE,
        "radar": "RMA10",
        "ciudad": CITY,
        "lat": LAT,
        "lon": LON,
        "frame_actual_utc": timestamp,
        "frame_edad_minutos": (
            round(age, 2)
            if age is not None
            else None
        ),
        "frame_es_viejo": (
            age > 35
            if age is not None
            else None
        ),
        "timestamp": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }


@app.get("/estado")
def estado() -> dict[str, Any]:

    nowcast = get_nowcast()

    nc = nowcast.get(
        "nowcast",
        {},
    )

    if not isinstance(nc, dict):
        nc = {}

    meteo = get_openmeteo_data()

    timestamp = radar_timestamp()

    age = frame_age_minutes(
        timestamp
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
        isinstance(speed, (int, float))
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
        "temperatura": (
            meteo.get("temperatura")
            if meteo
            else None
        ),
        "punto_rocio": (
            meteo.get("punto_rocio")
            if meteo
            else None
        ),
        "viento": (
            meteo.get("viento_vel")
            if meteo
            else None
        ),
        "viento_direccion": (
            meteo.get("viento_dir")
            if meteo
            else None
        ),
        "presion": (
            meteo.get("presion")
            if meteo
            else None
        ),
        "precipitacion": (
            meteo.get("precipitacion")
            if meteo
            else None
        ),
        "cobertura_radar": bool(
            nucleos
        ),
        "intensidad_nivel": level,
        "intensidad": intensity_label(
            level
        ),
        "nucleos": len(nucleos),
        "actualizado": timestamp,
        "frame_edad_minutos": (
            round(age, 2)
            if age is not None
            else None
        ),
        "frame_es_viejo": (
            age > 35
            if age is not None
            else None
        ),
        "nowcast": nc.get(
            "estado",
            nowcast.get(
                "estado",
                "sin datos",
            ),
        ),
        "movimiento": movement,
        "movimiento_hacia_bahia": nc.get(
            "movimiento_hacia_bahia"
        ),
        "eta_minutos": nc.get(
            "eta_minutos"
        ),
        "fuente": FUENTE,
        "version": VERSION,
    }


@app.get("/radar")
def radar() -> dict[str, Any]:

    nowcast = get_nowcast()

    nc = nowcast.get(
        "nowcast",
        {},
    )

    if not isinstance(nc, dict):
        nc = {}

    return {
        "estado": (
            "ok"
            if nowcast
            else "sin_datos"
        ),
        "fuente": FUENTE,
        "radar": "RMA10",
        "timestamp": radar_timestamp(),
        "imagen_local": "/radar/imagen",
        "frames_historicos": len(
            load_history_list()
        ),
        "frames_analizados": nc.get(
            "frames_analizados"
        ),
        "actividad": nc.get(
            "actividad"
        ),
        "intensidad_nivel_max": nc.get(
            "intensidad_nivel_max"
        ),
        "intensidad_escala": nc.get(
            "intensidad_escala",
            "RMA10_0_6",
        ),
        "area_px": nc.get(
            "area_px"
        ),
        "distancia_km": nc.get(
            "distancia_km"
        ),
        "fortalecimiento": nc.get(
            "fortalecimiento"
        ),
        "cambio_area_pct": nc.get(
            "cambio_area_pct"
        ),
        "velocidad_pixeles_min": nc.get(
            "velocidad_pixeles_min"
        ),
        "direccion": nc.get(
            "direccion"
        ),
        "movimiento_hacia_bahia": nc.get(
            "movimiento_hacia_bahia"
        ),
        "eta_minutos": nc.get(
            "eta_minutos"
        ),
        "nucleos": get_nucleos(
            nowcast
        ),
    }


@app.get("/radar/imagen")
def radar_imagen():

    if not ACTUAL_IMAGE.exists():

        return JSONResponse(
            status_code=404,
            content={
                "estado": "sin_imagen",
                "fuente": FUENTE,
            },
        )

    return FileResponse(
        ACTUAL_IMAGE,
        media_type="image/png",
        filename="rma10_actual.png",
    )


@app.get("/secuencia")
def secuencia() -> dict[str, Any]:

    history = load_history_list()

    return {
        "fuente": FUENTE,
        "radar": "RMA10",
        "frames": [
            {
                "timestamp": timestamp,
                "url": (
                    f"/radar/historico/"
                    f"{timestamp}"
                ),
            }
            for timestamp, _ in history
        ],
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
                "error":
                    "timestamp_invalido"
            },
        )

    path = (
        HISTORY_DIR
        / f"{timestamp}.png"
    )

    if not path.exists():

        return JSONResponse(
            status_code=404,
            content={
                "error":
                    "frame_no_encontrado"
            },
        )

    return FileResponse(
        path,
        media_type="image/png",
        filename=(
            f"rma10_{timestamp}.png"
        ),
    )


@app.get("/nowcast")
def nowcast() -> dict[str, Any]:
    return get_nowcast()


@app.get("/prediccion")
def prediccion() -> dict[str, Any]:

    return (
        read_json(
            PREDICTION_FILE
        )
        or {
            "estado": "sin_datos",
            "fuente": FUENTE,
        }
    )


@app.get("/prediccion/en-vivo")
def prediccion_en_vivo() -> dict[str, Any]:
    radar_ts = radar_timestamp()
    age = frame_age_minutes(radar_ts)
    ml = read_json(ML_PREDICTION_FILE) or {}

    return {
        "estado": "ok" if ml else "sin_datos",
        "fuente_radar": FUENTE,
        "radar": "RMA10",
        "modo": "PREDICCION_IA_ENTRE_FRAMES",
        "ultimo_radar_real_utc": radar_ts,
        "ultimo_radar_edad_minutos": round(age, 2) if age is not None else None,
        "actualizado_utc": datetime.now(timezone.utc).isoformat(),
        "mensaje": "Predicción IA en curso · estimación entre datos de radar",
        "prediccion": ml,
    }


@app.get("/tormenta")
def tormenta() -> dict[str, Any]:

    return (
        read_json(
            STORM_FILE
        )
        or {
            "estado": "sin_datos",
            "fuente_radar": FUENTE,
        }
    )


@app.get("/modelo")
def modelo() -> dict[str, Any]:

    return (
        read_json(
            DECISION_FILE
        )
        or {
            "estado": "sin_datos",
            "fuente_radar": FUENTE,
        }
    )


@app.get("/modelo/nowcast")
def modelo_nowcast() -> dict[str, Any]:

    return {
        "fuente_radar": FUENTE,
        "radar": get_nowcast(),
        "prediccion": read_json(
            PREDICTION_FILE
        ),
        "decision": read_json(
            DECISION_FILE
        ),
    }


@app.get("/inteligencia")
def inteligencia() -> dict[str, Any]:
    return result_payload()


@app.get("/debug")
def debug() -> dict[str, Any]:

    return {
        "version": VERSION,
        "fuente_radar": FUENTE,
        "archivos": {
            "actual_png":
                ACTUAL_IMAGE.exists(),
            "radar_nowcast":
                NOWCAST_FILE.exists(),
            "resultado_procesamiento":
                RESULTADO_FILE.exists(),
            "resultado_final":
                DECISION_FILE.exists(),
            "storm_engine":
                STORM_FILE.exists(),
            "meteorologia":
                METEO_FILE.exists(),
            "prediccion":
                PREDICTION_FILE.exists(),
            "timestamp":
                TIMESTAMP_FILE.exists(),
        },
        "timestamp_radar":
            radar_timestamp(),
}
