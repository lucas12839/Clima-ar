from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

try:
    import joblib
except Exception:
    joblib = None

try:
    from PIL import Image
except Exception:
    Image = None


app = FastAPI(
    title="ClimaAR",
    description="Seguimiento operativo de precipitacion para Bahia Blanca.",
    version="3.0.0",
)


# ============================================================
# CONFIGURACION
# ============================================================

LAT = -38.71
LON = -62.26
ZOOM = 7

RAINVIEWER_API = (
    "https://api.rainviewer.com/public/weather-maps.json"
)

RAINVIEWER_DEFAULT_HOST = (
    "https://tilecache.rainviewer.com"
)

DATA_DIR = Path("data/radar")

RADAR_IMAGE = DATA_DIR / "actual.png"
RADAR_STATUS = DATA_DIR / "status.json"

OBS_STATUS = Path(
    "data/sazb/status.json"
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

_session = requests.Session()

_session.headers.update(
    {
        "User-Agent": "ClimaAR/3.0"
    }
)


# ============================================================
# FUNCIONES GENERALES
# ============================================================

def now_utc() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def latlon_to_tile(
    lat: float,
    lon: float,
    zoom: int
) -> tuple[int, int]:

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
# OBSERVACION SAZB
# ============================================================

def leer_observacion_sazb() -> dict:

    if not OBS_STATUS.exists():

        return {
            "integrado": False,
            "observacion_valida": False,
            "fuente": "Aviation Weather Center",
            "estacion": "SAZB",
            "error": "Sin status SAZB",
        }

    try:

        return json.loads(
            OBS_STATUS.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:

        return {
            "integrado": False,
            "observacion_valida": False,
            "fuente": "Aviation Weather Center",
            "estacion": "SAZB",
            "error": (
                f"status SAZB invalido: {exc}"
            ),
        }


# ============================================================
# RAINVIEWER - ULTIMO FRAME
# ============================================================

def rainviewer_latest() -> dict:

    response = _session.get(
        RAINVIEWER_API,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    radar = data.get(
        "radar"
    ) or {}

    frames = (
        radar.get("past")
        or []
    )

    if not frames:

        raise RuntimeError(
            "RainViewer no devolvio frames radar."
        )

    frame = frames[-1]

    timestamp = int(
        frame.get(
            "time",
            0
        )
    )

    path = frame.get(
        "path"
    )

    # IMPORTANTE:
    # RainViewer entrega host en el objeto raiz,
    # no dentro del frame.
    host = (
        data.get("host")
        or RAINVIEWER_DEFAULT_HOST
    ).rstrip("/")

    if timestamp <= 0:

        raise RuntimeError(
            "Timestamp RainViewer invalido."
        )

    if not path:

        raise RuntimeError(
            "Path RainViewer invalido."
        )

    generated = data.get(
        "generated"
    )

    return {

        "fuente": "RainViewer",

        "timestamp": timestamp,

        "timestamp_utc": (
            datetime.fromtimestamp(
                timestamp,
                tz=timezone.utc
            ).isoformat()
        ),

        "host": host,

        "path": path,

        "frames_disponibles": len(
            frames
        ),

        "generado_utc": (
            datetime.fromtimestamp(
                int(generated),
                tz=timezone.utc
            ).isoformat()
            if generated
            else None
        ),

        "latitud": LAT,

        "longitud": LON,

        "zoom": ZOOM,
    }


# ============================================================
# DESCARGA Y ARMADO DEL RADAR
# ============================================================

def descargar_radar() -> dict:

    if Image is None:

        raise RuntimeError(
            "Pillow no esta disponible."
        )

    info = rainviewer_latest()

    xt, yt = latlon_to_tile(
        LAT,
        LON,
        ZOOM
    )

    # 3 x 3 teselas de 512 px
    image = Image.new(
        "RGBA",
        (
            1536,
            1536
        ),
        (
            0,
            0,
            0,
            0
        )
    )

    tiles_ok = 0

    for dx in (-1, 0, 1):

        for dy in (-1, 0, 1):

            # Formato actual de RainViewer:
            # host + path + /size/z/x/y/color/options.png
            url = (
                f"{info['host']}"
                f"{info['path']}"
                f"/512/{ZOOM}/"
                f"{xt + dx}/"
                f"{yt + dy}/"
                f"2/1_1.png"
            )

            try:

                r = _session.get(
                    url,
                    timeout=15
                )

                if (
                    r.status_code != 200
                    or len(r.content) < 200
                ):
                    continue

                tile = Image.open(
                    BytesIO(
                        r.content
                    )
                ).convert(
                    "RGBA"
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
            "No se pudo descargar ninguna "
            "tesela de RainViewer."
        )

    image.save(
        RADAR_IMAGE
    )

    estado = {

        "version": "3.1",

        "actualizado_utc": now_utc(),

        "ubicacion": {

            "ciudad": "Bahia Blanca",

            "latitud": LAT,

            "longitud": LON,
        },

        "estado": "radar_disponible",

        "radar": {

            **info,

            "tiles_ok": tiles_ok,

            "tiles_total": 9,

            "imagen": "/radar",
        },

        "observacion_sazb":
            leer_observacion_sazb(),

        "nota": (
            "RainViewer es la fuente primaria "
            "del radar. Las observaciones SAZB "
            "son complementarias."
        ),
    }

    RADAR_STATUS.write_text(

        json.dumps(
            estado,
            ensure_ascii=False,
            indent=2
        ),

        encoding="utf-8"
    )

    return estado


# ============================================================
# ESTADO GUARDADO
# ============================================================

def leer_estado_guardado() -> dict:

    if not RADAR_STATUS.exists():

        return {

            "estado": "sin_estado",

            "radar": {
                "fuente": "RainViewer"
            },

            "observacion_sazb":
                leer_observacion_sazb(),
        }

    try:

        return json.loads(
            RADAR_STATUS.read_text(
                encoding="utf-8"
            )
        )

    except Exception:

        return {

            "estado": "estado_invalido",

            "radar": {
                "fuente": "RainViewer"
            },

            "observacion_sazb":
                leer_observacion_sazb(),
        }


# ============================================================
# ESTADO RADAR EN VIVO
# ============================================================

def radar_status_live() -> dict:

    try:

        return rainviewer_latest()

    except Exception as exc:

        saved = leer_estado_guardado()

        saved[
            "error_radar_live"
        ] = str(exc)

        return saved


# ============================================================
# MODELO IA
# ============================================================

def modelo_info() -> dict:

    model_path = Path(
        "modelo/climaar_modelo_dia_severo.joblib"
    )

    summary_path = Path(
        "modelo/resumen_modelo.json"
    )

    if (
        joblib is None
        or not model_path.exists()
    ):

        return {

            "disponible": False,

            "motivo":
                "Modelo no disponible "
                "en el despliegue.",
        }

    result = {

        "disponible": True,

        "archivo":
            str(model_path),
    }

    try:

        model = joblib.load(
            model_path
        )

        if isinstance(
            model,
            dict
        ):

            result["tipo"] = model.get(
                "tipo"
            )

            result["version"] = model.get(
                "version"
            )

    except Exception as exc:

        return {

            "disponible": False,

            "motivo": (
                "No se pudo cargar el modelo: "
                f"{exc}"
            ),
        }

    if summary_path.exists():

        try:

            result["resumen"] = json.loads(
                summary_path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:

            pass

    return result


# ============================================================
# INICIO
# ============================================================

@app.get("/")
def inicio():

    return {

        "servicio": "ClimaAR",

        "estado": "ok",

        "version": app.version,

        "ubicacion": "Bahia Blanca",

        "fuente_radar": "RainViewer",

        "endpoints": {

            "health": "/health",

            "estado": "/estado",

            "radar": "/radar",

            "radar_status":
                "/radar/status",

            "observacion":
                "/observacion",

            "modelo": "/modelo",

            "tormenta": "/tormenta",

            "nowcast": "/nowcast",
        },
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "estado": "ok",

        "servicio": "ClimaAR",

        "version": app.version,

        "fuente_radar": "RainViewer",

        "radar_live":
            radar_status_live(),

        "observacion_sazb":
            leer_observacion_sazb(),

        "modelo":
            modelo_info(),

        "hora_utc":
            now_utc(),
    }


# ============================================================
# RADAR STATUS
# ============================================================

@app.get("/radar/status")
def radar_status():

    try:

        return descargar_radar()

    except Exception as exc:

        return JSONResponse(

            status_code=503,

            content={

                "estado":
                    "radar_no_disponible",

                "fuente":
                    "RainViewer",

                "error":
                    str(exc),

                "hora_utc":
                    now_utc(),
            }
        )


# ============================================================
# RADAR
# ============================================================

@app.get("/radar")
def radar():

    try:

        descargar_radar()

    except Exception:

        pass

    if not RADAR_IMAGE.exists():

        return JSONResponse(

            status_code=503,

            content={

                "estado":
                    "radar_no_disponible",

                "fuente":
                    "RainViewer",

                "error":
                    "No hay imagen radar local.",
            }
        )

    return FileResponse(

        RADAR_IMAGE,

        media_type="image/png",

        filename="climaar-radar.png",
    )


# ============================================================
# OBSERVACION
# ============================================================

@app.get("/observacion")
def observacion():

    return leer_observacion_sazb()


# ============================================================
# ESTADO GENERAL
# ============================================================

@app.get("/estado")
def estado():

    try:

        live = descargar_radar()

    except Exception:

        live = leer_estado_guardado()

    return {

        "servicio": "ClimaAR",

        "estado":
            live.get(
                "estado",
                "desconocido"
            ),

        "radar":
            live.get(
                "radar",
                {}
            ),

        "observacion_sazb":
            leer_observacion_sazb(),

        "modelo_atmosferico":
            modelo_info(),

        "actualizado_utc":
            live.get(
                "actualizado_utc",
                now_utc()
            ),
    }


# ============================================================
# MODELO
# ============================================================

@app.get("/modelo")
def modelo():

    return modelo_info()


# ============================================================
# TORMENTA
# ============================================================

@app.get("/tormenta")
def tormenta():

    actual = radar_status_live()

    radar = actual.get(
        "radar",
        {}
    )

    return {

        "estado": (

            "radar_disponible"

            if radar.get(
                "timestamp"
            )

            else "sin_datos"
        ),

        "ubicacion":
            "Bahia Blanca",

        "fuente":
            "RainViewer",

        "timestamp":
            radar.get(
                "timestamp"
            ),

        "timestamp_utc":
            radar.get(
                "timestamp_utc"
            ),

        "tiles_ok":
            radar.get(
                "tiles_ok"
            ),

        "mensaje": (
            "Informa disponibilidad "
            "y frescura del radar; "
            "no es una alerta meteorologica "
            "oficial."
        ),
    }


# ============================================================
# NOWCAST
# ============================================================

@app.get("/nowcast")
def nowcast():

    info = modelo_info()

    radar = radar_status_live()

    return {

        "estado":
            "experimental",

        "fuente_radar":
            "RainViewer",

        "radar":
            radar,

        "horizonte_minutos":
            10,

        "modelo_disponible":
            info.get(
                "disponible",
                False
            ),

        "modelo":
            info,

        "nota": (
            "El nowcast propio de ClimaAR "
            "requiere validacion historica "
            "antes de considerarse una "
            "prediccion meteorologica "
            "calibrada."
        ),
    }


# ============================================================
# EJECUCION LOCAL
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000
    )
