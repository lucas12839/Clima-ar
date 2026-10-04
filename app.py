from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from datetime import datetime, timezone
import requests
import json
import math
import os

app = FastAPI(
    title="ClimaAR",
    version="3.0.0",
    description="Seguimiento de tormentas y nowcasting para Bahía Blanca."
)

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RADAR_DIR = BASE_DIR / "data" / "radar"
RADAR_IMAGE = RADAR_DIR / "actual.png"
RADAR_STATUS = RADAR_DIR / "status.json"

OBS_STATUS = BASE_DIR / "data" / "sazb" / "status.json"

MODEL_PATH = BASE_DIR / "modelo" / "climaar_modelo_dia_severo.joblib"

RAINVIEWER_API = "https://api.rainviewer.com/public/weather-maps.json"

LAT = -38.71
LON = -62.26

REQUEST_TIMEOUT = 20


# ============================================================
# FUNCIONES GENERALES
# ============================================================

def ahora_utc():
    return datetime.now(timezone.utc).isoformat()


def leer_json(path):
    try:
        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return None


def guardar_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# OBSERVACIÓN SAZB
# ============================================================

def leer_observacion_sazb():
    data = leer_json(OBS_STATUS)

    if not isinstance(data, dict):
        return {
            "integrado": False,
            "observacion_valida": False,
            "fuente": "Aviation Weather Center",
            "estacion": "SAZB",
            "estacion_nombre": "Bahia Blanca Aero",
            "motivo": "No existe status.json"
        }

    return data


# ============================================================
# RAINVIEWER
# ============================================================

def obtener_rainviewer():
    try:
        r = requests.get(
            RAINVIEWER_API,
            timeout=REQUEST_TIMEOUT
        )

        r.raise_for_status()

        data = r.json()

        radar = data.get("radar", {})

        past = radar.get("past", [])

        if not past:
            raise RuntimeError("RainViewer no devolvió frames históricos.")

        latest = past[-1]

        return {
            "ok": True,
            "timestamp": latest.get("time"),
            "path": latest.get("path"),
            "host": latest.get("host"),
            "generated": data.get("generated"),
            "frames_disponibles": len(past),
            "nowcast_disponible": bool(radar.get("nowcast"))
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }


# ============================================================
# TILES DEL RADAR
# ============================================================

def descargar_tile(host, path, z, x, y, destino):
    url = f"https://{host}{path}/{z}/{x}/{y}/256/1/1_1.png"

    r = requests.get(
        url,
        timeout=REQUEST_TIMEOUT
    )

    r.raise_for_status()

    destino.write_bytes(r.content)


def latlon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)

    n = 2.0 ** zoom

    xtile = int(
        (lon + 180.0) / 360.0 * n
    )

    ytile = int(
        (
            1.0
            - math.asinh(math.tan(lat_rad)) / math.pi
        )
        / 2.0
        * n
    )

    return xtile, ytile


def descargar_radar():
    info = obtener_rainviewer()

    if not info.get("ok"):
        raise RuntimeError(
            info.get("error", "No se pudo consultar RainViewer.")
        )

    host = info["host"]
    path = info["path"]
    timestamp = info["timestamp"]

    zoom = 7

    center_x, center_y = latlon_to_tile(
        LAT,
        LON,
        zoom
    )

    # Imagen 3 x 3 tiles.
    tiles = []

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):

            x = center_x + dx
            y = center_y + dy

            tiles.append(
                {
                    "x": x,
                    "y": y,
                    "dx": dx,
                    "dy": dy
                }
            )

    try:
        from PIL import Image

        canvas = Image.new(
            "RGBA",
            (768, 768)
        )

        tiles_ok = 0

        for tile in tiles:

            url = (
                f"https://{host}"
                f"{path}/{zoom}/"
                f"{tile['x']}/{tile['y']}/"
                f"256/1/1_1.png"
            )

            response = requests.get(
                url,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            from io import BytesIO

            image = Image.open(
                BytesIO(response.content)
            ).convert("RGBA")

            px = (tile["dx"] + 1) * 256
            py = (tile["dy"] + 1) * 256

            canvas.alpha_composite(
                image,
                (px, py)
            )

            tiles_ok += 1

        RADAR_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        canvas.save(
            RADAR_IMAGE
        )

        status = {
            "version": "3.0",
            "actualizado_utc": ahora_utc(),
            "ubicacion": {
                "ciudad": "Bahia Blanca",
                "latitud": LAT,
                "longitud": LON
            },
            "radar": {
                "fuente": "RainViewer",
                "timestamp": timestamp,
                "timestamp_utc": datetime.fromtimestamp(
                    timestamp,
                    timezone.utc
                ).isoformat(),
                "host": host,
                "path": path,
                "zoom": zoom,
                "tiles_ok": tiles_ok,
                "tiles_total": 9
            }
        }

        guardar_json(
            RADAR_STATUS,
            status
        )

        return status

    except Exception as e:

        raise RuntimeError(
            f"No se pudo descargar el radar: {e}"
        )


# ============================================================
# INFORMACIÓN DEL MODELO
# ============================================================

def modelo_info():

    if not MODEL_PATH.exists():
        return {
            "disponible": False,
            "archivo": str(MODEL_PATH),
            "motivo": "Modelo todavía no disponible."
        }

    try:
        import joblib

        modelo = joblib.load(
            MODEL_PATH
        )

        return {
            "disponible": True,
            "archivo": str(MODEL_PATH),
            "tipo": type(modelo).__name__,
            "tamano_bytes": MODEL_PATH.stat().st_size
        }

    except Exception as e:

        return {
            "disponible": False,
            "archivo": str(MODEL_PATH),
            "motivo": str(e)
        }


# ============================================================
# ESTADO DEL RADAR
# ============================================================

def radar_status_local():

    data = leer_json(
        RADAR_STATUS
    )

    if isinstance(data, dict):
        return data

    return {
        "disponible": False,
        "motivo": "Todavía no existe status.json"
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "servicio": "ClimaAR",
        "version": "3.0.0",
        "estado": "ok",
        "ubicacion": "Bahia Blanca",
        "radar": "RainViewer",
        "observacion": "/observacion",
        "radar_status": "/radar/status",
        "radar_imagen": "/radar",
        "estado_general": "/estado",
        "modelo": "/modelo",
        "tormenta": "/tormenta",
        "nowcast": "/nowcast",
        "health": "/health"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    rainviewer = obtener_rainviewer()

    observacion = leer_observacion_sazb()

    modelo = modelo_info()

    return {
        "estado": "ok",
        "servicio": "ClimaAR",
        "version": "3.0.0",

        "fuente_radar": "RainViewer",

        "rainviewer": rainviewer,

        "observacion_sazb": observacion,

        "modelo": modelo,

        "hora_utc": ahora_utc()
    }


# ============================================================
# OBSERVACIÓN
# ============================================================

@app.get("/observacion")
def observacion():

    return leer_observacion_sazb()


# ============================================================
# RADAR STATUS
# ============================================================

@app.get("/radar/status")
def radar_status():

    info = obtener_rainviewer()

    local = radar_status_local()

    return {
        "fuente": "RainViewer",
        "online": info.get("ok", False),
        "rainviewer": info,
        "ultimo_procesamiento_local": local
    }


# ============================================================
# RADAR
# ============================================================

@app.get("/radar")
def radar():

    try:

        status = descargar_radar()

        if not RADAR_IMAGE.exists():

            raise HTTPException(
                status_code=503,
                detail="Radar actualizado pero imagen no disponible."
            )

        return FileResponse(
            RADAR_IMAGE,
            media_type="image/png",
            headers={
                "Cache-Control": "no-store"
            }
        )

    except Exception as e:

        raise HTTPException(
            status_code=503,
            detail=str(e)
        )


# ============================================================
# ESTADO GENERAL
# ============================================================

@app.get("/estado")
def estado():

    radar = obtener_rainviewer()

    observacion = leer_observacion_sazb()

    modelo = modelo_info()

    return {
        "servicio": "ClimaAR",
        "estado": "operativo",

        "ubicacion": {
            "ciudad": "Bahia Blanca",
            "latitud": LAT,
            "longitud": LON
        },

        "radar": {
            "fuente": "RainViewer",
            "disponible": radar.get("ok", False),
            "datos": radar
        },

        "observacion_sazb": observacion,

        "modelo": modelo,

        "actualizado_utc": ahora_utc()
    }


# ============================================================
# MODELO
# ============================================================

@app.get("/modelo")
def modelo():

    return modelo_info()


# ============================================================
# ANÁLISIS BÁSICO DE TORMENTA
# ============================================================

@app.get("/tormenta")
def tormenta():

    radar = obtener_rainviewer()

    observacion = leer_observacion_sazb()

    if not radar.get("ok"):

        return {
            "disponible": False,
            "fuente_radar": "RainViewer",
            "motivo": radar.get(
                "error",
                "Radar no disponible."
            )
        }

    temperatura = observacion.get(
        "temperatura_c"
    )

    punto_rocio = observacion.get(
        "punto_rocio_c"
    )

    viento = observacion.get(
        "viento_kt"
    )

    humedad_proxy = None

    if (
        isinstance(temperatura, (int, float))
        and isinstance(punto_rocio, (int, float))
    ):

        diferencia = temperatura - punto_rocio

        if diferencia <= 2:
            humedad_proxy = "muy_alta"

        elif diferencia <= 5:
            humedad_proxy = "alta"

        elif diferencia <= 8:
            humedad_proxy = "moderada"

        else:
            humedad_proxy = "baja"

    return {

        "disponible": True,

        "fuente_radar": "RainViewer",

        "timestamp_radar": radar.get(
            "timestamp"
        ),

        "timestamp_radar_utc": (
            datetime.fromtimestamp(
                radar["timestamp"],
                timezone.utc
            ).isoformat()
            if radar.get("timestamp")
            else None
        ),

        "observacion_sazb": {

            "temperatura_c": temperatura,

            "punto_rocio_c": punto_rocio,

            "viento_kt": viento,

            "humedad_proxy": humedad_proxy
        },

        "nota": (
            "Este endpoint entrega información "
            "operativa disponible. No constituye "
            "una predicción meteorológica definitiva."
        )
    }


# ============================================================
# NOWCAST
# ============================================================

@app.get("/nowcast")
def nowcast():

    radar = obtener_rainviewer()

    modelo = modelo_info()

    if not radar.get("ok"):

        return {
            "disponible": False,
            "motivo": "Radar no disponible.",
            "radar": radar
        }

    return {

        "disponible": True,

        "fuente_radar": "RainViewer",

        "radar": {
            "timestamp": radar.get(
                "timestamp"
            ),

            "frames_disponibles": radar.get(
                "frames_disponibles"
            ),

            "nowcast_rainviewer": radar.get(
                "nowcast_disponible"
            )
        },

        "modelo_climaar": modelo,

        "prediccion": {

            "estado": "experimental",

            "mensaje": (
                "El sistema dispone del flujo "
                "de radar en tiempo real y del "
                "modelo cuando se encuentra "
                "entrenado. La predicción IA "
                "avanzada se habilita cuando "
                "el modelo correspondiente "
                "está disponible."
            )
        },

        "actualizado_utc": ahora_utc()
    }


# ============================================================
# MANUAL: ACTUALIZAR RADAR
# ============================================================

@app.post("/radar/actualizar")
def actualizar_radar():

    try:

        status = descargar_radar()

        return {
            "ok": True,
            "mensaje": "Radar actualizado correctamente.",
            "status": status
        }

    except Exception as e:

        raise HTTPException(
            status_code=503,
            detail=str(e)
        )


# ============================================================
# EJECUCIÓN LOCAL
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port
    )
