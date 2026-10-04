from __future__ import annotations

import json
import math
from datetime import datetime, timezone, timedelta
from io import BytesIO
from pathlib import Path

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse

try:
    import joblib
except Exception:
    joblib = None

try:
    from PIL import Image
except Exception:
    Image = None


# ============================================================
# CLIMAAR
# ============================================================

app = FastAPI(
    title="ClimaAR",
    description="Seguimiento operativo de precipitacion para Bahia Blanca.",
    version="4.0.0-debug",
)

LAT = -38.71
LON = -62.26

RADAR_ZOOM = 7

RAINVIEWER_API = (
    "https://api.rainviewer.com/public/weather-maps.json"
)

RAINVIEWER_DEFAULT_HOST = (
    "https://tilecache.rainviewer.com"
)

DATA_DIR = Path("data/radar")
RADAR_IMAGE = DATA_DIR / "actual.png"
RADAR_STATUS = DATA_DIR / "status.json"

OBS_STATUS = Path("data/sazb/status.json")

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SESION HTTP
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": "ClimaAR/4.0",
    "Accept": "*/*",
})


# ============================================================
# UTILIDADES
# ============================================================

def now_utc() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def argentina_time(
    utc_timestamp: int | None
) -> str | None:

    if not utc_timestamp:
        return None

    dt = datetime.fromtimestamp(
        int(utc_timestamp),
        timezone.utc
    )

    dt_ar = dt.astimezone(
        timezone(timedelta(hours=-3))
    )

    return dt_ar.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def latlon_to_tile(
    lat: float,
    lon: float,
    zoom: int
) -> tuple[int, int]:

    n = 2 ** zoom

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
            ) / math.pi
        )
        / 2.0
        * n
    )

    return x, y


def tile_x_to_lon(
    x: int,
    zoom: int
) -> float:

    return (
        x / (2 ** zoom)
    ) * 360.0 - 180.0


def tile_y_to_lat(
    y: int,
    zoom: int
) -> float:

    n = math.pi - (
        2.0
        * math.pi
        * y
        / (2 ** zoom)
    )

    return math.degrees(
        math.atan(
            math.sinh(n)
        )
    )


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
            "error": str(exc),
        }


# ============================================================
# RAINVIEWER - API
# ============================================================

def rainviewer_latest() -> dict:

    response = session.get(
        RAINVIEWER_API,
        timeout=20,
        headers={
            "User-Agent": "ClimaAR/4.0",
            "Accept": "application/json",
            "Cache-Control": "no-cache",
        },
    )

    response.raise_for_status()

    data = response.json()

    radar = data.get(
        "radar",
        {}
    )

    frames = radar.get(
        "past",
        []
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
                timezone.utc
            ).isoformat()
        ),
        "timestamp_argentina": (
            argentina_time(timestamp)
        ),
        "host": host,
        "path": path,
        "frames_disponibles": len(frames),
        "generado_utc": (
            datetime.fromtimestamp(
                int(generated),
                timezone.utc
            ).isoformat()
            if generated
            else None
        ),
        "latitud": LAT,
        "longitud": LON,
        "zoom": RADAR_ZOOM,
    }


# ============================================================
# DIAGNOSTICO RAINVIEWER
# UNA SOLA TESELA 256x256
# ============================================================

def diagnostico_rainviewer() -> dict:

    resultado = {
        "diagnostico": (
            "RainViewer - prueba de una "
            "tesela 256x256"
        ),
        "hora_utc": now_utc(),
        "api": {},
        "tile": {},
        "resultado": "INICIANDO",
    }

    try:

        # ----------------------------------------------------
        # 1. API
        # ----------------------------------------------------

        response_api = session.get(
            RAINVIEWER_API,
            timeout=20,
            headers={
                "User-Agent": "ClimaAR/4.0",
                "Accept": "application/json",
                "Cache-Control": "no-cache",
            },
        )

        resultado["api"]["status_code"] = (
            response_api.status_code
        )

        resultado["api"]["content_type"] = (
            response_api.headers.get(
                "content-type"
            )
        )

        resultado["api"]["bytes"] = (
            len(response_api.content)
        )

        response_api.raise_for_status()

        data = response_api.json()

        resultado["api"]["generated"] = (
            data.get("generated")
        )

        resultado["api"]["host"] = (
            data.get("host")
        )

        frames = (
            data
            .get("radar", {})
            .get("past", [])
        )

        resultado["api"]["frames"] = len(
            frames
        )

        if not frames:

            raise RuntimeError(
                "La API no devolvio frames radar."
            )

        frame = frames[-1]

        ts = int(
            frame.get(
                "time",
                0
            )
        )

        path = frame.get(
            "path"
        )

        resultado["api"]["frame_time"] = ts

        if ts:

            resultado["api"]["frame_utc"] = (
                datetime.fromtimestamp(
                    ts,
                    timezone.utc
                ).isoformat()
            )

            resultado["api"]["frame_argentina"] = (
                argentina_time(ts)
            )

        resultado["api"]["path"] = path

        if not path:

            raise RuntimeError(
                "El frame no contiene path."
            )

        # ----------------------------------------------------
        # 2. CALCULO DE TESELA CENTRAL
        # ----------------------------------------------------

        cx, cy = latlon_to_tile(
            LAT,
            LON,
            RADAR_ZOOM
        )

        host = (
            data.get("host")
            or RAINVIEWER_DEFAULT_HOST
        ).rstrip("/")

        tile_url = (
            f"{host}"
            f"{path}"
            f"/256/"
            f"{RADAR_ZOOM}/"
            f"{cx}/"
            f"{cy}/"
            f"2/1_1.png"
        )

        resultado["tile"]["x"] = cx
        resultado["tile"]["y"] = cy
        resultado["tile"]["zoom"] = RADAR_ZOOM
        resultado["tile"]["size"] = 256
        resultado["tile"]["url"] = tile_url

        # ----------------------------------------------------
        # 3. DESCARGA DE UNA SOLA TESELA
        # ----------------------------------------------------

        tile_response = session.get(
            tile_url,
            timeout=20,
            headers={
                "User-Agent": "ClimaAR/4.0",
                "Referer": "https://www.rainviewer.com/",
                "Accept": (
                    "image/png,image/*;"
                    "q=0.8,*/*;q=0.5"
                ),
                "Cache-Control": "no-cache",
            },
        )

        resultado["tile"]["status_code"] = (
            tile_response.status_code
        )

        resultado["tile"]["content_type"] = (
            tile_response.headers.get(
                "content-type"
            )
        )

        resultado["tile"]["bytes"] = (
            len(tile_response.content)
        )

        resultado["tile"]["server"] = (
            tile_response.headers.get(
                "server"
            )
        )

        resultado["tile"]["content_encoding"] = (
            tile_response.headers.get(
                "content-encoding"
            )
        )

        resultado["tile"]["first_bytes_hex"] = (
            tile_response.content[:32].hex()
        )

        # ----------------------------------------------------
        # 4. HTTP
        # ----------------------------------------------------

        if tile_response.status_code != 200:

            resultado["tile"]["imagen_valida"] = False

            resultado["tile"]["error"] = (
                f"HTTP "
                f"{tile_response.status_code}"
            )

            resultado["resultado"] = (
                "FALLO_TILE_HTTP"
            )

            return resultado

        # ----------------------------------------------------
        # 5. PIL
        # ----------------------------------------------------

        if Image is None:

            resultado["tile"]["imagen_valida"] = False

            resultado["tile"]["error"] = (
                "Pillow no esta disponible."
            )

            resultado["resultado"] = (
                "FALLO_PILLOW"
            )

            return resultado

        try:

            imagen = Image.open(
                BytesIO(
                    tile_response.content
                )
            )

            imagen.load()

            resultado["tile"]["imagen_valida"] = True

            resultado["tile"]["formato"] = (
                imagen.format
            )

            resultado["tile"]["modo"] = (
                imagen.mode
            )

            resultado["tile"]["ancho"] = (
                imagen.width
            )

            resultado["tile"]["alto"] = (
                imagen.height
            )

            resultado["resultado"] = (
                "TESela_OK"
            )

        except Exception as exc:

            resultado["tile"]["imagen_valida"] = False

            resultado["tile"]["error"] = str(
                exc
            )

            resultado["resultado"] = (
                "FALLO_IMAGEN"
            )

    except requests.RequestException as exc:

        resultado["resultado"] = (
            "FALLO_RED"
        )

        resultado["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    except Exception as exc:

        resultado["resultado"] = (
            "FALLO_API"
        )

        resultado["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    return resultado


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

    except Exception as exc:

        return {
            "estado": "estado_invalido",
            "radar": {
                "fuente": "RainViewer"
            },
            "error": str(exc),
            "observacion_sazb":
                leer_observacion_sazb(),
        }


# ============================================================
# MODELO
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
            "motivo": (
                "Modelo no disponible "
                "en el despliegue."
            ),
        }

    resultado = {
        "disponible": True,
        "archivo": str(model_path),
    }

    try:

        modelo = joblib.load(
            model_path
        )

        if isinstance(
            modelo,
            dict
        ):

            resultado["tipo"] = (
                modelo.get("tipo")
            )

            resultado["version"] = (
                modelo.get("version")
            )

    except Exception as exc:

        return {
            "disponible": False,
            "motivo": (
                "No se pudo cargar "
                f"el modelo: {exc}"
            ),
        }

    if summary_path.exists():

        try:

            resultado["resumen"] = json.loads(
                summary_path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:

            pass

    return resultado


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
        "modo_actual": (
            "DIAGNOSTICO - una tesela 256px"
        ),
        "endpoints": {
            "health": "/health",
            "radar": "/radar",
            "radar_png": "/radar.png",
            "radar_status": "/radar/status",
            "radar_debug": "/radar/debug",
            "observacion": "/observacion",
            "estado": "/estado",
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

    debug = diagnostico_rainviewer()

    return {
        "estado": "ok",
        "servicio": "ClimaAR",
        "version": app.version,
        "fuente_radar": "RainViewer",
        "diagnostico_rainviewer": debug,
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

    debug = diagnostico_rainviewer()

    return JSONResponse(
        status_code=200,
        content=debug
    )


# ============================================================
# RADAR DEBUG
# ============================================================

@app.get("/radar/debug")
def radar_debug():

    return JSONResponse(
        status_code=200,
        content=diagnostico_rainviewer()
    )


# ============================================================
# RADAR PNG
# ============================================================

@app.get("/radar.png")
def radar_png():

    if not RADAR_IMAGE.exists():

        return JSONResponse(
            status_code=404,
            content={
                "estado": "sin_imagen",
                "mensaje": (
                    "Todavia no existe "
                    "data/radar/actual.png."
                ),
                "diagnostico":
                    "/radar/debug",
            }
        )

    return FileResponse(
        RADAR_IMAGE,
        media_type="image/png",
        headers={
            "Cache-Control":
                "no-store, no-cache, "
                "must-revalidate",
            "Pragma": "no-cache",
        },
    )


# ============================================================
# RADAR VISUAL
# ============================================================

@app.get("/radar")
def radar():

    debug = diagnostico_rainviewer()

    debug_json = json.dumps(
        debug,
        ensure_ascii=False,
        indent=2
    )

    resultado = debug.get(
        "resultado",
        "DESCONOCIDO"
    )

    ok = (
        resultado == "TESela_OK"
    )

    color = (
        "#19c37d"
        if ok
        else "#ff4d4f"
    )

    estado_texto = (
        "TESela RainViewer OK"
        if ok
        else "DIAGNOSTICO: FALLA"
    )

    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>ClimaAR - Diagnóstico radar</title>

<style>

body {{
    margin: 0;
    background: #101216;
    color: #ffffff;
    font-family: Arial, sans-serif;
}}

header {{
    padding: 18px;
    background: #181b21;
    border-bottom: 1px solid #30343b;
}}

h1 {{
    margin: 0 0 6px 0;
    font-size: 24px;
}}

.sub {{
    color: #aeb4bf;
    font-size: 14px;
}}

.estado {{
    margin: 18px;
    padding: 16px;
    border-radius: 12px;
    background: {color};
    color: #000000;
    font-size: 20px;
    font-weight: bold;
}}

.card {{
    margin: 18px;
    padding: 16px;
    background: #181b21;
    border-radius: 12px;
}}

pre {{
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 12px;
    line-height: 1.5;
    color: #d8dce3;
}}

a {{
    color: #55aaff;
}}

</style>
</head>

<body>

<header>
    <h1>ClimaAR</h1>
    <div class="sub">
        Diagnóstico real de conexión con RainViewer
    </div>
</header>

<div class="estado">
    {estado_texto}
</div>

<div class="card">

    <h2>Resultado</h2>

    <p>
        Resultado:
        <strong>{resultado}</strong>
    </p>

    <p>
        Bahía Blanca:
        {LAT}, {LON}
    </p>

    <p>
        Zoom:
        {RADAR_ZOOM}
    </p>

    <p>
        Tamaño probado:
        256 × 256 px
    </p>

    <p>
        <a href="/radar/debug">
            Abrir diagnóstico JSON
        </a>
    </p>

</div>

<div class="card">

    <h2>Respuesta completa</h2>

    <pre>{debug_json}</pre>

</div>

</body>
</html>
"""

    return HTMLResponse(
        content=html
    )


# ============================================================
# OBSERVACION
# ============================================================

@app.get("/observacion")
def observacion():

    return {
        "fuente":
            "Aviation Weather Center",
        "estacion":
            "SAZB",
        "observacion":
            leer_observacion_sazb(),
    }


# ============================================================
# ESTADO GENERAL
# ============================================================

@app.get("/estado")
def estado():

    return {
        "servicio":
            "ClimaAR",
        "estado":
            "ok",
        "version":
            app.version,
        "radar":
            diagnostico_rainviewer(),
        "observacion_sazb":
            leer_observacion_sazb(),
        "modelo":
            modelo_info(),
        "hora_utc":
            now_utc(),
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

    return {
        "servicio":
            "ClimaAR",
        "estado":
            "experimental",
        "ubicacion": {
            "latitud": LAT,
            "longitud": LON,
            "ciudad":
                "Bahia Blanca",
        },
        "radar":
            diagnostico_rainviewer(),
        "observacion_sazb":
            leer_observacion_sazb(),
    }


# ============================================================
# NOWCAST
# ============================================================

@app.get("/nowcast")
def nowcast():

    return {
        "servicio":
            "ClimaAR",
        "estado":
            "experimental",
        "mensaje":
            (
                "El diagnostico de RainViewer "
                "debe quedar operativo antes "
                "de habilitar el nowcast."
            ),
        "radar":
            diagnostico_rainviewer(),
        "modelo":
            modelo_info(),
    }


# ============================================================
# EJECUCION
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000
    )
