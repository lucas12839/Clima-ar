from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from io import BytesIO

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


app = FastAPI(
    title="ClimaAR",
    description="Seguimiento operativo de precipitacion para Bahia Blanca.",
    version="3.2.0",
)

LAT = -38.71
LON = -62.26
ZOOM = 7

RAINVIEWER_API = "https://api.rainviewer.com/public/weather-maps.json"
RAINVIEWER_DEFAULT_HOST = "https://tilecache.rainviewer.com"

DATA_DIR = Path("data/radar")
RADAR_IMAGE = DATA_DIR / "actual.png"
RADAR_STATUS = DATA_DIR / "status.json"
OBS_STATUS = Path("data/sazb/status.json")

DATA_DIR.mkdir(parents=True, exist_ok=True)

_session = requests.Session()
_session.headers.update({"User-Agent": "ClimaAR/3.2"})


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def latlon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
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
            "error": f"status SAZB invalido: {exc}",
        }


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

    frames = radar.get(
        "past"
    ) or []

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

    image = Image.new(
        "RGBA",
        (
            1536,
            1536
        ),
        (
            245,
            245,
            245,
            255
        )
    )

    tiles_ok = 0

    for dx in (-1, 0, 1):

        for dy in (-1, 0, 1):

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
            "No se pudo descargar ninguna tesela de RainViewer."
        )

    image.save(
        RADAR_IMAGE,
        "PNG"
    )

    estado = {

        "version": "3.2",

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

            "imagen": "/radar.png",

            "visor": "/radar",
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


def radar_status_live() -> dict:

    try:

        return rainviewer_latest()

    except Exception as exc:

        saved = leer_estado_guardado()

        saved[
            "error_radar_live"
        ] = str(exc)

        return saved


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

            "radar_png": "/radar.png",

            "radar_status":
                "/radar/status",

            "observacion":
                "/observacion",

            "modelo": "/modelo",

            "tormenta": "/tormenta",

            "nowcast": "/nowcast",
        },
    }


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


@app.get("/radar.png")
def radar_png():

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
    )


@app.get(
    "/radar",
    response_class=HTMLResponse
)
def radar():

    try:

        info = descargar_radar()

    except Exception:

        info = leer_estado_guardado().get(
            "radar",
            {}
        )

        if not info.get(
            "timestamp"
        ):

            info = radar_status_live()

    timestamp = info.get(
        "timestamp"
    )

    timestamp_text = info.get(
        "timestamp_utc",
        "sin datos"
    )

    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport"
content="width=device-width,initial-scale=1">
<title>ClimaAR — Radar Bahía Blanca</title>

<link rel="stylesheet"
href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">

<style>

html,
body,
#map {{
    height: 100%;
    margin: 0;
}}

#info {{
    position: absolute;
    z-index: 1000;
    top: 10px;
    left: 10px;
    background: rgba(255,255,255,.94);
    padding: 10px 12px;
    border-radius: 8px;
    font: 14px Arial,sans-serif;
    box-shadow:
        0 1px 5px rgba(0,0,0,.25);
}}

</style>
</head>

<body>

<div id="info">

<b>ClimaAR — Radar Bahía Blanca</b>

<br>

Fuente: RainViewer

<br>

Frame: {timestamp_text}

</div>

<div id="map"></div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js">
</script>

<script>

const lat = {LAT};

const lon = {LON};

const radarPath =
{json.dumps(info.get("path", ""))};

const radarHost =
{json.dumps(
    info.get(
        "host",
        RAINVIEWER_DEFAULT_HOST
    )
)};

const radarTime =
{int(timestamp or 0)};

const map =
L.map('map').setView(
    [lat, lon],
    8
);

L.tileLayer(
    'https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
    {{
        maxZoom: 19,
        attribution:
            '&copy; OpenStreetMap contributors'
    }}
).addTo(map);

if (
    radarPath
    && radarTime
) {{

    const radarUrl =
        radarHost
        + radarPath
        + '/256/{{z}}/{{x}}/{{y}}/2/1_1.png';

    L.tileLayer(
        radarUrl,
        {{
            opacity: 0.72,
            maxZoom: 7,
            minZoom: 0,
            tileSize: 256,
            attribution: 'RainViewer'
        }}
    ).addTo(map);
}}

L.marker(
    [lat, lon]
).addTo(map)

.bindPopup(
    'Bahía Blanca'
)

.openPopup();

</script>

</body>
</html>"""

    return HTMLResponse(
        content=html
    )


@app.get("/observacion")
def observacion():

    return leer_observacion_sazb()


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


@app.get("/modelo")
def modelo():

    return modelo_info()


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


@app.get("/nowcast")
def nowcast():

    info = modelo_info()

    radar = radar_status_live()

    return {

        "estado":
            "experimental",

        "radar":
            radar,

        "modelo":
            info,

        "mensaje": (
            "La prediccion de evolucion "
            "de tormentas aun es experimental. "
            "No debe interpretarse como "
            "alerta oficial."
        ),
    }
