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
    version="3.3.0",
)

LAT = -38.71
LON = -62.26

# Zoom fijo de RainViewer para la composicion del radar.
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

OBS_STATUS = Path(
    "data/sazb/status.json"
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

session = requests.Session()

session.headers.update({
    "User-Agent": "ClimaAR/3.3"
})


# ============================================================
# UTILIDADES
# ============================================================

def now_utc() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def argentina_time(utc_timestamp: int | None) -> str | None:

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
            "error": (
                "status SAZB invalido: "
                f"{exc}"
            ),
        }


# ============================================================
# RAINVIEWER
# ============================================================

def rainviewer_latest() -> dict:

    response = session.get(
        RAINVIEWER_API,
        timeout=20
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

        "frames_disponibles": len(
            frames
        ),

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
# DESCARGA Y COMPOSICION DEL RADAR
# ============================================================

def descargar_radar() -> dict:

    if Image is None:

        raise RuntimeError(
            "Pillow no esta disponible."
        )

    info = rainviewer_latest()

    center_x, center_y = latlon_to_tile(
        LAT,
        LON,
        RADAR_ZOOM
    )

    tile_size = 512

    # Tres por tres teselas.
    min_x = center_x - 1
    max_x = center_x + 1

    min_y = center_y - 1
    max_y = center_y + 1

    image = Image.new(
        "RGBA",
        (
            tile_size * 3,
            tile_size * 3
        ),
        (
            0,
            0,
            0,
            0
        )
    )

    tiles_ok = 0

    errores = []

    for ix, tile_x in enumerate(
        range(min_x, max_x + 1)
    ):

        for iy, tile_y in enumerate(
            range(min_y, max_y + 1)
        ):

            url = (
                f"{info['host']}"
                f"{info['path']}"
                f"/512/"
                f"{RADAR_ZOOM}/"
                f"{tile_x}/"
                f"{tile_y}/"
                f"2/1_1.png"
            )

            try:

                response = session.get(
                    url,
                    timeout=15
                )

                if response.status_code != 200:

                    errores.append(
                        f"HTTP {response.status_code}"
                    )

                    continue

                if len(
                    response.content
                ) < 200:

                    errores.append(
                        "Tesela demasiado pequena"
                    )

                    continue

                tile = Image.open(
                    BytesIO(
                        response.content
                    )
                ).convert(
                    "RGBA"
                )

                image.paste(
                    tile,
                    (
                        ix * tile_size,
                        iy * tile_size
                    ),
                    tile
                )

                tiles_ok += 1

            except Exception as exc:

                errores.append(
                    str(exc)
                )

    if tiles_ok == 0:

        raise RuntimeError(
            "RainViewer respondio, "
            "pero no se pudo descargar "
            "ninguna tesela del radar."
        )

    # Guardamos la imagen transparente.
    image.save(
        RADAR_IMAGE,
        "PNG"
    )

    # Limites geograficos exactos de la imagen.
    west = tile_x_to_lon(
        min_x,
        RADAR_ZOOM
    )

    east = tile_x_to_lon(
        max_x + 1,
        RADAR_ZOOM
    )

    north = tile_y_to_lat(
        min_y,
        RADAR_ZOOM
    )

    south = tile_y_to_lat(
        max_y + 1,
        RADAR_ZOOM
    )

    estado = {

        "version": "3.3",

        "actualizado_utc":
            now_utc(),

        "ubicacion": {

            "ciudad":
                "Bahia Blanca",

            "latitud":
                LAT,

            "longitud":
                LON,
        },

        "estado":
            (
                "radar_disponible"
                if tiles_ok == 9
                else "radar_parcial"
            ),

        "radar": {

            **info,

            "tiles_ok":
                tiles_ok,

            "tiles_total":
                9,

            "imagen":
                "/radar.png",

            "visor":
                "/radar",

            "bounds": {

                "north":
                    north,

                "south":
                    south,

                "west":
                    west,

                "east":
                    east,
            },

            "errores":
                errores[:10],
        },

        "observacion_sazb":
            leer_observacion_sazb(),

        "nota":
            (
                "RainViewer es la fuente "
                "primaria del radar. "
                "SAZB es una observacion "
                "complementaria."
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

            "estado":
                "sin_estado",

            "radar": {

                "fuente":
                    "RainViewer"
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

            "estado":
                "estado_invalido",

            "radar": {

                "fuente":
                    "RainViewer"
            },

            "observacion_sazb":
                leer_observacion_sazb(),
        }


def radar_live() -> dict:

    try:

        return descargar_radar()

    except Exception as exc:

        estado = leer_estado_guardado()

        estado[
            "error_radar_live"
        ] = str(exc)

        return estado


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

            "disponible":
                False,

            "motivo":
                "Modelo no disponible "
                "en el despliegue.",
        }

    resultado = {

        "disponible":
            True,

        "archivo":
            str(model_path),
    }

    try:

        modelo = joblib.load(
            model_path
        )

        if isinstance(
            modelo,
            dict
        ):

            resultado[
                "tipo"
            ] = modelo.get(
                "tipo"
            )

            resultado[
                "version"
            ] = modelo.get(
                "version"
            )

    except Exception as exc:

        return {

            "disponible":
                False,

            "motivo":
                (
                    "No se pudo cargar "
                    "el modelo: "
                    f"{exc}"
                ),
        }

    if summary_path.exists():

        try:

            resultado[
                "resumen"
            ] = json.loads(
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

        "servicio":
            "ClimaAR",

        "estado":
            "ok",

        "version":
            app.version,

        "ubicacion":
            "Bahia Blanca",

        "fuente_radar":
            "RainViewer",

        "endpoints": {

            "health":
                "/health",

            "radar":
                "/radar",

            "radar_png":
                "/radar.png",

            "radar_status":
                "/radar/status",

            "observacion":
                "/observacion",

            "estado":
                "/estado",

            "modelo":
                "/modelo",

            "tormenta":
                "/tormenta",

            "nowcast":
                "/nowcast",
        },
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    radar = radar_live()

    return {

        "estado":
            "ok",

        "servicio":
            "ClimaAR",

        "version":
            app.version,

        "fuente_radar":
            "RainViewer",

        "radar":
            radar,

        "observacion_sazb":
            leer_observacion_sazb(),

        "modelo":
            modelo_info(),

        "hora_utc":
            now_utc(),
    }


# ============================================================
# STATUS RADAR
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
# IMAGEN RADAR
# ============================================================

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
                    "No existe imagen radar.",
            }
        )

    return FileResponse(

        RADAR_IMAGE,

        media_type="image/png",

        headers={
            "Cache-Control":
                "no-store, no-cache, "
                "must-revalidate"
        }
    )


# ============================================================
# VISOR RADAR
# ============================================================

@app.get(
    "/radar",
    response_class=HTMLResponse
)
def radar():

    try:

        info = descargar_radar()

    except Exception as exc:

        estado = leer_estado_guardado()

        radar_info = estado.get(
            "radar",
            {}
        )

        if not radar_info.get(
            "timestamp"
        ):

            return HTMLResponse(

                content=f"""
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport"
content="width=device-width,initial-scale=1">
<title>ClimaAR</title>
</head>

<body style="
font-family:Arial;
padding:25px;
">

<h2>ClimaAR — Radar Bahía Blanca</h2>

<p>
<b>Radar no disponible.</b>
</p>

<p>
{str(exc)}
</p>

</body>
</html>
""",

                status_code=503
            )

        info = {
            **radar_info
        }

    timestamp = info.get(
        "timestamp"
    )

    timestamp_utc = info.get(
        "timestamp_utc",
        "sin datos"
    )

    timestamp_ar = info.get(
        "timestamp_argentina",
        "sin datos"
    )

    bounds = info.get(
        "bounds",
        {}
    )

    north = bounds.get(
        "north"
    )

    south = bounds.get(
        "south"
    )

    west = bounds.get(
        "west"
    )

    east = bounds.get(
        "east"
    )

    tiles_ok = info.get(
        "tiles_ok",
        0
    )

    html = f"""
<!doctype html>

<html lang="es">

<head>

<meta charset="utf-8">

<meta name="viewport"
content="width=device-width,initial-scale=1">

<title>
ClimaAR — Radar Bahía Blanca
</title>

<link
rel="stylesheet"
href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
/>

<style>

html,
body,
#map {{

    height:100%;

    width:100%;

    margin:0;

    padding:0;
}}

#info {{

    position:absolute;

    z-index:1000;

    top:10px;

    left:10px;

    right:10px;

    max-width:390px;

    background:rgba(
        255,
        255,
        255,
        .95
    );

    padding:12px;

    border-radius:10px;

    font-family:Arial,sans-serif;

    font-size:14px;

    box-shadow:
        0 2px 8px
        rgba(0,0,0,.25);
}}

#info b {{

    font-size:16px;
}}

#loading {{

    margin-top:5px;

    font-size:13px;
}}

</style>

</head>

<body>

<div id="info">

<b>
ClimaAR — Radar Bahía Blanca
</b>

<br>

Fuente: RainViewer

<br>

Frame UTC:
{timestamp_utc}

<br>

Hora Argentina:
{timestamp_ar}

<br>

Teselas:
{tiles_ok}/9

<div id="loading">
Radar cargado.
</div>

</div>

<div id="map"></div>

<script
src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js">
</script>

<script>

const lat = {LAT};

const lon = {LON};

const map = L.map(
    'map',
    {{
        zoomControl:true
    }}
).setView(
    [lat, lon],
    8
);

L.tileLayer(
    'https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
    {{

        maxZoom:19,

        attribution:
            '&copy; OpenStreetMap contributors'
    }}
).addTo(map);


const radarBounds = [
    [{south}, {west}],
    [{north}, {east}]
];


const radarLayer =
L.imageOverlay(
    '/radar.png?ts={int(timestamp or 0)}',
    radarBounds,
    {{

        opacity:0.78,

        interactive:false
    }}
).addTo(map);


L.marker(
    [lat, lon]
)
.addTo(map)
.bindPopup(
    'Bahía Blanca'
)
.openPopup();


map.fitBounds(
    radarBounds,
    {{
        padding:[10,10]
    }}
);


radarLayer.on(
    'load',
    function() {{

        document.getElementById(
            'loading'
        ).textContent =
            'Radar visible y cargado.';
    }}
);

</script>

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

    return leer_observacion_sazb()


# ============================================================
# ESTADO GENERAL
# ============================================================

@app.get("/estado")
def estado():

    radar = radar_live()

    return {

        "servicio":
            "ClimaAR",

        "estado":
            radar.get(
                "estado",
                "desconocido"
            ),

        "radar":
            radar.get(
                "radar",
                {}
            ),

        "observacion_sazb":
            leer_observacion_sazb(),

        "modelo_atmosferico":
            modelo_info(),

        "actualizado_utc":
            radar.get(
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

    radar = radar_live()

    datos = radar.get(
        "radar",
        {}
    )

    timestamp = datos.get(
        "timestamp"
    )

    return {

        "estado":
            (
                "radar_disponible"
                if timestamp
                else "sin_datos"
            ),

        "ubicacion":
            "Bahia Blanca",

        "fuente":
            "RainViewer",

        "timestamp":
            timestamp,

        "timestamp_utc":
            datos.get(
                "timestamp_utc"
            ),

        "timestamp_argentina":
            datos.get(
                "timestamp_argentina"
            ),

        "tiles_ok":
            datos.get(
                "tiles_ok"
            ),

        "mensaje":
            (
                "Estado de disponibilidad "
                "y frescura del radar. "
                "No es una alerta oficial."
            ),
    }


# ============================================================
# NOWCAST
# ============================================================

@app.get("/nowcast")
def nowcast():

    radar = radar_live()

    return {

        "estado":
            "experimental",

        "radar":
            radar,

        "modelo":
            modelo_info(),

        "mensaje":
            (
                "La prediccion de evolucion "
                "de tormentas aun es experimental. "
                "No debe interpretarse como "
                "alerta oficial."
            ),
            }
