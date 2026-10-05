from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from datetime import datetime, timezone
from pathlib import Path
from io import BytesIO
from zoneinfo import ZoneInfo
import math
import json
import time
import requests
from PIL import Image


# ============================================================
# CLIMAAR
# ============================================================

VERSION = "4.3.2"

app = FastAPI(
    title="ClimaAR",
    description="Radar meteorológico y seguimiento de tormentas para Bahía Blanca",
    version=VERSION
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RADAR_DIR = BASE_DIR / "data" / "radar"

RADAR_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ACTUAL_FILE = RADAR_DIR / "actual.png"
PREVIEW_FILE = RADAR_DIR / "preview.png"
STATUS_FILE = RADAR_DIR / "status.json"
NOWCAST_FILE = RADAR_DIR / "radar_nowcast.json"

LAT = -38.71
LON = -62.26

RADAR_ZOOM = 7
TILE_SIZE = 256
GRID_RADIUS = 1

RAINVIEWER_API = (
    "https://api.rainviewer.com/public/weather-maps.json"
)

HEADERS = {
    "User-Agent": f"ClimaAR/{VERSION}",
    "Referer": "https://www.rainviewer.com/",
    "Accept": "image/png,image/*;q=0.8,*/*;q=0.5",
    "Cache-Control": "no-cache"
}


# ============================================================
# UTILIDADES
# ============================================================

def iso_now():
    return datetime.now(timezone.utc).isoformat()


def argentina_time(dt):

    try:

        return dt.astimezone(
            ZoneInfo(
                "America/Argentina/Buenos_Aires"
            )
        )

    except Exception:

        return dt


def clean_json_value(value):

    if isinstance(value, float):

        if math.isfinite(value):
            return value

        return None

    if isinstance(value, dict):

        return {
            str(key): clean_json_value(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):

        return [
            clean_json_value(item)
            for item in value
        ]

    return value


def load_json(path):

    try:

        if not path.exists():
            return None

        raw = path.read_text(
            encoding="utf-8"
        )

        if not raw.strip():
            return None

        return clean_json_value(
            json.loads(raw)
        )

    except Exception:

        return None


def save_status(data):

    STATUS_FILE.write_text(
        json.dumps(
            clean_json_value(data),
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        ),
        encoding="utf-8"
    )


# ============================================================
# COORDENADAS / TILES
# ============================================================

def lon_to_tile_x(lon, zoom):

    return int(
        (lon + 180.0)
        / 360.0
        * (2 ** zoom)
    )


def lat_to_tile_y(lat, zoom):

    lat_rad = math.radians(lat)

    n = 2 ** zoom

    return int(
        (
            1
            - math.asinh(
                math.tan(lat_rad)
            ) / math.pi
        )
        / 2
        * n
    )


def tile_to_lon(x, zoom):

    return (
        x / (2 ** zoom)
        * 360.0
        - 180.0
    )


def tile_to_lat(y, zoom):

    return math.degrees(
        math.atan(
            math.sinh(
                math.pi
                * (
                    1
                    - 2 * y / (2 ** zoom)
                )
            )
        )
    )


# ============================================================
# RAINVIEWER
# ============================================================

def rainviewer_data():

    response = requests.get(
        RAINVIEWER_API,
        headers=HEADERS,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    past = data.get(
        "radar",
        {}
    ).get(
        "past",
        []
    )

    if not past:

        raise RuntimeError(
            "RainViewer no devolvió frames históricos."
        )

    frame = past[-1]

    path = frame.get(
        "path"
    )

    if not path:

        raise RuntimeError(
            "RainViewer no devolvió path para el frame."
        )

    return {
        "host": data.get(
            "host",
            "https://tilecache.rainviewer.com"
        ),
        "path": path,
        "time": frame.get("time"),
        "frames": len(past)
    }


def tile_url(
    host,
    path,
    x,
    y
):

    return (
        f"{host}"
        f"{path}"
        f"/{TILE_SIZE}"
        f"/{RADAR_ZOOM}"
        f"/{x}"
        f"/{y}"
        f"/2"
        f"/1_1.png"
    )


def download_tile(url):

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=20
    )

    response.raise_for_status()

    image = Image.open(
        BytesIO(
            response.content
        )
    ).convert("RGBA")

    return image, response


# ============================================================
# CONSTRUIR RADAR
# ============================================================

def build_9tiles():

    started = time.time()

    rv = rainviewer_data()

    center_x = lon_to_tile_x(
        LON,
        RADAR_ZOOM
    )

    center_y = lat_to_tile_y(
        LAT,
        RADAR_ZOOM
    )

    canvas = Image.new(
        "RGBA",
        (
            TILE_SIZE * 3,
            TILE_SIZE * 3
        ),
        (
            0,
            0,
            0,
            0
        )
    )

    tiles = []

    ok_count = 0

    for dy in range(-1, 2):

        for dx in range(-1, 2):

            x = center_x + dx
            y = center_y + dy

            url = tile_url(
                rv["host"],
                rv["path"],
                x,
                y
            )

            info = {
                "x": x,
                "y": y,
                "url": url,
                "http": None,
                "valid_png": False,
                "size": 0,
                "error": None
            }

            try:

                image, response = (
                    download_tile(url)
                )

                info["http"] = (
                    response.status_code
                )

                info["valid_png"] = True

                info["size"] = len(
                    response.content
                )

                canvas.alpha_composite(
                    image,
                    (
                        (dx + 1) * TILE_SIZE,
                        (dy + 1) * TILE_SIZE
                    )
                )

                ok_count += 1

            except Exception as exc:

                info["error"] = str(exc)

            tiles.append(info)

    canvas.save(
        ACTUAL_FILE,
        "PNG"
    )

    canvas.save(
        PREVIEW_FILE,
        "PNG"
    )

    frame_utc = None
    frame_argentina = None

    if rv["time"]:

        try:

            dt = datetime.fromtimestamp(
                int(rv["time"]),
                tz=timezone.utc
            )

            frame_utc = dt.isoformat()

            frame_argentina = (
                argentina_time(dt)
                .strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

        except Exception:

            frame_utc = str(
                rv["time"]
            )

    status = {

        "estado":
            "ok"
            if ok_count == 9
            else "parcial",

        "fuente": "RainViewer",

        "version": VERSION,

        "actualizado_utc":
            iso_now(),

        "frame_utc":
            frame_utc,

        "frame_argentina":
            frame_argentina,

        "frames_disponibles":
            rv["frames"],

        "teselas_ok":
            ok_count,

        "teselas_total":
            9,

        "zoom":
            RADAR_ZOOM,

        "centro": {
            "lat": LAT,
            "lon": LON
        },

        "bounds": {

            "north":
                tile_to_lat(
                    center_y - 1,
                    RADAR_ZOOM
                ),

            "south":
                tile_to_lat(
                    center_y + 2,
                    RADAR_ZOOM
                ),

            "west":
                tile_to_lon(
                    center_x - 1,
                    RADAR_ZOOM
                ),

            "east":
                tile_to_lon(
                    center_x + 2,
                    RADAR_ZOOM
                )
        },

        "duracion_segundos":
            round(
                time.time() - started,
                2
            ),

        "tiles":
            tiles
    }

    save_status(status)

    return clean_json_value(
        status
    )


# ============================================================
# INICIO
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def inicio():

    return """
    <h1>ClimaAR</h1>

    <p>
        <a href="/radar">
            Abrir radar
        </a>
    </p>

    <p>
        <a href="/health">
            Estado
        </a>
    </p>
    """


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "estado": "ok",
        "servicio": "ClimaAR",
        "version": VERSION,
        "hora_utc": iso_now()
    }


# ============================================================
# RADAR 9 TESELAS
# ============================================================

@app.get("/radar/9tiles")
def radar_9tiles():

    try:

        return build_9tiles()

    except Exception as exc:

        return {
            "estado": "error",
            "fuente": "RainViewer",
            "error": str(exc),
            "hora_utc": iso_now()
        }


# ============================================================
# RADAR STATUS
# ============================================================

@app.get("/radar/status")
def radar_status():

    status = load_json(
        STATUS_FILE
    )

    if status is None:

        return {
            "estado": "sin_datos",
            "fuente": "RainViewer",
            "mensaje":
                "Todavía no hay un radar generado."
        }

    return status


# ============================================================
# DEBUG
# ============================================================

@app.get("/radar/debug")
def radar_debug():

    try:

        rv = rainviewer_data()

        x = lon_to_tile_x(
            LON,
            RADAR_ZOOM
        )

        y = lat_to_tile_y(
            LAT,
            RADAR_ZOOM
        )

        url = tile_url(
            rv["host"],
            rv["path"],
            x,
            y
        )

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20
        )

        result = {

            "estado": "ok",

            "api_http": 200,

            "tile_http":
                response.status_code,

            "tile_url":
                url,

            "x": x,

            "y": y,

            "zoom":
                RADAR_ZOOM,

            "bytes":
                len(response.content)
        }

        try:

            image = Image.open(
                BytesIO(
                    response.content
                )
            )

            result["png"] = {

                "valido": True,

                "formato":
                    image.format,

                "modo":
                    image.mode,

                "ancho":
                    image.width,

                "alto":
                    image.height
            }

        except Exception as exc:

            result["png"] = {

                "valido": False,

                "error":
                    str(exc)
            }

        return clean_json_value(
            result
        )

    except Exception as exc:

        return {
            "estado": "error",
            "error": str(exc)
        }


# ============================================================
# IMAGEN RADAR
# ============================================================

@app.get("/radar.png")
def radar_png():

    if not ACTUAL_FILE.exists():

        try:

            build_9tiles()

        except Exception:

            pass

    if not ACTUAL_FILE.exists():

        return JSONResponse(
            {
                "estado":
                    "sin_imagen"
            },
            status_code=503
        )

    return FileResponse(
        ACTUAL_FILE,
        media_type="image/png"
    )


@app.get("/radar/preview.png")
def radar_preview():

    return radar_png()


# ============================================================
# ESTADO GENERAL
# ============================================================

@app.get("/estado")
def estado():

    try:

        radar = load_json(
            STATUS_FILE
        )

        nowcast_data = load_json(
            NOWCAST_FILE
        )

        respuesta = {

            "app":
                "ClimaAR",

            "version":
                VERSION,

            "radar":
                radar,

            "nowcast_radar": {

                "disponible":
                    nowcast_data is not None,

                "estado":
                    "ok"
                    if nowcast_data is not None
                    else "sin_datos",

                "datos":
                    nowcast_data
            },

            "nowcast_radar_operativo":
                nowcast_data is not None,

            "nowcast_radar_ia":
                False,

            "hora_utc":
                iso_now()
        }

        return JSONResponse(
            content=clean_json_value(
                respuesta
            )
        )

    except Exception as exc:

        return JSONResponse(
            status_code=200,
            content={

                "app":
                    "ClimaAR",

                "version":
                    VERSION,

                "estado":
                    "parcial",

                "error":
                    str(exc),

                "radar":
                    None,

                "nowcast_radar": {

                    "disponible":
                        False,

                    "estado":
                        "error"
                },

                "nowcast_radar_operativo":
                    False,

                "nowcast_radar_ia":
                    False,

                "hora_utc":
                    iso_now()
            }
        )


# ============================================================
# NOWCAST
# ============================================================

@app.get("/nowcast")
def nowcast():

    data = load_json(
        NOWCAST_FILE
    )

    if data is None:

        return {

            "disponible":
                False,

            "estado":
                "sin_datos",

            "zona":
                "Bahía Blanca",

            "mensaje":
                "Todavía no existe radar_nowcast.json."
        }

    return {

        "disponible":
            True,

        "estado":
            "ok",

        "zona":
            "Bahía Blanca",

        "fuente":
            "RainViewer",

        "nowcast":
            data
    }


# ============================================================
# OTROS ENDPOINTS
# ============================================================

@app.get("/observacion")
def observacion():

    return {

        "estado":
            "ok",

        "fuente":
            "SAZB",

        "mensaje":
            "Endpoint de observación meteorológica.",

        "hora_utc":
            iso_now()
    }


@app.get("/modelo")
def modelo():

    return {

        "estado":
            "experimental",

        "mensaje":
            "Módulo de análisis meteorológico de ClimaAR."
    }


@app.get("/tormenta")
def tormenta():

    return {

        "estado":
            "ok",

        "zona":
            "Bahía Blanca",

        "mensaje":
            "Análisis de tormenta disponible cuando existan datos suficientes."
    }


# ============================================================
# PÁGINA RADAR
# ============================================================

@app.get(
    "/radar",
    response_class=HTMLResponse
)
def radar_page():

    return r'''
<!doctype html>

<html lang="es">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width,
             initial-scale=1,
             maximum-scale=1,
             user-scalable=no"
>

<title>
ClimaAR — Radar Bahía Blanca
</title>

<link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
>

<style>

* {
    box-sizing: border-box;
}

html,
body {

    margin: 0;

    background: #111;

    color: white;

    font-family: Arial, sans-serif;
}

.top {

    padding: 12px;

    background: #171717;

    border-bottom: 1px solid #333;

    position: relative;

    z-index: 1000;
}

h1 {

    margin: 0 0 5px;

    font-size: 20px;
}

#estado {

    font-size: 13px;

    color: #ccc;

    margin-bottom: 9px;
}

button {

    width: 100%;

    padding: 12px;

    border: 0;

    border-radius: 8px;

    background: #1976d2;

    color: white;

    font-size: 16px;

    font-weight: bold;
}

#resultado {

    margin-top: 7px;

    font-size: 13px;

    min-height: 18px;
}

#map {

    width: 100%;

    height:
        calc(100vh - 145px);

    min-height: 520px;
}

.leaflet-control-attribution {

    font-size: 9px;
}

.info-panel {

    position: absolute;

    left: 10px;

    right: 10px;

    bottom: 15px;

    z-index: 999;

    background:
        rgba(17,17,17,.94);

    border: 1px solid #444;

    border-radius: 12px;

    padding: 12px;

    box-shadow:
        0 4px 16px
        rgba(0,0,0,.45);

    max-width: 520px;

    margin: auto;
}

.estado-principal {

    font-size: 18px;

    font-weight: bold;

    margin-bottom: 7px;
}

.estado-secundario {

    font-size: 12px;

    color: #bbb;

    margin-bottom: 10px;
}

.grid {

    display: grid;

    grid-template-columns:
        repeat(2,1fr);

    gap: 7px;
}

.dato {

    background: #222;

    border-radius: 8px;

    padding: 8px;
}

.dato-titulo {

    font-size: 10px;

    color: #999;

    margin-bottom: 3px;
}

.dato-valor {

    font-size: 14px;

    font-weight: bold;
}

.separador {

    margin: 10px 0 8px;

    border-top:
        1px solid #333;
}

.fuentes {

    font-size: 10px;

    color: #888;
}

.sin-actividad {

    color: #58d68d;
}

.actividad {

    color: #ffb84d;
}

.alerta {

    color: #ff6b6b;
}

.marcador-bahia {

    background: white;

    border:
        3px solid #1976d2;

    width: 18px;

    height: 18px;

    border-radius: 50%;

    box-shadow:
        0 0 0 4px
        rgba(25,118,210,.35);
}

</style>

</head>

<body>

<div class="top">

<h1>
ClimaAR — Radar Bahía Blanca
</h1>

<div id="estado">
RainViewer · cargando radar...
</div>

<button onclick="actualizarRadar()">
Actualizar radar
</button>

<div id="resultado"></div>

</div>

<div id="map"></div>


<div class="info-panel">

<div
    id="estadoPrincipal"
    class="estado-principal"
>
Analizando condiciones...
</div>

<div
    id="estadoSecundario"
    class="estado-secundario"
>
Cargando nowcast...
</div>


<div class="grid">

<div class="dato">

<div class="dato-titulo">
DISTANCIA
</div>

<div
    id="distancia"
    class="dato-valor"
>
—
</div>

</div>


<div class="dato">

<div class="dato-titulo">
VELOCIDAD
</div>

<div
    id="velocidad"
    class="dato-valor"
>
—
</div>

</div>


<div class="dato">

<div class="dato-titulo">
DIRECCIÓN
</div>

<div
    id="direccion"
    class="dato-valor"
>
—
</div>

</div>


<div class="dato">

<div class="dato-titulo">
TENDENCIA
</div>

<div
    id="fortalecimiento"
    class="dato-valor"
>
—
</div>

</div>


<div class="dato">

<div class="dato-titulo">
ETA
</div>

<div
    id="eta"
    class="dato-valor"
>
—
</div>

</div>


<div class="dato">

<div class="dato-titulo">
CONFIANZA
</div>

<div
    id="confianza"
    class="dato-valor"
>
—
</div>

</div>


<div class="dato">

<div class="dato-titulo">
PROYECCIÓN 30 MIN
</div>

<div
    id="proy30"
    class="dato-valor"
>
—
</div>

</div>


<div class="dato">

<div class="dato-titulo">
PROYECCIÓN 60 MIN
</div>

<div
    id="proy60"
    class="dato-valor"
>
—
</div>

</div>

</div>


<div class="separador"></div>

<div
    id="fuentes"
    class="fuentes"
>
Radar: RainViewer · SAZB · Open-Meteo · ClimaAR
</div>

</div>


<script
src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js">
</script>


<script>

const centro = [
    -38.71,
    -62.26
];


const map = L.map(
    "map"
).setView(
    centro,
    9
);


L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    {
        maxZoom: 19,

        attribution:
            "&copy; OpenStreetMap contributors"
    }
).addTo(map);


const iconBahia =
    L.divIcon({

        className: "",

        html:
            '<div class="marcador-bahia"></div>',

        iconSize: [
            18,
            18
        ],

        iconAnchor: [
            9,
            9
        ]
    });


L.marker(
    centro,
    {
        icon: iconBahia
    }
)
.addTo(map)
.bindTooltip(
    "Bahía Blanca"
);


let radar = null;


function txt(
    value,
    suffix = ""
) {

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {

        return "—";
    }

    return String(value)
        + suffix;
}


function pct(value) {

    const number =
        Number(value);

    if (
        !Number.isFinite(number)
    ) {

        return "—";
    }

    return Math.round(
        number * 100
    ) + "%";
}


function dir(
    direction,
    degrees
) {

    if (direction) {

        return direction;
    }

    const number =
        Number(degrees);

    if (
        !Number.isFinite(number)
    ) {

        return "—";
    }

    const names = [
        "N",
        "NE",
        "E",
        "SE",
        "S",
        "SO",
        "O",
        "NO"
    ];

    const index =
        (
            (
                Math.round(
                    number / 45
                ) % 8
            ) + 8
        ) % 8;

    return names[index];
}


/*
 * CORRECCIÓN:
 *
 * El JSON real es:
 *
 * nowcast_radar
 *     datos
 *         nowcast
 *
 * Por eso buscamos exactamente
 * ese nivel.
 */

function getNowcast(data) {

    const container =
        data?.nowcast_radar?.datos;


    if (
        container?.nowcast
    ) {

        return container.nowcast;
    }


    /*
     * Compatibilidad con
     * estructuras anteriores.
     */

    if (
        container?.actividad !== undefined ||
        container?.frames_analizados !== undefined
    ) {

        return container;
    }


    if (
        data?.nowcast_radar?.nowcast
    ) {

        return data.nowcast_radar.nowcast;
    }


    return null;
}


function mostrarNowcast(
    data
) {

    const n =
        getNowcast(data);


    const principal =
        document.getElementById(
            "estadoPrincipal"
        );


    const secundario =
        document.getElementById(
            "estadoSecundario"
        );


    if (!n) {

        principal.textContent =
            "Nowcast no disponible";

        principal.className =
            "estado-principal alerta";

        secundario.textContent =
            "No hay datos suficientes.";

        return;
    }


    if (
        n.actividad === true
    ) {

        principal.textContent =
            "🟠 Precipitación detectada";

        principal.className =
            "estado-principal actividad";

    } else {

        principal.textContent =
            "🟢 Sin precipitación detectada";

        principal.className =
            "estado-principal sin-actividad";
    }


    secundario.textContent =
        "Radar operativo · "
        + txt(
            n.frames_analizados
        )
        + " frames analizados";


    document.getElementById(
        "distancia"
    ).textContent =
        n.distancia_km != null
        ? txt(
            Number(
                n.distancia_km
            ).toFixed(1),
            " km"
        )
        : "—";


    document.getElementById(
        "velocidad"
    ).textContent =
        n.velocidad_kmh != null
        ? txt(
            Number(
                n.velocidad_kmh
            ).toFixed(1),
            " km/h"
        )
        : "—";


    document.getElementById(
        "direccion"
    ).textContent =
        dir(
            n.direccion,
            n.direccion_grados
        );


    document.getElementById(
        "fortalecimiento"
    ).textContent =
        n.fortalecimiento ||
        "—";


    document.getElementById(
        "eta"
    ).textContent =
        n.eta_minutos != null
        ? txt(
            Math.round(
                Number(
                    n.eta_minutos
                )
            ),
            " min"
        )
        : "—";


    document.getElementById(
        "confianza"
    ).textContent =
        pct(
            n.confianza_movimiento
        );


    document.getElementById(
        "proy30"
    ).textContent =
        n.proyeccion_30_min != null
        ? String(
            n.proyeccion_30_min
        )
        : "—";


    document.getElementById(
        "proy60"
    ).textContent =
        n.proyeccion_60_min != null
        ? String(
            n.proyeccion_60_min
        )
        : "—";
}


function colocarRadar(
    data
) {

    if (
        !data ||
        !data.bounds
    ) {

        return false;
    }


    const bounds = [

        [
            data.bounds.south,
            data.bounds.west
        ],

        [
            data.bounds.north,
            data.bounds.east
        ]
    ];


    if (!radar) {

        radar =
            L.imageOverlay(
                "/radar.png?ts="
                + Date.now(),

                bounds,

                {
                    opacity: 0.65,

                    interactive: false
                }
            )
            .addTo(map);

    } else {

        radar.setBounds(
            bounds
        );

        radar.setUrl(
            "/radar.png?ts="
            + Date.now()
        );
    }


    return true;
}


/*
 * CARGA AUTOMÁTICA
 *
 * Si Render no tiene status.json
 * al arrancar, se genera el radar
 * automáticamente.
 */

async function cargarEstado() {

    try {

        let response =
            await fetch(
                "/estado?ts="
                + Date.now()
            );


        if (!response.ok) {

            throw new Error(
                "HTTP "
                + response.status
            );
        }


        let data =
            await response.json();


        /*
         * Si no hay radar local,
         * lo construimos sin que
         * el usuario tenga que
         * tocar el botón.
         */

        if (!data.radar) {

            document.getElementById(
                "resultado"
            ).textContent =
                "Generando radar...";


            const radarResponse =
                await fetch(
                    "/radar/9tiles?ts="
                    + Date.now()
                );


            const radarData =
                await radarResponse.json();


            if (
                radarData.estado === "ok"
            ) {

                data.radar =
                    radarData;


                document.getElementById(
                    "resultado"
                ).textContent =
                    "Radar generado automáticamente · "
                    + radarData.teselas_ok
                    + "/"
                    + radarData.teselas_total;
            }
        }


        mostrarNowcast(
            data
        );


        if (
            data.radar &&
            data.radar.bounds
        ) {

            colocarRadar(
                data.radar
            );
        }


        if (
            data.radar &&
            data.radar.frame_argentina
        ) {

            document.getElementById(
                "estado"
            ).textContent =
                "RainViewer · último frame: "
                + data.radar.frame_argentina
                + " · teselas: "
                + txt(
                    data.radar.teselas_ok
                )
                + "/"
                + txt(
                    data.radar.teselas_total
                );
        }


    } catch (error) {

        document.getElementById(
            "resultado"
        ).textContent =
            "No se pudo actualizar el análisis.";

        console.log(
            error
        );
    }
}


/*
 * ACTUALIZACIÓN MANUAL
 */

async function actualizarRadar() {

    const output =
        document.getElementById(
            "resultado"
        );


    output.textContent =
        "Actualizando radar...";


    try {

        const response =
            await fetch(
                "/radar/9tiles?ts="
                + Date.now()
            );


        const data =
            await response.json();


        if (
            data.estado !== "ok"
        ) {

            output.textContent =
                "Error: "
                + (
                    data.error ||
                    "No se pudo actualizar."
                );

            return;
        }


        colocarRadar(
            data
        );


        map.setView(
            centro,
            9
        );


        output.textContent =
            "Radar actualizado · "
            + data.teselas_ok
            + "/"
            + data.teselas_total;


        await cargarEstado();


    } catch (error) {

        output.textContent =
            "Error de conexión: "
            + error;
    }
}


setTimeout(
    function() {

        map.invalidateSize();

    },
    300
);


/*
 * ARRANQUE AUTOMÁTICO
 */

cargarEstado();


/*
 * ACTUALIZACIÓN CADA 60 SEGUNDOS
 */

setInterval(
    cargarEstado,
    60000
);

</script>

</body>

</html>
''' 
