from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from datetime import datetime, timezone
from pathlib import Path
from io import BytesIO
from zoneinfo import ZoneInfo
import json
import math
import time

import requests
from PIL import Image


VERSION = "5.0.2"

BASE_DIR = Path(__file__).resolve().parent

RADAR_DIR = BASE_DIR / "data" / "radar"
SAZB_DIR = BASE_DIR / "data" / "sazb"

RADAR_DIR.mkdir(parents=True, exist_ok=True)
SAZB_DIR.mkdir(parents=True, exist_ok=True)

ACTUAL_FILE = RADAR_DIR / "actual.png"
PREVIEW_FILE = RADAR_DIR / "preview.png"
STATUS_FILE = RADAR_DIR / "status.json"
NOWCAST_FILE = RADAR_DIR / "radar_nowcast.json"
SAZB_STATUS_FILE = SAZB_DIR / "status.json"

LAT = -38.71
LON = -62.26

RADAR_ZOOM = 7
TILE_SIZE = 256

RAINVIEWER_API = "https://api.rainviewer.com/public/weather-maps.json"

HEADERS = {
    "User-Agent": f"ClimaAR/{VERSION}",
    "Referer": "https://www.rainviewer.com/",
}

app = FastAPI(
    title="ClimaAR",
    description="Radar meteorológico y seguimiento de tormentas para Bahía Blanca",
    version=VERSION,
)


def iso_now():
    return datetime.now(timezone.utc).isoformat()


def argentina_time(dt):
    try:
        return dt.astimezone(
            ZoneInfo("America/Argentina/Buenos_Aires")
        )
    except Exception:
        return dt


def clean(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}

    if isinstance(value, list):
        return [clean(v) for v in value]

    return value


def load_json(path):
    try:
        if not path.exists():
            return None

        return clean(
            json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        )
    except Exception:
        return None


def save_json(path, data):
    path.write_text(
        json.dumps(
            clean(data),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def lon_to_x(lon):
    return int(
        (lon + 180.0)
        / 360.0
        * (2 ** RADAR_ZOOM)
    )


def lat_to_y(lat):
    return int(
        (
            1
            - math.asinh(
                math.tan(
                    math.radians(lat)
                )
            ) / math.pi
        )
        / 2
        * (2 ** RADAR_ZOOM)
    )


def x_to_lon(x):
    return (
        x
        / (2 ** RADAR_ZOOM)
        * 360.0
        - 180.0
    )


def y_to_lat(y):
    return math.degrees(
        math.atan(
            math.sinh(
                math.pi
                * (
                    1
                    - 2 * y
                    / (2 ** RADAR_ZOOM)
                )
            )
        )
    )


def rainviewer_data():

    response = requests.get(
        RAINVIEWER_API,
        headers=HEADERS,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    past = (
        data
        .get("radar", {})
        .get("past", [])
    )

    if not past:
        raise RuntimeError(
            "RainViewer no devolvió frames históricos."
        )

    frame = past[-1]

    if not frame.get("path"):
        raise RuntimeError(
            "RainViewer no devolvió path para el frame."
        )

    return {
        "host": data.get(
            "host",
            "https://tilecache.rainviewer.com"
        ),
        "path": frame["path"],
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


def build_radar():

    started = time.time()

    rv = rainviewer_data()

    cx = lon_to_x(LON)
    cy = lat_to_y(LAT)

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
    ok = 0

    for dy in range(-1, 2):

        for dx in range(-1, 2):

            x = cx + dx
            y = cy + dy

            url = tile_url(
                rv["host"],
                rv["path"],
                x,
                y
            )

            item = {
                "x": x,
                "y": y,
                "url": url,
                "http": None,
                "valid_png": False,
                "size": 0,
                "error": None
            }

            try:

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

                canvas.alpha_composite(
                    image,
                    (
                        (dx + 1) * TILE_SIZE,
                        (dy + 1) * TILE_SIZE
                    )
                )

                item.update({
                    "http": response.status_code,
                    "valid_png": True,
                    "size": len(response.content)
                })

                ok += 1

            except Exception as exc:

                item["error"] = str(exc)

            tiles.append(item)

    canvas.save(
        ACTUAL_FILE,
        "PNG"
    )

    canvas.save(
        PREVIEW_FILE,
        "PNG"
    )

    frame_utc = None
    frame_ar = None

    if rv.get("time") is not None:

        try:

            dt = datetime.fromtimestamp(
                int(rv["time"]),
                tz=timezone.utc
            )

            frame_utc = dt.isoformat()

            frame_ar = (
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
            if ok == 9
            else "parcial",

        "fuente":
            "RainViewer",

        "version":
            VERSION,

        "esquema_color":
            2,

        "esquema_color_nombre":
            "Universal Blue",

        "actualizado_utc":
            iso_now(),

        "frame_utc":
            frame_utc,

        "frame_argentina":
            frame_ar,

        "frames_disponibles":
            rv["frames"],

        "teselas_ok":
            ok,

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
                y_to_lat(
                    cy - 1
                ),

            "south":
                y_to_lat(
                    cy + 2
                ),

            "west":
                x_to_lon(
                    cx - 1
                ),

            "east":
                x_to_lon(
                    cx + 2
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

    save_json(
        STATUS_FILE,
        status
    )

    return status


def cargar_nowcast():

    data = load_json(
        NOWCAST_FILE
    )

    if not isinstance(
        data,
        dict
    ):
        return None

    return data


@app.get(
    "/",
    response_class=HTMLResponse
)
def home():
    return radar_page()


@app.get("/health")
def health():

    return {
        "estado": "ok",
        "servicio": "ClimaAR",
        "version": VERSION,
        "hora_utc": iso_now()
    }


@app.get("/radar/9tiles")
def radar_9tiles():

    try:

        return build_radar()

    except Exception as exc:

        return {
            "estado": "error",
            "fuente": "RainViewer",
            "error": str(exc),
            "hora_utc": iso_now()
        }


@app.get("/radar/status")
def radar_status():

    return (
        load_json(
            STATUS_FILE
        )
        or
        {
            "estado": "sin_datos",
            "fuente": "RainViewer"
        }
    )


@app.get("/radar/debug")
def radar_debug():

    try:

        rv = rainviewer_data()

        x = lon_to_x(LON)
        y = lat_to_y(LAT)

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

        response.raise_for_status()

        image = Image.open(
            BytesIO(
                response.content
            )
        )

        return {

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
                len(
                    response.content
                ),

            "png": {

                "valido":
                    True,

                "formato":
                    image.format,

                "modo":
                    image.mode,

                "ancho":
                    image.width,

                "alto":
                    image.height
            }
        }

    except Exception as exc:

        return {
            "estado": "error",
            "error": str(exc)
        }


@app.get("/radar.png")
def radar_png():

    if not ACTUAL_FILE.exists():

        try:
            build_radar()

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


@app.get("/nowcast")
def nowcast():

    data = cargar_nowcast()

    return {

        "disponible":
            data is not None,

        "estado":
            "ok"
            if data is not None
            else "sin_datos",

        "zona":
            "Bahía Blanca",

        "fuente":
            "RainViewer",

        "nowcast":
            data
    }


@app.get("/observacion")
def observacion():

    return (
        load_json(
            SAZB_STATUS_FILE
        )
        or
        {
            "estado":
                "sin_datos",

            "fuente":
                "Aviation Weather Center",

            "estacion":
                "SAZB"
        }
    )


@app.get("/estado")
def estado():

    nowcast_data = cargar_nowcast()

    return clean({

        "app":
            "ClimaAR",

        "version":
            VERSION,

        "radar":
            load_json(
                STATUS_FILE
            ),

        "observacion_sazb":
            load_json(
                SAZB_STATUS_FILE
            ),

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
    })


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

    data = cargar_nowcast()

    if not data:

        return {
            "estado":
                "sin_datos",

            "zona":
                "Bahía Blanca"
        }

    n = data.get(
        "nowcast",
        data
    )

    return {

        "estado":
            "ok",

        "zona":
            "Bahía Blanca",

        "actividad":
            n.get(
                "actividad",
                False
            ),

        "nucleos":
            n.get(
                "nucleos",
                []
            ),

        "nucleo_principal_id":
            n.get(
                "nucleo_principal_id"
            ),

        "proyecciones":
            n.get(
                "proyecciones",
                {}
            ),

        "confianza":
            n.get(
                "confianza_movimiento",
                0
            )
    }


def radar_page():

    return r"""
<!doctype html>
<html lang="es">

<head>

<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width,initial-scale=1"
>

<title>ClimaAR — Radar Bahía Blanca</title>

<link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
/>

<style>

* {
    box-sizing: border-box;
}

html,
body {

    margin: 0;
    padding: 0;

    width: 100%;
    height: 100%;

    overflow: hidden;

    background: #111;

    color: #fff;

    font-family: Arial, sans-serif;
}

#map {

    position: absolute;

    inset: 0;

    width: 100%;
    height: 100%;

    z-index: 1;
}

.top {

    position: absolute;

    top: 0;
    left: 0;
    right: 0;

    z-index: 1000;

    padding: 10px;

    background:
        rgba(17,17,17,.94);

    border-bottom:
        1px solid #333;
}

.title {

    font-size: 20px;

    font-weight: bold;
}

.sub {

    color: #aaa;

    font-size: 12px;

    margin-top: 3px;
}

.actions {

    display: flex;

    gap: 8px;

    margin-top: 8px;
}

.actions button {

    flex: 1;

    border: 0;

    border-radius: 8px;

    padding: 10px;

    background: #1976d2;

    color: #fff;

    font-size: 15px;

    font-weight: bold;
}

.actions button.secondary {

    background: #444;
}

.panel {

    position: absolute;

    left: 10px;
    right: 10px;

    bottom: 10px;

    z-index: 999;

    max-width: 620px;

    max-height: 58vh;

    overflow-y: auto;

    background:
        rgba(17,17,17,.94);

    border:
        1px solid #444;

    border-radius: 14px;

    padding: 12px;

    box-shadow:
        0 4px 18px
        rgba(0,0,0,.55);
}

.panel.hidden {

    display: none;
}

.panel-header {

    display: flex;

    align-items: center;

    justify-content: space-between;

    gap: 10px;

    margin-bottom: 8px;
}

.panel-header strong {

    font-size: 17px;
}

.close-btn {

    width: 42px;
    height: 42px;

    border: 0;

    border-radius: 50%;

    background: #333;

    color: #fff;

    font-size: 24px;

    line-height: 42px;

    text-align: center;
}

.status {

    font-size: 20px;

    font-weight: bold;

    margin: 5px 0;
}

.ok {
    color: #55e08a;
}

.warn {
    color: #ffd45a;
}

.danger {
    color: #ff6868;
}

.info {

    color: #aaa;

    font-size: 13px;

    margin: 5px 0 10px;
}

.grid {

    display: grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap: 7px;
}

.card {

    background: #222;

    border-radius: 9px;

    padding: 9px;
}

.label {

    color: #999;

    font-size: 11px;

    text-transform: uppercase;
}

.value {

    margin-top: 3px;

    font-size: 17px;

    font-weight: bold;
}

.legend-title {

    margin-top: 12px;

    font-size: 12px;

    font-weight: bold;
}

.legend {

    display: flex;

    height: 18px;

    border-radius: 5px;

    overflow: hidden;

    margin-top: 5px;
}

.legend span {

    flex: 1;
}

.leaflet-control-attribution {

    font-size: 9px !important;
}

@media (max-width: 520px) {

    .panel {

        max-height: 55vh;

        bottom: 8px;

        left: 7px;
        right: 7px;
    }

    .title {

        font-size: 18px;
    }
}

</style>

</head>

<body>

<div id="map"></div>

<div class="top">

    <div class="title">
        ClimaAR — Radar Bahía Blanca
    </div>

    <div
        class="sub"
        id="frame"
    >
        Cargando radar...
    </div>

    <div class="actions">

        <button
            onclick="actualizarRadar()"
        >
            Actualizar radar
        </button>

        <button
            class="secondary"
            onclick="mostrarPanel()"
        >
            Mostrar información
        </button>

    </div>

</div>


<div
    class="panel"
    id="panel"
>

    <div class="panel-header">

        <strong>
            Estado meteorológico
        </strong>

        <button
            class="close-btn"
            onclick="ocultarPanel()"
            aria-label="Ocultar"
        >
            ×
        </button>

    </div>


    <div
        id="estado"
        class="status"
    >
        Cargando...
    </div>

    <div
        id="detalle"
        class="info"
    ></div>


    <div class="grid">

        <div class="card">

            <div class="label">
                Temperatura
            </div>

            <div
                class="value"
                id="temp"
            >
                —
            </div>

        </div>


        <div class="card">

            <div class="label">
                Punto de rocío
            </div>

            <div
                class="value"
                id="dew"
            >
                —
            </div>

        </div>


        <div class="card">

            <div class="label">
                Viento
            </div>

            <div
                class="value"
                id="wind"
            >
                —
            </div>

        </div>


        <div class="card">

            <div class="label">
                Presión
            </div>

            <div
                class="value"
                id="pressure"
            >
                —
            </div>

        </div>


        <div class="card">

            <div class="label">
                Distancia
            </div>

            <div
                class="value"
                id="distance"
            >
                —
            </div>

        </div>


        <div class="card">

            <div class="label">
                Velocidad célula
            </div>

            <div
                class="value"
                id="speed"
            >
                —
            </div>

        </div>


        <div class="card">

            <div class="label">
                Dirección célula
            </div>

            <div
                class="value"
                id="direction"
            >
                —
            </div>

        </div>


        <div class="card">

            <div class="label">
                Tendencia
            </div>

            <div
                class="value"
                id="trend"
            >
                —
            </div>

        </div>


        <div class="card">

            <div class="label">
                ETA
            </div>

            <div
                class="value"
                id="eta"
            >
                —
            </div>

        </div>


        <div class="card">

            <div class="label">
                Confianza tracking
            </div>

            <div
                class="value"
                id="confidence"
            >
                —
            </div>

        </div>

    </div>


    <div class="legend-title">

        REFLECTIVIDAD RADAR · dBZ ·
        RAINVIEWER UNIVERSAL BLUE

    </div>


    <div class="legend">

        <span style="background:#cec087"></span>
        <span style="background:#6e0dc6"></span>
        <span style="background:#c06487"></span>
        <span style="background:#fac431"></span>
        <span style="background:#fe9a58"></span>
        <span style="background:#fd341c"></span>
        <span style="background:#bebebe"></span>

    </div>

</div>


<script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
></script>


<script>

let map;

let radarOverlay = null;


function mostrarPanel() {

    document
        .getElementById("panel")
        .classList
        .remove("hidden");
}


function ocultarPanel() {

    document
        .getElementById("panel")
        .classList
        .add("hidden");
}


function val(
    v,
    suffix = ""
) {

    if (
        v === null ||
        v === undefined ||
        v === ""
    ) {

        return "—";
    }

    return String(v) + suffix;
}


function fmtConfidence(v) {

    if (
        v === null ||
        v === undefined
    ) {

        return "—";
    }

    let n = Number(v);

    if (n <= 1) {
        n *= 100;
    }

    return Math.round(n) + "%";
}


async function cargarEstado() {

    try {

        const response =
            await fetch(
                "/radar/status?t="
                + Date.now()
            );

        const status =
            await response.json();

        const bounds =
            status.bounds;


        if (

            bounds &&

            bounds.north !== undefined &&

            bounds.south !== undefined &&

            bounds.west !== undefined &&

            bounds.east !== undefined

        ) {

            const imageBounds = [

                [
                    bounds.south,
                    bounds.west
                ],

                [
                    bounds.north,
                    bounds.east
                ]

            ];


            if (radarOverlay) {

                map.removeLayer(
                    radarOverlay
                );
            }


            radarOverlay =
                L.imageOverlay(

                    "/radar.png?t="
                    + Date.now(),

                    imageBounds,

                    {
                        opacity: 0.88,

                        interactive: false
                    }

                );


            radarOverlay.addTo(
                map
            );
        }


        const frame =
            status.frame_argentina ||
            status.frame_utc ||
            "sin frame";


        document
            .getElementById("frame")
            .textContent =
                "RainViewer · último frame: "
                + frame
                + " · teselas: "
                + (
                    status.teselas_ok ?? "—"
                )
                + "/"
                + (
                    status.teselas_total ?? "—"
                );

    } catch (e) {

        document
            .getElementById("frame")
            .textContent =
                "No se pudo cargar el estado del radar";
    }
}


async function cargarInformacion() {

    try {

        const [
            nowcastResponse,
            obsResponse
        ] =
            await Promise.all([

                fetch(
                    "/nowcast?t="
                    + Date.now()
                ),

                fetch(
                    "/observacion?t="
                    + Date.now()
                )

            ]);


        const nowcastData =
            await nowcastResponse.json();

        const obs =
            await obsResponse.json();


        const n =
            nowcastData.nowcast || {};


        const actividad =
            n.actividad === true;


        const estado =
            document.getElementById(
                "estado"
            );


        estado.textContent =
            actividad
            ? "Precipitación detectada"
            : "Sin precipitación detectada";


        estado.className =
            "status "
            + (
                actividad
                ? "warn"
                : "ok"
            );


        document
            .getElementById("detalle")
            .textContent =

                "Nowcast V7 · "
                + (
                    n.frames_analizados
                    ?? "—"
                )
                + " frames · "
                + (
                    n.tiles_ok
                    ?? "—"
                )
                + "/9 teselas";


        const ambiente =
            obs.ambiente ||
            obs ||
            {};


        document
            .getElementById("temp")
            .textContent =
                val(
                    ambiente.temperature_2m,
                    " °C"
                );


        document
            .getElementById("dew")
            .textContent =
                val(
                    ambiente.dew_point_2m,
                    " °C"
                );


        document
            .getElementById("wind")
            .textContent =
                val(
                    ambiente.wind_speed_10m,
                    " km/h"
                );


        document
            .getElementById("pressure")
            .textContent =
                val(
                    ambiente.pressure_msl,
                    " hPa"
                );


        document
            .getElementById("distance")
            .textContent =
                val(
                    n.distancia_km,
                    " km"
                );


        document
            .getElementById("speed")
            .textContent =
                val(
                    n.velocidad_kmh,
                    " km/h"
                );


        document
            .getElementById("direction")
            .textContent =
                n.direccion ||
                val(
                    n.direccion_grados,
                    "°"
                );


        document
            .getElementById("trend")
            .textContent =
                n.fortalecimiento ||
                "sin_datos";


        document
            .getElementById("eta")
            .textContent =
                val(
                    n.eta_minutos,
                    " min"
                );


        document
            .getElementById("confidence")
            .textContent =
                fmtConfidence(
                    n.confianza_movimiento
                );


    } catch (e) {

        document
            .getElementById("estado")
            .textContent =
                "Error leyendo datos";

        document
            .getElementById("estado")
            .className =
                "status danger";

        document
            .getElementById("detalle")
            .textContent =
                String(e);
    }
}


async function actualizarRadar() {

    document
        .getElementById("frame")
        .textContent =
            "Actualizando radar...";


    try {

        await fetch(
            "/radar/9tiles?t="
            + Date.now(),
            {
                cache:
                    "no-store"
            }
        );

    } catch (e) {

        console.error(e);
    }


    await cargarEstado();

    await cargarInformacion();
}


function iniciarMapa() {

    map =
        L.map(
            "map",
            {
                zoomControl:
                    true,

                attributionControl:
                    true
            }
        )
        .setView(
            [
                -38.71,
                -62.26
            ],
            7
        );


    L.tileLayer(

        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",

        {
            maxZoom: 18,

            attribution:
                "&copy; OpenStreetMap contributors"
        }

    ).addTo(map);


    L.marker(
        [
            -38.71,
            -62.26
        ]
    )
    .addTo(map)
    .bindTooltip(
        "Bahía Blanca"
    );


    cargarEstado();

    cargarInformacion();


    setInterval(

        () => {

            cargarEstado();

            cargarInformacion();

        },

        60000

    );
}


iniciarMapa();

</script>

</body>

</html>
"""


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
