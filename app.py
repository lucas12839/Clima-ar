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


# ============================================================
# CLIMAAR
# Radar Bahía Blanca + RainViewer + Nowcast V7
# ============================================================

VERSION = "5.0.0"

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

RAINVIEWER_API = (
    "https://api.rainviewer.com/public/weather-maps.json"
)

HEADERS = {
    "User-Agent": f"ClimaAR/{VERSION}",
    "Referer": "https://www.rainviewer.com/"
}


app = FastAPI(
    title="ClimaAR",
    description=(
        "Radar meteorológico y seguimiento "
        "de tormentas para Bahía Blanca"
    ),
    version=VERSION
)


# ============================================================
# UTILIDADES
# ============================================================

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

        return {
            str(k): clean(v)
            for k, v in value.items()
        }

    if isinstance(value, list):

        return [
            clean(v)
            for v in value
        ]

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
            allow_nan=False
        ),
        encoding="utf-8"
    )


# ============================================================
# COORDENADAS
# ============================================================

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
            -
            math.asinh(
                math.tan(
                    math.radians(lat)
                )
            )
            / math.pi
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
                *
                (
                    1
                    -
                    2 * y
                    / (2 ** RADAR_ZOOM)
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

        "host":
            data.get(
                "host",
                "https://tilecache.rainviewer.com"
            ),

        "path":
            frame["path"],

        "time":
            frame.get("time"),

        "frames":
            len(past)
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

                    "http":
                        response.status_code,

                    "valid_png":
                        True,

                    "size":
                        len(
                            response.content
                        )
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


# ============================================================
# API
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return (
        "<h1>ClimaAR</h1>"
        "<p><a href='/radar'>Abrir radar</a></p>"
        "<p><a href='/health'>Estado</a></p>"
    )


@app.get("/health")
def health():

    return {

        "estado":
            "ok",

        "servicio":
            "ClimaAR",

        "version":
            VERSION,

        "hora_utc":
            iso_now()
    }


@app.get("/radar/9tiles")
def radar_9tiles():

    try:

        return build_radar()

    except Exception as exc:

        return {

            "estado":
                "error",

            "fuente":
                "RainViewer",

            "error":
                str(exc),

            "hora_utc":
                iso_now()
        }


@app.get("/radar/status")
def radar_status():

    return (
        load_json(
            STATUS_FILE
        )
        or
        {
            "estado":
                "sin_datos",

            "fuente":
                "RainViewer"
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

            "estado":
                "ok",

            "api_http":
                200,

            "tile_http":
                response.status_code,

            "tile_url":
                url,

            "x":
                x,

            "y":
                y,

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

            "estado":
                "error",

            "error":
                str(exc)
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


# ============================================================
# NOWCAST V7
# ============================================================

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


# ============================================================
# SAZB
# ============================================================

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


# ============================================================
# ESTADO GENERAL
# ============================================================

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


# ============================================================
# MODELO
# ============================================================

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
        {}
    )

    nucleos = n.get(
        "nucleos",
        []
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
            nucleos,

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


# ============================================================
# INTERFAZ
# ============================================================

HTML = r'''
<!doctype html>

<html lang="es">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width,initial-scale=1"
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
    color: #fff;
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
    color: #fff;
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
    height: calc(100vh - 150px);
    min-height: 520px;
}

.panel {
    position: absolute;
    left: 10px;
    right: 10px;
    bottom: 15px;
    z-index: 999;
    background: rgba(17,17,17,.95);
    border: 1px solid #444;
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,.45);
    max-width: 600px;
    margin: auto;
    max-height: 65vh;
    overflow-y: auto;
}

.main {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 5px;
}

.sub {
    font-size: 12px;
    color: #bbb;
    margin-bottom: 10px;
}

.grid {
    display: grid;
    grid-template-columns: repeat(2,1fr);
    gap: 7px;
}

.dato {
    background: #222;
    border-radius: 8px;
    padding: 8px;
}

.dt {
    font-size: 10px;
    color: #999;
    margin-bottom: 3px;
}

.dv {
    font-size: 14px;
    font-weight: bold;
}

.sep {
    margin: 10px 0 8px;
    border-top: 1px solid #333;
}

.section-title {
    font-size: 12px;
    font-weight: bold;
    color: #ddd;
    margin: 9px 0 6px;
}

.core {
    background: #202020;
    border: 1px solid #383838;
    border-radius: 9px;
    padding: 9px;
    margin-bottom: 7px;
}

.core-main {
    font-size: 14px;
    font-weight: bold;
    margin-bottom: 5px;
}

.core-sub {
    color: #aaa;
    font-size: 11px;
    margin-bottom: 6px;
}

.proj {
    display: grid;
    grid-template-columns: repeat(3,1fr);
    gap: 5px;
}

.proj-item {
    background: #292929;
    border-radius: 6px;
    padding: 6px;
    text-align: center;
}

.proj-time {
    font-size: 9px;
    color: #999;
}

.proj-value {
    font-size: 11px;
    font-weight: bold;
    margin-top: 2px;
}

.fuentes,
.obs {
    font-size: 10px;
    color: #888;
    margin-top: 7px;
}

.legend {
    font-size: 11px;
    font-weight: bold;
    margin: 8px 0 5px;
}

.bar {
    display: flex;
    height: 14px;
    border-radius: 4px;
    overflow: hidden;
}

.bar span {
    flex: 1;
}

.labels {
    display: flex;
    justify-content: space-between;
    font-size: 9px;
    color: #aaa;
    margin-top: 3px;
}

.ok {
    color: #58d68d;
}

.warn {
    color: #ffb84d;
}

.bad {
    color: #ff6b6b;
}

.info {
    color: #65b9ff;
}

@media (max-width: 520px) {

    .panel {
        bottom: 8px;
        left: 7px;
        right: 7px;
        max-height: 68vh;
    }

    .proj {
        grid-template-columns: repeat(2,1fr);
    }
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

<div class="panel">

<div
    id="principal"
    class="main"
>
Analizando condiciones...
</div>

<div
    id="sub"
    class="sub"
>
Cargando datos...
</div>


<div class="grid">

<div class="dato">
<div class="dt">TEMPERATURA</div>
<div id="temp" class="dv">—</div>
</div>

<div class="dato">
<div class="dt">PUNTO DE ROCÍO</div>
<div id="rocio" class="dv">—</div>
</div>

<div class="dato">
<div class="dt">VIENTO</div>
<div id="viento" class="dv">—</div>
</div>

<div class="dato">
<div class="dt">DIRECCIÓN</div>
<div id="dir" class="dv">—</div>
</div>

<div class="dato">
<div class="dt">PRESIÓN</div>
<div id="presion" class="dv">—</div>
</div>

<div class="dato">
<div class="dt">VISIBILIDAD</div>
<div id="vis" class="dv">—</div>
</div>

</div>


<div class="sep"></div>


<div class="grid">

<div class="dato">
<div class="dt">DISTANCIA</div>
<div id="dist" class="dv">—</div>
</div>

<div class="dato">
<div class="dt">VELOCIDAD</div>
<div id="vel" class="dv">—</div>
</div>

<div class="dato">
<div class="dt">DIRECCIÓN CÉLULA</div>
<div id="celldir" class="dv">—</div>
</div>

<div class="dato">
<div class="dt">TENDENCIA</div>
<div id="trend" class="dv">—</div>
</div>

<div class="dato">
<div class="dt">ETA</div>
<div id="eta" class="dv">—</div>
</div>

<div class="dato">
<div class="dt">CONFIANZA</div>
<div id="conf" class="dv">—</div>
</div>

</div>


<div class="sep"></div>

<div
    id="coresTitle"
    class="section-title"
>
NÚCLEOS DE PRECIPITACIÓN
</div>

<div id="cores">
Sin núcleos detectados.
</div>


<div class="sep"></div>

<div class="legend">
REFLECTIVIDAD RADAR · dBZ · RAINVIEWER UNIVERSAL BLUE
</div>

<div class="bar">

<span style="background:#cec087"></span>
<span style="background:#6e0dc6"></span>
<span style="background:#c06487"></span>
<span style="background:#fac431"></span>
<span style="background:#fe9a58"></span>
<span style="background:#fd341c"></span>
<span style="background:#bebebe"></span>

</div>

<div class="labels">

<span>10</span>
<span>20</span>
<span>30</span>
<span>40</span>
<span>50</span>
<span>60</span>
<span>65+</span>

</div>


<div
    id="obs"
    class="obs"
>
SAZB · cargando observación...
</div>

<div class="fuentes">
Radar: RainViewer · Observación: Aviation Weather Center SAZB · Seguimiento: ClimaAR V7
</div>

</div>


<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<script>

const center = [
    -38.71,
    -62.26
];


const map = L.map(
    'map'
).setView(
    center,
    9
);


L.tileLayer(
    'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    {
        maxZoom: 19,
        attribution:
            '&copy; OpenStreetMap contributors'
    }
).addTo(map);


L.marker(
    center
).addTo(map)
.bindTooltip(
    'Bahía Blanca'
);


let radar = null;


const el = id =>
    document.getElementById(id);


const val = (
    v,
    suffix = ''
) => {

    if (
        v === null ||
        v === undefined ||
        v === ''
    ) {
        return '—';
    }

    return String(v) + suffix;
};


function direction(value) {

    if (
        value === null ||
        value === undefined ||
        value === ''
    ) {
        return '—';
    }

    const n = Number(value);

    if (!Number.isFinite(n)) {
        return String(value);
    }

    return [
        'N',
        'NE',
        'E',
        'SE',
        'S',
        'SO',
        'O',
        'NO'
    ][
        Math.round(n / 45) % 8
    ];
}


function trendText(value) {

    if (!value) {
        return '—';
    }

    const s = String(value)
        .toLowerCase();

    if (
        s.includes('fortal')
    ) {
        return 'Fortaleciendo';
    }

    if (
        s.includes('debil')
    ) {
        return 'Debilitando';
    }

    if (
        s.includes('estable')
    ) {
        return 'Estable';
    }

    return String(value);
}


function confidence(value) {

    if (
        value === null ||
        value === undefined
    ) {
        return '—';
    }

    const n = Number(value);

    if (!Number.isFinite(n)) {
        return '—';
    }

    return Math.round(
        n * 100
    ) + '%';
}


function formatDistance(value) {

    if (
        value === null ||
        value === undefined
    ) {
        return '—';
    }

    const n = Number(value);

    if (!Number.isFinite(n)) {
        return '—';
    }

    return n.toFixed(1) + ' km';
}


function formatSpeed(value) {

    if (
        value === null ||
        value === undefined
    ) {
        return '—';
    }

    const n = Number(value);

    if (!Number.isFinite(n)) {
        return '—';
    }

    return n.toFixed(1) + ' km/h';
}


function formatEta(value) {

    if (
        value === null ||
        value === undefined
    ) {
        return '—';
    }

    const n = Number(value);

    if (!Number.isFinite(n)) {
        return '—';
    }

    return Math.round(n) + ' min';
}


function projectionValue(
    projection
) {

    if (!projection) {
        return '—';
    }

    if (
        projection.distancia_km !== null &&
        projection.distancia_km !== undefined
    ) {

        return (
            Number(
                projection.distancia_km
            ).toFixed(1)
            +
            ' km'
        );
    }

    if (
        projection.latitud !== null &&
        projection.latitud !== undefined
    ) {

        return Number(
            projection.latitud
        ).toFixed(2);
    }

    return '—';
}


function renderProjections(
    projections
) {

    if (
        !projections ||
        typeof projections !== 'object'
    ) {

        return '';
    }

    const order = [
        '15',
        '30',
        '45',
        '60',
        '90'
    ];

    let html = '';

    for (
        const minutes of order
    ) {

        const p =
            projections[minutes] ??
            projections[
                minutes + '_min'
            ];

        if (!p) {
            continue;
        }

        const inside =
            p.en_zona_radar === true
            ? 'zona radar'
            : 'fuera';

        html += `
            <div class="proj-item">
                <div class="proj-time">
                    +${minutes} min
                </div>
                <div class="proj-value">
                    ${projectionValue(p)}
                </div>
                <div class="proj-time">
                    ${inside}
                </div>
            </div>
        `;
    }

    return html;
}


function renderCores(
    n
) {

    const container =
        el('cores');

    const cores =
        Array.isArray(n.nucleos)
        ? n.nucleos
        : [];

    if (
        cores.length === 0
    ) {

        container.innerHTML =
            '<div class="sub">Sin núcleos de precipitación detectados en los frames analizados.</div>';

        return;
    }

    let html = '';

    cores.forEach(
        (core, index) => {

            const id =
                core.track_id ??
                core.id ??
                (index + 1);

            const principal =
                Number(id) ===
                Number(
                    n.nucleo_principal_id
                );

            const hacia =
                core.movimiento_hacia_bahia === true;

            const speed =
                core.velocidad_kmh;

            const dir =
                core.direccion_grados ??
                core.direccion;

            const dist =
                core.distancia_km;

            const eta =
                core.eta_minutos;

            const conf =
                core.confianza ??
                core.confianza_movimiento;

            const projections =
                core.proyecciones ??
                core.projection ??
                {};

            html += `

                <div class="core">

                    <div class="core-main">
                        Núcleo ${id}
                        ${principal ? ' · PRINCIPAL' : ''}
                    </div>

                    <div class="core-sub">
                        ${hacia
                            ? '🟠 Movimiento hacia Bahía Blanca'
                            : 'Movimiento no dirigido hacia Bahía Blanca'}
                    </div>

                    <div class="grid">

                        <div class="dato">
                            <div class="dt">
                                DISTANCIA
                            </div>
                            <div class="dv">
                                ${formatDistance(dist)}
                            </div>
                        </div>

                        <div class="dato">
                            <div class="dt">
                                VELOCIDAD
                            </div>
                            <div class="dv">
                                ${formatSpeed(speed)}
                            </div>
                        </div>

                        <div class="dato">
                            <div class="dt">
                                DIRECCIÓN
                            </div>
                            <div class="dv">
                                ${direction(dir)}
                            </div>
                        </div>

                        <div class="dato">
                            <div class="dt">
                                ETA
                            </div>
                            <div class="dv">
                                ${formatEta(eta)}
                            </div>
                        </div>

                    </div>

                    <div class="section-title">
                        PROYECCIÓN
                    </div>

                    <div class="proj">
                        ${renderProjections(
                            projections
                        )}
                    </div>

                    <div class="core-sub">
                        Confianza:
                        ${confidence(conf)}
                    </div>

                </div>
            `;
        }
    );

    container.innerHTML = html;
}


function nowcast(
    data
) {

    const n =
        data?.nowcast_radar?.datos;

    if (!n) {

        el('principal')
            .textContent =
            'Nowcast no disponible';

        el('principal')
            .className =
            'main bad';

        el('cores')
            .textContent =
            'No hay datos del motor V7.';

        return;
    }


    const active =
        n.actividad === true;


    el('principal')
        .textContent =
        active
        ? '🟠 Precipitación detectada'
        : '🟢 Sin precipitación detectada';


    el('principal')
        .className =
        'main ' +
        (
            active
            ? 'warn'
            : 'ok'
        );


    el('sub')
        .textContent =
        'Nowcast V7 · ' +
        val(
            n.frames_analizados
        ) +
        ' frames · ' +
        val(
            n.tiles_ok
        ) +
        '/9 teselas';


    el('dist')
        .textContent =
        formatDistance(
            n.distancia_km
        );


    el('vel')
        .textContent =
        formatSpeed(
            n.velocidad_kmh
        );


    el('celldir')
        .textContent =
        direction(
            n.direccion_grados ??
            n.direccion
        );


    el('trend')
        .textContent =
        trendText(
            n.fortalecimiento
        );


    el('eta')
        .textContent =
        formatEta(
            n.eta_minutos
        );


    el('conf')
        .textContent =
        confidence(
            n.confianza_movimiento
        );


    renderCores(n);
}


function sazb(
    s
) {

    if (
        !s ||
        s.observacion_valida !== true
    ) {

        el('obs')
            .textContent =
            'SAZB · sin observación válida';

        return;
    }


    const kmh =
        s.viento_kt != null
        ? Number(
            s.viento_kt
        ) * 1.852
        : null;


    el('temp')
        .textContent =
        val(
            s.temperatura_c,
            ' °C'
        );


    el('rocio')
        .textContent =
        val(
            s.punto_rocio_c,
            ' °C'
        );


    el('viento')
        .textContent =
        kmh != null
        ? kmh.toFixed(1)
          + ' km/h'
        : '—';


    el('dir')
        .textContent =
        direction(
            s.direccion_viento
        );


    el('presion')
        .textContent =
        s.presion_hpa != null
        ? Number(
            s.presion_hpa
        ).toFixed(1)
          + ' hPa'
        : '—';


    el('vis')
        .textContent =
        val(
            s.visibilidad_millas,
            ' mi'
        );


    el('obs')
        .textContent =
        'SAZB · observación hace ' +
        val(
            s.edad_minutos,
            ' min'
        );
}


function overlay(
    radarData
) {

    if (
        !radarData?.bounds
    ) {
        return;
    }


    const bounds = [

        [
            radarData.bounds.south,
            radarData.bounds.west
        ],

        [
            radarData.bounds.north,
            radarData.bounds.east
        ]
    ];


    if (!radar) {

        radar =
            L.imageOverlay(
                '/radar.png?ts=' +
                Date.now(),
                bounds,
                {
                    opacity: .65,
                    interactive: false
                }
            ).addTo(map);

    } else {

        radar.setBounds(
            bounds
        );

        radar.setUrl(
            '/radar.png?ts=' +
            Date.now()
        );
    }
}


async function load() {

    try {

        const response =
            await fetch(
                '/estado?ts=' +
                Date.now()
            );


        const data =
            await response.json();


        if (!data.radar) {

            const radarResponse =
                await fetch(
                    '/radar/9tiles?ts=' +
                    Date.now()
                );

            data.radar =
                await radarResponse.json();
        }


        nowcast(data);


        sazb(
            data.observacion_sazb
        );


        overlay(
            data.radar
        );


        if (
            data.radar?.frame_argentina
        ) {

            el('estado')
                .textContent =
                'RainViewer · último frame: ' +
                data.radar.frame_argentina +
                ' · teselas: ' +
                val(
                    data.radar.teselas_ok
                ) +
                '/' +
                val(
                    data.radar.teselas_total
                );
        }


    } catch (error) {

        el('resultado')
            .textContent =
            'No se pudo actualizar el análisis.';

        console.log(error);
    }
}


async function actualizarRadar() {

    el('resultado')
        .textContent =
        'Actualizando radar...';


    try {

        const response =
            await fetch(
                '/radar/9tiles?ts=' +
                Date.now()
            );


        const data =
            await response.json();


        if (
            data.estado !== 'ok'
        ) {

            el('resultado')
                .textContent =
                'Error: ' +
                (
                    data.error ||
                    'No se pudo actualizar.'
                );

            return;
        }


        overlay(data);


        el('resultado')
            .textContent =
            'Radar actualizado · ' +
            data.teselas_ok +
            '/' +
            data.teselas_total;


        await load();


    } catch (error) {

        el('resultado')
            .textContent =
            'Error de conexión: ' +
            error;
    }
}


setTimeout(
    () => map.invalidateSize(),
    300
);


load();


setInterval(
    load,
    60000
);

</script>

</body>

</html>
'''


# ============================================================
# PÁGINA RADAR
# ============================================================

@app.get(
    "/radar",
    response_class=HTMLResponse
)
def radar_page():

    return HTML
