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


VERSION = "5.1.0"

BASE = Path(__file__).resolve().parent

RADAR = BASE / "data" / "radar"
SAZB = BASE / "data" / "sazb"

RADAR.mkdir(parents=True, exist_ok=True)
SAZB.mkdir(parents=True, exist_ok=True)

ACTUAL = RADAR / "actual.png"
PREVIEW = RADAR / "preview.png"
STATUS = RADAR / "status.json"
NOWCAST = RADAR / "radar_nowcast.json"
SAZB_STATUS = SAZB / "status.json"

LAT = -38.71
LON = -62.26

ZOOM = 7
TILE = 256

RAINVIEWER_API = (
    "https://api.rainviewer.com/public/weather-maps.json"
)

HEAD = {
    "User-Agent": f"ClimaAR/{VERSION}",
    "Referer": "https://www.rainviewer.com/",
}


app = FastAPI(
    title="ClimaAR",
    version=VERSION,
    description=(
        "Radar meteorológico y seguimiento de tormentas "
        "para Bahía Blanca"
    ),
)


def now():
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
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def lon_to_x(lon):
    return int(
        (lon + 180)
        / 360
        * (2 ** ZOOM)
    )


def lat_to_y(lat):
    return int(
        (
            1
            - math.asinh(
                math.tan(
                    math.radians(lat)
                )
            )
            / math.pi
        )
        / 2
        * (2 ** ZOOM)
    )


def x_to_lon(x):
    return (
        x
        / (2 ** ZOOM)
        * 360
        - 180
    )


def y_to_lat(y):
    return math.degrees(
        math.atan(
            math.sinh(
                math.pi
                * (
                    1
                    - 2 * y
                    / (2 ** ZOOM)
                )
            )
        )
    )


def rainviewer_data():

    response = requests.get(
        RAINVIEWER_API,
        headers=HEAD,
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
        "host": (
            data.get("host")
            or "https://tilecache.rainviewer.com"
        ),
        "path": frame["path"],
        "time": frame.get("time"),
        "frames": len(past),
    }


def tile_url(host, path, x, y):

    return (
        f"{host}"
        f"{path}"
        f"/{TILE}"
        f"/{ZOOM}"
        f"/{x}"
        f"/{y}"
        f"/2"
        f"/1_1.png"
    )


def analyze_radar_image(path):

    if not path.exists():
        return {
            "precipitacion": False,
            "cobertura_pct": 0.0,
            "intensidad": "sin_datos",
            "pixeles_radar": 0,
        }

    image = Image.open(path).convert("RGBA")

    pixels = image.load()

    width, height = image.size

    radar_pixels = 0
    moderate_pixels = 0
    strong_pixels = 0

    for y in range(height):

        for x in range(width):

            r, g, b, a = pixels[x, y]

            if (
                a >= 20
                and (r + g + b) > 25
            ):

                radar_pixels += 1

                brightness = (
                    r + g + b
                ) / 3

                if brightness >= 150:
                    strong_pixels += 1

                elif brightness >= 80:
                    moderate_pixels += 1

    total = width * height

    coverage = (
        radar_pixels
        / total
        * 100
        if total
        else 0
    )

    if radar_pixels == 0:

        intensity = "sin_precipitacion"

    elif (
        strong_pixels
        / max(radar_pixels, 1)
        > 0.08
    ):

        intensity = "fuerte"

    elif (
        moderate_pixels
        / max(radar_pixels, 1)
        > 0.12
    ):

        intensity = "moderada"

    else:

        intensity = "debil"

    return {
        "precipitacion":
            radar_pixels > 0,

        "cobertura_pct":
            round(coverage, 3),

        "intensidad":
            intensity,

        "pixeles_radar":
            radar_pixels,

        "pixeles_moderados":
            moderate_pixels,

        "pixeles_fuertes":
            strong_pixels,
    }


def build_radar():

    started = time.time()

    rv = rainviewer_data()

    center_x = lon_to_x(LON)
    center_y = lat_to_y(LAT)

    canvas = Image.new(
        "RGBA",
        (
            TILE * 3,
            TILE * 3
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

            x = center_x + dx
            y = center_y + dy

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
                    headers=HEAD,
                    timeout=20
                )

                response.raise_for_status()

                image = Image.open(
                    BytesIO(
                        response.content
                    )
                ).convert("RGBA")

                if image.size != (
                    TILE,
                    TILE
                ):

                    image = image.resize(
                        (TILE, TILE)
                    )

                canvas.alpha_composite(
                    image,
                    (
                        (dx + 1) * TILE,
                        (dy + 1) * TILE
                    )
                )

                item.update({
                    "http":
                        response.status_code,

                    "valid_png":
                        True,

                    "size":
                        len(response.content)
                })

                ok += 1

            except Exception as exc:

                item["error"] = str(exc)

            tiles.append(item)

    canvas.save(
        ACTUAL,
        "PNG"
    )

    canvas.save(
        PREVIEW,
        "PNG"
    )

    analysis = analyze_radar_image(
        ACTUAL
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
            now(),

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
            ZOOM,

        "centro": {
            "lat": LAT,
            "lon": LON
        },

        "bounds": {

            "north":
                y_to_lat(
                    center_y - 1
                ),

            "south":
                y_to_lat(
                    center_y + 2
                ),

            "west":
                x_to_lon(
                    center_x - 1
                ),

            "east":
                x_to_lon(
                    center_x + 2
                )
        },

        "duracion_segundos":
            round(
                time.time() - started,
                2
            ),

        "analisis":
            analysis,

        "tiles":
            tiles
    }

    save_json(
        STATUS,
        status
    )

    return status


def cargar_nowcast():

    data = load_json(
        NOWCAST
    )

    if isinstance(
        data,
        dict
    ):
        return data

    return None


def radar_analysis():

    status = load_json(
        STATUS
    ) or {}

    analysis = status.get(
        "analisis"
    )

    if isinstance(
        analysis,
        dict
    ):
        return analysis

    return analyze_radar_image(
        ACTUAL
    )


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
        "hora_utc": now()
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
            "hora_utc": now()
        }


@app.get("/radar/status")
def radar_status():

    return (
        load_json(STATUS)
        or
        {
            "estado":
                "sin_datos",

            "fuente":
                "RainViewer"
        }
    )


@app.get("/radar/analysis")
def radar_analysis_endpoint():

    return {
        "estado": "ok",
        "fuente": "RainViewer",
        "analisis":
            radar_analysis(),
        "hora_utc":
            now()
    }


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
            headers=HEAD,
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
            "zoom": ZOOM,

            "bytes":
                len(response.content),

            "png": {

                "valido": True,

                "formato":
                    image.format,

                "modo":
                    image.mode,

                "ancho":
                    image.width,

                "alto":
                    image.height
            },

            "analisis":
                radar_analysis()
        }

    except Exception as exc:

        return {
            "estado": "error",
            "error": str(exc)
        }


@app.get("/radar.png")
def radar_png():

    if not ACTUAL.exists():

        try:
            build_radar()

        except Exception:
            pass

    if not ACTUAL.exists():

        return JSONResponse(
            {
                "estado":
                    "sin_imagen"
            },
            status_code=503
        )

    return FileResponse(
        ACTUAL,
        media_type="image/png",
        headers={
            "Cache-Control":
                "no-store"
        }
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
            SAZB_STATUS
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

    data = cargar_nowcast()

    return clean({

        "app":
            "ClimaAR",

        "version":
            VERSION,

        "radar":
            load_json(STATUS),

        "observacion_sazb":
            load_json(SAZB_STATUS),

        "nowcast_radar": {

            "disponible":
                data is not None,

            "estado":
                "ok"
                if data is not None
                else "sin_datos",

            "datos":
                data
        },

        "nowcast_radar_operativo":
            data is not None,

        "nowcast_radar_ia":
            False,

        "hora_utc":
            now()
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
                "Bahía Blanca",

            "actividad":
                False,

            "nucleos":
                [],

            "proyecciones":
                {},

            "confianza":
                0
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

    return '''<!doctype html>
<html lang="es">

<head>

<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width,initial-scale=1,viewport-fit=cover"
>

<title>ClimaAR — Radar Bahía Blanca</title>

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

    width: 100%;
    height: 100%;

    overflow: hidden;

    background: #08111b;

    color: #fff;

    font-family: Arial, sans-serif;
}

#map {

    position: fixed;

    inset: 0;

    z-index: 1;
}

.top {

    position: fixed;

    z-index: 1000;

    top: 0;
    left: 0;
    right: 0;

    min-height: 58px;

    padding: 8px 10px;

    display: flex;

    align-items: center;

    gap: 10px;

    background:
        rgba(5,12,20,.92);

    backdrop-filter:
        blur(8px);

    box-shadow:
        0 2px 12px
        rgba(0,0,0,.35);
}

.brand {

    font-weight: 800;

    font-size: 15px;
}

.sub {

    font-size: 10px;

    color: #aebdcc;

    margin-top: 3px;
}

.actions {

    margin-left: auto;

    display: flex;

    gap: 6px;
}

button {

    border: 0;

    border-radius: 9px;

    padding: 9px 11px;

    font-weight: 700;

    cursor: pointer;
}

#refresh {

    background: #1677ff;

    color: #fff;
}

#toggle {

    background: #293847;

    color: #fff;
}

.sheet {

    position: fixed;

    z-index: 1100;

    left: 9px;
    right: 9px;

    bottom: 9px;

    max-height: 39vh;

    overflow: auto;

    background:
        rgba(8,13,19,.95);

    border:
        1px solid
        rgba(255,255,255,.14);

    border-radius: 16px;

    box-shadow:
        0 10px 35px
        rgba(0,0,0,.5);

    backdrop-filter:
        blur(12px);

    transition:
        transform .22s ease,
        opacity .22s ease;
}

.sheet.closed {

    transform:
        translateY(
            calc(100% + 20px)
        );

    opacity: .1;

    pointer-events: none;
}

.head {

    position: sticky;

    top: 0;

    z-index: 2;

    display: flex;

    justify-content: space-between;

    align-items: center;

    padding: 9px 12px;

    background:
        rgba(8,13,19,.99);

    border-bottom:
        1px solid
        rgba(255,255,255,.08);
}

.title {

    font-weight: 800;

    font-size: 14px;
}

.close {

    width: 34px;
    height: 34px;

    padding: 0;

    border-radius: 50%;

    background: #3a4857;

    color: #fff;

    font-size: 21px;
}

.grid {

    display: grid;

    grid-template-columns:
        repeat(4,1fr);

    gap: 7px;

    padding: 9px 11px;
}

.card {

    background:
        rgba(255,255,255,.055);

    border-radius: 10px;

    padding: 8px;

    min-height: 50px;
}

.label {

    font-size: 9px;

    color: #91a0ae;

    text-transform: uppercase;
}

.value {

    font-size: 13px;

    font-weight: 800;

    margin-top: 4px;
}

.status {

    padding:
        0 11px 10px;

    font-size: 12px;
}

.good {
    color: #72e3a0;
}

.warn {
    color: #ffd166;
}

.bad {
    color: #ff7777;
}

#open {

    display: none;

    position: fixed;

    z-index: 1050;

    right: 10px;
    bottom: 10px;

    background: #111d29;

    color: #fff;

    border:
        1px solid
        #44515f;
}

@media(max-width:700px) {

    .brand {
        font-size: 14px;
    }

    .sub {
        display: none;
    }

    .actions button {

        padding:
            8px 9px;

        font-size: 11px;
    }

    .sheet {

        max-height: 43vh;
    }

    .grid {

        grid-template-columns:
            repeat(2,1fr);
    }
}

</style>

</head>

<body>

<div id="map"></div>

<header class="top">

    <div>

        <div class="brand">
            ClimaAR — Radar Bahía Blanca
        </div>

        <div
            class="sub"
            id="frame"
        >
            Cargando radar…
        </div>

    </div>

    <div class="actions">

        <button id="refresh">
            Actualizar radar
        </button>

        <button id="toggle">
            Ocultar datos
        </button>

    </div>

</header>

<section
    class="sheet"
    id="sheet"
>

    <div class="head">

        <div
            class="title"
            id="title"
        >
            Estado meteorológico
        </div>

        <button
            class="close"
            id="close"
        >
            ×
        </button>

    </div>

    <div
        class="status"
        id="main"
    >
        Cargando…
    </div>

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
                id="press"
            >
                —
            </div>
        </div>

        <div class="card">
            <div class="label">
                Cobertura radar
            </div>
            <div
                class="value"
                id="cov"
            >
                —
            </div>
        </div>

        <div class="card">
            <div class="label">
                Intensidad
            </div>
            <div
                class="value"
                id="int"
            >
                —
            </div>
        </div>

        <div class="card">
            <div class="label">
                Teselas
            </div>
            <div
                class="value"
                id="tiles"
            >
                —
            </div>
        </div>

        <div class="card">
            <div class="label">
                Actualizado
            </div>
            <div
                class="value"
                id="upd"
            >
                —
            </div>
        </div>

    </div>

    <div class="status">

        <b>Nowcast:</b>
        <span id="nc">
            —
        </span>

        <br>

        <b>Movimiento:</b>
        <span id="motion">
            —
        </span>

    </div>

</section>

<button id="open">
    Mostrar datos
</button>

<script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
></script>

<script>

const map =
    L.map(
        'map',
        {
            zoomControl: true,
            preferCanvas: true
        }
    ).setView(
        [-38.71,-62.26],
        8
    );

L.tileLayer(
    'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    {
        maxZoom: 18,
        attribution:
            '© OpenStreetMap'
    }
).addTo(map);

let layer = null;

const $ =
    id =>
        document.getElementById(id);

const text =
    value =>
        value == null
            ? '—'
            : String(value);


function overlay(status) {

    if (layer) {

        map.removeLayer(
            layer
        );
    }

    if (!status.bounds) {
        return;
    }

    layer =
        L.imageOverlay(
            '/radar.png?ts='
                + Date.now(),

            [
                [
                    status.bounds.south,
                    status.bounds.west
                ],

                [
                    status.bounds.north,
                    status.bounds.east
                ]
            ],

            {
                opacity: .84,
                interactive: false
            }
        );

    layer.addTo(map);
}


function renderRadar(status) {

    overlay(status);

    const analysis =
        status.analisis || {};

    const rain =
        !!analysis.precipitacion;

    $('frame').textContent =
        'RainViewer · último frame: '
        + text(
            status.frame_argentina
            || status.frame_utc
        )
        + ' · teselas: '
        + text(status.teselas_ok)
        + '/'
        + text(status.teselas_total);

    $('tiles').textContent =
        text(status.teselas_ok)
        + '/'
        + text(status.teselas_total);

    $('cov').textContent =
        text(
            analysis.cobertura_pct
        )
        + '%';

    const intensities = {

        sin_precipitacion:
            'Sin precipitación',

        debil:
            'Débil',

        moderada:
            'Moderada',

        fuerte:
            'Fuerte'
    };

    $('int').textContent =
        intensities[
            analysis.intensidad
        ] || '—';

    $('upd').textContent =
        status.actualizado_utc
            ? new Date(
                status.actualizado_utc
              ).toLocaleTimeString(
                'es-AR'
              )
            : '—';

    $('title').textContent =
        rain
            ? 'Precipitación detectada'
            : 'Estado meteorológico';

    $('main').innerHTML =
        rain
            ? '<span class="warn">'
              + '● Precipitación detectada '
              + 'en el mosaico actual.'
              + '</span>'
            : '<span class="good">'
              + '● Sin precipitación detectada '
              + 'en el mosaico actual.'
              + '</span>';
}


function renderObservation(data) {

    const d =
        data &&
        (
            data.observacion
            || data
        ) || {};

    const temperature =
        d.temperatura_c
        ?? d.temperature_c
        ?? d.temp_c
        ?? d.temperatura;

    const dew =
        d.punto_rocio_c
        ?? d.dewpoint_c
        ?? d.dew_point_c
        ?? d.dewpoint;

    const wind =
        d.viento_kmh
        ?? d.wind_kmh
        ?? d.wind_speed_kmh
        ?? d.viento;

    const pressure =
        d.presion_hpa
        ?? d.pressure_hpa
        ?? d.presion;

    $('temp').textContent =
        temperature != null
            ? Number(
                temperature
              ).toFixed(1)
              + ' °C'
            : '—';

    $('dew').textContent =
        dew != null
            ? Number(
                dew
              ).toFixed(1)
              + ' °C'
            : '—';

    $('wind').textContent =
        wind != null
            ? Number(
                wind
              ).toFixed(1)
              + ' km/h'
            : '—';

    $('press').textContent =
        pressure != null
            ? Number(
                pressure
              ).toFixed(1)
              + ' hPa'
            : '—';
}


function renderNowcast(data) {

    const value =
        data &&
        data.nowcast;

    if (!value) {

        $('nc').textContent =
            'sin datos';

        $('motion').textContent =
            'sin datos';

        return;
    }

    const n =
        value.nowcast
        || value;

    $('nc').textContent =
        n.actividad === true
            ? 'actividad detectada'
            : n.actividad === false
                ? 'sin actividad'
                : 'datos disponibles';

    const confidence =
        n.confianza_movimiento;

    $('motion').textContent =
        confidence != null
            ? 'confianza '
              + Math.round(
                    Number(
                        confidence
                    )
                )
              + '%'
            : 'sin datos';
}


async function loadRadar(
    full
) {

    $('refresh').disabled =
        true;

    $('main').textContent =
        'Actualizando radar…';

    try {

        const response =
            await fetch(
                (
                    full
                        ? '/radar/9tiles'
                        : '/radar/status'
                )
                + '?ts='
                + Date.now(),
                {
                    cache:
                        'no-store'
                }
            );

        const status =
            await response.json();

        if (
            status.estado
            === 'error'
        ) {

            throw new Error(
                status.error
                || 'Error de radar'
            );
        }

        renderRadar(
            status
        );

        const results =
            await Promise.all(
                [
                    fetch(
                        '/observacion?ts='
                        + Date.now(),
                        {
                            cache:
                                'no-store'
                        }
                    ).then(
                        r => r.json()
                    ),

                    fetch(
                        '/nowcast?ts='
                        + Date.now(),
                        {
                            cache:
                                'no-store'
                        }
                    ).then(
                        r => r.json()
                    )
                ]
            );

        renderObservation(
            results[0]
        );

        renderNowcast(
            results[1]
        );

    } catch (error) {

        $('main').innerHTML =
            '<span class="bad">'
            + 'No se pudo actualizar: '
            + text(error.message)
            + '</span>';

    } finally {

        $('refresh').disabled =
            false;
    }
}


function closeSheet() {

    $('sheet')
        .classList
        .add('closed');

    $('open').style.display =
        'block';

    setTimeout(
        () => map.invalidateSize(),
        250
    );
}


function openSheet() {

    $('sheet')
        .classList
        .remove('closed');

    $('open').style.display =
        'none';

    setTimeout(
        () => map.invalidateSize(),
        250
    );
}


$('refresh').onclick =
    () => loadRadar(true);

$('close').onclick =
    closeSheet;

$('open').onclick =
    openSheet;

$('toggle').onclick =
    () =>
        $('sheet')
            .classList
            .contains('closed')
            ? openSheet()
            : closeSheet;


loadRadar(true);

setInterval(
    () => loadRadar(false),
    60000
);

</script>

</body>
</html>'''


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
