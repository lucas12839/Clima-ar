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

VERSION = "4.4.0"

BASE_DIR = Path(__file__).resolve().parent

RADAR_DIR = BASE_DIR / "data" / "radar"
SAZB_DIR = BASE_DIR / "data" / "sazb"

RADAR_DIR.mkdir(parents=True, exist_ok=True)

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
    description="Radar meteorológico y seguimiento de tormentas para Bahía Blanca",
    version=VERSION
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


def clean(v):
    if isinstance(v, float):
        if not math.isfinite(v):
            return None

    if isinstance(v, dict):
        return {
            str(k): clean(x)
            for k, x in v.items()
        }

    if isinstance(v, list):
        return [
            clean(x)
            for x in v
        ]

    return v


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


def rainviewer_data():

    r = requests.get(
        RAINVIEWER_API,
        headers=HEADERS,
        timeout=20
    )

    r.raise_for_status()

    data = r.json()

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

                r = requests.get(
                    url,
                    headers=HEADERS,
                    timeout=20
                )

                r.raise_for_status()

                im = Image.open(
                    BytesIO(
                        r.content
                    )
                ).convert("RGBA")

                canvas.alpha_composite(
                    im,
                    (
                        (dx + 1) * TILE_SIZE,
                        (dy + 1) * TILE_SIZE
                    )
                )

                item.update({
                    "http": r.status_code,
                    "valid_png": True,
                    "size": len(r.content)
                })

                ok += 1

            except Exception as e:

                item["error"] = str(e)

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
        "estado": "ok",
        "servicio": "ClimaAR",
        "version": VERSION,
        "hora_utc": iso_now()
    }


@app.get("/radar/9tiles")
def radar_9tiles():

    try:
        return build_radar()

    except Exception as e:

        return {
            "estado": "error",
            "fuente": "RainViewer",
            "error": str(e),
            "hora_utc": iso_now()
        }


@app.get("/radar/status")
def radar_status():

    return (
        load_json(STATUS_FILE)
        or {
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

        r = requests.get(
            url,
            headers=HEADERS,
            timeout=20
        )

        r.raise_for_status()

        im = Image.open(
            BytesIO(
                r.content
            )
        )

        return {

            "estado": "ok",

            "api_http":
                200,

            "tile_http":
                r.status_code,

            "tile_url":
                url,

            "x":
                x,

            "y":
                y,

            "zoom":
                RADAR_ZOOM,

            "bytes":
                len(r.content),

            "png": {
                "valido":
                    True,

                "formato":
                    im.format,

                "modo":
                    im.mode,

                "ancho":
                    im.width,

                "alto":
                    im.height
            }
        }

    except Exception as e:

        return {
            "estado": "error",
            "error": str(e)
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
                "estado": "sin_imagen"
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


@app.get("/estado")
def estado():

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
                NOWCAST_FILE.exists(),

            "estado":
                "ok"
                if NOWCAST_FILE.exists()
                else "sin_datos",

            "datos":
                load_json(
                    NOWCAST_FILE
                )
        },

        "nowcast_radar_operativo":
            NOWCAST_FILE.exists(),

        "nowcast_radar_ia":
            False,

        "hora_utc":
            iso_now()
    })


@app.get("/nowcast")
def nowcast():

    data = load_json(
        NOWCAST_FILE
    )

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
        or {
            "estado": "sin_datos",
            "fuente":
                "Aviation Weather Center",
            "estacion":
                "SAZB"
        }
    )


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
    box-sizing: border-box
}

html,
body {
    margin: 0;
    background: #111;
    color: #fff;
    font-family: Arial, sans-serif
}

.top {
    padding: 12px;
    background: #171717;
    border-bottom: 1px solid #333;
    position: relative;
    z-index: 1000
}

h1 {
    margin: 0 0 5px;
    font-size: 20px
}

#estado {
    font-size: 13px;
    color: #ccc;
    margin-bottom: 9px
}

button {
    width: 100%;
    padding: 12px;
    border: 0;
    border-radius: 8px;
    background: #1976d2;
    color: #fff;
    font-size: 16px;
    font-weight: bold
}

#resultado {
    margin-top: 7px;
    font-size: 13px;
    min-height: 18px
}

#map {
    width: 100%;
    height: calc(100vh - 150px);
    min-height: 520px
}

.panel {
    position: absolute;
    left: 10px;
    right: 10px;
    bottom: 15px;
    z-index: 999;
    background: rgba(17,17,17,.94);
    border: 1px solid #444;
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,.45);
    max-width: 560px;
    margin: auto;
    max-height: 55vh;
    overflow-y: auto
}

.main {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 5px
}

.sub {
    font-size: 12px;
    color: #bbb;
    margin-bottom: 10px
}

.grid {
    display: grid;
    grid-template-columns: repeat(2,1fr);
    gap: 7px
}

.dato {
    background: #222;
    border-radius: 8px;
    padding: 8px
}

.dt {
    font-size: 10px;
    color: #999;
    margin-bottom: 3px
}

.dv {
    font-size: 14px;
    font-weight: bold
}

.sep {
    margin: 10px 0 8px;
    border-top: 1px solid #333
}

.fuentes,
.obs {
    font-size: 10px;
    color: #888;
    margin-top: 7px
}

.legend {
    font-size: 11px;
    font-weight: bold;
    margin: 8px 0 5px
}

.bar {
    display: flex;
    height: 14px;
    border-radius: 4px;
    overflow: hidden
}

.bar span {
    flex: 1
}

.labels {
    display: flex;
    justify-content: space-between;
    font-size: 9px;
    color: #aaa;
    margin-top: 3px
}

.ok {
    color: #58d68d
}

.warn {
    color: #ffb84d
}

.bad {
    color: #ff6b6b
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

<div class="dato">
<div class="dt">RÁFAGA</div>
<div id="rafaga" class="dv">—</div>
</div>

<div class="dato">
<div class="dt">HUMEDAD</div>
<div id="humedad" class="dv">—</div>
</div>

</div>

<div class="sep"></div>

<div class="grid">

<div class="dato">
<div class="dt">DISTANCIA</div>
<div id="dist" class="dv">—</div>
</div>

<div class="dato">
<div class="dt">VELOCIDAD CÉLULA</div>
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
Radar: RainViewer · Observación: Aviation Weather Center SAZB · Nowcast: ClimaAR
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
    s = ''
) =>
    (
        v === null ||
        v === undefined ||
        v === ''
    )
    ? '—'
    : String(v) + s;


function direction(d) {

    if (
        d === null ||
        d === undefined ||
        d === ''
    ) {
        return '—';
    }

    const n = Number(d);

    if (!Number.isFinite(n)) {
        return String(d);
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


function nowcast(d) {

    const n =
        d?.nowcast_radar?.datos;

    if (!n) {

        el('principal')
            .textContent =
            'Nowcast no disponible';

        el('principal')
            .className =
            'main bad';

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
        'Radar operativo · ' +
        val(n.frames_analizados) +
        ' frames analizados';

    el('dist')
        .textContent =
        n.distancia_km != null
        ? Number(
            n.distancia_km
        ).toFixed(1) + ' km'
        : '—';

    el('vel')
        .textContent =
        n.velocidad_kmh != null
        ? Number(
            n.velocidad_kmh
        ).toFixed(1) + ' km/h'
        : '—';

    el('celldir')
        .textContent =
        direction(
            n.direccion_grados ??
            n.direccion
        );

    el('trend')
        .textContent =
        val(
            n.fortalecimiento
        );

    el('eta')
        .textContent =
        n.eta_minutos != null
        ? Math.round(
            n.eta_minutos
        ) + ' min'
        : '—';

    el('conf')
        .textContent =
        n.confianza_movimiento != null
        ? Math.round(
            Number(
                n.confianza_movimiento
            ) * 100
        ) + '%'
        : '—';
}


function sazb(s) {

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
        ? kmh.toFixed(1) +
          ' km/h'
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
        ).toFixed(1) +
        ' hPa'
        : '—';

    el('vis')
        .textContent =
        val(
            s.visibilidad_millas,
            ' mi'
        );

    el('rafaga')
        .textContent =
        '—';

    el('humedad')
        .textContent =
        '—';

    el('obs')
        .textContent =
        'SAZB · observación hace ' +
        val(
            s.edad_minutos,
            ' min'
        );
}


function overlay(r) {

    if (!r?.bounds) {
        return;
    }

    const b = [
        [
            r.bounds.south,
            r.bounds.west
        ],
        [
            r.bounds.north,
            r.bounds.east
        ]
    ];

    if (!radar) {

        radar =
            L.imageOverlay(
                '/radar.png?ts=' +
                Date.now(),
                b,
                {
                    opacity: .65
                }
            ).addTo(map);

    } else {

        radar.setBounds(b);

        radar.setUrl(
            '/radar.png?ts=' +
            Date.now()
        );
    }
}


async function load() {

    try {

        let r =
            await fetch(
                '/estado?ts=' +
                Date.now()
            );

        let d =
            await r.json();

        if (!d.radar) {

            const rr =
                await fetch(
                    '/radar/9tiles?ts=' +
                    Date.now()
                );

            d.radar =
                await rr.json();
        }

        nowcast(d);

        sazb(
            d.observacion_sazb
        );

        overlay(
            d.radar
        );

        if (
            d.radar?.frame_argentina
        ) {

            el('estado')
                .textContent =
                'RainViewer · último frame: ' +
                d.radar.frame_argentina +
                ' · teselas: ' +
                val(
                    d.radar.teselas_ok
                ) +
                '/' +
                val(
                    d.radar.teselas_total
                );
        }

    } catch (e) {

        el('resultado')
            .textContent =
            'No se pudo actualizar el análisis.';

        console.log(e);
    }
}


async function actualizarRadar() {

    el('resultado')
        .textContent =
        'Actualizando radar...';

    try {

        const r =
            await fetch(
                '/radar/9tiles?ts=' +
                Date.now()
            );

        const d =
            await r.json();

        if (
            d.estado !== 'ok'
        ) {

            el('resultado')
                .textContent =
                'Error: ' +
                (
                    d.error ||
                    'No se pudo actualizar.'
                );

            return;
        }

        overlay(d);

        el('resultado')
            .textContent =
            'Radar actualizado · ' +
            d.teselas_ok +
            '/' +
            d.teselas_total;

        await load();

    } catch (e) {

        el('resultado')
            .textContent =
            'Error de conexión: ' +
            e;
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


@app.get(
    "/radar",
    response_class=HTMLResponse
)
def radar_page():

    return HTML
