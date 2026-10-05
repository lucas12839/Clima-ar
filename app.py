from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from datetime import datetime, timezone
from pathlib import Path
import math
import json
import time
import requests
from PIL import Image
from io import BytesIO


# ============================================================
# CLIMAAR
# ============================================================

VERSION = "4.3.1"

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

LAT = -38.71
LON = -62.26

RADAR_ZOOM = 7
TILE_SIZE = 256
GRID_RADIUS = 1

RAINVIEWER_API = (
    "https://api.rainviewer.com/public/weather-maps.json"
)

USER_AGENT = f"ClimaAR/{VERSION}"

HEADERS = {
    "User-Agent": USER_AGENT,
    "Referer": "https://www.rainviewer.com/",
    "Accept": "image/png,image/*;q=0.8,*/*;q=0.5",
    "Cache-Control": "no-cache"
}


# ============================================================
# ARCHIVOS
# ============================================================

ACTUAL_FILE = RADAR_DIR / "actual.png"

PREVIEW_FILE = RADAR_DIR / "preview.png"

STATUS_FILE = RADAR_DIR / "status.json"

NOWCAST_FILE = RADAR_DIR / "radar_nowcast.json"


# ============================================================
# UTILIDADES
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


def iso_now():
    return utc_now().isoformat()


def argentina_time(utc_dt):

    try:

        from zoneinfo import ZoneInfo

        return utc_dt.astimezone(
            ZoneInfo("America/Argentina/Buenos_Aires")
        )

    except Exception:

        return utc_dt


def clean_json_value(value):

    if isinstance(value, float):

        if not math.isfinite(value):
            return None

        return value

    if isinstance(value, dict):

        return {
            str(key): clean_json_value(val)
            for key, val in value.items()
        }

    if isinstance(value, list):

        return [
            clean_json_value(item)
            for item in value
        ]

    if isinstance(value, tuple):

        return [
            clean_json_value(item)
            for item in value
        ]

    return value


def load_json():

    if not STATUS_FILE.exists():
        return None

    try:

        raw = STATUS_FILE.read_text(
            encoding="utf-8"
        )

        if not raw.strip():
            return None

        data = json.loads(raw)

        return clean_json_value(data)

    except Exception:

        return None


def load_nowcast():

    if not NOWCAST_FILE.exists():
        return None

    try:

        raw = NOWCAST_FILE.read_text(
            encoding="utf-8"
        )

        if not raw.strip():
            return None

        data = json.loads(raw)

        return clean_json_value(data)

    except Exception:

        return None


def save_json(data):

    clean = clean_json_value(data)

    STATUS_FILE.write_text(
        json.dumps(
            clean,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        ),
        encoding="utf-8"
    )


# ============================================================
# COORDENADAS → TILE
# ============================================================

def lon_to_tile_x(lon, zoom):

    n = 2 ** zoom

    return int(
        (lon + 180.0)
        / 360.0
        * n
    )


def lat_to_tile_y(lat, zoom):

    lat_rad = math.radians(lat)

    n = 2 ** zoom

    y = (
        1
        - math.asinh(
            math.tan(lat_rad)
        ) / math.pi
    ) / 2 * n

    return int(y)


def tile_to_lon(x, zoom):

    n = 2 ** zoom

    return (
        x / n * 360.0
        - 180.0
    )


def tile_to_lat(y, zoom):

    n = 2 ** zoom

    lat_rad = math.atan(
        math.sinh(
            math.pi
            * (1 - 2 * y / n)
        )
    )

    return math.degrees(lat_rad)


# ============================================================
# RAINVIEWER API
# ============================================================

def rainviewer_data():

    response = requests.get(
        RAINVIEWER_API,
        headers=HEADERS,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    radar = data.get(
        "radar",
        {}
    )

    past = radar.get(
        "past",
        []
    )

    if not past:

        raise RuntimeError(
            "RainViewer no devolvió frames históricos."
        )

    frame = past[-1]

    host = data.get(
        "host",
        "https://tilecache.rainviewer.com"
    )

    path = frame.get(
        "path"
    )

    if not path:

        raise RuntimeError(
            "RainViewer no devolvió path para el frame."
        )

    timestamp = frame.get(
        "time"
    )

    return {
        "host": host,
        "path": path,
        "time": timestamp,
        "frames": len(past)
    }


# ============================================================
# URL DE TESELA
# ============================================================

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


# ============================================================
# DESCARGA DE TESELA
# ============================================================

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
# CONSTRUIR RADAR 3x3
# ============================================================

def build_9tiles():

    started = time.time()

    rv = rainviewer_data()

    host = rv["host"]

    path = rv["path"]

    timestamp = rv["time"]

    center_x = lon_to_tile_x(
        LON,
        RADAR_ZOOM
    )

    center_y = lat_to_tile_y(
        LAT,
        RADAR_ZOOM
    )

    canvas_size = (
        TILE_SIZE * 3
    )

    canvas = Image.new(
        "RGBA",
        (
            canvas_size,
            canvas_size
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

    for dy in range(
        -GRID_RADIUS,
        GRID_RADIUS + 1
    ):

        for dx in range(
            -GRID_RADIUS,
            GRID_RADIUS + 1
        ):

            x = center_x + dx

            y = center_y + dy

            url = tile_url(
                host,
                path,
                x,
                y
            )

            tile_info = {
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

                tile_info["http"] = (
                    response.status_code
                )

                tile_info["valid_png"] = True

                tile_info["size"] = len(
                    response.content
                )

                px = (
                    dx + GRID_RADIUS
                ) * TILE_SIZE

                py = (
                    dy + GRID_RADIUS
                ) * TILE_SIZE

                canvas.alpha_composite(
                    image,
                    (
                        px,
                        py
                    )
                )

                ok_count += 1

            except Exception as exc:

                tile_info["error"] = str(
                    exc
                )

            tiles.append(
                tile_info
            )

    canvas.save(
        ACTUAL_FILE,
        format="PNG"
    )

    canvas.save(
        PREVIEW_FILE,
        format="PNG"
    )

    left_x = (
        center_x
        - GRID_RADIUS
    )

    right_x = (
        center_x
        + GRID_RADIUS
        + 1
    )

    top_y = (
        center_y
        - GRID_RADIUS
    )

    bottom_y = (
        center_y
        + GRID_RADIUS
        + 1
    )

    west = tile_to_lon(
        left_x,
        RADAR_ZOOM
    )

    east = tile_to_lon(
        right_x,
        RADAR_ZOOM
    )

    north = tile_to_lat(
        top_y,
        RADAR_ZOOM
    )

    south = tile_to_lat(
        bottom_y,
        RADAR_ZOOM
    )

    frame_utc = None

    frame_argentina = None

    if timestamp:

        try:

            frame_dt = datetime.fromtimestamp(
                int(timestamp),
                tz=timezone.utc
            )

            frame_utc = (
                frame_dt.isoformat()
            )

            frame_argentina = (
                argentina_time(
                    frame_dt
                ).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

        except Exception:

            frame_utc = str(
                timestamp
            )

    status = {
        "estado": (
            "ok"
            if ok_count == 9
            else "parcial"
        ),
        "fuente": "RainViewer",
        "version": VERSION,
        "actualizado_utc": iso_now(),
        "frame_utc": frame_utc,
        "frame_argentina": frame_argentina,
        "frames_disponibles": rv[
            "frames"
        ],
        "teselas_ok": ok_count,
        "teselas_total": 9,
        "zoom": RADAR_ZOOM,
        "centro": {
            "lat": LAT,
            "lon": LON
        },
        "tile_centro": {
            "x": center_x,
            "y": center_y
        },
        "bounds": {
            "north": north,
            "south": south,
            "west": west,
            "east": east
        },
        "duracion_segundos": round(
            time.time() - started,
            2
        ),
        "tiles": tiles
    }

    save_json(status)

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
    <!DOCTYPE html>
    <html lang="es">

    <head>

        <meta charset="UTF-8">

        <meta name="viewport"
              content="width=device-width,
                       initial-scale=1.0">

        <title>ClimaAR</title>

        <style>

            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 30px;
                background: #111;
                color: white;
            }

            a {
                color: #4da6ff;
            }

        </style>

    </head>

    <body>

        <h1>ClimaAR</h1>

        <p>
            Sistema meteorológico experimental
            para Bahía Blanca.
        </p>

        <p>
            <a href="/radar">
                Abrir radar
            </a>
        </p>

        <p>
            <a href="/health">
                Estado del servicio
            </a>
        </p>

    </body>

    </html>
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
# ESTADO RADAR
# ============================================================

@app.get("/radar/status")
def radar_status():

    status = load_json()

    if status is None:

        return {
            "estado": "sin_datos",
            "fuente": "RainViewer",
            "mensaje":
                "Todavía no hay un radar generado."
        }

    return clean_json_value(
        status
    )


# ============================================================
# DEBUG
# ============================================================

@app.get("/radar/debug")
def radar_debug():

    try:

        rv = rainviewer_data()

        center_x = lon_to_tile_x(
            LON,
            RADAR_ZOOM
        )

        center_y = lat_to_tile_y(
            LAT,
            RADAR_ZOOM
        )

        url = tile_url(
            rv["host"],
            rv["path"],
            center_x,
            center_y
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
            "tile_url": url,
            "x": center_x,
            "y": center_y,
            "zoom": RADAR_ZOOM,
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
                "formato": image.format,
                "modo": image.mode,
                "ancho": image.width,
                "alto": image.height
            }

        except Exception as exc:

            result["png"] = {
                "valido": False,
                "error": str(exc)
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

        return {
            "estado": "sin_imagen"
        }

    return FileResponse(
        ACTUAL_FILE,
        media_type="image/png"
    )


@app.get("/radar/preview.png")
def radar_preview():

    if not PREVIEW_FILE.exists():

        try:
            build_9tiles()

        except Exception:
            pass

    if not PREVIEW_FILE.exists():

        return {
            "estado": "sin_imagen"
        }

    return FileResponse(
        PREVIEW_FILE,
        media_type="image/png"
    )


# ============================================================
# PÁGINA RADAR + NOWCAST
# ============================================================

@app.get(
    "/radar",
    response_class=HTMLResponse
)
def radar_page():

    return """
    <!DOCTYPE html>
    <html lang="es">

    <head>

        <meta charset="UTF-8">

        <meta name="viewport"
              content="width=device-width,
                       initial-scale=1.0,
                       maximum-scale=1.0,
                       user-scalable=no">

        <title>
            ClimaAR — Radar Bahía Blanca
        </title>

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
                min-height: 100%;
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
                margin: 0 0 5px 0;
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
                cursor: pointer;
            }

            button:active {
                opacity: 0.8;
            }

            #resultado {
                margin-top: 7px;
                font-size: 13px;
                min-height: 18px;
            }

            #map {
                width: 100%;
                height: calc(100vh - 145px);
                min-height: 520px;
                background: #222;
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

                background: rgba(17,17,17,0.94);

                border: 1px solid #444;

                border-radius: 12px;

                padding: 12px;

                box-shadow:
                    0 4px 16px
                    rgba(0,0,0,0.45);

                backdrop-filter: blur(5px);

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
                    repeat(2, 1fr);

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

                margin: 10px 0 8px 0;

                border-top: 1px solid #333;
            }

            .fuentes {

                font-size: 10px;

                color: #888;

                line-height: 1.4;
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

                border: 3px solid #1976d2;

                width: 18px;
                height: 18px;

                border-radius: 50%;

                box-shadow:
                    0 0 0 4px
                    rgba(25,118,210,0.35);
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
                class="estado-principal">
                Analizando condiciones...
            </div>

            <div
                id="estadoSecundario"
                class="estado-secundario">
                Cargando nowcast...
            </div>

            <div class="grid">

                <div class="dato">

                    <div class="dato-titulo">
                        DISTANCIA
                    </div>

                    <div
                        id="distancia"
                        class="dato-valor">
                        —
                    </div>

                </div>

                <div class="dato">

                    <div class="dato-titulo">
                        VELOCIDAD
                    </div>

                    <div
                        id="velocidad"
                        class="dato-valor">
                        —
                    </div>

                </div>

                <div class="dato">

                    <div class="dato-titulo">
                        DIRECCIÓN
                    </div>

                    <div
                        id="direccion"
                        class="dato-valor">
                        —
                    </div>

                </div>

                <div class="dato">

                    <div class="dato-titulo">
                        TENDENCIA
                    </div>

                    <div
                        id="fortalecimiento"
                        class="dato-valor">
                        —
                    </div>

                </div>

                <div class="dato">

                    <div class="dato-titulo">
                        ETA
                    </div>

                    <div
                        id="eta"
                        class="dato-valor">
                        —
                    </div>

                </div>

                <div class="dato">

                    <div class="dato-titulo">
                        CONFIANZA
                    </div>

                    <div
                        id="confianza"
                        class="dato-valor">
                        —
                    </div>

                </div>

                <div class="dato">

                    <div class="dato-titulo">
                        PROYECCIÓN 30 MIN
                    </div>

                    <div
                        id="proy30"
                        class="dato-valor">
                        —
                    </div>

                </div>

                <div class="dato">

                    <div class="dato-titulo">
                        PROYECCIÓN 60 MIN
                    </div>

                    <div
                        id="proy60"
                        class="dato-valor">
                        —
                    </div>

                </div>

            </div>

            <div class="separador"></div>

            <div
                id="fuentes"
                class="fuentes">
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
                "map",
                {
                    zoomControl: true,
                    attributionControl: true
                }
            );

            map.setView(
                centro,
                9
            );

            L.tileLayer(
                "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                {
                    maxZoom: 19,
                    attribution:
                        '&copy; OpenStreetMap contributors'
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
                "Bahía Blanca",
                {
                    permanent: false
                }
            );


            let radar = null;


            function texto(
                valor,
                sufijo = ""
            ) {

                if (
                    valor === null ||
                    valor === undefined ||
                    valor === ""
                ) {
                    return "—";
                }

                return String(valor) + sufijo;
            }


            function porcentaje(valor) {

                if (
                    valor === null ||
                    valor === undefined
                ) {
                    return "—";
                }

                const numero =
                    Number(valor);

                if (
                    !Number.isFinite(numero)
                ) {
                    return "—";
                }

                return Math.round(
                    numero * 100
                ) + "%";
            }


            function direccionTexto(
                direccion,
                grados
            ) {

                if (direccion) {
                    return direccion;
                }

                if (
                    grados === null ||
                    grados === undefined
                ) {
                    return "—";
                }

                const d =
                    Number(grados);

                if (
                    !Number.isFinite(d)
                ) {
                    return "—";
                }

                const nombres = [
                    "N",
                    "NE",
                    "E",
                    "SE",
                    "S",
                    "SO",
                    "O",
                    "NO"
                ];

                const indice =
                    (
                        Math.round(
                            d / 45
                        ) % 8 + 8
                    ) % 8;

                return nombres[
                    indice
                ];
            }


            /*
             * CORRECCIÓN PRINCIPAL:
             *
             * El JSON real tiene:
             *
             * nowcast_radar
             *   └── datos
             *        └── nowcast
             *
             * Por eso entramos hasta
             * datos.nowcast.
             */

            function obtenerNowcast(
                estado
            ) {

                if (
                    estado &&
                    estado.nowcast_radar &&
                    estado.nowcast_radar.datos &&
                    estado.nowcast_radar.datos.nowcast
                ) {

                    return estado
                        .nowcast_radar
                        .datos
                        .nowcast;

                }

                /*
                 * Compatibilidad por si en el futuro
                 * el formato cambia.
                 */

                if (
                    estado &&
                    estado.nowcast_radar &&
                    estado.nowcast_radar.datos
                ) {

                    const datos =
                        estado.nowcast_radar.datos;

                    if (
                        datos.actividad !== undefined ||
                        datos.frames_analizados !== undefined
                    ) {

                        return datos;

                    }

                }

                if (
                    estado &&
                    estado.nowcast_radar &&
                    estado.nowcast_radar.nowcast
                ) {

                    return estado
                        .nowcast_radar
                        .nowcast;

                }

                return null;
            }


            function mostrarNowcast(
                estado
            ) {

                const n =
                    obtenerNowcast(
                        estado
                    );

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


                const actividad =
                    n.actividad === true;


                if (!actividad) {

                    principal.textContent =
                        "🟢 Sin precipitación detectada";

                    principal.className =
                        "estado-principal sin-actividad";

                    secundario.textContent =
                        "Radar operativo · " +
                        texto(
                            n.frames_analizados
                        ) +
                        " frames analizados";

                } else {

                    principal.textContent =
                        "🟠 Precipitación detectada";

                    principal.className =
                        "estado-principal actividad";

                    secundario.textContent =
                        "Seguimiento radar activo · " +
                        texto(
                            n.frames_analizados
                        ) +
                        " frames analizados";
                }


                document.getElementById(
                    "distancia"
                ).textContent =
                    n.distancia_km !== null &&
                    n.distancia_km !== undefined
                    ? texto(
                        Number(
                            n.distancia_km
                        ).toFixed(1),
                        " km"
                    )
                    : "—";


                document.getElementById(
                    "velocidad"
                ).textContent =
                    n.velocidad_kmh !== null &&
                    n.velocidad_kmh !== undefined
                    ? texto(
                        Number(
                            n.velocidad_kmh
                        ).toFixed(1),
                        " km/h"
                    )
                    : "—";


                document.getElementById(
                    "direccion"
                ).textContent =
                    direccionTexto(
                        n.direccion,
                        n.direccion_grados
                    );


                document.getElementById(
                    "fortalecimiento"
                ).textContent =
                    n.fortalecimiento
                    ? n.fortalecimiento
                    : "—";


                document.getElementById(
                    "eta"
                ).textContent =
                    n.eta_minutos !== null &&
                    n.eta_minutos !== undefined
                    ? texto(
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
                    porcentaje(
                        n.confianza_movimiento
                    );


                document.getElementById(
                    "proy30"
                ).textContent =
                    n.proyeccion_30_min !== null &&
                    n.proyeccion_30_min !== undefined
                    ? String(
                        n.proyeccion_30_min
                    )
                    : "—";


                document.getElementById(
                    "proy60"
                ).textContent =
                    n.proyeccion_60_min !== null &&
                    n.proyeccion_60_min !== undefined
                    ? String(
                        n.proyeccion_60_min
                    )
                    : "—";


                const fuentes =
                    document.getElementById(
                        "fuentes"
                    );

                let textoFuentes =
                    "Radar: RainViewer";


                if (
                    estado &&
                    estado.sazb &&
                    estado.sazb.disponible
                ) {

                    textoFuentes +=
                        " · SAZB";

                }


                if (
                    estado &&
                    estado.openmeteo &&
                    estado.openmeteo.disponible
                ) {

                    textoFuentes +=
                        " · Open-Meteo";

                }


                if (
                    estado &&
                    estado.modelo_historico &&
                    estado.modelo_historico.disponible
                ) {

                    textoFuentes +=
                        " · Modelo histórico";

                }


                fuentes.textContent =
                    textoFuentes;
            }


            /*
             * ACTUALIZAR OVERLAY
             */

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
                            "/radar.png?ts=" +
                            Date.now(),
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
                        "/radar.png?ts=" +
                        Date.now()
                    );
                }

                return true;
            }


            /*
             * CARGAR ESTADO
             */

            async function cargarEstado() {

                try {

                    let response =
                        await fetch(
                            "/estado?ts=" +
                            Date.now()
                        );


                    if (!response.ok) {

                        throw new Error(
                            "HTTP " +
                            response.status
                        );
                    }


                    let data =
                        await response.json();


                    /*
                     * CORRECCIÓN:
                     *
                     * Si Render arrancó sin
                     * status.json, generamos
                     * automáticamente el radar.
                     *
                     * Ya no hace falta tocar
                     * "Actualizar radar".
                     */

                    if (
                        !data.radar
                    ) {

                        document.getElementById(
                            "resultado"
                        ).textContent =
                            "Generando radar...";


                        const radarResponse =
                            await fetch(
                                "/radar/9tiles?ts=" +
                                Date.now()
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
                                "Radar generado automáticamente · " +
                                radarData.teselas_ok +
                                "/" +
                                radarData.teselas_total;

                        }

                    }


                    mostrarNowcast(
                        data
                    );


                    /*
                     * RADAR
                     */

                    if (
                        data.radar &&
                        data.radar.bounds &&
                        data.radar.bounds.north !==
                        undefined
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
                            "RainViewer · último frame: " +
                            data.radar.frame_argentina +
                            " · teselas: " +
                            texto(
                                data.radar.teselas_ok
                            ) +
                            "/" +
                            texto(
                                data.radar.teselas_total
                            );

                    } else {

                        const n =
                            obtenerNowcast(
                                data
                            );

                        if (
                            n &&
                            n.frame_actual_utc
                        ) {

                            document.getElementById(
                                "estado"
                            ).textContent =
                                "RainViewer · frame: " +
                                n.frame_actual_utc;
                        }
                    }


                    /*
                     * DATOS SAZB
                     */

                    if (
                        data.sazb &&
                        data.sazb.disponible
                    ) {

                        const s =
                            data.sazb;

                        const temp =
                            s.temperatura_c;

                        const viento =
                            s.viento_kt;

                        const presion =
                            s.presion_hpa;

                        let extra =
                            "";

                        if (
                            temp !== null &&
                            temp !== undefined
                        ) {

                            extra +=
                                " · " +
                                Number(
                                    temp
                                ).toFixed(1) +
                                "°C";
                        }

                        if (
                            viento !== null &&
                            viento !== undefined
                        ) {

                            extra +=
                                " · viento " +
                                Number(
                                    viento
                                ).toFixed(0) +
                                " kt";
                        }

                        if (
                            presion !== null &&
                            presion !== undefined
                        ) {

                            extra +=
                                " · " +
                                Number(
                                    presion
                                ).toFixed(0) +
                                " hPa";
                        }

                        document.getElementById(
                            "resultado"
                        ).textContent =
                            "SAZB" +
                            extra;
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

                const resultado =
                    document.getElementById(
                        "resultado"
                    );

                resultado.textContent =
                    "Actualizando radar...";


                try {

                    const response =
                        await fetch(
                            "/radar/9tiles?ts=" +
                            Date.now()
                        );

                    const data =
                        await response.json();


                    if (
                        data.estado !== "ok"
                    ) {

                        resultado.textContent =
                            "Error: " +
                            (
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


                    resultado.textContent =
                        "Radar actualizado · " +
                        data.teselas_ok +
                        "/" +
                        data.teselas_total;


                    document.getElementById(
                        "estado"
                    ).textContent =
                        "RainViewer · último frame: " +
                        (
                            data.frame_argentina ||
                            "actualizado"
                        ) +
                        " · teselas: " +
                        data.teselas_ok +
                        "/" +
                        data.teselas_total;


                    await cargarEstado();


                } catch (error) {

                    resultado.textContent =
                        "Error de conexión: " +
                        error;
                }
            }


            /*
             * INICIO
             */

            setTimeout(
                function() {

                    map.invalidateSize();

                },
                300
            );


            /*
             * CARGA INICIAL AUTOMÁTICA
             */

            cargarEstado();


            /*
             * ACTUALIZACIÓN DEL ANÁLISIS
             * CADA 60 SEGUNDOS
             */

            setInterval(
                cargarEstado,
                60000
            );

        </script>

    </body>

    </html>
    """


# ============================================================
# OBSERVACIÓN SAZB
# ============================================================

@app.get("/observacion")
def observacion():

    return {
        "estado": "ok",
        "fuente": "SAZB",
        "mensaje":
            "Endpoint de observación meteorológica.",
        "hora_utc": iso_now()
    }


# ============================================================
# ESTADO GENERAL + NOWCAST
# ============================================================

@app.get("/estado")
def estado():

    try:

        radar = load_json()

        /*
         * Si Render arrancó sin status.json,
         * generamos el radar automáticamente.
         */

        if radar is None:

            try:

                radar = build_9tiles()

            except Exception:

                radar = None


        nowcast_data = load_nowcast()


        if nowcast_data is None:

            nowcast = {
                "disponible": False,
                "estado": "sin_datos",
                "datos": None,
                "motivo":
                    "Todavia no existe radar_nowcast.json."
            }

        else:

            nowcast = {
                "disponible": True,
                "estado": "ok",
                "datos": nowcast_data
            }


        respuesta = {
            "app": "ClimaAR",
            "version": VERSION,
            "radar": radar,
            "nowcast_radar": nowcast,
            "nowcast_radar_operativo":
                bool(
                    nowcast.get(
                        "disponible",
                        False
                    )
                ),
            "nowcast_radar_ia": False,
            "hora_utc": iso_now()
        }


        respuesta =
            clean_json_value(
                respuesta
            )


        return JSONResponse(
            content=respuesta
        )


    except Exception as exc:

        return JSONResponse(
            status_code=200,
            content={
                "app": "ClimaAR",
                "version": VERSION,
                "estado": "parcial",
                "error": str(exc),
                "radar": None,
                "nowcast_radar": {
                    "disponible": False,
                    "estado": "error"
                },
                "nowcast_radar_operativo": False,
                "nowcast_radar_ia": False,
                "hora_utc": iso_now()
            }
        )


# ============================================================
# MODELO
# ============================================================

@app.get("/modelo")
def modelo():

    return {
        "estado": "experimental",
        "mensaje":
            "Módulo de análisis meteorológico de ClimaAR."
    }


# ============================================================
# TORMENTA
# ============================================================

@app.get("/tormenta")
def tormenta():

    return {
        "estado": "ok",
        "zona": "Bahía Blanca",
        "mensaje":
            "Análisis de tormenta disponible cuando existan datos suficientes."
    }


# ============================================================
# NOWCAST
# ============================================================

@app.get("/nowcast")
def nowcast():

    data = load_nowcast()

    if data is None:

        return {
            "disponible": False,
            "estado": "sin_datos",
            "zona": "Bahía Blanca",
            "mensaje":
                "Todavia no existe radar_nowcast.json."
        }

    return {
        "disponible": True,
        "estado": "ok",
        "zona": "Bahía Blanca",
        "fuente": "RainViewer",
        "nowcast": clean_json_value(
            data
        )
        }
