from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from datetime import datetime, timezone
from pathlib import Path
import math
import json
import time
import requests
from PIL import Image
from io import BytesIO

app = FastAPI(
    title="ClimaAR",
    description="Radar meteorológico y seguimiento de tormentas para Bahía Blanca",
    version="4.2.1"
)

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RADAR_DIR = BASE_DIR / "data" / "radar"

RADAR_DIR.mkdir(parents=True, exist_ok=True)

LAT = -38.71
LON = -62.26

RADAR_ZOOM = 7
TILE_SIZE = 256
GRID_RADIUS = 1

RAINVIEWER_API = "https://api.rainviewer.com/public/weather-maps.json"

USER_AGENT = "ClimaAR/4.2.1"

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


def argentina_time(utc_dt):
    try:
        from zoneinfo import ZoneInfo
        return utc_dt.astimezone(
            ZoneInfo("America/Argentina/Buenos_Aires")
        )
    except Exception:
        return utc_dt


def iso_now():
    return utc_now().isoformat()


def save_json(data):
    STATUS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def load_json():
    if not STATUS_FILE.exists():
        return None

    try:
        return json.loads(
            STATUS_FILE.read_text(encoding="utf-8")
        )
    except Exception:
        return None


def load_nowcast():
    if not NOWCAST_FILE.exists():
        return None

    try:
        return json.loads(
            NOWCAST_FILE.read_text(encoding="utf-8")
        )
    except Exception:
        return None


# ============================================================
# COORDENADAS → TILE
# ============================================================

def lon_to_tile_x(lon, zoom):
    n = 2 ** zoom
    return int((lon + 180.0) / 360.0 * n)


def lat_to_tile_y(lat, zoom):
    lat_rad = math.radians(lat)

    n = 2 ** zoom

    y = (
        1
        - math.asinh(math.tan(lat_rad)) / math.pi
    ) / 2 * n

    return int(y)


def tile_to_lon(x, zoom):
    n = 2 ** zoom
    return x / n * 360.0 - 180.0


def tile_to_lat(y, zoom):
    n = 2 ** zoom

    lat_rad = math.atan(
        math.sinh(
            math.pi * (1 - 2 * y / n)
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

    radar = data.get("radar", {})

    past = radar.get("past", [])

    if not past:
        raise RuntimeError(
            "RainViewer no devolvió frames históricos."
        )

    frame = past[-1]

    host = data.get(
        "host",
        "https://tilecache.rainviewer.com"
    )

    path = frame.get("path")

    if not path:
        raise RuntimeError(
            "RainViewer no devolvió path para el frame."
        )

    timestamp = frame.get("time")

    return {
        "host": host,
        "path": path,
        "time": timestamp,
        "frames": len(past)
    }


# ============================================================
# URL DE TESELA
# ============================================================

def tile_url(host, path, x, y):

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
        BytesIO(response.content)
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

    canvas_size = TILE_SIZE * 3

    canvas = Image.new(
        "RGBA",
        (canvas_size, canvas_size),
        (0, 0, 0, 0)
    )

    tiles = []

    ok_count = 0

    for dy in range(-GRID_RADIUS, GRID_RADIUS + 1):

        for dx in range(-GRID_RADIUS, GRID_RADIUS + 1):

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

                image, response = download_tile(url)

                tile_info["http"] = response.status_code
                tile_info["valid_png"] = True
                tile_info["size"] = len(
                    response.content
                )

                px = (dx + GRID_RADIUS) * TILE_SIZE
                py = (dy + GRID_RADIUS) * TILE_SIZE

                canvas.alpha_composite(
                    image,
                    (px, py)
                )

                ok_count += 1

            except Exception as exc:

                tile_info["error"] = str(exc)

            tiles.append(tile_info)

    canvas.save(
        ACTUAL_FILE,
        format="PNG"
    )

    canvas.save(
        PREVIEW_FILE,
        format="PNG"
    )

    # Límites geográficos de las 3x3 teselas

    left_x = center_x - GRID_RADIUS
    right_x = center_x + GRID_RADIUS + 1

    top_y = center_y - GRID_RADIUS
    bottom_y = center_y + GRID_RADIUS + 1

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

            frame_utc = frame_dt.isoformat()

            frame_argentina = (
                argentina_time(frame_dt)
                .strftime("%Y-%m-%d %H:%M:%S")
            )

        except Exception:
            frame_utc = str(timestamp)

    status = {
        "estado": "ok" if ok_count == 9 else "parcial",
        "fuente": "RainViewer",
        "version": "4.2.1",
        "actualizado_utc": iso_now(),
        "frame_utc": frame_utc,
        "frame_argentina": frame_argentina,
        "frames_disponibles": rv["frames"],
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

    return status


# ============================================================
# PÁGINA PRINCIPAL
# ============================================================

@app.get("/", response_class=HTMLResponse)
def inicio():

    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>

        <meta charset="UTF-8">

        <meta name="viewport"
              content="width=device-width, initial-scale=1.0">

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
        "version": "4.2.1",
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
            "mensaje": "Todavía no hay un radar generado."
        }

    return status


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
            "tile_http": response.status_code,
            "tile_url": url,
            "x": center_x,
            "y": center_y,
            "zoom": RADAR_ZOOM,
            "bytes": len(response.content)
        }

        try:

            image = Image.open(
                BytesIO(response.content)
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

        return result

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
# PÁGINA RADAR
# ============================================================

@app.get("/radar", response_class=HTMLResponse)
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
                min-height: 100%;
                background: #111;
                color: white;
                font-family: Arial, sans-serif;
            }

            .top {
                padding: 12px;
                background: #171717;
                border-bottom: 1px solid #333;
            }

            h1 {
                margin: 0 0 6px 0;
                font-size: 20px;
            }

            #estado {
                font-size: 13px;
                color: #ccc;
                margin-bottom: 10px;
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
                margin-top: 8px;
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

        </style>

    </head>

    <body>

        <div class="top">

            <h1>
                ClimaAR — Radar Bahía Blanca
            </h1>

            <div id="estado">
                RainViewer · radar listo
            </div>

            <button onclick="actualizarRadar()">
                Actualizar radar
            </button>

            <div id="resultado"></div>

        </div>

        <div id="map"></div>


        <script
            src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js">
        </script>


        <script>

            // ==================================================
            // CENTRO EXACTO DE BAHÍA BLANCA
            // ==================================================

            const centro = [-38.71, -62.26];


            // ==================================================
            // MAPA
            // ==================================================

            const map = L.map("map", {
                zoomControl: true,
                attributionControl: true
            });


            map.setView(centro, 9);


            // ==================================================
            // MAPA BASE
            // ==================================================

            L.tileLayer(
                "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                {
                    maxZoom: 19,
                    attribution:
                        '&copy; OpenStreetMap contributors'
                }
            ).addTo(map);


            // ==================================================
            // OVERLAY RADAR
            // ==================================================

            let radar = null;


            async function cargarRadarGuardado() {

                try {

                    const response =
                        await fetch(
                            "/radar/status?ts=" +
                            Date.now()
                        );

                    if (!response.ok) {
                        return;
                    }

                    const data =
                        await response.json();

                    if (
                        data.bounds &&
                        data.bounds.north !== undefined
                    ) {

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

                        radar =
                            L.imageOverlay(
                                "/radar.png?ts=" +
                                Date.now(),
                                bounds,
                                {
                                    opacity: 0.65,
                                    interactive: false
                                }
                            ).addTo(map);

                    }

                    if (data.frame_argentina) {

                        document.getElementById(
                            "estado"
                        ).textContent =
                            "RainViewer · último frame: " +
                            data.frame_argentina +
                            " · teselas: " +
                            data.teselas_ok +
                            "/" +
                            data.teselas_total;

                    }

                } catch (error) {

                    console.log(
                        "No hay radar guardado todavía."
                    );

                }
            }


            // ==================================================
            // ACTUALIZAR RADAR
            // ==================================================

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


                    if (data.estado !== "ok") {

                        resultado.textContent =
                            "Error: " +
                            (
                                data.error ||
                                "No se pudo actualizar."
                            );

                        return;
                    }


                    if (radar) {

                        radar.setUrl(
                            "/radar.png?ts=" +
                            Date.now()
                        );

                    } else {

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

                        radar =
                            L.imageOverlay(
                                "/radar.png?ts=" +
                                Date.now(),
                                bounds,
                                {
                                    opacity: 0.65,
                                    interactive: false
                                }
                            ).addTo(map);

                    }


                    map.setView(
                        centro,
                        9
                    );


                    resultado.textContent =
                        "OK · " +
                        data.teselas_ok +
                        "/" +
                        data.teselas_total +
                        " · " +
                        (
                            data.frame_argentina ||
                            "frame actualizado"
                        );


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


                } catch (error) {

                    resultado.textContent =
                        "Error de conexión: " +
                        error;

                }

            }


            // ==================================================
            // CORREGIR TAMAÑO DEL MAPA
            // ==================================================

            setTimeout(
                function() {
                    map.invalidateSize();
                },
                300
            );


            // ==================================================
            // CARGAR RADAR GUARDADO
            // ==================================================

            cargarRadarGuardado();

        </script>

    </body>

    </html>
    """


# ============================================================
# OBSERVACIÓN
# ============================================================

@app.get("/observacion")
def observacion():

    return {
        "estado": "ok",
        "fuente": "SAZB",
        "mensaje": "Endpoint de observación meteorológica.",
        "hora_utc": iso_now()
    }


# ============================================================
# ESTADO GENERAL + NOWCAST
# ============================================================

@app.get("/estado")
def estado():

    radar = load_json()

    nowcast = load_nowcast()

    if nowcast is None:

        nowcast = {
            "disponible": False,
            "estado": "sin_datos",
            "motivo":
                "Todavia no existe radar_nowcast.json."
        }

    else:

        nowcast = {
            "disponible": True,
            "estado": "ok",
            "datos": nowcast
        }

    return {
        "app": "ClimaAR",
        "version": "4.2.1",
        "radar": radar,
        "nowcast_radar": nowcast,
        "nowcast_radar_operativo":
            nowcast.get(
                "disponible",
                False
            ),
        "nowcast_radar_ia": False,
        "hora_utc": iso_now()
    }


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
        "nowcast": data
    }
