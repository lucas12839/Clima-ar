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


app = FastAPI(
    title="ClimaAR",
    description="Seguimiento operativo de precipitacion para Bahia Blanca.",
    version="4.1.0-9tiles",
)

LAT = -38.71
LON = -62.26
RADAR_ZOOM = 7
TILE_SIZE = 256
GRID_RADIUS = 1

RAINVIEWER_API = "https://api.rainviewer.com/public/weather-maps.json"
RAINVIEWER_DEFAULT_HOST = "https://tilecache.rainviewer.com"

DATA_DIR = Path("data/radar")
RADAR_IMAGE = DATA_DIR / "actual.png"
RADAR_PREVIEW = DATA_DIR / "preview.png"
RADAR_STATUS = DATA_DIR / "status.json"

OBS_STATUS = Path("data/sazb/status.json")

DATA_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()

session.headers.update({
    "User-Agent": "ClimaAR/4.1",
    "Accept": "*/*",
})


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def argentina_time(ts: int | None) -> str | None:
    if not ts:
        return None

    return datetime.fromtimestamp(
        int(ts),
        timezone.utc
    ).astimezone(
        timezone(timedelta(hours=-3))
    ).strftime("%Y-%m-%d %H:%M:%S")


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
        x
        / (2 ** zoom)
        * 360.0
        - 180.0
    )


def tile_y_to_lat(
    y: int,
    zoom: int
) -> float:

    n = (
        math.pi
        - 2.0
        * math.pi
        * y
        / (2 ** zoom)
    )

    return math.degrees(
        math.atan(
            math.sinh(n)
        )
    )


def rainviewer_data() -> tuple[dict, dict]:

    response = session.get(
        RAINVIEWER_API,
        timeout=20,
        headers={
            "User-Agent": "ClimaAR/4.1",
            "Accept": "application/json",
            "Cache-Control": "no-cache",
        },
    )

    response.raise_for_status()

    data = response.json()

    frames = (
        data
        .get("radar", {})
        .get("past", [])
    )

    if not frames:
        raise RuntimeError(
            "RainViewer no devolvio frames radar."
        )

    frame = frames[-1]

    timestamp = int(
        frame.get("time", 0)
    )

    path = frame.get("path")

    if timestamp <= 0 or not path:
        raise RuntimeError(
            "Frame RainViewer invalido."
        )

    host = (
        data.get("host")
        or RAINVIEWER_DEFAULT_HOST
    ).rstrip("/")

    return data, {
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
        "path": path,
        "host": host,
        "frames_disponibles": len(frames),
        "generado_utc": (
            datetime.fromtimestamp(
                int(data["generated"]),
                timezone.utc
            ).isoformat()
            if data.get("generated")
            else None
        ),
    }


def tile_url(
    host: str,
    path: str,
    x: int,
    y: int
) -> str:

    return (
        f"{host}"
        f"{path}"
        f"/{TILE_SIZE}"
        f"/{RADAR_ZOOM}"
        f"/{x}"
        f"/{y}"
        f"/2/1_1.png"
    )


def download_tile(
    url: str
) -> tuple[bytes, dict]:

    response = session.get(
        url,
        timeout=20,
        headers={
            "User-Agent": "ClimaAR/4.1",
            "Referer": "https://www.rainviewer.com/",
            "Accept": (
                "image/png,image/*;"
                "q=0.8,*/*;q=0.5"
            ),
            "Cache-Control": "no-cache",
        },
    )

    meta = {
        "status_code": response.status_code,
        "content_type": response.headers.get(
            "content-type"
        ),
        "bytes": len(response.content),
        "server": response.headers.get(
            "server"
        ),
        "first_bytes_hex": (
            response.content[:16].hex()
        ),
        "imagen_valida": False,
    }

    if response.status_code != 200:
        return response.content, meta

    if Image is None:
        meta["error"] = (
            "Pillow no esta disponible."
        )
        return response.content, meta

    try:

        with Image.open(
            BytesIO(response.content)
        ) as image:

            image.load()

            meta.update({
                "imagen_valida": True,
                "formato": image.format,
                "modo": image.mode,
                "ancho": image.width,
                "alto": image.height,
            })

    except Exception as exc:

        meta["error"] = str(exc)

    return response.content, meta


def build_9tiles() -> dict:

    if Image is None:
        raise RuntimeError(
            "Pillow no esta disponible."
        )

    started = now_utc()

    _, frame = rainviewer_data()

    center_x, center_y = latlon_to_tile(
        LAT,
        LON,
        RADAR_ZOOM
    )

    min_x = center_x - GRID_RADIUS
    max_x = center_x + GRID_RADIUS

    min_y = center_y - GRID_RADIUS
    max_y = center_y + GRID_RADIUS

    composite = Image.new(
        "RGBA",
        (
            TILE_SIZE * 3,
            TILE_SIZE * 3
        ),
        (0, 0, 0, 0)
    )

    preview = Image.new(
        "RGBA",
        (
            TILE_SIZE * 3,
            TILE_SIZE * 3
        ),
        (20, 24, 30, 255)
    )

    tiles = []

    ok = 0

    for y in range(
        min_y,
        max_y + 1
    ):

        for x in range(
            min_x,
            max_x + 1
        ):

            url = tile_url(
                frame["host"],
                frame["path"],
                x,
                y
            )

            try:

                content, meta = (
                    download_tile(url)
                )

            except Exception as exc:

                content = b""

                meta = {
                    "status_code": None,
                    "content_type": None,
                    "bytes": 0,
                    "imagen_valida": False,
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }

            item = {
                "x": x,
                "y": y,
                "url": url,
                **meta,
            }

            tiles.append(item)

            if meta.get(
                "imagen_valida"
            ):

                try:

                    with Image.open(
                        BytesIO(content)
                    ).convert("RGBA") as image:

                        position = (
                            (x - min_x)
                            * TILE_SIZE,
                            (y - min_y)
                            * TILE_SIZE
                        )

                        composite.alpha_composite(
                            image,
                            position
                        )

                        preview.alpha_composite(
                            image,
                            position
                        )

                    ok += 1

                except Exception as exc:

                    item[
                        "imagen_valida"
                    ] = False

                    item[
                        "error_composicion"
                    ] = str(exc)

    composite.save(
        RADAR_IMAGE,
        format="PNG",
        optimize=True
    )

    preview.save(
        RADAR_PREVIEW,
        format="PNG",
        optimize=True
    )

    bounds = {
        "north": tile_y_to_lat(
            min_y,
            RADAR_ZOOM
        ),
        "south": tile_y_to_lat(
            max_y + 1,
            RADAR_ZOOM
        ),
        "west": tile_x_to_lon(
            min_x,
            RADAR_ZOOM
        ),
        "east": tile_x_to_lon(
            max_x + 1,
            RADAR_ZOOM
        ),
    }

    status = {
        "estado": (
            "ok"
            if ok == 9
            else "parcial"
            if ok > 0
            else "fallo"
        ),
        "fuente": "RainViewer",
        "frame_utc": frame[
            "timestamp_utc"
        ],
        "frame_argentina": frame[
            "timestamp_argentina"
        ],
        "path": frame["path"],
        "host": frame["host"],
        "frames_disponibles": frame[
            "frames_disponibles"
        ],
        "teselas_ok": ok,
        "teselas_total": 9,
        "zoom": RADAR_ZOOM,
        "tile_size": TILE_SIZE,
        "centro": {
            "lat": LAT,
            "lon": LON,
            "x": center_x,
            "y": center_y,
        },
        "bounds": bounds,
        "inicio_utc": started,
        "fin_utc": now_utc(),
        "tiles": tiles,
    }

    RADAR_STATUS.write_text(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    return status


def leer_observacion_sazb() -> dict:

    if not OBS_STATUS.exists():

        return {
            "integrado": False,
            "observacion_valida": False,
            "fuente": (
                "Aviation Weather Center"
            ),
            "estacion": "SAZB",
            "error": (
                "Sin status SAZB"
            ),
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
            "fuente": (
                "Aviation Weather Center"
            ),
            "estacion": "SAZB",
            "error": str(exc),
        }


def estado_guardado() -> dict:

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
            "error": str(exc),
            "observacion_sazb":
                leer_observacion_sazb(),
        }


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

    output = {
        "disponible": True,
        "archivo": str(model_path),
    }

    try:

        model = joblib.load(
            model_path
        )

        if isinstance(
            model,
            dict
        ):

            output["tipo"] = (
                model.get("tipo")
            )

            output["version"] = (
                model.get("version")
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

            output["resumen"] = json.loads(
                summary_path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:
            pass

    return output


@app.get("/")
def inicio():

    return {
        "servicio": "ClimaAR",
        "estado": "ok",
        "version": app.version,
        "ubicacion": "Bahia Blanca",
        "fuente_radar": "RainViewer",
        "radar": "/radar",
        "diagnostico_9_teselas":
            "/radar/9tiles",
    }


@app.get("/health")
def health():

    return {
        "estado": "ok",
        "servicio": "ClimaAR",
        "version": app.version,
        "hora_utc": now_utc(),
    }


@app.get("/radar/9tiles")
def radar_9tiles():

    try:

        return build_9tiles()

    except Exception as exc:

        return JSONResponse(
            status_code=502,
            content={
                "estado": "fallo",
                "error": (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            }
        )


@app.get("/radar/status")
def radar_status():

    return estado_guardado()


@app.get("/radar/debug")
def radar_debug():

    return {
        "mensaje": (
            "El diagnostico anterior de "
            "una tesela fue superado por "
            "la prueba 3x3."
        ),
        "prueba": "/radar/9tiles",
        "version": app.version,
    }


@app.get("/radar.png")
def radar_png():

    if not RADAR_IMAGE.exists():
        build_9tiles()

    return FileResponse(
        RADAR_IMAGE,
        media_type="image/png",
        headers={
            "Cache-Control":
                "no-store, max-age=0"
        }
    )


@app.get("/radar/preview.png")
def radar_preview_png():

    if not RADAR_PREVIEW.exists():
        build_9tiles()

    return FileResponse(
        RADAR_PREVIEW,
        media_type="image/png",
        headers={
            "Cache-Control":
                "no-store, max-age=0"
        }
    )


@app.get("/radar")
def radar():

    status = estado_guardado()

    return HTMLResponse(
        f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport"
content="width=device-width,initial-scale=1">

<title>ClimaAR Radar</title>

<style>
body {{
    font-family: Arial;
    margin: 0;
    background: #111;
    color: #eee;
}}

.box {{
    padding: 14px;
}}

button {{
    padding: 12px 16px;
    font-size: 16px;
}}

img {{
    max-width: 100%;
    display: block;
    margin-top: 12px;
}}

pre {{
    white-space: pre-wrap;
    font-size: 12px;
}}
</style>

</head>

<body>

<div class="box">

<h2>
ClimaAR — Radar RainViewer
</h2>

<p>
Prueba 3×3 de teselas alrededor
de Bahía Blanca.
</p>

<button
onclick="location.href='/radar/9tiles'">

Actualizar y probar 9 teselas

</button>

<p>

<a
href="/radar/preview.png"
style="color:#8cf">

Ver composición sobre fondo oscuro

</a>

</p>

<img
src="/radar.png?t={datetime.now().timestamp()}"
alt="Radar RainViewer">

<pre>
{json.dumps(
    status,
    ensure_ascii=False,
    indent=2
)}
</pre>

</div>

</body>
</html>
"""
    )


@app.get("/observacion")
def observacion():

    return leer_observacion_sazb()


@app.get("/estado")
def estado():

    return {
        "radar": estado_guardado(),
        "observacion":
            leer_observacion_sazb(),
        "modelo": modelo_info(),
    }


@app.get("/modelo")
def modelo():

    return modelo_info()


@app.get("/tormenta")
def tormenta():

    radar_state = estado_guardado()

    return {
        "estado":
            radar_state.get("estado"),
        "fuente_radar":
            "RainViewer",
        "frame_utc":
            radar_state.get(
                "frame_utc"
            ),
        "frame_argentina":
            radar_state.get(
                "frame_argentina"
            ),
        "teselas_ok":
            radar_state.get(
                "teselas_ok"
            ),
        "teselas_total":
            radar_state.get(
                "teselas_total"
            ),
        "observacion":
            leer_observacion_sazb(),
    }


@app.get("/nowcast")
def nowcast():

    return {
        "estado":
            "base_operativa",
        "mensaje": (
            "El motor de nowcasting IA "
            "se integra despues de validar "
            "el radar 3x3."
        ),
        "radar":
            estado_guardado(),
    }
