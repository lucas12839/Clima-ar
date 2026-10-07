from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import math
import time

VERSION = "5.2.0"

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
INTELLIGENCE = BASE / "data" / "ia" / "inteligencia_tormenta.json"

LAT = -38.71
LON = -62.26
ZOOM = 7
TILE = 512

app = FastAPI(
    title="ClimaAR",
    version=VERSION,
    description="Radar meteorológico y seguimiento de tormentas para Bahía Blanca",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def argentina_time(dt):
    try:
        return dt.astimezone(ZoneInfo("America/Argentina/Buenos_Aires"))
    except Exception:
        return dt


def clean(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def load_json(path):
    try:
        if not path.exists():
            return None
        return clean(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def save_json(path, data):
    path.write_text(
        json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def lon_to_world_x(lon):
    return ((lon + 180.0) / 360.0) * (2 ** ZOOM)


def lat_to_world_y(lat):
    return (
        (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi)
        / 2.0
        * (2 ** ZOOM)
    )


def x_to_lon(x):
    return x / (2 ** ZOOM) * 360.0 - 180.0


def y_to_lat(y):
    return math.degrees(
        math.atan(
            math.sinh(
                math.pi * (1.0 - 2.0 * y / (2 ** ZOOM))
            )
        )
    )


def radar_bounds():
    center_x = math.floor(lon_to_world_x(LON))
    center_y = math.floor(lat_to_world_y(LAT))
    return {
        "north": y_to_lat(center_y - 1),
        "south": y_to_lat(center_y + 2),
        "west": x_to_lon(center_x - 1),
        "east": x_to_lon(center_x + 2),
    }


def load_radar_nowcast():
    data = load_json(NOWCAST)
    return data if isinstance(data, dict) else None


def load_old_status():
    data = load_json(STATUS)
    return data if isinstance(data, dict) else None


def build_status_from_nowcast():
    """
    La interfaz no vuelve a descargar 9 teselas al abrir la página.
    El workflow de ClimaAR ya genera actual.png + radar_nowcast.json.
    """
    n = load_radar_nowcast() or {}
    old = load_old_status() or {}

    # El archivo generado por radar_rainviewer.py tiene los datos
    # dentro de la clave nowcast.
    data = n.get("nowcast", n)
    if not isinstance(data, dict):
        data = {}

    frame_ts = data.get("frame_actual")
    frame_utc = data.get("frame_actual_utc")

    if frame_ts is not None and not frame_utc:
        try:
            dt = datetime.fromtimestamp(int(frame_ts), tz=timezone.utc)
            frame_utc = dt.isoformat()
        except Exception:
            frame_utc = None

    frame_ar = None
    if frame_utc:
        try:
            dt = datetime.fromisoformat(str(frame_utc).replace("Z", "+00:00"))
            frame_ar = argentina_time(dt).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            frame_ar = str(frame_utc)

    area_px = data.get("area_px")
    try:
        coverage = float(area_px or 0) / float((TILE * 3) ** 2) * 100.0
    except Exception:
        coverage = 0.0

    if data.get("actividad") is True:
        intensity = data.get("intensidad") or "debil"
    else:
        intensity = "sin_precipitacion"

    status = {
        "estado": "ok" if ACTUAL.exists() else "sin_imagen",
        "fuente": "RainViewer",
        "version": VERSION,
        "esquema_color": 2,
        "esquema_color_nombre": "Universal Blue",
        "actualizado_utc": now(),
        "frame_utc": frame_utc,
        "frame_argentina": frame_ar,
        "frames_disponibles": data.get("frames_disponibles", 12),
        "teselas_ok": data.get("tiles_ok", 9),
        "teselas_total": 9,
        "zoom": ZOOM,
        "centro": {"lat": LAT, "lon": LON},
        "bounds": radar_bounds(),
        "duracion_segundos": old.get("duracion_segundos"),
        "analisis": {
            "precipitacion": bool(data.get("actividad", False)),
            "cobertura_pct": round(coverage, 3),
            "intensidad": intensity,
            "pixeles_radar": data.get("dbz_pixels", 0),
            "dbz_max": data.get("dbz_max"),
            "dbz_mean": data.get("dbz_mean"),
            "dbz_p90": data.get("dbz_p90"),
            "distancia_km": data.get("distancia_km"),
            "movimiento_hacia_bahia": data.get("movimiento_hacia_bahia", False),
        },
    }
    return status


@app.get("/", response_class=HTMLResponse)
def home():
    return radar_page()


@app.get("/health")
def health():
    return {
        "estado": "ok",
        "servicio": "ClimaAR",
        "version": VERSION,
        "hora_utc": now(),
    }


@app.get("/radar/9tiles")
def radar_9tiles():
    # Importante: ya NO bloqueamos el navegador descargando 9 teselas.
    # El workflow es quien actualiza los archivos del radar.
    return build_status_from_nowcast()


@app.get("/radar/status")
def radar_status():
    return build_status_from_nowcast()


@app.get("/radar/analysis")
def radar_analysis_endpoint():
    status = build_status_from_nowcast()
    return {
        "estado": "ok",
        "fuente": "RainViewer",
        "analisis": status.get("analisis", {}),
        "hora_utc": now(),
    }


@app.get("/radar/debug")
def radar_debug():
    status = build_status_from_nowcast()
    return {
        "estado": status.get("estado"),
        "version": VERSION,
        "zoom": ZOOM,
        "centro": {"lat": LAT, "lon": LON},
        "bounds": status.get("bounds"),
        "imagen_actual": ACTUAL.exists(),
        "imagen_bytes": ACTUAL.stat().st_size if ACTUAL.exists() else 0,
        "nowcast_disponible": load_radar_nowcast() is not None,
        "analisis": status.get("analisis"),
    }


@app.get("/radar.png")
def radar_png():
    if not ACTUAL.exists():
        return JSONResponse(
            {"estado": "sin_imagen", "mensaje": "El workflow todavía no generó actual.png."},
            status_code=503,
        )
    return FileResponse(
        ACTUAL,
        media_type="image/png",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/radar/preview.png")
def radar_preview():
    return radar_png()


@app.get("/nowcast")
def nowcast():
    data = load_radar_nowcast()
    return {
        "disponible": data is not None,
        "estado": "ok" if data is not None else "sin_datos",
        "zona": "Bahía Blanca",
        "fuente": "RainViewer",
        "nowcast": data,
    }


@app.get("/observacion")
def observacion():
    return (
        load_json(SAZB_STATUS)
        or {
            "estado": "sin_datos",
            "fuente": "Aviation Weather Center",
            "estacion": "SAZB",
        }
    )


@app.get("/intelligence")
def intelligence():
    data = load_json(INTELLIGENCE)
    return data or {
        "estado": "sin_datos",
        "fuente": "ClimaAR Intelligence",
    }


@app.get("/tormenta")
def tormenta():
    data = load_radar_nowcast() or {}
    n = data.get("nowcast", data)
    if not isinstance(n, dict):
        n = {}
    return {
        "estado": "ok" if data else "sin_datos",
        "zona": "Bahía Blanca",
        "actividad": n.get("actividad", False),
        "nucleos": n.get("nucleos", []),
        "nucleo_principal_id": n.get("nucleo_principal_id"),
        "proyecciones": n.get("proyecciones", {}),
        "confianza": n.get("confianza_movimiento", 0),
    }


@app.get("/estado")
def estado():
    radar = build_status_from_nowcast()
    data = load_radar_nowcast()
    return clean({
        "app": "ClimaAR",
        "version": VERSION,
        "radar": radar,
        "observacion_sazb": load_json(SAZB_STATUS),
        "intelligence": load_json(INTELLIGENCE),
        "nowcast_radar": {
            "disponible": data is not None,
            "estado": "ok" if data is not None else "sin_datos",
            "datos": data,
        },
        "nowcast_radar_operativo": data is not None,
        "nowcast_radar_ia": False,
        "hora_utc": now(),
    })


@app.get("/modelo")
def modelo():
    return {
        "estado": "experimental",
        "mensaje": "Módulo de análisis meteorológico de ClimaAR.",
    }


def radar_page():
    return """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>ClimaAR — Radar Bahía Blanca</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
*{box-sizing:border-box}
html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#08111b;color:#fff;font-family:Arial,sans-serif}
#map{position:fixed;inset:0;z-index:1}
.top{position:fixed;z-index:1000;top:0;left:0;right:0;min-height:58px;padding:8px 10px;display:flex;align-items:center;gap:10px;background:rgba(5,12,20,.92);backdrop-filter:blur(8px);box-shadow:0 2px 12px rgba(0,0,0,.35)}
.brand{font-weight:800;font-size:15px}
.sub{font-size:10px;color:#aebdcc;margin-top:3px}
.actions{margin-left:auto;display:flex;gap:6px}
button{border:0;border-radius:9px;padding:9px 11px;font-weight:700;cursor:pointer}
#refresh{background:#1677ff;color:#fff}
#toggle{background:#293847;color:#fff}
.sheet{position:fixed;z-index:1100;left:9px;right:9px;bottom:9px;max-height:43vh;overflow:auto;background:rgba(8,13,19,.95);border:1px solid rgba(255,255,255,.14);border-radius:16px;box-shadow:0 10px 35px rgba(0,0,0,.5);backdrop-filter:blur(12px);transition:transform .22s ease,opacity .22s ease}
.sheet.closed{transform:translateY(calc(100% + 20px));opacity:.1;pointer-events:none}
.head{position:sticky;top:0;z-index:2;display:flex;justify-content:space-between;align-items:center;padding:9px 12px;background:rgba(8,13,19,.99);border-bottom:1px solid rgba(255,255,255,.08)}
.title{font-weight:800;font-size:14px}
.close{width:34px;height:34px;padding:0;border-radius:50%;background:#3a4857;color:#fff;font-size:21px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;padding:9px 11px}
.card{background:rgba(255,255,255,.055);border-radius:10px;padding:8px;min-height:50px}
.label{font-size:9px;color:#91a0ae;text-transform:uppercase}
.value{font-size:13px;font-weight:800;margin-top:4px}
.status{padding:0 11px 10px;font-size:12px}
.good{color:#72e3a0}.warn{color:#ffd166}.bad{color:#ff7777}
#open{display:none;position:fixed;z-index:1050;right:10px;bottom:10px;background:#111d29;color:#fff;border:1px solid #44515f}
@media(max-width:700px){
.brand{font-size:14px}.sub{display:none}.actions button{padding:8px 9px;font-size:11px}
.grid{grid-template-columns:repeat(2,1fr)}
}
</style>
</head>
<body>
<div id="map"></div>
<header class="top">
<div>
<div class="brand">ClimaAR — Radar Bahía Blanca</div>
<div class="sub" id="frame">Cargando radar…</div>
</div>
<div class="actions">
<button id="refresh">Actualizar radar</button>
<button id="toggle">Ocultar datos</button>
</div>
</header>

<section class="sheet" id="sheet">
<div class="head">
<div class="title" id="title">Estado meteorológico</div>
<button class="close" id="close">×</button>
</div>
<div class="status" id="main">Cargando…</div>
<div class="grid">
<div class="card"><div class="label">Temperatura</div><div class="value" id="temp">—</div></div>
<div class="card"><div class="label">Punto de rocío</div><div class="value" id="dew">—</div></div>
<div class="card"><div class="label">Viento</div><div class="value" id="wind">—</div></div>
<div class="card"><div class="label">Presión</div><div class="value" id="press">—</div></div>
<div class="card"><div class="label">Cobertura radar</div><div class="value" id="cov">—</div></div>
<div class="card"><div class="label">Intensidad</div><div class="value" id="int">—</div></div>
<div class="card"><div class="label">Teselas</div><div class="value" id="tiles">—</div></div>
<div class="card"><div class="label">Actualizado</div><div class="value" id="upd">—</div></div>
</div>
<div class="status">
<b>Nowcast:</b> <span id="nc">—</span><br>
<b>Movimiento:</b> <span id="motion">—</span>
</div>
</section>

<button id="open">Mostrar datos</button>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const map=L.map('map',{zoomControl:true,preferCanvas:true}).setView([-38.71,-62.26],8);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:18,attribution:'© OpenStreetMap'}).addTo(map);

let layer=null;
const $=id=>document.getElementById(id);
const text=value=>value==null?'—':String(value);

function overlay(status){
    if(layer){map.removeLayer(layer);layer=null}
    if(!status.bounds){return}
    layer=L.imageOverlay('/radar.png?ts='+Date.now(),[
        [status.bounds.south,status.bounds.west],
        [status.bounds.north,status.bounds.east]
    ],{opacity:.84,interactive:false});
    layer.addTo(map);
}

function renderRadar(status){
    overlay(status);
    const a=status.analisis||{};
    const rain=!!a.precipitacion;

    $('frame').textContent='RainViewer · frame: '+text(status.frame_argentina||status.frame_utc)+' · teselas: '+text(status.teselas_ok)+'/9';
    $('tiles').textContent=text(status.teselas_ok)+'/9';
    $('cov').textContent=text(a.cobertura_pct)+'%';

    const labels={
        sin_precipitacion:'Sin precipitación',
        debil:'Débil',
        moderada:'Moderada',
        fuerte:'Fuerte'
    };
    $('int').textContent=labels[a.intensidad]||'—';

    $('upd').textContent=status.actualizado_utc
        ? new Date(status.actualizado_utc).toLocaleTimeString('es-AR')
        : '—';

    $('title').textContent=rain?'Precipitación detectada':'Estado meteorológico';
    $('main').innerHTML=rain
        ? '<span class="warn">● Precipitación detectada en el radar.</span>'
        : '<span class="good">● Sin precipitación detectada en el radar.</span>';
}

function renderObservation(data){
    const d=data&&(data.observacion||data)||{};
    const temperature=d.temperatura_c??d.temperature_c??d.temp_c??d.temperatura;
    const dew=d.punto_rocio_c??d.dewpoint_c??d.dew_point_c??d.dewpoint;
    const wind=d.viento_kmh??d.wind_kmh??d.wind_speed_kmh??d.viento;
    const pressure=d.presion_hpa??d.pressure_hpa??d.presion;

    $('temp').textContent=temperature!=null?Number(temperature).toFixed(1)+' °C':'—';
    $('dew').textContent=dew!=null?Number(dew).toFixed(1)+' °C':'—';
    $('wind').textContent=wind!=null?Number(wind).toFixed(1)+' km/h':'—';
    $('press').textContent=pressure!=null?Number(pressure).toFixed(1)+' hPa':'—';
}

function renderNowcast(data){
    const value=data&&data.nowcast;
    if(!value){
        $('nc').textContent='sin datos';
        $('motion').textContent='sin datos';
        return;
    }

    const n=value.nowcast||value;
    $('nc').textContent=n.actividad===true?'actividad detectada':n.actividad===false?'sin actividad':'datos disponibles';

    const confidence=n.confianza_movimiento;
    $('motion').textContent=confidence!=null
        ? 'confianza '+Math.round(Number(confidence)*100)+'%'
        : 'sin datos';
}

async function loadRadar(){
    $('refresh').disabled=true;
    $('main').textContent='Cargando datos del último radar…';

    try{
        const ts=Date.now();

        const responses=await Promise.all([
            fetch('/radar/status?ts='+ts,{cache:'no-store'}),
            fetch('/observacion?ts='+ts,{cache:'no-store'}),
            fetch('/nowcast?ts='+ts,{cache:'no-store'})
        ]);

        for(const r of responses){
            if(!r.ok) throw new Error('HTTP '+r.status);
        }

        const status=await responses[0].json();
        const observation=await responses[1].json();
        const nowcast=await responses[2].json();

        if(status.estado==='error') throw new Error(status.error||'Error de radar');

        renderRadar(status);
        renderObservation(observation);
        renderNowcast(nowcast);
    }catch(error){
        $('main').innerHTML='<span class="bad">No se pudieron cargar los datos: '+text(error.message)+'</span>';
    }finally{
        $('refresh').disabled=false;
    }
}

function closeSheet(){
    $('sheet').classList.add('closed');
    $('open').style.display='block';
    setTimeout(()=>map.invalidateSize(),250);
}

function openSheet(){
    $('sheet').classList.remove('closed');
    $('open').style.display='none';
    setTimeout(()=>map.invalidateSize(),250);
}

$('refresh').onclick=loadRadar;
$('close').onclick=closeSheet;
$('open').onclick=openSheet;
$('toggle').onclick=()=>$('sheet').classList.contains('closed')?openSheet():closeSheet();

loadRadar();
setInterval(loadRadar,60000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
