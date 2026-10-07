"""
ClimaAR - app.py COMPLETO V7.7 FIXED
Reemplazo total para tu app.py viejo
Fuente: RainViewer -> RMA10 Bahia Blanca (sin token SMN)
Fix: cobertura 0% falso, viento "-", frame 14:50 UTC clavado

Endpoints:
- /health
- /radar
- /radar/imagen
- /secuencia
- /tormenta
- /nowcast
- /modelo
- /modelo/nowcast
- /prediccion
- /estado
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from datetime import datetime, timezone
import json
import requests

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parent
RADAR_DIR = BASE_DIR / "data" / "radar"
HISTORY_DIR = RADAR_DIR / "historico"
RAW_DIR = RADAR_DIR / "raw"
NOWCAST_FILE = RADAR_DIR / "radar_nowcast.json"
LATEST_IMAGE = RADAR_DIR / "latest.png"

LAT = -38.0055
LON = -62.0
CITY = "Bahia Blanca"
VERSION = "7.7-fixed"
FUENTE = "RainViewer -> RMA10 Bahia Blanca"

app = FastAPI(
    title="ClimaAR - Radar Bahía Blanca",
    description="Radar RMA10 via RainViewer sin token SMN - V7.7 FIXED",
    version=VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HELPERS ---

def get_nowcast_data():
    if not NOWCAST_FILE.exists():
        return None
    try:
        return json.loads(NOWCAST_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error leyendo nowcast: {e}")
        return None

def get_openmeteo_data():
    """Datos reales para no mostrar '-' en viento"""
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={LAT}&longitude={LON}"
            f"&current=temperature_2m,dewpoint_2m,pressure_msl,wind_speed_10m,wind_direction_10m,precipitation"
            f"&timezone=America/Argentina/Buenos_Aires"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        curr = r.json().get("current", {})
        return {
            "temperatura": curr.get("temperature_2m"),
            "punto_rocio": curr.get("dewpoint_2m"),
            "presion": curr.get("pressure_msl"),
            "viento_vel": curr.get("wind_speed_10m"),
            "viento_dir": curr.get("wind_direction_10m"),
            "precipitacion": curr.get("precipitation"),
        }
    except Exception as e:
        print(f"Open-Meteo error: {e}")
        return None

def get_rainviewer_frames():
    """Frames frescos de RainViewer sin cache - FIX frame viejo 14:50 UTC"""
    try:
        r = requests.get(
            "https://api.rainviewer.com/public/weather-maps.json",
            timeout=10,
            headers={"Cache-Control": "no-cache", "Pragma": "no-cache"}
        )
        r.raise_for_status()
        data = r.json()
        past = data.get("radar", {}).get("past", [])
        return past
    except Exception as e:
        print(f"RainViewer error: {e}")
        return []

def load_history_list():
    if not HISTORY_DIR.exists():
        return []
    files = []
    for p in HISTORY_DIR.glob("radar_*.png"):
        try:
            ts = int(p.stem.split("_")[-1])
            files.append((ts, p))
        except:
            continue
    files.sort(key=lambda x: x[0])
    return files[-13:]

# --- ENDPOINTS ---

@app.get("/health")
def health():
    nowcast = get_nowcast_data()
    edad = nowcast.get("frame_edad_minutos", 0) if nowcast else 999
    return {
        "status": "ok",
        "app": "ClimaAR",
        "version": VERSION,
        "fuente": FUENTE,
        "ciudad": CITY,
        "lat": LAT,
        "lon": LON,
        "frame_edad_minutos": edad,
        "frame_es_viejo": edad > 35,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/estado")
def estado():
    nowcast = get_nowcast_data()
    meteo = get_openmeteo_data()

    if not nowcast:
        return {
            "temperatura": f"{meteo['temperatura']} °C" if meteo and meteo["temperatura"] is not None else "—",
            "punto_rocio": f"{meteo['punto_rocio']} °C" if meteo and meteo["punto_rocio"] is not None else "—",
            "viento": f"{meteo['viento_vel']} km/h" if meteo and meteo["viento_vel"] is not None else "Calma",
            "presion": f"{meteo['presion']} hPa" if meteo and meteo["presion"] is not None else "—",
            "cobertura_radar": "Iniciando...",
            "intensidad": "Cargando...",
            "teselas": "0/0",
            "actualizado": datetime.now(timezone.utc).isoformat(),
            "nowcast": "Iniciando",
            "movimiento": "Sin datos",
            "fuente": FUENTE,
            "version": VERSION
        }

    diag = nowcast.get("diagnostico", {})
    nc = nowcast.get("nowcast", {})
    historial = nowcast.get("historial", {})

    if diag.get("sin_precipitacion"):
        cobertura = "0% - Sin precipitación (normal)"
        intensidad = "Sin precipitación"
        movimiento_str = "N/A - Sin celdas para trackear"
    else:
        cobertura = f"{diag.get('cobertura_pct', 0)}%"
        intensidad = f"{nc.get('dbz_max', 0)} dBZ - {nc.get('estado', 'precipitación')}"
        mov = nc.get("movimiento", {})
        if mov.get("velocidad_kmh"):
            movimiento_str = f"{mov.get('direccion_texto','?')} a {mov.get('velocidad_kmh')} km/h (conf {mov.get('confianza',0)}%)"
        else:
            movimiento_str = f"Confianza {mov.get('confianza',0)}%"

    if meteo and meteo["viento_vel"] is not None:
        viento_str = f"{meteo['viento_vel']} km/h del {meteo['viento_dir']}°"
    else:
        viento_str = "Calma o sin datos SAZB"

    teselas_ok = historial.get("frames_procesados", 0)
    edad = nowcast.get("frame_edad_minutos", 0)

    if edad > 35:
        actualizado_str = f"{nowcast.get('frame_actual_utc','')} - ⚠️ Frame viejo hace {edad:.0f} min"
    else:
        actualizado_str = nowcast.get("frame_actual_utc", datetime.now(timezone.utc).isoformat())

    return {
        "temperatura": f"{meteo['temperatura']} °C" if meteo and meteo["temperatura"] is not None else "—",
        "punto_rocio": f"{meteo['punto_rocio']} °C" if meteo and meteo["punto_rocio"] is not None else "—",
        "viento": viento_str,
        "presion": f"{meteo['presion']} hPa" if meteo and meteo["presion"] is not None else "—",
        "cobertura_radar": cobertura,
        "intensidad": intensidad,
        "teselas": f"{teselas_ok}/{teselas_ok}",
        "actualizado": actualizado_str,
        "nowcast": nc.get("estado", "sin datos"),
        "movimiento": movimiento_str,
        "fuente": nowcast.get("fuente", FUENTE),
        "version": VERSION,
        "frame_edad_minutos": edad,
        "frame_es_viejo": edad > 35
    }

@app.get("/radar")
def radar():
    frames = get_rainviewer_frames()
    if not frames:
