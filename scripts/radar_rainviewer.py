"""
ClimaAR - radar_rainviewer.py FIXED
Módulo que reemplaza el overwrite del JSON y maneja correctamente
RainViewer sin futuro (desde enero 2026 solo pasado)
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LAT = -38.0055
LON = -62.0
MAX_HISTORY_FRAMES = 144  # 24hs de historia cada 10 min
RAINVIEWER_API = "https://api.rainviewer.com/public/weather-maps.json"

def analyze_sequence(results):
    """
    Analiza secuencia de frames para movimiento y nowcast propio
    Ya que RainViewer eliminó futuro en enero 2026, hacemos nuestro motor
    """
    if not results:
        return {
            "estado": "sin_datos",
            "confianza": 0,
            "movimiento": {"direccion": None, "velocidad_kmh": 0, "confianza": 0},
            "nowcast_disponible": False
        }
    
    latest = results[-1]
    
    # Si no hay precipitación, no hay movimiento que calcular
    if latest.get("sin_precipitacion") or latest.get("area", 0) == 0:
        return {
            "estado": "sin_precipitacion",
            "intensidad": "Sin precipitación",
            "confianza": 100,  # FIX: antes devolvía 0%, ahora 100% cuando no llueve es normal
            "movimiento": {
                "direccion": None,
                "velocidad_kmh": 0,
                "confianza": 0,
                "nota": "Sin celdas para trackear"
            },
            "nowcast_disponible": False,
            "cobertura_pct": latest.get("cobertura_pct", 0),
            "dbz_max": latest.get("dbz_max", 0),
            "area": latest.get("area", 0)
        }
    
    # Si hay precipitación, calcular movimiento entre últimos 2 frames
    if len(results) < 2:
        return {
            "estado": "precipitacion_detectada",
            "intensidad": f"{latest.get('dbz_max', 0)} dBZ",
            "confianza": 75,
            "movimiento": {"direccion": None, "velocidad_kmh": 0, "confianza": 50, "nota": "Solo 1 frame, sin historial para movimiento"},
            "nowcast_disponible": False,
            "cobertura_pct": latest.get("cobertura_pct", 0),
            "dbz_max": latest.get("dbz_max"),
            "area": latest.get("area")
        }
    
    # Cálculo simple de movimiento por centroides
    prev = results[-2]
    curr_components = latest.get("components", [])
    prev_components = prev.get("components", [])
    
    if not curr_components or not prev_components:
        # Sin componentes pero hay precipitación dispersa
        return {
            "estado": "precipitacion_dispersa",
            "intensidad": f"{latest.get('dbz_max', 0)} dBZ",
            "confianza": 70,
            "movimiento": {"direccion": None, "velocidad_kmh": 0, "confianza": 40},
            "nowcast_disponible": True,  # podemos hacer extrapolación simple
            "cobertura_pct": latest.get("cobertura_pct", 0),
            "dbz_max": latest.get("dbz_max"),
            "area": latest.get("area")
        }
    
    # Calcular desplazamiento del centroide principal
    curr_main = curr_components[0]
    prev_main = prev_components[0]
    
    dx = curr_main["x"] - prev_main["x"]
    dy = curr_main["y"] - prev_main["y"]
    
    # Convertir pixels a km aprox (muy aproximado para RMA10, ~1km por pixel)
    distance_pixels = math.sqrt(dx*dx + dy*dy)
    distance_km = distance_pixels * 1.0  # ajustar según resolución real
    
    # Tiempo entre frames (10 min = 600 seg)
    time_diff = latest["timestamp"] - prev["timestamp"]
    if time_diff == 0:
        time_diff = 600
    
    speed_kmh = (distance_km / time_diff) * 3600
    
    # Dirección en grados (0=N, 90=E)
    direction = (math.degrees(math.atan2(dx, -dy)) + 360) % 360
    direction_text = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"][int((direction + 22.5) // 45) % 8]
    
    return {
        "estado": "tormenta_activa" if latest.get("dbz_max", 0) > 40 else "precipitacion_activa",
        "intensidad": f"{latest.get('dbz_max', 0)} dBZ",
        "confianza": 85,
        "movimiento": {
            "direccion_grados": round(direction, 1),
            "direccion_texto": direction_text,
            "velocidad_kmh": round(speed_kmh, 1),
            "distancia_km": round(distance_km, 1),
            "confianza": 75,
            "dx_pixels": int(dx),
            "dy_pixels": int(dy)
        },
        "nowcast_disponible": True,
        "nowcast_metodo": "extrapolacion_centroide + optical flow futuro",
        "cobertura_pct": latest.get("cobertura_pct", 0),
        "dbz_max": latest.get("dbz_max"),
        "dbz_mean": latest.get("dbz_mean"),
        "area": latest.get("area"),
        "componentes": len(curr_components)
    }


def get_rainviewer_frames():
    """
    Obtiene frames frescos de RainViewer sin cache
    Uso: desde app.py para no depender solo de historico/
    """
    import requests
    try:
        r = requests.get(RAINVIEWER_API, timeout=10, headers={"Cache-Control": "no-cache"})
        r.raise_for_status()
        data = r.json()
        past = data.get("radar", {}).get("past", [])
        # Últimos 4 frames para secuencia
        return past[-4:] if len(past) >= 4 else past
    except Exception as e:
        print(f"Error obteniendo RainViewer: {e}")
        return []


# Para compatibilidad con código existente
def analyze_radar_image(image_path):
    """Wrapper para compatibilidad"""
    from PIL import Image
    from radar_repair_decoder_fixed import analyze_frame
    with Image.open(image_path) as img:
        return analyze_frame(img.convert("RGBA"), timestamp=Path(image_path).stat().st_mtime)
