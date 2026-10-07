"""
ClimaAR - Radar Repair Decoder V7.7 FIXED
Fix para cobertura 0% y frame clavado en 14:50 UTC
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

try:
    from scripts import radar_rainviewer as rv
except ImportError:
    import radar_rainviewer as rv

BASE = Path(__file__).resolve().parents[1]
RADAR_DIR = BASE / "data" / "radar"
HISTORY_DIR = RADAR_DIR / "historico"
NOWCAST_FILE = RADAR_DIR / "radar_nowcast.json"

MAX_FRAMES = 13
MIN_DBZ = 10.0
PALETTE_MATCH_DISTANCE = 45.0
REPAIR_VERSION = "7.7-fixed"
MAX_AGE_MINUTES = 35

DBZ_HEX = [
    "cec08796", "d2c48ba0", "d6c88faa", "dacc93b4", "ded097be",
    "88ddeeff", "6cd1ebff", "51c5e8ff", "36bae5ff", "1baee2ff",
    "00a3e0ff", "009ad5ff", "0091caff", "0088bfff", "007fb4ff",
    "0077aaff", "0070a3ff", "00699cff", "006295ff", "005b8eff",
    "005588ff", "005180ff", "004e78ff", "004a70ff", "004768ff",
    "ffee00ff", "ffe000ff", "ffd200ff", "ffc500ff", "ffb700ff",
    "ffaa00ff", "ff9f00ff", "ff9500ff", "ff8b00ff", "ff8100ff",
    "ff4400ff", "f23600ff", "e62800ff", "d91b00ff", "cd0d00ff",
    "c10000ff", "a80000ff", "8f0000ff", "760000ff", "5d0000ff",
    "ffaaffff", "ff9fffff", "ff95ffff", "ff8bffff", "ff81ffff",
    "ff77ffff", "ff6cffff", "ff62ffff", "ff58ffff", "ff4effff",
]

DBZ_VALUES = np.linspace(10, 65, len(DBZ_HEX), dtype=np.float32)
DBZ_RGB = np.array(
    [[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in DBZ_HEX],
    dtype=np.float32,
)

def rgba_to_dbz(image):
    arr = np.asarray(image).astype(np.float32)
    rgb = arr[:, :, :3]
    flat = rgb.reshape(-1, 3)
    distances = ((flat[:, None, :] - DBZ_RGB[None, :, :]) ** 2).sum(axis=2)
    indices = np.argmin(distances, axis=1)
    nearest = np.sqrt(np.min(distances, axis=1)).reshape(rgb.shape[:2])
    dbz = DBZ_VALUES[indices].reshape(rgb.shape[:2]).astype(np.float32)
    valid = (nearest <= PALETTE_MATCH_DISTANCE) & np.any(rgb > 5, axis=2)
    dbz[~valid] = np.nan
    return dbz

def metrics(dbz):
    valid = dbz[np.isfinite(dbz)]
    precip = valid[valid >= MIN_DBZ]
    if precip.size == 0:
        return {"dbz_max": 0.0, "dbz_mean": 0.0, "dbz_p90": 0.0, "dbz_pixels": 0, "sin_precipitacion": True}
    return {
        "dbz_max": float(np.max(precip)),
        "dbz_mean": round(float(np.mean(precip)), 1),
        "dbz_p90": round(float(np.percentile(precip, 90)), 1),
        "dbz_pixels": int(precip.size),
        "sin_precipitacion": False
    }

def find_components(dbz, min_pixels=20):
    precip_mask = np.isfinite(dbz) & (dbz >= 25)
    if not np.any(precip_mask):
        return []
    labeled, num_features = ndi.label(precip_mask)
    components = []
    for i in range(1, num_features + 1):
        pixels = np.count_nonzero(labeled == i)
        if pixels < min_pixels:
            continue
        y, x = ndi.center_of_mass(precip_mask, labeled, i)
        max_dbz = np.max(dbz[labeled == i])
        components.append({"id": i, "x": int(x), "y": int(y), "pixels": int(pixels), "dbz_max": float(max_dbz)})
    components.sort(key=lambda c: c["dbz_max"], reverse=True)
    return components[:10]

def analyze_frame(image, timestamp, tiles_loaded=1):
    arr = np.asarray(image)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    dbz = rgba_to_dbz(image)
    m = metrics(dbz)
    precip = np.isfinite(dbz) & (dbz >= MIN_DBZ)
    finite = np.isfinite(dbz)
    total_con_datos = np.count_nonzero(finite)
    cobertura_pct = 0.0
    if total_con_datos > 0:
        cobertura_pct = (np.count_nonzero(precip) / total_con_datos) * 100

    diagnostic = {
        "modo_imagen": image.mode,
        "ancho": int(image.width),
        "alto": int(image.height),
        "pixeles_paleta_detectados": int(np.count_nonzero(finite)),
        "pixeles_precipitacion": int(np.count_nonzero(precip)),
        "cobertura_pct_real": round(float(cobertura_pct), 2),
        "tolerancia_color": PALETTE_MATCH_DISTANCE,
    }
    components = find_components(dbz)
    return {
        "timestamp": int(timestamp),
        "tiles_ok": int(tiles_loaded),
        "tiles_total": 9,
        "frame_utc": datetime.fromtimestamp(int(timestamp), timezone.utc).isoformat(),
        "area": int(np.count_nonzero(precip)),
        "cobertura_pct": round(float(cobertura_pct), 2),
        "components": components,
        "distance_km": None,
        **m,
        "diagnostico_paleta": diagnostic,
    }

def load_history():
    found = []
    if not HISTORY_DIR.exists():
        return found
    for path in HISTORY_DIR.glob("radar_*.png"):
        try:
            timestamp = int(path.stem.split("_")[-1])
        except:
            continue
        if path.is_file() and path.stat().st_size > 100:
            found.append((timestamp, path))
    found.sort(key=lambda item: item[0])
    return found[-MAX_FRAMES:]

def main():
    files = load_history()
    if not files:
        raise RuntimeError(f"No hay frames en {HISTORY_DIR}")

    print(f"CLIMAAR DECODER V{REPAIR_VERSION} - Frames: {len(files)}")
    results = []
    for timestamp, path in files:
        with Image.open(path) as source:
            image = source.convert("RGBA")
        result = analyze_frame(image, timestamp, tiles_loaded=1)
        results.append(result)
        print(f"{path.name} | Cobertura: {result['cobertura_pct']}% | dBZ max: {result['dbz_max']}")

    results.sort(key=lambda item: item["timestamp"])
    latest = results[-1]
    now_utc = datetime.now(timezone.utc).timestamp()
    age_minutes = (now_utc - latest["timestamp"]) / 60

    if age_minutes > MAX_AGE_MINUTES:
        print(f"ADVERTENCIA: Frame viejo {age_minutes:.0f} min")

    try:
        nowcast = rv.analyze_sequence(results)
    except Exception as e:
        nowcast = {
            "estado": "sin_precipitacion" if latest.get("sin_precipitacion") else "ok",
            "confianza": 100 if latest.get("sin_precipitacion") else 70,
            "movimiento": {"direccion": None, "velocidad_kmh": 0, "confianza": 0}
        }

    nowcast["frame_actual"] = latest["timestamp"]
    nowcast["frame_actual_utc"] = latest["frame_utc"]
    nowcast["frame_timestamp"] = latest["timestamp"]
    nowcast["frame_edad_minutos"] = round(age_minutes, 1)
    nowcast["frame_es_viejo"] = age_minutes > MAX_AGE_MINUTES

    output = {
        "version": REPAIR_VERSION,
        "app": "ClimaAR",
        "ubicacion": {"latitud": getattr(rv, 'LAT', -38.0055), "longitud": getattr(rv, 'LON', -62.0), "ciudad": "Bahia Blanca"},
        "fuente": "RainViewer -> RMA10 Bahia Blanca",
        "frame_actual": latest["timestamp"],
        "frame_actual_utc": latest["frame_utc"],
        "frame_edad_minutos": round(age_minutes, 1),
        "nowcast": nowcast,
        "historial": {"frames_procesados": len(results), "frames_disponibles": len(files)},
        "diagnostico": {"cobertura_pct": latest["cobertura_pct"], "sin_precipitacion": latest.get("sin_precipitacion", False)}
    }

    NOWCAST_FILE.parent.mkdir(parents=True, exist_ok=True)
    NOWCAST_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"NOWCAST V{REPAIR_VERSION} guardado: {NOWCAST_FILE}")

if __name__ == "__main__":
    main()
