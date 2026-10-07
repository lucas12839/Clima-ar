import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import radar_rainviewer as rv

BASE = Path(__file__).resolve().parents[1]
RADAR_DIR = BASE / "data" / "radar"
HISTORY_DIR = RADAR_DIR / "historico"
NOWCAST_FILE = RADAR_DIR / "radar_nowcast.json"
ACTUAL_FILE = RADAR_DIR / "actual.png"

MAX_FRAMES = 13
PALETTE_MATCH_DISTANCE = 36.0
MIN_ALPHA_DETECT = 3.0


def rgba_to_dbz_robusto(image):
    array = np.asarray(image).astype(np.float32)
    rgb = array[:, :, :3]
    alpha = array[:, :, 3]
    flat = rgb.reshape(-1, 3)

    distances = ((flat[:, None, :] - rv.DBZ_RGB[None, :, :]) ** 2).sum(axis=2)
    indices = np.argmin(distances, axis=1)
    nearest = np.sqrt(np.min(distances, axis=1)).reshape(rgb.shape[:2])

    dbz = rv.DBZ_VALUES[indices].reshape(rgb.shape[:2]).astype(np.float32)
    valid = (alpha >= MIN_ALPHA_DETECT) & (nearest <= PALETTE_MATCH_DISTANCE)
    dbz[~valid] = np.nan
    return dbz


def analyze(image, timestamp):
    dbz = rgba_to_dbz_robusto(image)
    mask = rv.dbz_mask(dbz)
    components = rv.components_from_mask(mask, dbz)
    metrics = rv.dbz_metrics(dbz)

    for component in components:
        component["distance_km"] = round(
            rv.distance_point_to_bahia(
                component["centroid_x"], component["centroid_y"]
            ),
            2,
        )

    distance = rv.distance_to_bahia(mask)
    alpha = np.asarray(image)[:, :, 3]

    diagnostics = {
        "pixeles_no_transparentes": int(np.count_nonzero(alpha >= MIN_ALPHA_DETECT)),
        "pixeles_paleta_detectados": int(np.count_nonzero(np.isfinite(dbz))),
        "pixeles_precipitacion": int(np.count_nonzero(np.isfinite(dbz) & (dbz >= 10.0))),
        "tolerancia_color": PALETTE_MATCH_DISTANCE,
        "alpha_minimo": MIN_ALPHA_DETECT,
    }

    return {
        "timestamp": int(timestamp),
        "frame_utc": datetime.fromtimestamp(int(timestamp), timezone.utc).isoformat(),
        "tiles_ok": 9,
        "path": "historico",
        "area": int(np.count_nonzero(mask)),
        "components": components,
        "distance_km": round(distance, 2) if distance is not None else None,
        **metrics,
        "diagnostico_paleta": diagnostics,
    }


def main():
    files = []
    for path in HISTORY_DIR.glob("radar_*.png"):
        try:
            timestamp = int(path.stem.split("_")[-1])
        except (ValueError, IndexError):
            continue
        if path.is_file() and path.stat().st_size > 100:
            files.append((timestamp, path))

    files.sort(key=lambda item: item[0])
    files = files[-MAX_FRAMES:]

    if not files:
        raise RuntimeError("No hay frames históricos para reparar.")

    results = []
    for timestamp, path in files:
        image = Image.open(path).convert("RGBA")
        result = analyze(image, timestamp)
        results.append(result)
        d = result["diagnostico_paleta"]
        print(
            f"frame={timestamp} area={result['area']} dbz_max={result['dbz_max']} "
            f"dbz_pixels={result['dbz_pixels']} nontransparent={d['pixeles_no_transparentes']}"
        )

    results.sort(key=lambda item: item["timestamp"])
    nowcast = rv.analyze_sequence(results)

    output = {
        "version": "7.3",
        "app": "ClimaAR",
        "ubicacion": {"latitud": rv.LAT, "longitud": rv.LON, "ciudad": "Bahia Blanca"},
        "fuente": "RainViewer",
        "rainviewer_color_scheme": "Universal Blue (2)",
        "rainviewer_smooth": 0,
        "rainviewer_snow": 0,
        "dbz_decode": "tabla oficial Universal Blue + tolerancia robusta",
        "bahia_pixel_mosaico": {
            "x": round(rv.bahia_pixel_position()[0], 2),
            "y": round(rv.bahia_pixel_position()[1], 2),
        },
        "nowcast": nowcast,
        "historial": {
            "frames_procesados": len(results),
            "frames_disponibles": len(files),
            "max_history_frames": rv.MAX_HISTORY_FRAMES,
        },
        "diagnostico": {
            "metodo": "redecodificacion_robusta_desde_PNG_historico",
            "tolerancia_color": PALETTE_MATCH_DISTANCE,
            "alpha_minimo": MIN_ALPHA_DETECT,
            "ultimo_frame": results[-1]["diagnostico_paleta"],
        },
    }

    NOWCAST_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("NOWCAST REPARADO")
    print(json.dumps(nowcast, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
