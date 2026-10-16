from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RADAR = BASE / "data/radar/radar_nowcast.json"
METEO = BASE / "data/ia/estado_meteorologico.json"
INTEL = BASE / "data/ia/inteligencia_tormenta.json"
OUT = BASE / "data/ia/riesgo_severo.json"

VERSION = "1.0.0"
METODO = "HEURISTICO_NO_CALIBRADO_SIN_DOPPLER"

def load(path):
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}

def finite(value, default=None):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default

def clamp(value, lo=0.0, hi=1.0):
    x = finite(value, lo)
    return max(lo, min(hi, x))

def main():
    radar = load(RADAR)
    meteo = load(METEO)
    intel = load(INTEL)

    nc = radar.get("nowcast", {})
    if not isinstance(nc, dict):
        nc = {}

    env = meteo.get("ambiente_actual", {})
    if not isinstance(env, dict):
        env = {}

    nuclei = nc.get("nucleos_georreferenciados", nc.get("nucleos_analizados", []))
    if not isinstance(nuclei, list):
        nuclei = []

    active = [n for n in nuclei if isinstance(n, dict)]
    strong = [n for n in active if finite(n.get("intensidad"), 0) >= 5]
    severe = [n for n in active if finite(n.get("intensidad"), 0) >= 6]

    growth_values = [
        finite(n.get("cambio_area_pct"))
        for n in active
        if finite(n.get("cambio_area_pct")) is not None
    ]
    growth = max(growth_values) if growth_values else finite(nc.get("cambio_area_pct"), 0)

    shape_scores = []
    for n in active:
        w = finite(n.get("ancho"), 0)
        h = finite(n.get("alto"), 0)
        if w > 0 and h > 0:
            ratio = max(w, h) / max(1.0, min(w, h))
            shape_scores.append(clamp((ratio - 1.0) / 4.0))
    shape_signal = max(shape_scores) if shape_scores else 0.0

    cape = finite(env.get("cape"), 0)
    humidity = finite(env.get("relative_humidity_2m"), 0)
    gust = finite(env.get("wind_gusts_10m"), 0)

    env_signal = (
        0.45 * clamp(cape / 2500.0)
        + 0.30 * clamp(humidity / 100.0)
        + 0.25 * clamp(gust / 100.0)
    )

    intensity_signal = (
        0.55 * clamp((len(strong)) / 3.0)
        + 0.45 * clamp((len(severe)) / 2.0)
    )

    growth_signal = clamp(max(0.0, growth) / 50.0)

    movement_conf = clamp(nc.get("confianza_movimiento", 0))
    active_signal = clamp(len(active) / 30.0)

    # Este índice NO es una probabilidad de tornado.
    # Es un indicador de condiciones compatibles con vigilancia severa.
    severe_index = clamp(
        0.30 * intensity_signal
        + 0.20 * growth_signal
        + 0.15 * shape_signal
        + 0.15 * movement_conf
        + 0.15 * env_signal
        + 0.05 * active_signal
    )

    if severe_index >= 0.75:
        level = "MUY_ALTO"
    elif severe_index >= 0.55:
        level = "ALTO"
    elif severe_index >= 0.35:
        level = "MODERADO"
    else:
        level = "BAJO"

    result = {
        "version": VERSION,
        "actualizado_utc": datetime.now(timezone.utc).isoformat(),
        "metodo": METODO,
        "advertencia": "No usar este índice como probabilidad de tornado. Falta Doppler/cizalladura y calibración histórica.",
        "indice_severo": round(severe_index, 3),
        "nivel": level,
        "senales": {
            "nucleos_activos": len(active),
            "nucleos_intensidad_5_o_mayor": len(strong),
            "nucleos_intensidad_6": len(severe),
            "crecimiento_max_pct": round(growth, 2),
            "senal_forma": round(shape_signal, 3),
            "confianza_movimiento": round(movement_conf, 3),
            "cape": cape,
            "humedad": humidity,
            "rafaga": gust,
        },
        "tornado": {
            "indice_vigilancia": round(severe_index, 3),
            "estado": level,
            "probabilidad": None,
            "motivo": "No se publica probabilidad de tornado sin firma Doppler y validación histórica.",
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    # Integra el índice al motor existente sin inventar una probabilidad.
    if isinstance(intel, dict):
        peligros = intel.setdefault("peligros", {})
        if isinstance(peligros, dict):
            peligros["indice_severo"] = round(severe_index, 3)

        probabilidades = intel.setdefault("probabilidades", {})
        if isinstance(probabilidades, dict):
            probabilidades["tornado"] = None

        intel["riesgo_severo_climaar"] = result
        INTEL.write_text(
            json.dumps(intel, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    print("RIESGO SEVERO CLIMAAR OK")
    print("Índice:", round(severe_index, 3))
    print("Nivel:", level)
    print("Salida:", OUT)

if __name__ == "__main__":
    main()
