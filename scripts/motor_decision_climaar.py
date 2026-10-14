import json
import math
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RADAR = BASE / "data/radar/radar_nowcast.json"
PRED = BASE / "data/radar/seguimiento/prediccion_evolucion.json"
INTEL = BASE / "data/ia/inteligencia_tormenta.json"
METEO = BASE / "data/ia/estado_meteorologico.json"
OUT = BASE / "data/ia/resultado_final.json"

VERSION = "ClimaAR-final-2.0"


def load(path):
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def finite(v, default=None):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def clamp(v, lo=0.0, hi=1.0):
    x = finite(v, lo)
    return max(lo, min(hi, x))


def first_number(*values):
    for v in values:
        x = finite(v)
        if x is not None:
            return x
    return None


def radar_intensity(nowcast):
    nuclei = nowcast.get("nucleos_georreferenciados")
    if not isinstance(nuclei, list):
        nuclei = nowcast.get("nucleos_analizados", [])
    if not isinstance(nuclei, list):
        nuclei = []

    intensities = []
    areas = []
    changes = []

    for n in nuclei:
        if not isinstance(n, dict):
            continue

        intensity = first_number(
            n.get("intensidad"),
            n.get("intensidad_max"),
            n.get("reflectividad_max"),
        )
        area = first_number(n.get("area_px"), n.get("area"))
        change = first_number(
            n.get("cambio_area_pct"),
            n.get("cambio_intensidad_pct"),
        )

        if intensity is not None:
            intensities.append(intensity)
        if area is not None:
            areas.append(area)
        if change is not None:
            changes.append(change)

    top = max(intensities) if intensities else first_number(
        nowcast.get("intensidad_max"),
        nowcast.get("reflectividad_max"),
    )

    area = sum(areas) if areas else first_number(
        nowcast.get("area_px"),
        nowcast.get("area"),
    )

    change = max(changes) if changes else first_number(
        nowcast.get("cambio_area_pct")
    )

    score = 0.0
    reasons = []

    if top is not None:
        if top >= 60:
            score += 0.55
            reasons.append("reflectividad radar muy alta")
        elif top >= 50:
            score += 0.42
            reasons.append("reflectividad radar alta")
        elif top >= 40:
            score += 0.28
            reasons.append("reflectividad radar moderada-alta")
        elif top >= 30:
            score += 0.15
            reasons.append("reflectividad radar presente")

    if area is not None:
        if area >= 5000:
            score += 0.20
        elif area >= 2000:
            score += 0.14
        elif area >= 500:
            score += 0.08

    if change is not None:
        if change >= 30:
            score += 0.20
            reasons.append("crecimiento espacial rapido")
        elif change >= 15:
            score += 0.12
            reasons.append("crecimiento espacial")
        elif change <= -30:
            score -= 0.15
            reasons.append("contraccion rapida")
        elif change <= -15:
            score -= 0.08
            reasons.append("contraccion")

    score = clamp(score)

    trend = "estable"
    if change is not None:
        if change >= 15:
            trend = "fortaleciendose"
        elif change <= -15:
            trend = "debilitandose"

    return {
        "score": round(score, 3),
        "max_reflectividad": round(top, 1) if top is not None else None,
        "area_total_px": round(area, 1) if area is not None else None,
        "cambio_max_pct": round(change, 1) if change is not None else None,
        "tendencia": trend,
        "nucleos_considerados": len(nuclei),
        "razones": reasons,
    }


def prediction_summary(pred):
    out = {
        "schema_version": pred.get("schema_version"),
        "radar_timestamp": pred.get("radar_timestamp"),
        "intervalo_minutos": pred.get("intervalo_minutos"),
    }

    resumen = pred.get("resumen", {})
    if isinstance(resumen, dict):
        out["resumen"] = resumen

    cal = pred.get("calibracion", {})
    if isinstance(cal, dict):
        out["calibracion"] = cal

    return out


def intelligence_score(intel):
    probs = intel.get("probabilidades", {})
    hazards = intel.get("peligros", {})

    if not isinstance(probs, dict):
        probs = {}
    if not isinstance(hazards, dict):
        hazards = {}

    candidates = [
        probs.get("tormenta"),
        probs.get("llegada_bahia"),
        probs.get("fortalecimiento"),
    ] + list(hazards.values())

    vals = [
        finite(x)
        for x in candidates
        if finite(x) is not None
    ]

    return max(vals) if vals else 0.0


def main():
    radar = load(RADAR)
    pred = load(PRED)
    intel = load(INTEL)
    meteo = load(METEO)

    nowcast = radar.get("nowcast", {})
    if not isinstance(nowcast, dict):
        nowcast = {}

    intensity = radar_intensity(nowcast)
    ia_score = intelligence_score(intel)

    model_conf = clamp(
        intel.get(
            "confianza_modelo",
            intel.get("confianza", 0.0),
        )
    )

    radar_activity = bool(nowcast.get("actividad"))
    movement = clamp(
        nowcast.get("confianza_movimiento", 0.0)
    )

    probs = intel.get("probabilidades", {})
    hazards = intel.get("peligros", {})

    if not isinstance(probs, dict):
        probs = {}
    if not isinstance(hazards, dict):
        hazards = {}

    storm_prob = clamp(
        probs.get("tormenta", 0.0)
    )

    arrival = clamp(
        probs.get("llegada_bahia", 0.0)
    )

    if radar_activity:
        decision_score = (
            0.35 * storm_prob
            + 0.25 * intensity["score"]
            + 0.20 * movement
            + 0.20 * ia_score
        )
    else:
        decision_score = (
            0.15 * storm_prob
            + 0.10 * intensity["score"]
            + 0.75 * ia_score
        )
        decision_score = min(
            decision_score,
            0.35,
        )

    decision_score = clamp(
        decision_score
    )

    if decision_score >= 0.80:
        state = "severa"
    elif decision_score >= 0.60:
        state = "fuerte"
    elif decision_score >= 0.35:
        state = "activa"
    elif decision_score >= 0.15:
        state = "incipiente"
    else:
        state = "sin_actividad_significativa"

    if (
        intensity["tendencia"]
        == "fortaleciendose"
        and state not in ("severa", "fuerte")
        and decision_score >= 0.45
    ):
        state = "fortaleciendose"

    elif (
        intensity["tendencia"]
        == "debilitandose"
        and decision_score < 0.60
    ):
        state = "debilitandose"

    geo = nowcast.get(
        "georreferenciacion",
        {},
    )

    if not isinstance(geo, dict):
        geo = {}

    bahia = nowcast.get(
        "bahia_blanca",
        {},
    )

    if not isinstance(bahia, dict):
        bahia = {}

    result = {
        "version": VERSION,
        "actualizado_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "fuente_radar": "RMA10",

        "motor_decision": {
            "estado": state,
            "score_integrado": round(
                decision_score,
                3,
            ),
            "actividad_radar": radar_activity,

            "componentes": {
                "radar_intensidad": intensity["score"],
                "radar_movimiento": movement,
                "ia": round(
                    ia_score,
                    3,
                ),
                "ia_confianza": model_conf,
            },

            "regla": (
                "radar + movimiento + intensidad + IA; "
                "sin actividad radar la IA queda limitada"
            ),
        },

        "radar": {
            "timestamp": radar.get(
                "radar_timestamp"
            ),
            "nowcast": nowcast,
            "intensidad": intensity,
            "georreferenciacion": geo,
            "bahia_blanca": bahia,
        },

        "prediccion": prediction_summary(
            pred
        ),

        "inteligencia": intel,

        "meteorologia": meteo,

        "resumen_operativo": {
            "probabilidad_tormenta": finite(
                probs.get("tormenta"),
                0.0,
            ),

            "probabilidad_llegada_bahia": finite(
                probs.get(
                    "llegada_bahia"
                ),
                arrival,
            ),

            "probabilidad_fortalecimiento": finite(
                probs.get(
                    "fortalecimiento"
                )
            ),

            "probabilidad_debilitamiento": finite(
                probs.get(
                    "debilitamiento"
                )
            ),

            "lluvia_fuerte": finite(
                hazards.get(
                    "lluvia_fuerte"
                )
            ),

            "viento_fuerte": finite(
                hazards.get(
                    "viento_fuerte"
                )
            ),

            "granizo": finite(
                hazards.get(
                    "granizo"
                )
            ),

            "rayo": finite(
                hazards.get(
                    "rayo"
                )
            ),

            "eta_minutos": first_number(
                nowcast.get(
                    "eta_minutos"
                ),
                bahia.get(
                    "eta_minutos"
                ),
            ),

            "distancia_km": first_number(
                nowcast.get(
                    "distancia_km"
                ),
                bahia.get(
                    "distancia_principal_km"
                ),
            ),
        },
    }

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = OUT.with_suffix(
        ".json.tmp"
    )

    tmp.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    tmp.replace(OUT)

    print(
        "CLIMAAR MOTOR DECISION OK"
    )
    print(
        "Estado:",
        state,
    )
    print(
        "Score:",
        round(
            decision_score,
            3,
        ),
    )
    print(
        "Intensidad:",
        intensity["score"],
    )
    print(
        "Salida:",
        OUT,
    )


if __name__ == "__main__":
    main()
