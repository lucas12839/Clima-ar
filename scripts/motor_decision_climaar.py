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

VERSION = "ClimaAR-final-3.0"
VALID_SEVERITIES = {"verde", "amarillo", "naranja", "rojo"}
VALID_CLASSES = {
    "sin_tormenta",
    "tormenta",
    "tormenta_fuerte",
    "tormenta_severa",
}


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
        area = first_number(n.get("area_px"), n.get("area"), n.get("pixeles"))
        change = first_number(
            n.get("cambio_area_pct"),
            n.get("cambio_intensidad_pct"),
            n.get("movimiento", {}).get("cambio_area_pct")
            if isinstance(n.get("movimiento"), dict)
            else None,
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
        nowcast.get("intensidad_nivel_max"),
    )
    area = sum(areas) if areas else first_number(nowcast.get("area_px"), nowcast.get("area"))
    change = max(changes) if changes else first_number(nowcast.get("cambio_area_pct"))

    # Esto es SOLO una señal de intensidad radar. Nunca se traduce directamente
    # en severidad meteorológica.
    score = 0.0
    reasons = []
    nivel_max = top

    if nivel_max is not None:
        score += min(nivel_max / 6.0, 1.0) * 0.55
        if nivel_max >= 6:
            reasons.append("RMA10 nivel MUY_FUERTE")
        elif nivel_max >= 5:
            reasons.append("RMA10 nivel FUERTE")
        elif nivel_max >= 4:
            reasons.append("RMA10 nivel MODERADO_FUERTE")
        elif nivel_max >= 3:
            reasons.append("RMA10 nivel MODERADO")

    if area is not None:
        if area >= 5000:
            score += 0.15
        elif area >= 2000:
            score += 0.10
        elif area >= 500:
            score += 0.05

    if change is not None:
        if change >= 30:
            score += 0.20
            reasons.append("crecimiento espacial rapido")
        elif change >= 15:
            score += 0.12
            reasons.append("crecimiento espacial")
        elif change <= -30:
            score -= 0.12
            reasons.append("contraccion rapida")
        elif change <= -15:
            score -= 0.06
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
        "clasificacion": "INTENSIDAD_RADAR_NO_SEVERIDAD",
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
    indices = intel.get("indices_heuristicos", {})
    hazards = indices.get("peligros", {})
    if not isinstance(indices, dict):
        indices = {}
    if not isinstance(hazards, dict):
        hazards = {}
    candidates = [
        indices.get("tormenta"),
        indices.get("llegada_bahia"),
        indices.get("fortalecimiento"),
        indices.get("indice_hibrido_severidad"),
    ] + list(hazards.values())
    vals = [finite(x) for x in candidates if finite(x) is not None]
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
        intel.get("confianza_modelo", intel.get("confianza", 0.0))
    )

    radar_activity = bool(nowcast.get("actividad"))
    raw_movement = clamp(nowcast.get("confianza_movimiento", 0.0))
    speed = finite(nowcast.get("velocidad_kmh"))
    direction = finite(nowcast.get("direccion_grados"))
    moving_to_bahia = bool(nowcast.get("movimiento_hacia_bahia"))

    if speed is None or speed <= 5 or direction is None:
        movement = 0.0
    else:
        movement = raw_movement if moving_to_bahia else min(raw_movement, 0.5)

    indices = intel.get("indices_heuristicos", {})
    hazard_signals = indices.get("peligros", {})
    if not isinstance(indices, dict):
        indices = {}
    if not isinstance(hazard_signals, dict):
        hazard_signals = {}

    storm_signal = clamp(indices.get("tormenta", 0.0))
    arrival_signal = clamp(indices.get("llegada_bahia", 0.0))

    # Estado operativo: describe actividad/evolución. NO es severidad meteorológica.
    if radar_activity:
        operational_score = (
            0.30 * storm_signal
            + 0.25 * intensity["score"]
            + 0.20 * movement
            + 0.25 * ia_score
        )
    else:
        operational_score = min(
            0.15 * storm_signal + 0.10 * intensity["score"] + 0.75 * ia_score,
            0.35,
        )

    operational_score = clamp(operational_score)

    if operational_score >= 0.75:
        state = "actividad_muy_fuerte"
    elif operational_score >= 0.55:
        state = "fuerte"
    elif operational_score >= 0.35:
        state = "activa"
    elif operational_score >= 0.15:
        state = "incipiente"
    else:
        state = "sin_actividad_significativa"

    if intensity["tendencia"] == "fortaleciendose" and operational_score >= 0.35:
        state = "fortaleciendose"
    elif intensity["tendencia"] == "debilitandose" and operational_score < 0.60:
        state = "debilitandose"

    threat = intel.get("amenaza_bahia")
    if not isinstance(threat, dict):
        threat = {}
    threatens = threat.get("amenaza") is True

    # Un sistema que no amenaza a Bahia Blanca no puede presentarse como
    # estado fuerte/muy fuerte para la ciudad solo por su eco radar.
    if radar_activity and not threatens and state in (
        "actividad_muy_fuerte",
        "fuerte",
    ):
        state = "activa"

    severity_raw = intel.get("severidad")
    severity = (
        str(severity_raw).strip().lower()
        if severity_raw is not None
        else ""
    )
    severity = {"roja": "rojo"}.get(severity, severity)
    if severity not in VALID_SEVERITIES:
        severity = "no_determinada"

    intel_radar = intel.get("radar")
    if not isinstance(intel_radar, dict):
        intel_radar = {}

    weather_class = intel.get("clasificacion_meteorologica")
    if weather_class not in VALID_CLASSES:
        weather_class = "no_determinada"

    geo = nowcast.get("georreferenciacion", {})
    if not isinstance(geo, dict):
        geo = {}
    bahia = nowcast.get("bahia_blanca", {})
    if not isinstance(bahia, dict):
        bahia = {}

    principal_raw = nowcast.get("nucleo_principal_id")
    bahia_principal_raw = bahia.get("nucleo_principal_id")

    if (
        principal_raw is not None
        and bahia_principal_raw is not None
        and str(principal_raw) != str(bahia_principal_raw)
    ):
        raise RuntimeError(
            "ERROR: Decision Engine recibió principales inconsistentes: "
            f"nowcast={principal_raw} bahia={bahia_principal_raw}"
        )

    principal_id = first_number(principal_raw)

    if principal_id is None:
        raise RuntimeError(
            "ERROR: Decision Engine no recibió un principal operativo."
        )

    principal_id = int(principal_id)

    result = {
        "version": VERSION,
        "actualizado_utc": datetime.now(timezone.utc).isoformat(),
        "fuente_radar": "RMA10",
        "motor_decision": {
            "estado": state,
            "estado_operativo": state,
            "severidad_meteorologica": severity,
            "clasificacion_meteorologica": weather_class,
            "score_integrado": round(operational_score, 3),
            "actividad_radar": radar_activity,
            "componentes": {
                "radar_intensidad": intensity["score"],
                "radar_movimiento": movement,
                "indice_inteligencia": round(ia_score, 3),
                "ia": round(ia_score, 3),
                "ia_tipo_salida": "indice_heuristico",
                "ia_confianza": model_conf,
            },
            "regla": (
                "La intensidad RMA10 es una señal; la severidad meteorológica "
                "proviene de Intelligence/IA/histórico y no del color radar."
            ),
        },
        "clasificacion_meteorologica": weather_class,
        "amenaza_bahia": threat,
        "clasificacion_historica": intel.get("clasificacion_historica"),
        "severidad_meteorologica": {
            "valor": severity,
            "fuente": "inteligencia_tormenta",
            "determinada": severity != "no_determinada",
        },
        "principal": {
            "track_id": principal_id,
            "fuente": "radar_nowcast",
        },
        "radar": {
            "timestamp": (
                radar.get("radar_timestamp")
                if isinstance(radar.get("radar_timestamp"), str)
                and radar.get("radar_timestamp").strip()
                else nowcast.get("frame_actual_utc")
            ),
            "nowcast": nowcast,
            "intensidad": intensity,
            "georreferenciacion": geo,
            "bahia_blanca": bahia,
        },
        "prediccion": prediction_summary(pred),
        "inteligencia": intel,
        "meteorologia": meteo,
        "senales_heuristicas": {
            "tormenta": storm_signal,
            "llegada_bahia": arrival_signal,
            "fortalecimiento": finite(indices.get("fortalecimiento")),
            "debilitamiento": finite(indices.get("debilitamiento")),
            "lluvia_fuerte": finite(hazard_signals.get("lluvia_fuerte")),
            "viento_fuerte": finite(hazard_signals.get("viento_fuerte")),
            "granizo": finite(hazard_signals.get("granizo")),
            "rayo": finite(hazard_signals.get("rayo")),
        },
        "resumen_operativo": {
            "probabilidad_tormenta": None,
            "probabilidad_llegada_bahia": None,
            "probabilidad_fortalecimiento": None,
            "probabilidad_debilitamiento": None,
            "lluvia_fuerte": None,
            "viento_fuerte": None,
            "granizo": None,
            "rayo": None,
            "eta_minutos": first_number(
                intel_radar.get("eta_minutos"),
                nowcast.get("eta_minutos"),
                bahia.get("eta_minutos"),
            ),
            "distancia_km": first_number(
                intel_radar.get("distancia_km"),
                nowcast.get("distancia_km"),
                bahia.get("distancia_principal_km"),
            ),
            "severidad_meteorologica": severity,
            "clasificacion_meteorologica": weather_class,
            "amenaza_bahia": threat,
            "vigencia_radar": intel_radar.get("vigencia_radar"),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(OUT)

    print("CLIMAAR MOTOR DECISION OK")
    print("Estado operativo:", state)
    print("Severidad meteorologica:", severity)
    print("Score operativo:", round(operational_score, 3))
    print("Intensidad radar:", intensity["score"])
    print("Track principal:", principal_id)
    print("Salida:", OUT)


if __name__ == "__main__":
    main()
