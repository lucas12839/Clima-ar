import csv
import json
import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


VERSION = "3.0.0"

BASE_DIR = Path(__file__).resolve().parents[1]
IA_DIR = BASE_DIR / "data" / "ia"
RADAR_FILE = BASE_DIR / "data" / "radar" / "radar_nowcast.json"
FUSION_FILE = IA_DIR / "estado_meteorologico.json"
OUTPUT_FILE = IA_DIR / "inteligencia_tormenta.json"
HISTORY_FILE = IA_DIR / "historial_inteligencia_tormenta.csv"

LAT = -38.71
LON = -62.26

# Referencia historica documental. No se usa como "contador de tormentas":
# sirve para reconocer ambientes compatibles con episodios severos conocidos.
HISTORICAL_EVENTS = (
    ("2019-12-30", "extremo"),
    ("2023-12-16", "extremo"),
    ("2025-03-07", "extremo"),
)

HISTORY_FIELDS = [
    "timestamp_utc", "estado", "severidad",
    "prob_tormenta", "prob_llegada_bahia",
    "prob_fortalecimiento", "prob_debilitamiento",
    "prob_lluvia_fuerte", "prob_viento_fuerte",
    "prob_granizo", "prob_rayo",
    "eta_minutos", "duracion_estimada_minutos",
    "velocidad_kmh", "direccion_grados",
    "cape", "humedad", "rafaga_kmh",
    "calidad_datos", "confianza_modelo",
]


def now_dt():
    return datetime.now(timezone.utc)


def now_utc():
    return now_dt().isoformat()


def finite(value, default=None):
    try:
        if value is None:
            return default
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def clamp(value, low=0.0, high=1.0):
    value = finite(value, 0.0)
    return max(low, min(high, value))


def weighted_average(values):
    pairs = [(finite(v), finite(w, 1.0)) for v, w in values]
    pairs = [(v, w) for v, w in pairs if v is not None and w > 0]
    if not pairs:
        return 0.0
    return clamp(sum(v * w for v, w in pairs) / sum(w for _, w in pairs))


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(path)


def parse_dt(value):
    if not value:
        return None
    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def nearest_record(records, target=None):
    if not isinstance(records, list) or not records:
        return None
    target = target or now_dt()
    best = None
    best_diff = None
    for row in records:
        if not isinstance(row, dict):
            continue
        dt = parse_dt(row.get("tiempo_utc"))
        if dt is None:
            continue
        diff = abs((dt - target).total_seconds())
        if best_diff is None or diff < best_diff:
            best, best_diff = row, diff
    return best


def load_inputs():
    fusion = load_json(FUSION_FILE)
    radar = load_json(RADAR_FILE)
    if not isinstance(fusion, dict):
        raise RuntimeError("No existe estado_meteorologico.json valido.")
    return fusion, radar if isinstance(radar, dict) else {}


def current_environment(fusion):
    actual = fusion.get("ambiente_actual")
    if isinstance(actual, dict):
        return actual

    future = fusion.get("ambiente_futuro", {})
    points = future.get("puntos", []) if isinstance(future, dict) else []
    if not isinstance(points, list):
        return None

    for point in points:
        if isinstance(point, dict) and point.get("punto") == "BB_CENTRO":
            row = nearest_record(point.get("datos", []))
            if row:
                return row
    return None


def future_points(fusion):
    future = fusion.get("ambiente_futuro", {})
    points = future.get("puntos", []) if isinstance(future, dict) else []
    return points if isinstance(points, list) else []


def radar_summary(radar):
    nowcast = radar.get("nowcast", {})
    return nowcast if isinstance(nowcast, dict) else {}


def projections(nowcast):
    data = nowcast.get("proyecciones", {})
    return data if isinstance(data, dict) else {}


def environment_along_projection(points, projection):
    if not isinstance(projection, dict):
        return None

    projected_lat = finite(projection.get("latitud"), LAT)
    projected_lon = finite(projection.get("longitud"), LON)
    minutes = finite(projection.get("minutos"), 0.0)
    target = now_dt() + timedelta(minutes=minutes)

    best = None
    best_score = None

    for point in points:
        if not isinstance(point, dict):
            continue
        plat = finite(point.get("latitud"), LAT)
        plon = finite(point.get("longitud"), LON)
        row = nearest_record(point.get("datos", []), target)
        if row is None:
            continue
        dt = parse_dt(row.get("tiempo_utc"))
        if dt is None:
            continue

        spatial = math.hypot(plat - projected_lat, plon - projected_lon)
        temporal = abs((dt - target).total_seconds()) / 3600.0
        score = spatial * 3.0 + temporal

        if best_score is None or score < best_score:
            best, best_score = row, score

    return best


def environment_score(env):
    if not env:
        return 0.05, []

    cape = finite(env.get("cape"), 0)
    humidity = finite(env.get("relative_humidity_2m"), 0)
    temperature = finite(env.get("temperature_2m"))
    dew = finite(env.get("dew_point_2m"))
    rain = finite(env.get("rain"), 0)
    showers = finite(env.get("showers"), 0)
    precipitation = finite(env.get("precipitation"), 0)
    gust = finite(env.get("wind_gusts_10m"), 0)

    score = 0.0
    reasons = []

    if cape >= 1500:
        score += 0.35
        reasons.append("CAPE alto")
    elif cape >= 800:
        score += 0.25
        reasons.append("CAPE moderado-alto")
    elif cape >= 300:
        score += 0.12
        reasons.append("CAPE presente")

    if humidity >= 75:
        score += 0.15
        reasons.append("humedad alta")
    elif humidity >= 60:
        score += 0.08

    if temperature is not None and dew is not None:
        spread = temperature - dew
        if spread <= 8:
            score += 0.15
            reasons.append("capa baja humeda")
        elif spread <= 12:
            score += 0.07

    if rain > 0 or showers > 0 or precipitation > 0:
        score += 0.18
        reasons.append("precipitacion prevista")

    if gust >= 45:
        score += 0.08
        reasons.append("rafagas potencialmente fuertes")

    return clamp(score), reasons


def active_storm_probability(nowcast, env):
    env_prob, _ = environment_score(env)
    activity = bool(nowcast.get("actividad"))
    area = finite(nowcast.get("area_px"), 0)
    trend = nowcast.get("fortalecimiento")

    if not activity or area <= 0:
        return clamp(env_prob * 0.8)

    probability = 0.42
    if area >= 500:
        probability += 0.12
    if area >= 2000:
        probability += 0.10
    if trend == "fortaleciendose":
        probability += 0.12
    elif trend == "debilitandose":
        probability -= 0.08

    probability += env_prob * 0.20
    return clamp(probability, 0.20, 0.98)


def movement_quality(nowcast):
    if not nowcast.get("actividad"):
        return 0.0

    confidence = clamp(nowcast.get("confianza_movimiento"))
    speed = finite(nowcast.get("velocidad_kmh"))
    distance = finite(nowcast.get("distancia_km"))
    eta = finite(nowcast.get("eta_minutos"))
    direction = finite(nowcast.get("direccion_grados"))
    toward = bool(nowcast.get("movimiento_hacia_bahia"))

    quality = 0.45 * confidence
    if speed is not None and 2 <= speed <= 180:
        quality += 0.15
    if distance is not None:
        quality += 0.10
    if eta is not None:
        quality += 0.10
    if direction is not None and 0 <= direction <= 360:
        quality += 0.05
    if projections(nowcast):
        quality += 0.05
    if toward:
        quality += 0.10

    return clamp(quality)


def impact_probability(nowcast):
    if not nowcast.get("actividad"):
        return 0.0

    toward = bool(nowcast.get("movimiento_hacia_bahia"))
    distance = finite(nowcast.get("distancia_km"))
    eta = finite(nowcast.get("eta_minutos"))
    tracking = movement_quality(nowcast)

    if not toward:
        return round(0.08 * tracking, 3)

    proximity = 0.20 if distance is None else clamp(
        1.0 - distance / 180.0, 0.10, 1.0
    )

    if eta is not None:
        if eta <= 120:
            proximity += 0.12
        if eta <= 60:
            proximity += 0.12
        if eta <= 30:
            proximity += 0.10

    return clamp(0.20 + 0.45 * proximity + 0.35 * tracking)


def strengthening_probability(nowcast, env):
    if not nowcast.get("actividad"):
        return None

    trend = nowcast.get("fortalecimiento")
    change = finite(nowcast.get("cambio_area_pct"))
    env_probability, _ = environment_score(env)

    probability = 0.35
    if trend == "fortaleciendose":
        probability += 0.30
    elif trend == "debilitandose":
        probability -= 0.25

    if change is not None:
        if change >= 30:
            probability += 0.15
        elif change >= 15:
            probability += 0.08
        elif change <= -30:
            probability -= 0.12

    probability += (env_probability - 0.25) * 0.30
    return clamp(probability)


def historical_severe_environment_score(env):
    """
    Referencia de ambiente compatible con los episodios severos documentados.
    No afirma que el ambiente actual sea igual a un evento historico.
    """
    if not env:
        return 0.0, []

    cape = finite(env.get("cape"), 0)
    humidity = finite(env.get("relative_humidity_2m"), 0)
    gust = finite(env.get("wind_gusts_10m"), 0)
    temp = finite(env.get("temperature_2m"))
    dew = finite(env.get("dew_point_2m"))
    rain = finite(env.get("rain"), 0)
    precip = finite(env.get("precipitation"), 0)

    score = 0.0
    reasons = []

    if cape >= 1500:
        score += 0.35
        reasons.append("ambiente compatible con conveccion muy inestable")
    elif cape >= 800:
        score += 0.22
        reasons.append("inestabilidad compatible con tormentas fuertes")
    elif cape >= 500:
        score += 0.10

    if humidity >= 75:
        score += 0.15

    if gust >= 60:
        score += 0.20
        reasons.append("rafagas ambientales altas")
    elif gust >= 45:
        score += 0.10

    if temp is not None and dew is not None and temp - dew <= 8:
        score += 0.10

    if rain >= 5 or precip >= 5:
        score += 0.10

    return clamp(score), reasons


def radar_intensity_score(nowcast):
    if not nowcast.get("actividad"):
        return 0.0, []

    area = finite(nowcast.get("area_px"), 0)
    change = finite(nowcast.get("cambio_area_pct"), 0)
    confidence = clamp(nowcast.get("confianza_movimiento"))

    score = 0.12
    reasons = []

    if area >= 500:
        score += 0.10
    if area >= 2000:
        score += 0.15
    if area >= 5000:
        score += 0.15

    if change >= 30:
        score += 0.18
        reasons.append("crecimiento radar rapido")
    elif change >= 15:
        score += 0.10
        reasons.append("crecimiento radar")

    if nowcast.get("fortalecimiento") == "fortaleciendose":
        score += 0.12
        reasons.append("tendencia de fortalecimiento")

    score += 0.10 * confidence
    return clamp(score), reasons


def severity_hybrid_score(nowcast, env, hazards, impact, strengthening):
    radar_score, radar_reasons = radar_intensity_score(nowcast)
    historical_score, historical_reasons = historical_severe_environment_score(env)

    hazard_score = max(hazards.values()) if hazards else 0.0
    strengthening_score = strengthening if strengthening is not None else 0.0

    # Radar actual y evolucion tienen mayor peso que una analogia historica.
    raw = (
        0.38 * radar_score
        + 0.27 * hazard_score
        + 0.20 * historical_score
        + 0.10 * impact
        + 0.05 * strengthening_score
    )

    return clamp(raw), radar_reasons, historical_reasons


def hazard_probabilities(nowcast, env):
    if not nowcast.get("actividad"):
        return {
            "lluvia_fuerte": 0.02,
            "viento_fuerte": 0.02,
            "granizo": 0.01,
            "rayo": 0.01,
        }

    area = finite(nowcast.get("area_px"), 0)
    trend = nowcast.get("fortalecimiento")
    cape = finite(env.get("cape") if env else None, 0)
    gust = finite(env.get("wind_gusts_10m") if env else None, 0)
    rain = finite(env.get("rain") if env else None, 0)
    showers = finite(env.get("showers") if env else None, 0)
    precipitation = finite(env.get("precipitation") if env else None, 0)
    lightning = finite(env.get("lightning_density") if env else None, 0)

    area_factor = clamp(area / 4000.0)

    heavy_rain = 0.12 + 0.25 * area_factor
    if rain >= 5 or showers >= 5 or precipitation >= 5:
        heavy_rain += 0.25
    elif rain > 0 or showers > 0 or precipitation > 0:
        heavy_rain += 0.08
    if trend == "fortaleciendose":
        heavy_rain += 0.10

    strong_wind = 0.08 + 0.20 * area_factor
    if gust >= 60:
        strong_wind += 0.30
    elif gust >= 45:
        strong_wind += 0.18
    if trend == "fortaleciendose":
        strong_wind += 0.08

    hail = 0.02 + 0.18 * area_factor
    if cape >= 1500:
        hail += 0.15
    elif cape >= 800:
        hail += 0.08
    if trend == "fortaleciendose" and cape >= 800:
        hail += 0.05

    lightning_probability = 0.03 + 0.18 * area_factor
    if cape >= 800:
        lightning_probability += 0.18
    if lightning > 0:
        lightning_probability += min(0.45, lightning / 5.0)

    return {
        "lluvia_fuerte": clamp(heavy_rain),
        "viento_fuerte": clamp(strong_wind),
        "granizo": clamp(hail),
        "rayo": clamp(lightning_probability),
    }


def classify_storm(nowcast, env, hazards, hybrid_score):
    if not nowcast.get("actividad"):
        return "sin_tormenta_activa"

    trend = nowcast.get("fortalecimiento")
    rain = finite(env.get("rain") if env else None, 0)
    max_hazard = max(hazards.values()) if hazards else 0.0

    if hybrid_score >= 0.68 and max_hazard >= 0.55:
        return "tormenta_severa_potencial"

    if trend == "fortaleciendose" and hybrid_score >= 0.40:
        return "tormenta_convectiva_en_desarrollo"

    if rain > 0 or hazards.get("lluvia_fuerte", 0) >= 0.30:
        return "tormenta_con_lluvia"

    return "tormenta_activa"


def severity(nowcast, hybrid_score, confidence, impact):
    if not nowcast.get("actividad"):
        return "verde"

    raw = 0.60 * hybrid_score + 0.25 * impact + 0.15 * confidence

    if raw >= 0.72:
        return "naranja"
    if raw >= 0.45:
        return "amarillo"
    return "verde"


def duration_estimate(nowcast, env, impact):
    if not nowcast.get("actividad"):
        return None

    speed = finite(nowcast.get("velocidad_kmh"))
    rain = finite(env.get("rain") if env else None, 0)
    showers = finite(env.get("showers") if env else None, 0)

    if speed is None or speed <= 5:
        passage = 60.0
    else:
        passage = max(20.0, min(180.0, 60.0 * 80.0 / speed))

    if rain > 0 or showers > 0:
        passage += 20.0

    if impact < 0.35:
        passage *= 0.7

    return int(round(max(20.0, min(240.0, passage))))


def confidence_score(fusion, nowcast, env):
    quality_data = fusion.get("calidad_datos", {})
    if not isinstance(quality_data, dict):
        quality_data = {}

    quality = clamp(finite(quality_data.get("score"), 0) / 100.0)
    tracking = movement_quality(nowcast)
    environment = 1.0 if env else 0.0
    radar_present = 1.0 if nowcast else 0.0

    # La confianza refleja calidad de los datos, no severidad.
    return round(
        min(
            0.82,
            0.35 * quality
            + 0.35 * tracking
            + 0.15 * environment
            + 0.15 * radar_present,
        ),
        2,
    )


def build_reasons(
    nowcast,
    env,
    hazards,
    impact,
    hybrid_score,
    radar_reasons,
    historical_reasons,
):
    reasons = []

    if not nowcast.get("actividad"):
        reasons.append("No se detecta precipitacion activa en el radar.")
    else:
        reasons.append("Existe actividad de precipitacion en el radar.")

        if nowcast.get("movimiento_hacia_bahia"):
            reasons.append("El movimiento estimado apunta hacia Bahia Blanca.")
        else:
            reasons.append("No se confirma movimiento hacia Bahia Blanca.")

        if nowcast.get("fortalecimiento") == "fortaleciendose":
            reasons.append("El area radar esta aumentando.")
        elif nowcast.get("fortalecimiento") == "debilitandose":
            reasons.append("El area radar esta disminuyendo.")

    reasons.extend(radar_reasons)

    if historical_reasons:
        reasons.append(
            "El ambiente actual presenta algunas caracteristicas compatibles "
            "con ambientes de episodios severos historicos."
        )
        reasons.extend(historical_reasons)

    if env:
        cape = finite(env.get("cape"))
        gust = finite(env.get("wind_gusts_10m"))
        rain = finite(env.get("rain"))

        if cape is not None:
            reasons.append(f"CAPE ECMWF: {cape:.0f} J/kg.")
        if gust is not None:
            reasons.append(f"Rafaga prevista ECMWF: {gust:.1f} km/h.")
        if rain is not None and rain > 0:
            reasons.append(f"Lluvia prevista: {rain:.1f} mm.")

    reasons.append(f"Indice hibrido de severidad: {hybrid_score:.0%}.")
    reasons.append(f"Probabilidad estimada de impacto: {impact:.0%}.")
    reasons.append(
        "La severidad y la trayectoria se calculan como dimensiones separadas."
    )
    reasons.append(
        "La estimacion no reemplaza avisos oficiales y requiere validacion continua."
    )

    return reasons


def append_history(result):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    exists = HISTORY_FILE.exists()

    probabilities = result["probabilidades"]
    hazards = result["peligros"]
    environment = result.get("ambiente") or {}
    radar = result.get("radar") or {}

    row = {
        "timestamp_utc": result["actualizado_utc"],
        "estado": result["estado"],
        "severidad": result["severidad"],
        "prob_tormenta": probabilities["tormenta"],
        "prob_llegada_bahia": probabilities["llegada_bahia"],
        "prob_fortalecimiento": probabilities["fortalecimiento"],
        "prob_debilitamiento": probabilities["debilitamiento"],
        "prob_lluvia_fuerte": hazards["lluvia_fuerte"],
        "prob_viento_fuerte": hazards["viento_fuerte"],
        "prob_granizo": hazards["granizo"],
        "prob_rayo": hazards["rayo"],
        "eta_minutos": result["eta_minutos"],
        "duracion_estimada_minutos": result["duracion_estimada_minutos"],
        "velocidad_kmh": radar.get("velocidad_kmh"),
        "direccion_grados": radar.get("direccion_grados"),
        "cape": environment.get("cape"),
        "humedad": environment.get("relative_humidity_2m"),
        "rafaga_kmh": environment.get("wind_gusts_10m"),
        "calidad_datos": result["calidad_datos"],
        "confianza_modelo": result["confianza_modelo"],
    }

    with HISTORY_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    fusion, radar_file = load_inputs()
    nowcast = radar_summary(radar_file)
    environment = current_environment(fusion)
    future = future_points(fusion)

    storm_probability = active_storm_probability(nowcast, environment)
    impact = impact_probability(nowcast)
    strengthening = strengthening_probability(nowcast, environment)
    weakening = None if strengthening is None else clamp(1.0 - strengthening)

    hazards = hazard_probabilities(nowcast, environment)

    hybrid_score, radar_reasons, historical_reasons = severity_hybrid_score(
        nowcast, environment, hazards, impact, strengthening
    )

    storm_class = classify_storm(
        nowcast, environment, hazards, hybrid_score
    )

    tracking_quality = movement_quality(nowcast)
    confidence = confidence_score(fusion, nowcast, environment)
    severity_level = severity(
        nowcast, hybrid_score, confidence, impact
    )

    duration = duration_estimate(
        nowcast, environment, impact
    )

    trajectory_environment = {}
    for key, projection in projections(nowcast).items():
        projected_environment = environment_along_projection(
            future, projection
        )
        if projected_environment:
            trajectory_environment[str(key)] = {
                "minutos": projection.get("minutos"),
                "latitud": projection.get("latitud"),
                "longitud": projection.get("longitud"),
                "ambiente": projected_environment,
            }

    reasons = build_reasons(
        nowcast,
        environment,
        hazards,
        impact,
        hybrid_score,
        radar_reasons,
        historical_reasons,
    )

    result = {
        "version": VERSION,
        "motor": "ClimaAR Intelligence - Storm Engine Central",
        "actualizado_utc": now_utc(),
        "ubicacion": {
            "ciudad": "Bahia Blanca",
            "latitud": LAT,
            "longitud": LON,
        },
        "estado": storm_class,
        "severidad": severity_level,
        "probabilidades": {
            "tormenta": round(storm_probability, 3),
            "llegada_bahia": round(impact, 3),
            "fortalecimiento": (
                round(strengthening, 3)
                if strengthening is not None
                else None
            ),
            "debilitamiento": (
                round(weakening, 3)
                if weakening is not None
                else None
            ),
        },
        "peligros": {
            key: round(value, 3)
            for key, value in hazards.items()
        },
        "indice_hibrido_severidad": round(hybrid_score, 3),
        "referencia_historica": {
            "eventos": [
                {"fecha": date, "tipo": kind}
                for date, kind in HISTORICAL_EVENTS
            ],
            "score_ambiente_compatible": round(
                historical_severe_environment_score(environment)[0], 3
            ),
        },
        "eta_minutos": nowcast.get("eta_minutos"),
        "duracion_estimada_minutos": duration,
        "radar": nowcast,
        "ambiente": environment,
        "ambiente_en_trayectoria": trajectory_environment,
        "calidad_datos": finite(
            (
                fusion.get("calidad_datos", {}).get("score")
                if isinstance(fusion.get("calidad_datos", {}), dict)
                else None
            ),
            0,
        ),
        "calidad_tracking": round(tracking_quality, 3),
        "confianza_modelo": confidence,
        "limitaciones": [
            "El radar actual no aporta dBZ cuantitativo validado en este motor.",
            "No hay observacion directa de rayos integrada en esta etapa.",
            "La referencia historica no sustituye observaciones radar en tiempo real.",
            "La severidad es una estimacion y no reemplaza avisos oficiales.",
            "Las probabilidades requieren calibracion continua con historico temporal.",
        ],
        "razones": reasons,
    }

    save_json(OUTPUT_FILE, result)
    append_history(result)

    print("============================================")
    print("ClimaAR Storm Engine Central")
    print(f"Version {VERSION}")
    print("============================================")
    print(f"Estado: {storm_class}")
    print(f"Severidad: {severity_level}")
    print(f"Probabilidad tormenta: {storm_probability:.0%}")
    print(f"Impacto Bahia Blanca: {impact:.0%}")
    print(f"Indice severidad: {hybrid_score:.0%}")
    print(f"Calidad tracking: {tracking_quality:.0%}")
    print(f"Confianza modelo: {confidence:.0%}")
    print("STORM ENGINE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
