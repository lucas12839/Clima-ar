from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[1]

RADAR_FILE = BASE / "data" / "radar" / "radar_nowcast.json"
ML_FILE = BASE / "data" / "radar" / "seguimiento" / "prediccion_ml_rma10.json"
METEO_FILE = BASE / "data" / "ia" / "estado_meteorologico.json"
GOES_STATE_FILE = BASE / "data" / "goes19" / "goes19_state.json"
GOES_INTELLIGENCE_FILE = BASE / "data" / "ia" / "goes19_intelligence.json"
INTELLIGENCE_FILE = BASE / "data" / "ia" / "inteligencia_tormenta.json"

GOES_MAX_REUSE_MINUTES = 180.0
GOES_STATE_TIMEOUT_SECONDS = 300


def parse_timestamp(value: str) -> datetime:
    value = str(value).strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_optional_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        return parse_timestamp(str(value))
    except Exception:
        return None


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"JSON inválido: {path}") from exc


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    number = finite(value)
    if number is None:
        return low
    return max(low, min(high, number))


def load_radar_timestamp() -> datetime:
    data = load_json(RADAR_FILE)
    nowcast = data.get("nowcast")
    if not isinstance(nowcast, dict):
        raise RuntimeError("radar_nowcast.json no contiene un bloque nowcast válido.")
    timestamp = nowcast.get("frame_actual_utc")
    if not timestamp:
        raise RuntimeError("No existe nowcast.frame_actual_utc.")
    return parse_timestamp(timestamp)


def validate_radar_nowcast_ml(radar_timestamp: datetime) -> None:
    radar_data = load_json(RADAR_FILE)
    nowcast = radar_data.get("nowcast")
    if not isinstance(nowcast, dict):
        raise RuntimeError("ERROR: falta el objeto nowcast.")

    nowcast_timestamp = parse_timestamp(str(nowcast.get("frame_actual_utc")))
    if nowcast_timestamp != radar_timestamp:
        raise RuntimeError(
            "ERROR: Radar ↔ Nowcast desincronizados: "
            f"radar={radar_timestamp.isoformat()} "
            f"nowcast={nowcast_timestamp.isoformat()}"
        )

    ml_data = load_json(ML_FILE)
    if ml_data.get("ok") is not True:
        raise RuntimeError("ERROR: predicción ML no válida.")

    ml_value = ml_data.get("frame_radar")
    if not ml_value:
        raise RuntimeError("ERROR: ML no contiene frame_radar.")

    ml_timestamp = parse_timestamp(str(ml_value))
    if ml_timestamp != radar_timestamp:
        raise RuntimeError(
            "ERROR: Radar ↔ ML desincronizados: "
            f"radar={radar_timestamp.isoformat()} "
            f"ml={ml_timestamp.isoformat()}"
        )

    print(
        "Sincronización estricta OK: "
        f"Radar={radar_timestamp.isoformat()} | "
        f"Nowcast={nowcast_timestamp.isoformat()} | "
        f"ML={ml_timestamp.isoformat()}"
    )


def parse_row_time(value: Any) -> datetime | None:
    if not value:
        return None
    return parse_optional_timestamp(value)


def temporal_interpolation(rows, target):
    valid = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        dt = parse_row_time(row.get("tiempo_utc"))
        if dt is not None:
            valid.append((dt, row))

    if not valid:
        return None

    valid.sort(key=lambda item: item[0])

    for dt, row in valid:
        if dt == target:
            result = dict(row)
            result["tiempo_utc"] = target.isoformat()
            result["interpolacion_temporal"] = False
            return result

    before = None
    after = None
    for dt, row in valid:
        if dt < target:
            before = (dt, row)
        elif dt > target:
            after = (dt, row)
            break

    if before is None or after is None:
        return None

    t0, r0 = before
    t1, r1 = after
    total = (t1 - t0).total_seconds()
    alpha = (target - t0).total_seconds() / total if total > 0 else 0.0
    result = dict(r0)

    for key in set(r0) | set(r1):
        a = finite(r0.get(key))
        b = finite(r1.get(key))
        if a is not None and b is not None:
            result[key] = a + (b - a) * alpha
        elif key in r1 and key not in r0:
            result[key] = r1[key]

    result["tiempo_utc"] = target.isoformat()
    result["interpolacion_temporal"] = True
    result["factor_temporal"] = round(alpha, 4)
    result["tiempo_anterior_utc"] = t0.isoformat()
    result["tiempo_siguiente_utc"] = t1.isoformat()
    return result


def distance_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def environment_along_projection(points, projection, radar_timestamp):
    if not isinstance(projection, dict):
        return None

    projected_lat = finite(projection.get("latitud"))
    projected_lon = finite(projection.get("longitud"))
    minutes = finite(projection.get("minutos"))

    if projected_lat is None or projected_lon is None:
        return None
    if minutes is None:
        minutes = 0.0

    target = radar_timestamp + timedelta(minutes=minutes)
    candidates = []

    for point in points:
        if not isinstance(point, dict):
            continue
        plat = finite(point.get("latitud"))
        plon = finite(point.get("longitud"))
        rows = point.get("datos", [])
        if plat is None or plon is None or not isinstance(rows, list):
            continue

        row = temporal_interpolation(rows, target)
        if row is None:
            continue

        distance = distance_km(projected_lat, projected_lon, plat, plon)
        candidates.append((distance, point, row))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    candidates = candidates[:4]

    if candidates[0][0] < 0.001:
        result = dict(candidates[0][2])
        result["latitud"] = projected_lat
        result["longitud"] = projected_lon
        result["punto"] = candidates[0][1].get("punto", "GRILLA")
        result["interpolacion_espacial"] = False
        result["distancia_punto_mas_cercano_km"] = 0.0
        return result

    weights = []
    for distance, point, row in candidates:
        weight = 1.0 / max(distance, 0.001) ** 2
        weights.append((weight, point, row, distance))

    nearest_row = candidates[0][2]
    result = dict(nearest_row)
    numeric_keys = set()
    for _, _, row, _ in weights:
        numeric_keys.update(row.keys())

    for key in numeric_keys:
        values = []
        for weight, _, row, _ in weights:
            value = finite(row.get(key))
            if value is not None:
                values.append((weight, value))
        if values:
            denominator = sum(weight for weight, _ in values)
            if denominator > 0:
                result[key] = sum(weight * value for weight, value in values) / denominator

    result["latitud"] = projected_lat
    result["longitud"] = projected_lon
    result["punto"] = "INTERPOLADO_GRILLA_ECMWF"
    result["interpolacion_espacial"] = True
    result["puntos_utilizados"] = [point.get("punto") for _, point, _, _ in weights]
    result["distancias_puntos_km"] = {
        point.get("punto", "GRILLA"): round(distance, 3)
        for _, point, _, distance in weights
    }
    return result


def run_goes_state() -> tuple[dict[str, Any] | None, str]:
    """Actualiza GOES-19 con el mismo frame RMA10 y conserva degradación controlada."""
    command = [sys.executable, "-m", "goes19.state"]
    print("Actualizando GOES-19 con el frame RMA10...")

    try:
        completed = subprocess.run(
            command,
            cwd=BASE,
            text=True,
            capture_output=True,
            timeout=GOES_STATE_TIMEOUT_SECONDS,
            check=False,
        )
    except Exception as exc:
        print(f"AVISO GOES-19: no se pudo ejecutar state: {exc}")
        completed = None

    if completed is not None:
        if completed.stdout:
            print(completed.stdout.rstrip())
        if completed.stderr:
            print(completed.stderr.rstrip(), file=sys.stderr)

    if GOES_STATE_FILE.exists():
        try:
            state = load_json(GOES_STATE_FILE)
            if isinstance(state, dict):
                return state, "actualizado" if completed and completed.returncode == 0 else "reutilizado"
        except Exception as exc:
            print(f"AVISO GOES-19: estado existente inválido: {exc}")

    return None, "unavailable"


def product_temporal(data: Any, reference: datetime) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {
            "status": "unavailable",
            "age_minutes": None,
            "temporal_validity": "unknown",
        }

    source = data.get("source")
    if not isinstance(source, dict):
        source = {}

    timestamp = parse_optional_timestamp(source.get("last_modified_utc"))
    if timestamp is None:
        timestamp = parse_optional_timestamp(data.get("last_modified_utc"))

    status = str(data.get("status", "unavailable"))

    if timestamp is None:
        return {
            "status": status,
            "age_minutes": None,
            "temporal_validity": "unknown",
        }

    offset = (reference - timestamp).total_seconds() / 60.0
    if offset < 0:
        validity = "future_relative_to_frame"
    elif offset > GOES_MAX_REUSE_MINUTES:
        validity = "stale"
    else:
        validity = "valid"

    return {
        "status": status,
        "age_minutes": round(max(0.0, offset), 3),
        "offset_minutes": round(offset, 3),
        "temporal_validity": validity,
        "last_modified_utc": timestamp.isoformat().replace("+00:00", "Z"),
    }


def find_number(mapping: Any, keys: tuple[str, ...]) -> float | None:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = finite(mapping.get(key))
        if value is not None:
            return value
    return None


def stats_of(product: Any) -> dict[str, Any]:
    if not isinstance(product, dict):
        return {}
    stats = product.get("statistics")
    return stats if isinstance(stats, dict) else {}


def mean_value(product: Any, keys: tuple[str, ...] = ("mean", "mean_value", "average")) -> float | None:
    stats = stats_of(product)
    value = find_number(stats, keys)
    if value is not None:
        return value
    return find_number(product, keys)


def min_value(product: Any) -> float | None:
    stats = stats_of(product)
    return find_number(stats, ("min", "minimum", "min_value")) or find_number(product, ("min", "minimum", "min_value"))


def max_value(product: Any) -> float | None:
    stats = stats_of(product)
    return find_number(stats, ("max", "maximum", "max_value")) or find_number(product, ("max", "maximum", "max_value"))


def product_is_usable(product: Any, reference: datetime) -> bool:
    if not isinstance(product, dict):
        return False
    status = str(product.get("status", "")).lower()
    temporal = product_temporal(product, reference)
    return status in {"ok", "sin_actividad"} and temporal.get("temporal_validity") == "valid"


def c13_signal(product: Any, reference: datetime) -> tuple[float, dict[str, Any]]:
    if not product_is_usable(product, reference):
        return 0.0, {"usable": False}

    mean_k = mean_value(product)
    min_k = min_value(product)
    max_k = max_value(product)

    def celsius(k):
        if k is None:
            return None
        return k - 273.15 if k > 100 else k

    mean_c = celsius(mean_k)
    min_c = celsius(min_k)
    max_c = celsius(max_k)

    cold_score = 0.0
    if min_c is not None:
        cold_score = clamp((235.0 - min_c) / 35.0)
    if mean_c is not None:
        cold_score = max(cold_score, clamp((265.0 - mean_c) / 45.0) * 0.55)

    return clamp(cold_score), {
        "usable": True,
        "mean_c": round(mean_c, 2) if mean_c is not None else None,
        "min_c": round(min_c, 2) if min_c is not None else None,
        "max_c": round(max_c, 2) if max_c is not None else None,
        "signal": round(cold_score, 3),
    }


def cloud_top_signals(cloudtop: Any, reference: datetime) -> tuple[float, dict[str, Any]]:
    if not isinstance(cloudtop, dict):
        return 0.0, {"usable": False}

    height = cloudtop.get("cloudtop_height")
    pressure = cloudtop.get("cloudtop_pressure")

    usable_height = product_is_usable(height, reference)
    usable_pressure = product_is_usable(pressure, reference)

    height_m = mean_value(height)
    pressure_hpa = mean_value(pressure)

    if height_m is not None and height_m < 100:
        height_m *= 1000.0

    height_score = clamp((height_m - 8000.0) / 7000.0) if height_m is not None else 0.0
    pressure_score = clamp((350.0 - pressure_hpa) / 180.0) if pressure_hpa is not None else 0.0

    values = []
    if usable_height and height_m is not None:
        values.append(height_score)
    if usable_pressure and pressure_hpa is not None:
        values.append(pressure_score)

    signal = sum(values) / len(values) if values else 0.0

    return clamp(signal), {
        "usable_height": usable_height,
        "usable_pressure": usable_pressure,
        "mean_height_m": round(height_m, 1) if height_m is not None else None,
        "mean_pressure_hpa": round(pressure_hpa, 1) if pressure_hpa is not None else None,
        "signal": round(signal, 3),
    }


def glm_signal(glm: Any, reference: datetime) -> tuple[float, dict[str, Any]]:
    if not isinstance(glm, dict):
        return 0.0, {"usable": False}

    temporal = product_temporal(glm, reference)
    status = str(glm.get("status", "unavailable"))
    if temporal.get("temporal_validity") != "valid" or status not in {"ok", "sin_actividad"}:
        return 0.0, {
            "usable": False,
            "status": status,
            "age_minutes": temporal.get("age_minutes"),
        }

    counts = glm.get("counts")
    if not isinstance(counts, dict):
        counts = {}

    events = find_number(counts, ("events", "eventos")) or 0.0
    groups = find_number(counts, ("groups", "grupos")) or 0.0
    flashes = find_number(counts, ("flashes", "rayos", "flash")) or 0.0

    energy = find_number(glm.get("flash_energy"), ("mean", "mean_energy", "total"))
    area = find_number(glm.get("flash_area"), ("mean", "mean_area", "total"))

    count_score = clamp(math.log1p(max(events, groups * 5.0, flashes * 20.0)) / math.log1p(500.0))
    energy_score = clamp(math.log1p(max(0.0, energy or 0.0)) / math.log1p(100.0)) if energy is not None else 0.0
    signal = max(count_score, 0.7 * count_score + 0.3 * energy_score)

    return clamp(signal), {
        "usable": True,
        "status": status,
        "events": int(events),
        "groups": int(groups),
        "flashes": int(flashes),
        "flash_energy": energy,
        "flash_area": area,
        "signal": round(signal, 3),
    }


def dmw_products(dmw: Any) -> list[dict[str, Any]]:
    if not isinstance(dmw, dict):
        return []
    products = dmw.get("products")
    if not isinstance(products, list):
        return []
    return [p for p in products if isinstance(p, dict)]


def dmw_signal(dmw: Any, reference: datetime) -> tuple[float, dict[str, Any]]:
    records = dmw_products(dmw)
    usable = []
    dmwv = []

    for record in records:
        name = str(record.get("product") or record.get("source", {}).get("product") or record.get("name") or "").lower()
        temporal = product_temporal(record, reference)
        status = str(record.get("status", ""))
        if status not in {"ok", "sin_actividad"} or temporal.get("temporal_validity") != "valid":
            continue

        vector_summary = record.get("vector_summary")
        if not isinstance(vector_summary, dict):
            vector_summary = {}

        speed = find_number(record, ("mean_speed", "mean_wind_speed", "wind_speed"))
        direction = find_number(record, ("mean_direction_from", "mean_wind_direction", "wind_direction"))
        coherence = find_number(record, ("direction_resultant", "coherence", "resultant"))
        vectors = find_number(record, ("vectors", "vector_count", "total_vectors", "measurements"))

        if speed is None:
            speed = find_number(vector_summary, ("mean_speed", "mean_wind_speed", "wind_speed"))
        if direction is None:
            direction = find_number(vector_summary, ("mean_direction_from", "mean_wind_direction", "wind_direction"))
        if coherence is None:
            coherence = find_number(vector_summary, ("direction_resultant", "coherence", "resultant"))
        if vectors is None:
            vectors = find_number(vector_summary, ("vectors", "count", "n"))

        item = {
            "product": record.get("product") or record.get("source", {}).get("product") or record.get("name"),
            "age_minutes": temporal.get("age_minutes"),
            "mean_speed_ms": speed,
            "mean_direction_from_deg": direction,
            "direction_coherence": coherence,
            "vectors": int(vectors) if vectors is not None else None,
        }

        if "dmwv" in name:
            dmwv.append(item)
        else:
            usable.append(item)

    selected = usable + dmwv
    speeds = [x["mean_speed_ms"] for x in selected if x["mean_speed_ms"] is not None]
    directions = [x["mean_direction_from_deg"] for x in selected if x["mean_direction_from_deg"] is not None]
    coherences = [x["direction_coherence"] for x in selected if x["direction_coherence"] is not None]

    speed = sum(speeds) / len(speeds) if speeds else None
    coherence = sum(coherences) / len(coherences) if coherences else None

    # DMW is an atmospheric steering signal, not a surface-gust measurement.
    steering_quality = clamp((coherence or 0.0) * 0.7 + clamp((speed or 0.0) / 30.0) * 0.3)

    circular_direction = circular_mean_degrees(directions)

    return steering_quality, {
        "usable": bool(selected),
        "products": selected,
        "dmw_products": len(usable),
        "dmwv_products": len(dmwv),
        "mean_speed_ms": round(speed, 2) if speed is not None else None,
        "mean_direction_from_deg": round(sum(directions) / len(directions), 2) if directions else None,
        "mean_direction_circular_deg": round(circular_direction, 2) if circular_direction is not None else None,
        "mean_coherence": round(coherence, 3) if coherence is not None else None,
        "steering_quality": round(steering_quality, 3),
    }


def circular_mean_degrees(values: list[float]) -> float | None:
    if not values:
        return None
    radians = [math.radians(v % 360.0) for v in values]
    x = sum(math.cos(v) for v in radians)
    y = sum(math.sin(v) for v in radians)
    if x == 0 and y == 0:
        return None
    return math.degrees(math.atan2(y, x)) % 360.0


def build_goes_intelligence(state: dict[str, Any], reference: datetime, acquisition: str) -> dict[str, Any]:
    products = state.get("products") if isinstance(state.get("products"), dict) else {}
    cloudtop = state.get("cloudtop") if isinstance(state.get("cloudtop"), dict) else {}
    glm = state.get("glm")
    dmw = state.get("dmw")

    c13_score, c13 = c13_signal(products.get("abi_c13"), reference)
    c08_temporal = product_temporal(products.get("abi_c08"), reference)
    cloud_score, clouds = cloud_top_signals(cloudtop, reference)
    lightning_score, lightning = glm_signal(glm, reference)
    steering_score, winds = dmw_signal(dmw, reference)

    # C08 es un canal de vapor de agua: se conserva como diagnóstico de humedad
    # y calidad temporal, pero no se convierte artificialmente en severidad.
    moisture = {
        "usable": product_is_usable(products.get("abi_c08"), reference),
        "age_minutes": c08_temporal.get("age_minutes"),
        "status": products.get("abi_c08", {}).get("status") if isinstance(products.get("abi_c08"), dict) else "unavailable",
    }

    component_scores = [
        c13_score,
        cloud_score,
        lightning_score,
    ]
    available_scores = [value for value in component_scores if value > 0.0]

    convective = (
        0.45 * c13_score
        + 0.35 * cloud_score
        + 0.20 * lightning_score
    )
    if not available_scores:
        convective = 0.0

    quality_components = [
        product_is_usable(products.get("abi_c08"), reference),
        product_is_usable(products.get("abi_c13"), reference),
        product_is_usable(cloudtop.get("cloudtop_height"), reference),
        product_is_usable(cloudtop.get("cloudtop_pressure"), reference),
        lightning.get("usable", False),
        winds.get("usable", False),
    ]

    quality = sum(1 for item in quality_components if item) / len(quality_components)

    return {
        "version": "2.0.0",
        "motor": "ClimaAR GOES-19 Integrated Intelligence",
        "satellite": state.get("satellite", "GOES-19"),
        "reference_frame": {
            "source": "RMA10 radar_nowcast",
            "frame_actual_utc": reference.isoformat().replace("+00:00", "Z"),
        },
        "acquisition": acquisition,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_state_version": state.get("version"),
        "source_state_status": state.get("status"),
        "domain": state.get("domain"),
        "productos": {
            "abi_c08": products.get("abi_c08"),
            "abi_c13": products.get("abi_c13"),
        },
        "cloudtop": cloudtop,
        "glm": glm,
        "dmw": dmw,
        "senales": {
            "conveccion_satelital": round(clamp(convective), 3),
            "abi_c13_nubes_frías": c13,
            "abi_c08_vapor_agua": moisture,
            "cloud_top": clouds,
            "glm_rayo": lightning,
            "dmw_viento_guia": winds,
            "calidad_goes": round(quality, 3),
        },
        "raw_state": state,
    }


def apply_goes_to_intelligence(goes: dict[str, Any], radar_timestamp: datetime) -> None:
    if not INTELLIGENCE_FILE.exists():
        raise RuntimeError("No existe inteligencia_tormenta.json para integrar GOES-19.")

    intelligence = load_json(INTELLIGENCE_FILE)
    if not isinstance(intelligence, dict):
        raise RuntimeError("inteligencia_tormenta.json inválido.")

    reference = parse_timestamp(goes["reference_frame"]["frame_actual_utc"])
    if reference != radar_timestamp:
        raise RuntimeError(
            "ERROR: GOES-19 y Radar desincronizados: "
            f"goes={reference.isoformat()} radar={radar_timestamp.isoformat()}"
        )

    intelligence["goes19"] = goes

    indices = intelligence.get("indices_heuristicos")
    if not isinstance(indices, dict):
        indices = {}
        intelligence["indices_heuristicos"] = indices

    signals = goes.get("senales", {})
    convective = clamp(signals.get("conveccion_satelital"))
    lightning = clamp(signals.get("glm_rayo", {}).get("signal"))
    steering = clamp(signals.get("dmw_viento_guia", {}).get("steering_quality"))

    indices["goes19_conveccion"] = round(convective, 3)
    indices["goes19_rayo"] = round(lightning, 3)
    # DMW y calidad GOES se conservan como diagnóstico; no se convierten
    # directamente en amenaza meteorológica.

    radar_data = load_json(RADAR_FILE)
    nowcast = radar_data.get("nowcast", {})
    if not isinstance(nowcast, dict):
        nowcast = {}

    radar_direction = finite(nowcast.get("direccion_grados"))
    dmw_direction = finite(signals.get("dmw_viento_guia", {}).get("mean_direction_circular_deg"))
    if dmw_direction is None:
        dmw_direction = finite(signals.get("dmw_viento_guia", {}).get("mean_direction_from_deg"))

    direction_agreement = None
    if radar_direction is not None and dmw_direction is not None:
        difference = abs((radar_direction - dmw_direction + 180.0) % 360.0 - 180.0)
        direction_agreement = round(1.0 - difference / 180.0, 3)

    signals["dmw_viento_guia"]["radar_direction_agreement"] = direction_agreement

    hazards = indices.get("peligros")
    if not isinstance(hazards, dict):
        hazards = {}
        indices["peligros"] = hazards

    old_storm = clamp(indices.get("tormenta"))
    old_strength = clamp(indices.get("fortalecimiento"))

    # GOES aporta una fuente independiente; nunca reemplaza Radar/ML/ECMWF.
    # El máximo evita que una observación satelital degrade una señal terrestre.
    indices["tormenta"] = round(max(old_storm, 0.75 * old_storm + 0.25 * convective), 3)
    indices["fortalecimiento"] = round(max(old_strength, 0.80 * old_strength + 0.20 * convective), 3)
    hazards["rayo"] = round(max(clamp(hazards.get("rayo")), lightning), 3)

    intelligence["fuentes_integradas"] = [
        "RMA10",
        "ECMWF",
        "SAZB",
        "GOES-19 ABI C08",
        "GOES-19 ABI C13",
        "GOES-19 Cloud Top Height",
        "GOES-19 Cloud Top Pressure",
        "GOES-19 GLM",
        "GOES-19 DMW",
        "GOES-19 DMWV",
    ]

    intelligence["sincronizacion_goes19"] = {
        "radar_frame_utc": radar_timestamp.isoformat(),
        "goes_reference_frame_utc": reference.isoformat(),
        "sincronizado": True,
    }

    save_json(INTELLIGENCE_FILE, intelligence)


def main() -> None:
    radar_timestamp = load_radar_timestamp()
    print("Timestamp real del radar:", radar_timestamp.isoformat())

    validate_radar_nowcast_ml(radar_timestamp)

    radar_data = load_json(RADAR_FILE)
    nowcast = radar_data.get("nowcast", {})
    principal = nowcast.get("nucleo_principal_id")
    bahia = nowcast.get("bahia_blanca", {})
    if principal is None:
        raise RuntimeError("ERROR: Intelligence recibió un Nowcast sin principal.")
    if isinstance(bahia, dict) and bahia.get("nucleo_principal_id") is not None and str(bahia.get("nucleo_principal_id")) != str(principal):
        raise RuntimeError(
            "ERROR: Intelligence recibió principales inconsistentes: "
            f"nowcast={principal} bahia={bahia.get('nucleo_principal_id')}"
        )

    meteo_data = load_json(METEO_FILE)
    meteo_radar = meteo_data.get("radar", {})
    if not isinstance(meteo_radar, dict):
        raise RuntimeError("ERROR: estado_meteorologico.radar inválido.")

    meteo_frame = parse_timestamp(meteo_radar.get("frame_actual_utc"))
    if meteo_frame != radar_timestamp:
        raise RuntimeError(
            "ERROR: meteorología y Radar están desincronizados: "
            f"radar={radar_timestamp.isoformat()} meteo={meteo_frame.isoformat() if meteo_frame else None}"
        )

    meteo_principal = meteo_radar.get("nucleo_principal_id")
    if meteo_principal is None:
        raise RuntimeError("ERROR: meteorología no contiene nucleo_principal_id.")
    if str(meteo_principal) != str(principal):
        raise RuntimeError(
            "ERROR: meteorología recibió un principal distinto al Radar: "
            f"radar={principal} meteo={meteo_principal}"
        )

    goes_state, acquisition = run_goes_state()
    if goes_state is not None:
        goes_intelligence = build_goes_intelligence(
            goes_state,
            radar_timestamp,
            acquisition,
        )
        save_json(GOES_INTELLIGENCE_FILE, goes_intelligence)
        print(
            "GOES-19 integrado y sincronizado: "
            f"frame={radar_timestamp.isoformat()} "
            f"adquisición={acquisition}"
        )
    else:
        goes_intelligence = {
            "version": "2.0.0",
            "motor": "ClimaAR GOES-19 Integrated Intelligence",
            "satellite": "GOES-19",
            "reference_frame": {
                "source": "RMA10 radar_nowcast",
                "frame_actual_utc": radar_timestamp.isoformat().replace("+00:00", "Z"),
            },
            "acquisition": "unavailable",
            "senales": {
                "conveccion_satelital": 0.0,
                "calidad_goes": 0.0,
            },
        }
        save_json(GOES_INTELLIGENCE_FILE, goes_intelligence)
        print("AVISO: GOES-19 no disponible; Intelligence continúa sin señal GOES.")

    sys.path.insert(0, str(BASE / "scripts"))
    engine = importlib.import_module("tormenta_intelligence")

    original_now_dt = getattr(engine, "now_dt", None)
    original_environment_function = getattr(engine, "environment_along_projection", None)

    if original_now_dt is None:
        raise RuntimeError("tormenta_intelligence.py no expone la función now_dt().")
    if original_environment_function is None:
        raise RuntimeError("tormenta_intelligence.py no expone environment_along_projection().")

    try:
        engine.now_dt = lambda: radar_timestamp
        engine.environment_along_projection = lambda points, projection: environment_along_projection(
            points,
            projection,
            radar_timestamp,
        )

        print("Storm Engine ejecutándose con el timestamp del frame Radar.")
        print("Meteorología de trayectoria: interpolación temporal + espacial ECMWF.")
        result = engine.main()
    finally:
        engine.now_dt = original_now_dt
        engine.environment_along_projection = original_environment_function

    apply_goes_to_intelligence(goes_intelligence, radar_timestamp)

    print("GOES-19 aplicado al Storm Engine.")
    print("Storm Engine finalizado correctamente.")
    return result


if __name__ == "__main__":
    main()
