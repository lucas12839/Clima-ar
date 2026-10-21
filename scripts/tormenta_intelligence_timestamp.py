from __future__ import annotations

import importlib
import json
import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]

RADAR_FILE = BASE / "data" / "radar" / "radar_nowcast.json"
ML_FILE = BASE / "data" / "radar" / "seguimiento" / "prediccion_ml_rma10.json"
METEO_FILE = BASE / "data" / "ia" / "estado_meteorologico.json"


def parse_timestamp(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"JSON inválido: {path}") from exc


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
    """Bloquea Intelligence ante cualquier desincronización Radar/Nowcast/ML."""
    radar_data = load_json(RADAR_FILE)
    nowcast = radar_data.get("nowcast")
    if not isinstance(nowcast, dict):
        raise RuntimeError("ERROR: falta el objeto nowcast.")

    nowcast_value = nowcast.get("frame_actual_utc")
    if not nowcast_value:
        raise RuntimeError("ERROR: Nowcast no contiene frame_actual_utc.")

    nowcast_timestamp = parse_timestamp(str(nowcast_value))
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


def parse_row_time(value):
    if not value:
        return None
    try:
        return parse_timestamp(str(value))
    except Exception:
        return None


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def temporal_interpolation(rows, target):
    """Interpola una serie horaria al instante exacto solicitado."""
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

    if before is None:
        result = dict(valid[0][1])
        result["tiempo_utc"] = target.isoformat()
        result["interpolacion_temporal"] = False
        return result

    if after is None:
        result = dict(valid[-1][1])
        result["tiempo_utc"] = target.isoformat()
        result["interpolacion_temporal"] = False
        return result

    t0, r0 = before
    t1, r1 = after
    total = (t1 - t0).total_seconds()
    alpha = (target - t0).total_seconds() / total if total > 0 else 0.0
    result = dict(r0)

    keys = set(r0) | set(r1)
    for key in keys:
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
    """Obtiene el ambiente meteorológico en la posición y hora de cada proyección."""
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


def main() -> None:
    radar_timestamp = load_radar_timestamp()
    print("Timestamp real del radar:", radar_timestamp.isoformat())

    validate_radar_nowcast_ml(radar_timestamp)

    radar_data = load_json(RADAR_FILE)
    nowcast = radar_data.get("nowcast", {})
    principal = nowcast.get("nucleo_principal_id")
    bahia = nowcast.get("bahia_blanca", {})
    if principal is None:
        raise RuntimeError(
            "ERROR: Intelligence recibió un Nowcast sin principal."
        )
    if (
        isinstance(bahia, dict)
        and bahia.get("nucleo_principal_id") is not None
        and str(bahia.get("nucleo_principal_id")) != str(principal)
    ):
        raise RuntimeError(
            "ERROR: Intelligence recibió principales inconsistentes: "
            f"nowcast={principal} bahia={bahia.get('nucleo_principal_id')}"
        )

    meteo_data = load_json(METEO_FILE)
    meteo_radar = meteo_data.get("radar", {})
    if not isinstance(meteo_radar, dict):
        raise RuntimeError(
            "ERROR: estado_meteorologico.radar inválido."
        )

    meteo_frame = parse_timestamp(
        meteo_radar.get("frame_actual_utc")
    )
    if meteo_frame is None or meteo_frame != radar_timestamp:
        raise RuntimeError(
            "ERROR: meteorología y Radar están desincronizados: "
            f"radar={radar_timestamp.isoformat()} "
            f"meteo={meteo_radar.get('frame_actual_utc')}"
        )

    meteo_principal = meteo_radar.get("nucleo_principal_id")
    if meteo_principal is None:
        raise RuntimeError(
            "ERROR: meteorología no contiene nucleo_principal_id."
        )
    if str(meteo_principal) != str(principal):
        raise RuntimeError(
            "ERROR: meteorología recibió un principal distinto al Radar: "
            f"radar={principal} meteo={meteo_principal}"
        )

    sys.path.insert(0, str(BASE / "scripts"))
    engine = importlib.import_module("tormenta_intelligence")

    original_now_dt = getattr(engine, "now_dt", None)
    original_environment_function = getattr(engine, "environment_along_projection", None)

    if original_now_dt is None:
        raise RuntimeError("tormenta_intelligence.py no expone la función now_dt().")
    if original_environment_function is None:
        raise RuntimeError(
            "tormenta_intelligence.py no expone environment_along_projection()."
        )

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

    print("Storm Engine finalizado correctamente.")
    return result


if __name__ == "__main__":
    main()
