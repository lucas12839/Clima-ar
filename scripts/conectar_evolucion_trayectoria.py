import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

RADAR_FILE = BASE_DIR / "data" / "radar" / "radar_nowcast.json"
EVOLUTION_FILE = (
    BASE_DIR
    / "data"
    / "radar"
    / "seguimiento"
    / "prediccion_evolucion.json"
)

RADAR_LAT = -38.7342
RADAR_LON = -62.1654
RADAR_RANGE_KM = 240.0
EARTH_RADIUS_KM = 6371.0088

HORIZONTES = (5, 10, 20, 30)


def finite(value, default=None):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


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


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"JSON inválido: {path}")

    return data


def save_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_suffix(path.suffix + ".tmp")

    temp.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    temp.replace(path)


def destination_point(lat, lon, distance_km, bearing_deg):
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    bearing = math.radians(bearing_deg)

    angular_distance = distance_km / EARTH_RADIUS_KM

    sin_lat2 = (
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1)
        * math.sin(angular_distance)
        * math.cos(bearing)
    )

    lat2 = math.asin(
        max(-1.0, min(1.0, sin_lat2))
    )

    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance)
        - math.sin(lat1) * math.sin(lat2),
    )

    return (
        math.degrees(lat2),
        ((math.degrees(lon2) + 540.0) % 360.0) - 180.0,
    )


def pixel_to_latlon(x, y, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Dimensiones de imagen inválidas")

    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0

    radius_px = min(width, height) / 2.0

    dx = float(x) - center_x
    dy = center_y - float(y)

    distance_px = math.hypot(dx, dy)

    distance_km = (
        distance_px / radius_px
    ) * RADAR_RANGE_KM

    if distance_px <= 1e-9:
        bearing = 0.0
    else:
        bearing = (
            math.degrees(
                math.atan2(dx, dy)
            )
            + 360.0
        ) % 360.0

    lat, lon = destination_point(
        RADAR_LAT,
        RADAR_LON,
        distance_km,
        bearing,
    )

    return (
        lat,
        lon,
        distance_km,
        bearing,
    )


def find_current_nucleus(nowcast, track_id):
    nuclei = nowcast.get("nucleos", [])

    if not isinstance(nuclei, list):
        return None

    for nucleus in nuclei:
        if not isinstance(nucleus, dict):
            continue

        if nucleus.get("track_id") == track_id:
            return nucleus

    return None


def resolve_principal_track(nowcast, evolution):
    principal_id = nowcast.get(
        "nucleo_principal_id"
    )

    current_nucleus = find_current_nucleus(
        nowcast,
        principal_id,
    )

    if current_nucleus:
        track_id = current_nucleus.get(
            "track_id"
        )

        if track_id is not None:
            return track_id

    nuclei = evolution.get("nucleos", [])

    candidates = []

    if isinstance(nuclei, list):
        for item in nuclei:
            if not isinstance(item, dict):
                continue

            if not item.get(
                "pronostico_disponible",
                False,
            ):
                continue

            track_id = item.get("track_id")

            confidence = finite(
                item.get(
                    "confianza_prediccion"
                ),
                0.0,
            )

            if track_id is not None:
                candidates.append(
                    (
                        confidence,
                        track_id,
                    )
                )

    if not candidates:
        return None

    candidates.sort(
        reverse=True
    )

    return candidates[0][1]


def find_prediction(evolution, track_id):
    nuclei = evolution.get("nucleos", [])

    if not isinstance(nuclei, list):
        return None

    for item in nuclei:
        if not isinstance(item, dict):
            continue

        if item.get("track_id") == track_id:
            return item

    return None


def main():
    radar = load_json(RADAR_FILE)
    evolution = load_json(EVOLUTION_FILE)

    nowcast = radar.get("nowcast")

    if not isinstance(nowcast, dict):
        raise RuntimeError(
            "radar_nowcast.json no contiene nowcast"
        )

    radar_timestamp = parse_dt(
        nowcast.get("frame_actual_utc")
        or evolution.get("radar_timestamp")
        or radar.get("actualizado")
    )

    if radar_timestamp is None:
        raise RuntimeError(
            "No se pudo determinar el timestamp real del radar"
        )

    image = radar.get("imagen", {})

    width = int(
        finite(
            image.get("ancho"),
            512,
        )
    )

    height = int(
        finite(
            image.get("alto"),
            512,
        )
    )

    track_id = resolve_principal_track(
        nowcast,
        evolution,
    )

    if track_id is None:
        raise RuntimeError(
            "No se pudo determinar el track principal"
        )

    prediction = find_prediction(
        evolution,
        track_id,
    )

    if prediction is None:
        raise RuntimeError(
            f"No existe predicción para track_id={track_id}"
        )

    horizons = prediction.get(
        "horizontes"
    )

    if not isinstance(horizons, list):
        raise RuntimeError(
            "La predicción no contiene horizontes"
        )

    projections = {}

    for horizon in HORIZONTES:
        selected = None

        for item in horizons:
            if not isinstance(item, dict):
                continue

            minutes = int(
                finite(
                    item.get(
                        "horizonte_minutos"
                    ),
                    -1,
                )
            )

            if minutes == horizon:
                selected = item
                break

        if selected is None:
            continue

        x = finite(
            selected.get("x_estimado")
        )

        y = finite(
            selected.get("y_estimado")
        )

        if x is None or y is None:
            continue

        (
            lat,
            lon,
            distance_km,
            bearing,
        ) = pixel_to_latlon(
            x,
            y,
            width,
            height,
        )

        timestamp = (
            radar_timestamp
            + timedelta(minutes=horizon)
        )

        projections[str(horizon)] = {
            "minutos": horizon,
            "timestamp_utc": timestamp.isoformat(),
            "x": round(x, 3),
            "y": round(y, 3),
            "latitud": round(lat, 6),
            "longitud": round(lon, 6),
            "distancia_radar_km": round(
                distance_km,
                3,
            ),
            "rumbo_radar_grados": round(
                bearing,
                2,
            ),
            "track_id": track_id,
            "incertidumbre_pixeles": finite(
                selected.get(
                    "incertidumbre_pixeles"
                )
            ),
            "confianza_prediccion": finite(
                selected.get(
                    "confianza_prediccion"
                )
            ),
            "direccion_estimada": selected.get(
                "direccion_estimada"
            ),
        }

    if not projections:
        raise RuntimeError(
            "No se pudo generar ninguna proyección"
        )

    nowcast["proyecciones"] = projections

    nowcast[
        "proyecciones_metadata"
    ] = {
        "fuente": "prediccion_evolucion.json",
        "track_id_principal": track_id,
        "radar_timestamp_utc": radar_timestamp.isoformat(),
        "horizontes_generados": [
            int(key)
            for key in sorted(
                projections,
                key=lambda value: int(value),
            )
        ],
        "geometria": {
            "radar_lat": RADAR_LAT,
            "radar_lon": RADAR_LON,
            "alcance_km": RADAR_RANGE_KM,
            "imagen_ancho": width,
            "imagen_alto": height,
        },
    }

    radar["nowcast"] = nowcast

    save_json_atomic(
        RADAR_FILE,
        radar,
    )

    print(
        "OK: proyecciones "
        f"{'/'.join(projections.keys())} "
        f"generadas para track {track_id}"
    )

    for key in sorted(
        projections,
        key=lambda value: int(value),
    ):
        item = projections[key]

        print(
            f"{key} min -> "
            f"{item['timestamp_utc']} | "
            f"{item['latitud']}, "
            f"{item['longitud']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
