from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parents[1]

RADAR_FILE = BASE / "data" / "radar" / "radar_nowcast.json"
EVOLUTION_FILE = (
    BASE
    / "data"
    / "radar"
    / "seguimiento"
    / "prediccion_evolucion.json"
)

RADAR_LAT = -38.7342
RADAR_LON = -62.1654
RADAR_RANGE_KM = 240.0

DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512

HORIZONS = (5, 10, 20, 30)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"JSON inválido: {path}")

    return data


def parse_datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"Timestamp inválido: {value!r}")

    text = value.strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    dt = datetime.fromisoformat(text)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def iso_utc(dt: datetime) -> str:
    return (
        dt.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    earth_radius_km = 6371.0088

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dlon / 2) ** 2
    )

    a = min(1.0, max(0.0, a))

    return (
        2
        * earth_radius_km
        * math.asin(math.sqrt(a))
    )


def bearing_deg(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dlambda = math.radians(lon2 - lon1)

    y = math.sin(dlambda) * math.cos(phi2)

    x = (
        math.cos(phi1) * math.sin(phi2)
        - math.sin(phi1)
        * math.cos(phi2)
        * math.cos(dlambda)
    )

    bearing = math.degrees(math.atan2(y, x))

    return bearing % 360.0


def destination_point(
    lat: float,
    lon: float,
    distance_km: float,
    bearing: float,
) -> tuple[float, float]:
    earth_radius_km = 6371.0088

    angular_distance = (
        distance_km / earth_radius_km
    )

    theta = math.radians(bearing)

    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)

    sin_phi2 = (
        math.sin(phi1)
        * math.cos(angular_distance)
        + math.cos(phi1)
        * math.sin(angular_distance)
        * math.cos(theta)
    )

    sin_phi2 = min(1.0, max(-1.0, sin_phi2))

    phi2 = math.asin(sin_phi2)

    y = (
        math.sin(theta)
        * math.sin(angular_distance)
        * math.cos(phi1)
    )

    x = (
        math.cos(angular_distance)
        - math.sin(phi1) * math.sin(phi2)
    )

    lambda2 = (
        lambda1
        + math.atan2(y, x)
    )

    return (
        math.degrees(phi2),
        (
            math.degrees(lambda2) + 540.0
        ) % 360.0 - 180.0,
    )


def get_image_geometry(
    radar_nowcast: dict[str, Any],
) -> tuple[int, int, float, float]:
    """
    Obtiene dimensiones y centro geométrico.

    El radar RMA10 utiliza una representación cuadrada
    centrada en el radar y con alcance nominal de 240 km.
    """

    width = DEFAULT_WIDTH
    height = DEFAULT_HEIGHT

    image_info = radar_nowcast.get("imagen")

    if isinstance(image_info, dict):
        width = int(
            image_info.get("ancho", width)
        )
        height = int(
            image_info.get("alto", height)
        )

    if width <= 0 or height <= 0:
        width = DEFAULT_WIDTH
        height = DEFAULT_HEIGHT

    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0

    return (
        width,
        height,
        center_x,
        center_y,
    )


def pixel_to_latlon(
    x: float,
    y: float,
    width: int,
    height: int,
    center_x: float,
    center_y: float,
) -> tuple[float, float, float, float]:
    """
    Convierte coordenadas de imagen RMA10 a lat/lon.

    Convención:
      X positivo = Este
      Y positivo hacia abajo en imagen
      Norte = -Y

    El radio geométrico corresponde a 240 km.
    """

    radius_px = min(
        center_x,
        center_y,
        width - 1 - center_x,
        height - 1 - center_y,
    )

    if radius_px <= 0:
        raise ValueError(
            "Geometría de imagen inválida."
        )

    east_km = (
        (float(x) - center_x)
        / radius_px
        * RADAR_RANGE_KM
    )

    north_km = (
        (center_y - float(y))
        / radius_px
        * RADAR_RANGE_KM
    )

    distance_km = math.hypot(
        east_km,
        north_km,
    )

    if distance_km == 0:
        return (
            RADAR_LAT,
            RADAR_LON,
            0.0,
            0.0,
        )

    bearing = (
        math.degrees(
            math.atan2(
                east_km,
                north_km,
            )
        )
        % 360.0
    )

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


def get_principal_nucleus(
    nowcast: dict[str, Any],
) -> dict[str, Any] | None:
    nuclei = nowcast.get("nucleos")

    if not isinstance(nuclei, list):
        nuclei = []

    principal_id = nowcast.get(
        "nucleo_principal_id"
    )

    if principal_id is not None:
        for nucleus in nuclei:
            if not isinstance(nucleus, dict):
                continue

            if nucleus.get("nucleo_id") == principal_id:
                return nucleus

            if str(
                nucleus.get("nucleo_id")
            ) == str(principal_id):
                return nucleus

    # Fallback: buscar núcleo con mayor intensidad/confianza.
    candidates = [
        nucleus
        for nucleus in nuclei
        if isinstance(nucleus, dict)
        and nucleus.get("track_id") is not None
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: (
            float(
                item.get(
                    "intensidad_max",
                    0,
                )
                or 0
            ),
            float(
                item.get(
                    "confianza",
                    0,
                )
                or 0
            ),
            int(
                item.get(
                    "pixeles",
                    0,
                )
                or 0
            ),
        ),
    )


def get_principal_track(
    nowcast: dict[str, Any],
    evolution: dict[str, Any],
) -> dict[str, Any]:
    nuclei = evolution.get("nucleos")

    if not isinstance(nuclei, list):
        nuclei = []

    principal = get_principal_nucleus(
        nowcast
    )

    if principal is not None:
        principal_track_id = principal.get(
            "track_id"
        )

        if principal_track_id is not None:
            for track in nuclei:
                if not isinstance(track, dict):
                    continue

                if track.get("track_id") == principal_track_id:
                    return track

                if str(
                    track.get("track_id")
                ) == str(principal_track_id):
                    return track

    # Fallback seguro: buscar predicción activa
    # con mayor confianza.
    active = [
        track
        for track in nuclei
        if isinstance(track, dict)
        and track.get("activo") is True
        and track.get("pronostico_disponible") is True
    ]

    if not active:
        active = [
            track
            for track in nuclei
            if isinstance(track, dict)
            and track.get("pronostico_disponible") is True
        ]

    if not active:
        raise RuntimeError(
            "No existe ninguna predicción de evolución disponible."
        )

    return max(
        active,
        key=lambda item: float(
            item.get(
                "confianza_prediccion",
                0,
            )
            or 0
        ),
    )


def get_horizon_prediction(
    track: dict[str, Any],
    horizon: int,
) -> dict[str, Any] | None:
    horizons = track.get("horizontes")

    if not isinstance(horizons, list):
        return None

    for item in horizons:
        if not isinstance(item, dict):
            continue

        value = item.get(
            "horizonte_minutos"
        )

        if value is None:
            continue

        try:
            minutes = int(round(float(value)))
        except (
            TypeError,
            ValueError,
        ):
            continue

        if minutes == horizon:
            return item

    return None


def finite_number(
    value: Any,
) -> bool:
    try:
        number = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return False

    return math.isfinite(number)


def build_projection(
    prediction: dict[str, Any],
    horizon: int,
    radar_timestamp: datetime,
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    track: dict[str, Any],
) -> dict[str, Any]:
    x = prediction.get("x_estimado")
    y = prediction.get("y_estimado")

    if not finite_number(x) or not finite_number(y):
        raise ValueError(
            f"Proyección {horizon} min "
            "sin coordenadas X/Y válidas."
        )

    x = float(x)
    y = float(y)

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
        center_x,
        center_y,
    )

    timestamp = (
        radar_timestamp
        + timedelta(minutes=horizon)
    )

    projection = {
        "minutos": horizon,
        "timestamp_utc": iso_utc(timestamp),
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
        "track_id": track.get(
            "track_id"
        ),
        "incertidumbre_pixeles": (
            prediction.get(
                "incertidumbre_pixeles"
            )
        ),
        "confianza_prediccion": (
            prediction.get(
                "confianza_prediccion"
            )
        ),
        "direccion_estimada": (
            prediction.get(
                "direccion_estimada"
            )
        ),
        "velocidad_estimada_pixeles_min": (
            prediction.get(
                "velocidad_estimada_pixeles_min"
            )
        ),
        "rumbo_estimado_grados": (
            prediction.get(
                "rumbo_estimado_grados"
            )
        ),
        "calibracion_muestras": (
            prediction.get(
                "calibracion_muestras"
            )
        ),
        "correccion_bias_x_pixeles": (
            prediction.get(
                "correccion_bias_x_pixeles"
            )
        ),
        "correccion_bias_y_pixeles": (
            prediction.get(
                "correccion_bias_y_pixeles"
            )
        ),
    }

    return projection


def atomic_write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(
                data,
                fh,
                ensure_ascii=False,
                indent=2,
            )
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())

        os.replace(
            temp_name,
            path,
        )

    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    radar = load_json(
        RADAR_FILE
    )

    evolution = load_json(
        EVOLUTION_FILE
    )

    nowcast = radar.get("nowcast")

    if not isinstance(nowcast, dict):
        raise RuntimeError(
            "radar_nowcast.json no contiene un bloque nowcast válido."
        )

    radar_timestamp_value = nowcast.get(
        "frame_actual_utc"
    )

    if not radar_timestamp_value:
        radar_timestamp_value = radar.get(
            "actualizado"
        )

    if not radar_timestamp_value:
        radar_timestamp_value = evolution.get(
            "radar_timestamp"
        )

    radar_timestamp = parse_datetime(
        radar_timestamp_value
    )

    evolution_timestamp = evolution.get(
        "radar_timestamp"
    )

    if evolution_timestamp:
        evolution_dt = parse_datetime(
            evolution_timestamp
        )

        delta_seconds = abs(
            (
                evolution_dt
                - radar_timestamp
            ).total_seconds()
        )

        if delta_seconds > 1:
            raise RuntimeError(
                "Timestamp desincronizado entre "
                "radar_nowcast y prediccion_evolucion: "
                f"{radar_timestamp.isoformat()} vs "
                f"{evolution_dt.isoformat()}"
            )

    width, height, center_x, center_y = (
        get_image_geometry(nowcast)
    )

    track = get_principal_track(
        nowcast,
        evolution,
    )

    projections: dict[str, Any] = {}

    for horizon in HORIZONS:
        prediction = get_horizon_prediction(
            track,
            horizon,
        )

        if prediction is None:
            raise RuntimeError(
                "No existe predicción para "
                f"{horizon} minutos en track "
                f"{track.get('track_id')}."
            )

        projection = build_projection(
            prediction=prediction,
            horizon=horizon,
            radar_timestamp=radar_timestamp,
            width=width,
            height=height,
            center_x=center_x,
            center_y=center_y,
            track=track,
        )

        projections[str(horizon)] = projection

    nowcast["proyecciones"] = projections

    nowcast["proyecciones_metadata"] = {
        "fuente": "prediccion_evolucion",
        "track_id": track.get(
            "track_id"
        ),
        "radar_timestamp_utc": iso_utc(
            radar_timestamp
        ),
        "geometria": {
            "ancho": width,
            "alto": height,
            "centro_x": center_x,
            "centro_y": center_y,
            "radar_latitud": RADAR_LAT,
            "radar_longitud": RADAR_LON,
            "alcance_km": RADAR_RANGE_KM,
        },
        "horizontes_minutos": list(
            HORIZONS
        ),
    }

    radar["nowcast"] = nowcast

    atomic_write_json(
        RADAR_FILE,
        radar,
    )

    print(
        "OK: proyecciones "
        "5/10/20/30 generadas "
        f"para track {track.get('track_id')}"
    )

    for horizon in HORIZONS:
        projection = projections[
            str(horizon)
        ]

        print(
            f"{horizon:>2} min -> "
            f"{projection['timestamp_utc']} | "
            f"x={projection['x']} "
            f"y={projection['y']} | "
            f"lat={projection['latitud']} "
            f"lon={projection['longitud']}"
        )


if __name__ == "__main__":
    main()
