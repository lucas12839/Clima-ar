from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parents[1]

RADAR_FILE = (
    BASE
    / "data"
    / "radar"
    / "radar_nowcast.json"
)

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
        raise ValueError(
            f"Timestamp inválido: {value!r}"
        )

    text = value.strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    dt = datetime.fromisoformat(text)

    if dt.tzinfo is None:
        dt = dt.replace(
            tzinfo=timezone.utc
        )

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
        math.sin(dlat / 2.0) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dlon / 2.0) ** 2
    )

    a = min(1.0, max(0.0, a))

    return (
        2.0
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

    y = (
        math.sin(dlambda)
        * math.cos(phi2)
    )

    x = (
        math.cos(phi1)
        * math.sin(phi2)
        - math.sin(phi1)
        * math.cos(phi2)
        * math.cos(dlambda)
    )

    return math.degrees(
        math.atan2(y, x)
    ) % 360.0


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

    sin_phi2 = min(
        1.0,
        max(-1.0, sin_phi2),
    )

    phi2 = math.asin(sin_phi2)

    y = (
        math.sin(theta)
        * math.sin(angular_distance)
        * math.cos(phi1)
    )

    x = (
        math.cos(angular_distance)
        - math.sin(phi1)
        * math.sin(phi2)
    )

    lambda2 = (
        lambda1
        + math.atan2(y, x)
    )

    return (
        math.degrees(phi2),
        (
            math.degrees(lambda2)
            + 540.0
        ) % 360.0 - 180.0,
    )


def get_image_geometry(
    nowcast: dict[str, Any],
) -> tuple[int, int, float, float]:
    width = DEFAULT_WIDTH
    height = DEFAULT_HEIGHT

    image_info = nowcast.get("imagen")

    if isinstance(image_info, dict):
        try:
            width = int(
                image_info.get(
                    "ancho",
                    width,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            width = DEFAULT_WIDTH

        try:
            height = int(
                image_info.get(
                    "alto",
                    height,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            height = DEFAULT_HEIGHT

    if width <= 0 or height <= 0:
        width = DEFAULT_WIDTH
        height = DEFAULT_HEIGHT

    center_x = (
        width - 1
    ) / 2.0

    center_y = (
        height - 1
    ) / 2.0

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
) -> tuple[
    float,
    float,
    float,
    float,
]:
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


def get_bahia_principal_track_id(
    nowcast: dict[str, Any],
) -> Any:
    """Devuelve el único principal operativo ya definido por Tracking."""
    return nowcast.get("nucleo_principal_id")


def get_radar_principal_track_id(
    nowcast: dict[str, Any],
) -> Any:
    return nowcast.get("nucleo_principal_id")


def find_track(
    evolution: dict[str, Any],
    track_id: Any,
) -> dict[str, Any] | None:
    if track_id is None:
        return None

    nuclei = evolution.get(
        "nucleos"
    )

    if not isinstance(
        nuclei,
        list,
    ):
        return None

    for track in nuclei:
        if not isinstance(
            track,
            dict,
        ):
            continue

        current = track.get(
            "track_id"
        )

        if current == track_id:
            return track

        if str(current) == str(track_id):
            return track

    return None


def select_principal_track(
    nowcast: dict[str, Any],
    evolution: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Obtiene exclusivamente el principal definido por Tracking/Nowcast.

    Esta etapa no puede cambiar de track ni usar un fallback.
    """
    principal_id = nowcast.get("nucleo_principal_id")
    if principal_id is None:
        raise RuntimeError(
            "ERROR: Nowcast no contiene nucleo_principal_id."
        )

    candidate_id = nowcast.get("track_principal_candidato_id")
    if candidate_id is not None and str(candidate_id) != str(principal_id):
        raise RuntimeError(
            "ERROR: Nowcast tiene principales inconsistentes: "
            f"nucleo_principal_id={principal_id} "
            f"track_principal_candidato_id={candidate_id}."
        )

    track = find_track(evolution, principal_id)
    if track is None:
        raise RuntimeError(
            "ERROR: el principal "
            f"{principal_id} no existe en prediccion_evolucion.json."
        )

    if (
        track.get("pronostico_disponible") is not True
        and track.get("motivo") != "SIN_MOVIMIENTO_VALIDO"
    ):
        raise RuntimeError(
            "ERROR: el principal "
            f"{principal_id} tiene un estado de evolución no reconocido: "
            f"{track.get('motivo')}."
        )

    return track, "tracking_principal"


def get_horizon_prediction(
    track: dict[str, Any],
    horizon: int,
) -> dict[str, Any] | None:
    horizons = track.get(
        "horizontes"
    )

    if not isinstance(
        horizons,
        list,
    ):
        return None

    for item in horizons:
        if not isinstance(
            item,
            dict,
        ):
            continue

        value = item.get(
            "horizonte_minutos"
        )

        if value is None:
            continue

        try:
            minutes = int(
                round(float(value))
            )
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

    return math.isfinite(
        number
    )


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
    x = prediction.get(
        "x_estimado"
    )

    y = prediction.get(
        "y_estimado"
    )

    if not finite_number(
        x
    ) or not finite_number(
        y
    ):
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
        + timedelta(
            minutes=horizon
        )
    )

    return {
        "minutos": horizon,
        "timestamp_utc": iso_utc(
            timestamp
        ),
        "x": round(x, 3),
        "y": round(y, 3),
        "latitud": round(
            lat,
            6,
        ),
        "longitud": round(
            lon,
            6,
        ),
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
            os.fsync(
                fh.fileno()
            )

        os.replace(
            temp_name,
            path,
        )

    except Exception:
        try:
            os.unlink(
                temp_name
            )
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

    nowcast = radar.get(
        "nowcast"
    )

    if not isinstance(
        nowcast,
        dict,
    ):
        raise RuntimeError(
            "radar_nowcast.json no contiene "
            "un bloque nowcast válido."
        )

    radar_timestamp_value = (
        nowcast.get(
            "frame_actual_utc"
        )
        or radar.get(
            "actualizado"
        )
        or evolution.get(
            "radar_timestamp"
        )
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

    (
        width,
        height,
        center_x,
        center_y,
    ) = get_image_geometry(
        nowcast
    )

    (
        track,
        selection_source,
    ) = select_principal_track(
        nowcast,
        evolution,
    )

    track_id = track.get(
        "track_id"
    )

    projections: dict[str, Any] = {}

    forecast_available = (
        track.get("pronostico_disponible") is True
    )

    if forecast_available:
        for horizon in HORIZONS:
            prediction = get_horizon_prediction(
                track,
                horizon,
            )

            if prediction is None:
                raise RuntimeError(
                    f"No existe predicción para "
                    f"{horizon} minutos en track "
                    f"{track_id}."
                )

            projections[
                str(horizon)
            ] = build_projection(
                prediction=prediction,
                horizon=horizon,
                radar_timestamp=radar_timestamp,
                width=width,
                height=height,
                center_x=center_x,
                center_y=center_y,
                track=track,
            )

    # El mismo principal de Tracking atraviesa toda la trayectoria.
    if str(track_id) != str(nowcast.get("nucleo_principal_id")):
        raise RuntimeError(
            "ERROR: trayectoria intentó cambiar el principal: "
            f"nowcast={nowcast.get('nucleo_principal_id')} "
            f"trayectoria={track_id}"
        )

    nowcast["track_principal_id"] = track_id
    nowcast["track_principal_candidato_id"] = track_id

    nowcast[
        "proyecciones"
    ] = projections

    nowcast[
        "proyecciones_metadata"
    ] = {
        "fuente": (
            "prediccion_evolucion"
        ),
        "objetivo": (
            "bahia_blanca"
        ),
        "track_id": track_id,
        "selection_source": (
            selection_source
        ),
        "pronostico_disponible": forecast_available,
        "motivo_no_disponible": (
            None
            if forecast_available
            else track.get("motivo")
        ),
        "radar_principal_track_id": (
            get_radar_principal_track_id(
                nowcast
            )
        ),
        "bahia_eta_track_id": (
            get_bahia_principal_track_id(
                nowcast
            )
        ),
        "radar_timestamp_utc": (
            iso_utc(
                radar_timestamp
            )
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

    radar[
        "nowcast"
    ] = nowcast

    atomic_write_json(
        RADAR_FILE,
        radar,
    )

    # Verificación interna adicional:
    # las cuatro proyecciones deben pertenecer
    # al mismo track seleccionado.
    if forecast_available:
        for horizon in HORIZONS:
            projection = projections[
                str(horizon)
            ]

            if str(
                projection.get("track_id")
            ) != str(track_id):
                raise RuntimeError(
                    "Inconsistencia interna: "
                    f"proyección {horizon} no pertenece "
                    f"al track seleccionado {track_id}."
                )

    print(
        "OK: trayectoria sincronizada con "
        f"track {track_id}"
    )

    print(
        f"Origen selección: {selection_source}"
    )

    print(
        "Track Radar principal:",
        get_radar_principal_track_id(
            nowcast
        ),
    )

    print(
        "Track ETA Bahía:",
        get_bahia_principal_track_id(
            nowcast
        ),
    )

    if forecast_available:
        for horizon in HORIZONS:
            projection = projections[
                str(horizon)
            ]

            print(
                f"{horizon:>2} min -> "
                f"track={projection['track_id']} | "
                f"{projection['timestamp_utc']} | "
                f"x={projection['x']} "
                f"y={projection['y']} | "
                f"lat={projection['latitud']} "
                f"lon={projection['longitud']}"
            )
    else:
        print(
            f"Sin proyecciones: {track.get('motivo')}"
        )


if __name__ == "__main__":
    main()
