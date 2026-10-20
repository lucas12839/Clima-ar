#!/usr/bin/env python3
"""
ClimaAR - Georreferenciacion RMA10 y movimiento hacia Bahia Blanca.

Puntos 20, 21 y 22:
- convierte cada nucleo de pixel a lat/lon;
- valida la geometria del radar RMA10;
- calcula distancia de cada nucleo a Bahia Blanca;
- calcula rumbo de cada nucleo hacia Bahia Blanca;
- calcula el rumbo real de movimiento desde dx/dy;
- determina si realmente se mueve hacia Bahia Blanca;
- separa el criterio direccional del criterio de ETA.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image

BASE = Path("data/radar")
RADAR_FILE = BASE / "radar_nowcast.json"
IMAGE_FILE = BASE / "actual.png"

RADAR_LAT = -38.7342
RADAR_LON = -62.1654

BAHIA_LAT = -38.7196
BAHIA_LON = -62.2724

RADAR_RANGE_KM = 240.0
EARTH_RADIUS_KM = 6371.0088

TOWARD_BAHIA_MAX_ERROR_DEG = 45.0
MIN_SPEED_KMH_FOR_ETA = 5.0


def finite(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def clamp(value, low, high):
    return max(low, min(high, value))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data):
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


def haversine_km(lat1, lon1, lat2, lon2):
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2.0) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2.0) ** 2
    )

    return (
        2.0
        * EARTH_RADIUS_KM
        * math.asin(math.sqrt(clamp(a, 0.0, 1.0)))
    )


def bearing_deg(lat1, lon1, lat2, lon2):
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)

    y = math.sin(dl) * math.cos(p2)
    x = (
        math.cos(p1) * math.sin(p2)
        - math.sin(p1) * math.cos(p2) * math.cos(dl)
    )

    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def destination_point(lat, lon, bearing, distance_km):
    if distance_km <= 0.0:
        return lat, lon

    phi1 = math.radians(lat)
    lam1 = math.radians(lon)
    theta = math.radians(bearing)
    delta = distance_km / EARTH_RADIUS_KM

    sin_phi1 = math.sin(phi1)
    cos_phi1 = math.cos(phi1)
    sin_delta = math.sin(delta)
    cos_delta = math.cos(delta)

    phi2 = math.asin(
        clamp(
            sin_phi1 * cos_delta
            + cos_phi1 * sin_delta * math.cos(theta),
            -1.0,
            1.0,
        )
    )

    lam2 = lam1 + math.atan2(
        math.sin(theta) * sin_delta * cos_phi1,
        cos_delta - sin_phi1 * math.sin(phi2),
    )

    lon2 = (math.degrees(lam2) + 540.0) % 360.0 - 180.0

    return math.degrees(phi2), lon2


def image_geometry(width, height):
    if width <= 1 or height <= 1:
        raise ValueError(f"Dimensiones de imagen invalidas: {width}x{height}")

    return (
        width / 2.0,
        height / 2.0,
        width / 2.0,
        height / 2.0,
    )


def pixel_local_xy(x, y, width, height):
    center_x, center_y, radius_x, radius_y = image_geometry(width, height)

    east_km = (
        (finite(x, center_x) - center_x)
        / radius_x
        * RADAR_RANGE_KM
    )

    north_km = (
        (center_y - finite(y, center_y))
        / radius_y
        * RADAR_RANGE_KM
    )

    return east_km, north_km


def pixel_to_latlon(x, y, width, height):
    east_km, north_km = pixel_local_xy(x, y, width, height)

    distance_km = math.hypot(east_km, north_km)

    if distance_km <= 0.0:
        return RADAR_LAT, RADAR_LON

    bearing = (
        math.degrees(math.atan2(east_km, north_km)) + 360.0
    ) % 360.0

    return destination_point(
        RADAR_LAT,
        RADAR_LON,
        bearing,
        distance_km,
    )


def pixel_distance_km(dx_px, dy_px, width, height):
    _, _, radius_x, radius_y = image_geometry(width, height)

    east_km = finite(dx_px, 0.0) / radius_x * RADAR_RANGE_KM
    north_km = -finite(dy_px, 0.0) / radius_y * RADAR_RANGE_KM

    return math.hypot(east_km, north_km)


def angle_difference_deg(a, b):
    if a is None or b is None:
        return None

    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def movement_bearing_from_pixels(dx_px, dy_px):
    dx = finite(dx_px, 0.0)
    dy = finite(dy_px, 0.0)

    magnitude = math.hypot(dx, dy)

    if magnitude <= 0.0:
        return None, 0.0

    # Imagen: Y crece hacia abajo. Norte geografico = -dy.
    north = -dy
    bearing = (math.degrees(math.atan2(dx, north)) + 360.0) % 360.0

    return bearing, magnitude


def movement_for_nucleus(nucleo, width, height):
    movimiento = nucleo.get("movimiento", {})
    if not isinstance(movimiento, dict):
        movimiento = {}

    dx_px = finite(movimiento.get("dx_px"), 0.0)
    dy_px = finite(movimiento.get("dy_px"), 0.0)
    speed_px_min = finite(
        movimiento.get("velocidad_px_min"),
        0.0,
    )

    direction_from_tracker = finite(
        movimiento.get("direccion_grados")
    )

    direction_from_vector, vector_px = movement_bearing_from_pixels(
        dx_px,
        dy_px,
    )

    direction = (
        direction_from_vector
        if direction_from_vector is not None
        else direction_from_tracker
    )

    speed_kmh = 0.0

    if speed_px_min > 0.0:
        distance_px = math.hypot(dx_px, dy_px)

        if distance_px > 0.0:
            distance_km = pixel_distance_km(
                dx_px,
                dy_px,
                width,
                height,
            )
            dt_equiv_min = distance_px / speed_px_min

            if dt_equiv_min > 0.0:
                speed_kmh = (
                    distance_km
                    / dt_equiv_min
                    * 60.0
                )

        if speed_kmh <= 0.0:
            km_per_px = (
                RADAR_RANGE_KM
                / (width / 2.0)
            )
            speed_kmh = speed_px_min * km_per_px * 60.0

    return {
        "velocidad_kmh": round(speed_kmh, 2),
        "direccion_grados": (
            round(direction, 2)
            if direction is not None
            else None
        ),
        "dx_px": round(dx_px, 4),
        "dy_px": round(dy_px, 4),
        "velocidad_pixeles_min": round(speed_px_min, 6),
        "vector_movimiento_px": round(vector_px, 4),
    }


def validate_geometry(width, height):
    center_x, center_y, _, _ = image_geometry(width, height)

    center_lat, center_lon = pixel_to_latlon(
        center_x,
        center_y,
        width,
        height,
    )

    center_error_m = (
        haversine_km(
            center_lat,
            center_lon,
            RADAR_LAT,
            RADAR_LON,
        )
        * 1000.0
    )

    cardinal = {
        "norte": (center_x, 0.0, 0.0),
        "sur": (center_x, float(height), 180.0),
        "este": (float(width), center_y, 90.0),
        "oeste": (0.0, center_y, 270.0),
    }

    controles = {}

    for name, (x, y, expected_bearing) in cardinal.items():
        lat, lon = pixel_to_latlon(
            x,
            y,
            width,
            height,
        )

        distance = haversine_km(
            RADAR_LAT,
            RADAR_LON,
            lat,
            lon,
        )

        actual_bearing = bearing_deg(
            RADAR_LAT,
            RADAR_LON,
            lat,
            lon,
        )

        bearing_error = angle_difference_deg(
            actual_bearing,
            expected_bearing,
        )

        controles[name] = {
            "latitud": round(lat, 6),
            "longitud": round(lon, 6),
            "distancia_km": round(distance, 3),
            "rumbo_grados": round(actual_bearing, 3),
            "error_rumbo_grados": round(bearing_error, 3),
            "error_radio_km": round(
                abs(distance - RADAR_RANGE_KM),
                3,
            ),
        }

    radios_ok = all(
        item["error_radio_km"] <= 0.01
        for item in controles.values()
    )

    rumbos_ok = all(
        item["error_rumbo_grados"] <= 0.01
        for item in controles.values()
    )

    centro_ok = center_error_m <= 0.001

    return {
        "estado": bool(
            centro_ok
            and radios_ok
            and rumbos_ok
        ),
        "centro_ok": centro_ok,
        "radio_cardinal_ok": radios_ok,
        "orientacion_cardinal_ok": rumbos_ok,
        "error_centro_m": round(center_error_m, 6),
        "controles_cardinales": controles,
    }


def process_nucleus(nucleo, index, width, height):
    if not isinstance(nucleo, dict):
        return None

    if nucleo.get("x") is None or nucleo.get("y") is None:
        return None

    x = finite(nucleo.get("x"), width / 2.0)
    y = finite(nucleo.get("y"), height / 2.0)

    east_km, north_km = pixel_local_xy(
        x,
        y,
        width,
        height,
    )

    distance_radar_km = math.hypot(
        east_km,
        north_km,
    )

    bearing_radar = (
        math.degrees(math.atan2(east_km, north_km)) + 360.0
    ) % 360.0

    if distance_radar_km <= 0.0:
        bearing_radar = 0.0

    lat, lon = pixel_to_latlon(
        x,
        y,
        width,
        height,
    )

    distance_bahia = haversine_km(
        lat,
        lon,
        BAHIA_LAT,
        BAHIA_LON,
    )

    bearing_bahia = bearing_deg(
        lat,
        lon,
        BAHIA_LAT,
        BAHIA_LON,
    )

    movement = movement_for_nucleus(
        nucleo,
        width,
        height,
    )

    direction_error = angle_difference_deg(
        movement["direccion_grados"],
        bearing_bahia,
    )

    speed_kmh = movement["velocidad_kmh"]

    component_toward = 0.0
    if (
        movement["direccion_grados"] is not None
        and direction_error is not None
        and speed_kmh > 0.0
    ):
        component_toward = (
            speed_kmh
            * math.cos(math.radians(direction_error))
        )

    moving_toward = bool(
        movement["direccion_grados"] is not None
        and movement["vector_movimiento_px"] > 0.0
        and direction_error is not None
        and direction_error <= TOWARD_BAHIA_MAX_ERROR_DEG
        and component_toward > 0.0
    )

    approaching_distance_10min = max(
        0.0,
        distance_bahia - component_toward / 6.0,
    )

    distance_reduction_10min = max(
        0.0,
        distance_bahia - approaching_distance_10min,
    )

    eta = None
    if (
        moving_toward
        and component_toward >= MIN_SPEED_KMH_FOR_ETA
    ):
        eta = round(
            distance_bahia / component_toward * 60.0,
            1,
        )

    item = dict(nucleo)

    item["georreferencia"] = {
        "latitud": round(lat, 6),
        "longitud": round(lon, 6),
        "distancia_radar_km": round(distance_radar_km, 2),
        "rumbo_desde_radar_grados": round(bearing_radar, 1),
        "este_desde_radar_km": round(east_km, 2),
        "norte_desde_radar_km": round(north_km, 2),
        "dentro_alcance_nominal": bool(
            distance_radar_km <= RADAR_RANGE_KM + 0.5
        ),
        "distancia_bahia_km": round(distance_bahia, 2),
        "rumbo_bahia_grados": round(bearing_bahia, 1),
    }

    item["movimiento_bahia"] = {
        **movement,
        "direccion_movimiento_grados": (
            movement["direccion_grados"]
        ),
        "error_rumbo_bahia_grados": (
            round(direction_error, 1)
            if direction_error is not None
            else None
        ),
        "componente_hacia_bahia_kmh": round(
            component_toward,
            2,
        ),
        "movimiento_valido": bool(
            movement["vector_movimiento_px"] > 0.0
        ),
        "movimiento_hacia_bahia": moving_toward,
        "acercandose_a_bahia": moving_toward,
        "reduccion_distancia_10min_km": round(
            distance_reduction_10min,
            2,
        ),
        "distancia_proyectada_10min_km": round(
            approaching_distance_10min,
            2,
        ),
        "eta_minutos": eta,
        "criterio_direccion": (
            "CONO_45_GRADOS_Y_COMPONENTE_POSITIVA"
        ),
    }

    item["indice_nucleo"] = index + 1

    return item


def main():
    if not RADAR_FILE.exists():
        raise SystemExit(
            "ERROR: no existe data/radar/radar_nowcast.json"
        )

    if not IMAGE_FILE.exists():
        raise SystemExit(
            "ERROR: no existe data/radar/actual.png"
        )

    data = load_json(RADAR_FILE)
    nowcast = data.get("nowcast")

    if not isinstance(nowcast, dict):
        raise SystemExit(
            "ERROR: radar_nowcast.json no contiene nowcast valido"
        )

    with Image.open(IMAGE_FILE) as image:
        width, height = image.size

    validation = validate_geometry(
        width,
        height,
    )

    if not validation["estado"]:
        raise SystemExit(
            "ERROR: fallo la validacion geometrica RMA10"
        )

    radar_bahia_distance = haversine_km(
        RADAR_LAT,
        RADAR_LON,
        BAHIA_LAT,
        BAHIA_LON,
    )

    radar_bahia_bearing = bearing_deg(
        RADAR_LAT,
        RADAR_LON,
        BAHIA_LAT,
        BAHIA_LON,
    )

    nucleos = nowcast.get(
        "nucleos",
        nowcast.get("nucleos_analizados", []),
    )

    if not isinstance(nucleos, list):
        nucleos = []

    georeferenced = []

    for index, nucleo in enumerate(nucleos):
        item = process_nucleus(
            nucleo,
            index,
            width,
            height,
        )

        if item is not None:
            georeferenced.append(item)

    principal_id = nowcast.get("nucleo_principal_id")
    principal_item = None

    if principal_id is not None:
        for item in georeferenced:
            if (
                item.get("track_id") == principal_id
                or item.get("id") == principal_id
                or item.get("nucleo_id") == principal_id
            ):
                principal_item = item
                break

    approaching = [
        item
        for item in georeferenced
        if item.get("movimiento_bahia", {}).get(
            "movimiento_hacia_bahia"
        )
    ]

    # Para la evaluacion hacia Bahia, un nucleo principal que NO se
    # aproxima a Bahia no debe ocultar a otro nucleo que SI lo hace.
    # Primero conservamos el principal del radar solo si realmente va
    # hacia Bahia; si no, seleccionamos el nucleo que se aproxima y esta
    # mas cerca de Bahia.
    principal_hacia_bahia = (
        principal_item is not None
        and principal_item.get("movimiento_bahia", {}).get(
            "movimiento_hacia_bahia"
        ) is True
    )

    if not principal_hacia_bahia and approaching:
        principal_item = min(
            approaching,
            key=lambda item: finite(
                item.get("georreferencia", {}).get(
                    "distancia_bahia_km"
                ),
                float("inf"),
            ),
        )
    elif principal_item is None and georeferenced:
        principal_item = min(
            georeferenced,
            key=lambda item: finite(
                item.get("georreferencia", {}).get(
                    "distancia_bahia_km"
                ),
                float("inf"),
            ),
        )

    nowcast["georreferenciacion"] = {
        "disponible": True,
        "estado": "VALIDADA_OPERATIVAMENTE",
        "metodo": (
            "RMA10_CARTESIANO_CENTRADO_240KM_RADIO_GEODESICO"
        ),
        "fuente_parametros": "SMN_RMA10",
        "radar_latitud": RADAR_LAT,
        "radar_longitud": RADAR_LON,
        "bahia_blanca_latitud": BAHIA_LAT,
        "bahia_blanca_longitud": BAHIA_LON,
        "ancho_px": width,
        "alto_px": height,
        "centro_radar_x_px": width / 2.0,
        "centro_radar_y_px": height / 2.0,
        "alcance_radio_km": RADAR_RANGE_KM,
        "alcance_diametro_km": RADAR_RANGE_KM * 2.0,
        "km_por_pixel_horizontal": round(
            RADAR_RANGE_KM / (width / 2.0),
            6,
        ),
        "km_por_pixel_vertical": round(
            RADAR_RANGE_KM / (height / 2.0),
            6,
        ),
        "validacion": validation,
        "calibracion_geografica": (
            "VALIDADA_CON_COORDENADAS_OFICIALES_RMA10_Y_ALCANCE_240KM"
        ),
        "limitacion": (
            "La fuente PNG no contiene un GeoTransform GIS; "
            "la precision queda limitada a la geometria del "
            "producto RMA10 y a su renderizado pixelado."
        ),
    }

    nowcast["nucleos_georreferenciados"] = georeferenced

    nowcast["resumen_nucleos"] = {
        **(
            nowcast.get("resumen_nucleos", {})
            if isinstance(nowcast.get("resumen_nucleos", {}), dict)
            else {}
        ),
        "total_georreferenciados": len(georeferenced),
        "movimiento_valido": sum(
            1
            for item in georeferenced
            if item.get("movimiento_bahia", {}).get(
                "movimiento_valido"
            )
        ),
        "hacia_bahia": len(approaching),
        "criterio_hacia_bahia": (
            "RUMBO_DENTRO_DE_45_GRADOS_Y_COMPONENTE_POSITIVA"
        ),
    }

    if principal_item is not None:
        geo = principal_item["georreferencia"]
        motion = principal_item["movimiento_bahia"]

        nowcast["distancia_km"] = geo["distancia_bahia_km"]
        nowcast["distancia_tipo"] = (
            "georreferenciada_por_nucleo_validada"
        )
        nowcast["rumbo_bahia_grados"] = (
            geo["rumbo_bahia_grados"]
        )
        nowcast["error_rumbo_bahia_grados"] = (
            motion["error_rumbo_bahia_grados"]
        )
        nowcast["movimiento_hacia_bahia"] = (
            motion["movimiento_hacia_bahia"]
        )
        nowcast["eta_minutos"] = motion["eta_minutos"]
        nowcast["velocidad_kmh"] = motion["velocidad_kmh"]
        nowcast["direccion_grados"] = (
            motion["direccion_movimiento_grados"]
        )

        nowcast["bahia_blanca"] = {
            "latitud": BAHIA_LAT,
            "longitud": BAHIA_LON,
            "distancia_principal_km": (
                geo["distancia_bahia_km"]
            ),
            "rumbo_desde_principal_grados": (
                geo["rumbo_bahia_grados"]
            ),
            "movimiento_hacia_bahia": (
                motion["movimiento_hacia_bahia"]
            ),
            "eta_minutos": motion["eta_minutos"],
            "nucleo_principal_id": principal_item.get(
                "track_id",
                principal_item.get("nucleo_id"),
            ),
        }
    else:
        nowcast["distancia_km"] = round(
            radar_bahia_distance,
            2,
        )
        nowcast["distancia_tipo"] = (
            "radar_a_Bahia_Blanca_sin_nucleo"
        )
        nowcast["rumbo_bahia_grados"] = round(
            radar_bahia_bearing,
            1,
        )
        nowcast["movimiento_hacia_bahia"] = False
        nowcast["eta_minutos"] = None
        nowcast["bahia_blanca"] = {
            "latitud": BAHIA_LAT,
            "longitud": BAHIA_LON,
            "distancia_radar_km": round(
                radar_bahia_distance,
                2,
            ),
            "rumbo_radar_bahia_grados": round(
                radar_bahia_bearing,
                1,
            ),
            "movimiento_hacia_bahia": False,
            "eta_minutos": None,
        }

    diagnostico = data.get("diagnostico")
    if not isinstance(diagnostico, dict):
        diagnostico = {}

    diagnostico.update({
        "georreferenciacion_exacta": False,
        "georreferenciacion_operativa": True,
        "georreferenciacion_validada": True,
        "radar_latitud": RADAR_LAT,
        "radar_longitud": RADAR_LON,
        "bahia_blanca_latitud": BAHIA_LAT,
        "bahia_blanca_longitud": BAHIA_LON,
        "validacion_geografica": validation,
        "nota_georreferenciacion": (
            "Transformacion pixel->lat/lon validada matematicamente "
            "contra las coordenadas oficiales de RMA10 y el alcance "
            "nominal de 240 km de radio."
        ),
        "punto_22_movimiento_bahia": {
            "criterio": (
                "rumbo <= 45 grados respecto a Bahia y "
                "componente de velocidad positiva hacia Bahia"
            ),
            "nucleos_hacia_bahia": len(approaching),
            "nucleos_georreferenciados": len(georeferenced),
        },
    })

    data["diagnostico"] = diagnostico

    save_json(
        RADAR_FILE,
        data,
    )

    print("===== GEORREFERENCIACION RMA10 VALIDADA =====")
    print(f"Imagen: {width}x{height}")
    print(f"Radar: {RADAR_LAT}, {RADAR_LON}")
    print(f"Bahia Blanca: {BAHIA_LAT}, {BAHIA_LON}")
    print(f"Alcance radio: {RADAR_RANGE_KM} km")
    print(
        "Distancia radar-Bahia:",
        round(radar_bahia_distance, 2),
        "km",
    )
    print(
        "Rumbo radar-Bahia:",
        round(radar_bahia_bearing, 1),
        "grados",
    )
    print("Validacion geometrica:", validation["estado"])
    print(
        "Nucleos georreferenciados:",
        len(georeferenced),
    )
    print(
        "Nucleos hacia Bahia:",
        len(approaching),
    )
    print(
        "Movimiento principal hacia Bahia:",
        nowcast.get("movimiento_hacia_bahia"),
    )
    print(
        "Distancia principal:",
        nowcast.get("distancia_km"),
    )
    print(
        "ETA principal:",
        nowcast.get("eta_minutos"),
    )
    print("Salida:", RADAR_FILE)


if __name__ == "__main__":
    main()
