#!/usr/bin/env python3
"""
ClimaAR - Georreferenciacion operativa RMA10.

Convierte coordenadas de pixel del RMA10 a lat/lon usando:
- antena RMA10: -38.7342, -62.1654
- alcance nominal: 240 km de radio
- Bahia Blanca: -38.7196, -62.2724

La imagen se trata como un dominio centrado en la antena.
La transformacion es una aproximacion cartografica local y queda
marcada en el JSON para poder calibrarla posteriormente contra
una capa geografica oficial.
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


def finite(value, default=0.0):
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return default


def clamp(value, low, high):
    return max(low, min(high, value))


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def save_json(path, data):
    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

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
    radius = 6371.0088

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
        * radius
        * math.asin(
            math.sqrt(
                clamp(a, 0.0, 1.0)
            )
        )
    )


def bearing_deg(lat1, lon1, lat2, lon2):
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)

    y = math.sin(dl) * math.cos(p2)

    x = (
        math.cos(p1) * math.sin(p2)
        - math.sin(p1)
        * math.cos(p2)
        * math.cos(dl)
    )

    return (
        math.degrees(
            math.atan2(y, x)
        )
        + 360.0
    ) % 360.0


def pixel_to_latlon(
    x,
    y,
    width,
    height,
):
    """
    x positivo hacia este.
    y positivo hacia abajo.

    El alcance nominal de 240 km se toma como radio.
    Por eso el ancho completo representa aproximadamente 480 km.
    """

    center_x = (
        width / 2.0
    )

    center_y = (
        height / 2.0
    )

    radius_px_x = (
        width / 2.0
    )

    radius_px_y = (
        height / 2.0
    )

    east_km = (
        (x - center_x)
        / radius_px_x
        * RADAR_RANGE_KM
    )

    north_km = (
        (center_y - y)
        / radius_px_y
        * RADAR_RANGE_KM
    )

    lat = (
        RADAR_LAT
        + north_km / 111.32
    )

    lon_scale = (
        111.32
        * math.cos(
            math.radians(
                RADAR_LAT
            )
        )
    )

    lon = (
        RADAR_LON
        + east_km / lon_scale
    )

    return (
        round(lat, 6),
        round(lon, 6),
    )


def vector_to_bahia():
    """
    Vector local desde la antena RMA10 hacia Bahia Blanca.

    Devuelve:
    este_km, norte_km, distancia_km, bearing_deg
    """

    mean_lat = math.radians(
        (
            RADAR_LAT
            + BAHIA_LAT
        ) / 2.0
    )

    east_km = (
        (
            BAHIA_LON
            - RADAR_LON
        )
        * 111.32
        * math.cos(mean_lat)
    )

    north_km = (
        (
            BAHIA_LAT
            - RADAR_LAT
        )
        * 111.32
    )

    distance = math.hypot(
        east_km,
        north_km
    )

    bearing = bearing_deg(
        RADAR_LAT,
        RADAR_LON,
        BAHIA_LAT,
        BAHIA_LON,
    )

    return (
        east_km,
        north_km,
        distance,
        bearing,
    )


def motion_vector_from_direction(
    speed_kmh,
    direction_deg,
):
    if (
        speed_kmh is None
        or direction_deg is None
        or speed_kmh <= 0
    ):
        return (
            0.0,
            0.0,
        )

    direction = math.radians(
        direction_deg
    )

    east_kmh = (
        speed_kmh
        * math.sin(direction)
    )

    north_kmh = (
        speed_kmh
        * math.cos(direction)
    )

    return (
        east_kmh,
        north_kmh,
    )


def angle_difference_deg(
    a,
    b,
):
    if a is None or b is None:
        return None

    value = abs(
        (
            float(a)
            - float(b)
            + 180.0
        ) % 360.0
        - 180.0
    )

    return round(
        value,
        1
    )


def main():
    if not RADAR_FILE.exists():
        raise SystemExit(
            "ERROR: no existe radar_nowcast.json"
        )

    if not IMAGE_FILE.exists():
        raise SystemExit(
            "ERROR: no existe data/radar/actual.png"
        )

    data = load_json(
        RADAR_FILE
    )

    nowcast = data.get(
        "nowcast"
    )

    if not isinstance(
        nowcast,
        dict
    ):
        raise SystemExit(
            "ERROR: radar_nowcast.json "
            "no contiene nowcast valido"
        )

    with Image.open(
        IMAGE_FILE
    ) as image:
        width, height = image.size

    center_x = width / 2.0
    center_y = height / 2.0

    (
        bahia_east_km,
        bahia_north_km,
        bahia_distance_km,
        bahia_bearing_deg,
    ) = vector_to_bahia()

    principal = nowcast.get(
        "nucleo_principal_id"
    )

    nucleos = nowcast.get(
        "nucleos",
        []
    )

    if not isinstance(
        nucleos,
        list
    ):
        nucleos = []

    georeferenced = []

    for nucleo in nucleos:
        if not isinstance(
            nucleo,
            dict
        ):
            continue

        x = nucleo.get(
            "x"
        )

        y = nucleo.get(
            "y"
        )

        if x is None or y is None:
            continue

        x = finite(
            x,
            center_x
        )

        y = finite(
            y,
            center_y
        )

        lat, lon = pixel_to_latlon(
            x,
            y,
            width,
            height,
        )

        distance_bahia = (
            haversine_km(
                lat,
                lon,
                BAHIA_LAT,
                BAHIA_LON,
            )
        )

        bearing_bahia = (
            bearing_deg(
                lat,
                lon,
                BAHIA_LAT,
                BAHIA_LON,
            )
        )

        item = dict(
            nucleo
        )

        item["georeferencia"] = {
            "latitud": lat,
            "longitud": lon,
            "distancia_bahia_km": round(
                distance_bahia,
                2
            ),
            "rumbo_bahia_grados": round(
                bearing_bahia,
                1
            ),
        }

        georeferenced.append(
            item
        )

    nowcast["georreferenciacion"] = {
        "disponible": True,
        "metodo": (
            "RMA10_centrado_240km_radio"
        ),
        "radar_latitud": RADAR_LAT,
        "radar_longitud": RADAR_LON,
        "bahia_blanca_latitud": BAHIA_LAT,
        "bahia_blanca_longitud": BAHIA_LON,
        "ancho_px": width,
        "alto_px": height,
        "centro_radar_x_px": center_x,
        "centro_radar_y_px": center_y,
        "alcance_radio_km": RADAR_RANGE_KM,
        "alcance_diametro_km": (
            RADAR_RANGE_KM * 2.0
        ),
        "calibracion_geografica": (
            "operativa_no_validada_con_capa_cartografica"
        ),
    }

    nowcast["nucleos_georreferenciados"] = (
        georeferenced
    )

    if principal is not None:
        principal_item = None

        for item in georeferenced:
            if (
                item.get(
                    "track_id"
                )
                == principal
            ):
                principal_item = item
                break

        if principal_item is None:
            for item in georeferenced:
                if (
                    item.get("id")
                    == principal
                ):
                    principal_item = item
                    break

        if principal_item is not None:
            geo = principal_item[
                "georeferencia"
            ]

            distance_bahia = finite(
                geo.get(
                    "distancia_bahia_km"
                ),
                bahia_distance_km,
            )

            bearing_from_cell = finite(
                geo.get(
                    "rumbo_bahia_grados"
                ),
                bahia_bearing_deg,
            )

            direction = nowcast.get(
                "direccion_grados"
            )

            speed_kmh = nowcast.get(
                "velocidad_kmh"
            )

            direction_error = (
                angle_difference_deg(
                    direction,
                    bearing_from_cell,
                )
            )

            toward_bahia = (
                direction_error is not None
                and direction_error <= 45.0
                and finite(
                    speed_kmh,
                    0.0
                ) > 5.0
            )

            eta = None

            if toward_bahia:
                eta = round(
                    distance_bahia
                    / float(speed_kmh)
                    * 60.0,
                    1,
                )

            nowcast[
                "distancia_km"
            ] = round(
                distance_bahia,
                2
            )

            nowcast[
                "distancia_tipo"
            ] = (
                "georreferenciada_a_Bahia_Blanca"
            )

            nowcast[
                "movimiento_hacia_bahia"
            ] = bool(
                toward_bahia
            )

            nowcast[
                "rumbo_bahia_grados"
            ] = round(
                bearing_from_cell,
                1
            )

            nowcast[
                "error_rumbo_bahia_grados"
            ] = direction_error

            nowcast[
                "eta_minutos"
            ] = eta

            nowcast[
                "bahia_blanca"
            ] = {
                "latitud": BAHIA_LAT,
                "longitud": BAHIA_LON,
                "distancia_principal_km": round(
                    distance_bahia,
                    2
                ),
                "rumbo_desde_principal_grados": round(
                    bearing_from_cell,
                    1
                ),
                "movimiento_hacia_bahia": bool(
                    toward_bahia
                ),
                "eta_minutos": eta,
            }

    diagnostico = data.get(
        "diagnostico"
    )

    if not isinstance(
        diagnostico,
        dict
    ):
        diagnostico = {}

    diagnostico[
        "georreferenciacion_exacta"
    ] = False

    diagnostico[
        "georreferenciacion_operativa"
    ] = True

    diagnostico[
        "radar_latitud"
    ] = RADAR_LAT

    diagnostico[
        "radar_longitud"
    ] = RADAR_LON

    diagnostico[
        "bahia_blanca_latitud"
    ] = BAHIA_LAT

    diagnostico[
        "bahia_blanca_longitud"
    ] = BAHIA_LON

    diagnostico[
        "nota_georreferenciacion"
    ] = (
        "Transformacion pixel->lat/lon basada "
        "en el centro de RMA10 y alcance nominal "
        "de 240 km de radio. Debe validarse "
        "contra una capa geografica oficial antes "
        "de considerarse georreferenciacion exacta."
    )

    data["diagnostico"] = diagnostico

    save_json(
        RADAR_FILE,
        data
    )

    print(
        "===== GEORREFERENCIACION RMA10 ====="
    )
    print(
        "Imagen:",
        f"{width}x{height}"
    )
    print(
        "Radar:",
        RADAR_LAT,
        RADAR_LON
    )
    print(
        "Bahia Blanca:",
        BAHIA_LAT,
        BAHIA_LON
    )
    print(
        "Distancia radar-Bahia:",
        round(
            bahia_distance_km,
            2
        ),
        "km"
    )
    print(
        "Rumbo radar-Bahia:",
        round(
            bahia_bearing_deg,
            1
        ),
        "grados"
    )
    print(
        "Nucleos georreferenciados:",
        len(georeferenced)
    )
    print(
        "Movimiento hacia Bahia:",
        nowcast.get(
            "movimiento_hacia_bahia"
        )
    )
    print(
        "Distancia principal:",
        nowcast.get(
            "distancia_km"
        )
    )
    print(
        "ETA:",
        nowcast.get(
            "eta_minutos"
        )
    )
    print(
        "Salida:",
        RADAR_FILE
    )


if __name__ == "__main__":
    main()
