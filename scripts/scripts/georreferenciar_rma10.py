#!/usr/bin/env python3
"""
ClimaAR - Georreferenciacion operativa RMA10 por nucleo.

Cada nucleo mantiene su propia posicion, movimiento, distancia a
Bahia Blanca, rumbo, velocidad y ETA.

No se selecciona un unico nucleo para representar toda la tormenta.
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
    return max(
        low,
        min(high, value)
    )


def load_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
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


def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    radius = 6371.0088

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(
        lat2 - lat1
    )

    dl = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(
            dp / 2.0
        ) ** 2
        +
        math.cos(p1)
        * math.cos(p2)
        * math.sin(
            dl / 2.0
        ) ** 2
    )

    return (
        2.0
        * radius
        * math.asin(
            math.sqrt(
                clamp(
                    a,
                    0.0,
                    1.0
                )
            )
        )
    )


def bearing_deg(
    lat1,
    lon1,
    lat2,
    lon2
):

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dl = math.radians(
        lon2 - lon1
    )

    y = (
        math.sin(dl)
        * math.cos(p2)
    )

    x = (
        math.cos(p1)
        * math.sin(p2)
        -
        math.sin(p1)
        * math.cos(p2)
        * math.cos(dl)
    )

    return (
        math.degrees(
            math.atan2(
                y,
                x
            )
        )
        + 360.0
    ) % 360.0


def pixel_to_latlon(
    x,
    y,
    width,
    height
):

    center_x = width / 2.0
    center_y = height / 2.0

    radius_px_x = width / 2.0
    radius_px_y = height / 2.0

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
        round(lon, 6)
    )


def angle_difference_deg(
    a,
    b
):

    if (
        a is None
        or b is None
    ):
        return None

    return round(
        abs(
            (
                float(a)
                - float(b)
                + 180.0
            )
            % 360.0
            - 180.0
        ),
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

    radar_bahia_distance = (
        haversine_km(
            RADAR_LAT,
            RADAR_LON,
            BAHIA_LAT,
            BAHIA_LON
        )
    )

    radar_bahia_bearing = (
        bearing_deg(
            RADAR_LAT,
            RADAR_LON,
            BAHIA_LAT,
            BAHIA_LON
        )
    )

    nucleos = nowcast.get(
        "nucleos_analizados",
        nowcast.get(
            "nucleos",
            []
        )
    )

    if not isinstance(
        nucleos,
        list
    ):
        nucleos = []

    georeferenced = []

    for index, nucleo in enumerate(
        nucleos
    ):

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

        if (
            x is None
            or y is None
        ):
            continue

        x = finite(
            x,
            center_x
        )

        y = finite(
            y,
            center_y
        )

        lat, lon = (
            pixel_to_latlon(
                x,
                y,
                width,
                height
            )
        )

        distance_bahia = (
            haversine_km(
                lat,
                lon,
                BAHIA_LAT,
                BAHIA_LON
            )
        )

        bearing_bahia = (
            bearing_deg(
                lat,
                lon,
                BAHIA_LAT,
                BAHIA_LON
            )
        )

        movimiento = nucleo.get(
            "movimiento",
            {}
        )

        if not isinstance(
            movimiento,
            dict
        ):
            movimiento = {}

        direction = movimiento.get(
            "direccion_grados"
        )

        speed_px_min = finite(
            movimiento.get(
                "velocidad_px_min"
            ),
            0.0
        )

        speed_kmh = (
            speed_px_min
            *
            (
                RADAR_RANGE_KM
                /
                max(
                    width,
                    height
                )
            )
            *
            60.0
        )

        if speed_kmh <= 0:
            speed_kmh = None
        else:
            speed_kmh = round(
                speed_kmh,
                1
            )

        direction_error = (
            angle_difference_deg(
                direction,
                bearing_bahia
            )
        )

        toward_bahia = (
            direction_error is not None
            and direction_error <= 45.0
            and speed_kmh is not None
            and speed_kmh > 5.0
        )

        eta = None

        if toward_bahia:

            eta = round(
                distance_bahia
                /
                speed_kmh
                *
                60.0,
                1
            )

        item = dict(
            nucleo
        )

        item[
            "georreferencia"
        ] = {

            "latitud":
                lat,

            "longitud":
                lon,

            "distancia_bahia_km":
                round(
                    distance_bahia,
                    2
                ),

            "rumbo_bahia_grados":
                round(
                    bearing_bahia,
                    1
                ),
        }

        item[
            "movimiento_bahia"
        ] = {

            "velocidad_kmh":
                speed_kmh,

            "direccion_grados":
                (
                    round(
                        float(
                            direction
                        ),
                        1
                    )
                    if direction
                    is not None
                    else None
                ),

            "error_rumbo_bahia_grados":
                direction_error,

            "movimiento_hacia_bahia":
                bool(
                    toward_bahia
                ),

            "eta_minutos":
                eta,
        }

        item[
            "indice_nucleo"
        ] = index + 1

        georeferenced.append(
            item
        )

    candidates = [
        item
        for item
        in georeferenced
        if item.get(
            "movimiento_bahia",
            {}
        ).get(
            "movimiento_hacia_bahia"
        )
    ]

    candidates.sort(
        key=lambda item: (
            finite(
                item.get(
                    "georreferencia",
                    {}
                ).get(
                    "distancia_bahia_km"
                ),
                float("inf")
            ),
            -finite(
                item.get(
                    "intensidad"
                ),
                0
            )
        )
    )

    all_sorted = sorted(
        georeferenced,
        key=lambda item: finite(
            item.get(
                "georreferencia",
                {}
            ).get(
                "distancia_bahia_km"
            ),
            float("inf")
        )
    )

    nowcast[
        "georreferenciacion"
    ] = {

        "disponible":
            True,

        "metodo":
            "RMA10_centrado_240km_radio",

        "radar_latitud":
            RADAR_LAT,

        "radar_longitud":
            RADAR_LON,

        "bahia_blanca_latitud":
            BAHIA_LAT,

        "bahia_blanca_longitud":
            BAHIA_LON,

        "ancho_px":
            width,

        "alto_px":
            height,

        "centro_radar_x_px":
            center_x,

        "centro_radar_y_px":
            center_y,

        "alcance_radio_km":
            RADAR_RANGE_KM,

        "alcance_diametro_km":
            RADAR_RANGE_KM * 2.0,

        "calibracion_geografica":
            "operativa_no_validada_con_capa_cartografica",
    }

    nowcast[
        "nucleos_georreferenciados"
    ] = georeferenced

    nowcast[
        "resumen_nucleos"
    ] = {

        **(
            nowcast.get(
                "resumen_nucleos",
                {}
            )
            if isinstance(
                nowcast.get(
                    "resumen_nucleos",
                    {}
                ),
                dict
            )
            else {}
        ),

        "total_georreferenciados":
            len(
                georeferenced
            ),

        "hacia_bahia":
            len(
                candidates
            ),
    }

    if candidates:

        selected = candidates[0]

        selected_geo = selected[
            "georreferencia"
        ]

        selected_motion = selected[
            "movimiento_bahia"
        ]

        nowcast[
            "distancia_km"
        ] = selected_geo[
            "distancia_bahia_km"
        ]

        nowcast[
            "distancia_tipo"
        ] = (
            "georreferenciada_por_nucleo"
        )

        nowcast[
            "movimiento_hacia_bahia"
        ] = True

        nowcast[
            "rumbo_bahia_grados"
        ] = selected_geo[
            "rumbo_bahia_grados"
        ]

        nowcast[
            "error_rumbo_bahia_grados"
        ] = selected_motion[
            "error_rumbo_bahia_grados"
        ]

        nowcast[
            "eta_minutos"
        ] = selected_motion[
            "eta_minutos"
        ]

        nowcast[
            "velocidad_kmh"
        ] = selected_motion[
            "velocidad_kmh"
        ]

        nowcast[
            "direccion_grados"
        ] = selected_motion[
            "direccion_grados"
        ]

        nowcast[
            "bahia_blanca"
        ] = {

            "latitud":
                BAHIA_LAT,

            "longitud":
                BAHIA_LON,

            "distancia_principal_km":
                selected_geo[
                    "distancia_bahia_km"
                ],

            "rumbo_desde_principal_grados":
                selected_geo[
                    "rumbo_bahia_grados"
                ],

            "movimiento_hacia_bahia":
                True,

            "eta_minutos":
                selected_motion[
                    "eta_minutos"
                ],

            "nucleos_hacia_bahia": [

                {

                    "nucleo_id":
                        item.get(
                            "nucleo_id"
                        ),

                    "track_id":
                        item.get(
                            "track_id"
                        ),

                    "distancia_km":
                        item[
                            "georreferencia"
                        ][
                            "distancia_bahia_km"
                        ],

                    "eta_minutos":
                        item[
                            "movimiento_bahia"
                        ][
                            "eta_minutos"
                        ],

                    "intensidad":
                        item.get(
                            "intensidad"
                        ),
                }

                for item
                in candidates
            ],
        }

    else:

        nowcast[
            "distancia_km"
        ] = (

            all_sorted[0][
                "georreferencia"
            ][
                "distancia_bahia_km"
            ]

            if all_sorted

            else

            round(
                radar_bahia_distance,
                2
            )
        )

        nowcast[
            "distancia_tipo"
        ] = (

            "georreferenciada_sin_trayectoria_confirmada"

            if all_sorted

            else

            "radar_a_Bahia_Blanca"
        )

        nowcast[
            "movimiento_hacia_bahia"
        ] = False

        nowcast[
            "rumbo_bahia_grados"
        ] = (

            all_sorted[0][
                "georreferencia"
            ][
                "rumbo_bahia_grados"
            ]

            if all_sorted

            else None
        )

        nowcast[
            "error_rumbo_bahia_grados"
        ] = None

        nowcast[
            "eta_minutos"
        ] = None

        nowcast[
            "bahia_blanca"
        ] = {

            "latitud":
                BAHIA_LAT,

            "longitud":
                BAHIA_LON,

            "distancia_principal_km":
                (
                    all_sorted[0][
                        "georreferencia"
                    ][
                        "distancia_bahia_km"
                    ]
                    if all_sorted
                    else None
                ),

            "rumbo_desde_principal_grados":
                (
                    all_sorted[0][
                        "georreferencia"
                    ][
                        "rumbo_bahia_grados"
                    ]
                    if all_sorted
                    else None
                ),

            "movimiento_hacia_bahia":
                False,

            "eta_minutos":
                None,

            "nucleos_hacia_bahia":
                [],
        }

    diagnostico = data.get(
        "diagnostico",
        {}
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
        "georreferenciacion_por_nucleo"
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
        "nucleos_georreferenciados"
    ] = len(
        georeferenced
    )

    diagnostico[
        "nucleos_hacia_bahia"
    ] = len(
        candidates
    )

    diagnostico[
        "nota_georreferenciacion"
    ] = (
        "Cada nucleo se georreferencia y evalua "
        "individualmente. La posicion pixel->lat/lon "
        "usa el centro de RMA10 y 240 km de radio, "
        "por lo que continua siendo una transformacion "
        "operativa pendiente de validacion cartografica oficial."
    )

    data[
        "diagnostico"
    ] = diagnostico

    save_json(
        RADAR_FILE,
        data
    )

    print(
        "===== GEORREFERENCIACION RMA10 POR NUCLEO ====="
    )

    print(
        "Imagen:",
        f"{width}x{height}"
    )

    print(
        "Nucleos georreferenciados:",
        len(
            georeferenced
        )
    )

    print(
        "Nucleos hacia Bahia:",
        len(
            candidates
        )
    )

    for item in georeferenced:

        geo = item[
            "georreferencia"
        ]

        mov = item[
            "movimiento_bahia"
        ]

        print(
            "Nucleo",
            item.get(
                "nucleo_id"
            ),
            "| track",
            item.get(
                "track_id"
            ),
            "| distancia",
            geo[
                "distancia_bahia_km"
            ],
            "km | rumbo",
            geo[
                "rumbo_bahia_grados"
            ],
            "| movimiento hacia Bahia:",
            mov[
                "movimiento_hacia_bahia"
            ],
            "| ETA:",
            mov[
                "eta_minutos"
            ],
        )

    print(
        "Movimiento hacia Bahia:",
        nowcast.get(
            "movimiento_hacia_bahia"
        )
    )

    print(
        "Distancia seleccionada:",
        nowcast.get(
            "distancia_km"
        )
    )

    print(
        "ETA seleccionada:",
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
