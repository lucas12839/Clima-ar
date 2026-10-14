#!/usr/bin/env python3
"""
ClimaAR - Adaptador operativo RMA10 -> Storm Engine.

Convierte la salida REAL del pipeline RMA10 al contrato que consumen
meteorologia_fusion.py y tormenta_intelligence.py.

Importante:
- NO convierte el nivel de intensidad 0-6 en dBZ.
- NO inventa coordenadas geográficas.
- El movimiento se calcula desde el historial real del track.
- La distancia/ETA al centro del dominio son aproximadas y quedan
  marcadas como tales hasta calibrar la geometría geográfica del RMA10.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("data/radar")

RESULTADO = BASE / "resultado_procesamiento.json"
SEGUIMIENTO = BASE / "seguimiento" / "seguimiento_temporal.json"
ESTADO = BASE / "seguimiento" / "estado.json"
PREDICCION = BASE / "seguimiento" / "prediccion_evolucion.json"

SALIDA = BASE / "radar_nowcast.json"


# Aproximación provisional del diámetro del dominio RMA10.
# Se utiliza únicamente para estimaciones internas de distancia/velocidad.
RADAR_DIAMETRO_KM_APROX = 240.0


def load_json(path: Path, default):
    if not path.exists():
        return default

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return default


def number(value, default=0.0):
    try:
        value = float(value)

        if math.isfinite(value):
            return value

        return default

    except Exception:
        return default


def clamp(value, low=0.0, high=1.0):
    value = number(value, low)

    return max(
        low,
        min(high, value)
    )


def timestamp_from_result(resultado):
    name = str(
        resultado.get(
            "imagen_procesada",
            ""
        )
    )

    if (
        len(name) >= 14
        and name[:14].isdigit()
    ):
        dt = datetime.strptime(
            name[:14],
            "%Y%m%d%H%M%S"
        )

        return (
            dt.replace(
                tzinfo=timezone.utc
            )
            .isoformat()
            .replace("+00:00", "Z")
        )

    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def active_tracks(estado):
    tracks = estado.get(
        "tracks",
        []
    )

    if not isinstance(
        tracks,
        list
    ):
        return []

    result = []

    for track in tracks:

        if not isinstance(
            track,
            dict
        ):
            continue

        if track.get(
            "activo"
        ) is False:
            continue

        if track.get(
            "presente_en_ultimo_frame"
        ) is False:
            continue

        result.append(track)

    return result


def track_motion(track):
    """
    Calcula el movimiento usando los dos últimos puntos reales
    almacenados en el historial del track.

    No depende de dx_pixeles/dy_pixeles porque esos campos no forman
    parte del esquema actual de estado.json.
    """

    history = track.get(
        "historial",
        []
    )

    if (
        not isinstance(history, list)
        or len(history) < 2
    ):
        return (
            0.0,
            0.0,
            0.0,
            None
        )

    previous = history[-2]
    current = history[-1]

    if (
        not isinstance(previous, dict)
        or not isinstance(current, dict)
    ):
        return (
            0.0,
            0.0,
            0.0,
            None
        )

    dx = (
        number(current.get("x"))
        - number(previous.get("x"))
    )

    dy = (
        number(current.get("y"))
        - number(previous.get("y"))
    )

    distance_px = math.hypot(
        dx,
        dy
    )

    dt_min = None

    timestamp_previous = previous.get(
        "timestamp"
    )

    timestamp_current = current.get(
        "timestamp"
    )

    try:
        dt_previous = datetime.fromisoformat(
            str(timestamp_previous).replace(
                "Z",
                "+00:00"
            )
        )

        dt_current = datetime.fromisoformat(
            str(timestamp_current).replace(
                "Z",
                "+00:00"
            )
        )

        dt_min = (
            dt_current
            - dt_previous
        ).total_seconds() / 60.0

    except Exception:
        pass

    if (
        dt_min is not None
        and dt_min > 0
    ):
        speed_px_min = (
            distance_px
            / dt_min
        )
    else:
        speed_px_min = 0.0

    if distance_px > 0:

        # 0° = norte
        # 90° = este
        # 180° = sur
        # 270° = oeste
        direction = (
            math.degrees(
                math.atan2(
                    dx,
                    -dy
                )
            )
            + 360.0
        ) % 360.0

    else:
        direction = None

    return (
        dx,
        dy,
        speed_px_min,
        direction
    )


def area_trend(track):
    """
    Determina fortalecimiento/debilitamiento comparando el área
    del track entre los dos últimos frames.
    """

    history = track.get(
        "historial",
        []
    )

    if (
        not isinstance(history, list)
        or len(history) < 2
    ):
        return (
            "estable",
            0.0
        )

    previous = number(
        history[-2].get(
            "pixeles",
            0
        ),
        0
    )

    current = number(
        history[-1].get(
            "pixeles",
            0
        ),
        0
    )

    if previous <= 0:
        return (
            "estable",
            0.0
        )

    change = (
        (current - previous)
        / previous
        * 100.0
    )

    if change >= 10.0:
        return (
            "fortaleciendose",
            change
        )

    if change <= -10.0:
        return (
            "debilitandose",
            change
        )

    return (
        "estable",
        change
    )


def choose_principal(tracks):
    if not tracks:
        return None

    return max(
        tracks,
        key=lambda track: (
            max(
                0.0,
                number(
                    track.get(
                        "pixeles",
                        0
                    ),
                    0
                )
            )
            *
            max(
                1,
                number(
                    track.get(
                        "intensidad",
                        0
                    ),
                    0
                )
            )
        )
    )


def main():

    resultado = load_json(
        RESULTADO,
        {}
    )

    seguimiento = load_json(
        SEGUIMIENTO,
        {}
    )

    estado = load_json(
        ESTADO,
        {}
    )

    prediccion = load_json(
        PREDICCION,
        {}
    )

    if not isinstance(
        resultado,
        dict
    ):
        raise SystemExit(
            "ERROR: resultado_procesamiento.json inválido"
        )

    if not isinstance(
        estado,
        dict
    ):
        raise SystemExit(
            "ERROR: estado.json inválido"
        )

    width = max(
        1.0,
        number(
            resultado.get(
                "ancho_imagen"
            ),
            512
        )
    )

    height = max(
        1.0,
        number(
            resultado.get(
                "alto_imagen"
            ),
            512
        )
    )

    # Referencia geométrica del dominio.
    # NO representa automáticamente Bahía Blanca.
    center_x = width / 2.0
    center_y = height / 2.0

    km_per_px = (
        RADAR_DIAMETRO_KM_APROX
        / max(
            width,
            height
        )
    )

    nucleos = resultado.get(
        "nucleos",
        []
    )

    if not isinstance(
        nucleos,
        list
    ):
        nucleos = []

    precip = [
        nucleo
        for nucleo in nucleos
        if (
            isinstance(
                nucleo,
                dict
            )
            and nucleo.get(
                "estado"
            )
            in (
                "PRECIPITACION",
                "CAMBIO"
            )
        )
    ]

    tracks = active_tracks(
        estado
    )

    principal = choose_principal(
        tracks
    )

    actividad = bool(
        precip
    )

    area_px = sum(
        max(
            0.0,
            number(
                nucleo.get(
                    "pixeles"
                ),
                0
            )
        )
        for nucleo in precip
    )

    intensidad_max = max(
        (
            int(
                number(
                    nucleo.get(
                        "intensidad",
                        0
                    ),
                    0
                )
            )
            for nucleo in precip
        ),
        default=0
    )

    if principal:

        (
            dx,
            dy,
            speed_px_min,
            direction
        ) = track_motion(
            principal
        )

    else:

        dx = 0.0
        dy = 0.0
        speed_px_min = 0.0
        direction = None

    if principal:

        x = number(
            principal.get(
                "x"
            ),
            center_x
        )

        y = number(
            principal.get(
                "y"
            ),
            center_y

        )

    else:

        x = center_x
        y = center_y

    to_center_x = (
        center_x - x
    )

    to_center_y = (
        center_y - y
    )

    distance_to_center_px = math.hypot(
        to_center_x,
        to_center_y
    )

    toward_center = (
        dx * to_center_x
        +
        dy * to_center_y
        > 0
        if (
            principal
            and speed_px_min > 0
        )
        else False
    )

    if speed_px_min > 0:

        speed_kmh = round(
            speed_px_min
            * km_per_px
            * 60.0,
            1
        )

    else:

        speed_kmh = None

    if principal:

        distance_km_aprox = round(
            distance_to_center_px
            * km_per_px,
            1
        )

    else:

        distance_km_aprox = None

    eta_minutos_aprox = None

    if (
        toward_center
        and speed_kmh is not None
        and speed_kmh > 5
        and distance_km_aprox is not None
    ):

        eta_minutos_aprox = round(
            distance_km_aprox
            / speed_kmh
            * 60.0,
            1
        )

    if principal:

        (
            tendencia,
            cambio_area_pct
        ) = area_trend(
            principal
        )

    else:

        tendencia = "estable"
        cambio_area_pct = 0.0

    if principal:

        confianza = (
            clamp(
                number(
                    principal.get(
                        "confianza",
                        0
                    ),
                    0
                ),
                0,
                99
            )
            / 99.0
        )

    else:

        confianza = 0.0

    radar_ts = timestamp_from_result(
        resultado
    )

    salida = {

        "version":
            "RMA10-operativo-1.1",

        "fuente":
            "SMN RMA10",

        "actualizado":
            radar_ts,

        "nowcast": {

            "actualizado":
                radar_ts,

            "frame_actual_utc":
                radar_ts,

            "frames_analizados":
                len(nucleos),

            "tiles_ok":
                True,

            "actividad":
                actividad,

            "area_px":
                round(
                    area_px,
                    2
                ),

            # RMA10 actualmente entrega niveles 0-6.
            # NO se convierten artificialmente a dBZ.
            "intensidad_nivel_max":
                intensidad_max,

            "intensidad_escala":
                "RMA10_0_6",

            "dbz_max":
                None,

            "dbz_mean":
                None,

            "distancia_km":
                distance_km_aprox,

            "distancia_tipo":
                "centro_dominio_aproximada",

            "fortalecimiento":
                tendencia,

            "cambio_area_pct":
                round(
                    cambio_area_pct,
                    2
                ),

            "velocidad_kmh":
                speed_kmh,

            "velocidad_pixeles_min":
                round(
                    speed_px_min,
                    4
                ),

            "direccion":
                direction,

            "direccion_grados":
                direction,

            # Hasta calibrar la georreferenciación
            # NO se afirma que sea movimiento hacia Bahía Blanca.
            "movimiento_hacia_bahia":
                False,

            "movimiento_hacia_centro_dominio":
                toward_center,

            "eta_minutos":
                None,

            "eta_minutos_aproximada_centro_dominio":
                eta_minutos_aprox,

            "confianza_movimiento":
                round(
                    confianza,
                    3
                ),

            "estado":
                (
                    "actividad_detectada"
                    if actividad
                    else
                    "sin_precipitacion"
                ),

            "nucleos":
                precip,

            "nucleo_principal_id":
                (
                    principal.get(
                        "track_id"
                    )
                    if principal
                    else None
                ),

            "proyecciones":
                {},
        },

        "diagnostico": {

            "sin_precipitacion":
                not actividad,

            "cobertura_pct":
                round(
                    min(
                        100.0,
                        area_px
                        / (
                            width
                            * height
                        )
                        * 100.0
                    ),
                    2
                ),

            "nucleos_precipitacion":
                len(precip),

            "nucleos_total":
                len(nucleos),

            "georreferenciacion_exacta":
                False,

            "dBZ_disponible":
                False,

            "nota":
                (
                    "La distancia y ETA son referencias "
                    "aproximadas al centro del dominio. "
                    "No representan todavía distancia/ETA "
                    "exacta a Bahía Blanca."
                ),
        },

        "tracking": {

            "tracks_activos":
                len(tracks),

            "track_principal":
                (
                    principal.get(
                        "track_id"
                    )
                    if principal
                    else None
                ),

            "intervalo_minutos":
                seguimiento.get(
                    "intervalo_minutos"
                ),
        },

        "prediccion": {

            "disponible":
                bool(
                    prediccion
                ),

            "radar_timestamp":
                (
                    prediccion.get(
                        "radar_timestamp"
                    )
                    if isinstance(
                        prediccion,
                        dict
                    )
                    else None
                ),

            "resumen":
                (
                    prediccion.get(
                        "resumen",
                        {}
                    )
                    if isinstance(
                        prediccion,
                        dict
                    )
                    else {}
                ),
        },

        "calibracion":
            (
                prediccion.get(
                    "calibracion",
                    {}
                )
                if isinstance(
                    prediccion,
                    dict
                )
                else {}
            ),

        "escala": {

            "ancho_px":
                width,

            "alto_px":
                height,

            "km_por_pixel_estimado":
                round(
                    km_per_px,
                    5
                ),

            "rango_diametro_km_aproximado":
                RADAR_DIAMETRO_KM_APROX,
        },
    }

    SALIDA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    SALIDA.write_text(
        json.dumps(
            salida,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        ),
        encoding="utf-8"
    )

    print(
        "===== ADAPTADOR RMA10 -> INTELLIGENCE ====="
    )

    print(
        "Actividad:",
        actividad
    )

    print(
        "Núcleos precipitación:",
        len(precip)
    )

    print(
        "Tracks activos:",
        len(tracks)
    )

    print(
        "Intensidad RMA10 0-6:",
        intensidad_max
    )

    print(
        "Velocidad estimada km/h:",
        speed_kmh
    )

    print(
        "Dirección grados:",
        direction
    )

    print(
        "Hacia centro del dominio:",
        toward_center
    )

    print(
        "ETA aproximada al centro min:",
        eta_minutos_aprox
    )

    print(
        "dBZ: NO DISPONIBLE"
    )

    print(
        "Salida:",
        SALIDA
    )


if __name__ == "__main__":
    main()
