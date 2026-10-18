#!/usr/bin/env python3
"""ClimaAR - Adaptador operativo RMA10 -> Storm Engine."""

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
TIMESTAMP = BASE / "radar_timestamp.txt"

SALIDA = BASE / "radar_nowcast.json"


RADAR_DIAMETRO_KM_APROX = 240.0
MAX_TRACK_MATCH_DISTANCE_PX = 45.0
MAX_INTERVALO_TRACKING = 200.0

ESTADOS_EXCLUIDOS = {
    "ECO_ESTATICO",
    "SIN_ECO",
}


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


def clamp(
    value,
    low=0.0,
    high=1.0
):
    value = number(
        value,
        low
    )

    return max(
        low,
        min(high, value)
    )


def parse_ts(value):
    if not value:
        return None

    try:
        text = str(
            value
        ).strip()

        text = text.replace(
            "Z",
            "+00:00"
        )

        dt = datetime.fromisoformat(
            text
        )

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt.astimezone(
            timezone.utc
        )

    except Exception:
        return None


def iso(dt):
    return (
        dt.astimezone(
            timezone.utc
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z"
        )
    )


def radar_timestamp(resultado):

    if TIMESTAMP.exists():

        value = TIMESTAMP.read_text(
            encoding="utf-8"
        ).strip()

        dt = parse_ts(
            value
        )

        if dt:
            return iso(dt)

    for key in (
        "radar_timestamp",
        "frame_time",
        "timestamp_radar",
    ):

        dt = parse_ts(
            resultado.get(key)
        )

        if dt:
            return iso(dt)

    return iso(
        datetime.now(
            timezone.utc
        )
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

        result.append(
            track
        )

    return result


def track_current_position(track):

    history = track.get(
        "historial",
        []
    )

    if (
        isinstance(
            history,
            list
        )
        and history
    ):

        current = history[-1]

        if isinstance(
            current,
            dict
        ):

            if (
                current.get("x")
                is not None
                and
                current.get("y")
                is not None
            ):

                return (
                    number(
                        current.get("x")
                    ),
                    number(
                        current.get("y")
                    )
                )

    if (
        track.get("x")
        is not None
        and
        track.get("y")
        is not None
    ):

        return (
            number(
                track.get("x")
            ),
            number(
                track.get("y")
            )
        )

    return None


def track_motion(track):

    history = track.get(
        "historial",
        []
    )

    if (
        not isinstance(
            history,
            list
        )
        or
        len(history) < 2
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
        not isinstance(
            previous,
            dict
        )
        or
        not isinstance(
            current,
            dict
        )
    ):
        return (
            0.0,
            0.0,
            0.0,
            None
        )

    t0 = parse_ts(
        previous.get(
            "timestamp"
        )
    )

    t1 = parse_ts(
        current.get(
            "timestamp"
        )
    )

    if not t0 or not t1:
        return (
            0.0,
            0.0,
            0.0,
            None
        )

    dt_min = (
        t1 - t0
    ).total_seconds() / 60.0

    if (
        dt_min <= 0
        or
        dt_min > MAX_INTERVALO_TRACKING
    ):
        return (
            0.0,
            0.0,
            0.0,
            None
        )

    dx = (
        number(
            current.get("x")
        )
        -
        number(
            previous.get("x")
        )
    )

    dy = (
        number(
            current.get("y")
        )
        -
        number(
            previous.get("y")
        )
    )

    distance_px = math.hypot(
        dx,
        dy
    )

    speed_px_min = (
        distance_px
        /
        dt_min
    )

    direction = None

    if distance_px > 0:

        direction = (
            math.degrees(
                math.atan2(
                    dx,
                    -dy
                )
            )
            % 360.0
        )

    return (
        dx,
        dy,
        speed_px_min,
        direction
    )


def area_trend(track):

    history = track.get(
        "historial",
        []
    )

    if (
        not isinstance(
            history,
            list
        )
        or
        len(history) < 2
    ):
        return (
            "estable",
            0.0
        )

    previous = number(
        history[-2].get(
            "pixeles"
        ),
        0
    )

    current = number(
        history[-1].get(
            "pixeles"
        ),
        0
    )

    if previous <= 0:
        return (
            "estable",
            0.0
        )

    change = (
        (
            current
            -
            previous
        )
        /
        previous
        *
        100.0
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


def associate_tracks(
    nucleos,
    tracks
):

    enriched = []

    used = set()

    for index, nucleo in enumerate(
        nucleos
    ):

        item = dict(
            nucleo
        )

        item["nucleo_id"] = (
            item.get(
                "nucleo_id"
            )
            or
            item.get(
                "id"
            )
            or
            index + 1
        )

        x = item.get(
            "x"
        )

        y = item.get(
            "y"
        )

        best_index = None
        best_track = None
        best_distance = None

        if (
            x is not None
            and
            y is not None
        ):

            nx = number(x)
            ny = number(y)

            for (
                track_index,
                track
            ) in enumerate(
                tracks
            ):

                if track_index in used:
                    continue

                position = (
                    track_current_position(
                        track
                    )
                )

                if position is None:
                    continue

                tx, ty = position

                distance = math.hypot(
                    nx - tx,
                    ny - ty
                )

                if (
                    best_distance is None
                    or
                    distance < best_distance
                ):

                    best_index = (
                        track_index
                    )

                    best_track = (
                        track
                    )

                    best_distance = (
                        distance
                    )

        if (
            best_track is not None
            and
            best_distance
            <=
            MAX_TRACK_MATCH_DISTANCE_PX
        ):

            used.add(
                best_index
            )

            (
                dx,
                dy,
                speed,
                direction
            ) = track_motion(
                best_track
            )

            trend, change = (
                area_trend(
                    best_track
                )
            )

            item["track_id"] = (
                best_track.get(
                    "track_id",
                    best_track.get(
                        "id"
                    )
                )
            )

            item[
                "track_match_distance_px"
            ] = round(
                best_distance,
                2
            )

            item["movimiento"] = {

                "dx_px":
                    round(
                        dx,
                        4
                    ),

                "dy_px":
                    round(
                        dy,
                        4
                    ),

                "velocidad_px_min":
                    round(
                        speed,
                        5
                    ),

                "direccion_grados":
                    (
                        round(
                            direction,
                            2
                        )
                        if direction
                        is not None
                        else None
                    ),

                "tendencia":
                    trend,

                "cambio_area_pct":
                    round(
                        change,
                        2
                    ),

                "confianza_track":
                    round(
                        clamp(
                            number(
                                best_track.get(
                                    "confianza_movimiento",
                                    0
                                ),
                                0
                            ),
                            0,
                            1
                        ),
                        3
                    ),
            }

        else:

            item["track_id"] = None

            item[
                "track_match_distance_px"
            ] = None

            item["movimiento"] = {

                "dx_px": 0.0,

                "dy_px": 0.0,

                "velocidad_px_min": 0.0,

                "direccion_grados": None,

                "tendencia":
                    "sin_seguimiento",

                "cambio_area_pct":
                    0.0,

                "confianza_track":
                    0.0,
            }

        enriched.append(
            item
        )

    return enriched


def select_compatibility_track(
    tracks
):

    if not tracks:
        return None

    return max(
        tracks,
        key=lambda track:
            max(
                0.0,
                number(
                    track.get(
                        "pixeles"
                    ),
                    0
                )
            )
            *
            max(
                1.0,
                number(
                    track.get(
                        "intensidad"
                    ),
                    0
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
            "ERROR: resultado_procesamiento.json invalido"
        )

    if not isinstance(
        estado,
        dict
    ):
        raise SystemExit(
            "ERROR: estado.json invalido"
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

    km_per_px = (
        RADAR_DIAMETRO_KM_APROX
        /
        max(
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

    # El procesador RMA10 entrega
    # DINAMICO/CAMBIO.
    #
    # Los ecos estaticos ya fueron
    # eliminados antes de esta etapa.

    nucleos_dinamicos = [

        n

        for n in nucleos

        if (
            isinstance(
                n,
                dict
            )
            and
            str(
                n.get(
                    "estado",
                    ""
                )
            ).upper()
            not in
            ESTADOS_EXCLUIDOS
        )
    ]

    tracks = active_tracks(
        estado
    )

    nucleos_con_track = (
        associate_tracks(
            nucleos_dinamicos,
            tracks
        )
    )

    actividad = bool(
        nucleos_con_track
    )

    area_px = sum(
        max(
            0.0,
            number(
                n.get(
                    "pixeles"
                ),
                0
            )
        )
        for n
        in nucleos_con_track
    )

    intensidad_max = max(

        (
            int(
                number(
                    n.get(
                        "intensidad"
                    ),
                    0
                )
            )

            for n
            in nucleos_con_track
        ),

        default=0
    )

    fortaleciendo = sum(

        1

        for n
        in nucleos_con_track

        if (
            n.get(
                "movimiento",
                {}
            ).get(
                "tendencia"
            )
            ==
            "fortaleciendose"
        )
    )

    debilitando = sum(

        1

        for n
        in nucleos_con_track

        if (
            n.get(
                "movimiento",
                {}
            ).get(
                "tendencia"
            )
            ==
            "debilitandose"
        )
    )

    cambios = [

        number(
            n.get(
                "movimiento",
                {}
            ).get(
                "cambio_area_pct"
            ),
            0
        )

        for n
        in nucleos_con_track

        if (
            n.get(
                "movimiento",
                {}
            ).get(
                "tendencia"
            )
            !=
            "sin_seguimiento"
        )
    ]

    cambio_area_pct = (

        sum(cambios)
        /
        len(cambios)

        if cambios

        else 0.0
    )

    compat_track = (
        select_compatibility_track(
            tracks
        )
    )

    if compat_track:

        (
            _,
            _,
            compat_speed_px_min,
            compat_direction
        ) = track_motion(
            compat_track
        )

        compat_speed_kmh = (

            compat_speed_px_min
            *
            km_per_px
            *
            60.0

            if compat_speed_px_min > 0

            else None
        )

        compat_trend, _ = (
            area_trend(
                compat_track
            )
        )

        compat_confidence = (
            clamp(
                number(
                    compat_track.get(
                        "confianza_movimiento",
                        0
                    ),
                    0
                ),
                0,
                1
            )
        )

    else:

        compat_speed_px_min = 0.0
        compat_speed_kmh = None
        compat_direction = None
        compat_trend = "estable"
        compat_confidence = 0.0

    intervalo_raw = number(
        seguimiento.get(
            "intervalo_minutos"
        ),
        0
    )

    if (
        intervalo_raw <= 0
        or
        intervalo_raw
        >
        MAX_INTERVALO_TRACKING
    ):

        intervalo_publico = None

    else:

        intervalo_publico = round(
            intervalo_raw,
            2
        )

    calidad_intervalo = (
        seguimiento.get(
            "calidad_intervalo"
        )
    )

    if (
        intervalo_raw
        >
        MAX_INTERVALO_TRACKING
    ):

        calidad_intervalo = (
            "GAP_MAYOR_60_MIN"
        )

    radar_ts = radar_timestamp(
        resultado
    )

    salida = {

        "version":
            "RMA10-operativo-2.1",

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
                len(
                    nucleos_dinamicos
                ),

            "tiles_ok":
                True,

            "actividad":
                actividad,

            "area_px":
                round(
                    area_px,
                    2
                ),

            "intensidad_nivel_max":
                intensidad_max,

            "intensidad_escala":
                "RMA10_0_6",

            "dbz_max":
                None,

            "dbz_mean":
                None,

            "distancia_km":
                None,

            "distancia_tipo":
                "pendiente_georreferenciacion_por_nucleo",

            "fortalecimiento":
                compat_trend,

            "cambio_area_pct":
                round(
                    cambio_area_pct,
                    2
                ),

            "velocidad_kmh":
                (
                    round(
                        compat_speed_kmh,
                        1
                    )
                    if compat_speed_kmh
                    is not None
                    else None
                ),

            "velocidad_pixeles_min":
                round(
                    compat_speed_px_min,
                    5
                ),

            "direccion":
                compat_direction,

            "direccion_grados":
                (
                    round(
                        compat_direction,
                        2
                    )
                    if compat_direction
                    is not None
                    else None
                ),

            "movimiento_hacia_bahia":
                False,

            "movimiento_hacia_centro_dominio":
                False,

            "eta_minutos":
                None,

            "eta_minutos_aproximada_centro_dominio":
                None,

            "confianza_movimiento":
                round(
                    compat_confidence,
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
                nucleos_con_track,

            "nucleos_analizados":
                nucleos_con_track,

            "nucleo_principal_id":
                (
                    compat_track.get(
                        "track_id",
                        compat_track.get(
                            "id"
                        )
                    )
                    if compat_track
                    else None
                ),

            "resumen_nucleos": {

                "total_detectados":
                    len(
                        nucleos_con_track
                    ),

                "con_track":
                    sum(
                        1
                        for n
                        in nucleos_con_track
                        if n.get(
                            "track_id"
                        )
                        is not None
                    ),

                "sin_track":
                    sum(
                        1
                        for n
                        in nucleos_con_track
                        if n.get(
                            "track_id"
                        )
                        is None
                    ),

                "fortaleciendose":
                    fortaleciendo,

                "debilitandose":
                    debilitando,
            },

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
                        /
                        (
                            width
                            *
                            height
                        )
                        *
                        100.0
                    ),
                    2
                ),

            "nucleos_precipitacion":
                len(
                    nucleos_con_track
                ),

            "nucleos_total":
                len(
                    nucleos
                ),

            "tracks_activos":
                len(
                    tracks
                ),

            "georreferenciacion_exacta":
                False,

            "georreferenciacion_operativa":
                False,

            "dBZ_disponible":
                False,

            "nota":
                (
                    "Los ecos estaticos se "
                    "excluyen antes del tracking "
                    "y del ML. Los nucleos "
                    "DINAMICO/CAMBIO pasan "
                    "a Intelligence."
                ),
        },

        "tracking": {

            "tracks_activos":
                len(
                    tracks
                ),

            "track_principal":
                (
                    compat_track.get(
                        "track_id",
                        compat_track.get(
                            "id"
                        )
                    )
                    if compat_track
                    else None
                ),

            "intervalo_minutos":
                intervalo_publico,

            "calidad_intervalo":
                calidad_intervalo,
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
        "Frame radar:",
        radar_ts
    )

    print(
        "Actividad:",
        actividad
    )

    print(
        "Nucleos dinamicos:",
        len(
            nucleos_con_track
        )
    )

    print(
        "Tracks activos:",
        len(
            tracks
        )
    )

    print(
        "Intensidad RMA10 0-6:",
        intensidad_max
    )

    print(
        "Intervalo tracking:",
        intervalo_publico
    )

    print(
        "Calidad intervalo:",
        calidad_intervalo
    )

    print(
        "Salida:",
        SALIDA
    )


if __name__ == "__main__":
    main()
