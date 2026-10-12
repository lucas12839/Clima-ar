#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
from statistics import mean, pstdev
import json
import math


INPUT = Path(
    "data/radar/seguimiento/seguimiento_temporal.json"
)

STATE = Path(
    "data/radar/seguimiento/estado.json"
)

OUTPUT = Path(
    "data/radar/seguimiento/prediccion_evolucion.json"
)

HORIZONTES = (
    10,
    20,
    30
)

MIN_DT = 2.0
MAX_DT = 30.0

SECTORES = [
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSO",
    "SO",
    "OSO",
    "O",
    "ONO",
    "NO",
    "NNO"
]


def clamp(valor, minimo=0.0, maximo=100.0):
    return max(
        minimo,
        min(maximo, float(valor))
    )


def dt(value):

    if not isinstance(value, str):
        return None

    try:
        d = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

        if d.tzinfo is None:
            d = d.replace(
                tzinfo=timezone.utc
            )

        return d.astimezone(
            timezone.utc
        )

    except Exception:
        return None


def sector(grados):

    if grados is None:
        return None

    return SECTORES[
        int(
            (grados + 11.25) / 22.5
        ) % 16
    ]


def bearing(dx, dy):

    if (
        abs(dx) < 1e-9
        and
        abs(dy) < 1e-9
    ):
        return None

    return (
        math.degrees(
            math.atan2(
                dx,
                -dy
            )
        ) % 360
    )


def angle_diff(a, b):

    if a is None or b is None:
        return None

    return abs(
        (
            a - b + 180
        ) % 360 - 180
    )


def num(value):

    try:
        return float(value)
    except Exception:
        return None


def clean_history(track):

    raw = track.get(
        "historial",
        []
    )

    if not isinstance(raw, list):
        return []

    out = {}

    for h in raw:

        if not isinstance(h, dict):
            continue

        fecha = dt(
            h.get("timestamp")
        )

        if not fecha:
            continue

        if (
            h.get("x") is None
            or
            h.get("y") is None
        ):
            continue

        h = dict(h)

        h["timestamp"] = (
            fecha.isoformat()
            .replace("+00:00", "Z")
        )

        out[
            h["timestamp"]
        ] = h

    return sorted(
        out.values(),
        key=lambda h: dt(
            h["timestamp"]
        )
    )


def movements(historial):

    resultado = []

    for anterior, actual in zip(
        historial,
        historial[1:]
    ):

        t1 = dt(
            anterior["timestamp"]
        )

        t2 = dt(
            actual["timestamp"]
        )

        if not t1 or not t2:
            continue

        minutos = (
            t2 - t1
        ).total_seconds() / 60

        if (
            minutos < MIN_DT
            or
            minutos > MAX_DT
        ):
            continue

        dx = (
            float(actual["x"])
            -
            float(anterior["x"])
        )

        dy = (
            float(actual["y"])
            -
            float(anterior["y"])
        )

        vx = dx / minutos
        vy = dy / minutos

        velocidad = math.hypot(
            vx,
            vy
        )

        resultado.append({
            "dt": minutos,
            "dx": dx,
            "dy": dy,
            "vx": vx,
            "vy": vy,
            "speed": velocidad,
            "bearing": bearing(
                dx,
                dy
            )
        })

    return resultado


def area_series(historial):

    valores = []

    for h in historial:

        valor = num(
            h.get("pixeles")
        )

        if (
            valor is not None
            and
            valor > 0
        ):

            valores.append(
                (
                    h["timestamp"],
                    valor
                )
            )

    return valores


def intensity_num(h):

    valor = h.get(
        "nivel_intensidad"
    )

    mapa = {
        "MUY_DEBIL": 1,
        "DEBIL": 2,
        "MODERADA": 3,
        "FUERTE": 4,
        "MUY_FUERTE": 5,
        "SEVERA": 6
    }

    if (
        isinstance(valor, str)
        and
        valor in mapa
    ):
        return mapa[valor]

    return None


def evolution(historial):

    areas = area_series(
        historial
    )

    cambios_area = []

    for (_, anterior), (_, actual) in zip(
        areas,
        areas[1:]
    ):

        if anterior > 0:

            cambios_area.append(
                (
                    actual - anterior
                )
                /
                anterior
                *
                100
            )

    if cambios_area:

        tendencia_area = mean(
            cambios_area[-3:]
        )

    else:

        tendencia_area = None

    intensidades = [
        intensity_num(h)
        for h in historial
    ]

    intensidades = [
        x
        for x in intensidades
        if x is not None
    ]

    if len(intensidades) >= 2:

        tendencia_intensidad = (
            intensidades[-1]
            -
            intensidades[0]
        )

    else:

        tendencia_intensidad = 0

    score = 50.0

    if tendencia_area is not None:

        score += (
            clamp(
                tendencia_area,
                -25,
                25
            )
            *
            0.9
        )

    score += max(
        -18,
        min(
            18,
            tendencia_intensidad * 9
        )
    )

    intensificacion = clamp(
        score
    )

    debilitamiento = clamp(
        100 - intensificacion
    )

    return (
        intensificacion,
        debilitamiento,
        tendencia_area,
        tendencia_intensidad
    )


def predict(
    track,
    current_timestamp
):

    historial = clean_history(
        track
    )

    fecha_actual = dt(
        current_timestamp
    )

    if fecha_actual:

        historial = [
            h
            for h in historial
            if dt(
                h["timestamp"]
            ) <= fecha_actual
        ]

    if not historial:
        return None

    movimientos = movements(
        historial[-6:]
    )

    actual = historial[-1]

    (
        intensificacion,
        debilitamiento,
        tendencia_area,
        tendencia_intensidad
    ) = evolution(
        historial[-6:]
    )

    resultado = {

        "track_id":
            track.get("id"),

        "pronostico_disponible":
            False,

        "posicion_actual": {

            "x":
                round(
                    float(actual["x"]),
                    2
                ),

            "y":
                round(
                    float(actual["y"]),
                    2
                )
        },

        "frames_utilizados":
            len(historial),

        "intervalos_movimiento_validos":
            len(movimientos),

        "probabilidad_intensificacion":
            round(
                intensificacion,
                1
            ),

        "probabilidad_debilitamiento":
            round(
                debilitamiento,
                1
            )
    }

    if not movimientos:

        resultado["motivo"] = (
            "SIN_MOVIMIENTO_VALIDO"
        )

        return resultado

    pesos = list(
        range(
            1,
            len(movimientos) + 1
        )
    )

    total_pesos = sum(
        pesos
    )

    vx = sum(
        m["vx"] * peso
        for m, peso
        in zip(
            movimientos,
            pesos
        )
    ) / total_pesos

    vy = sum(
        m["vy"] * peso
        for m, peso
        in zip(
            movimientos,
            pesos
        )
    ) / total_pesos

    velocidad = math.hypot(
        vx,
        vy
    )

    rumbo_actual = bearing(
        vx,
        vy
    )

    cambios_rumbo = []

    for i in range(
        1,
        len(movimientos)
    ):

        cambio = angle_diff(
            movimientos[i]["bearing"],
            movimientos[i - 1]["bearing"]
        )

        if cambio is not None:
            cambios_rumbo.append(
                cambio
            )

    if cambios_rumbo:

        cambio_rumbo_medio = mean(
            cambios_rumbo
        )

    else:

        cambio_rumbo_medio = 0.0

    velocidades = [
        m["speed"]
        for m in movimientos
    ]

    if len(velocidades) > 1:

        variacion_velocidad = pstdev(
            velocidades
        )

    else:

        variacion_velocidad = 0.0

    consistencia = clamp(
        100
        -
        cambio_rumbo_medio * 0.9
        -
        (
            variacion_velocidad
            /
            max(
                velocidad,
                0.01
            )
        )
        * 20
    )

    confianza_movimiento = num(
        track.get(
            "confianza_movimiento"
        )
    )

    if confianza_movimiento is None:
        confianza_movimiento = 50.0

    if len(movimientos) >= 3:

        confianza_datos = 35.0

    elif len(movimientos) == 2:

        confianza_datos = 22.0

    else:

        confianza_datos = 8.0

    confianza = clamp(
        0.45 * confianza_movimiento
        +
        0.30 * consistencia
        +
        confianza_datos
    )

    aceleracion = None

    if len(movimientos) >= 2:

        aceleracion = (
            movimientos[-1]["speed"]
            -
            movimientos[-2]["speed"]
        ) / max(
            movimientos[-1]["dt"],
            1
        )

    elif num(
        track.get(
            "aceleracion_pixeles_min2"
        )
    ) is not None:

        aceleracion = num(
            track.get(
                "aceleracion_pixeles_min2"
            )
        )

    pronosticos = []

    x0 = float(
        actual["x"]
    )

    y0 = float(
        actual["y"]
    )

    for horizonte in HORIZONTES:

        velocidad_x = vx
        velocidad_y = vy

        if (
            aceleracion is not None
            and
            velocidad > 0.01
        ):

            ux = vx / velocidad
            uy = vy / velocidad

            aceleracion_limitada = max(
                -0.15,
                min(
                    0.15,
                    aceleracion
                )
            )

            velocidad_x = (
                vx
                +
                ux
                *
                aceleracion_limitada
                *
                horizonte
            )

            velocidad_y = (
                vy
                +
                uy
                *
                aceleracion_limitada
                *
                horizonte
            )

        x_estimado = (
            x0
            +
            velocidad_x
            *
            horizonte
        )

        y_estimado = (
            y0
            +
            velocidad_y
            *
            horizonte
        )

        velocidad_estimada = math.hypot(
            velocidad_x,
            velocidad_y
        )

        rumbo_estimado = bearing(
            velocidad_x,
            velocidad_y
        )

        incertidumbre = max(
            2.0,
            horizonte
            *
            max(
                0.05,
                velocidad * 0.35
            )
            +
            horizonte
            /
            10
            *
            cambio_rumbo_medio
            *
            0.08
        )

        pronosticos.append({

            "horizonte_minutos":
                horizonte,

            "x_estimado":
                round(
                    x_estimado,
                    2
                ),

            "y_estimado":
                round(
                    y_estimado,
                    2
                ),

            "velocidad_estimada_pixeles_min":
                round(
                    velocidad_estimada,
                    4
                ),

            "rumbo_estimado_grados":
                (
                    round(
                        rumbo_estimado,
                        1
                    )
                    if rumbo_estimado is not None
                    else None
                ),

            "direccion_estimada":
                sector(
                    rumbo_estimado
                ),

            "incertidumbre_pixeles":
                round(
                    incertidumbre,
                    1
                )
        })

    disponible = (
        len(movimientos) >= 2
    )

    resultado.update({

        "pronostico_disponible":
            disponible,

        "velocidad_actual_pixeles_min":
            round(
                velocidad,
                4
            ),

        "rumbo_actual_grados":
            (
                round(
                    rumbo_actual,
                    1
                )
                if rumbo_actual is not None
                else None
            ),

        "direccion_actual":
            sector(
                rumbo_actual
            ),

        "aceleracion_pixeles_min2":
            (
                round(
                    aceleracion,
                    5
                )
                if aceleracion is not None
                else None
            ),

        "tendencia_area_porcentaje":
            (
                round(
                    tendencia_area,
                    2
                )
                if tendencia_area is not None
                else None
            ),

        "tendencia_intensidad_niveles":
            tendencia_intensidad,

        "consistencia_movimiento":
            round(
                consistencia,
                1
            ),

        "confianza_prediccion":
            round(
                confianza if disponible
                else min(
                    confianza,
                    49
                ),
                1
            ),

        "horizontes":
            pronosticos if disponible else [],

        "motivo":
            (
                None
                if disponible
                else
                "SOLO_UN_INTERVALO_VALIDO"
            )
    })

    return resultado


def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            INPUT
        )

    if not STATE.exists():
        raise FileNotFoundError(
            STATE
        )

    seguimiento = json.loads(
        INPUT.read_text(
            encoding="utf-8"
        )
    )

    estado = json.loads(
        STATE.read_text(
            encoding="utf-8"
        )
    )

    radar_timestamp = (
        estado.get(
            "radar_timestamp"
        )
        or
        seguimiento.get(
            "radar_timestamp"
        )
    )

    intervalo = num(
        seguimiento.get(
            "intervalo_minutos"
        )
    )

    if intervalo is None:
        intervalo = 0.0

    intervalo_valido = (
        MIN_DT
        <= intervalo
        <= MAX_DT
    )

    # SOLAMENTE los núcleos del último frame.
    ids_actuales = set()

    for nucleo in estado.get(
        "nucleos",
        []
    ):

        try:

            ids_actuales.add(
                int(
                    nucleo["track_id"]
                )
            )

        except Exception:
            pass

    tracks_por_id = {}

    for track in estado.get(
        "tracks",
        []
    ):

        try:

            tracks_por_id[
                int(track["id"])
            ] = track

        except Exception:
            pass

    tracks_actuales = []

    for track_id in sorted(
        ids_actuales
    ):

        track = tracks_por_id.get(
            track_id
        )

        if not track:
            continue

        if not track.get(
            "activo",
            False
        ):
            continue

        tracks_actuales.append(
            track
        )

    resultado = {

        "version": 2,

        "radar_timestamp":
            radar_timestamp,

        "intervalo_minutos":
            intervalo,

        "tipo":
            "PREDICCION_HEURISTICA_MULTIFRAME",

        "prediccion_global_disponible":
            intervalo_valido,

        "nucleos": []
    }

    if not intervalo_valido:

        resultado["motivo"] = (
            "Intervalo temporal no válido para extrapolación."
        )

    else:

        for track in tracks_actuales:

            prediccion = predict(
                track,
                radar_timestamp
            )

            if prediccion:
                resultado[
                    "nucleos"
                ].append(
                    prediccion
                )

        disponibles = [
            p
            for p in resultado["nucleos"]
            if p.get(
                "pronostico_disponible"
            )
        ]

        resultado["resumen"] = {

            "nucleos_actuales":
                len(tracks_actuales),

            "nucleos_evaluados":
                len(
                    resultado["nucleos"]
                ),

            "nucleos_con_prediccion":
                len(disponibles),

            "nucleos_sin_prediccion":
                (
                    len(
                        resultado["nucleos"]
                    )
                    -
                    len(disponibles)
                ),

            "promedio_intensificacion":
                (
                    round(
                        mean(
                            p[
                                "probabilidad_intensificacion"
                            ]
                            for p in disponibles
                        ),
                        1
                    )
                    if disponibles
                    else None
                ),

            "promedio_debilitamiento":
                (
                    round(
                        mean(
                            p[
                                "probabilidad_debilitamiento"
                            ]
                            for p in disponibles
                        ),
                        1
                    )
                    if disponibles
                    else None
                ),

            "promedio_confianza":
                (
                    round(
                        mean(
                            p[
                                "confianza_prediccion"
                            ]
                            for p in disponibles
                        ),
                        1
                    )
                    if disponibles
                    else None
                )
        }

        # Movimiento global del sistema.
        if disponibles:

            vectores_x = []
            vectores_y = []

            for p in disponibles:

                velocidad = (
                    p.get(
                        "velocidad_actual_pixeles_min"
                    )
                    or 0
                )

                grados = p.get(
                    "rumbo_actual_grados"
                )

                if grados is None:
                    continue

                rad = math.radians(
                    grados
                )

                vectores_x.append(
                    velocidad
                    *
                    math.sin(rad)
                )

                vectores_y.append(
                    -velocity
                    *
                    math.cos(rad)
                )

            if vectores_x:

                vx = mean(
                    vectores_x
                )

                vy = mean(
                    vectores_y
                )

                velocidad_sistema = math.hypot(
                    vx,
                    vy
                )

                rumbo_sistema = bearing(
                    vx,
                    vy
                )

                resultado[
                    "movimiento_sistema"
                ] = {

                    "velocidad_media_vectorial_pixeles_min":
                        round(
                            velocidad_sistema,
                            4
                        ),

                    "rumbo_grados":
                        (
                            round(
                                rumbo_sistema,
                                1
                            )
                            if rumbo_sistema is not None
                            else None
                        ),

                    "direccion":
                        sector(
                            rumbo_sistema
                        ),

                    "nucleos_utilizados":
                        len(
                            vectores_x
                        )
                }

    resultado["limitacion"] = (
        "Predicción heurística. "
        "Velocidad y posición permanecen "
        "en píxeles hasta georreferenciar RMA10."
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT.write_text(
        json.dumps(
            resultado,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        f"OK: {OUTPUT}"
    )

    print(
        "Núcleos actuales evaluados: "
        f"{len(resultado['nucleos'])}"
    )


if __name__ == "__main__":
    main()
