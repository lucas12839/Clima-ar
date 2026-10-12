#!/usr/bin/env python3

"""
Predicción heurística de evolución de núcleos RMA10 para ClimaAR.

Todavía NO es IA meteorológica:
extrapola la trayectoria observada y combina:
- movimiento
- velocidad
- aceleración
- cambio de área
- tendencia de crecimiento/debilitamiento
- dirección

La velocidad permanece en píxeles/minuto hasta disponer
de georreferenciación.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean


INPUT = Path(
    "data/radar/seguimiento/seguimiento_temporal.json"
)

OUTPUT = Path(
    "data/radar/seguimiento/prediccion_evolucion.json"
)

HORIZONTES_MIN = (10, 20, 30)


def clamp(valor, minimo=0.0, maximo=100.0):
    return max(minimo, min(maximo, float(valor)))


def sector_meteorologico(grados):
    if grados is None:
        return None

    sectores = [
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
        "NNO",
    ]

    indice = int((grados % 360) / 22.5 + 0.5) % 16

    return sectores[indice]


def diferencia_angulos(a, b):
    if a is None or b is None:
        return None

    return abs((a - b + 180) % 360 - 180)


def obtener_historial_reciente(track, cantidad=4):

    historial = [
        entrada
        for entrada in track.get("historial", [])
        if (
            entrada.get("timestamp")
            and entrada.get("x") is not None
            and entrada.get("y") is not None
        )
    ]

    historial.sort(
        key=lambda entrada: entrada["timestamp"]
    )

    return historial[-cantidad:]


def calcular_movimientos(historial):

    movimientos = []

    for anterior, actual in zip(
        historial,
        historial[1:]
    ):

        try:

            tiempo_anterior = datetime.fromisoformat(
                anterior["timestamp"].replace("Z", "+00:00")
            )

            tiempo_actual = datetime.fromisoformat(
                actual["timestamp"].replace("Z", "+00:00")
            )

            intervalo = (
                tiempo_actual - tiempo_anterior
            ).total_seconds() / 60.0

            # Solo aceptamos intervalos temporales válidos.
            if intervalo < 2 or intervalo > 30:
                continue

            dx = (
                float(actual["x"])
                - float(anterior["x"])
            )

            dy = (
                float(actual["y"])
                - float(anterior["y"])
            )

            velocidad = (
                math.hypot(dx, dy)
                / intervalo
            )

            rumbo = (
                math.degrees(
                    math.atan2(dx, -dy)
                )
                % 360
            )

            movimientos.append(
                {
                    "intervalo": intervalo,
                    "vx": dx / intervalo,
                    "vy": dy / intervalo,
                    "velocidad": velocidad,
                    "rumbo": rumbo,
                }
            )

        except Exception:
            continue

    return movimientos


def predecir_nucleo(track):

    historial = obtener_historial_reciente(track)

    movimientos = calcular_movimientos(historial)

    if not historial:
        return None

    actual = historial[-1]

    # Sin movimiento temporal válido no extrapolamos.
    if not movimientos:

        return {
            "track_id": track.get("id"),
            "pronostico_disponible": False,
            "motivo": "SIN_MOVIMIENTO_VALIDO",
            "posicion_actual": {
                "x": actual["x"],
                "y": actual["y"],
            },
        }

    # Damos mayor peso a los movimientos más recientes.
    pesos = list(
        range(1, len(movimientos) + 1)
    )

    peso_total = sum(pesos)

    vx = (
        sum(
            movimiento["vx"] * peso
            for movimiento, peso
            in zip(movimientos, pesos)
        )
        / peso_total
    )

    vy = (
        sum(
            movimiento["vy"] * peso
            for movimiento, peso
            in zip(movimientos, pesos)
        )
        / peso_total
    )

    velocidad = math.hypot(vx, vy)

    if velocidad > 0.01:

        rumbo = (
            math.degrees(
                math.atan2(vx, -vy)
            )
            % 360
        )

    else:
        rumbo = None

    # Aceleración.
    aceleracion = track.get(
        "aceleracion_pixeles_min2"
    )

    if (
        aceleracion is None
        and len(movimientos) >= 2
    ):

        aceleracion = (
            movimientos[-1]["velocidad"]
            - movimientos[-2]["velocidad"]
        ) / max(
            movimientos[-1]["intervalo"],
            1
        )

    # Tendencia de área.
    cambios_area = []

    for entrada in historial[1:]:

        valor = entrada.get(
            "cambio_area_porcentaje"
        )

        if valor is not None:

            try:
                cambios_area.append(
                    float(valor)
                )
            except Exception:
                pass

    if cambios_area:

        tendencia_area = mean(
            cambios_area[-3:]
        )

    else:

        tendencia_area = None

    # Probabilidad heurística de intensificación.
    intensificacion = 50.0

    if tendencia_area is not None:

        intensificacion += max(
            -25,
            min(
                25,
                tendencia_area * 0.8
            )
        )

    if aceleracion is not None:

        intensificacion += max(
            -15,
            min(
                15,
                aceleracion * 300
            )
        )

    intensificacion = clamp(
        intensificacion
    )

    debilitamiento = clamp(
        100 - intensificacion
    )

    # Confianza del movimiento.
    confianza_movimiento = float(
        track.get(
            "confianza_movimiento",
            50
        )
        or 50
    )

    cambios_rumbo = []

    for indice in range(
        1,
        len(movimientos)
    ):

        cambio = diferencia_angulos(
            movimientos[indice]["rumbo"],
            movimientos[indice - 1]["rumbo"],
        )

        if cambio is not None:
            cambios_rumbo.append(cambio)

    if cambios_rumbo:

        consistencia = clamp(
            100
            - (
                mean(cambios_rumbo)
                * 0.25
            )
        )

    else:

        consistencia = 60.0

    confianza = clamp(
        0.55 * confianza_movimiento
        + 0.30 * consistencia
        + 15
        * min(len(movimientos), 3)
        / 3
    )

    # Posición actual.
    x_actual = float(actual["x"])
    y_actual = float(actual["y"])

    pronosticos = []

    for horizonte in HORIZONTES_MIN:

        vx_futuro = vx
        vy_futuro = vy

        # Solo aplicamos aceleración pequeña
        # y en la dirección actual.
        if (
            aceleracion is not None
            and abs(float(aceleracion)) < 0.15
            and velocidad > 0.01
        ):

            ux = vx / velocidad
            uy = vy / velocidad

            vx_futuro += (
                ux
                * float(aceleracion)
                * horizonte
            )

            vy_futuro += (
                uy
                * float(aceleracion)
                * horizonte
            )

        x_futuro = (
            x_actual
            + vx_futuro * horizonte
        )

        y_futuro = (
            y_actual
            + vy_futuro * horizonte
        )

        velocidad_futura = math.hypot(
            vx_futuro,
            vy_futuro
        )

        if velocidad_futura > 0.01:

            rumbo_futuro = (
                math.degrees(
                    math.atan2(
                        vx_futuro,
                        -vy_futuro
                    )
                )
                % 360
            )

        else:

            rumbo_futuro = None

        pronosticos.append(
            {
                "horizonte_minutos": horizonte,
                "x_estimado": round(
                    x_futuro,
                    2
                ),
                "y_estimado": round(
                    y_futuro,
                    2
                ),
                "velocidad_estimada_pixeles_min": round(
                    velocidad_futura,
                    4
                ),
                "rumbo_estimado_grados": (
                    round(
                        rumbo_futuro,
                        1
                    )
                    if rumbo_futuro is not None
                    else None
                ),
                "direccion_estimada": (
                    sector_meteorologico(
                        rumbo_futuro
                    )
                ),
            }
        )

    return {
        "track_id": track.get("id"),
        "pronostico_disponible": True,

        "posicion_actual": {
            "x": round(x_actual, 2),
            "y": round(y_actual, 2),
        },

        "velocidad_actual_pixeles_min": round(
            velocidad,
            4
        ),

        "rumbo_actual_grados": (
            round(rumbo, 1)
            if rumbo is not None
            else None
        ),

        "direccion_actual": (
            sector_meteorologico(rumbo)
        ),

        "aceleracion_pixeles_min2": (
            round(
                float(aceleracion),
                5
            )
            if aceleracion is not None
            else None
        ),

        "tendencia_area_porcentaje": (
            round(
                tendencia_area,
                2
            )
            if tendencia_area is not None
            else None
        ),

        "probabilidad_intensificacion": round(
            intensificacion,
            1
        ),

        "probabilidad_debilitamiento": round(
            debilitamiento,
            1
        ),

        "confianza_prediccion": round(
            confianza,
            1
        ),

        "horizontes": pronosticos,

        "advertencia": (
            "Velocidad y posición están "
            "en píxeles. Falta "
            "georreferenciación para "
            "convertir a km/h y distancia real."
        ),
    }


def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"No existe {INPUT}"
        )

    datos = json.loads(
        INPUT.read_text(
            encoding="utf-8"
        )
    )

    intervalo = float(
        datos.get(
            "intervalo_minutos",
            0
        )
        or 0
    )

    resultado = {

        "version": 1,

        "radar_timestamp":
            datos.get(
                "radar_timestamp"
            ),

        "intervalo_minutos":
            intervalo,

        "tipo":
            "PREDICCION_HEURISTICA",

        "prediccion_global_disponible":
            2 <= intervalo <= 30,

        "nucleos": [],
    }

    if not resultado[
        "prediccion_global_disponible"
    ]:

        resultado["motivo"] = (
            "Intervalo temporal no válido "
            "para extrapolación."
        )

    else:

        for track in datos.get(
            "tracks",
            []
        ):

            if not track.get(
                "activo"
            ):
                continue

            prediccion = (
                predecir_nucleo(track)
            )

            if prediccion:
                resultado[
                    "nucleos"
                ].append(
                    prediccion
                )

        disponibles = [
            nucleo
            for nucleo
            in resultado["nucleos"]
            if nucleo.get(
                "pronostico_disponible"
            )
        ]

        resultado["resumen"] = {

            "nucleos_con_prediccion":
                len(disponibles),

            "nucleos_sin_prediccion":
                len(resultado["nucleos"])
                - len(disponibles),

            "promedio_intensificacion":
                round(
                    mean(
                        nucleo[
                            "probabilidad_intensificacion"
                        ]
                        for nucleo
                        in disponibles
                    ),
                    1
                )
                if disponibles
                else None,

            "promedio_debilitamiento":
                round(
                    mean(
                        nucleo[
                            "probabilidad_debilitamiento"
                        ]
                        for nucleo
                        in disponibles
                    ),
                    1
                )
                if disponibles
                else None,

            "promedio_confianza":
                round(
                    mean(
                        nucleo[
                            "confianza_prediccion"
                        ]
                        for nucleo
                        in disponibles
                    ),
                    1
                )
                if disponibles
                else None,
        }

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


if __name__ == "__main__":
    main()
