#!/usr/bin/env python3
"""ClimaAR - Bloque 2: puente de tracking robusto.

Mantiene seguimiento_temporal.py como motor estable y corrige alrededor de
sus interfaces dos problemas reales detectados en auditoría:
1) los tracks que pierden uno o dos frames no podían volver a asociarse;
2) los tracks perdidos no incrementaban correctamente missed_frames cuando
   había más de un frame sin detecciones.

También publica un candidato de núcleo principal basado en persistencia,
movimiento, coherencia y aproximación a Bahía Blanca, sin usar intensidad
sola como criterio.
"""

from __future__ import annotations

import importlib.util
import math
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ORIGINAL = BASE / "seguimiento_temporal.py"

RADAR_LAT = -38.7342
RADAR_LON = -62.1654
BAHIA_LAT = -38.7196
BAHIA_LON = -62.2724
RADAR_RANGE_KM = 240.0


def load_original():
    spec = importlib.util.spec_from_file_location(
        "climaar_seguimiento_temporal_original",
        ORIGINAL,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar {ORIGINAL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_ts(value):
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


def number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def track_anterior(track):
    return {
        "x": number(track.get("x")),
        "y": number(track.get("y")),
        "pixeles": max(1, int(number(track.get("pixeles"), 1))),
        "ancho": max(1.0, number(track.get("ancho"), 1)),
        "alto": max(1.0, number(track.get("alto"), 1)),
    }


def calidad(intervalo):
    if intervalo < 5:
        return "BUENA"
    if intervalo <= 15:
        return "ACEPTABLE"
    return "BAJA"


def recover_missed_tracks(module, estado, nucleos, radar_ts, anterior_ts):
    """Reasocia tracks con missed_frames > 0 usando el intervalo real.

    Devuelve (nucleos_restantes, tracks_recuperados).
    """
    actual_dt = parse_ts(radar_ts)
    anterior_dt = parse_ts(anterior_ts)
    if actual_dt is None or anterior_dt is None:
        return nucleos, []

    candidatos = []
    tracks = estado.get("tracks", [])
    if not isinstance(tracks, list):
        return nucleos, []

    for track in tracks:
        if not isinstance(track, dict) or track.get("activo") is not True:
            continue

        missed = int(number(track.get("missed_frames"), 0))
        if missed <= 0:
            continue

        last_dt = parse_ts(track.get("ultimo_frame"))
        if last_dt is None or last_dt >= actual_dt:
            continue

        gap = (actual_dt - last_dt).total_seconds() / 60.0

        if gap < module.MIN_INTERVALO or gap > module.MAX_INTERVALO:
            continue

        anterior = track_anterior(track)

        for indice, nucleo in enumerate(nucleos):
            comp = module.compatibilidad(
                anterior,
                nucleo,
                gap,
                track,
            )

            if comp is None:
                continue

            # Para recuperación exigimos una asociación algo más fuerte que
            # una asociación normal, porque hay más incertidumbre temporal.
            if comp["score"] < 50.0:
                continue

            candidatos.append(
                {
                    "track": track,
                    "indice": indice,
                    "intervalo": gap,
                    **comp,
                }
            )

    candidatos.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    usados_tracks = set()
    usados_nucleos = set()
    recuperados = []

    for item in candidatos:
        tid = item["track"]["track_id"]
        indice = item["indice"]

        if tid in usados_tracks or indice in usados_nucleos:
            continue

        track = item["track"]
        nucleo = nucleos[indice]

        module.actualizar_track(
            track,
            nucleo,
            radar_ts,
            item["intervalo"],
            calidad(item["intervalo"]),
        )

        track["evento_ultimo"] = "RECUPERADO_DESPUES_DE_MISSED"
        track["origen_evento"] = "RECUPERACION"

        track["confianza_movimiento"] = min(
            0.95,
            max(
                number(track.get("confianza_movimiento"), 0.0),
                0.60 + min(
                    0.25,
                    item["score"] / 400.0,
                ),
            ),
        )

        usados_tracks.add(tid)
        usados_nucleos.add(indice)

        recuperados.append(
            {
                "track": track,
                "indice": indice,
                "score": item["score"],
                "intervalo": item["intervalo"],
            }
        )

    restantes = [
        nucleo
        for indice, nucleo in enumerate(nucleos)
        if indice not in usados_nucleos
    ]

    return restantes, recuperados


def advance_empty_frame(module, estado, radar_ts, anterior_ts):
    """Maneja un frame sin núcleos y cuenta correctamente cada missed frame."""
    actual_dt = parse_ts(radar_ts)
    anterior_dt = parse_ts(anterior_ts)

    activos = []

    for track in estado.get("tracks", []):
        if not isinstance(track, dict):
            continue

        if track.get("activo") is not True:
            continue

        last_dt = parse_ts(track.get("ultimo_frame"))

        if last_dt is None or actual_dt is None:
            track["activo"] = False
            track["presente_en_ultimo_frame"] = False
            continue

        if anterior_dt is not None and last_dt > actual_dt:
            continue

        track["presente_en_ultimo_frame"] = False

        track["missed_frames"] = (
            int(number(track.get("missed_frames"), 0)) + 1
        )

        track["evento_ultimo"] = "MISSED_FRAME"

        if track["missed_frames"] > module.MAX_MISSED_FRAMES:
            track["activo"] = False
        else:
            track["activo"] = True
            activos.append(track["track_id"])

    estado["tracks_activos"] = activos
    estado["ultimo_frame_nucleos"] = []
    estado["ultimo_radar_timestamp"] = radar_ts

    intervalo = 0

    if (
        actual_dt is not None
        and anterior_dt is not None
    ):
        intervalo = (
            actual_dt - anterior_dt
        ).total_seconds() / 60.0

    return module.salida_sin_nucleos(
        radar_ts,
        intervalo,
        "FRAME_SIN_NUCLEOS",
        "No se detectaron núcleos no-estáticos en este frame.",
    )


def select_principal_candidate(estado, radar_timestamp):
    """Selecciona un candidato principal por evidencia, no por intensidad sola."""

    tracks = [
        t
        for t in estado.get("tracks", [])
        if isinstance(t, dict)
        and t.get("activo") is True
        and t.get("presente_en_ultimo_frame") is True
        and t.get("ultimo_frame") == radar_timestamp
    ]

    if not tracks:
        return None

    # Conversión aproximada local suficiente para ordenar la dirección hacia
    # Bahía; la georreferenciación definitiva sigue siendo responsabilidad de
    # georreferenciar_rma10.py.
    km_lat = 110.574
    km_lon = 111.320 * math.cos(
        math.radians(RADAR_LAT)
    )

    target_east = (
        BAHIA_LON - RADAR_LON
    ) * km_lon

    target_north = (
        BAHIA_LAT - RADAR_LAT
    ) * km_lat

    km_px = RADAR_RANGE_KM / 256.0

    target_x = (
        256.0
        + target_east / km_px
    )

    target_y = (
        256.0
        - target_north / km_px
    )

    scored = []

    for track in tracks:
        x = number(
            track.get("x"),
            256,
        )

        y = number(
            track.get("y"),
            256,
        )

        distance_target_px = math.hypot(
            x - target_x,
            y - target_y,
        )

        proximity = max(
            0.0,
            1.0
            - min(
                distance_target_px,
                362.0,
            ) / 362.0,
        )

        speed = number(
            track.get("velocidad_pixeles_min")
        )

        displacement = number(
            track.get("desplazamiento_pixeles")
        )

        motion_conf = max(
            0.0,
            min(
                1.0,
                number(
                    track.get(
                        "confianza_movimiento"
                    )
                ),
            ),
        )

        persistence = min(
            1.0,
            len(
                track.get(
                    "historial",
                    [],
                )
            ) / 6.0,
        )

        area_trend = number(
            track.get(
                "cambio_area_porcentaje"
            )
        )

        growth = max(
            0.0,
            min(
                1.0,
                (area_trend + 30.0) / 60.0,
            ),
        )

        movement_score = 0.0

        direction = track.get(
            "rumbo_grados"
        )

        if (
            direction is not None
            and displacement
            >= module.MIN_DESPLAZAMIENTO_RUMBO
        ):
            target_bearing = (
                math.degrees(
                    math.atan2(
                        target_x - x,
                        -(target_y - y),
                    )
                )
                % 360.0
            )

            error = abs(
                (
                    float(direction)
                    - target_bearing
                    + 180.0
                )
                % 360.0
                - 180.0
            )

            movement_score = max(
                0.0,
                1.0 - error / 90.0,
            )

        motion_present = (
            1.0
            if (
                speed >= 0.10
                and displacement
                >= module.MIN_DESPLAZAMIENTO_MOVIMIENTO
            )
            else 0.0
        )

        # La intensidad radar participa apenas como una señal secundaria.
        intensity = max(
            0.0,
            min(
                1.0,
                number(
                    track.get("intensidad")
                ) / 6.0,
            ),
        )

        structure = max(
            0.0,
            min(
                1.0,
                number(
                    track.get("pixeles")
                ) / 1500.0,
            ),
        )

        score = (
            persistence * 0.22
            + motion_conf * 0.16
            + movement_score * 0.25
            + motion_present * 0.10
            + growth * 0.10
            + proximity * 0.07
            + structure * 0.07
            + intensity * 0.03
        )

        scored.append(
            (
                score,
                track,
            )
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    best_score, best = scored[0]

    return {
        "track_id": best.get("track_id"),
        "score": round(
            best_score,
            4,
        ),
        "criterio": (
            "persistencia_movimiento_"
            "rumbo_evolucion_proximidad_"
            "estructura"
        ),
        "candidatos": len(scored),
    }


def patched_procesar_frame(
    module,
    estado,
    nucleos,
    radar_ts,
    anterior_ts,
):
    if not nucleos:
        return advance_empty_frame(
            module,
            estado,
            radar_ts,
            anterior_ts,
        )

    if anterior_ts is None:
        return module.procesar_frame_secuencia_original(
            estado,
            nucleos,
            radar_ts,
            anterior_ts,
        )

    actual_dt = parse_ts(radar_ts)
    anterior_dt = parse_ts(anterior_ts)

    if (
        actual_dt is None
        or anterior_dt is None
    ):
        return module.procesar_frame_secuencia_original(
            estado,
            nucleos,
            radar_ts,
            anterior_ts,
        )

    restantes, recuperados = recover_missed_tracks(
        module,
        estado,
        nucleos,
        radar_ts,
        anterior_ts,
    )

    if not restantes and recuperados:
        resultado = {
            "schema_version": module.SCHEMA_VERSION,
            "radar_timestamp": radar_ts,
            "frame_anterior_disponible": True,
            "intervalo_minutos": round(
                (
                    actual_dt - anterior_dt
                ).total_seconds()
                / 60.0,
                2,
            ),
            "calidad_intervalo": calidad(
                (
                    actual_dt - anterior_dt
                ).total_seconds()
                / 60.0
            ),
            "nucleos_actuales": len(
                nucleos
            ),
            "nucleos_nuevos": 0,
            "nucleos_seguimiento": len(
                recuperados
            ),
            "nucleos_creciendo": sum(
                1
                for item in recuperados
                if number(
                    item["track"].get(
                        "cambio_area_porcentaje"
                    )
                ) >= 15
            ),
            "nucleos_debilitandose": sum(
                1
                for item in recuperados
                if number(
                    item["track"].get(
                        "cambio_area_porcentaje"
                    )
                ) <= -15
            ),
            "nucleos_movimiento": sum(
                1
                for item in recuperados
                if number(
                    item["track"].get(
                        "desplazamiento_pixeles"
                    )
                ) >= 1
            ),
            "nucleos_estables": sum(
                1
                for item in recuperados
                if abs(
                    number(
                        item["track"].get(
                            "cambio_area_porcentaje"
                        )
                    )
                ) < 15
            ),
            "nucleos_desaparecidos": 0,
            "eventos_split": [],
            "eventos_merge": [],
            "cantidad_split": 0,
            "cantidad_merge": 0,
            "asociaciones": [],
            "nucleos": [],
        }

    else:
        resultado = (
            module.procesar_frame_secuencia_original(
                estado,
                restantes,
                radar_ts,
                anterior_ts,
            )
        )

    if recuperados:
        recuperados_salida = [
            module.nucleo_salida(
                item["track"]
            )
            for item in recuperados
        ]

        resultado["nucleos"] = (
            recuperados_salida
            + resultado.get(
                "nucleos",
                [],
            )
        )

        resultado["nucleos_seguimiento"] = (
            int(
                resultado.get(
                    "nucleos_seguimiento",
                    0,
                )
            )
            + len(recuperados)
        )

        resultado["nucleos_actuales"] = len(
            nucleos
        )

        resultado["nucleos_nuevos"] = max(
            0,
            int(
                resultado.get(
                    "nucleos_nuevos",
                    0,
                )
            ),
        )

        resultado["nucleos_recuperados"] = (
            len(recuperados)
        )

        resultado["recuperaciones"] = [
            {
                "track_id": item["track"].get(
                    "track_id"
                ),
                "intervalo_real_minutos": round(
                    item["intervalo"],
                    2,
                ),
                "score": round(
                    item["score"],
                    2,
                ),
            }
            for item in recuperados
        ]

        estado["ultimo_frame_nucleos"] = (
            resultado["nucleos"]
        )

    principal = select_principal_candidate(
        estado,
        radar_ts,
    )

    resultado[
        "track_principal_candidato"
    ] = principal

    return resultado


def main():
    module = load_original()

    module.procesar_frame_secuencia_original = (
        module.procesar_frame_secuencia
    )

    module.procesar_frame_secuencia = (
        lambda estado, nucleos, radar_ts, anterior_ts:
        patched_procesar_frame(
            module,
            estado,
            nucleos,
            radar_ts,
            anterior_ts,
        )
    )

    return module.procesar()


if __name__ == "__main__":
    result = main()

    print(
        "BLOQUE 2 - TRACKING ROBUSTO TERMINADO"
    )

    print(
        "Núcleos actuales:",
        result.get(
            "nucleos_actuales",
            0,
        ),
    )

    print(
        "Seguimientos:",
        result.get(
            "nucleos_seguimiento",
            0,
        ),
    )

    print(
        "Recuperados:",
        result.get(
            "nucleos_recuperados",
            0,
        ),
    )

    print(
        "Principal candidato:",
        result.get(
            "track_principal_candidato"
        ),
                         )
