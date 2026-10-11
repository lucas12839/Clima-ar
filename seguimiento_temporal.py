#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
import json
import math
import re

RADAR_DIR = Path("data/radar")
RESULTADO_FILE = RADAR_DIR / "resultado_procesamiento.json"
SEGUIMIENTO_DIR = RADAR_DIR / "seguimiento"
ESTADO_FILE = SEGUIMIENTO_DIR / "estado.json"
RESULTADO_SEGUIMIENTO = SEGUIMIENTO_DIR / "seguimiento_temporal.json"
HISTORICO_DIR = RADAR_DIR / "historico"

# Seguimiento
DISTANCIA_BASE = 25.0
DISTANCIA_MAXIMA = 90.0
CAMBIO_AREA_CRECE = 15.0
CAMBIO_AREA_DEBILITA = -15.0
MOVIMIENTO_MINIMO = 5.0
MAX_HISTORIAL = 12
MAX_MISSED_FRAMES = 2


def ahora_utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def cargar_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"No se pudo leer {path}: {e}")
        return None


def guardar_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    tmp.replace(path)


def obtener_timestamp_radar():
    # Prioridad absoluta al nombre real del frame capturado.
    archivos = sorted(
        HISTORICO_DIR.glob("2026*.png"),
        key=lambda p: p.name
    )

    if archivos:
        m = re.search(r"(\d{14})", archivos[-1].stem)
        if m:
            try:
                return datetime.strptime(
                    m.group(1),
                    "%Y%m%d%H%M%S"
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                pass

    # Fallback: cualquier archivo que contenga timestamp YYYYMMDDHHMMSS.
    candidatos = []

    for p in HISTORICO_DIR.glob("*.png"):
        m = re.search(r"(\d{14})", p.stem)

        if m:
            try:
                dt = datetime.strptime(
                    m.group(1),
                    "%Y%m%d%H%M%S"
                ).replace(tzinfo=timezone.utc)

                candidatos.append(dt)

            except ValueError:
                continue

    return max(candidatos) if candidatos else None


def obtener_nucleos(resultado):
    if not isinstance(resultado, dict):
        return []

    for key in (
        "nucleos",
        "núcleos",
        "cells",
        "ecos",
        "detected_cells"
    ):
        valor = resultado.get(key)

        if isinstance(valor, list):
            return valor

    return []


def normalizar_nucleo(nucleo):
    if not isinstance(nucleo, dict):
        return None

    try:
        x = float(
            nucleo.get(
                "centro_x",
                nucleo.get("x", 0)
            )
        )

        y = float(
            nucleo.get(
                "centro_y",
                nucleo.get("y", 0)
            )
        )

        pixeles = int(
            nucleo.get(
                "pixeles",
                nucleo.get("area", 0)
            )
        )

        ancho = float(
            nucleo.get(
                "ancho",
                nucleo.get("width", 1)
            )
        )

        alto = float(
            nucleo.get(
                "alto",
                nucleo.get("height", 1)
            )
        )

        return {
            "x": x,
            "y": y,
            "pixeles": max(1, pixeles),
            "ancho": max(1.0, ancho),
            "alto": max(1.0, alto),
            "intensidad": nucleo.get(
                "intensidad_maxima",
                nucleo.get("intensity", "DESCONOCIDA")
            ),
            "nivel_intensidad": nucleo.get(
                "nivel_intensidad",
                "DESCONOCIDO"
            ),
            "estado": nucleo.get(
                "estado",
                "DESCONOCIDO"
            ),
            "confianza": float(
                nucleo.get("confianza", 0)
            ),
        }

    except Exception:
        return None


def distancia(a, b):
    return math.hypot(
        a["x"] - b["x"],
        a["y"] - b["y"]
    )


def iou(a, b):
    al = a["x"] - a["ancho"] / 2
    ar = a["x"] + a["ancho"] / 2
    at = a["y"] - a["alto"] / 2
    ab = a["y"] + a["alto"] / 2

    bl = b["x"] - b["ancho"] / 2
    br = b["x"] + b["ancho"] / 2
    bt = b["y"] - b["alto"] / 2
    bb = b["y"] + b["alto"] / 2

    w = max(
        0.0,
        min(ar, br) - max(al, bl)
    )

    h = max(
        0.0,
        min(ab, bb) - max(at, bt)
    )

    inter = w * h

    union = (
        a["ancho"] * a["alto"]
        + b["ancho"] * b["alto"]
        - inter
    )

    return inter / union if union > 0 else 0.0


def umbral_distancia(minutos):
    return min(
        DISTANCIA_MAXIMA,
        max(
            DISTANCIA_BASE,
            DISTANCIA_BASE + minutos * 5.0
        )
    )


def compatibilidad(actual, anterior, minutos):
    dist = distancia(actual, anterior)
    limite = umbral_distancia(minutos)

    if dist > limite:
        return None

    area_a = max(1.0, actual["pixeles"])
    area_b = max(1.0, anterior["pixeles"])

    ratio_area = (
        max(area_a, area_b) /
        min(area_a, area_b)
    )

    solapamiento = iou(actual, anterior)

    # Comparación de forma del eco.
    ratio_ancho = (
        max(actual["ancho"], anterior["ancho"]) /
        max(
            1.0,
            min(actual["ancho"], anterior["ancho"])
        )
    )

    ratio_alto = (
        max(actual["alto"], anterior["alto"]) /
        max(
            1.0,
            min(actual["alto"], anterior["alto"])
        )
    )

    # Si cambia demasiado de tamaño y no conserva solapamiento,
    # probablemente es otro eco.
    if ratio_area > 2.5 and solapamiento < 0.15:
        return None

    # Evita asociaciones entre ecos con formas completamente distintas.
    if ratio_ancho > 2.8 and ratio_alto > 2.0:
        return None

    if ratio_ancho > 3.5 or ratio_alto > 3.5:
        return None

    # Para desplazamientos grandes exigimos más evidencia.
    if dist > max(35.0, limite * 0.65):
        if solapamiento < 0.08:
            return None

        if ratio_area > 2.0:
            return None

    cercania = max(
        0.0,
        1.0 - dist / limite
    )

    similitud_area = max(
        0.0,
        1.0 -
        abs(math.log(area_a / area_b)) /
        math.log(2.5)
    )

    similitud_forma = max(
        0.0,
        1.0 -
        (
            abs(math.log(ratio_ancho)) +
            abs(math.log(ratio_alto))
        ) /
        math.log(3.0)
    )

    # Posición + solapamiento + tamaño + forma.
    puntaje = (
        35.0 * cercania +
        35.0 * solapamiento +
        20.0 * similitud_area +
        10.0 * similitud_forma
    )

    # Sin solapamiento: exigimos evidencia fuerte de que sigue siendo
    # el mismo eco. Esto evita asociaciones falsas entre núcleos
    # cercanos pero físicamente distintos.
    if solapamiento < 0.02:
        if dist > 20.0:
            return None

        if ratio_area > 1.5:
            return None

        if ratio_ancho > 1.6 or ratio_alto > 1.6:
            return None

        if puntaje < 55.0:
            return None

    cambio_area = (
        (area_a - area_b) /
        area_b *
        100.0
    )

    return {
        "puntaje": puntaje,
        "distancia": dist,
        "iou": solapamiento,
        "cambio_area": cambio_area,
        "ratio_area": ratio_area,
        "ratio_ancho": ratio_ancho,
        "ratio_alto": ratio_alto,
        "similitud_forma": similitud_forma,
    }


def siguiente_id(tracks):
    ids = []

    for t in tracks:
        try:
            ids.append(int(t.get("id", 0)))
        except Exception:
            pass

    return max(ids, default=0) + 1


def timestamp_iso(dt):
    return (
        dt.isoformat().replace("+00:00", "Z")
        if dt
        else None
    )


def crear_track(track_id, actual, timestamp):
    return {
        "id": track_id,
        "primer_frame": timestamp,
        "ultimo_frame": timestamp,
        "frames": 1,
        "missed_frames": 0,
        "x": actual["x"],
        "y": actual["y"],
        "pixeles": actual["pixeles"],
        "intensidad": actual["intensidad"],
        "nivel_intensidad": actual["nivel_intensidad"],
        "estado": "NUEVO",
        "historial": [
            {
                "timestamp": timestamp,
                "x": actual["x"],
                "y": actual["y"],
                "pixeles": actual["pixeles"],
                "intensidad": actual["intensidad"],
                "estado": "NUEVO",
            }
        ],
    }


def actualizar_track(
    track,
    actual,
    anterior,
    match,
    minutos,
    timestamp
):
    dx = actual["x"] - anterior["x"]
    dy = actual["y"] - anterior["y"]

    desplazamiento = math.hypot(dx, dy)

    velocidad = (
        desplazamiento /
        max(minutos, 0.1)
    )

    cambio_area = match["cambio_area"]

    if cambio_area >= CAMBIO_AREA_CRECE:
        evolucion = "CRECIENDO"

    elif cambio_area <= CAMBIO_AREA_DEBILITA:
        evolucion = "DEBILITANDOSE"

    else:
        evolucion = "ESTABLE"

    direccion = (
        math.degrees(
            math.atan2(dy, dx)
        ) + 360.0
    ) % 360.0

    track["ultimo_frame"] = timestamp
    track["frames"] = (
        int(track.get("frames", 0)) + 1
    )
    track["missed_frames"] = 0

    track["x"] = actual["x"]
    track["y"] = actual["y"]
    track["pixeles"] = actual["pixeles"]
    track["intensidad"] = actual["intensidad"]
    track["nivel_intensidad"] = (
        actual["nivel_intensidad"]
    )

    track["estado"] = evolucion

    track["desplazamiento_pixeles"] = round(
        desplazamiento,
        2
    )

    track["velocidad_pixeles_min"] = round(
        velocidad,
        3
    )

    track["direccion_grados"] = round(
        direccion,
        1
    )

    track["cambio_area_porcentaje"] = round(
        cambio_area,
        1
    )

    track["iou"] = round(
        match["iou"],
        3
    )

    track["ratio_area"] = round(
        match["ratio_area"],
        3
    )

    track["ratio_ancho"] = round(
        match["ratio_ancho"],
        3
    )

    track["ratio_alto"] = round(
        match["ratio_alto"],
        3
    )

    historial = track.setdefault(
        "historial",
        []
    )

    historial.append(
        {
            "timestamp": timestamp,
            "x": actual["x"],
            "y": actual["y"],
            "pixeles": actual["pixeles"],
            "intensidad": actual["intensidad"],
            "estado": evolucion,
            "desplazamiento_pixeles": round(
                desplazamiento,
                2
            ),
            "velocidad_pixeles_min": round(
                velocidad,
                3
            ),
            "direccion_grados": round(
                direccion,
                1
            ),
            "cambio_area_porcentaje": round(
                cambio_area,
                1
            ),
        }
    )

    track["historial"] = historial[
        -MAX_HISTORIAL:
    ]

    return evolucion, desplazamiento


def analizar():
    print("======================================")
    print("ClimaAR - SEGUIMIENTO TEMPORAL v3")
    print("======================================")

    resultado = cargar_json(
        RESULTADO_FILE
    )

    if resultado is None:
        raise RuntimeError(
            "No existe resultado_procesamiento.json"
        )

    actuales = []

    for raw in obtener_nucleos(resultado):
        n = normalizar_nucleo(raw)

        if n:
            actuales.append(n)

    radar_dt = obtener_timestamp_radar()
    radar_ts = timestamp_iso(radar_dt)

    print(
        f"Núcleos actuales: {len(actuales)}"
    )

    print(
        f"Radar timestamp: "
        f"{radar_ts or 'NO DISPONIBLE'}"
    )

    estado_anterior = cargar_json(
        ESTADO_FILE
    )

    if not estado_anterior:
        tracks = []
        estado_nucleos = []
        ts = radar_ts or ahora_utc()
        next_id = 1

        for actual in actuales:
            track = crear_track(
                next_id,
                actual,
                ts
            )

            tracks.append(track)

            estado_nucleos.append(
                {
                    **actual,
                    "track_id": next_id
                }
            )

            next_id += 1

        guardar_json(
            ESTADO_FILE,
            {
                "timestamp": ahora_utc(),
                "radar_timestamp": radar_ts,
                "nucleos": estado_nucleos,
                "tracks": tracks,
            }
        )

        guardar_json(
            RESULTADO_SEGUIMIENTO,
            {
                "timestamp": ahora_utc(),
                "radar_timestamp": radar_ts,
                "intervalo_minutos": None,
                "frame_anterior_disponible": False,
                "nucleos_actuales": len(actuales),
                "nucleos_nuevos": len(actuales),
                "nucleos_seguimiento": 0,
                "nucleos_creciendo": 0,
                "nucleos_debilitandose": 0,
                "nucleos_movimiento": 0,
                "nucleos_estables": 0,
                "nucleos_desaparecidos": 0,
                "asociaciones": [],
                "tracks": tracks,
            }
        )

        print(
            "Primer frame guardado. "
            "Ejecutar nuevamente para comparar."
        )

        return

    anteriores = []

    for raw in estado_anterior.get(
        "nucleos",
        []
    ):
        n = normalizar_nucleo(raw)

        if n:
            n["track_id"] = raw.get(
                "track_id"
            )

            anteriores.append(n)

    tracks_anteriores = estado_anterior.get(
        "tracks",
        []
    )

    tracks = [
        dict(t)
        for t in tracks_anteriores
        if isinstance(t, dict)
    ]

    prev_dt = None

    prev_text = estado_anterior.get(
        "radar_timestamp"
    )

    if prev_text:
        try:
            prev_dt = datetime.fromisoformat(
                prev_text.replace(
                    "Z",
                    "+00:00"
                )
            )
        except Exception:
            pass

    if radar_dt and prev_dt:
        minutos = max(
            0.1,
            (
                radar_dt - prev_dt
            ).total_seconds() / 60.0
        )

        if minutos > 180:
            print(
                "Intervalo del radar demasiado "
                "grande; se usa 10 min como respaldo."
            )

            minutos = 10.0

    else:
        minutos = 10.0

    print(
        f"Intervalo real estimado: "
        f"{minutos:.1f} minutos"
    )

    # Matriz de candidatos.
    # Se asigna globalmente por mejor puntaje,
    # evitando que dos ecos actuales reclamen
    # el mismo eco anterior.
    candidatos = []

    for ia, actual in enumerate(
        actuales
    ):
        for ip, anterior in enumerate(
            anteriores
        ):
            match = compatibilidad(
                actual,
                anterior,
                minutos
            )

            if match is not None:
                candidatos.append(
                    (
                        match["puntaje"],
                        ia,
                        ip,
                        match
                    )
                )

    candidatos.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    asignaciones_actual = {}
    usados_prev = set()

    for _, ia, ip, match in candidatos:
        if ia in asignaciones_actual:
            continue

        if ip in usados_prev:
            continue

        asignaciones_actual[ia] = (
            ip,
            match
        )

        usados_prev.add(ip)

    track_by_id = {}

    for t in tracks:
        try:
            track_by_id[int(t["id"])] = t
        except Exception:
            continue

    next_id = siguiente_id(
        tracks
    )

    estado_nucleos = []
    asociaciones = []

    nuevos = 0
    creciendo = 0
    debilitando = 0
    movimiento = 0
    estables = 0

    for ia, actual in enumerate(
        actuales
    ):
        if ia not in asignaciones_actual:
            track = crear_track(
                next_id,
                actual,
                radar_ts or ahora_utc()
            )

            tracks.append(track)
            track_by_id[next_id] = track

            estado_nucleos.append(
                {
                    **actual,
                    "track_id": next_id
                }
            )

            asociaciones.append(
                {
                    "track_id": next_id,
                    "tipo": "NUEVO",
                    "actual": {
                        "x": actual["x"],
                        "y": actual["y"],
                        "pixeles": actual["pixeles"]
                    },
                }
            )

            nuevos += 1
            next_id += 1

            continue

        ip, match = asignaciones_actual[
            ia
        ]

        anterior = anteriores[ip]

        track_id = anterior.get(
            "track_id"
        )

        if track_id is None:
            candidatos_track = [
                t
                for t in tracks
                if distancia(
                    actual,
                    {
                        "x": t.get(
                            "x",
                            0
                        ),
                        "y": t.get(
                            "y",
                            0
                        )
                    }
                )
                <= umbral_distancia(
                    minutos
                )
            ]

            if candidatos_track:
                track_id = min(
                    candidatos_track,
                    key=lambda t:
                        distancia(
                            actual,
                            {
                                "x": t.get(
                                    "x",
                                    0
                                ),
                                "y": t.get(
                                    "y",
                                    0
                                )
                            }
                        )
                ).get("id")

        try:
            track_id = int(track_id)

        except Exception:
            track_id = next_id
            next_id += 1

            track = crear_track(
                track_id,
                actual,
                radar_ts or ahora_utc()
            )

            tracks.append(track)
            track_by_id[track_id] = track

        track = track_by_id.get(
            track_id
        )

        if track is None:
            track = crear_track(
                track_id,
                actual,
                radar_ts or ahora_utc()
            )

            tracks.append(track)
            track_by_id[track_id] = track

        evolucion, desplazamiento = (
            actualizar_track(
                track,
                actual,
                anterior,
                match,
                minutos,
                radar_ts or ahora_utc()
            )
        )

        if evolucion == "CRECIENDO":
            creciendo += 1

        elif evolucion == "DEBILITANDOSE":
            debilitando += 1

        else:
            estables += 1

        if desplazamiento >= MOVIMIENTO_MINIMO:
            movimiento += 1

        estado_nucleos.append(
            {
                **actual,
                "track_id": track_id
            }
        )

        asociaciones.append(
            {
                "track_id": track_id,
                "tipo": "SEGUIMIENTO",
                "distancia_pixeles": round(
                    match["distancia"],
                    2
                ),
                "iou": round(
                    match["iou"],
                    3
                ),
                "ratio_area": round(
                    match["ratio_area"],
                    3
                ),
                "ratio_ancho": round(
                    match["ratio_ancho"],
                    3
                ),
                "ratio_alto": round(
                    match["ratio_alto"],
                    3
                ),
                "cambio_area_porcentaje": round(
                    match["cambio_area"],
                    1
                ),
                "puntaje": round(
                    match["puntaje"],
                    2
                ),
            }
        )

    # Los anteriores no asignados se consideran
    # desaparecidos en este frame.
    desaparecidos = 0

    activos_ids = {
        n.get("track_id")
        for n in estado_nucleos
    }

    for ip, anterior in enumerate(
        anteriores
    ):
        if ip in usados_prev:
            continue

        tid = anterior.get(
            "track_id"
        )

        if tid is None:
            continue

        try:
            tid = int(tid)
        except Exception:
            continue

        track = track_by_id.get(
            tid
        )

        if track:
            track["missed_frames"] = (
                int(
                    track.get(
                        "missed_frames",
                        0
                    )
                ) + 1
            )

            track["estado"] = (
                "DESAPARECIDO"
            )

            if (
                track["missed_frames"]
                > MAX_MISSED_FRAMES
            ):
                track["activo"] = False

        desaparecidos += 1

    for t in tracks:
        t.setdefault(
            "activo",
            True
        )

        if t.get("id") in activos_ids:
            t["activo"] = True

    tracks = [
        t
        for t in tracks
        if t.get("activo", True)
    ]

    seguimiento = {
        "timestamp": ahora_utc(),
        "radar_timestamp": radar_ts,
        "intervalo_minutos": round(
            minutos,
            2
        ),
        "frame_anterior_disponible": True,
        "nucleos_actuales": len(
            actuales
        ),
        "nucleos_nuevos": nuevos,
        "nucleos_seguimiento": len(
            asignaciones_actual
        ),
        "nucleos_creciendo": creciendo,
        "nucleos_debilitandose": debilitando,
        "nucleos_movimiento": movimiento,
        "nucleos_estables": estables,
        "nucleos_desaparecidos": desaparecidos,
        "asociaciones": asociaciones,
        "tracks": tracks,
    }

    guardar_json(
        RESULTADO_SEGUIMIENTO,
        seguimiento
    )

    guardar_json(
        ESTADO_FILE,
        {
            "timestamp": ahora_utc(),
            "radar_timestamp": radar_ts,
            "nucleos": estado_nucleos,
            "tracks": tracks,
        }
    )

    print("======================================")
    print("SEGUIMIENTO TEMPORAL v3")
    print("======================================")

    print(
        f"Actuales: {len(actuales)}"
    )

    print(
        f"Seguimiento: "
        f"{len(asignaciones_actual)}"
    )

    print(
        f"Nuevos: {nuevos}"
    )

    print(
        f"Creciendo: {creciendo}"
    )

    print(
        f"Debilitándose: {debilitando}"
    )

    print(
        f"Moviéndose: {movimiento}"
    )

    print(
        f"Estables: {estables}"
    )

    print(
        f"Desaparecidos: {desaparecidos}"
    )

    print("======================================")


if __name__ == "__main__":
    analizar()
