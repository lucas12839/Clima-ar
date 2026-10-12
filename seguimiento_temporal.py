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
TIMESTAMP_FILE = RADAR_DIR / "radar_timestamp.txt"

STATE_SCHEMA_VERSION = 4

DISTANCIA_BASE = 25.0
DISTANCIA_MAXIMA = 90.0

CAMBIO_AREA_CRECE = 15.0
CAMBIO_AREA_DEBILITA = -15.0
MOVIMIENTO_MINIMO = 5.0

MAX_HISTORIAL = 12
MAX_MISSED_FRAMES = 2

MIN_INTERVAL_MIN = 2.0
MAX_INTERVAL_MIN = 30.0


def ahora_utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_dt(value):
    if not value or not isinstance(value, str):
        return None

    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.astimezone(timezone.utc)

    except Exception:
        return None


def iso(dt):
    if not dt:
        return None

    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


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

    if TIMESTAMP_FILE.exists():

        dt = parse_dt(
            TIMESTAMP_FILE.read_text(
                encoding="utf-8"
            ).strip()
        )

        if dt:
            return dt

    candidatos = []

    for p in HISTORICO_DIR.glob("*.png"):

        m = re.search(r"(\d{14})", p.stem)

        if not m:
            continue

        try:

            candidatos.append(
                datetime.strptime(
                    m.group(1),
                    "%Y%m%d%H%M%S"
                ).replace(tzinfo=timezone.utc)
            )

        except ValueError:
            pass

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


def normalizar_nucleo(n):

    if not isinstance(n, dict):
        return None

    try:

        return {
            "x": float(
                n.get(
                    "centro_x",
                    n.get("x", 0)
                )
            ),

            "y": float(
                n.get(
                    "centro_y",
                    n.get("y", 0)
                )
            ),

            "pixeles": max(
                1,
                int(
                    n.get(
                        "pixeles",
                        n.get("area", 0)
                    )
                )
            ),

            "ancho": max(
                1.0,
                float(
                    n.get(
                        "ancho",
                        n.get("width", 1)
                    )
                )
            ),

            "alto": max(
                1.0,
                float(
                    n.get(
                        "alto",
                        n.get("height", 1)
                    )
                )
            ),

            "intensidad": n.get(
                "intensidad_maxima",
                n.get("intensity", "DESCONOCIDA")
            ),

            "nivel_intensidad": n.get(
                "nivel_intensidad",
                "DESCONOCIDO"
            ),

            "estado": n.get(
                "estado",
                "DESCONOCIDO"
            ),

            "confianza": float(
                n.get("confianza", 0)
            )
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

    inter = (
        max(0.0, min(ar, br) - max(al, bl))
        *
        max(0.0, min(ab, bb) - max(at, bt))
    )

    union = (
        a["ancho"] * a["alto"]
        +
        b["ancho"] * b["alto"]
        -
        inter
    )

    if union <= 0:
        return 0.0

    return inter / union


def umbral_distancia(minutos):

    return min(
        DISTANCIA_MAXIMA,
        max(
            DISTANCIA_BASE,
            DISTANCIA_BASE + minutos * 5.0
        )
    )


def compatibilidad(a, b, minutos):

    dist = distancia(a, b)

    limite = umbral_distancia(minutos)

    if dist > limite:
        return None

    aa = max(1, a["pixeles"])
    ab = max(1, b["pixeles"])

    ratio_area = max(aa, ab) / min(aa, ab)

    ov = iou(a, b)

    rw = (
        max(a["ancho"], b["ancho"])
        /
        max(1, min(a["ancho"], b["ancho"]))
    )

    rh = (
        max(a["alto"], b["alto"])
        /
        max(1, min(a["alto"], b["alto"]))
    )

    # Sin solapamiento:
    # solamente aceptamos cambios muy pequeños.

    if ov <= 0.0:

        if (
            dist > 7.0
            or ratio_area > 1.30
            or rw > 1.50
            or rh > 1.50
        ):
            return None

    elif ov < 0.05:

        if (
            dist > 10.0
            or ratio_area > 1.40
            or rw > 1.80
            or rh > 1.80
        ):
            return None

    elif ov < 0.12:

        if dist > 18.0 and ratio_area > 1.80:
            return None

    if ratio_area > 2.5 and ov < 0.15:
        return None

    if rw > 2.8 and rh > 2.0:
        return None

    if rw > 3.5 or rh > 3.5:
        return None

    if (
        dist > max(35, limite * 0.65)
        and
        (ov < 0.08 or ratio_area > 2.0)
    ):
        return None

    close = max(
        0,
        1 - dist / limite
    )

    sim_area = max(
        0,
        1 -
        abs(math.log(aa / ab))
        /
        math.log(2.5)
    )

    sim_shape = max(
        0,
        1 -
        (
            abs(math.log(rw))
            +
            abs(math.log(rh))
        )
        /
        math.log(3.0)
    )

    score = (
        35 * close
        +
        35 * ov
        +
        20 * sim_area
        +
        10 * sim_shape
    )

    min_score = 60.0 if ov < 0.05 else 50.0

    if score < min_score:
        return None

    return {
        "puntaje": score,
        "distancia": dist,
        "iou": ov,
        "cambio_area": (
            (aa - ab) / ab * 100
        ),
        "ratio_area": ratio_area,
        "ratio_ancho": rw,
        "ratio_alto": rh
    }


def rumbo(dx, dy):

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None, None

    grados = (
        math.degrees(
            math.atan2(dx, -dy)
        )
        + 360
    ) % 360

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
        "NNO"
    ]

    indice = int(
        (grados + 11.25) / 22.5
    ) % 16

    return grados, sectores[indice]


def crear_track(track_id, nucleo, timestamp):

    return {
        "id": track_id,
        "primer_frame": timestamp,
        "ultimo_frame": timestamp,
        "frames": 1,
        "missed_frames": 0,
        "activo": True,
        "presente_en_ultimo_frame": True,

        "x": nucleo["x"],
        "y": nucleo["y"],
        "pixeles": nucleo["pixeles"],

        "intensidad": nucleo["intensidad"],
        "nivel_intensidad": nucleo["nivel_intensidad"],

        "estado": "NUEVO",

        "historial": [
            {
                "timestamp": timestamp,
                "x": nucleo["x"],
                "y": nucleo["y"],
                "pixeles": nucleo["pixeles"],
                "intensidad": nucleo["intensidad"],
                "nivel_intensidad": nucleo["nivel_intensidad"],
                "estado": "NUEVO"
            }
        ]
    }


def historial_limpio(track):

    raw = track.get(
        "historial",
        []
    )

    if not isinstance(raw, list):
        raw = []

    por_timestamp = {}

    for h in raw:

        if not isinstance(h, dict):
            continue

        dt = parse_dt(
            h.get("timestamp")
        )

        if not dt:
            continue

        h = dict(h)

        h["timestamp"] = iso(dt)

        por_timestamp[h["timestamp"]] = h

    historial = sorted(
        por_timestamp.values(),
        key=lambda h: parse_dt(h["timestamp"])
    )

    historial = historial[-MAX_HISTORIAL:]

    track["historial"] = historial

    if historial:

        track["primer_frame"] = historial[0]["timestamp"]
        track["ultimo_frame"] = historial[-1]["timestamp"]

    return historial


def actualizar_movimiento(
    track,
    anterior,
    actual,
    intervalo
):

    dx = actual["x"] - anterior["x"]
    dy = actual["y"] - anterior["y"]

    desplazamiento = math.hypot(
        dx,
        dy
    )

    velocidad = (
        desplazamiento / intervalo
        if intervalo > 0
        else 0
    )

    grados, direccion = rumbo(
        dx,
        dy
    )

    cambio_area = (
        (actual["pixeles"] - anterior["pixeles"])
        /
        max(1, anterior["pixeles"])
        *
        100
    )

    aceleracion = None
    cambio_rumbo = None
    tendencia = "ESTABLE"

    historial = track.get(
        "historial",
        []
    )

    movimientos_previos = []

    for i in range(
        max(1, len(historial) - 5),
        len(historial)
    ):

        h1 = historial[i - 1]
        h2 = historial[i]

        t1 = parse_dt(
            h1.get("timestamp")
        )

        t2 = parse_dt(
            h2.get("timestamp")
        )

        if not t1 or not t2:
            continue

        mins = (
            t2 - t1
        ).total_seconds() / 60

        if (
            mins < MIN_INTERVAL_MIN
            or mins > MAX_INTERVAL_MIN
        ):
            continue

        ddx = h2["x"] - h1["x"]
        ddy = h2["y"] - h1["y"]

        sp = math.hypot(
            ddx,
            ddy
        ) / mins

        br, _ = rumbo(
            ddx,
            ddy
        )

        movimientos_previos.append(
            {
                "velocidad": sp,
                "rumbo": br
            }
        )

    if movimientos_previos:

        anterior_mov = movimientos_previos[-1]

        aceleracion = (
            velocidad
            -
            anterior_mov["velocidad"]
        ) / intervalo

        if grados is not None and anterior_mov["rumbo"] is not None:

            cambio_rumbo = abs(
                (
                    grados
                    -
                    anterior_mov["rumbo"]
                    +
                    180
                )
                % 360
                - 180
            )

    if velocidad > 0:

        if aceleracion is not None:

            if aceleracion > 0.01:
                tendencia = "ACELERANDO"

            elif aceleracion < -0.01:
                tendencia = "DESACELERANDO"

            else:
                tendencia = "VELOCIDAD_ESTABLE"

    confianza = 50.0

    if desplazamiento >= MOVIMIENTO_MINIMO:
        confianza += 20

    if track.get("frames", 1) >= 3:
        confianza += 10

    if cambio_area is not None:
        confianza += 5

    if cambio_rumbo is not None:
        if cambio_rumbo <= 30:
            confianza += 10
        elif cambio_rumbo > 90:
            confianza -= 10

    confianza = max(
        0,
        min(99, confianza)
    )

    track.update({

        "dx_pixeles": dx,
        "dy_pixeles": dy,

        "desplazamiento_pixeles":
            desplazamiento,

        "velocidad_pixeles_min":
            velocidad,

        "rumbo_grados":
            grados,

        "direccion":
            direccion,

        "cambio_rumbo_grados":
            cambio_rumbo,

        "aceleracion_pixeles_min2":
            aceleracion,

        "tendencia_velocidad":
            tendencia,

        "confianza_movimiento":
            confianza,

        "cambio_area_porcentaje":
            cambio_area
    })


def siguiente_id(tracks):

    ids = []

    for t in tracks:

        try:
            ids.append(
                int(t.get("id"))
            )
        except Exception:
            pass

    return max(ids or [0]) + 1


def main():

    resultado = cargar_json(
        RESULTADO_FILE
    )

    if resultado is None:
        raise RuntimeError(
            "No existe resultado_procesamiento.json"
        )

    radar_dt = obtener_timestamp_radar()

    if radar_dt is None:
        raise RuntimeError(
            "No se pudo determinar el timestamp real del radar."
        )

    radar_timestamp = iso(radar_dt)

    nucleos_raw = obtener_nucleos(
        resultado
    )

    nucleos = []

    for n in nucleos_raw:

        normalizado = normalizar_nucleo(n)

        if normalizado:
            nucleos.append(normalizado)

    SEGUIMIENTO_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    estado = cargar_json(
        ESTADO_FILE
    )

    if estado:
        estado = dict(estado)

    # Primer frame
    if not estado:

        tracks = []

        for nucleo in nucleos:

            tid = len(tracks) + 1

            track = crear_track(
                tid,
                nucleo,
                radar_timestamp
            )

            tracks.append(track)

        for i, nucleo in enumerate(nucleos):

            nucleo["track_id"] = (
                tracks[i]["id"]
            )

        estado = {
            "schema_version":
                STATE_SCHEMA_VERSION,

            "radar_timestamp":
                radar_timestamp,

            "nucleos":
                nucleos,

            "tracks":
                tracks
        }

        guardar_json(
            ESTADO_FILE,
            estado
        )

        resultado_temporal = {

            "schema_version":
                STATE_SCHEMA_VERSION,

            "radar_timestamp":
                radar_timestamp,

            "intervalo_minutos":
                None,

            "calidad_intervalo":
                "PRIMER_FRAME",

            "nucleos_actuales":
                len(nucleos),

            "nucleos_nuevos":
                len(nucleos),

            "nucleos_seguimiento":
                0,

            "nucleos_creciendo":
                0,

            "nucleos_debilitandose":
                0,

            "nucleos_movimiento":
                0,

            "nucleos_estables":
                0,

            "nucleos_desaparecidos":
                0,

            "asociaciones":
                []
        }

        guardar_json(
            RESULTADO_SEGUIMIENTO,
            resultado_temporal
        )

        print(
            f"PRIMER FRAME: {radar_timestamp}"
        )

        print(
            f"Núcleos: {len(nucleos)}"
        )

        return

    estado = dict(estado)

    estado["schema_version"] = (
        STATE_SCHEMA_VERSION
    )

    ultimo_timestamp = parse_dt(
        estado.get(
            "radar_timestamp"
        )
    )

    if ultimo_timestamp is None:

        ultimo_timestamp = radar_dt

    intervalo = (
        radar_dt - ultimo_timestamp
    ).total_seconds() / 60

    print(
        f"Frame anterior: {iso(ultimo_timestamp)}"
    )

    print(
        f"Frame actual: {radar_timestamp}"
    )

    print(
        f"Intervalo: {intervalo:.1f} min"
    )

    # Frame repetido o atrasado.
    if intervalo <= 0:

        resultado_temporal = {

            "schema_version":
                STATE_SCHEMA_VERSION,

            "timestamp":
                ahora_utc(),

            "radar_timestamp":
                radar_timestamp,

            "intervalo_minutos":
                intervalo,

            "frame_anterior_disponible":
                True,

            "calidad_intervalo":
                "FRAME_REPETIDO_O_ATRASADO",

            "motivo":
                "El frame actual no es posterior al último estado procesado.",

            "nucleos_actuales":
                len(nucleos),

            "nucleos_nuevos":
                0,

            "nucleos_seguimiento":
                0,

            "nucleos_creciendo":
                0,

            "nucleos_debilitandose":
                0,

            "nucleos_movimiento":
                0,

            "nucleos_estables":
                0,

            "nucleos_desaparecidos":
                0,

            "asociaciones":
                []
        }

        guardar_json(
            RESULTADO_SEGUIMIENTO,
            resultado_temporal
        )

        print(
            "FRAME REPETIDO O ATRASADO."
        )

        return

    # Si pasaron más de 30 minutos,
    # no inventamos movimiento.
    if intervalo > MAX_INTERVAL_MIN:

        tracks = estado.get(
            "tracks",
            []
        )

        nuevo_tracks = []

        for i, nucleo in enumerate(nucleos):

            tid = siguiente_id(
                tracks + nuevo_tracks
            )

            nuevo_tracks.append(
                crear_track(
                    tid,
                    nucleo,
                    radar_timestamp
                )
            )

            nucleo["track_id"] = tid

        for track in tracks:

            track["activo"] = False
            track["presente_en_ultimo_frame"] = False

        estado["radar_timestamp"] = (
            radar_timestamp
        )

        estado["nucleos"] = nucleos

        estado["tracks"] = (
            tracks + nuevo_tracks
        )

        guardar_json(
            ESTADO_FILE,
            estado
        )

        resultado_temporal = {

            "schema_version":
                STATE_SCHEMA_VERSION,

            "timestamp":
                ahora_utc(),

            "radar_timestamp":
                radar_timestamp,

            "intervalo_minutos":
                intervalo,

            "calidad_intervalo":
                "GAP_MAYOR_30_MIN",

            "motivo":
                "Se inicia una nueva base sin inventar movimiento.",

            "nucleos_actuales":
                len(nucleos),

            "nucleos_nuevos":
                len(nucleos),

            "nucleos_seguimiento":
                0,

            "nucleos_creciendo":
                0,

            "nucleos_debilitandose":
                0,

            "nucleos_movimiento":
                0,

            "nucleos_estables":
                0,

            "nucleos_desaparecidos":
                0,

            "asociaciones":
                []
        }

        guardar_json(
            RESULTADO_SEGUIMIENTO,
            resultado_temporal
        )

        print(
            "GAP MAYOR A 30 MINUTOS."
        )

        return

    tracks = estado.get(
        "tracks",
        []
    )

    # Sólo tracks activos pueden recibir asociación.
    activos = [
        t for t in tracks
        if t.get("activo", False)
    ]

    candidatos = []

    for ti, track in enumerate(activos):

        historial = historial_limpio(
            track
        )

        if not historial:
            continue

        anterior = historial[-1]

        anterior_nucleo = {
            "x": anterior["x"],
            "y": anterior["y"],
            "pixeles": anterior["pixeles"],
            "ancho": max(
                1,
                track.get("ancho", 1)
            ),
            "alto": max(
                1,
                track.get("alto", 1)
            )
        }

        for ni, nucleo in enumerate(nucleos):

            comp = compatibilidad(
                anterior_nucleo,
                nucleo,
                intervalo
            )

            if comp is None:
                continue

            candidatos.append(
                (
                    comp["puntaje"],
                    ti,
                    ni,
                    comp
                )
            )

    candidatos.sort(
        key=lambda x: x[0],
        reverse=True
    )

    usados_tracks = set()
    usados_nucleos = set()
    asociaciones = []

    for score, ti, ni, comp in candidatos:

        if ti in usados_tracks:
            continue

        if ni in usados_nucleos:
            continue

        usados_tracks.add(ti)
        usados_nucleos.add(ni)

        asociaciones.append(
            (
                ti,
                ni,
                comp
            )
        )

    asociaciones_por_track = {}

    for ti, ni, comp in asociaciones:
        asociaciones_por_track[ti] = (
            ni,
            comp
        )

    actuales_asociados = set()

    crecieron = 0
    debilitaron = 0
    movimiento = 0
    estables = 0

    salida_asociaciones = []

    for ti, track in enumerate(activos):

        if ti not in asociaciones_por_track:

            track["missed_frames"] = (
                int(
                    track.get(
                        "missed_frames",
                        0
                    )
                )
                + 1
            )

            track["presente_en_ultimo_frame"] = False

            if (
                track["missed_frames"]
                > MAX_MISSED_FRAMES
            ):
                track["activo"] = False

            continue

        ni, comp = (
            asociaciones_por_track[ti]
        )

        nucleo = nucleos[ni]

        actuales_asociados.add(
            ni
        )

        historial = historial_limpio(
            track
        )

        anterior = historial[-1]

        anterior_nucleo = {
            "x": anterior["x"],
            "y": anterior["y"],
            "pixeles": anterior["pixeles"]
        }

        actualizar_movimiento(
            track,
            anterior_nucleo,
            nucleo,
            intervalo
        )

        track["x"] = nucleo["x"]
        track["y"] = nucleo["y"]
        track["pixeles"] = nucleo["pixeles"]

        track["intensidad"] = (
            nucleo["intensidad"]
        )

        track["nivel_intensidad"] = (
            nucleo["nivel_intensidad"]
        )

        cambio_area = track.get(
            "cambio_area_porcentaje",
            0
        )

        if cambio_area >= CAMBIO_AREA_CRECE:

            track["estado"] = "CRECIENDO"

            crecieron += 1

        elif (
            cambio_area
            <= CAMBIO_AREA_DEBILITA
        ):

            track["estado"] = (
                "DEBILITANDOSE"
            )

            debilitaron += 1

        elif (
            track.get(
                "desplazamiento_pixeles",
                0
            )
            >= MOVIMIENTO_MINIMO
        ):

            track["estado"] = "MOVIENDOSE"

            movimiento += 1

        else:

            track["estado"] = "ESTABLE"

            estables += 1

        track["frames"] = (
            int(
                track.get(
                    "frames",
                    1
                )
            )
            + 1
        )

        track["missed_frames"] = 0

        track["activo"] = True

        track["presente_en_ultimo_frame"] = True

        track["ultimo_frame"] = (
            radar_timestamp
        )

        historial.append(
            {
                "timestamp":
                    radar_timestamp,

                "x":
                    nucleo["x"],

                "y":
                    nucleo["y"],

                "pixeles":
                    nucleo["pixeles"],

                "intensidad":
                    nucleo["intensidad"],

                "nivel_intensidad":
                    nucleo["nivel_intensidad"],

                "estado":
                    track["estado"],

                "iou":
                    comp["iou"],

                "confianza_movimiento":
                    track.get(
                        "confianza_movimiento"
                    )
            }
        )

        track["historial"] = (
            historial[-MAX_HISTORIAL:]
        )

        nucleo["track_id"] = (
            track["id"]
        )

        salida_asociaciones.append(
            {
                "track_id":
                    track["id"],

                "nucleo_actual":
                    ni,

                "puntaje":
                    round(
                        comp["puntaje"],
                        2
                    ),

                "distancia_pixeles":
                    round(
                        comp["distancia"],
                        2
                    ),

                "iou":
                    round(
                        comp["iou"],
                        3
                    ),

                "cambio_area_porcentaje":
                    round(
                        comp["cambio_area"],
                        2
                    ),

                "dx_pixeles":
                    track.get(
                        "dx_pixeles"
                    ),

                "dy_pixeles":
                    track.get(
                        "dy_pixeles"
                    ),

                "desplazamiento_pixeles":
                    track.get(
                        "desplazamiento_pixeles"
                    ),

                "velocidad_pixeles_min":
                    track.get(
                        "velocidad_pixeles_min"
                    ),

                "rumbo_grados":
                    track.get(
                        "rumbo_grados"
                    ),

                "direccion":
                    track.get(
                        "direccion"
                    ),

                "cambio_rumbo_grados":
                    track.get(
                        "cambio_rumbo_grados"
                    ),

                "aceleracion_pixeles_min2":
                    track.get(
                        "aceleracion_pixeles_min2"
                    ),

                "tendencia_velocidad":
                    track.get(
                        "tendencia_velocidad"
                    ),

                "confianza_movimiento":
                    track.get(
                        "confianza_movimiento"
                    )
            }
        )

    # Los núcleos que no pudieron asociarse
    # se consideran nuevos.
    nuevos = 0

    for ni, nucleo in enumerate(nucleos):

        if ni in actuales_asociados:
            continue

        tid = siguiente_id(
            tracks
        )

        nuevo_track = crear_track(
            tid,
            nucleo,
            radar_timestamp
        )

        tracks.append(
            nuevo_track
        )

        nucleo["track_id"] = tid

        nuevos += 1

    desaparecidos = 0

    for track in activos:

        if not track.get(
            "presente_en_ultimo_frame",
            False
        ):

            if track.get(
                "missed_frames",
                0
            ) == 1:

                desaparecidos += 1

    estado["schema_version"] = (
        STATE_SCHEMA_VERSION
    )

    estado["radar_timestamp"] = (
        radar_timestamp
    )

    estado["nucleos"] = nucleos

    estado["tracks"] = tracks

    guardar_json(
        ESTADO_FILE,
        estado
    )

    resultado_temporal = {

        "schema_version":
            STATE_SCHEMA_VERSION,

        "timestamp":
            ahora_utc(),

        "radar_timestamp":
            radar_timestamp,

        "intervalo_minutos":
            intervalo,

        "frame_anterior_disponible":
            True,

        "calidad_intervalo":
            "VALIDO",

        "nucleos_actuales":
            len(nucleos),

        "nucleos_nuevos":
            nuevos,

        "nucleos_seguimiento":
            len(salida_asociaciones),

        "nucleos_creciendo":
            crecieron,

        "nucleos_debilitandose":
            debilitaron,

        "nucleos_movimiento":
            movimiento,

        "nucleos_estables":
            estables,

        "nucleos_desaparecidos":
            desaparecidos,

        "asociaciones":
            salida_asociaciones
    }

    guardar_json(
        RESULTADO_SEGUIMIENTO,
        resultado_temporal
    )

    print(
        "======================================"
    )

    print(
        "SEGUIMIENTO TEMPORAL"
    )

    print(
        "======================================"
    )

    print(
        f"Radar: {radar_timestamp}"
    )

    print(
        f"Intervalo: {intervalo:.1f} min"
    )

    print(
        f"Núcleos actuales: {len(nucleos)}"
    )

    print(
        f"Nuevos: {nuevos}"
    )

    print(
        f"Seguimiento: {len(salida_asociaciones)}"
    )

    print(
        f"Creciendo: {crecieron}"
    )

    print(
        f"Debilitándose: {debilitaron}"
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


if __name__ == "__main__":
    main()
