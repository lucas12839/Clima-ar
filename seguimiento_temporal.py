import json
import math
from pathlib import Path
from datetime import datetime, timezone

BASE = Path("data/radar")
RESULTADO = BASE / "resultado_procesamiento.json"
SEGUIMIENTO = BASE / "seguimiento"
ESTADO = SEGUIMIENTO / "estado.json"
SALIDA = SEGUIMIENTO / "seguimiento_temporal.json"

SCHEMA_VERSION = 6

MIN_INTERVALO = 2.0
MAX_INTERVALO = 30.0
MAX_DISTANCIA = 45.0
MAX_DISTANCIA_SIN_OVERLAP = 25.0
MAX_MISSED_FRAMES = 2
MAX_HISTORIAL = 20


def cargar_json(path, default=None):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def guardar_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def numero(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def entero(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def timestamp_radar():
    archivos = sorted(BASE.glob("historico/*.png"), key=lambda p: p.name)
    candidatos = [
        p.stem for p in archivos
        if len(p.stem) == 14 and p.stem.isdigit()
    ]
    if not candidatos:
        raise RuntimeError("No se encontró timestamp de radar.")
    dt = datetime.strptime(candidatos[-1], "%Y%m%d%H%M%S")
    return dt.replace(tzinfo=timezone.utc)


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        ).astimezone(timezone.utc)
    except Exception:
        return None


def normalizar_nucleo(n, indice=0):
    if not isinstance(n, dict):
        n = {}

    x = numero(n.get("x", n.get("centro_x", n.get("centroid_x", 0))))
    y = numero(n.get("y", n.get("centro_y", n.get("centroid_y", 0))))
    pixeles = entero(n.get("pixeles", n.get("pixels", n.get("area", 0))))
    ancho = numero(n.get("ancho", n.get("width", 1)), 1)
    alto = numero(n.get("alto", n.get("height", 1)), 1)

    intensidad = entero(
        n.get(
            "intensidad_maxima",
            n.get("max_intensity", n.get("intensidad", 0))
        )
    )

    nivel = n.get("nivel_intensidad", n.get("level", ""))
    confianza = numero(n.get("confianza", n.get("confidence", 0)))
    estado = n.get("estado", n.get("state", ""))

    return {
        "indice": indice,
        "x": x,
        "y": y,
        "pixeles": pixeles,
        "ancho": max(ancho, 1),
        "alto": max(alto, 1),
        "intensidad": intensidad,
        "nivel_intensidad": nivel,
        "confianza": confianza,
        "estado": estado,
    }


def obtener_nucleos():
    data = cargar_json(RESULTADO, {})
    if not isinstance(data, dict):
        return []

    candidatos = (
        data.get("nucleos")
        or data.get("núcleos")
        or data.get("cells")
        or data.get("ecos")
        or data.get("detected_cells")
        or []
    )

    if not isinstance(candidatos, list):
        return []

    return [
        normalizar_nucleo(n, i)
        for i, n in enumerate(candidatos)
    ]


def bbox(n):
    x = n["x"]
    y = n["y"]
    w = max(n["ancho"], 1)
    h = max(n["alto"], 1)
    return (
        x - w / 2,
        y - h / 2,
        x + w / 2,
        y + h / 2,
    )


def iou(a, b):
    ax1, ay1, ax2, ay2 = bbox(a)
    bx1, by1, bx2, by2 = bbox(b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def distancia(a, b):
    return math.hypot(b["x"] - a["x"], b["y"] - a["y"])


def similitud_forma(a, b):
    wa = max(a["ancho"], 1)
    wb = max(b["ancho"], 1)
    ha = max(a["alto"], 1)
    hb = max(b["alto"], 1)

    return (
        min(wa, wb) / max(wa, wb)
        + min(ha, hb) / max(ha, hb)
    ) / 2


def compatibilidad(anterior, actual):
    d = distancia(anterior, actual)
    overlap = iou(anterior, actual)

    area_a = max(anterior["pixeles"], 1)
    area_b = max(actual["pixeles"], 1)
    area_min = min(area_a / area_b, area_b / area_a)

    forma = similitud_forma(anterior, actual)

    if d > MAX_DISTANCIA:
        return None

    if overlap < 0.02:
        if d > MAX_DISTANCIA_SIN_OVERLAP:
            return None
        if area_min < 0.65:
            return None
        if forma < 0.55:
            return None
    else:
        if area_min < 0.40:
            return None
        if forma < 0.35:
            return None

    proximidad = max(0.0, 1.0 - d / MAX_DISTANCIA)
    score = (
        proximidad * 40
        + overlap * 35
        + area_min * 15
        + forma * 10
    )

    return {
        "score": round(score, 2),
        "distancia": round(d, 3),
        "iou": round(overlap, 4),
        "similitud_area": round(area_min, 4),
        "similitud_forma": round(forma, 4),
    }


def rumbo(dx, dy):
    if abs(dx) < 0.01 and abs(dy) < 0.01:
        return None, None

    grados = math.degrees(math.atan2(dx, -dy)) % 360

    sectores = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSO", "SO", "OSO",
        "O", "ONO", "NO", "NNO",
    ]

    indice = int(((grados + 11.25) % 360) / 22.5)
    return grados, sectores[indice]


def estado_nuevo():
    return {
        "schema_version": SCHEMA_VERSION,
        "ultimo_radar_timestamp": None,
        "siguiente_track_id": 1,
        "tracks": [],
        "tracks_activos": [],
        "ultimo_frame_nucleos": [],
    }


def normalizar_estado(estado):
    if not isinstance(estado, dict):
        return estado_nuevo()

    nuevo = estado_nuevo()
    nuevo["ultimo_radar_timestamp"] = estado.get("ultimo_radar_timestamp")

    tracks = estado.get("tracks", [])
    if not isinstance(tracks, list):
        tracks = []

    max_id = 0

    for track in tracks:
        if not isinstance(track, dict):
            continue

        tid = track.get("track_id")
        try:
            tid = int(tid)
        except Exception:
            continue

        if tid <= 0:
            continue

        track["track_id"] = tid
        track["historial"] = (
            track.get("historial", [])
            if isinstance(track.get("historial", []), list)
            else []
        )

        nuevo["tracks"].append(track)
        max_id = max(max_id, tid)

    siguiente = entero(estado.get("siguiente_track_id"), max_id + 1)
    nuevo["siguiente_track_id"] = max(siguiente, max_id + 1)

    return nuevo


def crear_track(track_id, nucleo, radar_ts):
    return {
        "track_id": track_id,
        "activo": True,
        "primer_frame": radar_ts,
        "ultimo_frame": radar_ts,
        "presente_en_ultimo_frame": True,
        "missed_frames": 0,
        "x": nucleo["x"],
        "y": nucleo["y"],
        "pixeles": nucleo["pixeles"],
        "ancho": nucleo["ancho"],
        "alto": nucleo["alto"],
        "intensidad": nucleo["intensidad"],
        "nivel_intensidad": nucleo["nivel_intensidad"],
        "estado": nucleo["estado"],
        "confianza": nucleo["confianza"],
        "dx_pixeles": 0.0,
        "dy_pixeles": 0.0,
        "desplazamiento_pixeles": 0.0,
        "velocidad_pixeles_min": 0.0,
        "rumbo_grados": None,
        "direccion": None,
        "aceleracion_pixeles_min2": 0.0,
        "cambio_rumbo_grados": 0.0,
        "calidad_movimiento": "PRIMER_FRAME",
        "cambio_area_porcentaje": 0.0,
        "tendencia_velocidad": "ESTABLE",
        "confianza_movimiento": 0.0,
        "historial": [{
            "timestamp": radar_ts,
            "x": nucleo["x"],
            "y": nucleo["y"],
            "pixeles": nucleo["pixeles"],
            "ancho": nucleo["ancho"],
            "alto": nucleo["alto"],
            "intensidad": nucleo["intensidad"],
        }],
    }


def actualizar_track(track, nucleo, radar_ts, intervalo, calidad):
    anterior_x = numero(track.get("x"))
    anterior_y = numero(track.get("y"))
    anterior_velocidad = numero(
        track.get("velocidad_pixeles_min", 0)
    )
    anterior_rumbo = track.get("rumbo_grados")

    dx = nucleo["x"] - anterior_x
    dy = nucleo["y"] - anterior_y
    desplazamiento = math.hypot(dx, dy)

    if MIN_INTERVALO <= intervalo <= MAX_INTERVALO:
        velocidad = desplazamiento / intervalo
        aceleracion = (velocidad - anterior_velocidad) / intervalo
    else:
        velocidad = 0.0
        aceleracion = 0.0

    grados, direccion = rumbo(dx, dy)

    cambio_rumbo = 0.0
    if anterior_rumbo is not None and grados is not None:
        cambio_rumbo = abs(
            (grados - numero(anterior_rumbo) + 180) % 360 - 180
        )

    anterior_area = max(numero(track.get("pixeles"), 1), 1)
    cambio_area = (
        (nucleo["pixeles"] - anterior_area)
        / anterior_area
        * 100
    )

    if cambio_area >= 15:
        tendencia_area = "CRECIENDO"
    elif cambio_area <= -15:
        tendencia_area = "DEBILITANDOSE"
    else:
        tendencia_area = "ESTABLE"

    if velocidad > anterior_velocidad + 0.25:
        tendencia_velocidad = "AUMENTANDO"
    elif velocidad < anterior_velocidad - 0.25:
        tendencia_velocidad = "DISMINUYENDO"
    else:
        tendencia_velocidad = "ESTABLE"

    if desplazamiento < 1:
        confianza_movimiento = 0.35
    elif intervalo < MIN_INTERVALO or intervalo > MAX_INTERVALO:
        confianza_movimiento = 0.25
    else:
        confianza_movimiento = min(
            1.0,
            max(
                0.0,
                0.45
                + (1.0 - min(cambio_rumbo, 180) / 180) * 0.30
            ),
        )

    historial = track.get("historial")
    if not isinstance(historial, list):
        historial = []

    historial.append({
        "timestamp": radar_ts,
        "x": nucleo["x"],
        "y": nucleo["y"],
        "pixeles": nucleo["pixeles"],
        "ancho": nucleo["ancho"],
        "alto": nucleo["alto"],
        "intensidad": nucleo["intensidad"],
    })

    if len(historial) > MAX_HISTORIAL:
        historial = historial[-MAX_HISTORIAL:]

    track.update({
        "activo": True,
        "ultimo_frame": radar_ts,
        "presente_en_ultimo_frame": True,
        "missed_frames": 0,
        "x": nucleo["x"],
        "y": nucleo["y"],
        "pixeles": nucleo["pixeles"],
        "ancho": nucleo["ancho"],
        "alto": nucleo["alto"],
        "intensidad": nucleo["intensidad"],
        "nivel_intensidad": nucleo["nivel_intensidad"],
        "estado": nucleo["estado"],
        "confianza": nucleo["confianza"],
        "dx_pixeles": dx,
        "dy_pixeles": dy,
        "desplazamiento_pixeles": desplazamiento,
        "velocidad_pixeles_min": velocidad,
        "rumbo_grados": grados,
        "direccion": direccion,
        "aceleracion_pixeles_min2": aceleracion,
        "cambio_rumbo_grados": cambio_rumbo,
        "calidad_movimiento": calidad,
        "cambio_area_porcentaje": cambio_area,
        "tendencia_area": tendencia_area,
        "tendencia_velocidad": tendencia_velocidad,
        "confianza_movimiento": confianza_movimiento,
        "historial": historial,
    })


def nueva_base_desde_frame(estado, nucleos, radar_ts):
    for track in estado["tracks"]:
        track["activo"] = False
        track["presente_en_ultimo_frame"] = False

    actuales = []

    for nucleo in nucleos:
        tid = estado["siguiente_track_id"]
        estado["siguiente_track_id"] += 1

        track = crear_track(tid, nucleo, radar_ts)
        estado["tracks"].append(track)
        actuales.append(track)

    estado["tracks_activos"] = [
        t["track_id"] for t in actuales
    ]
    estado["ultimo_frame_nucleos"] = [
        nucleo_salida(t) for t in actuales
    ]
    estado["ultimo_radar_timestamp"] = radar_ts

    return actuales


def nucleo_salida(track):
    return {
        "track_id": track["track_id"],
        "x": track.get("x", 0),
        "y": track.get("y", 0),
        "pixeles": track.get("pixeles", 0),
        "ancho": track.get("ancho", 1),
        "alto": track.get("alto", 1),
        "intensidad": track.get("intensidad", 0),
        "nivel_intensidad": track.get("nivel_intensidad", ""),
        "estado": track.get("estado", ""),
        "confianza": track.get("confianza", 0),
        "dx_pixeles": track.get("dx_pixeles", 0),
        "dy_pixeles": track.get("dy_pixeles", 0),
        "desplazamiento_pixeles": track.get(
            "desplazamiento_pixeles", 0
        ),
        "velocidad_pixeles_min": track.get(
            "velocidad_pixeles_min", 0
        ),
        "rumbo_grados": track.get("rumbo_grados"),
        "direccion": track.get("direccion"),
        "aceleracion_pixeles_min2": track.get(
            "aceleracion_pixeles_min2", 0
        ),
        "cambio_rumbo_grados": track.get(
            "cambio_rumbo_grados", 0
        ),
        "calidad_movimiento": track.get(
            "calidad_movimiento", "SIN_MOVIMIENTO"
        ),
        "cambio_area_porcentaje": track.get(
            "cambio_area_porcentaje", 0
        ),
        "tendencia_area": track.get(
            "tendencia_area", "ESTABLE"
        ),
        "tendencia_velocidad": track.get(
            "tendencia_velocidad", "ESTABLE"
        ),
        "confianza_movimiento": track.get(
            "confianza_movimiento", 0
        ),
        "historial": track.get("historial", []),
    }


def salida_sin_nucleos(radar_ts, intervalo, calidad, motivo=None):
    salida = {
        "schema_version": SCHEMA_VERSION,
        "radar_timestamp": radar_ts,
        "frame_anterior_disponible": intervalo is not None,
        "intervalo_minutos": (
            round(intervalo, 2) if intervalo is not None else None
        ),
        "calidad_intervalo": calidad,
        "nucleos_actuales": 0,
        "nucleos_nuevos": 0,
        "nucleos_seguimiento": 0,
        "nucleos_creciendo": 0,
        "nucleos_debilitandose": 0,
        "nucleos_movimiento": 0,
        "nucleos_estables": 0,
        "nucleos_desaparecidos": 0,
        "asociaciones": [],
        "nucleos": [],
    }

    if motivo:
        salida["motivo"] = motivo

    return salida


def procesar():
    nucleos = obtener_nucleos()
    radar_dt = timestamp_radar()
    radar_ts = iso(radar_dt)

    estado = normalizar_estado(
        cargar_json(ESTADO, {})
    )

    anterior_ts = estado.get("ultimo_radar_timestamp")
    anterior_dt = parse_iso(anterior_ts)

    if anterior_dt is None:
        if not nucleos:
            salida = salida_sin_nucleos(
                radar_ts,
                None,
                "PRIMER_FRAME_SIN_NUCLEOS",
                "El procesador no detectó núcleos en este frame.",
            )

            estado["ultimo_radar_timestamp"] = radar_ts
            estado["tracks_activos"] = []
            estado["ultimo_frame_nucleos"] = []

            guardar_json(ESTADO, estado)
            guardar_json(SALIDA, salida)

            return salida

        actuales = nueva_base_desde_frame(
            estado,
            nucleos,
            radar_ts
        )

        salida = {
            "schema_version": SCHEMA_VERSION,
            "radar_timestamp": radar_ts,
            "frame_anterior_disponible": False,
            "intervalo_minutos": None,
            "calidad_intervalo": "PRIMER_FRAME",
            "nucleos_actuales": len(actuales),
            "nucleos_nuevos": len(actuales),
            "nucleos_seguimiento": 0,
            "nucleos_creciendo": 0,
            "nucleos_debilitandose": 0,
            "nucleos_movimiento": 0,
            "nucleos_estables": 0,
            "nucleos_desaparecidos": 0,
            "asociaciones": [],
            "nucleos": [
                nucleo_salida(t)
                for t in actuales
            ],
        }

        guardar_json(ESTADO, estado)
        guardar_json(SALIDA, salida)

        return salida

    intervalo = (
        radar_dt - anterior_dt
    ).total_seconds() / 60.0

    if intervalo <= 0:
        salida = {
            "schema_version": SCHEMA_VERSION,
            "radar_timestamp": radar_ts,
            "frame_anterior_disponible": True,
            "intervalo_minutos": round(intervalo, 2),
            "calidad_intervalo": "FRAME_REPETIDO_O_ATRASADO",
            "nucleos_actuales": len(nucleos),
            "nucleos_nuevos": 0,
            "nucleos_seguimiento": 0,
            "nucleos_creciendo": 0,
            "nucleos_debilitandose": 0,
            "nucleos_movimiento": 0,
            "nucleos_estables": 0,
            "nucleos_desaparecidos": 0,
            "asociaciones": [],
            "nucleos": [],
            "motivo": (
                "El frame actual no es posterior "
                "al último estado procesado."
            ),
        }

        guardar_json(SALIDA, salida)
        return salida

    if not nucleos:
        activos = []

        for track in estado["tracks"]:
            if track.get("ultimo_frame") != anterior_ts:
                continue

            track["presente_en_ultimo_frame"] = False
            track["missed_frames"] = (
                entero(track.get("missed_frames", 0)) + 1
            )

            if track["missed_frames"] > MAX_MISSED_FRAMES:
                track["activo"] = False
            else:
                track["activo"] = True
                activos.append(track["track_id"])

        estado["tracks_activos"] = activos
        estado["ultimo_frame_nucleos"] = []
        estado["ultimo_radar_timestamp"] = radar_ts

        salida = salida_sin_nucleos(
            radar_ts,
            intervalo,
            "FRAME_SIN_NUCLEOS",
            "El procesador no detectó núcleos en este frame; "
            "se conserva el historial sin inventar movimiento.",
        )

        guardar_json(ESTADO, estado)
        guardar_json(SALIDA, salida)

        return salida

    if intervalo > MAX_INTERVALO:
        actuales = nueva_base_desde_frame(
            estado,
            nucleos,
            radar_ts
        )

        salida = {
            "schema_version": SCHEMA_VERSION,
            "radar_timestamp": radar_ts,
            "frame_anterior_disponible": True,
            "intervalo_minutos": round(intervalo, 2),
            "calidad_intervalo": "GAP_MAYOR_30_MIN",
            "nucleos_actuales": len(actuales),
            "nucleos_nuevos": len(actuales),
            "nucleos_seguimiento": 0,
            "nucleos_creciendo": 0,
            "nucleos_debilitandose": 0,
            "nucleos_movimiento": 0,
            "nucleos_estables": 0,
            "nucleos_desaparecidos": 0,
            "asociaciones": [],
            "nucleos": [
                nucleo_salida(t)
                for t in actuales
            ],
            "motivo": (
                "El intervalo supera 30 minutos. "
                "Se creó una nueva base sin inventar "
                "movimiento entre frames."
            ),
        }

        guardar_json(ESTADO, estado)
        guardar_json(SALIDA, salida)

        return salida

    activos_previos = [
        track for track in estado["tracks"]
        if (
            track.get("activo") is True
            and track.get("ultimo_frame") == anterior_ts
        )
    ]

    candidatos = []

    for track in activos_previos:
        anterior = {
            "x": numero(track.get("x")),
            "y": numero(track.get("y")),
            "pixeles": entero(track.get("pixeles")),
            "ancho": numero(track.get("ancho"), 1),
            "alto": numero(track.get("alto"), 1),
        }

        for indice, actual in enumerate(nucleos):
            comp = compatibilidad(
                anterior,
                actual
            )

            if comp is None:
                continue

            candidatos.append({
                "track": track,
                "indice": indice,
                **comp,
            })

    candidatos.sort(
        key=lambda c: c["score"],
        reverse=True
    )

    asignados_tracks = set()
    asignados_nucleos = set()
    asociaciones = []

    for candidato in candidatos:
        tid = candidato["track"]["track_id"]
        indice = candidato["indice"]

        if tid in asignados_tracks:
            continue

        if indice in asignados_nucleos:
            continue

        asignados_tracks.add(tid)
        asignados_nucleos.add(indice)
        asociaciones.append(candidato)

    salida_nucleos = []

    creciendo = 0
    debilitandose = 0
    movimiento = 0
    estables = 0
    nuevos = 0

    for asociacion in asociaciones:
        track = asociacion["track"]
        nucleo = nucleos[asociacion["indice"]]

        if intervalo < 5:
            calidad = "BUENA"
        elif intervalo <= 15:
            calidad = "ACEPTABLE"
        else:
            calidad = "BAJA"

        actualizar_track(
            track,
            nucleo,
            radar_ts,
            intervalo,
            calidad
        )

        cambio = numero(
            track.get("cambio_area_porcentaje", 0)
        )

        if cambio >= 15:
            creciendo += 1
        elif cambio <= -15:
            debilitandose += 1
        else:
            estables += 1

        if track.get("desplazamiento_pixeles", 0) >= 1:
            movimiento += 1

        salida_nucleos.append(
            nucleo_salida(track)
        )

    desaparecidos = 0

    for track in activos_previos:
        tid = track["track_id"]

        if tid in asignados_tracks:
            continue

        track["presente_en_ultimo_frame"] = False
        track["missed_frames"] = (
            entero(track.get("missed_frames", 0)) + 1
        )

        if track["missed_frames"] > MAX_MISSED_FRAMES:
            track["activo"] = False
            desaparecidos += 1
        else:
            track["activo"] = True

    for indice, nucleo in enumerate(nucleos):
        if indice in asignados_nucleos:
            continue

        tid = estado["siguiente_track_id"]
        estado["siguiente_track_id"] += 1

        track = crear_track(
            tid,
            nucleo,
            radar_ts
        )

        estado["tracks"].append(track)

        salida_nucleos.append(
            nucleo_salida(track)
        )

        nuevos += 1

    tracks_activos = [
        track["track_id"]
        for track in estado["tracks"]
        if (
            track.get("activo") is True
            and track.get("presente_en_ultimo_frame") is True
        )
    ]

    estado["tracks_activos"] = tracks_activos
    estado["ultimo_frame_nucleos"] = salida_nucleos
    estado["ultimo_radar_timestamp"] = radar_ts

    salida = {
        "schema_version": SCHEMA_VERSION,
        "radar_timestamp": radar_ts,
        "frame_anterior_disponible": True,
        "intervalo_minutos": round(intervalo, 2),
        "calidad_intervalo": (
            "BUENA"
            if intervalo <= 10
            else "ACEPTABLE"
        ),
        "nucleos_actuales": len(nucleos),
        "nucleos_nuevos": nuevos,
        "nucleos_seguimiento": len(asociaciones),
        "nucleos_creciendo": creciendo,
        "nucleos_debilitandose": debilitandose,
        "nucleos_movimiento": movimiento,
        "nucleos_estables": estables,
        "nucleos_desaparecidos": desaparecidos,
        "asociaciones": [
            {
                "track_id": a["track"]["track_id"],
                "indice_nucleo": a["indice"],
                "score": a["score"],
                "distancia": a["distancia"],
                "iou": a["iou"],
                "similitud_area": a["similitud_area"],
                "similitud_forma": a["similitud_forma"],
            }
            for a in asociaciones
        ],
        "nucleos": salida_nucleos,
    }

    guardar_json(ESTADO, estado)
    guardar_json(SALIDA, salida)

    return salida


if __name__ == "__main__":
    resultado = procesar()

    print("")
    print("======================================")
    print("SEGUIMIENTO TEMPORAL TERMINADO")
    print("======================================")
    print(
        f"Núcleos actuales: "
        f"{resultado.get('nucleos_actuales', 0)}"
    )
    print(
        f"Seguimientos: "
        f"{resultado.get('nucleos_seguimiento', 0)}"
    )
    print(
        f"Nuevos: "
        f"{resultado.get('nucleos_nuevos', 0)}"
    )
    print(
        f"Desaparecidos: "
        f"{resultado.get('nucleos_desaparecidos', 0)}"
    )
