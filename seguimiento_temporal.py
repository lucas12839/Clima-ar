import json
import math
from pathlib import Path
from datetime import datetime, timezone, timedelta
from PIL import Image

BASE = Path("data/radar")
RESULTADO = BASE / "resultado_procesamiento.json"
SEGUIMIENTO = BASE / "seguimiento"
ESTADO = SEGUIMIENTO / "estado.json"
SALIDA = SEGUIMIENTO / "seguimiento_temporal.json"

SCHEMA_VERSION = 6

MIN_INTERVALO = 2.0
MAX_INTERVALO = 30.0

# Matching adaptativo para radar RMA10.
# El valor anterior de 45 px era demasiado restrictivo para frames
# separados ~10 minutos.
MAX_DISTANCIA_BASE = 70.0
MAX_DISTANCIA_SIN_OVERLAP = 85.0
MAX_MISSED_FRAMES = 2
MAX_HISTORIAL = 20
MAX_REPLAY_FRAMES = 6


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


def posicion_predicha(track, intervalo):
    x = numero(track.get("x"))
    y = numero(track.get("y"))
    vx = numero(track.get("velocidad_x_pixeles_min"), 0.0)
    vy = numero(track.get("velocidad_y_pixeles_min"), 0.0)

    return (
        x + vx * intervalo,
        y + vy * intervalo,
    )


def distancia_predicha(track, actual, intervalo):
    px, py = posicion_predicha(track, intervalo)
    return math.hypot(actual["x"] - px, actual["y"] - py)


def distancia_maxima(intervalo, track=None):
    # Permite más desplazamiento cuanto mayor sea el intervalo.
    # Si el track ya tiene velocidad, adapta el radio a su movimiento.
    base = max(
        MAX_DISTANCIA_BASE,
        7.0 * max(intervalo, 1.0),
    )

    if track is not None:
        velocidad = math.hypot(
            numero(track.get("velocidad_x_pixeles_min"), 0.0),
            numero(track.get("velocidad_y_pixeles_min"), 0.0),
        )
        base = max(base, velocidad * intervalo * 2.5 + 25.0)

    return min(base, 180.0)


def compatibilidad(anterior, actual, intervalo, track=None):
    d = distancia(anterior, actual)
    overlap = iou(anterior, actual)

    area_a = max(anterior["pixeles"], 1)
    area_b = max(actual["pixeles"], 1)
    area_min = min(area_a / area_b, area_b / area_a)

    forma = similitud_forma(anterior, actual)
    limite = distancia_maxima(intervalo, track)

    if d > limite:
        return None

    d_pred = None
    if track is not None:
        d_pred = distancia_predicha(track, actual, intervalo)

    # Quando existe movimento conhecido, a posição prevista ganha
    # prioridade para não perder o mesmo núcleo quando atravessa pixels.
    if d_pred is not None:
        limite_pred = min(180.0, max(60.0, limite * 1.25))
        if d_pred > limite_pred and overlap < 0.02:
            return None

    if overlap < 0.02:
        if d > MAX_DISTANCIA_SIN_OVERLAP and d_pred is None:
            return None
        if area_min < 0.45:
            return None
        if forma < 0.35:
            return None
    else:
        if area_min < 0.25:
            return None
        if forma < 0.25:
            return None

    proximidad = max(0.0, 1.0 - d / max(limite, 1.0))
    forma_score = forma * 10.0
    area_score = area_min * 15.0
    overlap_score = overlap * 30.0

    if d_pred is None:
        pred_score = 0.0
    else:
        pred_limite = max(60.0, limite * 1.25)
        pred_score = max(
            0.0,
            1.0 - d_pred / pred_limite
        ) * 35.0

    score = (
        proximidad * 25.0
        + overlap_score
        + area_score
        + forma_score
        + pred_score
    )

    return {
        "score": round(score, 2),
        "distancia": round(d, 3),
        "distancia_predicha": (
            round(d_pred, 3) if d_pred is not None else None
        ),
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

        # Migración transparente: tracks anteriores no tenían vx/vy.
        if "velocidad_x_pixeles_min" not in track:
            track["velocidad_x_pixeles_min"] = 0.0
        if "velocidad_y_pixeles_min" not in track:
            track["velocidad_y_pixeles_min"] = 0.0

        nuevo["tracks"].append(track)
        max_id = max(max_id, tid)

    siguiente = entero(
        estado.get("siguiente_track_id"),
        max_id + 1
    )
    nuevo["siguiente_track_id"] = max(
        siguiente,
        max_id + 1
    )

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
        "velocidad_x_pixeles_min": 0.0,
        "velocidad_y_pixeles_min": 0.0,
        "rumbo_grados": None,
        "direccion": None,
        "aceleracion_pixeles_min2": 0.0,
        "cambio_rumbo_grados": 0.0,
        "calidad_movimiento": "PRIMER_FRAME",
        "cambio_area_porcentaje": 0.0,
        "tendencia_area": "ESTABLE",
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
        velocidad_x = dx / intervalo
        velocidad_y = dy / intervalo
        aceleracion = (
            velocidad - anterior_velocidad
        ) / intervalo
    else:
        velocidad = 0.0
        velocidad_x = 0.0
        velocidad_y = 0.0
        aceleracion = 0.0

    grados, direccion = rumbo(dx, dy)

    cambio_rumbo = 0.0
    if anterior_rumbo is not None and grados is not None:
        cambio_rumbo = abs(
            (grados - numero(anterior_rumbo) + 180) % 360 - 180
        )

    anterior_area = max(
        numero(track.get("pixeles"), 1),
        1
    )
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
        estabilidad = max(
            0.0,
            1.0 - min(cambio_rumbo, 180) / 180
        )
        confianza_movimiento = min(
            1.0,
            0.50 + estabilidad * 0.35
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
        "velocidad_x_pixeles_min": velocidad_x,
        "velocidad_y_pixeles_min": velocidad_y,
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

        track = crear_track(
            tid,
            nucleo,
            radar_ts
        )
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
        "velocidad_x_pixeles_min": track.get(
            "velocidad_x_pixeles_min", 0
        ),
        "velocidad_y_pixeles_min": track.get(
            "velocidad_y_pixeles_min", 0
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
            "calidad_movimiento",
            "SIN_MOVIMIENTO"
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
            round(intervalo, 2)
            if intervalo is not None
            else None
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


def frames_historicos_validos(radar_dt):
    candidatos = []

    for archivo in BASE.glob("historico/*.png"):
        if not archivo.stem.isdigit() or len(archivo.stem) != 14:
            continue

        try:
            dt = datetime.strptime(
                archivo.stem,
                "%Y%m%d%H%M%S"
            ).replace(tzinfo=timezone.utc)
        except Exception:
            continue

        if dt > radar_dt + timedelta(minutes=2):
            continue

        candidatos.append((dt, archivo))

    candidatos.sort(key=lambda item: item[0])

    if not candidatos:
        return []

    return candidatos[-MAX_REPLAY_FRAMES:]


def nucleos_desde_imagen(path, mascara, detectar_func, analizar_func, confianza_func):
    imagen = Image.open(path).convert("RGBA")
    detectados = detectar_func(imagen)

    salida = []

    for indice, nucleo in enumerate(detectados):
        estado_nucleo, confianza_base = analizar_func(
            nucleo,
            mascara
        )

        confianza = confianza_func(
            nucleo,
            estado_nucleo,
            confianza_base
        )

        item = dict(nucleo)
        item["indice"] = indice
        item["estado"] = estado_nucleo
        item["confianza"] = confianza
        salida.append(normalizar_nucleo(item, indice))

    # Los ecos estáticos no entran al seguimiento temporal.
    return [
        n for n in salida
        if n.get("estado") != "ECO_ESTATICO"
    ]


def procesar_frame_secuencia(estado, nucleos, radar_ts, anterior_ts):
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

        return salida_sin_nucleos(
            radar_ts,
            None if anterior_ts is None else 0,
            "FRAME_SIN_NUCLEOS",
            "No se detectaron núcleos no-estáticos en este frame."
        )

    if anterior_ts is None:
        actuales = nueva_base_desde_frame(
            estado,
            nucleos,
            radar_ts
        )

        return {
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
            "nucleos": [nucleo_salida(t) for t in actuales],
        }

    anterior_dt = parse_iso(anterior_ts)
    actual_dt = parse_iso(radar_ts)

    if anterior_dt is None or actual_dt is None:
        actuales = nueva_base_desde_frame(
            estado,
            nucleos,
            radar_ts
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "radar_timestamp": radar_ts,
            "frame_anterior_disponible": False,
            "intervalo_minutos": None,
            "calidad_intervalo": "REINICIO_TIMESTAMP_INVALIDO",
            "nucleos_actuales": len(actuales),
            "nucleos_nuevos": len(actuales),
            "nucleos_seguimiento": 0,
            "nucleos_movimiento": 0,
            "nucleos": [nucleo_salida(t) for t in actuales],
        }

    intervalo = (actual_dt - anterior_dt).total_seconds() / 60.0

    if intervalo <= 0 or intervalo > MAX_INTERVALO:
        actuales = nueva_base_desde_frame(
            estado,
            nucleos,
            radar_ts
        )
        calidad = (
            "FRAME_REPETIDO_O_ATRASADO"
            if intervalo <= 0
            else "GAP_MAYOR_30_MIN"
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "radar_timestamp": radar_ts,
            "frame_anterior_disponible": intervalo > 0,
            "intervalo_minutos": round(intervalo, 2),
            "calidad_intervalo": calidad,
            "nucleos_actuales": len(actuales),
            "nucleos_nuevos": len(actuales),
            "nucleos_seguimiento": 0,
            "nucleos_creciendo": 0,
            "nucleos_debilitandose": 0,
            "nucleos_movimiento": 0,
            "nucleos_estables": 0,
            "nucleos_desaparecidos": 0,
            "asociaciones": [],
            "nucleos": [nucleo_salida(t) for t in actuales],
        }

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
                actual,
                intervalo,
                track
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

        if tid in asignados_tracks or indice in asignados_nucleos:
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

    calidad = "BUENA" if intervalo < 5 else "ACEPTABLE" if intervalo <= 15 else "BAJA"

    for asociacion in asociaciones:
        track = asociacion["track"]
        nucleo = nucleos[asociacion["indice"]]

        actualizar_track(
            track,
            nucleo,
            radar_ts,
            intervalo,
            calidad
        )

        cambio = numero(track.get("cambio_area_porcentaje", 0))

        if cambio >= 15:
            creciendo += 1
        elif cambio <= -15:
            debilitandose += 1
        else:
            estables += 1

        if track.get("desplazamiento_pixeles", 0) >= 1:
            movimiento += 1

        salida_nucleos.append(nucleo_salida(track))

    desaparecidos = 0

    for track in activos_previos:
        tid = track["track_id"]
        if tid in asignados_tracks:
            continue

        track["presente_en_ultimo_frame"] = False
        track["missed_frames"] = entero(track.get("missed_frames", 0)) + 1

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

        track = crear_track(tid, nucleo, radar_ts)
        estado["tracks"].append(track)
        salida_nucleos.append(nucleo_salida(track))
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

    return {
        "schema_version": SCHEMA_VERSION,
        "radar_timestamp": radar_ts,
        "frame_anterior_disponible": True,
        "intervalo_minutos": round(intervalo, 2),
        "calidad_intervalo": calidad,
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
                "distancia_predicha": a.get("distancia_predicha"),
                "iou": a["iou"],
                "similitud_area": a["similitud_area"],
                "similitud_forma": a["similitud_forma"],
            }
            for a in asociaciones
        ],
        "nucleos": salida_nucleos,
    }


def procesar():
    radar_dt = parse_iso(
        cargar_json(Path("data/radar/radar_timestamp.txt"), None)
    )

    timestamp_path = BASE / "radar_timestamp.txt"
    if timestamp_path.exists():
        radar_dt = parse_iso(timestamp_path.read_text(encoding="utf-8").strip())

    if radar_dt is None:
        radar_dt = timestamp_radar()

    radar_ts = iso(radar_dt)
    mascara = None

    try:
        from scripts.procesador_radar import (
            detectar_nucleos as detectar_nucleos_desde_procesador,
            cargar_mascara as cargar_mascara_desde_procesador,
            analizar_mascara as analizar_mascara_desde_procesador,
            calcular_confianza as calcular_confianza_desde_procesador,
        )
        mascara = cargar_mascara_desde_procesador()
    except Exception as exc:
        raise RuntimeError(
            "No se pudo cargar el procesador RMA10 para reconstruir el historial."
        ) from exc

    frames = frames_historicos_validos(radar_dt)

    actual_path = BASE / "actual.png"
    timestamps_presentes = {iso(dt) for dt, _ in frames}

    if actual_path.exists() and radar_ts not in timestamps_presentes:
        frames.append((radar_dt, actual_path))

    frames.sort(key=lambda item: item[0])

    if len(frames) > MAX_REPLAY_FRAMES:
        frames = frames[-MAX_REPLAY_FRAMES:]

    if not frames:
        raise RuntimeError("No hay frames RMA10 disponibles para reconstruir el seguimiento.")

    estado = estado_nuevo()
    ultimo_resultado = None
    resumen_replay = []

    for frame_dt, frame_path in frames:
        frame_ts = iso(frame_dt)
        nucleos = nucleos_desde_imagen(
            frame_path,
            mascara,
            detectar_nucleos_desde_procesador,
            analizar_mascara_desde_procesador,
            calcular_confianza_desde_procesador,
        )

        resultado = procesar_frame_secuencia(
            estado,
            nucleos,
            frame_ts,
            estado.get("ultimo_radar_timestamp")
        )

        ultimo_resultado = resultado
        resumen_replay.append({
            "timestamp": frame_ts,
            "archivo": frame_path.name,
            "nucleos": len(nucleos),
            "seguimiento": resultado.get("nucleos_seguimiento", 0),
            "movimiento": resultado.get("nucleos_movimiento", 0),
        })

    if ultimo_resultado is None:
        raise RuntimeError("No se pudo reconstruir el seguimiento RMA10.")

    ultimo_resultado["modo_seguimiento"] = "RECONSTRUCCION_HISTORICA"
    ultimo_resultado["frames_reconstruidos"] = len(frames)
    ultimo_resultado["historial_replay"] = resumen_replay

    guardar_json(ESTADO, estado)
    guardar_json(SALIDA, ultimo_resultado)

    return ultimo_resultado


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
