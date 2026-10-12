import json
import math
from pathlib import Path
from datetime import datetime, timezone

BASE = Path("data/radar")
RESULTADO = BASE / "resultado_procesamiento.json"
SEGUIMIENTO = BASE / "seguimiento"
ESTADO = SEGUIMIENTO / "estado.json"
SALIDA = SEGUIMIENTO / "seguimiento_temporal.json"

SCHEMA_VERSION = 5

MIN_INTERVALO = 2.0
MAX_INTERVALO = 30.0

MAX_DISTANCIA = 45.0
MAX_DISTANCIA_SIN_OVERLAP = 25.0

MAX_MISSED_FRAMES = 2


# ---------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------

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
    archivos = sorted(
        BASE.glob("historico/*.png"),
        key=lambda p: p.name
    )

    candidatos = []

    for p in archivos:
        if len(p.stem) == 14 and p.stem.isdigit():
            candidatos.append(p.stem)

    if not candidatos:
        raise RuntimeError("No se encontró timestamp de radar.")

    ts = candidatos[-1]

    dt = datetime.strptime(ts, "%Y%m%d%H%M%S")
    dt = dt.replace(tzinfo=timezone.utc)

    return dt


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        ).astimezone(timezone.utc)
    except Exception:
        return None


# ---------------------------------------------------------
# NORMALIZACIÓN DE NÚCLEOS
# ---------------------------------------------------------

def normalizar_nucleo(n, indice=0):

    x = numero(
        n.get("x", n.get("centro_x", n.get("centroid_x", 0)))
    )

    y = numero(
        n.get("y", n.get("centro_y", n.get("centroid_y", 0)))
    )

    pixeles = entero(
        n.get("pixeles", n.get("pixels", n.get("area", 0)))
    )

    ancho = numero(
        n.get("ancho", n.get("width", 1)),
        1
    )

    alto = numero(
        n.get("alto", n.get("height", 1)),
        1
    )

    intensidad = entero(
        n.get(
            "intensidad_maxima",
            n.get("max_intensity", n.get("intensidad", 0))
        )
    )

    nivel = n.get(
        "nivel_intensidad",
        n.get("level", "")
    )

    confianza = numero(
        n.get("confianza", n.get("confidence", 0))
    )

    estado = n.get(
        "estado",
        n.get("state", "")
    )

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
        "estado": estado
    }


def obtener_nucleos():

    data = cargar_json(RESULTADO, {})

    candidatos = (
        data.get("nucleos")
        or data.get("núcleos")
        or data.get("cells")
        or data.get("ecos")
        or data.get("detected_cells")
        or []
    )

    return [
        normalizar_nucleo(n, i)
        for i, n in enumerate(candidatos)
    ]


# ---------------------------------------------------------
# GEOMETRÍA
# ---------------------------------------------------------

def bbox(n):

    x = n["x"]
    y = n["y"]

    w = max(n["ancho"], 1)
    h = max(n["alto"], 1)

    return (
        x - w / 2,
        y - h / 2,
        x + w / 2,
        y + h / 2
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

    if union <= 0:
        return 0.0

    return inter / union


def distancia(a, b):

    return math.hypot(
        b["x"] - a["x"],
        b["y"] - a["y"]
    )


def similitud_area(a, b):

    aa = max(a["pixeles"], 1)
    ab = max(b["pixeles"], 1)

    ratio = aa / ab

    if ratio <= 0:
        return 0.0

    return min(ratio, 1 / ratio)


def similitud_forma(a, b):

    wa = max(a["ancho"], 1)
    wb = max(b["ancho"], 1)

    ha = max(a["alto"], 1)
    hb = max(b["alto"], 1)

    sw = min(wa, wb) / max(wa, wb)
    sh = min(ha, hb) / max(ha, hb)

    return (sw + sh) / 2


# ---------------------------------------------------------
# COMPATIBILIDAD
# ---------------------------------------------------------

def compatibilidad(anterior, actual):

    d = distancia(anterior, actual)
    overlap = iou(anterior, actual)

    area_ratio = max(
        anterior["pixeles"],
        1
    ) / max(
        actual["pixeles"],
        1
    )

    area_ratio_inv = 1 / area_ratio if area_ratio else 999

    area_min = min(area_ratio, area_ratio_inv)

    forma = similitud_forma(anterior, actual)

    # ---------------------------------------------
    # RECHAZOS DUROS
    # ---------------------------------------------

    if d > MAX_DISTANCIA:
        return None

    # Sin solapamiento necesitamos mucha más evidencia.
    if overlap < 0.02:

        if d > MAX_DISTANCIA_SIN_OVERLAP:
            return None

        if area_min < 0.65:
            return None

        if forma < 0.55:
            return None

    # Si hay solapamiento podemos aceptar cambios
    # moderados de área.
    else:

        if area_min < 0.40:
            return None

        if forma < 0.35:
            return None

    # ---------------------------------------------
    # PUNTAJES
    # ---------------------------------------------

    proximidad = max(
        0.0,
        1.0 - d / MAX_DISTANCIA
    )

    area_score = area_min

    forma_score = forma

    score = (
        proximidad * 40
        + overlap * 35
        + area_score * 15
        + forma_score * 10
    )

    return {
        "score": round(score, 2),
        "distancia": round(d, 3),
        "iou": round(overlap, 4),
        "similitud_area": round(area_score, 4),
        "similitud_forma": round(forma_score, 4)
    }


# ---------------------------------------------------------
# DIRECCIÓN
# ---------------------------------------------------------

def rumbo(dx, dy):

    if abs(dx) < 0.01 and abs(dy) < 0.01:
        return None, None

    grados = math.degrees(
        math.atan2(dx, -dy)
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
        ((grados + 11.25) % 360) / 22.5
    )

    return grados, sectores[indice]


# ---------------------------------------------------------
# TRACKS
# ---------------------------------------------------------

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

        "velocidad_pixeles_min": 0.0,
        "rumbo_grados": None,
        "direccion": None,

        "aceleracion_pixeles_min2": 0.0,
        "cambio_rumbo_grados": 0.0,

        "historial": [
            {
                "timestamp": radar_ts,
                "x": nucleo["x"],
                "y": nucleo["y"],
                "pixeles": nucleo["pixeles"],
                "ancho": nucleo["ancho"],
                "alto": nucleo["alto"],
                "intensidad": nucleo["intensidad"]
            }
        ]
    }


def actualizar_track(
    track,
    nucleo,
    radar_ts,
    intervalo,
    calidad
):

    anterior_x = track["x"]
    anterior_y = track["y"]

    anterior_velocidad = numero(
        track.get("velocidad_pixeles_min", 0)
    )

    anterior_rumbo = track.get("rumbo_grados")

    dx = nucleo["x"] - anterior_x
    dy = nucleo["y"] - anterior_y

    desplazamiento = math.hypot(dx, dy)

    velocidad = 0.0

    if (
        intervalo >= MIN_INTERVALO
        and intervalo <= MAX_INTERVALO
    ):
        velocidad = desplazamiento / intervalo

    grados, direccion = rumbo(dx, dy)

    aceleracion = 0.0

    if (
        intervalo >= MIN_INTERVALO
        and intervalo <= MAX_INTERVALO
    ):
        aceleracion = (
            velocidad - anterior_velocidad
        ) / intervalo

    cambio_rumbo = 0.0

    if (
        anterior_rumbo is not None
        and grados is not None
    ):
        cambio_rumbo = abs(
            (grados - anterior_rumbo + 180) % 360 - 180
        )

    historial = track.setdefault("historial", [])

    historial.append({
        "timestamp": radar_ts,
        "x": nucleo["x"],
        "y": nucleo["y"],
        "pixeles": nucleo["pixeles"],
        "ancho": nucleo["ancho"],
        "alto": nucleo["alto"],
        "intensidad": nucleo["intensidad"]
    })

    # Conservamos bastante historia para predicción.
    if len(historial) > 20:
        del historial[:-20]

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

        "calidad_movimiento": calidad
    })


# ---------------------------------------------------------
# ESTADO NUEVO
# ---------------------------------------------------------

def estado_nuevo():

    return {
        "schema_version": SCHEMA_VERSION,
        "ultimo_radar_timestamp": None,
        "siguiente_track_id": 1,

        # TODOS los tracks históricos.
        "tracks": [],

        # SOLO los tracks que pertenecen al último frame.
        "tracks_activos": [],

        "ultimo_frame_nucleos": []
    }


# ---------------------------------------------------------
# LIMPIAR ESTADO ANTIGUO
# ---------------------------------------------------------

def normalizar_estado_viejo(estado):

    if not isinstance(estado, dict):
        return estado_nuevo()

    nuevo = estado_nuevo()

    nuevo["ultimo_radar_timestamp"] = estado.get(
        "ultimo_radar_timestamp"
    )

    tracks = estado.get("tracks", [])

    if not isinstance(tracks, list):
        tracks = []

    # Reparar IDs inválidos.
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
        max_id = max(max_id, tid)

        if not isinstance(track.get("historial"), list):
            track["historial"] = []

        nuevo["tracks"].append(track)

    siguiente = entero(
        estado.get("siguiente_track_id"),
        max_id + 1
    )

    nuevo["siguiente_track_id"] = max(
        siguiente,
        max_id + 1
    )

    # Recalcular activos únicamente a partir del último
    # conjunto válido guardado.
    ultimo_ts = nuevo["ultimo_radar_timestamp"]

    activos = []

    for track in nuevo["tracks"]:

        if (
            track.get("activo") is True
            and track.get("ultimo_frame") == ultimo_ts
        ):
            activos.append(track["track_id"])

    nuevo["tracks_activos"] = activos

    nuevo["ultimo_frame_nucleos"] = estado.get(
        "ultimo_frame_nucleos",
        []
    )

    return nuevo


# ---------------------------------------------------------
# CREAR NUEVA BASE
# ---------------------------------------------------------

def nueva_base_desde_frame(
    estado,
    nucleos,
    radar_ts
):

    # Los tracks anteriores quedan históricos.
    for track in estado["tracks"]:
        track["activo"] = False
        track["presente_en_ultimo_frame"] = False

    activos = []
    nucleos_salida = []

    for nucleo in nucleos:

        tid = estado["siguiente_track_id"]

        estado["siguiente_track_id"] += 1

        track = crear_track(
            tid,
            nucleo,
            radar_ts
        )

        estado["tracks"].append(track)

        activos.append(tid)

        salida = dict(nucleo)
        salida["track_id"] = tid
        nucleos_salida.append(salida)

    estado["ultimo_radar_timestamp"] = radar_ts
    estado["tracks_activos"] = activos
    estado["ultimo_frame_nucleos"] = nucleos_salida

    return nucleos_salida


# ---------------------------------------------------------
# PROCESAR FRAME
# ---------------------------------------------------------

def procesar():

    nucleos = obtener_nucleos()

    if not nucleos:
        raise RuntimeError(
            "El procesador no entregó núcleos."
        )

    radar_dt = timestamp_radar()
    radar_ts = iso(radar_dt)

    estado = normalizar_estado_viejo(
        cargar_json(ESTADO, {})
    )

    anterior_ts = estado.get(
        "ultimo_radar_timestamp"
    )

    anterior_dt = parse_iso(anterior_ts)

    # -----------------------------------------------------
    # PRIMER FRAME
    # -----------------------------------------------------

    if anterior_dt is None:

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

            "nucleos": actuales
        }

        guardar_json(ESTADO, estado)
        guardar_json(SALIDA, salida)

        return salida

    intervalo = (
        radar_dt - anterior_dt
    ).total_seconds() / 60.0

    # -----------------------------------------------------
    # FRAME REPETIDO / ATRASADO
    # -----------------------------------------------------

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

            "nucleos": nucleos,

            "motivo": (
                "El frame actual no es posterior "
                "al último estado procesado."
            )
        }

        # MUY IMPORTANTE:
        # NO modificamos el estado.

        guardar_json(SALIDA, salida)

        return salida

    # -----------------------------------------------------
    # GAP MAYOR A 30 MIN
    # -----------------------------------------------------

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

            "nucleos": actuales,

            "motivo": (
                "El intervalo supera 30 minutos. "
                "Se creó una nueva base sin inventar "
                "movimiento entre frames."
            )
        }

        guardar_json(ESTADO, estado)
        guardar_json(SALIDA, salida)

        return salida

    # -----------------------------------------------------
    # FRAME VÁLIDO
    # -----------------------------------------------------

    activos_previos = []

    for track in estado["tracks"]:

        if (
            track.get("activo") is True
            and track.get("ultimo_frame") == anterior_ts
        ):
            activos_previos.append(track)

    # ---------------------------------------------
    # CANDIDATOS
    # ---------------------------------------------

    candidatos = []

    for track in activos_previos:

        anterior = {
            "x": numero(track.get("x")),
            "y": numero(track.get("y")),
            "pixeles": entero(track.get("pixeles")),
            "ancho": numero(track.get("ancho"), 1),
            "alto": numero(track.get("alto"), 1)
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
                **comp
            })

    # ---------------------------------------------
    # ASIGNACIÓN GLOBAL UNO A UNO
    # ---------------------------------------------

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

    # -----------------------------------------------------
    # ACTUALIZAR TRACKS ASOCIADOS
    # -----------------------------------------------------

    salida_nucleos = []

    creciendo = 0
    debilitandose = 0
    movimiento = 0
    estables = 0

    for asociacion in asociaciones:

        track = asociacion["track"]
        indice = asociacion["indice"]
        nucleo = nucleos[indice]

        anterior_area = max(
            track.get("pixeles", 1),
            1
        )

        area_change = (
            nucleo["pixeles"] - anterior_area
        ) / anterior_area * 100

        if area_change >= 15:
            creciendo += 1
            tendencia_area = "CRECIENDO"

        elif area_change <= -15:
            debilitandose += 1
            tendencia_area = "DEBILITANDOSE"

        else:
            estables += 1
            tendencia_area = "ESTABLE"

        anterior = {
            "x": numero(track.get("x")),
            "y": numero(track.get("y"))
        }

        dx = nucleo["x"] - anterior["x"]
        dy = nucleo["y"] - anterior["y"]

        desplazamiento = math.hypot(dx, dy)

        if desplazamiento >= 5:
            movimiento += 1

        actualizar_track(
            track,
            nucleo,
            radar_ts,
            intervalo,
            asociacion["score"]
        )

        salida = dict(nucleo)

        salida["track_id"] = track["track_id"]

        salida["area_change_pct"] = round(
            area_change,
            2
        )

        salida["tendencia_area"] = tendencia_area

        salida["dx_pixeles"] = round(dx, 3)
        salida["dy_pixeles"] = round(dy, 3)

        salida["desplazamiento_pixeles"] = round(
            desplazamiento,
            3
        )

        salida["velocidad_pixeles_min"] = round(
            track["velocidad_pixeles_min"],
            5
        )

        salida["rumbo_grados"] = track["rumbo_grados"]
        salida["direccion"] = track["direccion"]

        salida_nucleos.append(salida)

    # -----------------------------------------------------
    # NÚCLEOS NO ASOCIADOS = TRACKS NUEVOS
    # -----------------------------------------------------

    nuevos = 0

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

        nuevos += 1

        salida = dict(nucleo)
        salida["track_id"] = tid
        salida["nuevo"] = True

        salida_nucleos.append(salida)

    # -----------------------------------------------------
    # TRACKS ANTERIORES NO ASOCIADOS
    # -----------------------------------------------------

    desaparecidos = 0

    for track in activos_previos:

        tid = track["track_id"]

        if tid in asignados_tracks:
            continue

        track["presente_en_ultimo_frame"] = False
        track["missed_frames"] = (
            entero(track.get("missed_frames")) + 1
        )

        # Se cierra el track como activo.
        # Se conserva entero en históricos.
        track["activo"] = False

        desaparecidos += 1

    # -----------------------------------------------------
    # ACTIVOS = SOLO LOS DEL FRAME ACTUAL
    # -----------------------------------------------------

    activos_actuales = []

    for track in estado["tracks"]:

        if (
            track.get("activo") is True
            and track.get("ultimo_frame") == radar_ts
        ):
            activos_actuales.append(
                track["track_id"]
            )

    estado["ultimo_radar_timestamp"] = radar_ts
    estado["tracks_activos"] = activos_actuales
    estado["ultimo_frame_nucleos"] = salida_nucleos

    # -----------------------------------------------------
    # ASOCIACIONES DE SALIDA
    # -----------------------------------------------------

    asociaciones_salida = []

    for asociacion in asociaciones:

        track = asociacion["track"]

        asociaciones_salida.append({
            "track_id": track["track_id"],
            "nucleo_actual": asociacion["indice"],

            "score": asociacion["score"],
            "distancia_pixeles": asociacion["distancia"],
            "iou": asociacion["iou"],
            "similitud_area": asociacion["similitud_area"],
            "similitud_forma": asociacion["similitud_forma"],

            "dx_pixeles": round(
                track.get("dx_pixeles", 0),
                3
            ),

            "dy_pixeles": round(
                track.get("dy_pixeles", 0),
                3
            ),

            "desplazamiento_pixeles": round(
                track.get(
                    "desplazamiento_pixeles",
                    0
                ),
                3
            ),

            "velocidad_pixeles_min": round(
                track.get(
                    "velocidad_pixeles_min",
                    0
                ),
                5
            ),

            "rumbo_grados": track.get(
                "rumbo_grados"
            ),

            "direccion": track.get(
                "direccion"
            ),

            "aceleracion_pixeles_min2": round(
                track.get(
                    "aceleracion_pixeles_min2",
                    0
                ),
                6
            ),

            "cambio_rumbo_grados": round(
                track.get(
                    "cambio_rumbo_grados",
                    0
                ),
                2
            )
        })

    # -----------------------------------------------------
    # SALIDA FINAL
    # -----------------------------------------------------

    salida = {
        "schema_version": SCHEMA_VERSION,

        "radar_timestamp": radar_ts,

        "frame_anterior_disponible": True,

        "intervalo_minutos": round(
            intervalo,
            2
        ),

        "calidad_intervalo": "VALIDO",

        "nucleos_actuales": len(
            salida_nucleos
        ),

        "nucleos_nuevos": nuevos,

        "nucleos_seguimiento": len(
            asociaciones
        ),

        "nucleos_creciendo": creciendo,

        "nucleos_debilitandose": debilitandose,

        "nucleos_movimiento": movimiento,

        "nucleos_estables": estables,

        "nucleos_desaparecidos": desaparecidos,

        "asociaciones": asociaciones_salida,

        "nucleos": salida_nucleos
    }

    guardar_json(
        ESTADO,
        estado
    )

    guardar_json(
        SALIDA,
        salida
    )

    return salida


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    resultado = procesar()

    print("=" * 60)
    print("SEGUIMIENTO TEMPORAL")
    print("=" * 60)

    print(
        "Radar:",
        resultado["radar_timestamp"]
    )

    print(
        "Intervalo:",
        resultado["intervalo_minutos"],
        "min"
    )

    print(
        "Calidad:",
        resultado["calidad_intervalo"]
    )

    print(
        "Núcleos:",
        resultado["nucleos_actuales"]
    )

    print(
        "Nuevos:",
        resultado["nucleos_nuevos"]
    )

    print(
        "Seguimiento:",
        resultado["nucleos_seguimiento"]
    )

    print(
        "Creciendo:",
        resultado["nucleos_creciendo"]
    )

    print(
        "Debilitándose:",
        resultado["nucleos_debilitandose"]
    )

    print(
        "Movimiento:",
        resultado["nucleos_movimiento"]
    )

    print(
        "Desaparecidos:",
        resultado["nucleos_desaparecidos"]
    )

    print("=" * 60)
