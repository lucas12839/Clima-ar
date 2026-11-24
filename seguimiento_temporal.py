import json
import math
from pathlib import Path
from datetime import datetime, timezone, timedelta

try:
    from scripts.rma10_features import bahia_pixel_position
except ImportError:
    from rma10_features import bahia_pixel_position

BASE = Path("data/radar")
RESULTADO = BASE / "resultado_procesamiento.json"
SEGUIMIENTO = BASE / "seguimiento"
ESTADO = SEGUIMIENTO / "estado.json"
SALIDA = SEGUIMIENTO / "seguimiento_temporal.json"

SCHEMA_VERSION = 8

MIN_INTERVALO = 2.0
MAX_INTERVALO = 45.0
MIN_DESPLAZAMIENTO_MOVIMIENTO = 1.0
MIN_VELOCIDAD_RUMBO = 0.75
MIN_DESPLAZAMIENTO_RUMBO = 3.0
UMBRAL_ACELERACION = 0.10
UMBRAL_CAMBIO_RUMBO = 20.0

MAX_DISTANCIA_BASE = 70.0
MAX_DISTANCIA_SIN_OVERLAP = 85.0
# Límite físico para asociar celdas SIN superposición entre frames:
# 120 km/h (2,13 px/min en la grilla de 512 px) más un margen de 6 px por
# cambio de forma del centroide. Sin esto se unían celdas distintas a
# 50-80 km y salían velocidades de 150-850 km/h.
VELOCIDAD_MAX_PX_MIN = 120.0 / 60.0 / (240.0 / 255.5)
# 24/11/2026: 12 -> 6 px (unos 6 km). Con la velocidad del flujo óptico la
# predicción es mejor y alcanza un margen chico: error a 10 min 4,7 -> 4,5 km.
MARGEN_CENTROIDE_PX = 6.0
MAX_MISSED_FRAMES = 2
MAX_HISTORIAL = 20
MAX_REPLAY_FRAMES = 6


# Velocidad por flujo óptico (24/11/2026). El centro de una tormenta se corre
# cuando cambia de forma (crece de un lado, se desarma del otro) y eso daba
# velocidades imposibles (1 de cada 5 tramos > 80 km/h, máximo 400 km/h).
# El flujo óptico mide cómo se movió el eco entero entre las dos imágenes:
# la velocidad sale del flujo (ver elegir_velocidad).
KM_POR_PIXEL = 240.0 / 255.5
VELOCIDAD_FISICA_MAX_KMH = 120.0
PESO_MEDIDA_NUEVA = 0.4
GIRO_BRUSCO_GRADOS = 90.0


def kmh_a_px_min(kmh):
    return kmh / 60.0 / KM_POR_PIXEL


def px_min_a_kmh(px_min):
    return px_min * 60.0 * KM_POR_PIXEL


def ruta_frame(ts, actual=None):
    """Imagen guardada de un horario (historico) o None."""
    dt = parse_iso(ts)
    if dt is None:
        return None
    nombre = dt.strftime("%Y%m%d%H%M%S") + ".png"
    for carpeta in (BASE / "historico", BASE / "historico_tracking_backup"):
        ruta = carpeta / nombre
        if ruta.exists():
            return ruta
    if actual is not None:
        candidato = Path(actual).parent / nombre
        if candidato.exists():
            return candidato
    return None


def flujo_entre(frame_previo, frame_actual, intervalo):
    """Campo (u, v) en px/min entre dos imágenes, o None si no se puede.

    Usa el mismo flujo óptico del nowcast (dBZ limpios, sin ecos fijos,
    suavizado y acotado a 120 km/h)."""
    if frame_previo is None or frame_actual is None or not intervalo or intervalo <= 0:
        return None
    try:
        import contextlib
        import io
        import sys as _sys
        raiz = Path(__file__).resolve().parent
        for carpeta in (raiz, raiz / "scripts"):
            if str(carpeta) not in _sys.path:
                _sys.path.insert(0, str(carpeta))
        from scripts import nowcast_flujo_optico as nf
        with contextlib.redirect_stdout(io.StringIO()):
            previo = nf.cargar_dbz(Path(frame_previo))
            actual = nf.cargar_dbz(Path(frame_actual))
        if previo is None or actual is None or previo.shape != actual.shape:
            return None
        clutter = nf.cargar_mascara_clutter(actual.shape)
        flujo = nf.estimar_flujo(
            nf.campo_limpio(previo, clutter),
            nf.campo_limpio(actual, clutter),
            float(intervalo),
            nf.km_por_pixel(actual.shape[1]),
        )
        if not flujo.get("valido"):
            return None
        return flujo["u"], flujo["v"]
    except Exception:
        return None


def velocidad_flujo_nucleo(flujo, nucleo):
    """Vector medio (px/min) del flujo dentro del rectángulo del núcleo."""
    if flujo is None:
        return None
    u, v = flujo
    h, w = u.shape
    medio_ancho = max(3.0, numero(nucleo.get("ancho"), 1) / 2.0)
    medio_alto = max(3.0, numero(nucleo.get("alto"), 1) / 2.0)
    x, y = numero(nucleo.get("x")), numero(nucleo.get("y"))
    x0, x1 = max(0, int(x - medio_ancho)), min(w, int(x + medio_ancho) + 1)
    y0, y1 = max(0, int(y - medio_alto)), min(h, int(y + medio_alto) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    return float(u[y0:y1, x0:x1].mean()), float(v[y0:y1, x0:x1].mean())


def elegir_velocidad(vx_centro, vy_centro, flujo_vec, previa=None):
    """Velocidad final (px/min), si el tramo es dudoso y de dónde salió.

    - Con flujo óptico se mide el movimiento con el flujo (el centro se corre
      cuando la tormenta cambia de forma). Sin flujo, con el centro.
    - Si había una velocidad anterior válida se suaviza (40 % la medida nueva,
      60 % la anterior): una tormenta no cambia de velocidad de golpe.
    - Si el centro implica más de 120 km/h el tramo queda dudoso (puede ser
      otra tormenta). Sin flujo, en ese caso se mantiene la velocidad
      anterior (o quieta si no había).

    Medido con 10 días guardados (herramientas/medir_velocidades.py): tramos
    > 80 km/h 21,8 % -> 1,5 %; máximo 403 -> 120 km/h; error de posición a
    10/20/30 min 7,5/14,5/20,8 -> 4,7/8,9/13,2 km.
    """
    rapidez_centro = px_min_a_kmh(math.hypot(vx_centro, vy_centro))
    salto = rapidez_centro > VELOCIDAD_FISICA_MAX_KMH
    if flujo_vec is not None:
        mx, my = flujo_vec
        fuente = "FLUJO"
    elif not salto:
        mx, my = vx_centro, vy_centro
        fuente = "CENTRO"
    elif previa is not None:
        return previa[0], previa[1], True, "ANTERIOR"
    else:
        return 0.0, 0.0, True, "SIN_DATO"
    if previa is not None and math.hypot(previa[0], previa[1]) > 0:
        mx = PESO_MEDIDA_NUEVA * mx + (1.0 - PESO_MEDIDA_NUEVA) * previa[0]
        my = PESO_MEDIDA_NUEVA * my + (1.0 - PESO_MEDIDA_NUEVA) * previa[1]
        fuente += "_SUAVIZADA"
    return mx, my, salto, fuente


def cargar_json(path, default=None):
    if not path.exists():
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def guardar_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


def numero(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def entero(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


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


def parse_iso(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            str(value).replace(
                "Z",
                "+00:00"
            )
        ).astimezone(
            timezone.utc
        )

    except Exception:
        return None


def parse_timestamp_filename(stem):

    """Reconoce todos los formatos históricos usados por ClimaAR."""

    texto = str(stem).strip()

    # Formato actual: YYYYMMDDHHMMSS
    if len(texto) == 14 and texto.isdigit():

        try:
            return datetime.strptime(
                texto,
                "%Y%m%d%H%M%S"
            ).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass

    # Formato legacy: YYYYMMDD_HHMMSS
    if len(texto) == 15 and texto[8] == "_":

        try:
            return datetime.strptime(
                texto,
                "%Y%m%d_%H%M%S"
            ).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass

    # Historial legacy con epoch Unix en segundos.
    if len(texto) == 10 and texto.isdigit():

        try:
            timestamp = int(texto)
            dt = datetime.fromtimestamp(
                timestamp,
                tz=timezone.utc
            )
            if 2000 <= dt.year <= 2100:
                return dt
        except (OverflowError, OSError, ValueError):
            pass

    # Algunos históricos pueden conservar prefijos como radar_.
    for formato in (
        "radar_%Y%m%d%H%M%S",
        "radar_%Y%m%d_%H%M%S",
    ):
        try:
            return datetime.strptime(
                texto,
                formato
            ).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue

    return None


def timestamp_radar():

    candidatos = []

    for carpeta in (
        BASE / "historico",
        BASE / "historico_tracking_backup",
    ):

        if not carpeta.exists():
            continue

        for archivo in carpeta.glob("*.png"):

            dt = parse_timestamp_filename(
                archivo.stem
            )

            if dt is not None:
                candidatos.append(dt)

    if not candidatos:
        raise RuntimeError(
            "No se encontró timestamp de radar."
        )

    return max(candidatos)


def normalizar_nucleo(
    nucleo,
    indice=0
):

    if not isinstance(
        nucleo,
        dict
    ):
        nucleo = {}

    return {

        "indice":
            indice,

        "x":
            numero(
                nucleo.get(
                    "x",
                    nucleo.get(
                        "centro_x",
                        nucleo.get(
                            "centroid_x",
                            0
                        )
                    )
                )
            ),

        "y":
            numero(
                nucleo.get(
                    "y",
                    nucleo.get(
                        "centro_y",
                        nucleo.get(
                            "centroid_y",
                            0
                        )
                    )
                )
            ),

        "pixeles":
            entero(
                nucleo.get(
                    "pixeles",
                    nucleo.get(
                        "pixels",
                        nucleo.get(
                            "area",
                            0
                        )
                    )
                )
            ),

        "ancho":
            max(
                numero(
                    nucleo.get(
                        "ancho",
                        nucleo.get(
                            "width",
                            1
                        )
                    ),
                    1
                ),
                1
            ),

        "alto":
            max(
                numero(
                    nucleo.get(
                        "alto",
                        nucleo.get(
                            "height",
                            1
                        )
                    ),
                    1
                ),
                1
            ),

        "intensidad":
            entero(
                nucleo.get(
                    "intensidad_maxima",
                    nucleo.get(
                        "max_intensity",
                        nucleo.get(
                            "intensidad",
                            0
                        )
                    )
                )
            ),

        "nivel_intensidad":
            nucleo.get(
                "nivel_intensidad",
                nucleo.get(
                    "level",
                    ""
                )
            ),

        "confianza":
            numero(
                nucleo.get(
                    "confianza",
                    nucleo.get(
                        "confidence",
                        0
                    )
                )
            ),

        "estado":
            nucleo.get(
                "estado",
                nucleo.get(
                    "state",
                    ""
                )
            ),
    }


def bbox(nucleo):

    x = nucleo["x"]
    y = nucleo["y"]

    ancho = max(
        nucleo["ancho"],
        1
    )

    alto = max(
        nucleo["alto"],
        1
    )

    return (
        x - ancho / 2,
        y - alto / 2,
        x + ancho / 2,
        y + alto / 2,
    )


def iou(a, b):

    ax1, ay1, ax2, ay2 = bbox(a)
    bx1, by1, bx2, by2 = bbox(b)

    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )

    inter = (
        max(
            0,
            ix2 - ix1
        )
        *
        max(
            0,
            iy2 - iy1
        )
    )

    area_a = (
        max(
            0,
            ax2 - ax1
        )
        *
        max(
            0,
            ay2 - ay1
        )
    )

    area_b = (
        max(
            0,
            bx2 - bx1
        )
        *
        max(
            0,
            by2 - by1
        )
    )

    union = (
        area_a
        + area_b
        - inter
    )

    if union <= 0:
        return 0.0

    return inter / union


def distancia(a, b):

    return math.hypot(
        b["x"] - a["x"],
        b["y"] - a["y"]
    )


def similitud_forma(a, b):

    wa = max(
        a["ancho"],
        1
    )

    wb = max(
        b["ancho"],
        1
    )

    ha = max(
        a["alto"],
        1
    )

    hb = max(
        b["alto"],
        1
    )

    return (
        (
            min(
                wa,
                wb
            )
            /
            max(
                wa,
                wb
            )
        )
        +
        (
            min(
                ha,
                hb
            )
            /
            max(
                ha,
                hb
            )
        )
    ) / 2


def posicion_predicha(
    track,
    intervalo
):

    x = numero(
        track.get(
            "x"
        )
    )

    y = numero(
        track.get(
            "y"
        )
    )

    vx = numero(
        track.get(
            "velocidad_x_pixeles_min",
            0
        )
    )

    vy = numero(
        track.get(
            "velocidad_y_pixeles_min",
            0
        )
    )

    return (
        x + vx * intervalo,
        y + vy * intervalo
    )


def distancia_predicha(
    track,
    actual,
    intervalo
):

    px, py = posicion_predicha(
        track,
        intervalo
    )

    return math.hypot(
        actual["x"] - px,
        actual["y"] - py
    )


def distancia_maxima(
    intervalo,
    track=None
):

    limite = max(
        MAX_DISTANCIA_BASE,
        7.0 * max(
            intervalo,
            1.0
        )
    )

    if track is not None:

        velocidad = math.hypot(
            numero(
                track.get(
                    "velocidad_x_pixeles_min",
                    0
                )
            ),
            numero(
                track.get(
                    "velocidad_y_pixeles_min",
                    0
                )
            )
        )

        limite = max(
            limite,
            velocidad * intervalo * 2.5 + 25.0
        )

    return min(
        limite,
        180.0
    )


def compatibilidad(
    anterior,
    actual,
    intervalo,
    track=None
):

    d = distancia(
        anterior,
        actual
    )

    overlap = iou(
        anterior,
        actual
    )

    area_a = max(
        anterior["pixeles"],
        1
    )

    area_b = max(
        actual["pixeles"],
        1
    )

    area_min = min(
        area_a / area_b,
        area_b / area_a
    )

    forma = similitud_forma(
        anterior,
        actual
    )

    limite = distancia_maxima(
        intervalo,
        track
    )

    if d > limite:
        return None

    # Sin superposición, la identidad depende solo de la distancia: tiene
    # que ser físicamente alcanzable en el intervalo.
    if overlap < 0.02 and d > (
        VELOCIDAD_MAX_PX_MIN * max(intervalo, 1.0) + MARGEN_CENTROIDE_PX
    ):
        return None

    d_pred = None

    if track is not None:

        d_pred = distancia_predicha(
            track,
            actual,
            intervalo
        )

        limite_pred = min(
            180.0,
            max(
                60.0,
                limite * 1.25
            )
        )

        if (
            d_pred > limite_pred
            and
            overlap < 0.02
        ):
            return None

    if overlap < 0.02:

        if (
            d > MAX_DISTANCIA_SIN_OVERLAP
            and
            d_pred is None
        ):
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

    proximidad = max(
        0.0,
        1.0 - d / max(
            limite,
            1.0
        )
    )

    pred_score = 0.0

    if d_pred is not None:

        pred_limite = max(
            60.0,
            limite * 1.25
        )

        pred_score = max(
            0.0,
            1.0 - d_pred / pred_limite
        ) * 35.0

    score = (
        proximidad * 25.0
        +
        overlap * 30.0
        +
        area_min * 15.0
        +
        forma * 10.0
        +
        pred_score
    )

    return {

        "score":
            round(
                score,
                2
            ),

        "distancia":
            round(
                d,
                3
            ),

        "distancia_predicha":
            (
                round(
                    d_pred,
                    3
                )
                if d_pred is not None
                else None
            ),

        "iou":
            round(
                overlap,
                4
            ),

        "similitud_area":
            round(
                area_min,
                4
            ),

        "similitud_forma":
            round(
                forma,
                4
            ),
    }


def rumbo(
    dx,
    dy
):

    if (
        abs(dx) < 0.01
        and
        abs(dy) < 0.01
    ):
        return (
            None,
            None
        )

    grados = (
        math.degrees(
            math.atan2(
                dx,
                -dy
            )
        )
        % 360
    )

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

    indice = int(
        (
            grados + 11.25
        )
        % 360
        / 22.5
    )

    return (
        grados,
        sectores[indice]
    )


def estado_nuevo():

    return {

        "schema_version":
            SCHEMA_VERSION,

        "ultimo_radar_timestamp":
            None,

        "siguiente_track_id":
            1,

        "tracks":
            [],

        "tracks_activos":
            [],

        "ultimo_frame_nucleos":
            [],
    }


def normalizar_estado(
    estado
):

    if not isinstance(
        estado,
        dict
    ):
        return estado_nuevo()

    nuevo = estado_nuevo()

    nuevo[
        "ultimo_radar_timestamp"
    ] = estado.get(
        "ultimo_radar_timestamp"
    )

    tracks = estado.get(
        "tracks",
        []
    )

    if not isinstance(
        tracks,
        list
    ):
        tracks = []

    max_id = 0

    for track in tracks:

        if not isinstance(
            track,
            dict
        ):
            continue

        tid = entero(
            track.get(
                "track_id"
            ),
            0
        )

        if tid <= 0:
            continue

        track[
            "track_id"
        ] = tid

        if not isinstance(
            track.get(
                "historial",
                []
            ),
            list
        ):
            track[
                "historial"
            ] = []

        track.setdefault(
            "velocidad_x_pixeles_min",
            0.0
        )

        track.setdefault(
            "velocidad_y_pixeles_min",
            0.0
        )

        nuevo[
            "tracks"
        ].append(
            track
        )

        max_id = max(
            max_id,
            tid
        )

    siguiente = entero(
        estado.get(
            "siguiente_track_id"
        ),
        max_id + 1
    )

    nuevo[
        "siguiente_track_id"
    ] = max(
        siguiente,
        max_id + 1
    )

    return nuevo


def crear_track(
    track_id,
    nucleo,
    radar_ts
):

    return {

        "track_id":
            track_id,

        "activo":
            True,

        "primer_frame":
            radar_ts,

        "ultimo_frame":
            radar_ts,

        "presente_en_ultimo_frame":
            True,

        "missed_frames":
            0,

        "x":
            nucleo["x"],

        "y":
            nucleo["y"],

        "pixeles":
            nucleo["pixeles"],

        "ancho":
            nucleo["ancho"],

        "alto":
            nucleo["alto"],

        "intensidad":
            nucleo["intensidad"],

        "nivel_intensidad":
            nucleo["nivel_intensidad"],

        "estado":
            nucleo["estado"],

        "confianza":
            nucleo["confianza"],

        "dx_pixeles":
            0.0,

        "dy_pixeles":
            0.0,

        "desplazamiento_pixeles":
            0.0,

        "velocidad_pixeles_min":
            0.0,

        "velocidad_x_pixeles_min":
            0.0,

        "velocidad_y_pixeles_min":
            0.0,

        "rumbo_grados":
            None,

        "direccion":
            None,

        "aceleracion_pixeles_min2":
            0.0,

        "aceleracion_x_pixeles_min2":
            0.0,

        "aceleracion_y_pixeles_min2":
            0.0,

        "tendencia_aceleracion":
            "SIN_COMPARACION",

        "cambio_rumbo_grados":
            0.0,

        "cambio_rumbo_signado_grados":
            0.0,

        "cambio_rumbo_significativo":
            False,

        "tipo_cambio_rumbo":
            "SIN_COMPARACION",

        "rumbo_comparable":
            False,

        "calidad_movimiento":
            "PRIMER_FRAME",

        "cambio_area_porcentaje":
            0.0,

        "tendencia_area":
            "ESTABLE",

        "tendencia_velocidad":
            "ESTABLE",

        "confianza_movimiento":
            0.0,

        "parent_track_id":
            None,

        "origen_evento":
            "NUEVO",

        "fusionado_en_track_id":
            None,

        "evento_ultimo":
            None,

        "historial":
            [
                {
                    "timestamp":
                        radar_ts,

                    "x":
                        nucleo["x"],

                    "y":
                        nucleo["y"],

                    "pixeles":
                        nucleo["pixeles"],

                    "ancho":
                        nucleo["ancho"],

                    "alto":
                        nucleo["alto"],

                    "intensidad":
                        nucleo["intensidad"],
                }
            ],
    }


def nucleo_salida(
    track
):

    keys = [

        "track_id",
        "x",
        "y",
        "pixeles",
        "ancho",
        "alto",
        "intensidad",
        "nivel_intensidad",
        "estado",
        "confianza",
        "dx_pixeles",
        "dy_pixeles",
        "desplazamiento_pixeles",
        "velocidad_pixeles_min",
        "velocidad_x_pixeles_min",
        "velocidad_y_pixeles_min",
        "rumbo_grados",
        "direccion",
        "aceleracion_pixeles_min2",
        "aceleracion_x_pixeles_min2",
        "aceleracion_y_pixeles_min2",
        "tendencia_aceleracion",
        "cambio_rumbo_grados",
        "cambio_rumbo_signado_grados",
        "cambio_rumbo_significativo",
        "tipo_cambio_rumbo",
        "rumbo_comparable",
        "calidad_movimiento",
        "cambio_area_porcentaje",
        "tendencia_area",
        "tendencia_velocidad",
        "confianza_movimiento",
        "parent_track_id",
        "origen_evento",
        "fusionado_en_track_id",
        "evento_ultimo",
        "historial",
    ]

    return {
        key:
            track.get(
                key
            )
        for key in keys
    }


def actualizar_track(
    track,
    nucleo,
    radar_ts,
    intervalo,
    calidad,
    flujo_vec=None,
):

    anterior_x = numero(
        track.get(
            "x"
        )
    )

    anterior_y = numero(
        track.get(
            "y"
        )
    )

    anterior_velocidad = numero(
        track.get(
            "velocidad_pixeles_min",
            0
        )
    )

    anterior_desplazamiento = numero(
        track.get(
            "desplazamiento_pixeles",
            0
        )
    )

    anterior_velocidad_x = numero(
        track.get(
            "velocidad_x_pixeles_min",
            0
        )
    )

    anterior_velocidad_y = numero(
        track.get(
            "velocidad_y_pixeles_min",
            0
        )
    )

    anterior_rumbo = track.get(
        "rumbo_grados"
    )

    historial_previo = track.get(
        "historial",
        []
    )

    if not isinstance(
        historial_previo,
        list
    ):
        historial_previo = []

    movimiento_previo_valido = (
        len(historial_previo) >= 2
    )

    intervalo_valido = (
        MIN_INTERVALO
        <= intervalo
        <= MAX_INTERVALO
    )

    dx = (
        nucleo["x"]
        -
        anterior_x
    )

    dy = (
        nucleo["y"]
        -
        anterior_y
    )

    desplazamiento = math.hypot(
        dx,
        dy
    )

    velocidad_centro = 0.0
    movimiento_dudoso = False
    fuente_velocidad = "SIN_INTERVALO"

    if intervalo_valido:

        velocidad_centro = desplazamiento / intervalo
        velocidad_x, velocidad_y, movimiento_dudoso, fuente_velocidad = elegir_velocidad(
            dx / intervalo,
            dy / intervalo,
            flujo_vec,
            (
                (anterior_velocidad_x, anterior_velocidad_y)
                if movimiento_previo_valido
                else None
            ),
        )
        velocidad = math.hypot(
            velocidad_x,
            velocidad_y
        )

        if movimiento_previo_valido:
            aceleracion = (
                velocidad
                -
                anterior_velocidad
            ) / intervalo

            aceleracion_x = (
                velocidad_x
                -
                anterior_velocidad_x
            ) / intervalo

            aceleracion_y = (
                velocidad_y
                -
                anterior_velocidad_y
            ) / intervalo
        else:
            aceleracion = 0.0
            aceleracion_x = 0.0
            aceleracion_y = 0.0

    else:

        velocidad = 0.0
        velocidad_x = 0.0
        velocidad_y = 0.0
        aceleracion = 0.0
        aceleracion_x = 0.0
        aceleracion_y = 0.0

    # El rumbo sale de la velocidad elegida, no del salto del centro.
    if intervalo_valido and velocidad > 1e-6:
        grados, direccion = rumbo(
            velocidad_x,
            velocidad_y
        )
    else:
        grados, direccion = rumbo(
            dx,
            dy
        )

    rumbo_comparable = (
        movimiento_previo_valido
        and
        anterior_rumbo is not None
        and
        grados is not None
        and
        anterior_velocidad >= MIN_VELOCIDAD_RUMBO
        and
        velocidad >= MIN_VELOCIDAD_RUMBO
        and
        desplazamiento >= MIN_DESPLAZAMIENTO_RUMBO
        and
        anterior_desplazamiento >= MIN_DESPLAZAMIENTO_RUMBO
    )

    cambio_rumbo = 0.0
    cambio_rumbo_signado = 0.0

    if rumbo_comparable:

        cambio_rumbo_signado = (
            grados
            -
            numero(
                anterior_rumbo
            )
            +
            180
        ) % 360 - 180

        cambio_rumbo = abs(
            cambio_rumbo_signado
        )

    cambio_rumbo_significativo = (
        rumbo_comparable
        and
        cambio_rumbo >= UMBRAL_CAMBIO_RUMBO
    )

    if not rumbo_comparable:
        tipo_cambio_rumbo = "SIN_COMPARACION"
    elif cambio_rumbo < UMBRAL_CAMBIO_RUMBO:
        tipo_cambio_rumbo = "ESTABLE"
    elif cambio_rumbo_signado > 0:
        tipo_cambio_rumbo = "GIRO_DERECHA"
    else:
        tipo_cambio_rumbo = "GIRO_IZQUIERDA"

    if not movimiento_previo_valido:
        tendencia_aceleracion = "SIN_COMPARACION"
    elif aceleracion > UMBRAL_ACELERACION:
        tendencia_aceleracion = "ACELERANDO"
    elif aceleracion < -UMBRAL_ACELERACION:
        tendencia_aceleracion = "DESACELERANDO"
    else:
        tendencia_aceleracion = "ESTABLE"

    anterior_area = max(
        numero(
            track.get(
                "pixeles",
                1
            )
        ),
        1
    )

    cambio_area = (
        (
            nucleo["pixeles"]
            -
            anterior_area
        )
        /
        anterior_area
        * 100
    )

    if cambio_area >= 15:

        tendencia_area = (
            "CRECIENDO"
        )

    elif cambio_area <= -15:

        tendencia_area = (
            "DEBILITANDOSE"
        )

    else:

        tendencia_area = (
            "ESTABLE"
        )

    if not movimiento_previo_valido:

        tendencia_velocidad = (
            "SIN_COMPARACION"
        )

    elif velocidad > (
        anterior_velocidad + 0.25
    ):

        tendencia_velocidad = (
            "AUMENTANDO"
        )

    elif velocidad < (
        anterior_velocidad - 0.25
    ):

        tendencia_velocidad = (
            "DISMINUYENDO"
        )

    else:

        tendencia_velocidad = (
            "ESTABLE"
        )

    if desplazamiento < MIN_DESPLAZAMIENTO_MOVIMIENTO:

        confianza_movimiento = 0.35

    elif not movimiento_previo_valido:

        confianza_movimiento = 0.55

    else:

        estabilidad = max(
            0.0,
            1.0 -
            min(
                cambio_rumbo,
                180
            ) / 180
        )

        confianza_movimiento = min(
            0.95,
            0.50 +
            estabilidad * 0.45
        )

    # Un giro de más de 90° entre dos imágenes no es físico para una
    # tormenta: el tramo queda dudoso y baja la confianza del movimiento.
    if rumbo_comparable and cambio_rumbo >= GIRO_BRUSCO_GRADOS:
        movimiento_dudoso = True

    if movimiento_dudoso:
        confianza_movimiento = min(
            confianza_movimiento,
            0.45
        )

    historial = list(
        historial_previo
    )

    historial.append(
        {
            "timestamp":
                radar_ts,

            "x":
                nucleo["x"],

            "y":
                nucleo["y"],

            "pixeles":
                nucleo["pixeles"],

            "ancho":
                nucleo["ancho"],

            "alto":
                nucleo["alto"],

            "intensidad":
                nucleo["intensidad"],
        }
    )

    historial = historial[
        -MAX_HISTORIAL:
    ]

    track.update(
        {

            "activo":
                True,

            "ultimo_frame":
                radar_ts,

            "presente_en_ultimo_frame":
                True,

            "missed_frames":
                0,

            "x":
                nucleo["x"],

            "y":
                nucleo["y"],

            "pixeles":
                nucleo["pixeles"],

            "ancho":
                nucleo["ancho"],

            "alto":
                nucleo["alto"],

            "intensidad":
                nucleo["intensidad"],

            "nivel_intensidad":
                nucleo["nivel_intensidad"],

            "estado":
                nucleo["estado"],

            "confianza":
                nucleo["confianza"],

            "dx_pixeles":
                dx,

            "dy_pixeles":
                dy,

            "desplazamiento_pixeles":
                desplazamiento,

            "velocidad_pixeles_min":
                velocidad,

            "velocidad_x_pixeles_min":
                velocidad_x,

            "velocidad_y_pixeles_min":
                velocidad_y,

            "rumbo_grados":
                grados,

            "direccion":
                direccion,

            "aceleracion_pixeles_min2":
                aceleracion,

            "aceleracion_x_pixeles_min2":
                aceleracion_x,

            "aceleracion_y_pixeles_min2":
                aceleracion_y,

            "tendencia_aceleracion":
                tendencia_aceleracion,

            "cambio_rumbo_grados":
                cambio_rumbo,

            "cambio_rumbo_signado_grados":
                cambio_rumbo_signado,

            "cambio_rumbo_significativo":
                cambio_rumbo_significativo,

            "tipo_cambio_rumbo":
                tipo_cambio_rumbo,

            "rumbo_comparable":
                rumbo_comparable,

            "velocidad_centro_pixeles_min":
                velocidad_centro,

            "movimiento_dudoso":
                movimiento_dudoso,

            "fuente_velocidad":
                fuente_velocidad,

            "calidad_movimiento":
                calidad,

            "cambio_area_porcentaje":
                cambio_area,

            "tendencia_area":
                tendencia_area,

            "tendencia_velocidad":
                tendencia_velocidad,

            "confianza_movimiento":
                confianza_movimiento,

            "historial":
                historial,
        }
    )


def nueva_base_desde_frame(
    estado,
    nucleos,
    radar_ts
):

    for track in estado[
        "tracks"
    ]:

        track[
            "activo"
        ] = False

        track[
            "presente_en_ultimo_frame"
        ] = False

    actuales = []

    for nucleo in nucleos:

        tid = estado[
            "siguiente_track_id"
        ]

        estado[
            "siguiente_track_id"
        ] += 1

        track = crear_track(
            tid,
            nucleo,
            radar_ts
        )

        estado[
            "tracks"
        ].append(
            track
        )

        actuales.append(
            track
        )

    estado[
        "tracks_activos"
    ] = [
        track[
            "track_id"
        ]
        for track in actuales
    ]

    estado[
        "ultimo_frame_nucleos"
    ] = [
        nucleo_salida(
            track
        )
        for track in actuales
    ]

    estado[
        "ultimo_radar_timestamp"
    ] = radar_ts

    return actuales


def salida_sin_nucleos(
    radar_ts,
    intervalo,
    calidad,
    motivo=None
):

    salida = {

        "schema_version":
            SCHEMA_VERSION,

        "radar_timestamp":
            radar_ts,

        "frame_anterior_disponible":
            intervalo is not None,

        "intervalo_minutos":
            (
                round(
                    intervalo,
                    2
                )
                if intervalo is not None
                else None
            ),

        "calidad_intervalo":
            calidad,

        "nucleos_actuales":
            0,

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

        "eventos_split":
            [],

        "eventos_merge":
            [],

        "cantidad_split":
            0,

        "cantidad_merge":
            0,

        "asociaciones":
            [],

        "nucleos":
            [],
    }

    if motivo:

        salida[
            "motivo"
        ] = motivo

    return salida



# ============================================================
# DIVISIÓN Y UNIÓN: MOVIMIENTO Y TAMAÑO DEL SISTEMA COMPLETO
# ============================================================
# Cuando una tormenta se divide o dos se unen, el centro de cada pedazo salta
# aunque el sistema no se haya movido así, y el tamaño de cada pedazo cambia
# aunque el sistema no haya crecido ni se haya debilitado. Se corrige
# conservando el "centro de masa" (centro ponderado por superficie) del
# sistema completo: el pasado de cada pedazo se corre al lugar que le toca
# dentro del sistema, y su tamaño pasado se reparte en proporción. Así la
# velocidad, el rumbo y el cambio de tamaño de cada pedazo son los del sistema.
# Solo se aplica si las superficies cierran (los pedazos suman lo que había) y
# si el centro conjunto explica mejor el movimiento que el pedazo solo.
MIN_CIERRE_SUPERFICIE = 0.6
MAX_CIERRE_SUPERFICIE = 1.5
# Qué parte de cada pedazo tiene que caer dentro de lo que había antes
# (llevado con su velocidad): un pedazo de afuera es una tormenta nueva.
MIN_CONTENIDO = 0.5


def _contenido(pieza, contenedor):
    """Parte del rectángulo de la pieza que cae dentro del contenedor (0-1)."""
    ax1, ay1, ax2, ay2 = bbox(pieza)
    bx1, by1, bx2, by2 = bbox(contenedor)
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area = max(1e-9, (ax2 - ax1) * (ay2 - ay1))
    return inter / area


def _llevado(track, intervalo):
    """Rectángulo del track llevado con su velocidad al momento actual."""
    x, y = posicion_predicha(track, intervalo)
    return {"x": x, "y": y, "ancho": numero(track.get("ancho", 1), 1),
            "alto": numero(track.get("alto", 1), 1)}


def _cierra(area_nueva, area_previa):
    return MIN_CIERRE_SUPERFICIE * area_previa <= area_nueva <= MAX_CIERRE_SUPERFICIE * area_previa


def _centro_ponderado(partes):
    total = sum(max(0.0, area) for _x, _y, area in partes)
    if total <= 0:
        return None
    return (
        sum(x * max(0.0, area) for x, _y, area in partes) / total,
        sum(y * max(0.0, area) for _x, y, area in partes) / total,
        total,
    )


def _reanclar(track, dx, dy, factor):
    """Corre el pasado del track (dx, dy) y escala su superficie pasada."""
    track["x"] = numero(track.get("x")) + dx
    track["y"] = numero(track.get("y")) + dy
    track["pixeles"] = numero(track.get("pixeles"), 1) * factor
    historial = track.get("historial")
    if isinstance(historial, list):
        nuevo = []
        for punto in historial:
            if not isinstance(punto, dict):
                continue
            punto = dict(punto)
            if punto.get("x") is not None and punto.get("y") is not None:
                punto["x"] = numero(punto["x"]) + dx
                punto["y"] = numero(punto["y"]) + dy
            if punto.get("pixeles") is not None:
                punto["pixeles"] = numero(punto["pixeles"]) * factor
            nuevo.append(punto)
        track["historial"] = nuevo


def ajustar_union(principal, fusionados, nucleo, intervalo):
    """Une el pasado de los tracks fusionados al del principal.

    Devuelve True si se aplicó. El principal pasa a representar al sistema
    unido: su posición pasada es el centro ponderado de todos y su superficie
    pasada la suma de todas.
    """
    t_principal = parse_iso(principal.get("ultimo_frame"))
    partes = [(numero(principal.get("x")), numero(principal.get("y")),
               numero(principal.get("pixeles")))]
    for track in fusionados:
        x, y = numero(track.get("x")), numero(track.get("y"))
        t_track = parse_iso(track.get("ultimo_frame"))
        if t_principal is not None and t_track is not None and t_track != t_principal:
            # Lo que se vio antes se lleva al mismo momento con su propia velocidad.
            x, y = posicion_predicha(track, (t_principal - t_track).total_seconds() / 60.0)
        partes.append((x, y, numero(track.get("pixeles"))))
    conjunto = _centro_ponderado(partes)
    if conjunto is None or partes[0][2] <= 0:
        return False
    cx, cy, area_previa = conjunto
    if not _cierra(numero(nucleo.get("pixeles")), area_previa):
        return False  # lo que quedó no se parece a la suma: no es una unión limpia
    # Cada tormenta que se une tiene que caer sobre el núcleo nuevo.
    for track in [principal, *fusionados]:
        t_track = parse_iso(track.get("ultimo_frame"))
        lapso = intervalo + (
            (t_principal - t_track).total_seconds() / 60.0
            if t_principal is not None and t_track is not None else 0.0)
        if _contenido(_llevado(track, lapso), nucleo) < MIN_CONTENIDO:
            return False
    # El centro conjunto (movido con la velocidad del principal) tiene que
    # explicar mejor dónde quedó el núcleo que el principal solo.
    vx = numero(principal.get("velocidad_x_pixeles_min", 0))
    vy = numero(principal.get("velocidad_y_pixeles_min", 0))
    px, py = partes[0][0] + vx * intervalo, partes[0][1] + vy * intervalo
    if math.hypot(nucleo["x"] - (cx + vx * intervalo), nucleo["y"] - (cy + vy * intervalo)) \
            > math.hypot(nucleo["x"] - px, nucleo["y"] - py):
        return False
    _reanclar(principal, cx - partes[0][0], cy - partes[0][1], area_previa / partes[0][2])
    return True


def ajustar_division(padre, nucleo_padre, nucleo_hijo, intervalo):
    """Prepara al padre para una división.

    Devuelve (dx_hijo, dy_hijo, factor_hijo, pasado_original) para armar al
    hijo, o None si la división no es limpia. El padre se corre para que su
    pasado represente su parte del sistema.
    """
    area_a = numero(nucleo_padre.get("pixeles"))
    area_b = numero(nucleo_hijo.get("pixeles"))
    area_previa = numero(padre.get("pixeles"))
    conjunto = _centro_ponderado([
        (nucleo_padre["x"], nucleo_padre["y"], area_a),
        (nucleo_hijo["x"], nucleo_hijo["y"], area_b),
    ])
    if conjunto is None or area_previa <= 0:
        return None
    cx, cy, area_nueva = conjunto
    if not _cierra(area_nueva, area_previa):
        return None  # los pedazos no suman lo que había: no es una división limpia
    antes = _llevado(padre, intervalo)
    if min(_contenido(nucleo_padre, antes), _contenido(nucleo_hijo, antes)) < MIN_CONTENIDO:
        return None  # un pedazo cae afuera de la tormenta de antes: es otra tormenta
    px, py = posicion_predicha(padre, intervalo)
    if math.hypot(cx - px, cy - py) > math.hypot(nucleo_padre["x"] - px, nucleo_padre["y"] - py):
        return None  # el pedazo grande solo explica mejor el movimiento
    pasado = {
        "x": numero(padre.get("x")),
        "y": numero(padre.get("y")),
        "pixeles": area_previa,
        "historial": [dict(p) for p in padre.get("historial") or [] if isinstance(p, dict)],
    }
    _reanclar(padre, nucleo_padre["x"] - cx, nucleo_padre["y"] - cy, area_a / area_nueva)
    return (nucleo_hijo["x"] - cx, nucleo_hijo["y"] - cy, area_b / area_nueva, pasado)


def heredar_division(hijo, padre, ajuste):
    """El hijo hereda el pasado del sistema (en su lugar) y el movimiento del padre."""
    dx, dy, factor, pasado = ajuste
    previo = {
        "x": pasado["x"],
        "y": pasado["y"],
        "pixeles": pasado["pixeles"],
        "historial": pasado["historial"],
    }
    _reanclar(previo, dx, dy, factor)
    actual = hijo["historial"][-1]
    hijo["historial"] = (previo["historial"] + [actual])[-MAX_HISTORIAL:]
    for campo in (
        "dx_pixeles", "dy_pixeles", "desplazamiento_pixeles",
        "velocidad_pixeles_min", "velocidad_x_pixeles_min", "velocidad_y_pixeles_min",
        "rumbo_grados", "direccion",
        "aceleracion_pixeles_min2", "aceleracion_x_pixeles_min2", "aceleracion_y_pixeles_min2",
        "tendencia_aceleracion", "cambio_rumbo_grados", "cambio_rumbo_signado_grados",
        "cambio_rumbo_significativo", "tipo_cambio_rumbo", "rumbo_comparable",
        "calidad_movimiento", "cambio_area_porcentaje", "tendencia_area",
        "tendencia_velocidad", "confianza_movimiento",
    ):
        if campo in padre:
            hijo[campo] = padre[campo]


def analizar_split_merge(
    candidatos,
    asociaciones,
    asignados_tracks,
    asignados_nucleos,
    nucleos
):
    """Detecta divisiones y fusiones sin reutilizar tracks en un mismo frame.

    Split: un track previo tiene compatibilidad fuerte con más de un núcleo
    actual. El mejor conserva el track original y un núcleo secundario puede
    recibir un track hijo. Cada núcleo actual solo puede ser hijo de un padre.

    Merge: varios tracks previos pueden converger sobre el mismo núcleo actual.
    Se conserva la asociación principal y cada track secundario se cierra como
    fusionado. Un track secundario solo puede participar en UN merge por frame,
    y se elige globalmente su mejor destino para evitar fusiones duplicadas.
    """

    por_track = {}
    por_nucleo = {}

    for candidato in candidatos:
        tid = candidato["track"]["track_id"]
        indice = candidato["indice"]
        por_track.setdefault(tid, []).append(candidato)
        por_nucleo.setdefault(indice, []).append(candidato)

    for lista in por_track.values():
        lista.sort(key=lambda item: item["score"], reverse=True)

    for lista in por_nucleo.values():
        lista.sort(key=lambda item: item["score"], reverse=True)

    eventos_split = []
    eventos_merge = []
    nucleos_split = []
    tracks_merge = set()

    # SPLIT: el track original ya debe tener una asociación principal.
    for asociacion in asociaciones:
        tid = asociacion["track"]["track_id"]
        indice_principal = asociacion["indice"]
        score_principal = asociacion["score"]

        candidatos_split = []

        for candidato in por_track.get(tid, []):
            indice = candidato["indice"]

            if indice == indice_principal:
                continue

            if indice in asignados_nucleos:
                continue

            if candidato["score"] < max(
                45.0,
                score_principal * 0.55
            ):
                continue

            if candidato["distancia"] > 100.0:
                continue

            if (
                candidato["iou"] < 0.01
                and
                candidato["similitud_area"] < 0.35
                and
                candidato["similitud_forma"] < 0.45
            ):
                continue

            candidatos_split.append(candidato)

        if not candidatos_split:
            continue

        elegido = max(
            candidatos_split,
            key=lambda item: item["score"]
        )

        indice_hijo = elegido["indice"]
        if indice_hijo in nucleos_split:
            continue

        nucleos_split.append(indice_hijo)

        eventos_split.append(
            {
                "track_padre": tid,
                "indice_nucleo_padre": indice_principal,
                "indice_nucleo_hijo": indice_hijo,
                "score": round(elegido["score"], 2),
            }
        )

    # MERGE: primero construimos TODAS las opciones válidas.
    # Luego las ordenamos globalmente por score. Así un track secundario
    # solamente puede ser consumido una vez y siempre conserva su mejor
    # oportunidad de fusión.
    opciones_merge = []

    principales_por_nucleo = {
        asociacion["indice"]: asociacion
        for asociacion in asociaciones
    }

    for indice, lista in por_nucleo.items():
        principal = principales_por_nucleo.get(indice)

        if principal is None:
            continue

        tid_principal = principal["track"]["track_id"]
        score_principal = principal["score"]

        for candidato in lista:
            tid_secundario = candidato["track"]["track_id"]

            if tid_secundario == tid_principal:
                continue

            # El track ya tiene asociación principal en este frame.
            if tid_secundario in asignados_tracks:
                continue

            if candidato["score"] < max(
                45.0,
                score_principal * 0.50
            ):
                continue

            if candidato["distancia"] > 100.0:
                continue

            if (
                candidato["iou"] < 0.02
                and
                candidato["similitud_area"] < 0.35
                and
                candidato["similitud_forma"] < 0.45
            ):
                continue

            opciones_merge.append(
                {
                    "track_principal": tid_principal,
                    "track_fusionado": tid_secundario,
                    "indice_nucleo": indice,
                    "score": candidato["score"],
                }
            )

    opciones_merge.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    for opcion in opciones_merge:
        tid_secundario = opcion["track_fusionado"]

        # Un track previo no puede fusionarse con dos núcleos del mismo frame.
        if tid_secundario in tracks_merge:
            continue

        tracks_merge.add(tid_secundario)

        eventos_merge.append(
            {
                "track_principal": opcion["track_principal"],
                "track_fusionado": tid_secundario,
                "indice_nucleo": opcion["indice_nucleo"],
                "score": round(opcion["score"], 2),
            }
        )

    return (
        eventos_split,
        eventos_merge,
        nucleos_split,
        tracks_merge,
    )

def seleccionar_track_principal(tracks, radar_ts=None):
    """Selecciona el candidato principal sin usar intensidad como criterio dominante.

    La selección es operativa: continuidad, movimiento confiable, aproximación
    al dominio de Bahía Blanca (coordenadas de píxel respecto al centro del
    radar) y tamaño estructural. La intensidad aporta señal, pero no decide
    por sí sola cuál es el núcleo principal.
    """

    if not tracks:
        return None

    candidatos = []

    # Geometría RMA10 compartida: radio operativo 240 km y coordenadas
    # reales de Bahía Blanca respecto del centro del radar.
    centro_x = 255.5
    centro_y = 255.5
    bahia_x, bahia_y = bahia_pixel_position(512, 512)

    for track in tracks:
        if not isinstance(track, dict):
            continue
        if track.get("activo") is not True:
            continue
        if track.get("presente_en_ultimo_frame") is not True:
            continue

        x = numero(track.get("x"), centro_x)
        y = numero(track.get("y"), centro_y)

        distancia_bahia_px = math.hypot(
            x - bahia_x,
            y - bahia_y
        )

        distancia_centro_px = math.hypot(
            x - centro_x,
            y - centro_y
        )

        velocidad = max(
            0.0,
            numero(
                track.get("velocidad_pixeles_min"),
                0.0
            )
        )

        confianza_movimiento = max(
            0.0,
            min(
                1.0,
                numero(
                    track.get("confianza_movimiento"),
                    0.0
                )
            )
        )

        desplazamiento = max(
            0.0,
            numero(
                track.get("desplazamiento_pixeles"),
                0.0
            )
        )

        area = max(
            1.0,
            numero(
                track.get("pixeles"),
                1.0
            )
        )

        intensidad = max(
            0.0,
            numero(
                track.get("intensidad"),
                0.0
            )
        )

        # Continuidad: un track con varios frames y movimiento consistente
        # tiene prioridad sobre un núcleo aislado.
        historial = track.get("historial", [])
        persistencia = min(
            1.0,
            max(
                0,
                len(historial) - 1
            ) / 8.0
        )

        movimiento = min(
            1.0,
            velocidad / 3.0
        ) * 0.55 + confianza_movimiento * 0.45

        estructura = min(
            1.0,
            math.log1p(area) / math.log1p(500.0)
        )

        # Preferimos candidatos que ya se encuentren en la mitad exterior
        # hacia la zona de Bahía, pero sin convertir distancia en el criterio
        # absoluto: el cálculo definitivo se hace después con georreferencia.
        aproximacion_bahia = max(
            0.0,
            1.0 - distancia_bahia_px / 360.0
        )

        # Una pequeña contribución de intensidad evita ignorar una estructura
        # realmente intensa, pero no puede dominar la selección.
        senal_intensidad = min(
            1.0,
            intensidad / 6.0
        )

        score = (
            persistencia * 0.30
            + movimiento * 0.25
            + aproximacion_bahia * 0.20
            + estructura * 0.15
            + min(
                1.0,
                desplazamiento / 8.0
            ) * 0.07
            + senal_intensidad * 0.03
        )

        candidatos.append(
            (
                score,
                track
            )
        )

    if not candidatos:
        return None

    candidatos.sort(
        key=lambda item: (
            item[0],
            numero(
                item[1].get("confianza_movimiento"),
                0.0
            ),
            numero(
                item[1].get("pixeles"),
                0.0
            )
        ),
        reverse=True
    )

    return candidatos[0][1]


def marcar_track_perdido(track):
    """Avanza un track no observado un frame sin alterar su última posición."""

    track["presente_en_ultimo_frame"] = False
    track["missed_frames"] = (
        entero(
            track.get("missed_frames", 0)
        ) + 1
    )

    if track["missed_frames"] > MAX_MISSED_FRAMES:
        track["activo"] = False
        track["evento_ultimo"] = "DESAPARECIDO"
        return False

    track["activo"] = True
    track["evento_ultimo"] = "FRAME_PERDIDO"
    return True


def procesar_frame_secuencia(
    estado,
    nucleos,
    radar_ts,
    anterior_ts,
    frame_actual=None,
):

    """Procesa un frame RMA10 manteniendo recuperación de tracks perdidos.

    Regla clave de Block 2:
    - un track activo puede sobrevivir MAX_MISSED_FRAMES frames sin detección;
    - su última posición sigue siendo la referencia de asociación;
    - al reaparecer, se calcula el intervalo desde su último frame observado,
      no desde el frame inmediatamente anterior global;
    - al recuperarse, actualizar_track() reinicia missed_frames a cero;
    - cada frame perdido se contabiliza exactamente una vez.
    """

    if not nucleos:

        activos = []
        desaparecidos = 0
        recuperables = 0

        for track in estado["tracks"]:
            if track.get("activo") is not True:
                continue

            ultimo_track_ts = track.get("ultimo_frame")

            # No contamos dos veces el mismo frame si el pipeline vuelve a
            # ejecutar el mismo timestamp.
            ultimo_track_dt = parse_iso(
                ultimo_track_ts
            )
            radar_dt = parse_iso(
                radar_ts
            )

            if (
                ultimo_track_dt is not None
                and radar_dt is not None
                and ultimo_track_dt >= radar_dt
            ):
                continue

            sobrevivio = marcar_track_perdido(track)

            if sobrevivio:
                activos.append(track["track_id"])
                recuperables += 1
            else:
                desaparecidos += 1

        estado["tracks_activos"] = activos
        estado["ultimo_frame_nucleos"] = []
        estado["ultimo_radar_timestamp"] = radar_ts

        principal = seleccionar_track_principal(
            [
                track
                for track in estado["tracks"]
                if (
                    track.get("activo") is True
                    and track.get("presente_en_ultimo_frame") is True
                )
            ],
            radar_ts
        )

        salida = salida_sin_nucleos(
            radar_ts,
            (
                None
                if anterior_ts is None
                else 0
            ),
            "FRAME_SIN_NUCLEOS",
            "No se detectaron núcleos no-estáticos en este frame."
        )
        salida["nucleos_desaparecidos"] = desaparecidos
        salida["tracks_recuperables"] = recuperables
        salida["track_principal_candidato_id"] = (
            principal.get("track_id")
            if principal
            else None
        )
        return salida

    if anterior_ts is None:

        actuales = nueva_base_desde_frame(
            estado,
            nucleos,
            radar_ts
        )

        principal = seleccionar_track_principal(
            actuales,
            radar_ts
        )

        resultado = {
            "schema_version": SCHEMA_VERSION,
            "radar_timestamp": radar_ts,
            "frame_anterior_disponible": False,
            "intervalo_minutos": None,
            "calidad_intervalo": "PRIMER_FRAME",
            "nucleos_actuales": len(actuales),
            "nucleos_nuevos": len(actuales),
            "nucleos_seguimiento": 0,
            "nucleos_recuperados": 0,
            "tracks_recuperados": 0,
            "nucleos_creciendo": 0,
            "nucleos_debilitandose": 0,
            "nucleos_movimiento": 0,
            "nucleos_estables": 0,
            "nucleos_desaparecidos": 0,
            "eventos_split": [],
            "eventos_merge": [],
            "cantidad_split": 0,
            "cantidad_merge": 0,
            "asociaciones": [],
            "track_principal_candidato_id": (
                principal.get("track_id")
                if principal
                else None
            ),
            "nucleos": [
                nucleo_salida(track)
                for track in actuales
            ],
        }
        return resultado

    anterior_dt = parse_iso(anterior_ts)
    actual_dt = parse_iso(radar_ts)

    if anterior_dt is None or actual_dt is None:

        actuales = nueva_base_desde_frame(
            estado,
            nucleos,
            radar_ts
        )

        salida = salida_sin_nucleos(
            radar_ts,
            None,
            "REINICIO_TIMESTAMP_INVALIDO"
        )
        salida["nucleos_actuales"] = len(actuales)
        salida["nucleos_nuevos"] = len(actuales)
        salida["nucleos"] = [
            nucleo_salida(track)
            for track in actuales
        ]
        principal = seleccionar_track_principal(
            actuales,
            radar_ts
        )
        salida["track_principal_candidato_id"] = (
            principal.get("track_id")
            if principal
            else None
        )
        return salida

    intervalo = (
        actual_dt - anterior_dt
    ).total_seconds() / 60.0

    if intervalo <= 0 or intervalo > MAX_INTERVALO:

        actuales = nueva_base_desde_frame(
            estado,
            nucleos,
            radar_ts
        )

        calidad = (
            "FRAME_REPETIDO_O_ATRASADO"
            if intervalo <= 0
            else
            "GAP_MAYOR_45_MIN"
        )

        principal = seleccionar_track_principal(
            actuales,
            radar_ts
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
            "nucleos_recuperados": 0,
            "tracks_recuperados": 0,
            "nucleos_creciendo": 0,
            "nucleos_debilitandose": 0,
            "nucleos_movimiento": 0,
            "nucleos_estables": 0,
            "nucleos_desaparecidos": 0,
            "asociaciones": [],
            "track_principal_candidato_id": (
                principal.get("track_id")
                if principal
                else None
            ),
            "nucleos": [
                nucleo_salida(track)
                for track in actuales
            ],
        }

    # Block 2: cualquier track activo cuyo último frame observado sea anterior
    # al frame global puede intentar recuperarse, siempre que no haya superado
    # MAX_MISSED_FRAMES. No exigimos ultimo_frame == anterior_ts.
    activos_previos = []
    for track in estado["tracks"]:
        if track.get("activo") is not True:
            continue

        missed = entero(
            track.get("missed_frames", 0)
        )
        if missed > MAX_MISSED_FRAMES:
            track["activo"] = False
            continue

        track_last_dt = parse_iso(
            track.get("ultimo_frame")
        )
        if track_last_dt is None:
            continue

        if track_last_dt >= actual_dt:
            continue

        # El intervalo debe ser el tiempo real desde la última observación de
        # este track. Esto es lo que permite recuperar un track tras 1-2 frames
        # perdidos sin calcular una velocidad artificialmente alta.
        intervalo_track = (
            actual_dt - track_last_dt
        ).total_seconds() / 60.0

        if (
            intervalo_track < MIN_INTERVALO
            or intervalo_track > MAX_INTERVALO
        ):
            continue

        track["intervalo_desde_ultima_observacion_min"] = round(
            intervalo_track,
            3
        )
        activos_previos.append(
            (
                track,
                intervalo_track
            )
        )

    candidatos = []

    for track, intervalo_track in activos_previos:

        anterior = {
            "x": numero(track.get("x")),
            "y": numero(track.get("y")),
            "pixeles": entero(track.get("pixeles")),
            "ancho": numero(track.get("ancho", 1), 1),
            "alto": numero(track.get("alto", 1), 1),
        }

        for indice, actual in enumerate(nucleos):
            comp = compatibilidad(
                anterior,
                actual,
                intervalo_track,
                track
            )

            if comp is None:
                continue

            candidatos.append({
                "track": track,
                "indice": indice,
                "intervalo_track": intervalo_track,
                "recuperacion": (
                    entero(track.get("missed_frames", 0)) > 0
                ),
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

    eventos_split = []
    eventos_merge = []
    nucleos_split = []
    tracks_merge = set()

    # analizar_split_merge necesita solamente la estructura de candidatos y
    # asociaciones; la lógica histórica se conserva sin cambios.
    (
        eventos_split,
        eventos_merge,
        nucleos_split,
        tracks_merge,
    ) = analizar_split_merge(
        candidatos,
        asociaciones,
        asignados_tracks,
        asignados_nucleos,
        nucleos
    )

    # Las fusiones cierran los tracks secundarios.
    for track, _intervalo_track in activos_previos:
        tid = track["track_id"]

        if tid not in tracks_merge:
            continue

        track["activo"] = False
        track["presente_en_ultimo_frame"] = False
        track["fusionado_en_track_id"] = next(
            (
                evento["track_principal"]
                for evento in eventos_merge
                if evento["track_fusionado"] == tid
            ),
            None
        )
        track["evento_ultimo"] = "MERGE"
        asignados_tracks.add(tid)

    # El principal de cada unión pasa a representar al sistema unido.
    por_id_previos = {
        track["track_id"]: track
        for track, _intervalo_track in activos_previos
    }
    asociacion_por_track = {
        asociacion["track"]["track_id"]: asociacion
        for asociacion in asociaciones
    }
    fusionados_por_principal = {}
    for evento in eventos_merge:
        fusionados_por_principal.setdefault(
            evento["track_principal"], []
        ).append(por_id_previos.get(evento["track_fusionado"]))
    for tid_principal, fusionados in fusionados_por_principal.items():
        asociacion = asociacion_por_track.get(tid_principal)
        fusionados = [track for track in fusionados if track is not None]
        if asociacion is None or not fusionados:
            continue
        ajustar_union(
            asociacion["track"],
            fusionados,
            nucleos[asociacion["indice"]],
            asociacion.get("intervalo_track", intervalo),
        )

    # Las divisiones se preparan antes de actualizar al padre.
    ajustes_division = {}
    for evento in eventos_split:
        asociacion = asociacion_por_track.get(evento["track_padre"])
        if asociacion is None or evento["indice_nucleo_hijo"] in asignados_nucleos:
            continue
        ajuste = ajustar_division(
            asociacion["track"],
            nucleos[asociacion["indice"]],
            nucleos[evento["indice_nucleo_hijo"]],
            asociacion.get("intervalo_track", intervalo),
        )
        if ajuste is not None:
            ajustes_division[evento["indice_nucleo_hijo"]] = ajuste

    salida_nucleos = []
    creciendo = 0
    debilitandose = 0
    movimiento = 0
    estables = 0
    nuevos = 0
    desaparecidos = 0
    recuperados = 0

    calidad = (
        "BUENA"
        if intervalo < 5
        else
        "ACEPTABLE"
        if intervalo <= 15
        else
        "BAJA"
    )

    # Flujo óptico del par de imágenes (una vez por frame). El flujo está
    # indexado en la imagen anterior: se mide donde estaba cada tormenta.
    flujo = None
    if frame_actual is not None and asociaciones:
        flujo = flujo_entre(
            ruta_frame(anterior_ts, frame_actual),
            frame_actual,
            intervalo,
        )
    previo_por_track = {
        asociacion["track"]["track_id"]: {
            "x": numero(asociacion["track"].get("x")),
            "y": numero(asociacion["track"].get("y")),
            "ancho": numero(asociacion["track"].get("ancho", 1), 1),
            "alto": numero(asociacion["track"].get("alto", 1), 1),
        }
        for asociacion in asociaciones
    }

    for asociacion in asociaciones:

        track = asociacion["track"]
        nucleo = nucleos[asociacion["indice"]]
        era_recuperacion = bool(
            asociacion.get("recuperacion")
        )

        actualizar_track(
            track,
            nucleo,
            radar_ts,
            asociacion.get(
                "intervalo_track",
                intervalo
            ),
            calidad,
            velocidad_flujo_nucleo(
                flujo,
                previo_por_track.get(track["track_id"], nucleo)
            ),
        )

        if era_recuperacion:
            track["origen_evento"] = "RECUPERACION"
            track["evento_ultimo"] = "RECUPERADO_DESPUES_DE_FRAME_PERDIDO"
            recuperados += 1

        cambio = numero(
            track.get("cambio_area_porcentaje", 0)
        )

        if cambio >= 15:
            creciendo += 1
        elif cambio <= -15:
            debilitandose += 1
        else:
            estables += 1

        if numero(
            track.get("desplazamiento_pixeles", 0)
        ) >= 1:
            movimiento += 1

        salida_nucleos.append(
            nucleo_salida(track)
        )

    # Los núcleos secundarios de un split reciben un track hijo propio.
    for evento in eventos_split:
        indice_hijo = evento["indice_nucleo_hijo"]

        if indice_hijo in asignados_nucleos:
            continue

        padre = next(
            (
                track
                for track, _intervalo_track in activos_previos
                if track["track_id"] == evento["track_padre"]
            ),
            None
        )

        if padre is None:
            continue

        tid = estado["siguiente_track_id"]
        estado["siguiente_track_id"] += 1

        hijo = crear_track(
            tid,
            nucleos[indice_hijo],
            radar_ts
        )
        hijo["parent_track_id"] = padre["track_id"]
        hijo["origen_evento"] = "SPLIT"
        hijo["evento_ultimo"] = "SPLIT"
        if indice_hijo in ajustes_division:
            heredar_division(hijo, padre, ajustes_division[indice_hijo])

        estado["tracks"].append(hijo)
        asignados_nucleos.add(indice_hijo)
        asignados_tracks.add(tid)
        salida_nucleos.append(nucleo_salida(hijo))
        nuevos += 1

    # Todos los tracks activos que no fueron asociados pierden un frame.
    # Esto incluye tracks que ya habían perdido uno anteriormente.
    for track in estado["tracks"]:
        tid = track["track_id"]

        if track.get("activo") is not True:
            continue

        if tid in asignados_tracks:
            continue

        track_last_dt = parse_iso(
            track.get("ultimo_frame")
        )

        # Un track fusionado ya fue cerrado arriba.
        if track.get("evento_ultimo") == "MERGE":
            continue

        if (
            track_last_dt is not None
            and track_last_dt >= actual_dt
        ):
            continue

        if not marcar_track_perdido(track):
            desaparecidos += 1

    # Todo núcleo actual sin asociación crea un track nuevo.
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
        salida_nucleos.append(nucleo_salida(track))
        nuevos += 1

    # Sólo tracks realmente presentes en el frame actual son tracks activos
    # para los consumidores downstream. Los recuperables permanecen en estado
    # interno, pero no se publican como núcleos actuales.
    estado["tracks_activos"] = [
        track["track_id"]
        for track in estado["tracks"]
        if (
            track.get("activo") is True
            and track.get("presente_en_ultimo_frame") is True
        )
    ]

    estado["ultimo_frame_nucleos"] = salida_nucleos
    estado["ultimo_radar_timestamp"] = radar_ts

    tracks_presentes = [
        track
        for track in estado["tracks"]
        if (
            track.get("activo") is True
            and track.get("presente_en_ultimo_frame") is True
        )
    ]

    principal = seleccionar_track_principal(
        tracks_presentes,
        radar_ts
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "radar_timestamp": radar_ts,
        "frame_anterior_disponible": True,
        "intervalo_minutos": round(intervalo, 2),
        "calidad_intervalo": calidad,
        "nucleos_actuales": len(nucleos),
        "nucleos_nuevos": nuevos,
        "nucleos_seguimiento": len(asociaciones),
        "nucleos_recuperados": recuperados,
        "tracks_recuperados": recuperados,
        "nucleos_creciendo": creciendo,
        "nucleos_debilitandose": debilitandose,
        "nucleos_movimiento": movimiento,
        "nucleos_estables": estables,
        "nucleos_desaparecidos": desaparecidos,
        "eventos_split": eventos_split,
        "eventos_merge": eventos_merge,
        "cantidad_split": len(eventos_split),
        "cantidad_merge": len(eventos_merge),
        "asociaciones": [
            {
                "track_id": asociacion["track"]["track_id"],
                "indice_nucleo": asociacion["indice"],
                "score": asociacion["score"],
                "distancia": asociacion["distancia"],
                "distancia_predicha": asociacion.get("distancia_predicha"),
                "intervalo_track_minutos": round(
                    numero(
                        asociacion.get(
                            "intervalo_track",
                            intervalo
                        )
                    ),
                    3
                ),
                "recuperacion": bool(
                    asociacion.get("recuperacion")
                ),
                "iou": asociacion["iou"],
                "similitud_area": asociacion["similitud_area"],
                "similitud_forma": asociacion["similitud_forma"],
            }
            for asociacion in asociaciones
        ],
        "track_principal_candidato_id": (
            principal.get("track_id")
            if principal
            else None
        ),
        "nucleos": salida_nucleos,
    }

def frames_historicos_validos(
    radar_dt
):

    candidatos = {}

    for carpeta in (
        BASE / "historico",
        BASE / "historico_tracking_backup",
    ):

        if not carpeta.exists():
            continue

        for archivo in carpeta.glob("*.png"):

            dt = parse_timestamp_filename(
                archivo.stem
            )

            if dt is None:
                continue

            if (
                dt
                >
                radar_dt +
                timedelta(minutes=2)
            ):
                continue

            # Un solo frame por timestamp.
            candidatos[
                iso(dt)
            ] = (
                dt,
                archivo
            )

    salida = list(
        candidatos.values()
    )

    salida.sort(
        key=lambda item: item[0]
    )

    return salida[
        -MAX_REPLAY_FRAMES:
    ]


def nucleos_desde_imagen(
    path,
    mascara,
    detectar_func,
    analizar_func,
    confianza_func,
    aplicar_mascara_func,
    evidencia_func=None,
):
    from scripts.procesador_radar import cargar_imagen, intensidad_radar

    matriz = intensidad_radar(cargar_imagen(path))
    evidencia_temporal = None
    frames_temporales = 0

    if aplicar_mascara_func is not None and mascara is not None:
        if evidencia_func is not None:
            evidencia_temporal, frames_temporales = evidencia_func(
                matriz, mascara, path
            )
        else:
            try:
                from scripts.procesador_radar import _mascara_temporal_dinamica
                evidencia_temporal, frames_temporales = _mascara_temporal_dinamica(
                    matriz, mascara, ruta_actual=path
                )
            except Exception:
                evidencia_temporal = None

    if aplicar_mascara_func is not None:
        try:
            matriz, _, _ = aplicar_mascara_func(
                matriz,
                ruta_actual=path,
                mascara=mascara,
                evidencia_temporal=evidencia_temporal,
                frames_temporales=frames_temporales,
                quiet=True,
            )
        except TypeError:
            matriz, _, _ = aplicar_mascara_func(matriz, ruta_actual=path)
    elif mascara is not None and getattr(mascara, "shape", None) == matriz.shape:
        matriz = matriz.copy()
        matriz[mascara] = 0

    detectados = detectar_func(matriz)
    salida = []

    for indice, nucleo in enumerate(detectados):
        try:
            estado_nucleo, confianza_base = analizar_func(
                nucleo, mascara, evidencia_temporal
            )
        except TypeError:
            estado_nucleo, confianza_base = analizar_func(nucleo, mascara)

        item = dict(nucleo)
        item["indice"] = indice
        item["estado"] = estado_nucleo
        item["confianza"] = confianza_func(
            item, estado_nucleo, confianza_base
        )
        salida.append(normalizar_nucleo(item, indice))

    return [
        nucleo for nucleo in salida
        if nucleo.get("estado") != "ECO_ESTATICO"
    ]


def procesar():

    timestamp_path = (
        BASE /
        "radar_timestamp.txt"
    )

    radar_dt = None

    if timestamp_path.exists():

        radar_dt = parse_iso(
            timestamp_path.read_text(
                encoding="utf-8"
            ).strip()
        )

    if radar_dt is None:

        radar_dt = timestamp_radar()

    radar_ts = iso(
        radar_dt
    )

    try:

        from scripts.procesador_radar import (
            detectar_nucleos
                as detectar_nucleos_desde_procesador,

            cargar_mascara
                as cargar_mascara_desde_procesador,

            analizar_mascara
                as analizar_mascara_desde_procesador,

            calcular_confianza
                as calcular_confianza_desde_procesador,

            aplicar_mascara_estatica
                as aplicar_mascara_desde_procesador,
        )

    except Exception as exc:

        raise RuntimeError(
            "No se pudo cargar el procesador RMA10 "
            "para reconstruir el historial."
        ) from exc

    mascara = (
        cargar_mascara_desde_procesador()
    )

    frames = frames_historicos_validos(
        radar_dt
    )

    actual_path = (
        BASE /
        "actual.png"
    )

    presentes = {
        iso(dt)
        for dt, _ in frames
    }

    if (
        actual_path.exists()
        and
        radar_ts not in presentes
    ):

        frames.append(
            (
                radar_dt,
                actual_path
            )
        )

    frames.sort(
        key=lambda item:
            item[0]
    )

    frames = frames[
        -MAX_REPLAY_FRAMES:
    ]

    if not frames:

        raise RuntimeError(
            "No hay frames RMA10 disponibles "
            "para reconstruir el seguimiento."
        )

    estado_previo = normalizar_estado(
        cargar_json(
            ESTADO,
            {}
        )
    )

    estado_previo_ts = estado_previo.get(
        "ultimo_radar_timestamp"
    )

    estado_previo_dt = parse_iso(
        estado_previo_ts
    )

    ultimo_resultado = None

    resumen_replay = []

    # Primero usamos el estado persistido de la ejecución anterior.
    # Es la vía normal de continuidad y evita reiniciar los tracks
    # cada vez que el artefacto histórico llegue incompleto.
    if (
        estado_previo_dt is not None
        and estado_previo_ts != radar_ts
        and estado_previo_dt < radar_dt
        and (
            radar_dt - estado_previo_dt
        ).total_seconds() / 60.0 <= MAX_INTERVALO
        and estado_previo.get("tracks")
    ):

        nucleos = nucleos_desde_imagen(
            actual_path,
            mascara,
            detectar_nucleos_desde_procesador,
            analizar_mascara_desde_procesador,
            calcular_confianza_desde_procesador,
            aplicar_mascara_desde_procesador,
        )

        estado = estado_previo

        ultimo_resultado = (
            procesar_frame_secuencia(
                estado,
                nucleos,
                radar_ts,
                estado_previo_ts,
                frame_actual=actual_path,
            )
        )

        resumen_replay.append(
            {
                "timestamp": radar_ts,
                "archivo": actual_path.name,
                "nucleos": len(nucleos),
                "seguimiento": ultimo_resultado.get(
                    "nucleos_seguimiento",
                    0
                ),
                "movimiento": ultimo_resultado.get(
                    "nucleos_movimiento",
                    0
                ),
                "modo": "ESTADO_PREVIO_PERSISTIDO",
            }
        )

        frames_reconstruidos = 2
        modo_reconstruccion = (
            "CONTINUIDAD_ESTADO_PREVIO"
        )

    else:

        estado = estado_nuevo()

        # La numeración de las tormentas NUNCA vuelve a 1 (auditoría
        # 20/11/2026): al reconstruir (cielo limpio o hueco largo) los
        # números reiniciaban y una predicción vieja de la "tormenta 5" se
        # comparaba con otra tormenta distinta que también era la 5.
        estado["siguiente_track_id"] = max(
            estado["siguiente_track_id"],
            entero(estado_previo.get("siguiente_track_id"), 1),
        )

        for frame_dt, frame_path in frames:

            frame_ts = iso(
                frame_dt
            )

            nucleos = nucleos_desde_imagen(
                frame_path,
                mascara,
                detectar_nucleos_desde_procesador,
                analizar_mascara_desde_procesador,
                calcular_confianza_desde_procesador,
                aplicar_mascara_desde_procesador,
            )

            resultado = (
                procesar_frame_secuencia(
                    estado,
                    nucleos,
                    frame_ts,
                    estado.get(
                        "ultimo_radar_timestamp"
                    ),
                    frame_actual=frame_path,
                )
            )

            ultimo_resultado = resultado

            resumen_replay.append(
                {
                    "timestamp": frame_ts,
                    "archivo": frame_path.name,
                    "nucleos": len(nucleos),
                    "seguimiento": resultado.get(
                        "nucleos_seguimiento",
                        0
                    ),
                    "movimiento": resultado.get(
                        "nucleos_movimiento",
                        0
                    ),
                    "modo": "REPLAY_HISTORICO",
                }
            )

        frames_reconstruidos = len(
            frames
        )
        modo_reconstruccion = (
            "RECONSTRUCCION_HISTORICA"
        )

    if ultimo_resultado is None:

        raise RuntimeError(
            "No se pudo reconstruir "
            "el seguimiento RMA10."
        )

    ultimo_resultado[
        "modo_seguimiento"
    ] = modo_reconstruccion

    ultimo_resultado[
        "frames_reconstruidos"
    ] = frames_reconstruidos

    ultimo_resultado[
        "historial_replay"
    ] = resumen_replay

    guardar_json(
        ESTADO,
        estado
    )

    guardar_json(
        SALIDA,
        ultimo_resultado
    )

    return ultimo_resultado


if __name__ == "__main__":

    resultado = procesar()

    print("")

    print(
        "======================================"
    )

    print(
        "SEGUIMIENTO TEMPORAL TERMINADO"
    )

    print(
        "======================================"
    )

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
