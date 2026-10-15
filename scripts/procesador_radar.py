from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import re

import numpy as np
from PIL import Image
from scipy import ndimage


RADAR_DIR = Path("data/radar")
HISTORICO_DIR = RADAR_DIR / "historico"
MASK_PATH = RADAR_DIR / "mascara_ecos_estaticos.png"
OUTPUT_PATH = RADAR_DIR / "resultado_procesamiento.json"
TIMESTAMP_FILE = RADAR_DIR / "radar_timestamp.txt"
STATE_PATH = RADAR_DIR / "seguimiento" / "estado.json"
PENDING_PATH = RADAR_DIR / "seguimiento" / "predicciones_pendientes.json"

TIMESTAMP_RE = re.compile(r"^(\d{14})\.png$")

MAX_FUTURE_SKEW_MINUTES = 2.0

MAX_HISTORICOS_COMPARAR = 5
MAX_ANTIGUEDAD_HISTORICO_MIN = 60

MIN_FRAMES_COMPARABLES = 3
MIN_FRAMES_ESTABLES = 3
TOLERANCIA_INTENSIDAD_ESTATICA = 1
MIN_CAMBIOS_DINAMICOS = 1

MIN_NUCLEO_PIXELES = 10
MIN_SEMILLA_PIXELES = 5

# Distancia mínima entre centros de células.
SEPARACION_SEMILLAS_PX = 7


NIVELES = {
    0: "FONDO",
    1: "MUY_DEBIL",
    2: "DEBIL",
    3: "MODERADA",
    4: "FUERTE",
    5: "MUY_FUERTE",
    6: "SEVERA",
}


# ============================================================
# COLORES RMA10
# ============================================================

def clasificar_color(r, g, b):

    if max(r, g, b) < 25:
        return "FONDO", 0

    if b > g and g > r * 1.5:
        return "AZUL_CYAN", 1

    if g > r * 1.35 and g >= b * 0.95:
        return "VERDE", 2

    if (
        r > 100
        and g > 100
        and b < 100
        and abs(r - g) < 100
    ):
        return "AMARILLO", 3

    if (
        r > 130
        and 60 < g < 180
        and b < 90
        and r > g * 1.15
    ):
        return "NARANJA", 4

    if (
        r > 130
        and r > g * 1.35
        and r > b * 1.35
    ):
        return "ROJO", 5

    if (
        r > 100
        and b > 100
        and r > g * 1.3
    ):
        return "VIOLETA_MAGENTA", 6

    return "OTRO", 0


# ============================================================
# IMAGEN → MATRIZ DE INTENSIDAD
# ============================================================

def imagen_a_matriz_intensidad(imagen):

    arr = np.asarray(
        imagen.convert("RGBA")
    )

    matriz = np.zeros(
        arr.shape[:2],
        dtype=np.uint8
    )

    for y in range(arr.shape[0]):

        for x in range(arr.shape[1]):

            r, g, b = arr[y, x, :3]

            _, intensidad = clasificar_color(
                int(r),
                int(g),
                int(b)
            )

            matriz[y, x] = intensidad

    return matriz


# ============================================================
# TIMESTAMP
# ============================================================

def parse_timestamp_file():

    if not TIMESTAMP_FILE.exists():
        raise RuntimeError(
            "No existe data/radar/radar_timestamp.txt"
        )

    value = TIMESTAMP_FILE.read_text(
        encoding="utf-8"
    ).strip()

    if not value:
        raise RuntimeError(
            "radar_timestamp.txt esta vacio"
        )

    try:

        dt = datetime.fromisoformat(
            value.replace(
                "Z",
                "+00:00"
            )
        )

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt.astimezone(
            timezone.utc
        )

    except Exception as exc:

        raise RuntimeError(
            f"Timestamp radar invalido: {value}"
        ) from exc


def timestamp_compacto(dt):

    return dt.strftime(
        "%Y%m%d%H%M%S"
    )


# ============================================================
# HISTÓRICO
# ============================================================

def obtener_historicos_validos(
    radar_dt
):

    if not HISTORICO_DIR.exists():
        return []

    limite = (
        radar_dt
        - timedelta(
            minutes=MAX_ANTIGUEDAD_HISTORICO_MIN
        )
    )

    candidatos = []

    for archivo in HISTORICO_DIR.glob(
        "*.png"
    ):

        match = TIMESTAMP_RE.match(
            archivo.name
        )

        if not match:
            continue

        try:

            dt = datetime.strptime(
                match.group(1),
                "%Y%m%d%H%M%S"
            ).replace(
                tzinfo=timezone.utc
            )

        except Exception:
            continue

        if limite <= dt < radar_dt:

            candidatos.append(
                (
                    dt,
                    archivo
                )
            )

    candidatos.sort(
        key=lambda item: item[0],
        reverse=True
    )

    seleccionados = []

    for dt, archivo in candidatos[
        :MAX_HISTORICOS_COMPARAR
    ]:

        try:

            imagen = Image.open(
                archivo
            ).convert("RGBA")

            seleccionados.append(
                (
                    dt,
                    imagen
                )
            )

        except Exception as exc:

            print(
                f"No se pudo leer historico "
                f"{archivo.name}: {exc}"
            )

    seleccionados.sort(
        key=lambda item: item[0]
    )

    return seleccionados


# ============================================================
# ELIMINAR FRAMES FUTUROS
# ============================================================

def limpiar_frames_futuros(
    radar_dt
):

    eliminados = 0

    if not HISTORICO_DIR.exists():
        return 0

    limite = (
        radar_dt
        + timedelta(
            minutes=MAX_FUTURE_SKEW_MINUTES
        )
    )

    for archivo in HISTORICO_DIR.glob(
        "*.png"
    ):

        match = TIMESTAMP_RE.match(
            archivo.name
        )

        if not match:
            continue

        try:

            dt = datetime.strptime(
                match.group(1),
                "%Y%m%d%H%M%S"
            ).replace(
                tzinfo=timezone.utc
            )

        except Exception:
            continue

        if dt > limite:

            archivo.unlink(
                missing_ok=True
            )

            eliminados += 1

    return eliminados


# ============================================================
# REPARAR ESTADO FUTURO
# ============================================================

def reparar_estado_futuro(
    radar_dt
):

    if not STATE_PATH.exists():
        return False

    try:

        data = json.loads(
            STATE_PATH.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            data,
            dict
        ):
            return False

        value = data.get(
            "ultimo_radar_timestamp"
        )

        if not value:
            return False

        ultimo = datetime.fromisoformat(
            str(value).replace(
                "Z",
                "+00:00"
            )
        )

        if ultimo.tzinfo is None:
            ultimo = ultimo.replace(
                tzinfo=timezone.utc
            )

        ultimo = ultimo.astimezone(
            timezone.utc
        )

    except Exception:
        return False

    if ultimo <= (
        radar_dt
        + timedelta(
            minutes=MAX_FUTURE_SKEW_MINUTES
        )
    ):
        return False

    data[
        "ultimo_radar_timestamp"
    ] = None

    data[
        "tracks"
    ] = []

    data[
        "tracks_activos"
    ] = []

    data[
        "ultimo_frame_nucleos"
    ] = []

    data[
        "siguiente_track_id"
    ] = 1

    data[
        "reparacion_timestamp"
    ] = (
        datetime.now(
            timezone.utc
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z"
        )
    )

    data[
        "motivo_reparacion"
    ] = (
        "Estado anterior tenia "
        "un radar futuro invalido. "
        "Se reinicio el seguimiento."
    )

    STATE_PATH.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    return True


# ============================================================
# LIMPIAR PREDICCIONES FUTURAS
# ============================================================

def limpiar_predicciones_futuras(
    radar_dt
):

    if not PENDING_PATH.exists():
        return 0

    try:

        data = json.loads(
            PENDING_PATH.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        return 0

    if not isinstance(
        data,
        dict
    ):
        return 0

    pendientes = data.get(
        "pendientes",
        []
    )

    if not isinstance(
        pendientes,
        list
    ):
        return 0

    validos = []
    eliminados = 0

    limite = (
        radar_dt
        + timedelta(
            minutes=MAX_FUTURE_SKEW_MINUTES
        )
    )

    for item in pendientes:

        if not isinstance(
            item,
            dict
        ):
            continue

        value = item.get(
            "timestamp_origen"
        )

        futuro = False

        if value:

            try:

                dt = datetime.fromisoformat(
                    str(value).replace(
                        "Z",
                        "+00:00"
                    )
                )

                if dt.tzinfo is None:
                    dt = dt.replace(
                        tzinfo=timezone.utc
                    )

                futuro = (
                    dt.astimezone(
                        timezone.utc
                    )
                    > limite
                )

            except Exception:
                pass

        if futuro:

            eliminados += 1

        else:

            validos.append(
                item
            )

    data[
        "pendientes"
    ] = validos[-500:]

    data[
        "actualizado"
    ] = (
        radar_dt
        .isoformat()
        .replace(
            "+00:00",
            "Z"
        )
    )

    PENDING_PATH.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    return eliminados


# ============================================================
# MÁSCARA ESTÁTICA
# ============================================================

def cargar_mascara():

    if not MASK_PATH.exists():
        return None

    try:

        return Image.open(
            MASK_PATH
        ).convert("L")

    except Exception:

        return None


# ============================================================
# FILTRO PIXEL A PIXEL
# ============================================================

def construir_matriz_filtrada(
    imagen_actual,
    mascara_estatica,
    historicos
):

    actual = imagen_a_matriz_intensidad(
        imagen_actual
    )

    matrices = []

    for dt, imagen in historicos:

        if imagen.size != imagen_actual.size:
            continue

        matrices.append(
            (
                dt,
                imagen_a_matriz_intensidad(
                    imagen
                )
            )
        )

    comparables = len(
        matrices
    )

    if (
        mascara_estatica is None
        or comparables
        < MIN_FRAMES_COMPARABLES
    ):

        return (
            actual,
            0,
            0,
            comparables
        )

    mask = (
        np.asarray(
            mascara_estatica,
            dtype=np.uint8
        )
        > 0
    )

    if mask.shape != actual.shape:

        return (
            actual,
            0,
            0,
            comparables
        )

    resultado = actual.copy()

    estaticos = 0
    dinamicos = 0

    ys, xs = np.where(
        mask & (actual > 0)
    )

    for y, x in zip(
        ys,
        xs
    ):

        valores = np.array(
            [
                matriz[y, x]
                for _, matriz in matrices
            ],
            dtype=np.int16
        )

        presentes = valores[
            valores > 0
        ]

        if (
            len(presentes)
            < MIN_FRAMES_COMPARABLES
        ):
            continue

        cambios = int(
            np.sum(
                np.abs(
                    presentes
                    - int(actual[y, x])
                )
                > TOLERANCIA_INTENSIDAD_ESTATICA
            )
        )

        estables = (
            len(presentes)
            - cambios
        )

        if (
            estables
            >= MIN_FRAMES_ESTABLES
            and cambios
            < MIN_CAMBIOS_DINAMICOS
        ):

            resultado[
                y,
                x
            ] = 0

            estaticos += 1

        elif (
            cambios
            >= MIN_CAMBIOS_DINAMICOS
        ):

            dinamicos += 1

    return (
        resultado,
        estaticos,
        dinamicos,
        comparables
    )


# ============================================================
# SEMILLAS DE CÉLULAS
#
# ESTA ES LA CORRECCIÓN PRINCIPAL.
#
# Antes:
#     todo lo conectado = un núcleo.
#
# Ahora:
#     se buscan centros de reflectividad alta
#     y se separan las células mediante Voronoi.
# ============================================================

def detectar_semillas(
    matriz
):

    for umbral in (
        5,
        4,
        3
    ):

        candidatos = (
            matriz >= umbral
        )

        if not np.any(
            candidatos
        ):
            continue

        maximo = (
            ndimage.maximum_filter(
                matriz,
                size=15,
                mode="nearest"
            )
        )

        puntos = (
            candidatos
            & (matriz == maximo)
        )

        labels, cantidad = (
            ndimage.label(
                puntos,
                structure=np.ones(
                    (3, 3),
                    dtype=np.uint8
                )
            )
        )

        centros = []

        for idx in range(
            1,
            cantidad + 1
        ):

            ys, xs = np.where(
                labels == idx
            )

            if len(xs) == 0:
                continue

            nivel = int(
                matriz[
                    ys,
                    xs
                ].max()
            )

            centros.append(
                (
                    float(xs.mean()),
                    float(ys.mean()),
                    nivel
                )
            )

        if not centros:
            continue

        centros.sort(
            key=lambda item: item[2],
            reverse=True
        )

        aceptados = []

        for candidato in centros:

            cx, cy, nivel = candidato

            demasiado_cerca = False

            for ax, ay, _ in aceptados:

                distancia = (
                    (
                        (cx - ax) ** 2
                        +
                        (cy - ay) ** 2
                    )
                    ** 0.5
                )

                if (
                    distancia
                    < SEPARACION_SEMILLAS_PX
                ):

                    demasiado_cerca = True
                    break

            if not demasiado_cerca:

                aceptados.append(
                    candidato
                )

        semillas = np.zeros(
            matriz.shape,
            dtype=np.int32
        )

        for seed_id, (
            cx,
            cy,
            _
        ) in enumerate(
            aceptados,
            1
        ):

            xi = int(
                round(cx)
            )

            yi = int(
                round(cy)
            )

            if (
                0 <= yi
                < matriz.shape[0]
                and
                0 <= xi
                < matriz.shape[1]
            ):

                semillas[
                    yi,
                    xi
                ] = seed_id

        if np.any(
            semillas
        ):

            return semillas

    return np.zeros(
        matriz.shape,
        dtype=np.int32
    )


# ============================================================
# SEPARAR NÚCLEOS
# ============================================================

def separar_nucleos(
    matriz
):

    precipitacion = (
        matriz > 0
    )

    if not np.any(
        precipitacion
    ):
        return []

    semillas = detectar_semillas(
        matriz
    )

    cantidad_semillas = int(
        semillas.max()
    )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if cantidad_semillas == 0:

        labels, cantidad = (
            ndimage.label(
                precipitacion,
                structure=np.ones(
                    (3, 3),
                    dtype=np.uint8
                )
            )
        )

        salida = []

        for idx in range(
            1,
            cantidad + 1
        ):

            ys, xs = np.where(
                labels == idx
            )

            if (
                len(xs)
                >= MIN_NUCLEO_PIXELES
            ):

                salida.append(
                    (
                        xs,
                        ys
                    )
                )

        return salida

    # --------------------------------------------------------
    # VORONOI
    #
    # Cada píxel de precipitación se asigna
    # a la semilla más cercana.
    #
    # Esto rompe las "uniones" entre células.
    # --------------------------------------------------------

    _, indices = (
        ndimage.distance_transform_edt(
            ~(semillas > 0),
            return_indices=True
        )
    )

    nearest_y = indices[0]
    nearest_x = indices[1]

    labels = np.zeros(
        matriz.shape,
        dtype=np.int32
    )

    seed_ids = (
        semillas[
            nearest_y,
            nearest_x
        ]
    )

    labels[
        precipitacion
    ] = seed_ids[
        precipitacion
    ]

    salida = []

    for idx in range(
        1,
        cantidad_semillas + 1
    ):

        ys, xs = np.where(
            labels == idx
        )

        if (
            len(xs)
            >= MIN_NUCLEO_PIXELES
        ):

            salida.append(
                (
                    xs,
                    ys
                )
            )

    return salida


# ============================================================
# COLORES
# ============================================================

def colores_de_puntos(
    imagen,
    xs,
    ys
):

    arr = np.asarray(
        imagen.convert("RGBA")
    )

    presentes = set()

    for x, y in zip(
        xs,
        ys
    ):

        r, g, b = (
            arr[
                y,
                x,
                :3
            ]
        )

        color, _ = clasificar_color(
            int(r),
            int(g),
            int(b)
        )

        if color not in {
            "FONDO",
            "OTRO"
        }:

            presentes.add(
                color
            )

    return sorted(
        presentes
    )


# ============================================================
# DETECTAR NÚCLEOS
# ============================================================

def detectar_nucleos_desde_matriz(
    matriz,
    imagen_original
):

    componentes = separar_nucleos(
        matriz
    )

    nucleos = []

    for xs, ys in componentes:

        min_x = int(
            xs.min()
        )

        max_x = int(
            xs.max()
        )

        min_y = int(
            ys.min()
        )

        max_y = int(
            ys.max()
        )

        intensidades = (
            matriz[
                ys,
                xs
            ]
        )

        max_intensidad = int(
            intensidades.max()
        )

        nucleos.append({

            "x": round(
                float(xs.mean()),
                2
            ),

            "y": round(
                float(ys.mean()),
                2
            ),

            "pixeles": int(
                len(xs)
            ),

            "ancho": (
                max_x
                - min_x
                + 1
            ),

            "alto": (
                max_y
                - min_y
                + 1
            ),

            "intensidad": (
                max_intensidad
            ),

            "nivel_intensidad": (
                NIVELES.get(
                    max_intensidad,
                    "OTRO"
                )
            ),

            "colores": (
                colores_de_puntos(
                    imagen_original,
                    xs,
                    ys
                )
            ),
        })

    nucleos.sort(
        key=lambda nucleo: (
            nucleo["intensidad"],
            nucleo["pixeles"]
        ),
        reverse=True
    )

    return nucleos


# ============================================================
# CONFIANZA
# ============================================================

def calcular_confianza(
    nucleo,
    estado,
    base
):

    confianza = (
        float(base)
        + float(
            nucleo["intensidad"]
        ) * 2
    )

    if (
        nucleo["intensidad"]
        >= 5
    ):

        confianza += 2

    return round(
        max(
            1,
            min(
                99,
                confianza
            )
        ),
        1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    HISTORICO_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    STATE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    radar_dt = (
        parse_timestamp_file()
    )

    ahora = datetime.now(
        timezone.utc
    )

    if radar_dt > (
        ahora
        + timedelta(
            minutes=MAX_FUTURE_SKEW_MINUTES
        )
    ):

        raise RuntimeError(
            "El radar_timestamp.txt "
            "es futuro respecto al reloj "
            "del runner: "
            f"{radar_dt.isoformat()}"
        )

    actual_path = (
        RADAR_DIR
        / "actual.png"
    )

    if not actual_path.exists():

        raise RuntimeError(
            "No existe "
            "data/radar/actual.png"
        )

    eliminados = (
        limpiar_frames_futuros(
            radar_dt
        )
    )

    estado_reparado = (
        reparar_estado_futuro(
            radar_dt
        )
    )

    pendientes_eliminados = (
        limpiar_predicciones_futuras(
            radar_dt
        )
    )

    imagen = Image.open(
        actual_path
    ).convert("RGBA")

    mascara = (
        cargar_mascara()
    )

    historicos = (
        obtener_historicos_validos(
            radar_dt
        )
    )

    (
        matriz_filtrada,
        pixeles_estaticos,
        pixeles_dinamicos,
        frames_comparados
    ) = construir_matriz_filtrada(
        imagen,
        mascara,
        historicos
    )

    nucleos = (
        detectar_nucleos_desde_matriz(
            matriz_filtrada,
            imagen
        )
    )

    print(
        "========================================"
    )

    print(
        "CLIMAAR RMA10"
    )

    print(
        "========================================"
    )

    print(
        "Frame: "
        f"{timestamp_compacto(radar_dt)}"
    )

    print(
        f"Resolucion: "
        f"{imagen.width}x{imagen.height}"
    )

    print(
        "Frames historicos comparados: "
        f"{frames_comparados}"
    )

    print(
        "Pixeles eliminados como "
        f"ecos estaticos: "
        f"{pixeles_estaticos}"
    )

    print(
        "Pixeles recuperados como "
        f"dinamicos: "
        f"{pixeles_dinamicos}"
    )

    print(
        "Nucleos independientes detectados: "
        f"{len(nucleos)}"
    )

    print(
        "========================================"
    )

    resultado = []

    for nucleo_id, nucleo in enumerate(
        nucleos,
        1
    ):

        estado = (
            "CAMBIO"
            if pixeles_dinamicos > 0
            else "PRECIPITACION"
        )

        base = (
            70
            if estado == "CAMBIO"
            else 85
        )

        registro = {

            "id": nucleo_id,

            "x": nucleo["x"],

            "y": nucleo["y"],

            "pixeles": nucleo[
                "pixeles"
            ],

            "ancho": nucleo[
                "ancho"
            ],

            "alto": nucleo[
                "alto"
            ],

            "intensidad": nucleo[
                "intensidad"
            ],

            "nivel_intensidad": (
                nucleo[
                    "nivel_intensidad"
                ]
            ),

            "colores": nucleo[
                "colores"
            ],

            "estado": estado,

            "confianza": (
                calcular_confianza(
                    nucleo,
                    estado,
                    base
                )
            ),

            "es_trackeable": True,

            "filtro_estatico": (
                "pixelwise_temporal"
            ),

            "segmentacion": (
                "semillas_intensidad_voronoi"
            ),
        }

        resultado.append(
            registro
        )

    cantidad_nucleos = len(
        resultado
    )

    if cantidad_nucleos:

        confianza_promedio = round(
            sum(
                nucleo[
                    "confianza"
                ]
                for nucleo in resultado
            )
            / cantidad_nucleos,
            1
        )

    else:

        confianza_promedio = 0.0

    salida = {

        "version": 6,

        "timestamp": (
            radar_dt
            .isoformat()
            .replace(
                "+00:00",
                "Z"
            )
        ),

        "timestamp_compacto": (
            timestamp_compacto(
                radar_dt
            )
        ),

        "imagen": (
            "data/radar/actual.png"
        ),

        "resolucion": {

            "ancho": (
                imagen.width
            ),

            "alto": (
                imagen.height
            ),
        },

        "mascara_estatica": (
            MASK_PATH.exists()
        ),

        "metodo_filtro_estatico": (
            "pixelwise_temporal"
        ),

        "metodo_segmentacion": (
            "semillas_intensidad_voronoi"
        ),

        "parametros_segmentacion": {

            "min_nucleo_pixeles": (
                MIN_NUCLEO_PIXELES
            ),

            "min_semilla_pixeles": (
                MIN_SEMILLA_PIXELES
            ),

            "separacion_semillas_px": (
                SEPARACION_SEMILLAS_PX
            ),

            "semillas_prioridad": (
                "intensidad_5_6_4_3"
            ),
        },

        "frames_historicos_comparados": (
            frames_comparados
        ),

        "nucleos_detectados": (
            cantidad_nucleos
        ),

        "nucleos_trackeables": (
            cantidad_nucleos
        ),

        "nucleos_estaticos_excluidos": 0,

        "ecos_estaticos": [],

        "nucleos": resultado,

        "resumen": {

            "nucleos": (
                cantidad_nucleos
            ),

            "precipitacion": sum(
                1
                for nucleo in resultado
                if nucleo["estado"]
                == "PRECIPITACION"
            ),

            "cambios": sum(
                1
                for nucleo in resultado
                if nucleo["estado"]
                == "CAMBIO"
            ),

            "ecos_estaticos": 0,

            "pixeles_ecos_estaticos_excluidos": (
                pixeles_estaticos
            ),

            "pixeles_dinamicos_recuperados": (
                pixeles_dinamicos
            ),

            "confianza_promedio": (
                confianza_promedio
            ),

            "peso_total": round(
                sum(
                    nucleo[
                        "intensidad"
                    ]
                    for nucleo in resultado
                ),
                2
            ),
        },

        "limpieza": {

            "frames_futuros_eliminados": (
                eliminados
            ),

            "estado_reparado": (
                estado_reparado
            ),

            "predicciones_futuras_eliminadas": (
                pendientes_eliminados
            ),
        },
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            salida,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "Resultado guardado en: "
        f"{OUTPUT_PATH}"
    )

    print(
        "Procesamiento RMA10 "
        "finalizado correctamente."
    )


if __name__ == "__main__":
    main()
