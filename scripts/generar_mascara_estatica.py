from pathlib import Path
from datetime import datetime, timezone, timedelta

import json

import numpy as np
from PIL import Image
from scipy import ndimage


HISTORICO = Path(
    "data/radar/historico"
)

RADAR_TIMESTAMP = Path(
    "data/radar/radar_timestamp.txt"
)

SALIDA = Path(
    "data/radar/mascara_ecos_estaticos.png"
)

INFO = Path(
    "data/radar/mascara_ecos_estaticos.json"
)

CONFIANZA = Path(
    "data/radar/mascara_ecos_estaticos_confianza.png"
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

MAX_FRAMES = 40

PERSISTENCIA_MINIMA = 0.75

ESTABILIDAD_MINIMA = 0.75

MAX_FUTURE_SKEW_MINUTES = 2.0


# ============================================================
# TIMESTAMP
# ============================================================

def cargar_timestamp():

    if not RADAR_TIMESTAMP.exists():
        raise RuntimeError(
            "No existe data/radar/radar_timestamp.txt"
        )

    value = RADAR_TIMESTAMP.read_text(
        encoding="utf-8"
    ).strip()

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
            f"Timestamp de radar inválido: {value}"
        ) from exc


def timestamp_archivo(path):

    try:

        nombre = path.stem

        # Formato actual:
        # YYYYMMDDHHMMSS

        if (
            len(nombre) == 14
            and nombre.isdigit()
        ):

            return datetime.strptime(
                nombre,
                "%Y%m%d%H%M%S"
            ).replace(
                tzinfo=timezone.utc
            )

        # Compatibilidad:
        # YYYYMMDD_HHMMSS

        return datetime.strptime(
            nombre,
            "%Y%m%d_%H%M%S"
        ).replace(
            tzinfo=timezone.utc
        )

    except Exception:

        return None


# ============================================================
# CLASIFICACIÓN VECTORIAL
# ============================================================

def intensidad_radar(rgb):

    r = rgb[:, :, 0].astype(
        np.int16
    )

    g = rgb[:, :, 1].astype(
        np.int16
    )

    b = rgb[:, :, 2].astype(
        np.int16
    )

    salida = np.zeros(
        r.shape,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # NIVEL 1
    # --------------------------------------------------------

    salida[
        (b > g) &
        (g > r * 1.5)
    ] = 1

    # --------------------------------------------------------
    # NIVEL 2
    # --------------------------------------------------------

    salida[
        (g > r * 1.35) &
        (g >= b * 0.95)
    ] = 2

    # --------------------------------------------------------
    # NIVEL 3
    # --------------------------------------------------------

    salida[
        (r > 100) &
        (g > 100) &
        (b < 100) &
        (
            np.abs(
                r - g
            ) < 100
        )
    ] = 3

    # --------------------------------------------------------
    # NIVEL 4
    # --------------------------------------------------------

    salida[
        (r > 130) &
        (g > 60) &
        (g < 180) &
        (b < 90) &
        (r > g * 1.15)
    ] = 4

    # --------------------------------------------------------
    # NIVEL 5
    # --------------------------------------------------------

    salida[
        (r > 130) &
        (r > g * 1.35) &
        (r > b * 1.35)
    ] = 5

    # --------------------------------------------------------
    # NIVEL 6
    # --------------------------------------------------------

    salida[
        (r > 100) &
        (b > 100) &
        (r > g * 1.3)
    ] = 6

    # El radar necesita distinguir negro/gris
    # de ecos válidos.

    maxima = np.maximum.reduce(
        [r, g, b]
    )

    minima = np.minimum.reduce(
        [r, g, b]
    )

    poco_contraste = (
        maxima < 25
    ) | (
        (maxima - minima) < 20
    )

    salida[
        poco_contraste
    ] = 0

    return salida


# ============================================================
# CARGAR HISTORIAL
# ============================================================

def obtener_archivos():

    radar_dt = cargar_timestamp()

    limite = (
        radar_dt +
        timedelta(
            minutes=MAX_FUTURE_SKEW_MINUTES
        )
    )

    archivos = []

    futuros = []

    for path in HISTORICO.glob(
        "*.png"
    ):

        dt = timestamp_archivo(
            path
        )

        if dt is None:
            continue

        if dt > limite:

            futuros.append(
                path.name
            )

            continue

        archivos.append(
            (
                dt,
                path
            )
        )

    archivos.sort(
        key=lambda x: x[0]
    )

    archivos = archivos[
        -MAX_FRAMES:
    ]

    return (
        radar_dt,
        [path for _, path in archivos],
        futuros
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "======================================"
    )

    print(
        "ClimaAR - Máscara estática optimizada"
    )

    print(
        "======================================"
    )

    radar_dt, archivos, futuros = (
        obtener_archivos()
    )

    if len(archivos) < 3:

        raise RuntimeError(
            "Se necesitan al menos 3 "
            "frames históricos válidos. "
            f"Encontrados: {len(archivos)}"
        )

    print(
        f"Frames analizados: {len(archivos)}"
    )

    if futuros:

        print(
            "Frames futuros excluidos:",
            len(futuros)
        )

    # --------------------------------------------------------
    # CARGAR IMÁGENES
    # --------------------------------------------------------

    matrices = []

    tamaño = None

    for numero, archivo in enumerate(
        archivos,
        start=1
    ):

        print(
            f"[{numero}/{len(archivos)}] "
            f"{archivo.name}"
        )

        try:

            imagen = Image.open(
                archivo
            ).convert(
                "RGB"
            )

            if tamaño is None:
                tamaño = imagen.size

            if imagen.size != tamaño:

                raise RuntimeError(
                    "Tamaño diferente al "
                    "resto del historial."
                )

            rgb = np.asarray(
                imagen,
                dtype=np.uint8
            )

            intensidad = (
                intensidad_radar(
                    rgb
                )
            )

            matrices.append(
                intensidad
            )

        except Exception as exc:

            raise RuntimeError(
                f"No se pudo procesar "
                f"{archivo}: {exc}"
            ) from exc

    # --------------------------------------------------------
    # STACK TEMPORAL
    #
    # shape:
    #
    # frames x alto x ancho
    # --------------------------------------------------------

    historial = np.stack(
        matrices,
        axis=0
    )

    frames = historial.shape[0]

    print(
        "Matriz temporal:",
        historial.shape
    )

    # --------------------------------------------------------
    # PRESENCIA
    # --------------------------------------------------------

    presente = (
        historial > 0
    )

    cantidad_presente = np.sum(
        presente,
        axis=0,
        dtype=np.uint16
    )

    persistencia = (
        cantidad_presente /
        float(frames)
    )

    min_activos = max(
        2,
        int(
            np.ceil(
                frames *
                PERSISTENCIA_MINIMA
            )
        )
    )

    candidatos = (
        cantidad_presente >=
        min_activos
    )

    print(
        "Frames mínimos activos:",
        min_activos
    )

    # --------------------------------------------------------
    # ESTABILIDAD DE INTENSIDAD
    # --------------------------------------------------------
    #
    # Solamente evaluamos los candidatos.
    # No recorremos píxel por píxel.
    # --------------------------------------------------------

    if np.any(candidatos):

        valores = historial[
            :,
            candidatos
        ]

        minimo = np.min(
            valores,
            axis=0
        )

        maximo = np.max(
            valores,
            axis=0
        )

        variacion = (
            maximo -
            minimo
        )

        media = np.mean(
            valores,
            axis=0
        )

        # ----------------------------------------------------
        # INTENSIDAD ACTUAL
        # ----------------------------------------------------

        actual = historial[
            -1,
            candidatos
        ]

        diferencia = np.abs(
            actual.astype(
                np.int16
            ) -
            np.rint(
                media
            ).astype(
                np.int16
            )
        )

        estatico = (
            (variacion <= 2) &
            (diferencia <= 2)
        )

        indices = np.flatnonzero(
            candidatos
        )

        mascara_flat = np.zeros(
            candidatos.size,
            dtype=np.uint8
        )

        confianza_flat = np.zeros(
            candidatos.size,
            dtype=np.uint8
        )

        confianza = (
            0.65 *
            persistencia[
                candidatos
            ] +
            0.35 *
            (
                np.maximum(
                    0,
                    1 -
                    (
                        variacion /
                        6.0
                    )
                )
            )
        )

        confianza = np.clip(
            confianza,
            0.0,
            1.0
        )

        mascara_flat[
            indices[estatico]
        ] = 255

        confianza_flat[
            indices[estatico]
        ] = (
            np.rint(
                confianza[
                    estatico
                ] *
                255
            )
            .astype(
                np.uint8
            )
        )

        mascara_data = (
            mascara_flat.reshape(
                candidatos.shape
            )
        )

        confianza_data = (
            confianza_flat.reshape(
                candidatos.shape
            )
        )

    else:

        mascara_data = np.zeros(
            historial.shape[1:],
            dtype=np.uint8
        )

        confianza_data = np.zeros(
            historial.shape[1:],
            dtype=np.uint8
        )

    # --------------------------------------------------------
    # LIMPIEZA PEQUEÑA
    # --------------------------------------------------------

    mascara_bool = (
        mascara_data > 0
    )

    mascara_bool = ndimage.binary_opening(
        mascara_bool,
        structure=np.ones(
            (3, 3),
            dtype=bool
        )
    )

    mascara_bool = ndimage.binary_closing(
        mascara_bool,
        structure=np.ones(
            (3, 3),
            dtype=bool
        )
    )

    mascara_data = (
        mascara_bool
        .astype(np.uint8)
        * 255
    )

    # --------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------

    estaticos = int(
        np.sum(
            mascara_bool
        )
    )

    total = (
        mascara_bool.size
    )

    porcentaje = (
        estaticos /
        total *
        100
    )

    mascara = Image.fromarray(
        mascara_data,
        mode="L"
    )

    confianza = Image.fromarray(
        confianza_data,
        mode="L"
    )

    SALIDA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    mascara.save(
        SALIDA
    )

    confianza.save(
        CONFIANZA
    )

    radar_iso = (
        radar_dt
        .isoformat()
        .replace(
            "+00:00",
            "Z"
        )
    )

    info = {

        "schema_version": 4,

        "metodo":
            "NUMPY_PERSISTENCIA_TEMPORAL_ESTABILIDAD",

        "radar_timestamp_utc":
            radar_iso,

        "frames_analizados":
            len(archivos),

        "frames":
            [p.name for p in archivos],

        "frames_futuros_excluidos":
            len(futuros),

        "persistencia_minima":
            PERSISTENCIA_MINIMA,

        "estabilidad_minima":
            ESTABILIDAD_MINIMA,

        "min_frames_activos":
            min_activos,

        "pixeles_estaticos":
            estaticos,

        "porcentaje_mascara":
            round(
                porcentaje,
                3
            ),

        "ancho":
            tamaño[0],

        "alto":
            tamaño[1],

        "mascara_no_geografica":
            True,

        "permite_recuperacion_de_ecos_nuevos":
            True,

        "proteccion_temporal":
            True,

        "permite_que_un_ecO_dinamico_cruce_la_zona":
            True,
    }

    INFO.write_text(
        json.dumps(
            info,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print(
        "======================================"
    )
    print(
        "MÁSCARA ESTÁTICA COMPLETADA"
    )
    print(
        "======================================"
    )

    print(
        "Frames:",
        len(archivos)
    )

    print(
        "Píxeles estáticos:",
        estaticos
    )

    print(
        "Porcentaje:",
        round(
            porcentaje,
            3
        ),
        "%"
    )

    print(
        "Máscara:",
        SALIDA
    )

    print(
        "Confianza:",
        CONFIANZA
    )

    print(
        "Info:",
        INFO
    )


if __name__ == "__main__":
    main()
