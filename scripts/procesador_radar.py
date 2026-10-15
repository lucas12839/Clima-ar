import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from scipy import ndimage


BASE = Path("data/radar")
ACTUAL = BASE / "actual.png"
MASCARA = BASE / "mascara_ecos_estaticos.png"
HISTORICO = BASE / "historico"
SALIDA = BASE / "resultado_procesamiento.json"

MIN_AREA_NUCLEO = 8
INTENSIDAD_MINIMA = 1

# Cantidad máxima de frames históricos utilizados.
MAX_HISTORICOS = 40

# Para declarar un eco como estático:
MIN_FRAMES_COMPARABLES = 5
MIN_PRESENCIA_ESTATICO = 0.80
MAX_DESPLAZAMIENTO_ESTATICO_PX = 4.0
MAX_CAMBIO_INTENSIDAD_ESTATICO = 1

# Una región grande puede contener varios ecos.
# Esto NO se utiliza para declarar severidad.
UMBRAL_REGION_GRANDE = 1500


# ============================================================
# CARGA
# ============================================================

def cargar_imagen(path):
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    return np.array(
        Image.open(path).convert("RGB")
    )


# ============================================================
# INTENSIDAD RMA10
# ============================================================

def intensidad_radar(rgb):
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    intensidad = np.zeros(
        r.shape,
        dtype=np.uint8
    )

    # Azul / celeste
    intensidad[
        (b > r + 20)
        & (b > g + 10)
    ] = 1

    # Verde
    intensidad[
        (g > r + 15)
        & (g > b + 10)
    ] = 2

    # Amarillo
    intensidad[
        (r > 140)
        & (g > 100)
        & (b < 120)
        & (r > b + 40)
    ] = 3

    # Naranja
    intensidad[
        (r > 170)
        & (g > 60)
        & (g < 180)
        & (b < 100)
        & (r > g + 25)
    ] = 4

    # Rojo
    intensidad[
        (r > 170)
        & (g < 90)
        & (b < 100)
        & (r > g + 70)
    ] = 5

    # Magenta / rojo muy intenso
    intensidad[
        (r > 150)
        & (b > 80)
        & (g < 100)
        & (r > g + 50)
    ] = np.maximum(
        intensidad[
            (r > 150)
            & (b > 80)
            & (g < 100)
            & (r > g + 50)
        ],
        6
    )

    return intensidad


# ============================================================
# HISTORIAL
# ============================================================

def obtener_historico():

    if not HISTORICO.exists():
        return []

    archivos = list(
        HISTORICO.glob("*.png")
    )

    archivos.sort(
        key=lambda p: p.stat().st_mtime
    )

    return archivos[-MAX_HISTORICOS:]


def cargar_historicos():

    matrices = []

    for archivo in obtener_historico():

        try:
            imagen = cargar_imagen(
                archivo
            )

            matriz = intensidad_radar(
                imagen
            )

            matrices.append(
                matriz
            )

        except Exception as e:
            print(
                f"No se pudo leer "
                f"{archivo}: {e}"
            )

    return matrices


# ============================================================
# DETECCIÓN DE ECO ESTÁTICO
# ============================================================

def calcular_mascara_estatica(
    actual,
    historicos
):
    """
    Determina píxel por píxel qué zonas son
    persistentemente estáticas.

    IMPORTANTE:

    Un eco con intensidad alta NO se considera
    severo si históricamente permanece fijo.

    Para que un píxel sea estático debe:

    - aparecer persistentemente;
    - mantener aproximadamente la misma intensidad;
    - no mostrar evolución significativa.
    """

    if len(historicos) < MIN_FRAMES_COMPARABLES:

        return np.zeros(
            actual.shape,
            dtype=bool
        )

    historial = np.stack(
        historicos,
        axis=0
    )

    presente = (
        historial > 0
    )

    cantidad_presente = np.sum(
        presente,
        axis=0
    )

    presencia = (
        cantidad_presente
        / len(historicos)
    )

    candidatos = (
        actual > 0
    ) & (
        presencia >= MIN_PRESENCIA_ESTATICO
    )

    if not np.any(candidatos):

        return np.zeros(
            actual.shape,
            dtype=bool
        )

    valores = historial[
        :,
        candidatos
    ]

    # Intensidad media histórica.
    media = np.mean(
        valores,
        axis=0
    )

    minimo = np.min(
        valores,
        axis=0
    )

    maximo = np.max(
        valores,
        axis=0
    )

    variacion = (
        maximo - minimo
    )

    intensidad_actual = actual[
        candidatos
    ]

    diferencia_actual = np.abs(
        intensidad_actual.astype(
            np.int16
        )
        -
        np.rint(media).astype(
            np.int16
        )
    )

    estatico = (
        (variacion <= MAX_CAMBIO_INTENSIDAD_ESTATICO)
        &
        (
            diferencia_actual
            <= MAX_CAMBIO_INTENSIDAD_ESTATICO
        )
    )

    mascara = np.zeros(
        actual.shape,
        dtype=bool
    )

    indices = np.flatnonzero(
        candidatos
    )

    mascara.flat[
        indices[estatico]
    ] = True

    return mascara


# ============================================================
# COMPONENTES
# ============================================================

def detectar_componentes(matriz):

    mascara = (
        matriz >= INTENSIDAD_MINIMA
    )

    estructura = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    etiquetas, cantidad = (
        ndimage.label(
            mascara,
            structure=estructura
        )
    )

    componentes = []

    for etiqueta in range(
        1,
        cantidad + 1
    ):

        region = (
            etiquetas == etiqueta
        )

        area = int(
            np.sum(region)
        )

        if area < MIN_AREA_NUCLEO:
            continue

        componentes.append(
            region
        )

    return componentes


# ============================================================
# INFORMACIÓN DEL NÚCLEO
# ============================================================

def analizar_nucleo(
    region,
    matriz
):

    ys, xs = np.where(
        region
    )

    if len(xs) == 0:
        return None

    valores = matriz[
        region
    ]

    x_min = int(
        xs.min()
    )

    x_max = int(
        xs.max()
    )

    y_min = int(
        ys.min()
    )

    y_max = int(
        ys.max()
    )

    area = int(
        len(xs)
    )

    intensidad_max = int(
        valores.max()
    )

    intensidad_promedio = float(
        np.mean(valores)
    )

    return {
        "area_px": area,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "width_px": (
            x_max - x_min + 1
        ),
        "height_px": (
            y_max - y_min + 1
        ),
        "centroide_x": round(
            float(np.mean(xs)),
            2
        ),
        "centroide_y": round(
            float(np.mean(ys)),
            2
        ),
        "intensidad_max": intensidad_max,
        "intensidad_promedio": round(
            intensidad_promedio,
            3
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=== ClimaAR - Procesador RMA10 ==="
    )

    if not ACTUAL.exists():
        raise FileNotFoundError(
            f"No existe {ACTUAL}"
        )

    # --------------------------------------------------------
    # Imagen actual
    # --------------------------------------------------------

    rgb = cargar_imagen(
        ACTUAL
    )

    matriz_actual = intensidad_radar(
        rgb
    )

    pixeles_originales = int(
        np.sum(
            matriz_actual > 0
        )
    )

    print(
        f"Pixeles con eco actuales: "
        f"{pixeles_originales}"
    )

    # --------------------------------------------------------
    # Historial
    # --------------------------------------------------------

    historicos = cargar_historicos()

    print(
        f"Frames históricos cargados: "
        f"{len(historicos)}"
    )

    # --------------------------------------------------------
    # Máscara histórica
    # --------------------------------------------------------

    mascara_estatica = (
        calcular_mascara_estatica(
            matriz_actual,
            historicos
        )
    )

    pixeles_estaticos = int(
        np.sum(
            mascara_estatica
        )
    )

    print(
        f"Pixeles identificados como "
        f"estáticos: {pixeles_estaticos}"
    )

    # --------------------------------------------------------
    # IMPORTANTE:
    #
    # Los ecos estáticos se eliminan ANTES
    # de detectar núcleos.
    #
    # Por lo tanto no pueden convertirse
    # en núcleos severos.
    # --------------------------------------------------------

    matriz_dinamica = (
        matriz_actual.copy()
    )

    matriz_dinamica[
        mascara_estatica
    ] = 0

    pixeles_dinamicos = int(
        np.sum(
            matriz_dinamica > 0
        )
    )

    # --------------------------------------------------------
    # Detección de núcleos
    # --------------------------------------------------------

    componentes = detectar_componentes(
        matriz_dinamica
    )

    nucleos = []

    for region in componentes:

        nucleo = analizar_nucleo(
            region,
            matriz_dinamica
        )

        if nucleo is None:
            continue

        # Un núcleo que queda después
        # del filtro estático es dinámico.
        nucleo[
            "tipo"
        ] = "DINAMICO"

        nucleos.append(
            nucleo
        )

    # --------------------------------------------------------
    # Orden
    # --------------------------------------------------------

    nucleos.sort(
        key=lambda n: (
            -n["area_px"],
            -n["intensidad_max"]
        )
    )

    # --------------------------------------------------------
    # Estado general
    # --------------------------------------------------------

    if not nucleos:

        estado = "SIN_ECOS_DINAMICOS"

    else:

        max_intensidad = max(
            n["intensidad_max"]
            for n in nucleos
        )

        if max_intensidad >= 6:
            estado = (
                "PRECIPITACION_DINAMICA_MUY_FUERTE"
            )

        elif max_intensidad >= 5:
            estado = (
                "PRECIPITACION_DINAMICA_FUERTE"
            )

        elif max_intensidad >= 4:
            estado = (
                "PRECIPITACION_DINAMICA_MODERADA"
            )

        else:
            estado = (
                "PRECIPITACION_DINAMICA"
            )

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    resultado = {

        "version":
            "RMA10-procesador-8.0",

        "timestamp_procesamiento":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "fuente":
            "SMN RMA10",

        "metodo":
            "filtro_estatico_historico_antes_de_segmentacion",

        "frames_historicos_utilizados":
            len(historicos),

        "pixeles_con_eco_original":
            pixeles_originales,

        "pixeles_identificados_como_estaticos":
            pixeles_estaticos,

        "pixeles_dinamicos":
            pixeles_dinamicos,

        "nucleos_estaticos_excluidos":
            pixeles_estaticos,

        "nucleos_detectados":
            len(nucleos),

        "estado_general":
            estado,

        "ecos_estaticos_se_excluyen_antes_del_tracking":
            True,

        "ecos_estaticos_se_excluyen_del_ml":
            True,

        "recuperacion_de_ecos_nuevos":
            True,

        "nucleos":
            nucleos
    }

    SALIDA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        SALIDA,
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            resultado,
            archivo,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    print()
    print(
        "=== RESULTADO ==="
    )

    print(
        f"Originales: "
        f"{pixeles_originales}"
    )

    print(
        f"Estáticos eliminados: "
        f"{pixeles_estaticos}"
    )

    print(
        f"Dinámicos restantes: "
        f"{pixeles_dinamicos}"
    )

    print(
        f"Núcleos dinámicos: "
        f"{len(nucleos)}"
    )

    print(
        f"Estado: "
        f"{estado}"
    )

    print()

    for i, nucleo in enumerate(
        nucleos,
        start=1
    ):

        print(
            f"Núcleo {i}: "
            f"{nucleo['area_px']} px | "
            f"intensidad "
            f"{nucleo['intensidad_max']}/6 | "
            f"centro "
            f"("
            f"{nucleo['centroide_x']}, "
            f"{nucleo['centroide_y']}"
            f") | "
            f"{nucleo['tipo']}"
        )

    print()
    print(
        f"Guardado: {SALIDA}"
    )


if __name__ == "__main__":
    main()
