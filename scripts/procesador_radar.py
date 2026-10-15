import json
import math
import heapq
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

# ============================================================
# CONFIGURACIÓN
# ============================================================

MIN_AREA_NUCLEO = 8
MIN_AREA_CELDA_GRANDE = 80

# Una región grande se subdivide internamente.
UMBRAL_REGION_GRANDE = 1500

# Distancia mínima entre semillas.
DISTANCIA_MIN_SEMILLAS = 12

# Intensidad mínima considerada precipitación.
INTENSIDAD_MINIMA = 1

# Intensidades del radar.
# 0 = sin eco
# 1-2 = débil
# 3-4 = moderado
# 5-6 = fuerte / muy fuerte


# ============================================================
# UTILIDADES
# ============================================================

def cargar_imagen(path):
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    img = Image.open(path).convert("RGB")
    return np.array(img)


def intensidad_radar(rgb):
    """
    Convierte colores del radar RMA10 a una escala 0-6.

    La clasificación es deliberadamente conservadora.
    """

    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    intensidad = np.zeros(r.shape, dtype=np.uint8)

    # Azules / celestes
    intensidad[((b > r + 20) & (b > g + 10))] = 1

    # Verdes
    intensidad[
        (g > r + 15)
        & (g > b + 10)
    ] = 2

    # Amarillos
    intensidad[
        (r > 140)
        & (g > 100)
        & (b < 120)
        & (r > b + 40)
    ] = 3

    # Naranjas
    intensidad[
        (r > 170)
        & (g > 60)
        & (g < 180)
        & (b < 100)
        & (r > g + 25)
    ] = 4

    # Rojos
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
        6,
    )

    return intensidad


def distancia_px(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


# ============================================================
# MÁSCARA DE ECOS ESTÁTICOS
# ============================================================

def obtener_historico():
    if not HISTORICO.exists():
        return []

    archivos = sorted(
        HISTORICO.glob("*.png"),
        key=lambda p: p.stat().st_mtime
    )

    return archivos[-40:]


def aplicar_mascara_estatica(actual_intensidad):
    """
    La máscara NO elimina automáticamente un eco.

    Solo elimina un píxel si:
    - está dentro de la máscara histórica
    - aparece de forma persistente
    - prácticamente no cambia de intensidad

    De esta forma un eco nuevo puede recuperar la zona.
    """

    if not MASCARA.exists():
        return actual_intensidad.copy(), 0

    try:
        mascara_rgb = cargar_imagen(MASCARA)

        mascara = (
            np.mean(mascara_rgb, axis=2) > 100
        )

        historicos = obtener_historico()

        if len(historicos) < 3:
            return actual_intensidad.copy(), 0

        matrices = []

        for archivo in historicos:
            try:
                img = cargar_imagen(archivo)
                matrices.append(intensidad_radar(img))
            except Exception:
                continue

        if len(matrices) < 3:
            return actual_intensidad.copy(), 0

        hist = np.stack(matrices)

        # Solo nos interesan píxeles con eco actual.
        candidatos = mascara & (actual_intensidad > 0)

        if not np.any(candidatos):
            return actual_intensidad.copy(), 0

        valores = hist[:, candidatos]

        # Cantidad de frames donde hubo eco.
        presentes = np.sum(valores > 0, axis=0)

        # Variación histórica.
        minimo = np.min(valores, axis=0)
        maximo = np.max(valores, axis=0)

        estable = (
            (presentes >= 3)
            & ((maximo - minimo) <= 1)
        )

        indices = np.flatnonzero(candidatos)

        eliminar = indices[estable]

        resultado = actual_intensidad.copy()
        resultado.flat[eliminar] = 0

        return resultado, int(len(eliminar))

    except Exception:
        return actual_intensidad.copy(), 0


# ============================================================
# DETECCIÓN DE SEMILLAS
# ============================================================

def detectar_semillas(matriz, componente_mask):
    """
    Busca núcleos internos de intensidad.

    Prioridad:
    6 -> 5 -> 4 -> 3

    Se evita colocar varias semillas demasiado cerca.
    """

    semillas = []

    for nivel in (6, 5, 4, 3):
        candidatos = (
            componente_mask
            & (matriz >= nivel)
        )

        if not np.any(candidatos):
            continue

        # Máximo local.
        maximo = ndimage.maximum_filter(
            matriz,
            size=15,
            mode="constant",
            cval=0
        )

        peaks = (
            candidatos
            & (matriz == maximo)
        )

        ys, xs = np.where(peaks)

        puntos = []

        for y, x in zip(ys, xs):
            puntos.append(
                (
                    int(matriz[y, x]),
                    int(y),
                    int(x)
                )
            )

        # Primero los más intensos.
        puntos.sort(
            key=lambda p: (-p[0], p[1], p[2])
        )

        for intensidad, y, x in puntos:

            demasiado_cerca = False

            for _, sy, sx in semillas:
                if distancia_px(x, y, sx, sy) < DISTANCIA_MIN_SEMILLAS:
                    demasiado_cerca = True
                    break

            if not demasiado_cerca:
                semillas.append(
                    (intensidad, y, x)
                )

    return semillas


# ============================================================
# SEPARACIÓN INTERNA
# ============================================================

def separar_region(matriz, componente_mask):
    """
    Divide una región grande en células independientes.

    No utiliza simplemente conectividad.

    Cada semilla comienza en el máximo de intensidad
    y la región crece priorizando:
      1. mayor intensidad
      2. menor caída de intensidad
      3. cercanía

    Esto evita que un puente débil fusione dos núcleos fuertes.
    """

    area = int(np.sum(componente_mask))

    if area < UMBRAL_REGION_GRANDE:
        return [
            componente_mask
        ]

    semillas = detectar_semillas(
        matriz,
        componente_mask
    )

    if len(semillas) <= 1:
        return [
            componente_mask
        ]

    h, w = matriz.shape

    etiquetas = np.zeros(
        (h, w),
        dtype=np.int32
    )

    # --------------------------------------------------------
    # Crear regiones iniciales
    # --------------------------------------------------------

    for indice, (_, y, x) in enumerate(semillas, start=1):
        etiquetas[y, x] = indice

    # --------------------------------------------------------
    # Priority flood
    # --------------------------------------------------------

    heap = []

    for indice, (_, y, x) in enumerate(semillas, start=1):
        heapq.heappush(
            heap,
            (
                -int(matriz[y, x]),
                0,
                y,
                x,
                indice
            )
        )

    vecinos = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    while heap:

        prioridad_intensidad, distancia, y, x, etiqueta = heapq.heappop(heap)

        intensidad_actual = int(
            matriz[y, x]
        )

        for dy, dx in vecinos:

            ny = y + dy
            nx = x + dx

            if ny < 0 or ny >= h:
                continue

            if nx < 0 or nx >= w:
                continue

            if not componente_mask[ny, nx]:
                continue

            nueva_intensidad = int(
                matriz[ny, nx]
            )

            if nueva_intensidad <= 0:
                continue

            # No cruzar grandes saltos de intensidad.
            salto = abs(
                intensidad_actual -
                nueva_intensidad
            )

            penalizacion = salto * 100

            nueva_distancia = (
                distancia
                + 1
                + penalizacion
            )

            if etiquetas[ny, nx] == 0:

                etiquetas[ny, nx] = etiqueta

                heapq.heappush(
                    heap,
                    (
                        -nueva_intensidad,
                        nueva_distancia,
                        ny,
                        nx,
                        etiqueta
                    )
                )

            else:
                # Si ya pertenece a otra célula,
                # no se fusionan las dos regiones.
                continue

    # --------------------------------------------------------
    # Convertir etiquetas a máscaras
    # --------------------------------------------------------

    regiones = []

    for etiqueta in range(
        1,
        len(semillas) + 1
    ):

        region = (
            etiquetas == etiqueta
        )

        area_region = int(
            np.sum(region)
        )

        if area_region >= MIN_AREA_CELDA_GRANDE:
            regiones.append(region)

    # Si la segmentación produjo algo extraño,
    # conservar la región original.
    if not regiones:
        return [
            componente_mask
        ]

    # --------------------------------------------------------
    # Recuperar píxeles que hayan quedado fuera.
    # --------------------------------------------------------

    cubierto = np.zeros(
        componente_mask.shape,
        dtype=bool
    )

    for region in regiones:
        cubierto |= region

    sobrantes = (
        componente_mask
        & ~cubierto
    )

    if np.any(sobrantes):

        # Asignar sobrantes al núcleo más cercano.
        distancia_mas_corta = None
        indice_mejor = None

        for i, region in enumerate(regiones):

            distancia, _ = ndimage.distance_transform_edt(
                ~region,
                return_indices=True
            )

            if distancia_mas_corta is None:
                distancia_mas_corta = distancia
                indice_mejor = np.full(
                    distancia.shape,
                    i,
                    dtype=np.int32
                )
            else:
                mejor = distancia < distancia_mas_corta

                distancia_mas_corta[mejor] = distancia[mejor]
                indice_mejor[mejor] = i

        ys, xs = np.where(sobrantes)

        for y, x in zip(ys, xs):
            regiones[
                int(indice_mejor[y, x])
            ][y, x] = True

    return regiones


# ============================================================
# DETECCIÓN DE NÚCLEOS
# ============================================================

def detectar_nucleos(matriz):
    """
    Primera separación por conectividad.

    Después, las regiones grandes se subdividen
    internamente mediante intensidad.
    """

    mascara_precipitacion = (
        matriz >= INTENSIDAD_MINIMA
    )

    estructura = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    etiquetas, cantidad = ndimage.label(
        mascara_precipitacion,
        structure=estructura
    )

    nucleos = []

    for etiqueta in range(
        1,
        cantidad + 1
    ):

        componente = (
            etiquetas == etiqueta
        )

        area = int(
            np.sum(componente)
        )

        if area < MIN_AREA_NUCLEO:
            continue

        regiones = separar_region(
            matriz,
            componente
        )

        for region in regiones:

            area_region = int(
                np.sum(region)
            )

            if area_region < MIN_AREA_NUCLEO:
                continue

            ys, xs = np.where(region)

            if len(xs) == 0:
                continue

            x_min = int(xs.min())
            x_max = int(xs.max())
            y_min = int(ys.min())
            y_max = int(ys.max())

            valores = matriz[region]

            intensidad_max = int(
                valores.max()
            )

            intensidad_promedio = float(
                np.mean(valores)
            )

            x_centro = float(
                np.mean(xs)
            )

            y_centro = float(
                np.mean(ys)
            )

            nucleos.append(
                {
                    "area_px": area_region,
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                    "width_px": x_max - x_min + 1,
                    "height_px": y_max - y_min + 1,
                    "centroide_x": round(
                        x_centro,
                        2
                    ),
                    "centroide_y": round(
                        y_centro,
                        2
                    ),
                    "intensidad_max": intensidad_max,
                    "intensidad_promedio": round(
                        intensidad_promedio,
                        3
                    ),
                }
            )

    # Más grandes primero.
    nucleos.sort(
        key=lambda n: (
            -n["area_px"],
            -n["intensidad_max"]
        )
    )

    return nucleos


# ============================================================
# ESTADO GENERAL
# ============================================================

def calcular_estado(nucleos, matriz):

    if not nucleos:
        return "SIN_ECOS"

    intensidad_max = max(
        n["intensidad_max"]
        for n in nucleos
    )

    if intensidad_max >= 6:
        return "PRECIPITACION_MUY_FUERTE"

    if intensidad_max >= 5:
        return "PRECIPITACION_FUERTE"

    if intensidad_max >= 4:
        return "PRECIPITACION_MODERADA"

    return "PRECIPITACION"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=== ClimaAR - Procesador RMA10 ===")

    if not ACTUAL.exists():
        raise FileNotFoundError(
            f"No existe {ACTUAL}"
        )

    rgb = cargar_imagen(
        ACTUAL
    )

    matriz_original = intensidad_radar(
        rgb
    )

    pixeles_originales = int(
        np.sum(matriz_original > 0)
    )

    matriz, pixeles_estaticos_eliminados = (
        aplicar_mascara_estatica(
            matriz_original
        )
    )

    pixeles_finales = int(
        np.sum(matriz > 0)
    )

    nucleos = detectar_nucleos(
        matriz
    )

    estado = calcular_estado(
        nucleos,
        matriz
    )

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    resultado = {
        "version": "RMA10-procesador-7.0",

        "timestamp_procesamiento": timestamp,

        "fuente": "SMN RMA10",

        "metodo_segmentacion":
            "conectividad + semillas de intensidad + priority flood",

        "imagen": str(ACTUAL),

        "pixeles_con_eco_original":
            pixeles_originales,

        "pixeles_ecos_estaticos_eliminados":
            pixeles_estaticos_eliminados,

        "pixeles_con_eco_final":
            pixeles_finales,

        "nucleos_detectados":
            len(nucleos),

        "estado_general":
            estado,

        "nucleos_estaticos_excluidos":
            0,

        "recuperacion_de_ecos_nuevos":
            True,

        "nucleos":
            nucleos,
    }

    SALIDA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        SALIDA,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            resultado,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        f"Pixeles originales: "
        f"{pixeles_originales}"
    )

    print(
        f"Ecos estáticos eliminados: "
        f"{pixeles_estaticos_eliminados}"
    )

    print(
        f"Pixeles finales: "
        f"{pixeles_finales}"
    )

    print(
        f"Núcleos detectados: "
        f"{len(nucleos)}"
    )

    print(
        f"Estado: {estado}"
    )

    print()
    print("=== NÚCLEOS ===")

    for i, nucleo in enumerate(
        nucleos,
        start=1
    ):

        print(
            f"{i}: "
            f"{nucleo['area_px']} px | "
            f"intensidad {nucleo['intensidad_max']}/6 | "
            f"centro "
            f"({nucleo['centroide_x']}, "
            f"{nucleo['centroide_y']})"
        )

    print()
    print(
        f"Resultado guardado en: {SALIDA}"
    )


if __name__ == "__main__":
    main()
