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

MIN_AREA_NUCLEO = 8
MIN_AREA_CELDA_GRANDE = 80
UMBRAL_REGION_GRANDE = 1500
DISTANCIA_MIN_SEMILLAS = 12
INTENSIDAD_MINIMA = 1

MAX_HISTORICOS = 40
MIN_FRAMES_ESTATICOS = 5
MIN_PRESENCIA_ESTATICO = 0.70
MAX_CAMBIO_INTENSIDAD_ESTATICO = 2

NIVELES = {
    0: "SIN_ECO",
    1: "DEBIL",
    2: "DEBIL",
    3: "MODERADO",
    4: "MODERADO",
    5: "FUERTE",
    6: "MUY_FUERTE",
}


# ============================================================
# CARGA
# ============================================================

def cargar_imagen(path):
    if isinstance(path, Image.Image):
        return np.array(path.convert("RGB"))

    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    return np.array(
        Image.open(path).convert("RGB")
    )


def cargar_mascara():
    if not MASCARA.exists():
        return None

    try:
        imagen = cargar_imagen(MASCARA)

        mascara = (
            np.mean(imagen, axis=2) > 100
        )

        return mascara

    except Exception:
        return None


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

    intensidad[
        (b > r + 20) &
        (b > g + 10)
    ] = 1

    intensidad[
        (g > r + 15) &
        (g > b + 10)
    ] = 2

    intensidad[
        (r > 140) &
        (g > 100) &
        (b < 120) &
        (r > b + 40)
    ] = 3

    intensidad[
        (r > 170) &
        (g > 60) &
        (g < 180) &
        (b < 100) &
        (r > g + 25)
    ] = 4

    intensidad[
        (r > 170) &
        (g < 90) &
        (b < 100) &
        (r > g + 70)
    ] = 5

    mascara_6 = (
        (r > 150) &
        (b > 80) &
        (g < 100) &
        (r > g + 50)
    )

    intensidad[mascara_6] = 6

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
            matrices.append(
                intensidad_radar(
                    cargar_imagen(archivo)
                )
            )
        except Exception as exc:
            print(
                f"No se pudo leer {archivo}: {exc}"
            )

    return matrices


# ============================================================
# FILTRO PIXEL A PIXEL DE ECOS ESTÁTICOS
# ============================================================

def calcular_mascara_estatica(
    actual,
    historicos
):
    if len(historicos) < MIN_FRAMES_ESTATICOS:
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
        cantidad_presente /
        float(len(historicos))
    )

    candidatos = (
        (actual > 0) &
        (presencia >= MIN_PRESENCIA_ESTATICO)
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

    media = np.mean(
        valores,
        axis=0
    )

    actual_valor = actual[
        candidatos
    ]

    diferencia = np.abs(
        actual_valor.astype(np.int16) -
        np.rint(media).astype(np.int16)
    )

    estatico = (
        (variacion <= MAX_CAMBIO_INTENSIDAD_ESTATICO) &
        (diferencia <= MAX_CAMBIO_INTENSIDAD_ESTATICO)
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


def aplicar_mascara_estatica(
    actual
):
    historicos = cargar_historicos()

    mascara = calcular_mascara_estatica(
        actual,
        historicos
    )

    resultado = actual.copy()

    resultado[mascara] = 0

    return (
        resultado,
        int(np.sum(mascara)),
        len(historicos)
    )


# ============================================================
# SEMILLAS
# ============================================================

def distancia_px(
    x1,
    y1,
    x2,
    y2
):
    return math.hypot(
        x2 - x1,
        y2 - y1
    )


def detectar_semillas(
    matriz,
    componente
):
    semillas = []

    maximo = ndimage.maximum_filter(
        matriz,
        size=15,
        mode="constant",
        cval=0
    )

    for nivel in (6, 5, 4, 3):

        candidatos = (
            componente &
            (matriz >= nivel) &
            (matriz == maximo)
        )

        if not np.any(candidatos):
            continue

        ys, xs = np.where(
            candidatos
        )

        puntos = []

        for y, x in zip(ys, xs):
            puntos.append(
                (
                    int(matriz[y, x]),
                    int(y),
                    int(x)
                )
            )

        puntos.sort(
            key=lambda p: (
                -p[0],
                p[1],
                p[2]
            )
        )

        for intensidad, y, x in puntos:

            cerca = False

            for _, sy, sx in semillas:
                if distancia_px(
                    x,
                    y,
                    sx,
                    sy
                ) < DISTANCIA_MIN_SEMILLAS:
                    cerca = True
                    break

            if not cerca:
                semillas.append(
                    (
                        intensidad,
                        y,
                        x
                    )
                )

    return semillas


# ============================================================
# SEPARACIÓN DE REGIONES GRANDES
# ============================================================

def separar_region(
    matriz,
    componente
):
    area = int(
        np.sum(componente)
    )

    if area < UMBRAL_REGION_GRANDE:
        return [componente]

    semillas = detectar_semillas(
        matriz,
        componente
    )

    if len(semillas) <= 1:
        return [componente]

    h, w = matriz.shape

    etiquetas = np.zeros(
        (h, w),
        dtype=np.int32
    )

    heap = []

    for indice, (_, y, x) in enumerate(
        semillas,
        start=1
    ):
        etiquetas[y, x] = indice

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

        _, distancia, y, x, etiqueta = (
            heapq.heappop(heap)
        )

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

            if not componente[ny, nx]:
                continue

            nueva = int(
                matriz[ny, nx]
            )

            if nueva <= 0:
                continue

            if etiquetas[ny, nx] != 0:
                continue

            salto = abs(
                intensidad_actual -
                nueva
            )

            nuevo_costo = (
                distancia +
                1 +
                salto * 100
            )

            etiquetas[ny, nx] = etiqueta

            heapq.heappush(
                heap,
                (
                    -nueva,
                    nuevo_costo,
                    ny,
                    nx,
                    etiqueta
                )
            )

    regiones = []

    for etiqueta in range(
        1,
        len(semillas) + 1
    ):

        region = (
            etiquetas == etiqueta
        )

        if int(np.sum(region)) >= MIN_AREA_CELDA_GRANDE:
            regiones.append(
                region
            )

    if not regiones:
        return [componente]

    return regiones


# ============================================================
# DETECCIÓN DE NÚCLEOS
# ============================================================

def detectar_nucleos(
    imagen
):
    matriz = intensidad_radar(
        cargar_imagen(imagen)
    )

    mascara = (
        matriz >= INTENSIDAD_MINIMA
    )

    estructura = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    etiquetas, cantidad = ndimage.label(
        mascara,
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

            ys, xs = np.where(
                region
            )

            if len(xs) < MIN_AREA_NUCLEO:
                continue

            valores = matriz[
                region
            ]

            intensidad = int(
                valores.max()
            )

            x_min = int(xs.min())
            x_max = int(xs.max())

            y_min = int(ys.min())
            y_max = int(ys.max())

            x = float(
                np.mean(xs)
            )

            y = float(
                np.mean(ys)
            )

            nucleos.append(
                {
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "pixeles": int(len(xs)),
                    "ancho": int(
                        x_max - x_min + 1
                    ),
                    "alto": int(
                        y_max - y_min + 1
                    ),
                    "intensidad": intensidad,
                    "intensidad_max": intensidad,
                    "intensidad_promedio": round(
                        float(np.mean(valores)),
                        3
                    ),
                    "nivel_intensidad":
                        NIVELES.get(
                            intensidad,
                            "DESCONOCIDO"
                        ),
                }
            )

    nucleos.sort(
        key=lambda n: (
            -n["pixeles"],
            -n["intensidad"]
        )
    )

    return nucleos


# ============================================================
# ANÁLISIS DE MÁSCARA
# ============================================================

def analizar_mascara(
    nucleo,
    mascara
):
    if mascara is None:
        return (
            "DINAMICO",
            0.70
        )

    try:
        x = int(
            round(nucleo["x"])
        )

        y = int(
            round(nucleo["y"])
        )

        ancho = max(
            int(nucleo["ancho"]),
            1
        )

        alto = max(
            int(nucleo["alto"]),
            1
        )

        h, w = mascara.shape

        x1 = max(
            0,
            x - ancho // 2
        )

        x2 = min(
            w,
            x + ancho // 2 + 1
        )

        y1 = max(
            0,
            y - alto // 2
        )

        y2 = min(
            h,
            y + alto // 2 + 1
        )

        region = mascara[
            y1:y2,
            x1:x2
        ]

        if region.size == 0:
            return (
                "DINAMICO",
                0.70
            )

        porcentaje = (
            float(np.mean(region))
        )

        if porcentaje >= 0.70:
            return (
                "ECO_ESTATICO",
                round(
                    min(
                        0.99,
                        0.70 +
                        porcentaje * 0.29
                    ),
                    3
                )
            )

        if porcentaje >= 0.40:
            return (
                "CAMBIO",
                0.65
            )

        return (
            "DINAMICO",
            0.75
        )

    except Exception:
        return (
            "DINAMICO",
            0.50
        )


# ============================================================
# CONFIANZA
# ============================================================

def calcular_confianza(
    nucleo,
    estado,
    confianza_base
):
    if estado == "ECO_ESTATICO":
        return 0.0

    intensidad = int(
        nucleo.get(
            "intensidad",
            0
        )
    )

    area = int(
        nucleo.get(
            "pixeles",
            0
        )
    )

    confianza = float(
        confianza_base
    )

    if intensidad >= 5:
        confianza += 0.08

    if area >= 100:
        confianza += 0.05

    return round(
        min(
            1.0,
            confianza
        ),
        3
    )


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

    imagen = cargar_imagen(
        ACTUAL
    )

    matriz_original = intensidad_radar(
        imagen
    )

    pixeles_originales = int(
        np.sum(
            matriz_original > 0
        )
    )

    matriz_dinamica, pixeles_estaticos, frames = (
        aplicar_mascara_estatica(
            matriz_original
        )
    )

    pixeles_dinamicos = int(
        np.sum(
            matriz_dinamica > 0
        )
    )

    # Detectar sobre la matriz ya filtrada.
    componentes = detectar_componentes(
        matriz_dinamica
    )

    nucleos = []

    for componente in componentes:

        regiones = separar_region(
            matriz_dinamica,
            componente
        )

        for region in regiones:

            ys, xs = np.where(
                region
            )

            if len(xs) < MIN_AREA_NUCLEO:
                continue

            valores = matriz_dinamica[
                region
            ]

            intensidad = int(
                valores.max()
            )

            nucleos.append(
                {
                    "x": round(
                        float(np.mean(xs)),
                        2
                    ),
                    "y": round(
                        float(np.mean(ys)),
                        2
                    ),
                    "pixeles": int(
                        len(xs)
                    ),
                    "ancho": int(
                        xs.max() -
                        xs.min() + 1
                    ),
                    "alto": int(
                        ys.max() -
                        ys.min() + 1
                    ),
                    "intensidad": intensidad,
                    "intensidad_max": intensidad,
                    "intensidad_promedio": round(
                        float(np.mean(valores)),
                        3
                    ),
                    "nivel_intensidad":
                        NIVELES.get(
                            intensidad,
                            "DESCONOCIDO"
                        ),
                    "estado": "DINAMICO",
                }
            )

    nucleos.sort(
        key=lambda n: (
            -n["pixeles"],
            -n["intensidad"]
        )
    )

    if not nucleos:
        estado_general = "SIN_ECOS_DINAMICOS"

    else:
        intensidad_max = max(
            n["intensidad"]
            for n in nucleos
        )

        if intensidad_max >= 6:
            estado_general = (
                "PRECIPITACION_DINAMICA_MUY_FUERTE"
            )
        elif intensidad_max >= 5:
            estado_general = (
                "PRECIPITACION_DINAMICA_FUERTE"
            )
        elif intensidad_max >= 4:
            estado_general = (
                "PRECIPITACION_DINAMICA_MODERADA"
            )
        else:
            estado_general = (
                "PRECIPITACION_DINAMICA"
            )

    resultado = {
        "version":
            "RMA10-procesador-9.1",

        "timestamp_procesamiento":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "fuente":
            "SMN RMA10",

        "metodo_segmentacion":
            "filtro_estatico_pixelwise + conectividad + semillas + priority flood",

        "frames_historicos_utilizados":
            frames,

        "pixeles_con_eco_original":
            pixeles_originales,

        "pixeles_ecos_estaticos_eliminados":
            pixeles_estaticos,

        "pixeles_identificados_como_estaticos":
            pixeles_estaticos,

        "pixeles_con_eco_final":
            pixeles_dinamicos,

        "pixeles_dinamicos":
            pixeles_dinamicos,

        "nucleos_estaticos_excluidos":
            pixeles_estaticos,

        "nucleos_detectados":
            len(nucleos),

        "estado_general":
            estado_general,

        "ecos_estaticos_se_excluyen_antes_del_tracking":
            True,

        "ecos_estaticos_se_excluyen_del_ml":
            True,

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
        f"{estado_general}"
    )

    print()
    print(
        "=== NÚCLEOS ==="
    )

    for i, nucleo in enumerate(
        nucleos,
        start=1
    ):
        print(
            f"{i}: "
            f"{nucleo['pixeles']} px | "
            f"intensidad "
            f"{nucleo['intensidad']}/6 | "
            f"centro "
            f"({nucleo['x']}, "
            f"{nucleo['y']})"
        )

    print()
    print(
        f"Guardado: {SALIDA}"
    )


# ============================================================
# COMPATIBILIDAD
# ============================================================

def detectar_componentes(matriz):
    mascara = (
        matriz >= INTENSIDAD_MINIMA
    )

    estructura = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    etiquetas, cantidad = ndimage.label(
        mascara,
        structure=estructura
    )

    componentes = []

    for etiqueta in range(
        1,
        cantidad + 1
    ):
        region = (
            etiquetas == etiqueta
        )

        if int(np.sum(region)) >= MIN_AREA_NUCLEO:
            componentes.append(
                region
            )

    return componentes


if __name__ == "__main__":
    main()
