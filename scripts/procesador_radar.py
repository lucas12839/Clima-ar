import json
import math
import heapq
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from scipy import ndimage


# ============================================================
# RUTAS
# ============================================================

BASE = Path("data/radar")

ACTUAL = BASE / "actual.png"

MASCARA = BASE / "mascara_ecos_estaticos.png"

MASCARA_JSON = BASE / "mascara_ecos_estaticos.json"

SALIDA = BASE / "resultado_procesamiento.json"


# ============================================================
# CONFIGURACIÓN
# ============================================================

MIN_AREA_NUCLEO = 8

MIN_AREA_CELDA_GRANDE = 80

UMBRAL_REGION_GRANDE = 1500

DISTANCIA_MIN_SEMILLAS = 12

INTENSIDAD_MINIMA = 1


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
# CARGA DE IMAGEN
# ============================================================

def cargar_imagen(path):

    if isinstance(path, Image.Image):

        return np.array(
            path.convert("RGB")
        )

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"No existe: {path}"
        )

    return np.array(
        Image.open(path).convert("RGB")
    )


# ============================================================
# INTENSIDAD RMA10
# ============================================================

def intensidad_radar(rgb):

    if rgb.ndim != 3:

        raise ValueError(
            "La imagen radar no tiene 3 dimensiones."
        )

    if rgb.shape[2] != 3:

        raise ValueError(
            "La imagen radar no tiene 3 canales RGB."
        )

    r = rgb[:, :, 0].astype(np.int16)

    g = rgb[:, :, 1].astype(np.int16)

    b = rgb[:, :, 2].astype(np.int16)


    intensidad = np.zeros(
        r.shape,
        dtype=np.uint8
    )


    # Nivel 1
    intensidad[
        (b > r + 20) &
        (b > g + 10)
    ] = 1


    # Nivel 2
    intensidad[
        (g > r + 15) &
        (g > b + 10)
    ] = 2


    # Nivel 3
    intensidad[
        (r > 140) &
        (g > 100) &
        (b < 120) &
        (r > b + 40)
    ] = 3


    # Nivel 4
    intensidad[
        (r > 170) &
        (g > 60) &
        (g < 180) &
        (b < 100) &
        (r > g + 25)
    ] = 4


    # Nivel 5
    intensidad[
        (r > 170) &
        (g < 90) &
        (b < 100) &
        (r > g + 70)
    ] = 5


    # Nivel 6
    mascara_6 = (
        (r > 150) &
        (b > 80) &
        (g < 100) &
        (r > g + 50)
    )

    intensidad[
        mascara_6
    ] = 6


    return intensidad


# ============================================================
# CARGAR MÁSCARA ESTÁTICA
# ============================================================

def cargar_mascara():

    if not MASCARA.exists():

        print(
            "AVISO: no existe la máscara estática."
        )

        return None


    try:

        imagen = cargar_imagen(
            MASCARA
        )


        mascara = (
            np.mean(
                imagen,
                axis=2
            ) > 100
        )


        return mascara


    except Exception as exc:

        print(
            "AVISO: no se pudo cargar "
            f"la máscara estática: {exc}"
        )

        return None


# ============================================================
# INFORMACIÓN DE MÁSCARA
# ============================================================

def cargar_info_mascara():

    if not MASCARA_JSON.exists():

        return {}


    try:

        with open(
            MASCARA_JSON,
            "r",
            encoding="utf-8"
        ) as f:

            datos = json.load(f)


        if isinstance(
            datos,
            dict
        ):

            return datos


    except Exception as exc:

        print(
            "AVISO: no se pudo leer "
            f"información de máscara: {exc}"
        )


    return {}


# ============================================================
# APLICAR MÁSCARA ESTÁTICA
# ============================================================

def aplicar_mascara_estatica(
    actual
):

    """
    Utiliza exclusivamente la máscara temporal
    generada por generar_mascara_estatica.py.

    NO vuelve a analizar los históricos.

    Esto evita:

    - problemas entre imágenes de distinto tamaño
    - np.stack() con arrays incompatibles
    - doble filtrado de ecos
    - eliminar nuevamente un eco dinámico
    """


    mascara = cargar_mascara()


    if mascara is None:

        print(
            "AVISO: se procesa el radar "
            "sin máscara estática."
        )

        return (
            actual.copy(),
            0,
            0
        )


    if mascara.shape != actual.shape:

        print(
            "AVISO: tamaño de máscara "
            "diferente al radar actual."
        )

        print(
            "Máscara:",
            mascara.shape
        )

        print(
            "Radar:",
            actual.shape
        )

        print(
            "Se ignora la máscara para "
            "no producir un error."
        )

        return (
            actual.copy(),
            0,
            0
        )


    estaticos = (
        mascara &
        (actual > 0)
    )


    resultado = actual.copy()


    resultado[
        estaticos
    ] = 0


    info = cargar_info_mascara()


    frames = int(
        info.get(
            "frames_analizados",
            0
        )
    )


    cantidad = int(
        np.sum(
            estaticos
        )
    )


    print(
        "Máscara estática externa utilizada."
    )


    print(
        "Píxeles estáticos eliminados:",
        cantidad
    )


    print(
        "Frames utilizados para máscara:",
        frames
    )


    return (
        resultado,
        cantidad,
        frames
    )


# ============================================================
# DISTANCIA
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


# ============================================================
# DETECTAR SEMILLAS
# ============================================================

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


    for nivel in (
        6,
        5,
        4,
        3
    ):

        candidatos = (
            componente &
            (matriz >= nivel) &
            (matriz == maximo)
        )


        if not np.any(
            candidatos
        ):

            continue


        ys, xs = np.where(
            candidatos
        )


        puntos = []


        for y, x in zip(
            ys,
            xs
        ):

            puntos.append(
                (
                    int(
                        matriz[y, x]
                    ),
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
# SEPARAR REGIONES GRANDES
# ============================================================

def separar_region(
    matriz,
    componente
):

    area = int(
        np.sum(
            componente
        )
    )


    if area < UMBRAL_REGION_GRANDE:

        return [
            componente
        ]


    semillas = detectar_semillas(
        matriz,
        componente
    )


    if len(semillas) <= 1:

        return [
            componente
        ]


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

        etiquetas[
            y,
            x
        ] = indice


        heapq.heappush(
            heap,
            (
                -int(
                    matriz[y, x]
                ),
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

        _,
        distancia,
        y,
        x,
        etiqueta = heapq.heappop(
            heap
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


            if not componente[
                ny,
                nx
            ]:

                continue


            nueva = int(
                matriz[
                    ny,
                    nx
                ]
            )


            if nueva <= 0:

                continue


            if etiquetas[
                ny,
                nx
            ] != 0:

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


            etiquetas[
                ny,
                nx
            ] = etiqueta


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
            etiquetas ==
            etiqueta
        )


        if int(
            np.sum(region)
        ) >= MIN_AREA_CELDA_GRANDE:

            regiones.append(
                region
            )


    if not regiones:

        return [
            componente
        ]


    return regiones


# ============================================================
# DETECTAR COMPONENTES
# ============================================================

def detectar_componentes(
    matriz
):

    mascara = (
        matriz >=
        INTENSIDAD_MINIMA
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

        componente = (
            etiquetas ==
            etiqueta
        )


        area = int(
            np.sum(
                componente
            )
        )


        if area < MIN_AREA_NUCLEO:

            continue


        componentes.append(
            componente
        )


    return componentes


# ============================================================
# CREAR NÚCLEO
# ============================================================

def crear_nucleo(
    matriz,
    region
):

    ys, xs = np.where(
        region
    )


    if len(xs) < MIN_AREA_NUCLEO:

        return None


    valores = matriz[
        region
    ]


    intensidad = int(
        valores.max()
    )


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


    x = float(
        np.mean(xs)
    )

    y = float(
        np.mean(ys)
    )


    return {

        "x":
            round(
                x,
                2
            ),

        "y":
            round(
                y,
                2
            ),

        "pixeles":
            int(
                len(xs)
            ),

        "ancho":
            int(
                x_max -
                x_min +
                1
            ),

        "alto":
            int(
                y_max -
                y_min +
                1
            ),

        "intensidad":
            intensidad,

        "intensidad_max":
            intensidad,

        "intensidad_promedio":
            round(
                float(
                    np.mean(
                        valores
                    )
                ),
                3
            ),

        "nivel_intensidad":
            NIVELES.get(
                intensidad,
                "DESCONOCIDO"
            ),

        "estado":
            "DINAMICO",
    }


# ============================================================
# DETECTAR NÚCLEOS
# ============================================================

def detectar_nucleos(
    matriz
):

    componentes = detectar_componentes(
        matriz
    )


    nucleos = []


    for componente in componentes:

        regiones = separar_region(
            matriz,
            componente
        )


        for region in regiones:

            nucleo = crear_nucleo(
                matriz,
                region
            )


            if nucleo is not None:

                nucleos.append(
                    nucleo
                )


    nucleos.sort(
        key=lambda n: (
            -n["pixeles"],
            -n["intensidad"]
        )
    )


    return nucleos


# ============================================================
# ANALIZAR MÁSCARA DE NÚCLEO
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
            round(
                nucleo["x"]
            )
        )

        y = int(
            round(
                nucleo["y"]
            )
        )


        ancho = max(
            int(
                nucleo["ancho"]
            ),
            1
        )

        alto = max(
            int(
                nucleo["alto"]
            ),
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


        porcentaje = float(
            np.mean(
                region
            )
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
# ESTADO GENERAL
# ============================================================

def calcular_estado_general(
    nucleos
):

    if not nucleos:

        return (
            "SIN_ECOS_DINAMICOS"
        )


    intensidad_max = max(
        n["intensidad"]
        for n in nucleos
    )


    if intensidad_max >= 6:

        return (
            "PRECIPITACION_DINAMICA_MUY_FUERTE"
        )


    if intensidad_max >= 5:

        return (
            "PRECIPITACION_DINAMICA_FUERTE"
        )


    if intensidad_max >= 4:

        return (
            "PRECIPITACION_DINAMICA_MODERADA"
        )


    return (
        "PRECIPITACION_DINAMICA"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=== ClimaAR - Procesador RMA10 ==="
    )


    # --------------------------------------------------------
    # Comprobar radar
    # --------------------------------------------------------

    if not ACTUAL.exists():

        raise FileNotFoundError(
            f"No existe {ACTUAL}"
        )


    # --------------------------------------------------------
    # Cargar radar actual
    # --------------------------------------------------------

    imagen = cargar_imagen(
        ACTUAL
    )


    print(
        "Tamaño radar actual:",
        imagen.shape
    )


    matriz_original = intensidad_radar(
        imagen
    )


    pixeles_originales = int(
        np.sum(
            matriz_original > 0
        )
    )


    # --------------------------------------------------------
    # Aplicar máscara externa
    # --------------------------------------------------------

    (
        matriz_dinamica,
        pixeles_estaticos,
        frames
    ) = aplicar_mascara_estatica(
        matriz_original
    )


    pixeles_dinamicos = int(
        np.sum(
            matriz_dinamica > 0
        )
    )


    # --------------------------------------------------------
    # Detectar núcleos
    # --------------------------------------------------------

    nucleos = detectar_nucleos(
        matriz_dinamica
    )


    # --------------------------------------------------------
    # Analizar estado de cada núcleo
    # --------------------------------------------------------

    mascara = cargar_mascara()


    for nucleo in nucleos:

        estado,
        confianza_base = analizar_mascara(
            nucleo,
            mascara
        )


        confianza = calcular_confianza(
            nucleo,
            estado,
            confianza_base
        )


        nucleo[
            "estado"
        ] = estado


        nucleo[
            "confianza"
        ] = confianza


    # --------------------------------------------------------
    # El resultado principal se basa en núcleos dinámicos
    # --------------------------------------------------------

    nucleos_dinamicos = [

        nucleo

        for nucleo in nucleos

        if nucleo.get(
            "estado"
        ) != "ECO_ESTATICO"

    ]


    estado_general = calcular_estado_general(
        nucleos_dinamicos
    )


    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    resultado = {

        "version":
            "RMA10-procesador-10.0",

        "timestamp_procesamiento":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "fuente":
            "SMN RMA10",

        "metodo_segmentacion":
            (
                "mascara_estatica_temporal_externa"
                "_conectividad_semillas"
                "_priority_flood"
            ),

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
            sum(
                1
                for n in nucleos
                if n.get(
                    "estado"
                ) == "ECO_ESTATICO"
            ),

        "nucleos_detectados":
            len(nucleos),

        "nucleos_dinamicos":
            len(nucleos_dinamicos),

        "estado_general":
            estado_general,

        "ecos_estaticos_se_excluyen_antes_del_tracking":
            True,

        "ecos_estaticos_se_excluyen_del_ml":
            True,

        "recuperacion_de_ecos_nuevos":
            True,

        "mascara_estatica_externa":
            True,

        "historiales_reprocesados":
            False,

        "nucleos":
            nucleos,
    }


    # --------------------------------------------------------
    # Guardar resultado
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Consola
    # --------------------------------------------------------

    print()

    print(
        "Pixeles originales:",
        pixeles_originales
    )

    print(
        "Estáticos eliminados:",
        pixeles_estaticos
    )

    print(
        "Pixeles dinámicos:",
        pixeles_dinamicos
    )

    print(
        "Núcleos detectados:",
        len(nucleos)
    )

    print(
        "Núcleos dinámicos:",
        len(nucleos_dinamicos)
    )

    print(
        "Estado general:",
        estado_general
    )

    print()

    print(
        "Procesamiento RMA10 completado."
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    main()
