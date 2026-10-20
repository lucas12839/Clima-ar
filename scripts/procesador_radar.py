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
MASCARA_JSON = BASE / "mascara_ecos_estaticos.json"
SALIDA = BASE / "resultado_procesamiento.json"


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


def intensidad_radar(rgb):

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            "La imagen radar no tiene formato RGB valido."
        )

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

    intensidad[
        (r > 150) &
        (b > 80) &
        (g < 100) &
        (r > g + 50)
    ] = 6

    return intensidad


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

        return (
            np.mean(
                imagen,
                axis=2
            ) > 100
        )

    except Exception as exc:

        print(
            "AVISO: no se pudo cargar "
            f"información de máscara: {exc}"
        )

        return None


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


TEMPORAL_MAX_FRAMES = 3
TEMPORAL_MAX_GAP_MINUTOS = 60.0
TEMPORAL_DELTA_INTENSIDAD = 2
TEMPORAL_NUEVO_ECO_MIN_FRAMES = 2


def _parse_timestamp_text(texto):

    if texto is None:
        return None

    texto = str(texto).strip()

    if not texto:
        return None

    if texto.endswith("Z"):
        texto = texto[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(texto)
        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone.utc
            )
        return dt.astimezone(
            timezone.utc
        )
    except ValueError:
        pass

    formatos = (
        "%Y%m%d%H%M%S",
        "%Y%m%d_%H%M%S",
        "%Y-%m-%dT%H_%M_%S+00_00",
        "%Y-%m-%dT%H_%M_%S",
        "%Y-%m-%dT%H-%M-%S",
    )

    for formato in formatos:
        try:
            return datetime.strptime(
                texto,
                formato
            ).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue

    if texto.isdigit() and len(texto) == 10:
        try:
            dt = datetime.fromtimestamp(
                int(texto),
                tz=timezone.utc
            )
            if 2000 <= dt.year <= 2100:
                return dt
        except (OverflowError, OSError, ValueError):
            pass

    return None


def _timestamp_de_path(path):

    try:
        path = Path(path)
    except Exception:
        return None

    return _parse_timestamp_text(
        path.stem
    )


def _timestamp_actual():

    timestamp_path = (
        BASE /
        "radar_timestamp.txt"
    )

    if not timestamp_path.exists():
        return None

    try:
        return _parse_timestamp_text(
            timestamp_path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None


def _frames_previos(ruta_actual=None):

    if ruta_actual is None:
        objetivo = _timestamp_actual()
    else:
        objetivo = _timestamp_de_path(
            ruta_actual
        )

        if objetivo is None:
            objetivo = _timestamp_actual()

    if objetivo is None:
        return []

    candidatos = {}

    for carpeta in (
        BASE / "historico",
        BASE / "historico_tracking_backup",
    ):
        if not carpeta.exists():
            continue

        for path in carpeta.glob("*.png"):
            stamp = _timestamp_de_path(
                path
            )

            if stamp is None:
                continue

            if stamp >= objetivo:
                continue

            diferencia = (
                objetivo - stamp
            ).total_seconds() / 60.0

            if (
                diferencia <= 0
                or diferencia > TEMPORAL_MAX_GAP_MINUTOS
            ):
                continue

            clave = stamp.isoformat()
            anterior = candidatos.get(
                clave
            )

            if anterior is None:
                candidatos[clave] = (
                    stamp,
                    path
                )

    salida = list(
        candidatos.values()
    )

    salida.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        path
        for _, path in salida[
            :TEMPORAL_MAX_FRAMES
        ]
    ]


def _cargar_intensidades_previas(
    actual,
    ruta_actual=None
):

    previas = []

    for path in _frames_previos(
        ruta_actual
    ):
        try:
            rgb = cargar_imagen(
                path
            )
            matriz = intensidad_radar(
                rgb
            )

            if matriz.shape != actual.shape:
                continue

            previas.append(
                matriz
            )
        except Exception as exc:
            print(
                "AVISO: no se pudo usar "
                f"frame previo {path}: {exc}"
            )

    return previas


def _mascara_temporal_dinamica(
    actual,
    mascara,
    ruta_actual=None
):

    if mascara is None:
        return np.zeros(
            actual.shape,
            dtype=bool
        ), 0

    previas = _cargar_intensidades_previas(
        actual,
        ruta_actual
    )

    if not previas:
        return np.zeros(
            actual.shape,
            dtype=bool
        ), 0

    apilado = np.stack(
        previas,
        axis=0
    )

    presentes = (
        apilado > 0
    )

    cantidad_presente = np.sum(
        presentes,
        axis=0
    )

    mediana_previa = np.median(
        apilado,
        axis=0
    )

    maximo_previo = np.max(
        apilado,
        axis=0
    )

    actual_presente = (
        actual > 0
    )

    nuevo_eco = (
        actual_presente &
        (
            cantidad_presente
            <= max(
                0,
                len(previas) - TEMPORAL_NUEVO_ECO_MIN_FRAMES
            )
        )
    )

    cambio_intensidad = (
        actual_presente &
        (
            np.abs(
                actual.astype(np.int16)
                - mediana_previa.astype(np.int16)
            ) >= TEMPORAL_DELTA_INTENSIDAD
        )
    )

    intensificacion = (
        actual_presente &
        (
            actual.astype(np.int16)
            >= maximo_previo.astype(np.int16)
            + TEMPORAL_DELTA_INTENSIDAD
        )
    )

    evidencia = (
        mascara &
        (
            nuevo_eco |
            cambio_intensidad |
            intensificacion
        )
    )

    return (
        evidencia,
        len(previas)
    )


def aplicar_mascara_estatica(
    actual,
    ruta_actual=None
):

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

    evidencia_temporal, frames_temporales = (
        _mascara_temporal_dinamica(
            actual,
            mascara,
            ruta_actual
        )
    )

    eliminar = (
        estaticos &
        ~evidencia_temporal
    )

    resultado = actual.copy()

    resultado[
        eliminar
    ] = 0

    info = cargar_info_mascara()

    frames_mascara = int(
        info.get(
            "frames_analizados",
            0
        )
    )

    cantidad = int(
        np.sum(
            eliminar
        )
    )

    rescatados = int(
        np.sum(
            evidencia_temporal &
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
        "Píxeles dinámicos dentro de máscara "
        "conservados:",
        rescatados
    )

    print(
        "Frames temporales comparados:",
        frames_temporales
    )

    print(
        "Frames utilizados para máscara:",
        frames_mascara
    )

    return (
        resultado,
        cantidad,
        frames_mascara
    )


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
                0.0,
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

        (
            _,
            distancia,
            y,
            x,
            etiqueta
        ) = heapq.heappop(
            heap
        )

        intensidad_actual = int(
            matriz[y, x]
        )

        for dy, dx in vecinos:

            ny = y + dy
            nx = x + dx

            if not (
                0 <= ny < h
                and
                0 <= nx < w
            ):
                continue

            if not componente[
                ny,
                nx
            ]:
                continue

            if matriz[
                ny,
                nx
            ] <= 0:
                continue

            if etiquetas[
                ny,
                nx
            ] != 0:
                continue

            nueva = int(
                matriz[
                    ny,
                    nx
                ]
            )

            nuevo_costo = (
                distancia +
                1 +
                abs(
                    intensidad_actual -
                    nueva
                ) * 100
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
            np.sum(
                region
            )
        ) >= MIN_AREA_CELDA_GRANDE:

            regiones.append(
                region
            )

    if not regiones:

        return [
            componente
        ]

    return regiones


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

        if int(
            np.sum(
                componente
            )
        ) >= MIN_AREA_NUCLEO:

            componentes.append(
                componente
            )

    return componentes


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

    return {

        "x":
            round(
                float(
                    np.mean(xs)
                ),
                2
            ),

        "y":
            round(
                float(
                    np.mean(ys)
                ),
                2
            ),

        "pixeles":
            int(
                len(xs)
            ),

        "ancho":
            int(
                xs.max() -
                xs.min() +
                1
            ),

        "alto":
            int(
                ys.max() -
                ys.min() +
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


def detectar_nucleos(
    matriz
):

    nucleos = []

    componentes = detectar_componentes(
        matriz
    )

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


def _region_nucleo(nucleo, shape):

    x = int(round(nucleo["x"]))
    y = int(round(nucleo["y"]))
    ancho = max(int(nucleo["ancho"]), 1)
    alto = max(int(nucleo["alto"]), 1)

    h, w = shape
    x1 = max(0, x - ancho // 2)
    x2 = min(w, x + ancho // 2 + 1)
    y1 = max(0, y - alto // 2)
    y2 = min(h, y + alto // 2 + 1)
    return y1, y2, x1, x2


def analizar_mascara(
    nucleo,
    mascara,
    evidencia_temporal=None
):

    if mascara is None:
        return ("DINAMICO", 0.70)

    try:
        y1, y2, x1, x2 = _region_nucleo(nucleo, mascara.shape)
        region = mascara[y1:y2, x1:x2]

        if region.size == 0:
            return ("DINAMICO", 0.70)

        porcentaje = float(np.mean(region))

        if porcentaje >= 0.70:
            if evidencia_temporal is not None:
                evidencia = evidencia_temporal[y1:y2, x1:x2]
                if evidencia.size and float(np.mean(evidencia)) >= 0.15:
                    return ("CAMBIO", 0.78)
            return ("ECO_ESTATICO", 0.95)

        if porcentaje >= 0.40:
            return ("CAMBIO", 0.65)

        return ("DINAMICO", 0.75)

    except Exception:
        return ("DINAMICO", 0.50)


def calcular_confianza(
    nucleo,
    estado,
    confianza_base
):

    if estado == "ECO_ESTATICO":
        return 0.0

    confianza = float(
        confianza_base
    )

    if int(
        nucleo.get(
            "intensidad",
            0
        )
    ) >= 5:

        confianza += 0.08

    if int(
        nucleo.get(
            "pixeles",
            0
        )
    ) >= 100:

        confianza += 0.05

    return round(
        min(
            1.0,
            confianza
        ),
        3
    )


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

    (
        matriz_dinamica,
        pixeles_estaticos,
        frames
    ) = aplicar_mascara_estatica(
        matriz_original,
        ruta_actual=ACTUAL
    )

    pixeles_dinamicos = int(
        np.sum(
            matriz_dinamica > 0
        )
    )

    mascara = cargar_mascara()
    evidencia_temporal, _ = _mascara_temporal_dinamica(
        matriz_original,
        mascara,
        ruta_actual=ACTUAL
    )

    # La matriz limpia alimenta la cadena operativa.
    # El frame original se usa solo para conservar/contabilizar
    # ecos estaticos internamente.
    nucleos_operativos = detectar_nucleos(
        matriz_dinamica
    )

    nucleos_candidatos_estaticos = detectar_nucleos(
        matriz_original
    )

    nucleos_estaticos = []

    for nucleo in nucleos_candidatos_estaticos:
        estado, confianza_base = analizar_mascara(
            nucleo,
            mascara,
            evidencia_temporal
        )
        nucleo["estado"] = estado
        nucleo["confianza"] = calcular_confianza(
            nucleo,
            estado,
            confianza_base
        )
        if estado == "ECO_ESTATICO":
            nucleos_estaticos.append(nucleo)

    nucleos_dinamicos = []

    for nucleo in nucleos_operativos:
        estado, confianza_base = analizar_mascara(
            nucleo,
            mascara,
            evidencia_temporal
        )

        if estado == "ECO_ESTATICO":
            nucleos_estaticos.append(nucleo)
            continue

        nucleo["estado"] = estado
        nucleo["confianza"] = calcular_confianza(
            nucleo,
            estado,
            confianza_base
        )
        nucleos_dinamicos.append(nucleo)

    nucleos_estaticos.sort(
        key=lambda n: (
            -n.get("pixeles", 0),
            -n.get("intensidad", 0)
        )
    )

    nucleos_dinamicos.sort(
        key=lambda n: (
            -n.get("pixeles", 0),
            -n.get("intensidad", 0)
        )
    )

    # Este es el unico conjunto que pasa a tracking/ML/Intelligence.
    nucleos = nucleos_dinamicos

    estado_general = calcular_estado_general(
        nucleos_dinamicos
    )

    resultado = {

        "version":
            "RMA10-procesador-10.2",

        "timestamp_procesamiento":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "fuente":
            "SMN RMA10",

        "metodo_segmentacion":
            (
                "mascara_estatica_temporal_"
                "con_rescate_de_ecos_dinamicos_"
                "conectividad_semillas_priority_flood"
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
            len(nucleos_estaticos),

        "nucleos_detectados":
            len(nucleos_dinamicos) + len(nucleos_estaticos),

        "nucleos_dinamicos":
            len(nucleos_dinamicos),

        "nucleos_estaticos":
            nucleos_estaticos,

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

    print(
        "Procesamiento RMA10 completado."
    )


if __name__ == "__main__":

    main()
