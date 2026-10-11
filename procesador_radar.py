#!/usr/bin/env python3

from pathlib import Path
from collections import deque, Counter
from PIL import Image
import json
import math


RADAR_FILE = Path("data/radar/actual.png")

MASCARA_FILE = Path(
    "data/radar/mascara_ecos_estaticos.png"
)

INFO_MASCARA = Path(
    "data/radar/mascara_ecos_estaticos.json"
)

HISTORICO_DIR = Path(
    "data/radar/historico"
)

MIN_PIXELS = 10


# ============================================================
# COLORES DEL RADAR
# ============================================================

def clasificar_color(r, g, b):

    if max(r, g, b) < 25:
        return "FONDO"

    diferencia = max(r, g, b) - min(r, g, b)

    if diferencia < 20:
        return "FONDO"

    if b > g and g > r * 1.5:
        return "AZUL/CYAN"

    if g > r * 1.35 and g >= b * 0.95:
        return "VERDE"

    if r > 100 and g > 100 and b < 100:
        if abs(r - g) < 100:
            return "AMARILLO"

    if r > 130 and g > 60 and g < 180 and b < 90:
        if r > g * 1.15:
            return "NARANJA"

    if r > 130 and r > g * 1.35 and r > b * 1.35:
        return "ROJO"

    if r > 100 and b > 100 and r > g * 1.3:
        return "VIOLETA/MAGENTA"

    return "OTRO"


INTENSIDAD = {
    "FONDO": 0,
    "AZUL/CYAN": 1,
    "VERDE": 2,
    "AMARILLO": 3,
    "NARANJA": 4,
    "ROJO": 5,
    "VIOLETA/MAGENTA": 6,
    "OTRO": 0
}


# ============================================================
# CARGAR MÁSCARA ESTÁTICA
# ============================================================

def cargar_mascara():

    if not MASCARA_FILE.exists():
        print("Máscara estática: NO DISPONIBLE")
        return None

    mascara = Image.open(
        MASCARA_FILE
    ).convert("L")

    print("Máscara estática: ACTIVA")

    if INFO_MASCARA.exists():

        try:

            with open(
                INFO_MASCARA,
                "r",
                encoding="utf-8"
            ) as archivo:

                info = json.load(archivo)

            frames = info.get(
                "frames",
                []
            )

            print(
                f"Frames base: {len(frames)}"
            )

        except Exception:

            print(
                "No se pudo leer información de la máscara."
            )

    return mascara


# ============================================================
# CARGAR FRAMES BASE
# ============================================================

def cargar_frames_base():

    if not INFO_MASCARA.exists():
        return []

    try:

        with open(
            INFO_MASCARA,
            "r",
            encoding="utf-8"
        ) as archivo:

            info = json.load(archivo)

        frames = []

        for nombre in info.get(
            "frames",
            []
        ):

            archivo_frame = (
                HISTORICO_DIR / nombre
            )

            if archivo_frame.exists():

                frames.append(
                    Image.open(
                        archivo_frame
                    ).convert("RGB")
                )

        return frames

    except Exception:

        return []


# ============================================================
# INTENSIDAD DE UN PIXEL
# ============================================================

def intensidad_pixel(
    r,
    g,
    b
):

    categoria = clasificar_color(
        r,
        g,
        b
    )

    return INTENSIDAD.get(
        categoria,
        0
    )


# ============================================================
# PROCESAMIENTO
# ============================================================

def procesar_radar():

    print("======================================")
    print("ClimaAR - MOTOR DE DETECCIÓN RMA10")
    print("======================================")

    if not RADAR_FILE.exists():

        raise FileNotFoundError(
            f"No se encontró: {RADAR_FILE}"
        )

    imagen = Image.open(
        RADAR_FILE
    ).convert("RGB")

    ancho = imagen.width
    alto = imagen.height

    print("")
    print(
        f"Imagen: {RADAR_FILE}"
    )

    print(
        f"Resolución: {ancho} x {alto}"
    )

    pixeles = imagen.load()

    mascara = cargar_mascara()

    frames_base = cargar_frames_base()

    # ========================================================
    # CLASIFICAR PIXELES
    # ========================================================

    matriz = []

    for y in range(alto):

        fila = []

        for x in range(ancho):

            r, g, b = pixeles[x, y]

            fila.append(
                clasificar_color(
                    r,
                    g,
                    b
                )
            )

        matriz.append(fila)

    # ========================================================
    # BUSCAR COMPONENTES CONECTADOS
    # ========================================================

    visitados = set()

    nucleos = []

    direcciones = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1)
    ]

    for y in range(alto):

        for x in range(ancho):

            if (x, y) in visitados:
                continue

            categoria = matriz[y][x]

            if categoria in (
                "FONDO",
                "OTRO"
            ):
                continue

            cola = deque()

            cola.append(
                (x, y)
            )

            visitados.add(
                (x, y)
            )

            puntos = []

            colores = Counter()

            min_x = x
            max_x = x
            min_y = y
            max_y = y

            while cola:

                px, py = cola.popleft()

                puntos.append(
                    (px, py)
                )

                color_actual = matriz[
                    py
                ][
                    px
                ]

                colores[
                    color_actual
                ] += 1

                min_x = min(
                    min_x,
                    px
                )

                max_x = max(
                    max_x,
                    px
                )

                min_y = min(
                    min_y,
                    py
                )

                max_y = max(
                    max_y,
                    py
                )

                for dx, dy in direcciones:

                    nx = px + dx
                    ny = py + dy

                    if nx < 0 or nx >= ancho:
                        continue

                    if ny < 0 or ny >= alto:
                        continue

                    if (
                        nx,
                        ny
                    ) in visitados:
                        continue

                    nuevo_color = matriz[
                        ny
                    ][
                        nx
                    ]

                    if nuevo_color in (
                        "FONDO",
                        "OTRO"
                    ):
                        continue

                    visitados.add(
                        (
                            nx,
                            ny
                        )
                    )

                    cola.append(
                        (
                            nx,
                            ny
                        )
                    )

            cantidad = len(
                puntos
            )

            if cantidad < MIN_PIXELS:
                continue

            intensidad_maxima = max(
                colores,
                key=lambda c:
                INTENSIDAD[c]
            )

            intensidad_promedio = (
                sum(
                    INTENSIDAD[c]
                    * cantidad_color
                    for c,
                    cantidad_color
                    in colores.items()
                )
                /
                cantidad
            )

            centro_x = (
                sum(
                    p[0]
                    for p in puntos
                )
                /
                cantidad
            )

            centro_y = (
                sum(
                    p[1]
                    for p in puntos
                )
                /
                cantidad
            )

            ancho_nucleo = (
                max_x - min_x + 1
            )

            alto_nucleo = (
                max_y - min_y + 1
            )

            # =================================================
            # ANÁLISIS CONTRA MÁSCARA
            # =================================================

            dentro_mascara = 0
            fuera_mascara = 0

            intensidad_fuera = []

            for px, py in puntos:

                es_estatico = (
                    mascara is not None
                    and mascara.getpixel(
                        (px, py)
                    ) > 0
                )

                if es_estatico:

                    dentro_mascara += 1

                else:

                    fuera_mascara += 1

                    intensidad_fuera.append(
                        INTENSIDAD[
                            matriz[py][px]
                        ]
                    )

            porcentaje_mascara = (
                dentro_mascara
                /
                cantidad
                * 100
            )

            porcentaje_fuera = (
                fuera_mascara
                /
                cantidad
                * 100
            )

            # =================================================
            # CLASIFICACIÓN
            # =================================================

            estado = "CAMBIO"

            confianza = 50

            motivo = "CAMBIO_DE_ECO"

            if mascara is None:

                if intensidad_maxima == "VIOLETA/MAGENTA":
                    confianza = 70

                elif intensidad_maxima == "ROJO":
                    confianza = 65

                elif intensidad_maxima == "NARANJA":
                    confianza = 60

                else:
                    confianza = 50

                estado = "PRECIPITACION"

                motivo = "SIN_MASCARA"

            else:

                if (
                    porcentaje_mascara >= 80
                    and porcentaje_fuera <= 20
                ):

                    estado = "ECO_ESTATICO"

                    confianza = 92

                    motivo = (
                        "COINCIDE_CON_ECO_ESTÁTICO"
                    )

                elif porcentaje_fuera >= 30:

                    estado = "PRECIPITACION"

                    confianza = 85

                    motivo = (
                        "ECO_FUERA_DE_MASCARA"
                    )

                elif porcentaje_fuera >= 15:

                    estado = "CAMBIO"

                    confianza = 70

                    motivo = (
                        "CRECIMIENTO_FUERA_MASCARA"
                    )

                else:

                    estado = "CAMBIO"

                    confianza = 65

                    motivo = (
                        "CAMBIO_DE_ECO"
                    )

            # =================================================
            # AJUSTE POR INTENSIDAD
            # =================================================

            if estado != "ECO_ESTATICO":

                if intensidad_maxima == "VIOLETA/MAGENTA":

                    confianza += 5

                elif intensidad_maxima == "ROJO":

                    confianza += 4

                elif intensidad_maxima == "NARANJA":

                    confianza += 3

                elif intensidad_maxima == "AMARILLO":

                    confianza += 1

            confianza = min(
                confianza,
                99
            )

            nucleos.append({

                "pixeles": cantidad,

                "centro_x": round(
                    centro_x,
                    1
                ),

                "centro_y": round(
                    centro_y,
                    1
                ),

                "ancho": ancho_nucleo,

                "alto": alto_nucleo,

                "intensidad_maxima":
                    intensidad_maxima,

                "intensidad_promedio":
                    round(
                        intensidad_promedio,
                        2
                    ),

                "estado":
                    estado,

                "confianza":
                    confianza,

                "motivo":
                    motivo,

                "porcentaje_mascara":
                    round(
                        porcentaje_mascara,
                        1
                    ),

                "porcentaje_fuera_mascara":
                    round(
                        porcentaje_fuera,
                        1
                    ),

                "colores":
                    dict(
                        colores
                    )
            })

    # ========================================================
    # ORDENAR
    # ========================================================

    nucleos.sort(
        key=lambda n:
        n["pixeles"],
        reverse=True
    )

    # ========================================================
    # ESTADÍSTICAS
    # ========================================================

    estaticos = sum(
        1
        for n in nucleos
        if n["estado"]
        == "ECO_ESTATICO"
    )

    precipitacion = sum(
        1
        for n in nucleos
        if n["estado"]
        == "PRECIPITACION"
    )

    cambios = sum(
        1
        for n in nucleos
        if n["estado"]
        == "CAMBIO"
    )

    # ========================================================
    # PROBABILIDAD GLOBAL
    # ========================================================

    nucleos_reales = [
        n
        for n in nucleos
        if n["estado"]
        in (
            "PRECIPITACION",
            "CAMBIO"
        )
    ]

    if nucleos_reales:

        peso_total = sum(
            n["pixeles"]
            for n in nucleos_reales
        )

        if peso_total > 0:

            probabilidad = sum(
                n["confianza"]
                * n["pixeles"]
                for n in nucleos_reales
            ) / peso_total

        else:

            probabilidad = 0

    else:

        probabilidad = 0

    probabilidad = round(
        min(
            probabilidad,
            99
        ),
        1
    )

    # ========================================================
    # RESULTADO EN CONSOLA
    # ========================================================

    print("")
    print("======================================")
    print("RESULTADO")
    print("======================================")

    print(
        f"Núcleos detectados: {len(nucleos)}"
    )

    print(
        f"Ecos estáticos: {estaticos}"
    )

    print(
        f"Precipitación: {precipitacion}"
    )

    print(
        f"Cambios: {cambios}"
    )

    print(
        f"Probabilidad radar: {probabilidad}%"
    )

    print("")

    for numero, nucleo in enumerate(
        nucleos[:20],
        start=1
    ):

        print(
            f"NÚCLEO #{numero}"
        )

        print(
            f"  Estado: {nucleo['estado']}"
        )

        print(
            f"  Confianza: "
            f"{nucleo['confianza']}%"
        )

        print(
            f"  Píxeles: "
            f"{nucleo['pixeles']}"
        )

        print(
            f"  Centro: "
            f"X={nucleo['centro_x']} "
            f"Y={nucleo['centro_y']}"
        )

        print(
            f"  Tamaño: "
            f"{nucleo['ancho']} x "
            f"{nucleo['alto']}"
        )

        print(
            f"  Intensidad máxima: "
            f"{nucleo['intensidad_maxima']}"
        )

        print(
            f"  Intensidad promedio: "
            f"{nucleo['intensidad_promedio']}"
        )

        print(
            f"  Dentro máscara: "
            f"{nucleo['porcentaje_mascara']}%"
        )

        print(
            f"  Fuera máscara: "
            f"{nucleo['porcentaje_fuera_mascara']}%"
        )

        print(
            f"  Motivo: "
            f"{nucleo['motivo']}"
        )

        print("")

    # ========================================================
    # GUARDAR RESULTADO JSON
    # ========================================================

    resultado = {

        "radar": "RMA10",

        "resolucion": {
            "ancho": ancho,
            "alto": alto
        },

        "nucleos_detectados":
            len(nucleos),

        "ecos_estaticos":
            estaticos,

        "precipitacion":
            precipitacion,

        "cambios":
            cambios,

        "probabilidad_precipitacion":
            probabilidad,

        "nucleos":
            nucleos

    }

    salida = Path(
        "data/radar/resultado_procesamiento.json"
    )

    salida.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        salida,
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            resultado,
            archivo,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Resultado guardado: {salida}"
    )

    print("")
    print("======================================")
    print("ANÁLISIS TERMINADO")
    print("======================================")


if __name__ == "__main__":

    procesar_radar()
