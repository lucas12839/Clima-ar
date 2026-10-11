#!/usr/bin/env python3

from pathlib import Path
from collections import deque, Counter
from PIL import Image
import json


RADAR_FILE = Path("data/radar/actual.png")
MASCARA_FILE = Path("data/radar/mascara_ecos_estaticos.png")
MASCARA_INFO = Path("data/radar/mascara_ecos_estaticos.json")

MIN_PIXELS = 10


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


def cargar_mascara():

    if not MASCARA_FILE.exists():
        print("Máscara estática: NO DISPONIBLE")
        return None, None

    mascara = Image.open(MASCARA_FILE).convert("L")

    frames = []

    if MASCARA_INFO.exists():

        with open(MASCARA_INFO, "r") as archivo:
            informacion = json.load(archivo)

        for nombre in informacion.get("frames", []):

            archivo_frame = (
                Path("data/radar/historico") / nombre
            )

            if archivo_frame.exists():
                frames.append(
                    Image.open(archivo_frame).convert("RGB")
                )

    print("Máscara estática: ACTIVA")
    print(f"Frames base: {len(frames)}")

    return mascara, frames


def analizar_ecos_estaticos(
    puntos,
    mascara,
    frames_base
):

    if mascara is None:
        return False, "SIN_MASCARA"

    pixeles_mascara = mascara.load()

    dentro = 0
    fuera = 0

    aumentos_fuertes = 0
    puntos_comparables = 0

    # --------------------------------------------------
    # COMPARAR ECO ACTUAL CONTRA LA MÁSCARA
    # --------------------------------------------------

    for x, y, categoria_actual in puntos:

        if pixeles_mascara[x, y] > 0:
            dentro += 1

            if frames_base:

                intensidad_actual = INTENSIDAD[categoria_actual]

                intensidad_base = 0

                for frame in frames_base:

                    r, g, b = frame.getpixel((x, y))

                    categoria_base = clasificar_color(
                        r, g, b
                    )

                    intensidad_base = max(
                        intensidad_base,
                        INTENSIDAD[categoria_base]
                    )

                puntos_comparables += 1

                # Aumento importante respecto del eco base
                if intensidad_actual >= intensidad_base + 2:
                    aumentos_fuertes += 1

        else:
            fuera += 1

    total = dentro + fuera

    if total == 0:
        return False, "SIN_PIXELES"

    porcentaje_dentro = dentro / total
    porcentaje_fuera = fuera / total

    porcentaje_aumento = 0

    if puntos_comparables > 0:
        porcentaje_aumento = (
            aumentos_fuertes / puntos_comparables
        )

    # --------------------------------------------------
    # DECISIÓN
    # --------------------------------------------------

    # Si una parte importante apareció fuera
    # de la máscara, NO bloquear.
    if porcentaje_fuera >= 0.20:
        return False, "CRECIMIENTO_FUERA_MASCARA"

    # Si el eco se intensificó fuertemente,
    # recuperarlo como posible precipitación.
    if porcentaje_aumento >= 0.10:
        return False, "INTENSIFICACION"

    # Si casi todo coincide con el eco conocido,
    # considerarlo estático.
    if porcentaje_dentro >= 0.80:
        return True, "ECO_ESTATICO"

    return False, "CAMBIO_DE_ECO"


def procesar_radar():

    print("======================================")
    print("ClimaAR - DETECCIÓN DE NÚCLEOS RMA10")
    print("======================================")

    if not RADAR_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró la imagen: {RADAR_FILE}"
        )

    imagen = Image.open(RADAR_FILE).convert("RGB")

    ancho = imagen.width
    alto = imagen.height

    print()
    print("Imagen:")
    print(RADAR_FILE)

    print()
    print(f"Resolución: {ancho} x {alto}")

    mascara, frames_base = cargar_mascara()

    pixeles = imagen.load()

    # ======================================
    # CLASIFICAR CADA PIXEL
    # ======================================

    matriz = []

    for y in range(alto):

        fila = []

        for x in range(ancho):

            r, g, b = pixeles[x, y]

            fila.append(
                clasificar_color(r, g, b)
            )

        matriz.append(fila)

    # ======================================
    # BUSCAR NÚCLEOS
    # ======================================

    visitados = set()
    nucleos = []

    direcciones = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1)
    ]

    for y in range(alto):

        for x in range(ancho):

            if (x, y) in visitados:
                continue

            categoria = matriz[y][x]

            if categoria in ("FONDO", "OTRO"):
                continue

            cola = deque()

            cola.append((x, y))

            visitados.add((x, y))

            puntos = []
            colores = Counter()

            min_x = x
            max_x = x
            min_y = y
            max_y = y

            while cola:

                px, py = cola.popleft()

                puntos.append(
                    (px, py, matriz[py][px])
                )

                color_actual = matriz[py][px]

                colores[color_actual] += 1

                min_x = min(min_x, px)
                max_x = max(max_x, px)

                min_y = min(min_y, py)
                max_y = max(max_y, py)

                for dx, dy in direcciones:

                    nx = px + dx
                    ny = py + dy

                    if nx < 0 or nx >= ancho:
                        continue

                    if ny < 0 or ny >= alto:
                        continue

                    if (nx, ny) in visitados:
                        continue

                    nuevo_color = matriz[ny][nx]

                    if nuevo_color in (
                        "FONDO",
                        "OTRO"
                    ):
                        continue

                    visitados.add((nx, ny))

                    cola.append((nx, ny))

            cantidad = len(puntos)

            if cantidad < MIN_PIXELS:
                continue

            intensidad_maxima = max(
                colores,
                key=lambda c: INTENSIDAD[c]
            )

            centro_x = sum(
                p[0] for p in puntos
            ) // cantidad

            centro_y = sum(
                p[1] for p in puntos
            ) // cantidad

            # ======================================
            # FILTRO DE ECOS ESTÁTICOS
            # ======================================

            es_estatico, motivo = analizar_ecos_estaticos(
                puntos,
                mascara,
                frames_base
            )

            nucleos.append({
                "pixeles": cantidad,
                "centro_x": centro_x,
                "centro_y": centro_y,
                "ancho": max_x - min_x + 1,
                "alto": max_y - min_y + 1,
                "intensidad": intensidad_maxima,
                "colores": colores,
                "estatico": es_estatico,
                "motivo": motivo
            })

    # ======================================
    # ORDENAR
    # ======================================

    nucleos.sort(
        key=lambda n: n["pixeles"],
        reverse=True
    )

    # ======================================
    # RESULTADO
    # ======================================

    print()
    print("======================================")
    print("RESULTADO DEL FILTRO")
    print("======================================")

    total = len(nucleos)

    estaticos = sum(
        1 for n in nucleos
        if n["estatico"]
    )

    precipitacion = total - estaticos

    print(f"Núcleos detectados: {total}")
    print(f"Ecos estáticos ignorados: {estaticos}")
    print(f"Posibles núcleos de precipitación: {precipitacion}")

    print()
    print("======================================")
    print("NÚCLEOS")
    print("======================================")

    for numero, nucleo in enumerate(
        nucleos[:20],
        start=1
    ):

        estado = (
            "ECO ESTÁTICO"
            if nucleo["estatico"]
            else "PRECIPITACIÓN / CAMBIO"
        )

        print()
        print(f"NÚCLEO #{numero}")
        print(f"  Estado: {estado}")
        print(f"  Motivo: {nucleo['motivo']}")
        print(f"  Píxeles: {nucleo['pixeles']}")

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
            f"{nucleo['intensidad']}"
        )

        print("  Colores:")

        for color, cantidad in nucleo[
            "colores"
        ].most_common():

            print(
                f"    {color}: {cantidad}"
            )

    print()
    print("======================================")
    print("ANÁLISIS TERMINADO")
    print("======================================")


if __name__ == "__main__":
    procesar_radar()
