#!/usr/bin/env python3

from pathlib import Path
from collections import deque, Counter
from PIL import Image


RADAR_FILE = Path("data/radar/actual.png")

# Ignorar manchas demasiado pequeñas
MIN_PIXELS = 10


def clasificar_color(r, g, b):

    # Fondo
    if max(r, g, b) < 25:
        return "FONDO"

    diferencia = max(r, g, b) - min(r, g, b)

    if diferencia < 20:
        return "FONDO"

    # Azul / Cyan
    if b > g and g > r * 1.5:
        return "AZUL/CYAN"

    # Verde
    if g > r * 1.35 and g >= b * 0.95:
        return "VERDE"

    # Amarillo
    if r > 100 and g > 100 and b < 100:
        if abs(r - g) < 100:
            return "AMARILLO"

    # Naranja
    if r > 130 and g > 60 and g < 180 and b < 90:
        if r > g * 1.15:
            return "NARANJA"

    # Rojo
    if r > 130 and r > g * 1.35 and r > b * 1.35:
        return "ROJO"

    # Violeta / Magenta
    if r > 100 and b > 100 and r > g * 1.3:
        return "VIOLETA/MAGENTA"

    return "OTRO"


# Prioridad de intensidad
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

    print("")
    print("Imagen:")
    print(RADAR_FILE)

    print("")
    print(f"Resolución: {ancho} x {alto}")

    pixeles = imagen.load()

    # ======================================
    # CLASIFICAR CADA PÍXEL
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
    # BUSCAR NÚCLEOS CONECTADOS
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

            # Nuevo núcleo
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

                puntos.append((px, py))

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

            # Ignorar ruido muy pequeño
            if cantidad < MIN_PIXELS:
                continue

            # Intensidad máxima del núcleo
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

            nucleos.append({
                "pixeles": cantidad,
                "centro_x": centro_x,
                "centro_y": centro_y,
                "ancho": max_x - min_x + 1,
                "alto": max_y - min_y + 1,
                "intensidad": intensidad_maxima,
                "colores": colores
            })

    # ======================================
    # ORDENAR NÚCLEOS POR TAMAÑO
    # ======================================

    nucleos.sort(
        key=lambda n: n["pixeles"],
        reverse=True
    )

    # ======================================
    # RESULTADO
    # ======================================

    print("")
    print("======================================")
    print("NÚCLEOS DE PRECIPITACIÓN")
    print("======================================")

    print(
        f"Núcleos detectados: {len(nucleos)}"
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
            f"  Píxeles: {nucleo['pixeles']}"
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
            f"{nucleo['intensidad']}"
        )

        print("  Colores:")

        for color, cantidad in nucleo[
            "colores"
        ].most_common():

            print(
                f"    {color}: {cantidad}"
            )

        print("")

    print("======================================")
    print("ANÁLISIS TERMINADO")
    print("======================================")


if __name__ == "__main__":
    procesar_radar()
