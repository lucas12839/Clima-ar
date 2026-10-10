#!/usr/bin/env python3

from pathlib import Path
from PIL import Image


RADAR_FILE = Path("data/radar/actual.png")


def clasificar_color(r, g, b):

    # Fondo
    if max(r, g, b) < 25:
        return "FONDO"

    # Ignorar colores casi grises
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


def procesar_radar():

    print("======================================")
    print("ClimaAR - ANALISIS DE RADAR RMA10")
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
    print("Información:")
    print(f"Formato: {imagen.format}")
    print(f"Ancho: {ancho}")
    print(f"Alto: {alto}")

    pixeles = imagen.load()

    resultados = {
        "FONDO": 0,
        "AZUL/CYAN": 0,
        "VERDE": 0,
        "AMARILLO": 0,
        "NARANJA": 0,
        "ROJO": 0,
        "VIOLETA/MAGENTA": 0,
        "OTRO": 0
    }

    # Límites de cada color
    limites = {}

    for categoria in resultados:

        if categoria == "FONDO":
            continue

        limites[categoria] = {
            "min_x": ancho,
            "max_x": 0,
            "min_y": alto,
            "max_y": 0
        }

    # Analizar todos los píxeles
    for y in range(alto):

        for x in range(ancho):

            r, g, b = pixeles[x, y]

            categoria = clasificar_color(r, g, b)

            resultados[categoria] += 1

            if categoria != "FONDO":

                limite = limites[categoria]

                limite["min_x"] = min(
                    limite["min_x"], x
                )

                limite["max_x"] = max(
                    limite["max_x"], x
                )

                limite["min_y"] = min(
                    limite["min_y"], y
                )

                limite["max_y"] = max(
                    limite["max_y"], y
                )

    total = ancho * alto

    print("")
    print("======================================")
    print("CLASIFICACIÓN DEL RADAR")
    print("======================================")

    for categoria, cantidad in resultados.items():

        porcentaje = cantidad / total * 100

        print(
            f"{categoria:<18} "
            f"{cantidad:>7} píxeles "
            f"({porcentaje:.2f}%)"
        )

    print("")
    print("======================================")
    print("ZONAS DETECTADAS")
    print("======================================")

    for categoria, limite in limites.items():

        cantidad = resultados[categoria]

        if cantidad == 0:
            continue

        ancho_zona = (
            limite["max_x"] - limite["min_x"] + 1
        )

        alto_zona = (
            limite["max_y"] - limite["min_y"] + 1
        )

        centro_x = (
            limite["min_x"] + limite["max_x"]
        ) // 2

        centro_y = (
            limite["min_y"] + limite["max_y"]
        ) // 2

        print("")
        print(categoria)

        print(
            f"  Cantidad: {cantidad} píxeles"
        )

        print(
            f"  Zona: {ancho_zona} x {alto_zona}"
        )

        print(
            f"  Centro: X={centro_x} Y={centro_y}"
        )

    print("")
    print("======================================")
    print("ANÁLISIS TERMINADO")
    print("======================================")


if __name__ == "__main__":
    procesar_radar()
