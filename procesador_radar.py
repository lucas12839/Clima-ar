#!/usr/bin/env python3

from pathlib import Path
from PIL import Image
from collections import Counter


RADAR_FILE = Path("data/radar/actual.png")


def clasificar_color(r, g, b):
    """
    Clasificación aproximada por color.
    NO representa todavía valores dBZ.
    """

    # Fondo / transparente / negro
    if r < 20 and g < 20 and b < 20:
        return "FONDO"

    # Azules y cian
    if b > r * 1.25 and b > g * 1.05:
        return "AZUL/CYAN"

    # Verdes
    if g > r * 1.25 and g > b * 1.15:
        return "VERDE"

    # Amarillos
    if r > 150 and g > 150 and b < 120:
        return "AMARILLO"

    # Naranjas
    if r > 170 and g > 80 and g < 180 and b < 100:
        return "NARANJA"

    # Rojos
    if r > 150 and r > g * 1.35 and r > b * 1.35:
        return "ROJO"

    # Magenta / violeta
    if r > 100 and b > 100 and r > g * 1.25:
        return "VIOLETA/MAGENTA"

    return "OTRO"


def procesar_radar():

    print("======================================")
    print("ClimaAR - ANÁLISIS DE COLORES RMA10")
    print("======================================")

    if not RADAR_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró la imagen: {RADAR_FILE}"
        )

    print("")
    print("Imagen encontrada:")
    print(RADAR_FILE)

    imagen = Image.open(RADAR_FILE)

    print("")
    print("Información de la imagen:")
    print(f"Formato: {imagen.format}")
    print(f"Ancho: {imagen.width}")
    print(f"Alto: {imagen.height}")
    print(f"Modo: {imagen.mode}")

    # Convertimos a RGB para analizar correctamente los colores
    imagen = imagen.convert("RGB")

    pixeles = list(imagen.getdata())

    total_pixeles = len(pixeles)

    categorias = Counter()

    for r, g, b in pixeles:

        categoria = clasificar_color(r, g, b)

        categorias[categoria] += 1

    print("")
    print("======================================")
    print("CLASIFICACIÓN DEL RADAR")
    print("======================================")

    orden = [
        "FONDO",
        "AZUL/CYAN",
        "VERDE",
        "AMARILLO",
        "NARANJA",
        "ROJO",
        "VIOLETA/MAGENTA",
        "OTRO"
    ]

    for categoria in orden:

        cantidad = categorias[categoria]

        porcentaje = (
            cantidad / total_pixeles * 100
        )

        print(
            f"{categoria:<18} "
            f"{cantidad:>8} píxeles "
            f"({porcentaje:>6.2f}%)"
        )

    # ======================================
    # COLORES REALES MÁS FRECUENTES
    # ======================================

    print("")
    print("======================================")
    print("COLORES REALES MÁS FRECUENTES")
    print("======================================")

    colores = Counter(pixeles)

    mostrados = 0

    for (r, g, b), cantidad in colores.most_common():

        # Ignorar negro/fondo
        if r < 20 and g < 20 and b < 20:
            continue

        porcentaje = (
            cantidad / total_pixeles * 100
        )

        print(
            f"RGB ({r}, {g}, {b}) -> "
            f"{cantidad} píxeles "
            f"({porcentaje:.2f}%)"
        )

        mostrados += 1

        if mostrados >= 30:
            break

    print("")
    print("======================================")
    print("ANÁLISIS TERMINADO")
    print("======================================")


if __name__ == "__main__":
    procesar_radar()
