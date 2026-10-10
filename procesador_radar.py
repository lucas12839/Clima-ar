#!/usr/bin/env python3

from pathlib import Path
from PIL import Image
from collections import Counter


RADAR_FILE = Path("data/radar/actual.png")


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

    imagen = Image.open(RADAR_FILE).convert("RGB")

    print("")
    print("Información de la imagen:")
    print(f"Formato: {imagen.format}")
    print(f"Ancho: {imagen.width}")
    print(f"Alto: {imagen.height}")

    # ======================================
    # CONTAR COLORES
    # ======================================

    pixeles = list(imagen.getdata())

    total_pixeles = len(pixeles)

    contador = Counter(pixeles)

    print("")
    print("======================================")
    print("COLORES MÁS FRECUENTES")
    print("======================================")

    for color, cantidad in contador.most_common(30):

        porcentaje = (
            cantidad / total_pixeles * 100
        )

        print(
            f"RGB {color} -> "
            f"{cantidad} píxeles "
            f"({porcentaje:.2f}%)"
        )

    # ======================================
    # COLORES NO GRISES
    # ======================================

    colores = []

    for color, cantidad in contador.items():

        r, g, b = color

        diferencia = max(color) - min(color)

        if diferencia > 15:

            colores.append(
                (cantidad, color)
            )

    colores.sort(reverse=True)

    print("")
    print("======================================")
    print("COLORES DEL RADAR")
    print("======================================")

    total_colores = sum(
        cantidad
        for cantidad, color in colores
    )

    porcentaje_colores = (
        total_colores
        / total_pixeles
        * 100
    )

    print(
        f"Píxeles con color: {total_colores}"
    )

    print(
        f"Porcentaje: "
        f"{porcentaje_colores:.2f}%"
    )

    print("")
    print("Principales colores detectados:")

    for cantidad, color in colores[:30]:

        porcentaje = (
            cantidad / total_pixeles * 100
        )

        print(
            f"RGB {color} -> "
            f"{cantidad} píxeles "
            f"({porcentaje:.2f}%)"
        )

    print("")
    print("======================================")
    print("ANÁLISIS TERMINADO")
    print("======================================")


if __name__ == "__main__":
    procesar_radar()
