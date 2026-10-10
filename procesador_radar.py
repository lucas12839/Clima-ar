#!/usr/bin/env python3

from pathlib import Path
from PIL import Image


RADAR_FILE = Path("data/radar/actual.png")


def procesar_radar():

    print("======================================")
    print("ClimaAR - PROCESADOR RADAR")
    print("======================================")

    if not RADAR_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró la imagen: {RADAR_FILE}"
        )

    print("")
    print("Imagen encontrada:")
    print(RADAR_FILE)

    # Abrir imagen
    imagen = Image.open(RADAR_FILE)

    print("")
    print("Información de la imagen:")
    print(f"Formato: {imagen.format}")
    print(f"Ancho: {imagen.width}")
    print(f"Alto: {imagen.height}")
    print(f"Modo: {imagen.mode}")

    # Convertir a RGB
    imagen_rgb = imagen.convert("RGB")

    # Analizar píxeles
    pixeles = imagen_rgb.load()

    ancho = imagen_rgb.width
    alto = imagen_rgb.height

    total_pixeles = ancho * alto

    pixeles_con_color = 0

    for y in range(alto):

        for x in range(ancho):

            r, g, b = pixeles[x, y]

            # Detectar píxeles que no sean
            # prácticamente blancos/grises.
            if not (
                abs(r - g) < 8
                and abs(g - b) < 8
                and abs(r - b) < 8
            ):

                pixeles_con_color += 1

    porcentaje = (
        pixeles_con_color
        / total_pixeles
        * 100
    )

    print("")
    print("Análisis básico:")
    print(
        f"Píxeles con color: {pixeles_con_color}"
    )

    print(
        f"Porcentaje detectado: "
        f"{porcentaje:.2f}%"
    )

    print("")
    print("Procesamiento terminado.")


if __name__ == "__main__":
    procesar_radar()
