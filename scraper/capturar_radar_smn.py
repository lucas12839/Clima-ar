#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import re
import requests

PAGINA = "https://radares.hidricosargentina.gob.ar/"
BASE_URL = "https://radares.hidricosargentina.gob.ar"

RADAR = "RMA10"

OUTPUT_DIR = Path("data/radar")
HISTORICO_DIR = OUTPUT_DIR / "historico"
ACTUAL_FILE = OUTPUT_DIR / "actual.png"

HEADERS = {
    "User-Agent": "ClimaAR/1.0"
}


def buscar_imagen_radar():
    print("Consultando SINARAME...")

    respuesta = requests.get(
        PAGINA,
        headers=HEADERS,
        timeout=30
    )

    respuesta.raise_for_status()

    html = respuesta.text

    # Buscar cualquier PNG perteneciente a RMA10
    patron = rf"/cache/{RADAR}/([0-9]{{14}}\.png)"

    encontrados = re.findall(
        patron,
        html
    )

    if not encontrados:
        raise RuntimeError(
            "No se encontró ninguna imagen RMA10 en la página."
        )

    # Eliminar duplicados
    encontrados = list(dict.fromkeys(encontrados))

    # Ordenar cronológicamente
    encontrados.sort()

    archivo = encontrados[-1]

    url = f"{BASE_URL}/cache/{RADAR}/{archivo}"

    print("")
    print("Imagen encontrada:")
    print(url)

    return url, archivo


def descargar_imagen(url, archivo):
    print("")
    print("Descargando radar...")

    respuesta = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )

    respuesta.raise_for_status()

    if not respuesta.content:
        raise RuntimeError(
            "El servidor devolvió una imagen vacía."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    HISTORICO_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Guardar imagen actual
    ACTUAL_FILE.write_bytes(
        respuesta.content
    )

    # Guardar histórico usando el nombre original
    historico = HISTORICO_DIR / archivo

    historico.write_bytes(
        respuesta.content
    )

    print("")
    print("======================================")
    print("RADAR DESCARGADO CORRECTAMENTE")
    print("======================================")

    print(
        f"Archivo actual: {ACTUAL_FILE}"
    )

    print(
        f"Histórico: {historico}"
    )

    print(
        f"Tamaño: {len(respuesta.content)} bytes"
    )


def main():
    inicio = datetime.now()

    print("======================================")
    print("ClimaAR - DESCARGADOR RMA10")
    print("======================================")
    print(
        f"Inicio: {inicio.isoformat()}"
    )

    url, archivo = buscar_imagen_radar()

    descargar_imagen(
        url,
        archivo
    )

    fin = datetime.now()

    print("")
    print(
        f"Finalizado: {fin.isoformat()}"
    )


if __name__ == "__main__":
    main()
