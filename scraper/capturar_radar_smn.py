#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
import sys
import requests


OUTPUT_DIR = Path("data/radar")
HISTORIC_DIR = OUTPUT_DIR / "historico"
LATEST_FILE = OUTPUT_DIR / "actual.png"

API_URL = "https://ws1.smn.gob.ar/v1/images/radar/RMA10"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def descargar_radar():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORIC_DIR.mkdir(parents=True, exist_ok=True)

    print("======================================")
    print("CLIMAAR - PRUEBA DIRECTA RMA10")
    print("======================================")
    print()

    print("Consultando:")
    print(API_URL)

    respuesta = requests.get(
        API_URL,
        headers=HEADERS,
        timeout=30
    )

    print("HTTP:", respuesta.status_code)
    print("Content-Type:", respuesta.headers.get("content-type"))

    respuesta.raise_for_status()

    print()
    print("Respuesta recibida:")
    print(respuesta.text[:3000])

    # Guardamos la respuesta para inspeccionarla
    debug_file = OUTPUT_DIR / "respuesta_rma10.txt"

    debug_file.write_text(
        respuesta.text,
        encoding="utf-8"
    )

    print()
    print("Respuesta guardada en:")
    print(debug_file)

    print()
    print("======================================")
    print("PRUEBA TERMINADA")
    print("======================================")


if __name__ == "__main__":

    try:
        descargar_radar()

    except Exception as e:

        print()
        print("======================================")
        print("ERROR")
        print("======================================")
        print(str(e))

        sys.exit(1)
