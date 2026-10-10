#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timedelta, timezone
import requests
import sys


BASE_URL = "https://radares.hidricosargentina.gob.ar/cache/RMA10"

OUTPUT_DIR = Path("data/radar")
HISTORIC_DIR = OUTPUT_DIR / "historico"
LATEST_FILE = OUTPUT_DIR / "actual.png"

HEADERS = {
    "User-Agent": "ClimaAR/1.0"
}


def descargar_radar():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORIC_DIR.mkdir(parents=True, exist_ok=True)

    print("======================================")
    print("CLIMAAR - RMA10 SINARAME")
    print("======================================")
    print()

    ahora = datetime.now(timezone.utc).replace(
        second=0,
        microsecond=0
    )

    print("Buscando imagen RMA10...")
    print("Hora UTC:", ahora.strftime("%Y-%m-%d %H:%M"))

    # Buscamos hacia atrás hasta 2 horas.
    # El radar puede tener algunos minutos de demora.
    encontrada = None

    for minutos in range(0, 121):

        fecha = ahora - timedelta(minutes=minutos)

        nombre = fecha.strftime("%Y%m%d%H%M00.png")

        url = f"{BASE_URL}/{nombre}"

        print("Probando:", nombre)

        try:

            respuesta = requests.get(
                url,
                headers=HEADERS,
                timeout=15
            )

            if respuesta.status_code == 200:

                contenido = respuesta.content

                # Evitamos aceptar una respuesta vacía o demasiado pequeña.
                if len(contenido) > 5000:

                    encontrada = (
                        fecha,
                        nombre,
                        url,
                        contenido
                    )

                    print()
                    print("======================================")
                    print("RADAR ENCONTRADO")
                    print("======================================")
                    print("Archivo:", nombre)
                    print("URL:", url)
                    print("Tamaño:", len(contenido), "bytes")

                    break

        except requests.RequestException as e:

            print("Error de conexión:", e)

    if encontrada is None:

        raise RuntimeError(
            "No se encontró ninguna imagen RMA10 en las últimas 2 horas."
        )

    fecha, nombre, url, contenido = encontrada

    # Guardar imagen actual
    LATEST_FILE.write_bytes(contenido)

    # Guardar histórico
    historic_file = HISTORIC_DIR / nombre
    historic_file.write_bytes(contenido)

    print()
    print("Guardado:")
    print("Actual:", LATEST_FILE)
    print("Histórico:", historic_file)

    print()
    print("======================================")
    print("RMA10 CAPTURADO CORRECTAMENTE")
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
