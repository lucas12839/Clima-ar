#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timedelta, timezone
import os
import time
import requests


BASE_URL = "https://radares.hidricosargentina.gob.ar"

OUTPUT_DIR = Path("data/radar")
HISTORICO_DIR = OUTPUT_DIR / "historico"
ACTUAL_FILE = OUTPUT_DIR / "actual.png"

RADAR = "RMA10"

FRAMES_INICIALES = 3

# Cuántos minutos hacia atrás buscar
MINUTOS_BUSQUEDA = 120

# Intervalo habitual de los frames
INTERVALO_MINUTOS = 10


def descargar_frame(timestamp):

    nombre = timestamp.strftime("%Y%m%d%H%M%S") + ".png"

    url = (
        f"{BASE_URL}/cache/"
        f"{RADAR}/{nombre}"
    )

    print(f"Probando: {nombre}")

    try:

        respuesta = requests.get(
            url,
            timeout=15
        )

    except Exception as e:

        print(f"Error de conexión: {e}")
        return None

    if respuesta.status_code != 200:

        print(
            f"  HTTP {respuesta.status_code}"
        )

        return None

    contenido = respuesta.content

    # Evitar guardar respuestas inválidas
    if len(contenido) < 1000:

        print(
            f"  Respuesta demasiado pequeña: "
            f"{len(contenido)} bytes"
        )

        return None

    historico = HISTORICO_DIR / nombre

    historico.write_bytes(contenido)
    ACTUAL_FILE.write_bytes(contenido)

    print(
        f"  OK -> {nombre} "
        f"({len(contenido)} bytes)"
    )

    return nombre


def buscar_frames(cantidad):

    ahora = datetime.now(timezone.utc)

    # Redondeamos al múltiplo de 10 minutos
    minuto = (
        ahora.minute
        - ahora.minute % INTERVALO_MINUTOS
    )

    base = ahora.replace(
        minute=minuto,
        second=0,
        microsecond=0
    )

    encontrados = []

    print("")
    print("======================================")
    print("BUSCANDO FRAMES RMA10")
    print("======================================")
    print(f"Hora UTC: {ahora}")
    print(f"Desde: {base}")
    print("")

    pasos = (
        MINUTOS_BUSQUEDA //
        INTERVALO_MINUTOS
    )

    for i in range(pasos + 1):

        timestamp = (
            base -
            timedelta(
                minutes=i *
                INTERVALO_MINUTOS
            )
        )

        nombre = descargar_frame(timestamp)

        if nombre:

            if nombre not in encontrados:

                encontrados.append(nombre)

                print(
                    f"FRAME ENCONTRADO "
                    f"({len(encontrados)}/{cantidad})"
                )

            if len(encontrados) >= cantidad:
                break

        time.sleep(0.2)

    return encontrados


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    HISTORICO_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    modo = os.environ.get(
        "CLIMAAR_CAPTURE_MODE",
        ""
    ).strip().lower()

    if modo not in {
        "rapido",
        "inicial"
    }:

        if (
            OUTPUT_DIR /
            "mascara_ecos_estaticos.png"
        ).exists():

            modo = "rapido"

        else:

            modo = "inicial"

    if modo == "rapido":
        cantidad = 1
    else:
        cantidad = FRAMES_INICIALES

    print("")
    print("======================================")
    print("ClimaAR - CAPTURA DIRECTA RMA10")
    print("======================================")
    print(f"Modo: {modo.upper()}")
    print(f"Frames necesarios: {cantidad}")
    print("======================================")

    frames = buscar_frames(cantidad)

    print("")
    print("======================================")
    print("RESULTADO")
    print("======================================")

    if not frames:

        raise RuntimeError(
            "No se encontró ningún PNG "
            "RMA10 en los últimos "
            f"{MINUTOS_BUSQUEDA} minutos."
        )

    for frame in frames:

        print(frame)

    print("")
    print(
        f"Frames encontrados: "
        f"{len(frames)}"
    )

    print("CAPTURA EXITOSA")


if __name__ == "__main__":
    main()
