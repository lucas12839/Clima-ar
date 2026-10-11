#!/usr/bin/env python3

from pathlib import Path
import os
import time
import re

from playwright.sync_api import sync_playwright


URL = "https://radares.hidricosargentina.gob.ar/"

OUTPUT_DIR = Path("data/radar")
HISTORICO_DIR = OUTPUT_DIR / "historico"
ACTUAL_FILE = OUTPUT_DIR / "actual.png"

MASCARA_FILE = OUTPUT_DIR / "mascara_ecos_estaticos.png"

FRAMES_INICIALES = 3


def buscar_selector_rma10(page):

    selects = page.locator("select")

    for i in range(selects.count()):

        selector = selects.nth(i)

        try:
            texto = selector.inner_text()
        except Exception:
            continue

        if (
            "Bahía Blanca" not in texto
            and "Bahia Blanca" not in texto
            and "RMA10" not in texto
        ):
            continue

        opciones = selector.locator("option")

        for j in range(opciones.count()):

            opcion = opciones.nth(j)

            texto_opcion = opcion.inner_text().strip()
            valor = opcion.get_attribute("value")

            if (
                valor
                and (
                    "Bahía Blanca" in texto_opcion
                    or "Bahia Blanca" in texto_opcion
                    or "RMA10" in texto_opcion
                )
            ):
                return selector, valor

    raise RuntimeError(
        "No se encontró el selector de Bahía Blanca/RMA10."
    )


def descargar_png(page, radar_url):

    print("")
    print("======================================")
    print("DESCARGANDO RMA10")
    print("======================================")
    print(f"URL: {radar_url}")

    respuesta = page.request.get(
        radar_url,
        timeout=15000
    )

    print(f"HTTP: {respuesta.status}")

    if respuesta.status != 200:
        raise RuntimeError(
            f"HTTP {respuesta.status}"
        )

    contenido = respuesta.body()

    if len(contenido) < 1000:
        raise RuntimeError(
            f"PNG demasiado pequeño: {len(contenido)} bytes"
        )

    nombre = radar_url.split("/")[-1]

    historico = HISTORICO_DIR / nombre

    historico.write_bytes(contenido)
    ACTUAL_FILE.write_bytes(contenido)

    print(f"Guardado: {historico}")
    print(f"Tamaño: {len(contenido)} bytes")
    print("CAPTURA EXITOSA")

    return nombre


def capturar_frame(page):

    selector, valor = buscar_selector_rma10(page)

    print("Seleccionando Bahía Blanca (RMA10)...")

    selector.select_option(value=valor)

    print("Esperando imagen del radar...")

    inicio = time.monotonic()

    while time.monotonic() - inicio < 15:

        imagenes = page.locator(
            'img[src*="/cache/RMA10/"]'
        )

        cantidad = imagenes.count()

        for i in range(cantidad):

            src = imagenes.nth(i).get_attribute("src")

            if not src:
                continue

            if "/cache/RMA10/" not in src:
                continue

            if not src.endswith(".png"):
                continue

            if src.startswith("/"):
                radar_url = (
                    "https://radares.hidricosargentina.gob.ar"
                    + src
                )
            else:
                radar_url = src

            print("")
            print("URL ENCONTRADA EN HTML:")
            print(radar_url)

            return descargar_png(
                page,
                radar_url
            )

        page.wait_for_timeout(500)

    raise RuntimeError(
        "No apareció la imagen PNG de RMA10."
    )


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

        if MASCARA_FILE.exists():
            modo = "rapido"
        else:
            modo = "inicial"

    if modo == "rapido":
        frames_necesarios = 1
    else:
        frames_necesarios = FRAMES_INICIALES

    print("")
    print("======================================")
    print("ClimaAR - CAPTURA RMA10")
    print("======================================")
    print(f"Modo: {modo.upper()}")
    print(f"Frames necesarios: {frames_necesarios}")
    print("======================================")

    frames_guardados = set()

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        page = browser.new_page(
            viewport={
                "width": 1920,
                "height": 1080
            }
        )

        try:

            while len(frames_guardados) < frames_necesarios:

                print("")
                print("Abriendo SINARAME...")

                page.goto(
                    URL,
                    wait_until="domcontentloaded",
                    timeout=30000
                )

                page.wait_for_timeout(3000)

                nombre = capturar_frame(page)

                if nombre in frames_guardados:

                    print(
                        f"Frame repetido: {nombre}"
                    )

                    if modo == "rapido":
                        break

                    print(
                        "Esperando nuevo frame..."
                    )

                    time.sleep(30)

                else:

                    frames_guardados.add(nombre)

                    print(
                        f"NUEVO FRAME: {nombre}"
                    )

                    if (
                        len(frames_guardados)
                        < frames_necesarios
                    ):
                        time.sleep(30)

        finally:

            browser.close()

    print("")
    print("======================================")
    print("CAPTURA TERMINADA")
    print("======================================")

    for frame in sorted(frames_guardados):
        print(frame)


if __name__ == "__main__":
    main()
