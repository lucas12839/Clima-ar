#!/usr/bin/env python3

from pathlib import Path
import os
import time

from playwright.sync_api import sync_playwright


URL = "https://radares.hidricosargentina.gob.ar/"

OUTPUT_DIR = Path("data/radar")
HISTORICO_DIR = OUTPUT_DIR / "historico"
ACTUAL_FILE = OUTPUT_DIR / "actual.png"

MASCARA_FILE = OUTPUT_DIR / "mascara_ecos_estaticos.png"

# Solo se utilizan cuando se está creando la máscara inicial
FRAMES_INICIALES = 3

# Tiempo máximo esperando la respuesta del radar
TIMEOUT_RADAR = 20000

# Tiempo máximo para cargar la página
PAGE_TIMEOUT = 30000


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


def guardar_respuesta(respuesta):

    url = respuesta.url

    nombre = url.split("/")[-1]

    if (
        not nombre.endswith(".png")
        or "/cache/RMA10/" not in url
    ):
        return None

    contenido = respuesta.body()

    if not contenido:
        return None

    historico = HISTORICO_DIR / nombre

    historico.write_bytes(contenido)

    ACTUAL_FILE.write_bytes(contenido)

    print("")
    print("======================================")
    print("RADAR RMA10 ENCONTRADO")
    print("======================================")
    print(f"URL: {url}")
    print(f"HTTP: {respuesta.status}")
    print(f"Archivo: {historico}")
    print(f"Tamaño: {len(contenido)} bytes")

    return nombre


def capturar_frame(page):

    selector, valor_rma10 = buscar_selector_rma10(page)

    respuestas = []

    def recibir_respuesta(response):

        if (
            "/cache/RMA10/" in response.url
            and response.url.endswith(".png")
        ):
            print(
                f"Respuesta radar detectada: {response.url}"
            )

            respuestas.append(response)

    page.on(
        "response",
        recibir_respuesta
    )

    try:

        print(
            "Seleccionando Bahía Blanca (RMA10)..."
        )

        selector.select_option(
            value=valor_rma10
        )

        inicio = time.monotonic()

        while (
            time.monotonic() - inicio
            < TIMEOUT_RADAR / 1000
        ):

            for respuesta in respuestas:

                try:

                    nombre = guardar_respuesta(
                        respuesta
                    )

                    if nombre:
                        return nombre

                except Exception as error:

                    print(
                        f"Error guardando respuesta: {error}"
                    )

            page.wait_for_timeout(250)

        raise RuntimeError(
            "No se recibió una imagen PNG RMA10 "
            "dentro de 20 segundos."
        )

    finally:

        page.remove_listener(
            "response",
            recibir_respuesta
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

    # El workflow nos indica explícitamente el modo.
    modo = os.environ.get(
        "CLIMAAR_CAPTURE_MODE",
        ""
    ).strip().lower()

    # Compatibilidad de seguridad:
    # si no llega la variable, se decide por
    # existencia de la máscara.
    if modo not in {"rapido", "inicial"}:

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
    print(
        f"Frames necesarios: {frames_necesarios}"
    )
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

        intento = 0

        try:

            while (
                len(frames_guardados)
                < frames_necesarios
            ):

                intento += 1

                print("")
                print(
                    "======================================"
                )
                print(
                    f"INTENTO {intento}"
                )
                print(
                    f"Progreso: "
                    f"{len(frames_guardados)}/"
                    f"{frames_necesarios}"
                )
                print(
                    "======================================"
                )

                try:

                    page.goto(
                        URL,
                        wait_until="domcontentloaded",
                        timeout=PAGE_TIMEOUT
                    )

                    # Tiempo para que la página
                    # termine de construir los selectores.
                    page.wait_for_timeout(3000)

                    nombre = capturar_frame(page)

                    if nombre in frames_guardados:

                        print(
                            f"Frame repetido: {nombre}"
                        )

                    else:

                        frames_guardados.add(
                            nombre
                        )

                        print(
                            f"NUEVO FRAME GUARDADO: "
                            f"{nombre}"
                        )

                except Exception as error:

                    print(
                        f"Error en intento: {error}"
                    )

                if (
                    len(frames_guardados)
                    < frames_necesarios
                ):

                    if modo == "inicial":
                        espera = 15
                    else:
                        espera = 3

                    print(
                        f"Esperando {espera} "
                        "segundos antes del "
                        "siguiente intento..."
                    )

                    time.sleep(espera)

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
