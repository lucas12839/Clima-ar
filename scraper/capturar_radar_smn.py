#!/usr/bin/env python3

from pathlib import Path
import time
from playwright.sync_api import sync_playwright


URL = "https://radares.hidricosargentina.gob.ar/"

OUTPUT_DIR = Path("data/radar")
HISTORICO_DIR = OUTPUT_DIR / "historico"
ACTUAL_FILE = OUTPUT_DIR / "actual.png"

MASCARA_FILE = OUTPUT_DIR / "mascara_ecos_estaticos.png"

FRAMES_INICIALES = 3

# Mucho menor que los 90 segundos anteriores
TIMEOUT_RADAR = 20000


def buscar_selector_rma10(page):

    selects = page.locator("select")

    for i in range(selects.count()):

        selector = selects.nth(i)

        try:
            texto = selector.inner_text()
        except Exception:
            continue

        if (
            "Bahía Blanca" in texto
            or "Bahia Blanca" in texto
            or "RMA10" in texto
        ):

            opciones = selector.locator("option")

            for j in range(opciones.count()):

                opcion = opciones.nth(j)

                texto_opcion = opcion.inner_text().strip()
                valor = opcion.get_attribute("value")

                if (
                    "Bahía Blanca" in texto_opcion
                    or "Bahia Blanca" in texto_opcion
                    or "RMA10" in texto_opcion
                ):

                    return selector, valor

    raise RuntimeError(
        "No se encontró el selector de Bahía Blanca/RMA10."
    )


def guardar_respuesta(respuesta):

    url = respuesta.url
    nombre = url.split("/")[-1]

    if not nombre.endswith(".png"):
        return None

    if "/cache/RMA10/" not in url:
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

    print("Seleccionando Bahía Blanca (RMA10)...")

    try:

        selector.select_option(
            value=valor_rma10
        )

    except Exception as error:

        page.remove_listener(
            "response",
            recibir_respuesta
        )

        raise error

    # Esperar como máximo 20 segundos
    inicio = time.time()

    while (
        time.time() - inicio
        < TIMEOUT_RADAR / 1000
    ):

        if respuestas:

            for respuesta in respuestas:

                try:

                    nombre = guardar_respuesta(
                        respuesta
                    )

                    if nombre:

                        page.remove_listener(
                            "response",
                            recibir_respuesta
                        )

                        return nombre

                except Exception as error:

                    print(
                        f"Error guardando respuesta: {error}"
                    )

        time.sleep(0.5)

    page.remove_listener(
        "response",
        recibir_respuesta
    )

    raise RuntimeError(
        "No se recibió una imagen PNG RMA10 "
        "dentro de 20 segundos."
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

    mascara_existe = MASCARA_FILE.exists()

    print("======================================")
    print("ClimaAR - CAPTURA RMA10")
    print("======================================")

    if mascara_existe:

        print("Máscara estática: ACTIVA")
        print("Modo: CAPTURA RÁPIDA")
        print("Frames necesarios: 1")

        frames_necesarios = 1

    else:

        print("Máscara estática: NO DISPONIBLE")
        print("Modo: CAPTURA INICIAL")
        print(
            f"Frames necesarios: {FRAMES_INICIALES}"
        )

        frames_necesarios = FRAMES_INICIALES

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

        while (
            len(frames_guardados)
            < frames_necesarios
        ):

            intento += 1

            print("")
            print("======================================")
            print(
                f"INTENTO {intento}"
            )
            print(
                f"Progreso: "
                f"{len(frames_guardados)}/"
                f"{frames_necesarios}"
            )
            print("======================================")

            try:

                page.goto(
                    URL,
                    wait_until="domcontentloaded",
                    timeout=60000
                )

                # La página necesita unos segundos
                # para construir el selector.
                page.wait_for_timeout(5000)

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
                        "NUEVO FRAME GUARDADO"
                    )

                    print(
                        f"Progreso: "
                        f"{len(frames_guardados)}/"
                        f"{frames_necesarios}"
                    )

            except Exception as error:

                print("")
                print(
                    f"Error en intento: {error}"
                )

            if (
                len(frames_guardados)
                < frames_necesarios
            ):

                # Si estamos creando la máscara,
                # comprobamos nuevamente después
                # de 15 segundos.
                print(
                    "Esperando 15 segundos "
                    "antes del siguiente intento..."
                )

                time.sleep(15)

        print("")
        print("======================================")
        print("CAPTURA TERMINADA")
        print("======================================")

        for frame in sorted(
            frames_guardados
        ):

            print(frame)

        browser.close()


if __name__ == "__main__":
    main()
