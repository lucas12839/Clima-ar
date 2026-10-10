#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
from playwright.sync_api import sync_playwright
import time


URL = "https://radares.hidricosargentina.gob.ar/"

OUTPUT_DIR = Path("data/radar")
HISTORICO_DIR = OUTPUT_DIR / "historico"
ACTUAL_FILE = OUTPUT_DIR / "actual.png"


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORICO_DIR.mkdir(parents=True, exist_ok=True)

    radar_encontrado = False
    archivo_historico = None

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)

        page = browser.new_page(
            viewport={
                "width": 1920,
                "height": 1080
            }
        )

        # ======================================
        # CAPTURAR RESPUESTAS DEL RADAR
        # ======================================

        def recibir_respuesta(response):

            nonlocal radar_encontrado
            nonlocal archivo_historico

            url = response.url

            if (
                "/cache/RMA10/" in url
                and url.endswith(".png")
            ):

                print("")
                print("======================================")
                print("RADAR RMA10 ENCONTRADO")
                print("======================================")
                print(url)

                try:

                    contenido = response.body()

                    if len(contenido) == 0:
                        print("La imagen llegó vacía.")
                        return

                    # Guardar imagen actual
                    ACTUAL_FILE.write_bytes(
                        contenido
                    )

                    # Obtener nombre original
                    nombre = url.split("/")[-1]

                    archivo_historico = (
                        HISTORICO_DIR / nombre
                    )

                    archivo_historico.write_bytes(
                        contenido
                    )

                    radar_encontrado = True

                    print("")
                    print("RADAR DESCARGADO CORRECTAMENTE")
                    print(
                        f"Tamaño: {len(contenido)} bytes"
                    )
                    print(
                        f"Actual: {ACTUAL_FILE}"
                    )
                    print(
                        f"Histórico: {archivo_historico}"
                    )

                except Exception as error:

                    print(
                        f"Error guardando radar: {error}"
                    )

        page.on(
            "response",
            recibir_respuesta
        )

        # ======================================
        # ABRIR SINARAME
        # ======================================

        print("======================================")
        print("ABRIENDO SINARAME")
        print("======================================")

        page.goto(
            URL,
            wait_until="domcontentloaded",
            timeout=60000
        )

        print("Página cargada.")

        time.sleep(8)

        # ======================================
        # BUSCAR SELECTOR
        # ======================================

        print("")
        print("Buscando selector de radares...")

        selects = page.locator("select")

        selector = None

        for i in range(selects.count()):

            actual = selects.nth(i)

            try:
                texto = actual.inner_text()
            except Exception:
                continue

            if (
                "Bahía Blanca" in texto
                or "Bahia Blanca" in texto
                or "RMA10" in texto
            ):

                selector = actual
                break

        if selector is None:

            raise RuntimeError(
                "No se encontró el selector de RMA10."
            )

        print("Selector encontrado.")

        # ======================================
        # BUSCAR BAHÍA BLANCA
        # ======================================

        opciones = selector.locator("option")

        seleccionada = False

        for i in range(opciones.count()):

            opcion = opciones.nth(i)

            texto = opcion.inner_text().strip()
            valor = opcion.get_attribute("value")

            print(
                f"Opción {i}: {texto}"
            )

            if (
                "Bahía Blanca" in texto
                or "Bahia Blanca" in texto
                or "RMA10" in texto
            ):

                print("")
                print(
                    "Seleccionando Bahía Blanca (RMA10)..."
                )

                selector.select_option(
                    value=valor
                )

                seleccionada = True

                break

        if not seleccionada:

            raise RuntimeError(
                "No se pudo seleccionar RMA10."
            )

        print(
            "RMA10 seleccionado."
        )

        # ======================================
        # ESPERAR IMAGEN
        # ======================================

        print("")
        print("Esperando imagen del radar...")

        for segundo in range(40):

            if radar_encontrado:
                break

            time.sleep(1)

            print(
                f"Esperando: {segundo + 1}/40"
            )

        # ======================================
        # RESULTADO
        # ======================================

        print("")
        print("======================================")

        if radar_encontrado:

            print(
                "ÉXITO: RADAR RMA10 DESCARGADO"
            )

            print(
                f"Archivo: {ACTUAL_FILE}"
            )

            print(
                f"Histórico: {archivo_historico}"
            )

        else:

            print(
                "ERROR: NO SE ENCONTRÓ LA IMAGEN RMA10"
            )

            raise RuntimeError(
                "SINARAME no entregó una imagen "
                "PNG de RMA10 durante la prueba."
            )

        print("======================================")

        browser.close()


if __name__ == "__main__":
    main()
