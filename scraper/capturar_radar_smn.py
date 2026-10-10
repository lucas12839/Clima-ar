#!/usr/bin/env python3

from pathlib import Path
from playwright.sync_api import sync_playwright


URL = "https://radares.hidricosargentina.gob.ar/"

OUTPUT_DIR = Path("data/radar")
HISTORICO_DIR = OUTPUT_DIR / "historico"
ACTUAL_FILE = OUTPUT_DIR / "actual.png"


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORICO_DIR.mkdir(parents=True, exist_ok=True)

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

        print("======================================")
        print("ABRIENDO SINARAME")
        print("======================================")

        page.goto(
            URL,
            wait_until="domcontentloaded",
            timeout=60000
        )

        print("Página cargada.")

        # Esperar que Blazor termine de cargar
        page.wait_for_timeout(10000)

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
        # ENCONTRAR OPCIÓN RMA10
        # ======================================

        opciones = selector.locator("option")

        valor_rma10 = None

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

                valor_rma10 = valor
                break

        if valor_rma10 is None:

            raise RuntimeError(
                "No se encontró Bahía Blanca RMA10."
            )

        print("")
        print("Seleccionando RMA10...")
        print("Esperando imagen del radar...")

        # ======================================
        # SELECCIONAR Y ESPERAR PNG
        # ======================================

        try:

            with page.expect_response(
                lambda response:
                    "/cache/RMA10/" in response.url
                    and ".png" in response.url,
                timeout=90000
            ) as respuesta_esperada:

                selector.select_option(
                    value=valor_rma10
                )

            respuesta = respuesta_esperada.value

            print("")
            print("======================================")
            print("IMAGEN RMA10 ENCONTRADA")
            print("======================================")

            print(respuesta.url)
            print(
                f"HTTP: {respuesta.status}"
            )

            # ==================================
            # DESCARGAR CONTENIDO
            # ==================================

            contenido = respuesta.body()

            if not contenido:

                raise RuntimeError(
                    "La respuesta del radar está vacía."
                )

            # Guardar actual
            ACTUAL_FILE.write_bytes(
                contenido
            )

            # Nombre original
            nombre = respuesta.url.split("/")[-1]

            historico = HISTORICO_DIR / nombre

            historico.write_bytes(
                contenido
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
                f"Tamaño: {len(contenido)} bytes"
            )

        except Exception as error:

            print("")
            print("======================================")
            print("ERROR")
            print("======================================")

            print(error)

            # Guardar captura para diagnóstico
            page.screenshot(
                path=str(
                    OUTPUT_DIR / "error_rma10.png"
                ),
                full_page=True
            )

            raise

        browser.close()


if __name__ == "__main__":
    main()
