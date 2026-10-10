#!/usr/bin/env python3

from pathlib import Path
from playwright.sync_api import sync_playwright
import time

URL = "https://radares.hidricosargentina.gob.ar/"

OUTPUT_DIR = Path("data/radar")
REPORT_FILE = OUTPUT_DIR / "sinarame_network.txt"
SCREENSHOT_FILE = OUTPUT_DIR / "sinarame_rma10.png"


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    requests_found = []

    print("======================================")
    print("CLIMAAR - CAPTURA SINARAME RMA10")
    print("======================================")

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

        def registrar_request(request):

            url = request.url.lower()

            palabras = [
                "rma10",
                "radar",
                ".png",
                ".jpg",
                ".jpeg",
                "cache",
                "image",
                "frame"
            ]

            if any(palabra in url for palabra in palabras):

                linea = f"REQUEST {request.method} {request.url}"

                print(linea)

                requests_found.append(linea)

        def registrar_response(response):

            url = response.url.lower()

            palabras = [
                "rma10",
                "radar",
                ".png",
                ".jpg",
                ".jpeg",
                "cache",
                "image",
                "frame"
            ]

            if any(palabra in url for palabra in palabras):

                linea = (
                    f"RESPONSE {response.status} "
                    f"{response.url}"
                )

                print(linea)

                requests_found.append(linea)

        page.on("request", registrar_request)
        page.on("response", registrar_response)

        print()
        print("Abriendo SINARAME...")

        page.goto(
            URL,
            wait_until="networkidle",
            timeout=60000
        )

        print("Página cargada.")

        time.sleep(3)

        print()
        print("Buscando selector RMA10...")

        selector = page.locator("#radar-selector")

        selector.wait_for(
            state="visible",
            timeout=30000
        )

        print("Selector encontrado.")

        print()
        print("Seleccionando Bahía Blanca (RMA10)...")

        selector.select_option("RMA10")

        print("RMA10 seleccionado.")

        print()
        print("Esperando carga del radar...")

        time.sleep(15)

        # Guardar captura visual
        page.screenshot(
            path=str(SCREENSHOT_FILE),
            full_page=True
        )

        print()
        print("Captura guardada:")
        print(SCREENSHOT_FILE)

        # Buscar imágenes visibles
        print()
        print("IMAGENES EN LA PAGINA:")

        imagenes = page.locator("img")

        cantidad = imagenes.count()

        print("Cantidad:", cantidad)

        for i in range(cantidad):

            try:

                src = imagenes.nth(i).get_attribute("src")

                if src:
                    linea = f"IMG {src}"

                    print(linea)
                    requests_found.append(linea)

            except Exception:
                pass

        # Guardar HTML final
        html_file = OUTPUT_DIR / "sinarame_rma10.html"

        html_file.write_text(
            page.content(),
            encoding="utf-8"
        )

        print()
        print("HTML final guardado:")
        print(html_file)

        browser.close()

    # Eliminar duplicados manteniendo orden
    unicas = []

    for item in requests_found:

        if item not in unicas:
            unicas.append(item)

    REPORT_FILE.write_text(
        "\n".join(unicas),
        encoding="utf-8"
    )

    print()
    print("======================================")
    print("CAPTURA TERMINADA")
    print("======================================")
    print()
    print("Peticiones encontradas:", len(unicas))
    print("Reporte:", REPORT_FILE)
    print()


if __name__ == "__main__":
    main()
