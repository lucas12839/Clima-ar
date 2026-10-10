#!/usr/bin/env python3

from pathlib import Path
from playwright.sync_api import sync_playwright
import time

URL = "https://radares.hidricosargentina.gob.ar/"

OUTPUT_DIR = Path("data/radar")
REPORT_FILE = OUTPUT_DIR / "sinarame_network.txt"
SCREENSHOT_FILE = OUTPUT_DIR / "sinarame_rma10.png"
HTML_FILE = OUTPUT_DIR / "sinarame_rma10.html"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eventos = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        page = browser.new_page(
            viewport={"width": 1920, "height": 1080}
        )

        # ==============================
        # REQUESTS
        # ==============================
        def request_handler(request):
            line = (
                f"REQUEST | {request.method} | "
                f"{request.resource_type} | {request.url}"
            )
            eventos.append(line)
            print(line)

        page.on("request", request_handler)

        # ==============================
        # RESPUESTAS
        # ==============================
        def response_handler(response):
            line = (
                f"RESPONSE | {response.status} | "
                f"{response.request.resource_type} | "
                f"{response.url}"
            )
            eventos.append(line)
            print(line)

        page.on("response", response_handler)

        # ==============================
        # ERRORES JAVASCRIPT
        # ==============================
        def page_error_handler(error):
            line = f"PAGEERROR | {error}"
            eventos.append(line)
            print(line)

        page.on("pageerror", page_error_handler)

        # ==============================
        # CONSOLA
        # ==============================
        def console_handler(msg):
            line = f"CONSOLE | {msg.type} | {msg.text}"
            eventos.append(line)
            print(line)

        page.on("console", console_handler)

        # ==============================
        # ABRIR SINARAME
        # ==============================
        print("")
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

        # ==============================
        # BUSCAR SELECTOR
        # ==============================
        print("")
        print("Buscando selector de radares...")

        selects = page.locator("select")
        selector = None

        for i in range(selects.count()):

            current = selects.nth(i)

            try:
                text = current.inner_text()
            except Exception:
                continue

            if (
                "Bahía Blanca" in text
                or "Bahia Blanca" in text
                or "RMA10" in text
            ):
                selector = current
                break

        if selector is None:
            raise RuntimeError(
                "No se encontró el selector "
                "de Bahía Blanca/RMA10."
            )

        print("Selector encontrado.")

        # ==============================
        # BUSCAR RMA10
        # ==============================
        options = selector.locator("option")
        selected = False

        for i in range(options.count()):

            option = options.nth(i)

            text = option.inner_text().strip()
            value = option.get_attribute("value")

            print(
                f"Opción {i}: {text} | value={value}"
            )

            if (
                "Bahía Blanca" in text
                or "Bahia Blanca" in text
                or "RMA10" in text
            ):

                print("")
                print(
                    "SELECCIONANDO BAHÍA BLANCA (RMA10)"
                )

                selector.select_option(
                    value=value
                )

                selected = True
                break

        if not selected:
            raise RuntimeError(
                "No se pudo seleccionar "
                "Bahía Blanca RMA10."
            )

        print("")
        print("RMA10 seleccionado correctamente.")

        # IMPORTANTE:
        # NO hacemos dispatchEvent manual.
        # select_option() ya dispara el cambio.

        # ==============================
        # ESPERAR RADAR
        # ==============================
        print("")
        print("======================================")
        print("ESPERANDO RADAR RMA10")
        print("======================================")

        for second in range(40):

            time.sleep(1)

            print(
                f"Esperando: {second + 1}/40"
            )

        # ==============================
        # CAPTURA
        # ==============================
        print("")
        print("Guardando captura...")

        page.screenshot(
            path=str(SCREENSHOT_FILE),
            full_page=True
        )

        # ==============================
        # HTML
        # ==============================
        print("Guardando HTML...")

        HTML_FILE.write_text(
            page.content(),
            encoding="utf-8"
        )

        # ==============================
        # RECURSOS
        # ==============================
        print("Buscando recursos cargados...")

        resources = page.evaluate("""
            performance
                .getEntriesByType("resource")
                .map(function(x) {
                    return {
                        name: x.name,
                        initiatorType: x.initiatorType
                    };
                });
        """)

        eventos.append("")
        eventos.append(
            "========== RECURSOS =========="
        )

        for resource in resources:

            eventos.append(
                f"RESOURCE | "
                f"{resource.get('initiatorType')} | "
                f"{resource.get('name')}"
            )

        # ==============================
        # IMAGENES
        # ==============================
        print("Buscando imágenes...")

        images = page.locator("img")

        eventos.append("")
        eventos.append(
            "========== IMAGENES =========="
        )

        for i in range(images.count()):

            image = images.nth(i)

            try:
                src = image.get_attribute("src")
                alt = image.get_attribute("alt")
            except Exception:
                src = None
                alt = None

            eventos.append(
                f"IMG {i} | "
                f"src={src} | "
                f"alt={alt}"
            )

        # ==============================
        # GUARDAR INFORME
        # ==============================
        print("Guardando informe...")

        REPORT_FILE.write_text(
            "\n".join(eventos),
            encoding="utf-8"
        )

        print("")
        print("======================================")
        print("ARCHIVOS GENERADOS")
        print("======================================")

        print(REPORT_FILE)
        print(SCREENSHOT_FILE)
        print(HTML_FILE)

        browser.close()


if __name__ == "__main__":
    main()
