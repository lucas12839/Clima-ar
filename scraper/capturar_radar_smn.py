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

    print("======================================")
    print("CLIMAAR - DIAGNOSTICO SINARAME RMA10")
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

        # ---------------------------------
        # CAPTURAR REQUESTS
        # ---------------------------------

        def registrar_request(request):

            linea = (
                f"REQUEST | {request.method} | "
                f"{request.resource_type} | "
                f"{request.url}"
            )

            eventos.append(linea)

            print(linea)

        page.on("request", registrar_request)

        # ---------------------------------
        # CAPTURAR RESPONSES
        # ---------------------------------

        def registrar_response(response):

            linea = (
                f"RESPONSE | {response.status} | "
                f"{response.request.resource_type} | "
                f"{response.url}"
            )

            eventos.append(linea)

            print(linea)

        page.on("response", registrar_response)

        # ---------------------------------
        # WEBSOCKETS
        # ---------------------------------

        def registrar_websocket(ws):

            print()
            print("WEBSOCKET:")
            print(ws.url)

            eventos.append(
                f"WEBSOCKET OPEN | {ws.url}"
            )

            def recibido(mensaje):

                texto = str(mensaje)

                if len(texto) > 10000:
                    texto = texto[:10000] + " [TRUNCADO]"

                eventos.append(
                    f"WEBSOCKET RECEIVED | {ws.url} | {texto}"
                )

                print(
                    "WEBSOCKET RECEIVED:",
                    texto[:2000]
                )

            def enviado(mensaje):

                texto = str(mensaje)

                if len(texto) > 10000:
                    texto = texto[:10000] + " [TRUNCADO]"

                eventos.append(
                    f"WEBSOCKET SENT | {ws.url} | {texto}"
                )

            ws.on("framereceived", recibido)
            ws.on("framesent", enviado)

        page.on("websocket", registrar_websocket)

        # ---------------------------------
        # ABRIR SINARAME
        # ---------------------------------

        print()
        print("Abriendo SINARAME...")

        page.goto(
            URL,
            wait_until="domcontentloaded",
            timeout=60000
        )

        print("Página cargada.")

        time.sleep(8)

        # ---------------------------------
        # BUSCAR SELECT
        # ---------------------------------

        print()
        print("Buscando selector de radar...")

        selects = page.locator("select")

        cantidad_selects = selects.count()

        print(
            "Selectores encontrados:",
            cantidad_selects
        )

        selector = None

        # Buscar el select que contenga Bahía Blanca
        for i in range(cantidad_selects):

            actual = selects.nth(i)

            try:

                texto = actual.inner_text()

                print(
                    f"SELECT {i}:",
                    texto[:500]
                )

                if (
                    "Bahía Blanca" in texto
                    or
                    "Bahia Blanca" in texto
                    or
                    "RMA10" in texto
                ):

                    selector = actual

                    print()
                    print(
                        "Selector RMA10 encontrado:",
                        i
                    )

                    break

            except Exception:
                pass

        if selector is None:

            raise RuntimeError(
                "No se encontró el selector que contiene Bahía Blanca/RMA10."
            )

        # ---------------------------------
        # MOSTRAR OPCIONES
        # ---------------------------------

        print()
        print("Opciones:")

        opciones = selector.locator("option")

        for i in range(opciones.count()):

            option = opciones.nth(i)

            try:

                texto = option.inner_text()
                valor = option.get_attribute("value")

                print(
                    f"{i}: {texto} | value={valor}"
                )

            except Exception:
                pass

        # ---------------------------------
        # SELECCIONAR POR TEXTO
        # ---------------------------------

        print()
        print("======================================")
        print("SELECCIONANDO BAHÍA BLANCA RMA10")
        print("======================================")

        seleccionada = False

        for i in range(opciones.count()):

            option = opciones.nth(i)

            try:

                texto = option.inner_text().strip()

                if (
                    "Bahía Blanca" in texto
                    or
                    "Bahia Blanca" in texto
                    or
                    "RMA10" in texto
                ):

                    valor = option.get_attribute("value")

                    print(
                        "Texto encontrado:",
                        texto
                    )

                    print(
                        "Value:",
                        valor
                    )

                    selector.select_option(
                        value=valor
                    )

                    seleccionada = True

                    break

            except Exception:
                pass

        if not seleccionada:

            raise RuntimeError(
                "No se pudo seleccionar Bahía Blanca RMA10."
            )

        print("RMA10 seleccionado correctamente.")

        # ---------------------------------
        # FORZAR EVENTO CHANGE
        # ---------------------------------

        page.evaluate("""
            (selector) => {
                selector.dispatchEvent(
                    new Event("change", {
                        bubbles: true
                    })
                );
            }
        """, selector.element_handle())

        print("Evento change enviado.")

        # ---------------------------------
        # ESPERAR RADAR
        # ---------------------------------

        print()
        print("Esperando tráfico del radar...")

        for segundo in range(30):

            time.sleep(1)

            print(
                f"Esperando: {segundo + 1}/30"
            )

        # ---------------------------------
        # RECURSOS DEL NAVEGADOR
        # ---------------------------------

        print()
        print("======================================")
        print("RECURSOS CARGADOS")
        print("======================================")

        resources = page.evaluate("""
            performance
                .getEntriesByType("resource")
                .map(x => ({
                    name: x.name,
                    initiatorType: x.initiatorType
                }))
        """)

        eventos.append("")
        eventos.append(
            "===== PERFORMANCE RESOURCES ====="
        )

        for resource in resources:

            name = resource.get("name", "")
            tipo = resource.get(
                "initiatorType",
                ""
            )

            linea = (
                f"RESOURCE | {tipo} | {name}"
            )

            eventos.append(linea)

            print(linea)

        # ---------------------------------
        # IMÁGENES
        # ---------------------------------

        print()
        print("======================================")
        print("IMÁGENES")
        print("======================================")

        imagenes = page.locator("img")

        cantidad = imagenes.count()

        print(
            "Cantidad:",
            cantidad
        )

        eventos.append("")
        eventos.append(
            "===== IMG ELEMENTS ====="
        )

        for i in range(cantidad):

            try:

                img = imagenes.nth(i)

                src = img.get_attribute("src")
                alt = img.get_attribute("alt")

                linea = (
                    f"IMG | src={src} | alt={alt}"
                )

                eventos.append(linea)

                print(linea)

            except Exception:
                pass

        # ---------------------------------
        # TEXTO FINAL
        # ---------------------------------

        texto = page.locator(
            "body"
        ).inner_text()

        eventos.append("")
        eventos.append(
            "===== TEXTO VISIBLE ====="
        )

        eventos.append(
            texto[:30000]
        )

        # ---------------------------------
        # GUARDAR HTML
        # ---------------------------------

        HTML_FILE.write_text(
            page.content(),
            encoding="utf-8"
        )

        # ---------------------------------
        # SCREENSHOT
        # ---------------------------------

        page.screenshot(
            path=str(SCREENSHOT_FILE),
            full_page=True
        )

        browser.close()

    # ---------------------------------
    # GUARDAR REPORTE
    # ---------------------------------

    REPORT_FILE.write_text(
        "\n".join(eventos),
        encoding="utf-8"
    )

    print()
    print("======================================")
    print("DIAGNOSTICO TERMINADO")
    print("======================================")

    print(
        "Reporte:",
        REPORT_FILE
    )

    print(
        "Captura:",
        SCREENSHOT_FILE
    )

    print(
        "HTML:",
        HTML_FILE
    )


if __name__ == "__main__":
    main()
