#!/usr/bin/env python3

from pathlib import Path
import time

from playwright.sync_api import sync_playwright


URL = "https://radares.hidricosargentina.gob.ar/"
OUTPUT_DIR = Path("data/radar")
DIAGNOSTICO = OUTPUT_DIR / "diagnostico_sinarame.png"


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("======================================")
    print("ClimaAR - DIAGNOSTICO SINARAME")
    print("======================================")

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)

        page = browser.new_page(
            viewport={
                "width": 1920,
                "height": 1080
            }
        )

        requests = []
        websockets = []

        def detectar_request(request):
            url = request.url

            if "RMA10" in url or "/cache/" in url:
                requests.append(url)
                print("REQUEST:", url)

        def detectar_websocket(ws):
            websockets.append(ws.url)
            print("WEBSOCKET:", ws.url)

        page.on("request", detectar_request)
        page.on("websocket", detectar_websocket)

        try:

            print("1. Abriendo SINARAME...")

            page.goto(
                URL,
                wait_until="domcontentloaded",
                timeout=20000
            )

            print("Página cargada:", page.url)

            page.wait_for_timeout(5000)

            print("")
            print("2. Buscando selector RMA10...")

            selects = page.locator("select")
            print("Cantidad de SELECT:", selects.count())

            encontrado = False

            for i in range(selects.count()):

                selector = selects.nth(i)

                try:
                    texto = selector.inner_text()
                except Exception:
                    continue

                print(f"SELECT {i}: {texto[:500]}")

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

                        print(
                            f"  OPCION {j}: "
                            f"{texto_opcion} | value={valor}"
                        )

                        if (
                            valor
                            and (
                                "Bahía Blanca" in texto_opcion
                                or "Bahia Blanca" in texto_opcion
                                or "RMA10" in texto_opcion
                            )
                        ):

                            print("")
                            print("3. SELECCIONANDO RMA10...")
                            print(
                                f"Valor seleccionado: {valor}"
                            )

                            selector.select_option(
                                value=valor
                            )

                            encontrado = True
                            break

                if encontrado:
                    break

            if not encontrado:
                print("ERROR: NO SE ENCONTRÓ RMA10.")

            print("")
            print("4. Esperando respuesta del sitio...")

            page.wait_for_timeout(5000)

            print("")
            print("5. BUSCANDO IMÁGENES EN EL HTML")
            print("======================================")

            imagenes = page.locator("img")

            print(
                "Cantidad de imágenes:",
                imagenes.count()
            )

            for i in range(imagenes.count()):

                img = imagenes.nth(i)

                try:
                    src = img.get_attribute("src")
                    alt = img.get_attribute("alt")

                    if src:
                        print(
                            f"IMG {i}: "
                            f"src={src} "
                            f"alt={alt}"
                        )

                except Exception:
                    pass

            print("")
            print("6. TEXTO RELEVANTE DE LA PÁGINA")
            print("======================================")

            texto = page.locator("body").inner_text()

            for linea in texto.splitlines():

                linea = linea.strip()

                if (
                    "Fecha frame" in linea
                    or "Última actualización" in linea
                    or "Bahía Blanca" in linea
                    or "RMA10" in linea
                    or "Reconect" in linea
                    or "sesión" in linea
                    or "servidor" in linea
                    or "error" in linea.lower()
                ):
                    print(linea)

            print("")
            print("7. WEBSOCKETS")
            print("======================================")

            if websockets:
                for ws in websockets:
                    print(ws)
            else:
                print("NO SE DETECTARON WEBSOCKETS")

            print("")
            print("8. REQUESTS RMA10/CACHE")
            print("======================================")

            if requests:
                for url in requests:
                    print(url)
            else:
                print("NO SE DETECTARON REQUESTS RMA10/CACHE")

            print("")
            print("9. GUARDANDO CAPTURA DE PANTALLA...")

            page.screenshot(
                path=str(DIAGNOSTICO),
                full_page=True
            )

            print(
                "Captura:",
                DIAGNOSTICO
            )

            print("")
            print("======================================")
            print("DIAGNOSTICO TERMINADO")
            print("======================================")

        except Exception as error:

            print("")
            print("ERROR GENERAL:")
            print(error)

        finally:

            browser.close()


if __name__ == "__main__":
    main()
