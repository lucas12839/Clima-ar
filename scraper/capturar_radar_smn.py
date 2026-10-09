#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
import sys

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


SMN_URL = "https://www.smn.gob.ar/radar/bahiablanca"

OUTPUT_DIR = Path("data/radar")
HISTORIC_DIR = OUTPUT_DIR / "historico"

LATEST_FILE = OUTPUT_DIR / "actual.png"


def guardar_imagen(page):
    """
    Busca la imagen/canvas visible correspondiente al radar
    y guarda una captura del elemento.
    """

    candidatos = []

    # Buscar imágenes visibles
    imagenes = page.locator("img:visible")

    for i in range(imagenes.count()):
        elemento = imagenes.nth(i)

        try:
            box = elemento.bounding_box()

            if not box:
                continue

            ancho = box["width"]
            alto = box["height"]

            # Evitamos logos, iconos y elementos pequeños
            if ancho < 250 or alto < 200:
                continue

            # El radar normalmente tiene proporción cercana a cuadrada
            proporcion = ancho / alto

            if 0.60 <= proporcion <= 1.50:
                candidatos.append(
                    ("img", i, ancho * alto)
                )

        except Exception:
            continue

    # Buscar canvas visibles por si el radar se dibuja mediante canvas
    canvas = page.locator("canvas:visible")

    for i in range(canvas.count()):
        elemento = canvas.nth(i)

        try:
            box = elemento.bounding_box()

            if not box:
                continue

            ancho = box["width"]
            alto = box["height"]

            if ancho < 250 or alto < 200:
                continue

            proporcion = ancho / alto

            if 0.60 <= proporcion <= 1.50:
                candidatos.append(
                    ("canvas", i, ancho * alto)
                )

        except Exception:
            continue

    if not candidatos:
        raise RuntimeError(
            "No se encontró ninguna imagen/canvas compatible con el radar."
        )

    # Elegimos el elemento de mayor superficie
    candidatos.sort(key=lambda x: x[2], reverse=True)

    tipo, indice, superficie = candidatos[0]

    if tipo == "img":
        radar = imagenes.nth(indice)
    else:
        radar = canvas.nth(indice)

    box = radar.bounding_box()

    print("Elemento radar encontrado:")
    print(f"  tipo: {tipo}")
    print(f"  ancho: {box['width']:.0f}")
    print(f"  alto: {box['height']:.0f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORIC_DIR.mkdir(parents=True, exist_ok=True)

    # Captura del radar actual
    radar.screenshot(path=str(LATEST_FILE))

    # Histórico
    ahora = datetime.now(timezone.utc)
    timestamp = ahora.strftime("%Y%m%dT%H%M%SZ")

    historic_file = HISTORIC_DIR / f"RMA10_{timestamp}.png"

    radar.screenshot(path=str(historic_file))

    # Validación básica
    tamano = LATEST_FILE.stat().st_size

    if tamano < 5000:
        raise RuntimeError(
            f"La imagen parece inválida o vacía. Tamaño: {tamano} bytes."
        )

    print()
    print("RADAR CAPTURADO CORRECTAMENTE")
    print(f"Actual:    {LATEST_FILE}")
    print(f"Histórico: {historic_file}")
    print(f"Tamaño:    {tamano} bytes")


def main():

    print("========================================")
    print("CLIMAAR - CAPTURADOR RADAR SMN RMA10")
    print("========================================")

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        page = browser.new_page(
            viewport={
                "width": 1366,
                "height": 1000,
            }
        )

        print()
        print("1. Abriendo SMN...")

        page.goto(
            SMN_URL,
            wait_until="domcontentloaded",
            timeout=60000
        )

        print("Página cargada.")

        # Esperamos que aparezca el selector
        print()
        print("2. Buscando 'Seleccionar Radar'...")

        selector = page.get_by_text(
            "Seleccionar Radar",
            exact=True
        )

        selector.wait_for(
            state="visible",
            timeout=30000
        )

        print("Selector encontrado.")

        # Abrimos el menú
        print()
        print("3. Abriendo selector...")

        selector.click()

        # Esperamos la opción exacta de Bahía Blanca
        print()
        print("4. Buscando Bahía Blanca...")

        opcion = page.get_by_text(
            "Bahía Blanca (Buenos Aires)",
            exact=True
        )

        opcion.wait_for(
            state="visible",
            timeout=30000
        )

        print("Bahía Blanca encontrada.")

        # Seleccionamos RMA10
        print()
        print("5. Seleccionando Bahía Blanca...")

        opcion.click()

        # Esperar que cambie el contenido
        print()
        print("6. Esperando actualización del radar...")

        try:
            page.get_by_text(
                "Bahía Blanca-SINARAME",
                exact=False
            ).wait_for(
                state="visible",
                timeout=30000
            )

            print("Texto del radar RMA10 detectado.")

        except PlaywrightTimeoutError:

            print(
                "No apareció el texto exacto del radar."
            )

            # Igual damos tiempo para que termine de cargar
            page.wait_for_timeout(10000)

        # Tiempo adicional para que termine de renderizar
        page.wait_for_timeout(5000)

        print()
        print("7. Capturando imagen...")

        guardar_imagen(page)

        browser.close()


if __name__ == "__main__":
    try:
        main()

    except Exception as e:

        print()
        print("========================================")
        print("ERROR EN CAPTURADOR SMN")
        print("========================================")
        print(str(e))

        sys.exit(1)
