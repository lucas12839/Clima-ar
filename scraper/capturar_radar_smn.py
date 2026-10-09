#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
import sys

from playwright.sync_api import sync_playwright


SMN_URL = "https://www.smn.gob.ar/radar/bahiablanca"

OUTPUT_DIR = Path("data/radar")
HISTORIC_DIR = OUTPUT_DIR / "historico"
LATEST_FILE = OUTPUT_DIR / "actual.png"


def guardar_debug(page):
    """Guarda información útil si el SMN cambia la página."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    page.screenshot(
        path=str(OUTPUT_DIR / "debug_smn.png"),
        full_page=True
    )

    html = page.content()

    (OUTPUT_DIR / "debug_smn.html").write_text(
        html,
        encoding="utf-8"
    )

    print("Debug guardado:")
    print("  data/radar/debug_smn.png")
    print("  data/radar/debug_smn.html")


def seleccionar_bahia_blanca(page):

    print("Buscando selector de radar...")

    # Primero buscamos cualquier elemento visible que contenga
    # el texto "Seleccionar Radar".
    selector = page.locator(
        "text=Seleccionar Radar"
    ).first

    selector.wait_for(
        state="visible",
        timeout=60000
    )

    print("Selector encontrado.")

    selector.click()

    print("Selector abierto.")

    # Buscamos la opción exacta de Bahía Blanca.
    opcion = page.locator(
        "text=Bahía Blanca (Buenos Aires)"
    ).first

    opcion.wait_for(
        state="visible",
        timeout=30000
    )

    print("Bahía Blanca encontrada.")

    opcion.click()

    print("Bahía Blanca seleccionada.")


def encontrar_radar(page):

    print("Esperando imagen de RMA10...")

    # Esperamos un poco para que el visor cambie.
    page.wait_for_timeout(7000)

    # Buscamos imágenes grandes visibles.
    imagenes = page.locator("img:visible")

    candidatos = []

    for i in range(imagenes.count()):

        elemento = imagenes.nth(i)

        try:

            box = elemento.bounding_box()

            if not box:
                continue

            ancho = box["width"]
            alto = box["height"]

            if ancho < 250 or alto < 200:
                continue

            area = ancho * alto

            candidatos.append(
                (elemento, area, ancho, alto)
            )

        except Exception:
            continue

    if not candidatos:
        raise RuntimeError(
            "No se encontró ninguna imagen grande visible."
        )

    candidatos.sort(
        key=lambda x: x[1],
        reverse=True
    )

    radar, area, ancho, alto = candidatos[0]

    print("Imagen candidata encontrada:")
    print(f"  ancho: {ancho:.0f}")
    print(f"  alto:  {alto:.0f}")

    return radar


def capturar():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    HISTORIC_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        page = browser.new_page(
            viewport={
                "width": 390,
                "height": 844
            }
        )

        try:

            print()
            print("======================================")
            print("CLIMAAR - CAPTURADOR SMN RMA10")
            print("======================================")
            print()

            print("Abriendo:")
            print(SMN_URL)

            page.goto(
                SMN_URL,
                wait_until="domcontentloaded",
                timeout=60000
            )

            print("Página cargada.")

            page.wait_for_timeout(5000)

            seleccionar_bahia_blanca(page)

            radar = encontrar_radar(page)

            print("Capturando radar...")

            radar.screenshot(
                path=str(LATEST_FILE)
            )

            timestamp = datetime.now(
                timezone.utc
            ).strftime(
                "%Y%m%dT%H%M%SZ"
            )

            historic_file = (
                HISTORIC_DIR /
                f"RMA10_{timestamp}.png"
            )

            radar.screenshot(
                path=str(historic_file)
            )

            tamano = LATEST_FILE.stat().st_size

            if tamano < 5000:
                raise RuntimeError(
                    f"La captura parece inválida: {tamano} bytes."
                )

            print()
            print("======================================")
            print("RADAR CAPTURADO CORRECTAMENTE")
            print("======================================")
            print()
            print(f"Actual: {LATEST_FILE}")
            print(f"Histórico: {historic_file}")
            print(f"Tamaño: {tamano} bytes")

        except Exception as e:

            print()
            print("ERROR:")
            print(str(e))
            print()

            guardar_debug(page)

            raise

        finally:

            browser.close()


if __name__ == "__main__":

    try:
        capturar()

    except Exception:

        sys.exit(1)
