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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Guardando captura de debug...")

    try:
        page.screenshot(
            path=str(OUTPUT_DIR / "debug_smn.png"),
            full_page=True
        )
        print("OK: debug_smn.png")
    except Exception as e:
        print(f"No se pudo guardar screenshot: {e}")

    try:
        html = page.content()

        (OUTPUT_DIR / "debug_smn.html").write_text(
            html,
            encoding="utf-8"
        )

        print("OK: debug_smn.html")

    except Exception as e:
        print(f"No se pudo guardar HTML: {e}")


def seleccionar_bahia_blanca(page):

    print("Buscando selector de radar...")

    # Buscamos el texto sin exigir coincidencia exacta.
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

    page.wait_for_timeout(1000)

    print("Buscando Bahía Blanca...")

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

    page.wait_for_timeout(7000)

    imagenes = page.locator("img:visible")

    candidatos = []

    cantidad = imagenes.count()

    print(f"Imágenes visibles encontradas: {cantidad}")

    for i in range(cantidad):

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
            print("======================================")
            print("ERROR EN CAPTURA")
            print("======================================")
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
