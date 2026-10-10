#!/usr/bin/env python3

from pathlib import Path
import time
from playwright.sync_api import sync_playwright

URL = "https://radares.hidricosargentina.gob.ar/"
OUTPUT_DIR = Path("data/radar")
HISTORICO_DIR = OUTPUT_DIR / "historico"
ACTUAL_FILE = OUTPUT_DIR / "actual.png"

FRAMES_NECESARIOS = 3
ESPERA_ENTRE_INTENTOS = 60
TIMEOUT_RESPUESTA = 90000
MAX_INTENTOS = 40


def buscar_selector_rma10(page):
    selects = page.locator("select")
    selector = None

    for i in range(selects.count()):
        actual = selects.nth(i)
        try:
            texto = actual.inner_text()
        except Exception:
            continue

        if "Bahía Blanca" in texto or "Bahia Blanca" in texto or "RMA10" in texto:
            selector = actual
            break

    if selector is None:
        raise RuntimeError("No se encontró el selector de RMA10.")

    opciones = selector.locator("option")
    valor_rma10 = None

    for i in range(opciones.count()):
        opcion = opciones.nth(i)
        texto = opcion.inner_text().strip()
        valor = opcion.get_attribute("value")

        print(f"Opción {i}: {texto}")

        if "Bahía Blanca" in texto or "Bahia Blanca" in texto or "RMA10" in texto:
            valor_rma10 = valor
            break

    if valor_rma10 is None:
        raise RuntimeError("No se encontró Bahía Blanca RMA10.")

    return selector, valor_rma10


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORICO_DIR.mkdir(parents=True, exist_ok=True)

    frames_guardados = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})

        print("======================================")
        print("CAPTURA DE 3 FRAMES RMA10")
        print("======================================")
        print("Se guardarán únicamente frames distintos.")

        intento = 0

        while len(frames_guardados) < FRAMES_NECESARIOS:
            intento += 1

            if intento > MAX_INTENTOS:
                raise RuntimeError(
                    f"No se pudieron obtener {FRAMES_NECESARIOS} frames "
                    f"distintos después de {MAX_INTENTOS} intentos."
                )

            print("")
            print("======================================")
            print(f"INTENTO {intento}")
            print(f"Frames distintos: {len(frames_guardados)}/{FRAMES_NECESARIOS}")
            print("======================================")

            page.goto(URL, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(10000)

            selector, valor_rma10 = buscar_selector_rma10(page)

            print("Seleccionando RMA10...")
            print("Esperando imagen del radar...")

            try:
                with page.expect_response(
                    lambda response:
                        "/cache/RMA10/" in response.url and ".png" in response.url,
                    timeout=TIMEOUT_RESPUESTA
                ) as respuesta_esperada:
                    selector.select_option(value=valor_rma10)

                respuesta = respuesta_esperada.value
                url_imagen = respuesta.url
                nombre = url_imagen.split("/")[-1]

                print(f"Imagen encontrada: {url_imagen}")
                print(f"HTTP: {respuesta.status}")

                if nombre in frames_guardados:
                    print(f"Frame repetido: {nombre}")
                else:
                    contenido = respuesta.body()

                    if not contenido:
                        raise RuntimeError("La respuesta del radar está vacía.")

                    historico = HISTORICO_DIR / nombre
                    historico.write_bytes(contenido)
                    ACTUAL_FILE.write_bytes(contenido)

                    frames_guardados.add(nombre)

                    print("NUEVO FRAME GUARDADO")
                    print(f"Archivo: {historico}")
                    print(f"Tamaño: {len(contenido)} bytes")
                    print(f"Progreso: {len(frames_guardados)}/{FRAMES_NECESARIOS}")

            except Exception as error:
                print(f"No apareció un frame nuevo en este intento: {error}")

            if len(frames_guardados) < FRAMES_NECESARIOS:
                print(f"Esperando {ESPERA_ENTRE_INTENTOS} segundos...")
                time.sleep(ESPERA_ENTRE_INTENTOS)

        print("")
        print("======================================")
        print("3 FRAMES DISTINTOS OBTENIDOS")
        print("======================================")

        for frame in sorted(frames_guardados):
            print(frame)

        print("")
        print("Listos para generar la máscara de ecos estáticos.")

        browser.close()


if __name__ == "__main__":
    main()
