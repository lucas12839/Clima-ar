#!/usr/bin/env python3

from pathlib import Path
from playwright.sync_api import sync_playwright


URL = "https://radares.hidricosargentina.gob.ar/"

OUTPUT_DIR = Path("data/radar")
HISTORICO_DIR = OUTPUT_DIR / "historico"
ACTUAL_FILE = OUTPUT_DIR / "actual.png"

MASCARA_FILE = OUTPUT_DIR / "mascara_ecos_estaticos.png"

FRAMES_INICIALES = 3
TIMEOUT_RESPUESTA = 90000


def buscar_selector_rma10(page):

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

    opciones = selector.locator("option")
    valor_rma10 = None

    for i in range(opciones.count()):

        opcion = opciones.nth(i)

        texto = opcion.inner_text().strip()
        valor = opcion.get_attribute("value")

        print(f"Opción {i}: {texto}")

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

    return selector, valor_rma10


def capturar_frame(page):

    selector, valor_rma10 = buscar_selector_rma10(page)

    print("Seleccionando RMA10...")
    print("Esperando imagen del radar...")

    with page.expect_response(
        lambda response:
            "/cache/RMA10/" in response.url
            and ".png" in response.url,
        timeout=TIMEOUT_RESPUESTA
    ) as respuesta_esperada:

        selector.select_option(
            value=valor_rma10
        )

    respuesta = respuesta_esperada.value

    url_imagen = respuesta.url
    nombre = url_imagen.split("/")[-1]

    print(f"Imagen encontrada: {url_imagen}")
    print(f"HTTP: {respuesta.status}")

    contenido = respuesta.body()

    if not contenido:
        raise RuntimeError(
            "La respuesta del radar está vacía."
        )

    historico = HISTORICO_DIR / nombre

    historico.write_bytes(contenido)
    ACTUAL_FILE.write_bytes(contenido)

    print("FRAME GUARDADO")
    print(f"Archivo: {historico}")
    print(f"Tamaño: {len(contenido)} bytes")

    return nombre


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    HISTORICO_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # DETERMINAR MODO
    # ========================================================

    mascara_existe = MASCARA_FILE.exists()

    if mascara_existe:

        print("======================================")
        print("CAPTURA RÁPIDA RMA10")
        print("======================================")
        print("Máscara estática encontrada.")
        print("Se capturará solamente 1 frame.")
        print("No se esperarán 3 actualizaciones.")

    else:

        print("======================================")
        print("CAPTURA INICIAL DE RMA10")
        print("======================================")
        print("No existe máscara estática.")
        print(f"Se capturarán {FRAMES_INICIALES} frames distintos.")

    # ========================================================
    # NAVEGADOR
    # ========================================================

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

        # ====================================================
        # MODO RÁPIDO
        # ====================================================

        if mascara_existe:

            page.goto(
                URL,
                wait_until="domcontentloaded",
                timeout=60000
            )

            page.wait_for_timeout(5000)

            nombre = capturar_frame(page)

            print("")
            print("======================================")
            print("CAPTURA RÁPIDA TERMINADA")
            print("======================================")
            print(f"Frame: {nombre}")

        # ====================================================
        # MODO INICIAL
        # ====================================================

        else:

            frames_guardados = set()

            while len(frames_guardados) < FRAMES_INICIALES:

                print("")
                print("======================================")
                print(
                    f"PROGRESO: "
                    f"{len(frames_guardados)}/"
                    f"{FRAMES_INICIALES}"
                )
                print("======================================")

                page.goto(
                    URL,
                    wait_until="domcontentloaded",
                    timeout=60000
                )

                page.wait_for_timeout(5000)

                nombre = capturar_frame(page)

                if nombre in frames_guardados:

                    print(
                        f"Frame repetido: {nombre}"
                    )

                    print(
                        "Esperando 30 segundos "
                        "para comprobar si el radar actualizó..."
                    )

                    page.wait_for_timeout(30000)

                else:

                    frames_guardados.add(nombre)

                    print(
                        "NUEVO FRAME"
                    )

                    print(
                        f"Progreso: "
                        f"{len(frames_guardados)}/"
                        f"{FRAMES_INICIALES}"
                    )

            print("")
            print("======================================")
            print("3 FRAMES DISTINTOS OBTENIDOS")
            print("======================================")

            for frame in sorted(
                frames_guardados
            ):
                print(frame)

        browser.close()


if __name__ == "__main__":
    main()
