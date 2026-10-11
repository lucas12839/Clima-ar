#!/usr/bin/env python3

from pathlib import Path
import os
import time
import re

from playwright.sync_api import sync_playwright


URL = "https://radares.hidricosargentina.gob.ar/"

OUTPUT_DIR = Path("data/radar")
HISTORICO_DIR = OUTPUT_DIR / "historico"
ACTUAL_FILE = OUTPUT_DIR / "actual.png"

FRAMES_INICIALES = 3

MAX_INTENTOS = 3
ESPERA_FRAME = 30


def encontrar_url_radar(page):
    html = page.content()

    patrones = [
        r'src=["\']([^"\']*?/cache/RMA10/\d+\.png)["\']',
        r'(/cache/RMA10/\d+\.png)',
        r'(https?://[^"\']*?/cache/RMA10/\d+\.png)',
    ]

    for patron in patrones:
        resultados = re.findall(
            patron,
            html,
            flags=re.IGNORECASE
        )

        if resultados:
            url = resultados[-1]

            if url.startswith("/"):
                url = (
                    "https://radares.hidricosargentina.gob.ar"
                    + url
                )

            return url

    return None


def frame_valido(page):
    texto = page.locator("body").inner_text()

    match = re.search(
        r"Fecha frame:\s*(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})",
        texto
    )

    if match:
        return match.group(1)

    return None


def seleccionar_rma10(page):

    selects = page.locator("select")

    for i in range(selects.count()):

        selector = selects.nth(i)

        try:
            texto = selector.inner_text()
        except Exception:
            continue

        if (
            "Bahía Blanca" not in texto
            and "Bahia Blanca" not in texto
            and "RMA10" not in texto
        ):
            continue

        opciones = selector.locator("option")

        for j in range(opciones.count()):

            opcion = opciones.nth(j)

            texto_opcion = opcion.inner_text().strip()
            valor = opcion.get_attribute("value")

            if (
                valor
                and (
                    "Bahía Blanca" in texto_opcion
                    or "Bahia Blanca" in texto_opcion
                    or "RMA10" in texto_opcion
                )
            ):

                print(
                    f"Seleccionando: "
                    f"{texto_opcion} | {valor}"
                )

                selector.select_option(
                    value=valor
                )

                return True

    return False


def descargar_png(page, radar_url):

    print("")
    print("======================================")
    print("DESCARGANDO RMA10")
    print("======================================")
    print(radar_url)

    respuesta = page.request.get(
        radar_url,
        timeout=15000
    )

    print(f"HTTP: {respuesta.status}")

    if respuesta.status != 200:
        raise RuntimeError(
            f"HTTP {respuesta.status}"
        )

    contenido = respuesta.body()

    if len(contenido) < 1000:
        raise RuntimeError(
            f"PNG inválido: {len(contenido)} bytes"
        )

    nombre = radar_url.split("/")[-1]

    historico = HISTORICO_DIR / nombre

    historico.write_bytes(contenido)
    ACTUAL_FILE.write_bytes(contenido)

    print(f"Guardado: {historico}")
    print(f"Tamaño: {len(contenido)} bytes")
    print("CAPTURA EXITOSA")

    return nombre


def intentar_captura(page):

    print("Seleccionando RMA10...")

    if not seleccionar_rma10(page):
        raise RuntimeError(
            "No se encontró selector RMA10."
        )

    print(
        f"Esperando frame válido "
        f"(máximo {ESPERA_FRAME} segundos)..."
    )

    inicio = time.monotonic()

    while time.monotonic() - inicio < ESPERA_FRAME:

        frame = frame_valido(page)

        if frame:
            print(f"Fecha frame detectada: {frame}")

            radar_url = encontrar_url_radar(page)

            if radar_url:
                print("URL PNG detectada:")
                print(radar_url)

                return descargar_png(
                    page,
                    radar_url
                )

            print(
                "Hay fecha de frame, "
                "pero todavía no apareció la URL PNG."
            )

        else:
            print(
                "Esperando conexión/frame..."
            )

        time.sleep(1)

    raise RuntimeError(
        "SINARAME no entregó un frame válido."
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    HISTORICO_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    modo = os.environ.get(
        "CLIMAAR_CAPTURE_MODE",
        ""
    ).strip().lower()

    if modo not in {
        "rapido",
        "inicial"
    }:

        if (
            OUTPUT_DIR /
            "mascara_ecos_estaticos.png"
        ).exists():

            modo = "rapido"

        else:
            modo = "inicial"

    frames_necesarios = (
        1 if modo == "rapido"
        else FRAMES_INICIALES
    )

    print("")
    print("======================================")
    print("ClimaAR - CAPTURA RMA10")
    print("======================================")
    print(f"Modo: {modo.upper()}")
    print(
        f"Frames necesarios: "
        f"{frames_necesarios}"
    )
    print(
        f"Máximo de intentos: "
        f"{MAX_INTENTOS}"
    )
    print("======================================")

    frames_guardados = set()

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        try:

            for intento in range(
                1,
                MAX_INTENTOS + 1
            ):

                print("")
                print("======================================")
                print(
                    f"INTENTO {intento}/"
                    f"{MAX_INTENTOS}"
                )
                print("======================================")

                page = browser.new_page(
                    viewport={
                        "width": 1920,
                        "height": 1080
                    }
                )

                try:

                    print("Abriendo SINARAME...")

                    page.goto(
                        URL,
                        wait_until="domcontentloaded",
                        timeout=30000
                    )

                    page.wait_for_timeout(4000)

                    nombre = intentar_captura(page)

                    if nombre not in frames_guardados:

                        frames_guardados.add(nombre)

                        print(
                            f"Nuevo frame: {nombre}"
                        )

                    else:

                        print(
                            f"Frame repetido: {nombre}"
                        )

                    if (
                        len(frames_guardados)
                        >= frames_necesarios
                    ):
                        break

                except Exception as e:

                    print("")
                    print(
                        f"Intento {intento} falló:"
                    )
                    print(e)

                finally:
                    page.close()

                if (
                    len(frames_guardados)
                    < frames_necesarios
                    and intento < MAX_INTENTOS
                ):

                    print(
                        "Esperando 5 segundos "
                        "antes de reintentar..."
                    )

                    time.sleep(5)

        finally:
            browser.close()

    if not frames_guardados:

        raise RuntimeError(
            "No se pudo capturar ningún "
            "frame RMA10 después de "
            f"{MAX_INTENTOS} intentos."
        )

    print("")
    print("======================================")
    print("CAPTURA TERMINADA")
    print("======================================")

    for frame in sorted(frames_guardados):
        print(frame)


if __name__ == "__main__":
    main()
