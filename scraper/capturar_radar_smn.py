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

MAX_INTENTOS = 5
ESPERA_IMAGEN = 30

FRAMES_INICIALES = 3


def buscar_url_rma10(page):

    # Primero buscamos directamente en las imágenes del DOM
    try:

        imagenes = page.locator(
            'img[src*="/cache/RMA10/"]'
        )

        cantidad = imagenes.count()

        for i in range(cantidad):

            src = imagenes.nth(i).get_attribute("src")

            if src and re.search(
                r"/cache/RMA10/\d+\.png",
                src,
                re.IGNORECASE
            ):

                if src.startswith("/"):

                    src = (
                        "https://radares.hidricosargentina.gob.ar"
                        + src
                    )

                return src

    except Exception:
        pass

    # Si no apareció como locator, buscamos en el HTML
    try:

        html = page.content()

        patrones = [
            r'["\']([^"\']*?/cache/RMA10/\d+\.png)["\']',
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

    except Exception:
        pass

    return None


def seleccionar_rma10(page):

    selects = page.locator("select")

    print(
        f"Selectores encontrados: "
        f"{selects.count()}"
    )

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

            texto_opcion = (
                opcion.inner_text().strip()
            )

            valor = opcion.get_attribute(
                "value"
            )

            if (
                valor
                and (
                    "Bahía Blanca"
                    in texto_opcion
                    or
                    "Bahia Blanca"
                    in texto_opcion
                    or
                    "RMA10"
                    in texto_opcion
                )
            ):

                print(
                    f"Seleccionando: "
                    f"{texto_opcion} | "
                    f"value={valor}"
                )

                selector.select_option(
                    value=valor
                )

                return True

    return False


def descargar_radar(page, url):

    print("")
    print("======================================")
    print("URL RMA10 ENCONTRADA")
    print("======================================")
    print(url)

    respuesta = page.request.get(
        url,
        timeout=20000
    )

    print(
        f"HTTP: {respuesta.status}"
    )

    if respuesta.status != 200:

        raise RuntimeError(
            f"HTTP {respuesta.status}"
        )

    contenido = respuesta.body()

    if len(contenido) < 1000:

        raise RuntimeError(
            f"Archivo inválido: "
            f"{len(contenido)} bytes"
        )

    nombre = url.split("/")[-1]

    historico = (
        HISTORICO_DIR / nombre
    )

    historico.write_bytes(
        contenido
    )

    ACTUAL_FILE.write_bytes(
        contenido
    )

    print(
        f"Guardado: {historico}"
    )

    print(
        f"Tamaño: {len(contenido)} bytes"
    )

    print("")
    print("CAPTURA EXITOSA")

    return nombre


def intentar_captura(page):

    print("Abriendo SINARAME...")

    page.goto(
        URL,
        wait_until="domcontentloaded",
        timeout=30000
    )

    # Tiempo inicial para que cargue Blazor
    page.wait_for_timeout(5000)

    print("Seleccionando RMA10...")

    if not seleccionar_rma10(page):

        raise RuntimeError(
            "No se encontró el selector RMA10."
        )

    print("")
    print(
        f"Buscando imagen RMA10 "
        f"durante {ESPERA_IMAGEN} segundos..."
    )

    inicio = time.monotonic()

    while (
        time.monotonic() - inicio
        < ESPERA_IMAGEN
    ):

        radar_url = buscar_url_rma10(page)

        if radar_url:

            print(
                "Imagen RMA10 detectada."
            )

            return descargar_radar(
                page,
                radar_url
            )

        print(
            "Esperando imagen RMA10..."
        )

        time.sleep(1)

    # Información de diagnóstico
    print("")
    print("======================================")
    print("DIAGNÓSTICO")
    print("======================================")

    try:

        texto = page.locator(
            "body"
        ).inner_text()

        for linea in texto.splitlines():

            if (
                "Bahía Blanca" in linea
                or "RMA10" in linea
                or "Fecha frame" in linea
                or "Última actualización"
                in linea
            ):

                print(linea)

    except Exception:
        pass

    raise RuntimeError(
        "SINARAME no entregó una "
        "URL PNG RMA10."
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

       
