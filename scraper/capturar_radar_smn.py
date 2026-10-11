#!/usr/bin/env python3

from pathlib import Path
import os
import time
import re
import io

from PIL import Image
from playwright.sync_api import sync_playwright


URL = "https://radares.hidricosargentina.gob.ar/"

OUTPUT_DIR = Path("data/radar")
HISTORICO_DIR = OUTPUT_DIR / "historico"
ACTUAL_FILE = OUTPUT_DIR / "actual.png"

MAX_INTENTOS = 5
ESPERA_IMAGEN = 30


def buscar_url_rma10(page):
    try:
        imagenes = page.locator('img[src*="/cache/RMA10/"]')

        for i in range(imagenes.count()):
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

    print(f"Selectores encontrados: {selects.count()}")

    for i in range(selects.count()):
        selector = selects.nth(i)

        try:
            texto = selector.inner_text()
        except Exception:
            continue

        if not any(x in texto for x in [
            "Bahía Blanca",
            "Bahia Blanca",
            "RMA10"
        ]):
            continue

        opciones = selector.locator("option")

        for j in range(opciones.count()):
            opcion = opciones.nth(j)

            texto_opcion = opcion.inner_text().strip()
            valor = opcion.get_attribute("value")

            if valor and any(x in texto_opcion for x in [
                "Bahía Blanca",
                "Bahia Blanca",
                "RMA10"
            ]):
                print(
                    f"Seleccionando: "
                    f"{texto_opcion} | value={valor}"
                )

                selector.select_option(value=valor)

                return True

    return False


def validar_png(contenido):
    """
    Verifica que realmente sea un radar visible
    y no un PNG transparente/vacío.
    """

    if len(contenido) < 1000:
        return False, "Archivo demasiado pequeño"

    try:
        imagen = Image.open(io.BytesIO(contenido))
        imagen.load()

        print(
            f"PNG: {imagen.size} | "
            f"modo={imagen.mode}"
        )

        rgba = imagen.convert("RGBA")

        pixeles = list(rgba.getdata())

        visibles = 0
        no_transparentes = 0

        for r, g, b, a in pixeles:

            if a > 0:
                no_transparentes += 1

            if a > 0 and (r > 5 or g > 5 or b > 5):
                visibles += 1

        total = len(pixeles)

        porcentaje_visible = (
            visibles / total * 100
            if total else 0
        )

        porcentaje_alpha = (
            no_transparentes / total * 100
            if total else 0
        )

        print(
            f"Pixeles visibles: "
            f"{porcentaje_visible:.2f}%"
        )

        print(
            f"Pixeles no transparentes: "
            f"{porcentaje_alpha:.2f}%"
        )

        # Un radar completamente transparente
        # debe ser rechazado.
        if porcentaje_alpha < 1:
            return False, "PNG completamente transparente"

        if porcentaje_visible < 1:
            return False, "PNG prácticamente vacío"

        return True, "PNG válido"

    except Exception as e:
        return False, f"Error leyendo PNG: {e}"


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

    print(f"HTTP: {respuesta.status}")

    if respuesta.status != 200:
        raise RuntimeError(
            f"HTTP {respuesta.status}"
        )

    contenido = respuesta.body()

    valido, motivo = validar_png(contenido)

    print(f"Validación: {motivo}")

    if not valido:
        raise RuntimeError(
            f"RADAR INVÁLIDO: {motivo}"
        )

    nombre = url.split("/")[-1]

    historico = HISTORICO_DIR / nombre

    historico.write_bytes(contenido)
    ACTUAL_FILE.write_bytes(contenido)

    print(f"Guardado: {historico}")
    print(f"Tamaño: {len(contenido)} bytes")

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
            print("Imagen RMA10 detectada.")

            return descargar_radar(
                page,
                radar_url
            )

        print("Esperando imagen RMA10...")

        time.sleep(1)

    print("")
    print("======================================")
    print("DIAGNÓSTICO")
    print("======================================")

    try:
        texto = page.locator("body").inner_text()

        for linea in texto.splitlines():

            if any(x in linea for x in [
                "Bahía Blanca",
                "RMA10",
                "Fecha frame",
                "Última actualización"
            ]):
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

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        try:

            ultimo_error = None

            for intento in range(1, MAX_INTENTOS + 1):

                print("")
                print("======================================")
                print(
                    f"INTENTO {intento}/{MAX_INTENTOS}"
                )
                print("======================================")

                page = browser.new_page()

                try:

                    resultado = intentar_captura(page)

                    print("")
                    print(
                        "RADAR RMA10 CAPTURADO "
                        "CORRECTAMENTE."
                    )

                    return resultado

                except Exception as e:

                    ultimo_error = e

                    print("")
                    print(
                        f"Intento {intento} falló:"
                    )
                    print(e)

                finally:
                    page.close()

                if intento < MAX_INTENTOS:
                    print(
                        "Esperando antes del "
                        "siguiente intento..."
                    )
                    time.sleep(5)

            raise RuntimeError(
                "No se pudo obtener un radar RMA10 "
                "válido después de "
                f"{MAX_INTENTOS} intentos. "
                f"Último error: {ultimo_error}"
            )

        finally:
            browser.close()


if __name__ == "__main__":
    main()
