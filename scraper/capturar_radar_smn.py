#!/usr/bin/env python3
from pathlib import Path
import io
import re
import time
from datetime import datetime, timezone
from urllib.parse import urljoin

from PIL import Image
from playwright.sync_api import sync_playwright

URL = "https://radares.hidricosargentina.gob.ar/"

OUT = Path("data/radar")
HIST = OUT / "historico"
ACTUAL = OUT / "actual.png"
TSFILE = OUT / "radar_timestamp.txt"

MAX_RETRIES = 5
MAX_RADAR_AGE_MINUTES = 20


def validar_png(data):
    if not data or len(data) < 1000:
        return False, "archivo vacio/pequeno"

    try:
        im = Image.open(io.BytesIO(data)).convert("RGBA")

        if im.width < 100 or im.height < 100:
            return False, f"resolucion {im.width}x{im.height}"

        alpha = im.getchannel("A")

        if alpha.getextrema()[1] < 3:
            return False, "imagen completamente transparente"

        visible = sum(1 for a in alpha.getdata() if a > 2)

        if visible / (im.width * im.height) < 0.01:
            return False, "menos de 1% de pixeles visibles"

        return True, f"{im.width}x{im.height}, {len(data)} bytes"

    except Exception as exc:
        return False, f"PNG invalido: {exc}"


def timestamp_from_url(url):
    if not url:
        return None

    match = re.search(r"(\d{14})\.png(?:\?.*)?$", url)

    if not match:
        return None

    return match.group(1)


def timestamp_to_datetime(timestamp):
    return datetime.strptime(
        timestamp,
        "%Y%m%d%H%M%S",
    ).replace(tzinfo=timezone.utc)


def edad_minutos(timestamp):
    ahora = datetime.now(timezone.utc)
    fecha = timestamp_to_datetime(timestamp)

    return (ahora - fecha).total_seconds() / 60.0


def normalizar_url(url):
    if not url:
        return None

    url = url.strip().strip("\"'")

    if url.startswith("//"):
        return "https:" + url

    if url.startswith("/"):
        return urljoin(URL, url)

    if url.startswith("http://") or url.startswith("https://"):
        return url

    return None


def extraer_urls_rma10(page):
    encontrados = []

    def agregar(texto):
        if not texto:
            return

        patrones = [
            r"https?://[^\"'<>\\s]+/cache/RMA10/[^\"'<>\\s]+?\.png(?:\?[^\"'<>\\s]*)?",
            r"//[^\"'<>\\s]+/cache/RMA10/[^\"'<>\\s]+?\.png(?:\?[^\"'<>\\s]*)?",
            r"/cache/RMA10/[^\"'<>\\s]+?\.png(?:\?[^\"'<>\\s]*)?",
        ]

        for patron in patrones:
            for coincidencia in re.findall(
                patron,
                texto,
                re.I
            ):
                url = normalizar_url(coincidencia)
                timestamp = timestamp_from_url(url)

                if url and timestamp:
                    encontrados.append(
                        (timestamp, url)
                    )

    urls_img = page.locator("img").evaluate_all(
        "els => els.map(e => e.src || e.getAttribute('src') || '')"
    )

    for url in urls_img:
        agregar(url)

    agregar(page.content())

    try:
        scripts = page.locator("script").evaluate_all(
            "els => els.map(e => e.textContent || '')"
        )

        for script in scripts:
            agregar(script)

    except Exception:
        pass

    unicos = {}

    for timestamp, url in encontrados:
        unicos[(timestamp, url)] = (
            timestamp,
            url
        )

    return list(unicos.values())


def seleccionar_mas_reciente(candidatos):
    if not candidatos:
        return None

    return max(
        candidatos,
        key=lambda item: item[0]
    )


def descargar_imagen(page, url):
    response = page.request.get(
        url,
        timeout=30000
    )

    if not response.ok:
        raise RuntimeError(
            f"HTTP {response.status} al descargar {url}"
        )

    data = response.body()

    ok, detail = validar_png(data)

    if not ok:
        raise RuntimeError(detail)

    return data, detail


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    HIST.mkdir(
        parents=True,
        exist_ok=True
    )

    with sync_playwright() as playwright:

        browser = playwright.chromium.launch(
            headless=True
        )

        page = browser.new_page(
            viewport={
                "width": 1920,
                "height": 1080
            }
        )

        try:

            ultimo_error = None

            for intento in range(
                1,
                MAX_RETRIES + 1
            ):

                print(
                    f"Intento {intento}/"
                    f"{MAX_RETRIES}"
                )

                try:

                    respuestas_radar = []

                    def capturar_respuesta(response):
                        url = response.url

                        if (
                            "/cache/RMA10/" in url
                            and ".png" in url.lower()
                        ):

                            timestamp = (
                                timestamp_from_url(
                                    url
                                )
                            )

                            if timestamp:
                                respuestas_radar.append(
                                    (
                                        timestamp,
                                        url,
                                        response
                                    )
                                )

                    page.on(
                        "response",
                        capturar_respuesta
                    )

                    page.goto(
                        URL,
                        wait_until="domcontentloaded",
                        timeout=60000
                    )

                    page.wait_for_timeout(
                        10000
                    )

                    selects = page.locator(
                        "select"
                    )

                    selector = None

                    for index in range(
                        selects.count()
                    ):

                        select = selects.nth(
                            index
                        )

                        try:
                            texto = (
                                select.inner_text()
                            )

                        except Exception:
                            continue

                        if (
                            "Bahía Blanca" in texto
                            or
                            "Bahia Blanca" in texto
                            or
                            "RMA10" in texto
                        ):

                            selector = select
                            break

                    if selector is None:
                        raise RuntimeError(
                            "No se encontro "
                            "selector RMA10 "
                            "en SINARAME"
                        )

                    opciones = selector.locator(
                        "option"
                    )

                    value = None

                    for index in range(
                        opciones.count()
                    ):

                        option = opciones.nth(
                            index
                        )

                        texto = (
                            option.inner_text()
                            .strip()
                        )

                        valor = (
                            option.get_attribute(
                                "value"
                            )
                        )

                        if (
                            "Bahía Blanca" in texto
                            or
                            "Bahia Blanca" in texto
                            or
                            "RMA10" in texto
                        ):

                            value = valor
                            break

                    if value is None:
                        raise RuntimeError(
                            "No se encontro "
                            "Bahía Blanca RMA10 "
                            "en SINARAME"
                        )

                    selector.select_option(
                        value=value
                    )

                    page.wait_for_timeout(
                        12000
                    )

                    candidatos = []

                    for (
                        timestamp,
                        url,
                        response
                    ) in respuestas_radar:

                        candidatos.append(
                            (
                                timestamp,
                                url,
                                response
                            )
                        )

                    for (
                        timestamp,
                        url
                    ) in extraer_urls_rma10(
                        page
                    ):

                        candidatos.append(
                            (
                                timestamp,
                                url,
                                None
                            )
                        )

                    if not candidatos:
                        raise RuntimeError(
                            "SINARAME no expuso "
                            "ninguna URL "
                            "RMA10 PNG"
                        )

                    unicos = {}

                    for item in candidatos:

                        timestamp, url, response = (
                            item
                        )

                        unicos[
                            (timestamp, url)
                        ] = item

                    candidatos = list(
                        unicos.values()
                    )

                    candidatos.sort(
                        key=lambda item: item[0],
                        reverse=True
                    )

                    print(
                        "Candidatos RMA10 "
                        f"encontrados: "
                        f"{len(candidatos)}"
                    )

                    for (
                        timestamp,
                        url,
                        _
                    ) in candidatos[:10]:

                        print(
                            f"RMA10 candidato: "
                            f"{timestamp} | "
                            f"edad "
                            f"{edad_minutos(timestamp):.1f} "
                            f"min | {url}"
                        )

                    seleccionado = (
                        seleccionar_mas_reciente(
                            candidatos
                        )
                    )

                    timestamp, image_url, response = (
                        seleccionado
                    )

                    edad = edad_minutos(
                        timestamp
                    )

                    if edad < -5:
                        raise RuntimeError(
                            "Timestamp RMA10 "
                            "futuro o invalido: "
                            f"{timestamp}"
                        )

                    if (
                        edad
                        > MAX_RADAR_AGE_MINUTES
                    ):

                        raise RuntimeError(
                            "SINARAME NO TIENE "
                            "UN RMA10 ACTUALIZADO: "
                            f"el frame mas reciente "
                            f"encontrado es de "
                            f"{edad:.1f} minutos. "
                            f"URL: {image_url}"
                        )

                    if response is not None:

                        data = response.body()

                        ok, detail = (
                            validar_png(data)
                        )

                        if not ok:
                            data, detail = (
                                descargar_imagen(
                                    page,
                                    image_url
                                )
                            )

                    else:

                        data, detail = (
                            descargar_imagen(
                                page,
                                image_url
                            )
                        )

                    print(
                        "Imagen seleccionada:",
                        image_url
                    )

                    print(
                        "Validacion:",
                        detail
                    )

                    print(
                        f"Timestamp RMA10: "
                        f"{timestamp} | "
                        f"edad: "
                        f"{edad:.1f} min"
                    )

                    name = (
                        f"{timestamp}.png"
                    )

                    ACTUAL.write_bytes(
                        data
                    )

                    (
                        HIST / name
                    ).write_bytes(
                        data
                    )

                    iso = (
                        f"{timestamp[:4]}-"
                        f"{timestamp[4:6]}-"
                        f"{timestamp[6:8]}T"
                        f"{timestamp[8:10]}:"
                        f"{timestamp[10:12]}:"
                        f"{timestamp[12:14]}Z"
                    )

                    TSFILE.write_text(
                        iso,
                        encoding="utf-8"
                    )

                    print(
                        f"OK RMA10 "
                        f"{name} -> "
                        f"{iso} "
                        f"({edad:.1f} min)"
                    )

                    return

                except Exception as exc:

                    ultimo_error = exc

                    print(
                        f"Fallo: {exc}"
                    )

                    if (
                        intento
                        < MAX_RETRIES
                    ):
                        time.sleep(4)

            page.screenshot(
                path=str(
                    OUT /
                    "error_rma10.png"
                ),
                full_page=True
            )

            raise RuntimeError(
                "No se pudo obtener "
                "un RMA10 actualizado "
                "desde SINARAME. "
                f"Ultimo error: "
                f"{ultimo_error}"
            )

        finally:
            browser.close()


if __name__ == "__main__":
    main()
