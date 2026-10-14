#!/usr/bin/env python3
from pathlib import Path
import io
import re
import time
from datetime import datetime, timezone

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
        im = Image.open(
            io.BytesIO(data)
        ).convert("RGBA")

        if im.width < 100 or im.height < 100:
            return False, (
                f"resolucion {im.width}x{im.height}"
            )

        alpha = im.getchannel("A")

        if alpha.getextrema()[1] < 3:
            return False, (
                "imagen completamente transparente"
            )

        visible = sum(
            1
            for a in alpha.getdata()
            if a > 2
        )

        if visible / (
            im.width * im.height
        ) < 0.01:
            return False, (
                "menos de 1% de pixeles visibles"
            )

        return True, (
            f"{im.width}x{im.height}, "
            f"{len(data)} bytes"
        )

    except Exception as exc:
        return False, f"PNG invalido: {exc}"


def timestamp_from_url(url):
    match = re.search(
        r"(\d{14})\.png(?:\?.*)?$",
        url
    )

    if not match:
        return None

    return match.group(1)


def timestamp_to_datetime(timestamp):
    return datetime.strptime(
        timestamp,
        "%Y%m%d%H%M%S"
    ).replace(
        tzinfo=timezone.utc
    )


def edad_minutos(timestamp):
    ahora = datetime.now(
        timezone.utc
    )

    fecha = timestamp_to_datetime(
        timestamp
    )

    return (
        ahora - fecha
    ).total_seconds() / 60.0


def extraer_urls_rma10(page):
    urls = page.locator(
        "img"
    ).evaluate_all(
        "els => els.map("
        "e => e.src || "
        "e.getAttribute('src') || ''"
        ")"
    )

    resultado = []

    for url in urls:

        if "/cache/RMA10/" not in url:
            continue

        timestamp = timestamp_from_url(
            url
        )

        if timestamp:
            resultado.append(
                (
                    timestamp,
                    url
                )
            )

    return resultado


def seleccionar_url_mas_reciente(urls):
    if not urls:
        return None

    return max(
        urls,
        key=lambda item: item[0]
    )


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
                            "Bahía Blanca"
                            in texto
                            or
                            "Bahia Blanca"
                            in texto
                            or
                            "RMA10"
                            in texto
                        ):
                            selector = select
                            break

                    if selector is None:
                        raise RuntimeError(
                            "No se encontro "
                            "selector RMA10"
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
                            "Bahía Blanca"
                            in texto
                            or
                            "Bahia Blanca"
                            in texto
                            or
                            "RMA10"
                            in texto
                        ):
                            value = valor
                            break

                    if value is None:
                        raise RuntimeError(
                            "No se encontro "
                            "Bahía Blanca RMA10"
                        )

                    with page.expect_response(
                        lambda r: (
                            "/cache/RMA10/"
                            in r.url
                            and
                            ".png"
                            in r.url.lower()
                        ),
                        timeout=90000
                    ) as expected:

                        selector.select_option(
                            value=value
                        )

                    response = expected.value

                    data = response.body()

                    response_timestamp = (
                        timestamp_from_url(
                            response.url
                        )
                    )

                    ok, detail = validar_png(
                        data
                    )

                    print(
                        response.url
                    )

                    print(
                        f"HTTP {response.status} "
                        f"| {detail}"
                    )

                    if not ok:
                        raise RuntimeError(
                            detail
                        )

                    if response_timestamp is None:
                        raise RuntimeError(
                            "URL de respuesta "
                            "sin timestamp RMA10"
                        )

                    urls = extraer_urls_rma10(
                        page
                    )

                    seleccionado = (
                        seleccionar_url_mas_reciente(
                            urls
                        )
                    )

                    if seleccionado:

                        timestamp, image_url = (
                            seleccionado
                        )

                        print(
                            "URL RMA10 mas reciente "
                            "encontrada:",
                            image_url
                        )

                        try:

                            image_response = (
                                page.request.get(
                                    image_url,
                                    timeout=30000
                                )
                            )

                            if image_response.ok:

                                image_data = (
                                    image_response.body()
                                )

                                image_ok, image_detail = (
                                    validar_png(
                                        image_data
                                    )
                                )

                                if image_ok:

                                    data = image_data

                                    response_timestamp = (
                                        timestamp
                                    )

                                    print(
                                        "Imagen mas reciente "
                                        "descargada:",
                                        image_detail
                                    )

                        except Exception as exc:

                            print(
                                "No se pudo descargar "
                                "directamente la imagen "
                                f"mas reciente: {exc}"
                            )

                    edad = edad_minutos(
                        response_timestamp
                    )

                    print(
                        f"Timestamp RMA10: "
                        f"{response_timestamp} "
                        f"| edad: {edad:.1f} min"
                    )

                    if edad < -5:

                        raise RuntimeError(
                            "Timestamp RMA10 "
                            "futuro o invalido"
                        )

                    if (
                        edad
                        > MAX_RADAR_AGE_MINUTES
                    ):

                        raise RuntimeError(
                            "RADAR DESACTUALIZADO: "
                            f"{edad:.1f} minutos "
                            "de antiguedad "
                            f"(maximo "
                            f"{MAX_RADAR_AGE_MINUTES})"
                        )

                    name = (
                        f"{response_timestamp}.png"
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
                        f"{response_timestamp[:4]}-"
                        f"{response_timestamp[4:6]}-"
                        f"{response_timestamp[6:8]}T"
                        f"{response_timestamp[8:10]}:"
                        f"{response_timestamp[10:12]}:"
                        f"{response_timestamp[12:14]}Z"
                    )

                    TSFILE.write_text(
                        iso,
                        encoding="utf-8"
                    )

                    print(
                        f"OK RMA10 {name} "
                        f"-> {iso} "
                        f"({edad:.1f} min)"
                    )

                    return

                except Exception as exc:

                    ultimo_error = exc

                    print(
                        f"Fallo: {exc}"
                    )

                    if intento < MAX_RETRIES:
                        time.sleep(4)

            page.screenshot(
                path=str(
                    OUT / "error_rma10.png"
                ),
                full_page=True
            )

            raise RuntimeError(
                "No se pudo obtener "
                "un RMA10 actualizado. "
                f"Ultimo error: {ultimo_error}"
            )

        finally:
            browser.close()


if __name__ == "__main__":
    main()
