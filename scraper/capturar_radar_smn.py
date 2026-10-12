#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO
import re

from PIL import Image
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

URL = "https://radares.hidricosargentina.gob.ar/"
OUTPUT_DIR = Path("data/radar")
HISTORICO_DIR = OUTPUT_DIR / "historico"
ACTUAL_FILE = OUTPUT_DIR / "actual.png"
RADAR_TIMESTAMP_FILE = OUTPUT_DIR / "radar_timestamp.txt"

MAX_INTENTOS = 5
MIN_BYTES = 1000
MIN_VISIBLE_RATIO = 0.01


def timestamp_desde_nombre(nombre):
    m = re.search(r"(\d{14})", nombre)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    m = re.search(r"(?:^|_)(\d{10})(?:\D|$)", nombre)
    if m:
        try:
            return datetime.fromtimestamp(int(m.group(1)), timezone.utc)
        except (ValueError, OSError, OverflowError):
            pass
    return None


def validar_png(contenido):
    if len(contenido) < MIN_BYTES:
        return False, "archivo demasiado pequeño"
    try:
        with Image.open(BytesIO(contenido)) as img:
            img.load()
            rgba = img.convert("RGBA")
            pixels = list(rgba.getdata())
            if not pixels:
                return False, "imagen sin píxeles"
            visibles = sum(1 for r, g, b, a in pixels if a >= 10 and max(r, g, b) >= 8)
            ratio = visibles / len(pixels)
            if ratio < MIN_VISIBLE_RATIO:
                return False, f"imagen casi transparente/vacía ({ratio:.4%} visible)"
            return True, f"{img.width}x{img.height}, {ratio:.2%} visible"
    except Exception as exc:
        return False, f"PNG inválido: {exc}"


def encontrar_selector(page):
    selects = page.locator("select")
    for i in range(selects.count()):
        actual = selects.nth(i)
        try:
            texto = actual.inner_text()
        except Exception:
            continue
        if "Bahía Blanca" in texto or "Bahia Blanca" in texto or "RMA10" in texto:
            return actual
    return None


def encontrar_valor_rma10(selector):
    opciones = selector.locator("option")
    for i in range(opciones.count()):
        opcion = opciones.nth(i)
        texto = opcion.inner_text().strip()
        valor = opcion.get_attribute("value")
        if "Bahía Blanca" in texto or "Bahia Blanca" in texto or "RMA10" in texto:
            return valor
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORICO_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})

        ultimo_error = None

        for intento in range(1, MAX_INTENTOS + 1):
            print(f"======================================")
            print(f"INTENTO RMA10 {intento}/{MAX_INTENTOS}")
            print(f"======================================")

            try:
                page.goto(URL, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(10000)

                selector = encontrar_selector(page)
                if selector is None:
                    raise RuntimeError("No se encontró el selector de RMA10.")

                valor_rma10 = encontrar_valor_rma10(selector)
                if valor_rma10 is None:
                    raise RuntimeError("No se encontró Bahía Blanca RMA10.")

                with page.expect_response(
                    lambda response: "/cache/RMA10/" in response.url and ".png" in response.url,
                    timeout=90000,
                ) as respuesta_esperada:
                    selector.select_option(value=valor_rma10)

                respuesta = respuesta_esperada.value
                contenido = respuesta.body()
                ok, detalle = validar_png(contenido)
                print(f"URL: {respuesta.url}")
                print(f"HTTP: {respuesta.status}")
                print(f"Validación: {detalle}")

                if not ok:
                    raise RuntimeError(detalle)
                if respuesta.status != 200:
                    raise RuntimeError(f"HTTP {respuesta.status}")

                nombre = respuesta.url.split("/")[-1]
                dt = timestamp_desde_nombre(nombre)
                if dt is None:
                    raise RuntimeError(f"No se pudo obtener timestamp desde {nombre}")

                ACTUAL_FILE.write_bytes(contenido)
                (HISTORICO_DIR / nombre).write_bytes(contenido)
                RADAR_TIMESTAMP_FILE.write_text(
                    dt.isoformat().replace("+00:00", "Z"),
                    encoding="utf-8",
                )

                print("======================================")
                print("RADAR RMA10 VALIDADO Y GUARDADO")
                print("======================================")
                print(f"Frame: {dt.isoformat().replace('+00:00', 'Z')}")
                print(f"Actual: {ACTUAL_FILE}")
                print(f"Histórico: {HISTORICO_DIR / nombre}")
                print(f"Tamaño: {len(contenido)} bytes")
                browser.close()
                return

            except (PlaywrightTimeoutError, Exception) as exc:
                ultimo_error = exc
                print(f"Fallo intento {intento}: {exc}")
                if intento < MAX_INTENTOS:
                    page.wait_for_timeout(3000)

        page.screenshot(path=str(OUTPUT_DIR / "error_rma10.png"), full_page=True)
        browser.close()
        raise RuntimeError(f"SINARAME no entregó un PNG RMA10 válido después de {MAX_INTENTOS} intentos: {ultimo_error}")


if __name__ == "__main__":
    main()
