#!/usr/bin/env python3

from pathlib import Path
from playwright.sync_api import sync_playwright
import time

URL = "https://radares.hidricosargentina.gob.ar/"

OUTPUT_DIR = Path("data/radar")
REPORT_FILE = OUTPUT_DIR / "sinarame_network.txt"
SCREENSHOT_FILE = OUTPUT_DIR / "sinarame_rma10.png"
HTML_FILE = OUTPUT_DIR / "sinarame_rma10.html"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eventos = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        page = browser.new_page(
            viewport={
                "width": 1920,
                "height": 1080
            }
        )

        # ==============================
        # REQUESTS
        # ==============================
        def registrar_request(request):
            linea = (
                f"REQUEST | "
                f"{request.method} | "
                f"{request.resource_type} | "
                f"{request.url}"
            )

            eventos.append(linea)
            print(linea)

        page.on("request", registrar_request)

        # ==============================
        # RESPONSES
        # ==============================
        def registrar_response(response):
            linea = (
                f"RESPONSE | "
                f"{response.status} | "
                f"{response.request.resource_type} | "
                f"{response.url}"
            )

            eventos.append(linea)
            print(linea)

        page.on("response", registrar_response)

        # ==============================
        # ERRORES JAVASCRIPT
        # ==============================
        def registrar_error(error):
            linea = f"PAGEERROR | {error}"

            eventos.append(linea)
            print(linea)

        page.on("pageerror", registrar_error)

        # ==============================
        # CONSOLA
        # ==============================
        def registrar_console(msg):
            texto = msg.text

            linea = (
                f"CONSOLE | "
                f"{msg.type} | "
                f"{texto}"
            )

            eventos.append(linea)
            print(linea)

        page.on("console", registrar_console)

        # ==============================
        # WEBSOCKETS
        # ==============================
        def registrar_websocket(ws):
            print(f"WEBSOCKET OPEN | {ws.url}")

            eventos.append(
                f"WEBSOCKET OPEN | {ws.url}"
            )

            def recibido(mensaje):
                texto = str(mensaje)

                if len(texto) > 10000:
                    texto = (
                        texto[:10000]
                        + " [TRUNCADO]"
                    )

                linea = (
                    f"WEBSOCKET RECEIVED | "
                    f"{ws.url} | "
                    f"{texto}"
                )

                eventos.append(linea)

                print(
                    "WEBSOCKET RECEIVED:",
                    texto[:2000]
                )

            def enviado(mensaje):
                texto = str(mensaje)

                if len(texto)
