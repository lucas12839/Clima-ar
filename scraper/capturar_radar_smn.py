#!/usr/bin/env python3

from pathlib import Path
from urllib.parse import urljoin
import requests
import re

BASE_URL = "https://radares.hidricosargentina.gob.ar/"

OUTPUT_DIR = Path("data/radar")
HTML_FILE = OUTPUT_DIR / "sinarame.html"
REPORT_FILE = OUTPUT_DIR / "sinarame_debug.txt"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (ClimaAR Radar Diagnostic)"
}


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("======================================")
    print("CLIMAAR - DIAGNOSTICO SINARAME")
    print("======================================")
    print()

    print("Descargando página principal...")

    r = requests.get(
        BASE_URL,
        headers=HEADERS,
        timeout=30
    )

    print("HTTP:", r.status_code)
    print("Tamaño:", len(r.content), "bytes")

    r.raise_for_status()

    html = r.text

    HTML_FILE.write_text(
        html,
        encoding="utf-8"
    )

    print("HTML guardado:", HTML_FILE)
    print()

    # Buscar scripts
    scripts = re.findall(
        r'<script[^>]+src=["\']([^"\']+)["\']',
        html,
        re.IGNORECASE
    )

    print("Scripts encontrados:", len(scripts))
    print()

    report = []

    report.append("CLIMAAR - DIAGNOSTICO SINARAME")
    report.append("=" * 50)
    report.append("")
    report.append(f"HTTP principal: {r.status_code}")
    report.append(f"Tamaño HTML: {len(r.content)} bytes")
    report.append("")
    report.append("SCRIPTS:")
    report.append("")

    for script in scripts:

        url = urljoin(BASE_URL, script)

        print("Descargando JS:")
        print(url)

        try:

            js = requests.get(
                url,
                headers=HEADERS,
                timeout=30
            )

            print(
                "  HTTP:",
                js.status_code,
                "|",
                len(js.content),
                "bytes"
            )

            report.append("")
            report.append("=" * 70)
            report.append(f"JS: {url}")
            report.append(f"HTTP: {js.status_code}")
            report.append(f"Tamaño: {len(js.content)}")
            report.append("=" * 70)

            if js.status_code != 200:
                continue

            texto = js.text

            # Palabras importantes
            palabras = [
                "RMA10",
                "RMA",
                "/cache/",
                ".png",
                ".jpg",
                "radar",
                "frame",
                "frames",
                "imagen",
                "images",
                "latest",
                "api",
                "http://",
                "https://"
            ]

            encontrados = set()

            for palabra in palabras:

                if palabra.lower() in texto.lower():
                    encontrados.add(palabra)

            if encontrados:

                report.append("")
                report.append(
                    "PALABRAS ENCONTRADAS: "
                    + ", ".join(sorted(encontrados))
                )

                # Mostrar URLs que aparecen en JS
                urls = re.findall(
                    r'https?://[^\s"\'<>]+',
                    texto
                )

                if urls:
                    report.append("")
                    report.append("URLS ENCONTRADAS:")

                    for u in sorted(set(urls)):
                        report.append(u[:500])

                # Buscar líneas relacionadas
                report.append("")
                report.append("FRAGMENTOS RELEVANTES:")

                lineas = texto.splitlines()

                contador = 0

                for i, linea in enumerate(lineas):

                    linea_lower = linea.lower()

                    if any(
                        palabra.lower() in linea_lower
                        for palabra in palabras
                    ):

                        fragmento = linea.strip()

                        if fragmento:
                            report.append(
                                f"[linea {i + 1}] "
                                + fragmento[:1000]
                            )

                            contador += 1

                            if contador >= 80:
                                break

        except Exception as e:

            print("  ERROR:", e)

            report.append(
                f"ERROR descargando {url}: {e}"
            )

    # Buscar información directamente en HTML
    report.append("")
    report.append("=" * 70)
    report.append("BUSQUEDA DIRECTA EN HTML")
    report.append("=" * 70)

    for palabra in [
        "RMA10",
        "/cache/",
        ".png",
        ".jpg",
        "radar",
        "frame",
        "api"
    ]:

        if palabra.lower() in html.lower():

            report.append(
                f"ENCONTRADO EN HTML: {palabra}"
            )

    REPORT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print()
    print("======================================")
    print("DIAGNOSTICO TERMINADO")
    print("======================================")
    print()
    print("Archivos generados:")
    print(HTML_FILE)
    print(REPORT_FILE)
    print()
    print("Ahora subí estos archivos como artifacts.")
    print()


if __name__ == "__main__":
    main()
