#!/usr/bin/env python3
from pathlib import Path
import io
import json
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

from PIL import Image
from playwright.sync_api import sync_playwright


URL = "https://radares.hidricosargentina.gob.ar/"

OUT = Path("data/radar")
HIST = OUT / "historico"
ACTUAL = OUT / "actual.png"
TSFILE = OUT / "radar_timestamp.txt"

PERSISTENT_STATE = Path(".climaar/estado_rma10.json")
STATE = OUT / "seguimiento" / "estado.json"
SEGUIMIENTO = OUT / "seguimiento" / "seguimiento_temporal.json"
PREDICCION = OUT / "seguimiento" / "prediccion_evolucion.json"
PENDIENTES = OUT / "seguimiento" / "predicciones_pendientes.json"

MIGRATION_MARKER = OUT / ".timestamp_rma10_utc_v2"

MAX_RETRIES = 5
MAX_RADAR_AGE_MINUTES = 30

ARGENTINA_TZ = ZoneInfo(
    "America/Argentina/Buenos_Aires"
)


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

        if (
            visible / (im.width * im.height)
            < 0.01
        ):
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
    if not url:
        return None

    match = re.search(
        r"(\d{14})\.png(?:\?.*)?$",
        url
    )

    if not match:
        return None

    return match.group(1)


def timestamp_source_to_datetime(timestamp):
    return datetime.strptime(
        timestamp,
        "%Y%m%d%H%M%S"
    ).replace(
        tzinfo=ARGENTINA_TZ
    )


def source_timestamp_to_utc_timestamp(timestamp):
    return timestamp_source_to_datetime(
        timestamp
    ).astimezone(
        timezone.utc
    ).strftime(
        "%Y%m%d%H%M%S"
    )


def edad_minutos(timestamp):
    ahora = datetime.now(
        ARGENTINA_TZ
    )

    fecha = timestamp_source_to_datetime(
        timestamp
    )

    return (
        ahora - fecha
    ).total_seconds() / 60.0


def normalizar_url(url):
    if not url:
        return None

    url = url.strip().strip("\"'")

    if url.startswith("//"):
        return "https:" + url

    if url.startswith("/"):
        return urljoin(
            URL,
            url
        )

    if (
        url.startswith("http://")
        or
        url.startswith("https://")
    ):
        return url

    return None


def extraer_urls_rma10(page):
    encontrados = []

    def agregar(texto):
        if not texto:
            return

        patrones = [
            (
                r"https?://[^\"'<>"
                r"\s]+/cache/RMA10/"
                r"[^\"'<>"
                r"\s]+?\.png"
                r"(?:\?[^\"'<>"
                r"\s]*)?"
            ),
            (
                r"//[^\"'<>"
                r"\s]+/cache/RMA10/"
                r"[^\"'<>"
                r"\s]+?\.png"
                r"(?:\?[^\"'<>"
                r"\s]*)?"
            ),
            (
                r"/cache/RMA10/"
                r"[^\"'<>"
                r"\s]+?\.png"
                r"(?:\?[^\"'<>"
                r"\s]*)?"
            ),
        ]

        for patron in patrones:
            for coincidencia in re.findall(
                patron,
                texto,
                re.I
            ):
                url = normalizar_url(
                    coincidencia
                )

                timestamp = timestamp_from_url(
                    url
                )

                if url and timestamp:
                    encontrados.append(
                        (
                            timestamp,
                            url
                        )
                    )

    urls_img = page.locator(
        "img"
    ).evaluate_all(
        "els => els.map("
        "e => e.src || "
        "e.getAttribute('src') || ''"
        ")"
    )

    for url in urls_img:
        agregar(url)

    agregar(
        page.content()
    )

    try:
        scripts = page.locator(
            "script"
        ).evaluate_all(
            "els => els.map("
            "e => e.textContent || ''"
            ")"
        )

        for script in scripts:
            agregar(script)

    except Exception:
        pass

    unicos = {}

    for timestamp, url in encontrados:
        unicos[
            (timestamp, url)
        ] = (
            timestamp,
            url
        )

    return list(
        unicos.values()
    )


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
            f"HTTP {response.status} "
            f"al descargar {url}"
        )

    data = response.body()

    ok, detail = validar_png(
        data
    )

    if not ok:
        raise RuntimeError(
            detail
        )

    return data, detail


def migrar_iso_legacy(value):
    if not isinstance(value, str):
        return value

    texto = value.strip()

    if not texto:
        return value

    try:
        if texto.endswith("Z"):
            texto_parse = texto[:-1] + "+00:00"
        else:
            texto_parse = texto

        dt = datetime.fromisoformat(
            texto_parse
        )

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone.utc
            )
        else:
            dt = dt.astimezone(
                timezone.utc
            )

        dt += timedelta(
            hours=3
        )

        return (
            dt.astimezone(timezone.utc)
            .isoformat()
            .replace(
                "+00:00",
                "Z"
            )
        )

    except Exception:
        return value


def migrar_objeto_timestamps(obj):
    claves_radar = {
        "ultimo_radar_timestamp",
        "primer_frame",
        "ultimo_frame",
        "timestamp",
        "timestamp_origen",
        "timestamp_objetivo",
        "radar_timestamp",
    }

    if isinstance(obj, dict):
        for key, value in list(obj.items()):

            if key in claves_radar:
                if isinstance(value, str):
                    obj[key] = migrar_iso_legacy(
                        value
                    )
                elif isinstance(value, list):
                    obj[key] = [
                        migrar_iso_legacy(item)
                        if isinstance(item, str)
                        else item
                        for item in value
                    ]

            else:
                migrar_objeto_timestamps(
                    value
                )

    elif isinstance(obj, list):
        for item in obj:
            migrar_objeto_timestamps(
                item
            )


def migrar_json_legacy(path):
    if not path.exists():
        return False

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        migrar_objeto_timestamps(
            data
        )

        path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        return True

    except Exception as exc:
        print(
            f"AVISO: no se pudo migrar "
            f"{path}: {exc}"
        )
        return False


def migrar_historico_legacy():
    if MIGRATION_MARKER.exists():
        return

    print(
        "Migrando timestamps historicos "
        "RMA10 de hora Argentina a UTC..."
    )

    HIST.mkdir(
        parents=True,
        exist_ok=True
    )

    renombrados = 0

    for archivo in list(
        HIST.glob("*.png")
    ):

        if not re.fullmatch(
            r"\d{14}\.png",
            archivo.name
        ):
            continue

        source_timestamp = (
            archivo.stem
        )

        utc_timestamp = (
            source_timestamp_to_utc_timestamp(
                source_timestamp
            )
        )

        destino = (
            HIST
            / f"{utc_timestamp}.png"
        )

        if destino == archivo:
            continue

        if destino.exists():
            archivo.unlink()
        else:
            archivo.rename(
                destino
            )

        renombrados += 1

    archivos_json = [
        PERSISTENT_STATE,
        STATE,
        SEGUIMIENTO,
        PREDICCION,
        PENDIENTES,
    ]

    migrados = 0

    for path in archivos_json:
        if migrar_json_legacy(path):
            migrados += 1

    MIGRATION_MARKER.write_text(
        "RMA10 timestamps internos en UTC\n",
        encoding="utf-8"
    )

    print(
        "Migracion completada. "
        f"Frames renombrados: {renombrados}. "
        f"JSON migrados: {migrados}."
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

    migrar_historico_legacy()

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

                    def capturar_respuesta(
                        response
                    ):
                        url = response.url

                        if (
                            "/cache/RMA10/"
                            in url
                            and
                            ".png"
                            in url.lower()
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
                        wait_until=(
                            "domcontentloaded"
                        ),
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
                            "selector RMA10 "
                            "en SINARAME"
                        )

                    opciones = (
                        selector.locator(
                            "option"
                        )
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

                    print(
                        f"Hora fuente RMA10: "
                        f"{timestamp} "
                        f"(Argentina)"
                    )

                    print(
                        f"Edad real del frame: "
                        f"{edad:.1f} minutos"
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
                            validar_png(
                                data
                            )
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

                    utc_timestamp = (
                        source_timestamp_to_utc_timestamp(
                            timestamp
                        )
                    )

                    print(
                        "Timestamp interno UTC:",
                        utc_timestamp
                    )

                    print(
                        "Imagen seleccionada:",
                        image_url
                    )

                    print(
                        "Validacion:",
                        detail
                    )

                    name = (
                        f"{utc_timestamp}.png"
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
                        f"{utc_timestamp[:4]}-"
                        f"{utc_timestamp[4:6]}-"
                        f"{utc_timestamp[6:8]}T"
                        f"{utc_timestamp[8:10]}:"
                        f"{utc_timestamp[10:12]}:"
                        f"{utc_timestamp[12:14]}Z"
                    )

                    TSFILE.write_text(
                        iso,
                        encoding="utf-8"
                    )

                    print(
                        f"OK RMA10 "
                        f"{name} -> "
                        f"{iso} "
                        f"(fuente "
                        f"{timestamp} "
                        f"Argentina, "
                        f"{edad:.1f} min)"
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
