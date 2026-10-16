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
CAPTURE_STATUS = OUT / "captura_estado.txt"

PERSISTENT_STATE = Path(".climaar/estado_rma10.json")
STATE = OUT / "seguimiento" / "estado.json"
SEGUIMIENTO = OUT / "seguimiento" / "seguimiento_temporal.json"
PREDICCION = OUT / "seguimiento" / "prediccion_evolucion.json"
PENDIENTES = OUT / "seguimiento" / "predicciones_pendientes.json"

MIGRATION_MARKER = OUT / ".timestamp_rma10_utc_v2"

MAX_RETRIES = 5

# SINARAME puede entregar frames con bastante retraso.
# Antes estaba en 30 minutos y eso provocaba falsos fallos.
MAX_RADAR_AGE_MINUTES = 45

MAX_FUTURE_SKEW_MINUTES = 2

ARGENTINA_TZ = ZoneInfo("America/Argentina/Buenos_Aires")


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

        visible = sum(
            1
            for a in alpha.getdata()
            if a > 2
        )

        if visible / (im.width * im.height) < 0.01:
            return False, "menos de 1% de pixeles visibles"

        return True, f"{im.width}x{im.height}, {len(data)} bytes"

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
                url = normalizar_url(coincidencia)
                timestamp = timestamp_from_url(url)

                if url and timestamp:
                    encontrados.append(
                        (timestamp, url)
                    )

    urls_img = page.locator("img").evaluate_all(
        "els => els.map("
        "e => e.src || "
        "e.getAttribute('src') || ''"
        ")"
    )

    for url in urls_img:
        agregar(url)

    agregar(page.content())

    try:
        scripts = page.locator("script").evaluate_all(
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
        unicos[(timestamp, url)] = (
            timestamp,
            url
        )

    return list(unicos.values())


def leer_timestamp_utc(value):
    if not value:
        return None

    try:
        texto = str(value).strip()

        if texto.endswith("Z"):
            texto = texto[:-1] + "+00:00"

        dt = datetime.fromisoformat(texto)

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt.astimezone(
            timezone.utc
        )

    except Exception:
        return None


def timestamp_persistido():
    candidatos = []

    ahora = datetime.now(timezone.utc)

    limite_futuro = (
        ahora
        + timedelta(
            minutes=MAX_FUTURE_SKEW_MINUTES
        )
    )

    for path in (
        PERSISTENT_STATE,
        STATE,
    ):
        if not path.exists():
            continue

        try:
            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            value = data.get(
                "ultimo_radar_timestamp"
            )

            dt = leer_timestamp_utc(value)

            if dt is None:
                continue

            if dt <= limite_futuro:
                candidatos.append(dt)

        except Exception:
            pass

    if TSFILE.exists():
        try:
            texto = TSFILE.read_text(
                encoding="utf-8"
            ).strip()

            dt = leer_timestamp_utc(texto)

            if (
                dt is not None
                and dt <= limite_futuro
            ):
                candidatos.append(dt)

        except Exception:
            pass

    if not candidatos:
        return None

    return max(candidatos)


def seleccionar_mas_reciente(candidatos):
    """
    Selecciona el frame mas reciente.

    Si existe uno posterior al ultimo aceptado,
    devuelve ese.

    Si SINARAME todavia no publico uno nuevo,
    devuelve el ultimo disponible sin generar error.
    """

    if not candidatos:
        return None

    ultimo = timestamp_persistido()

    if ultimo is not None:

        nuevos = []

        for item in candidatos:

            timestamp = item[0]

            fecha = (
                timestamp_source_to_datetime(
                    timestamp
                )
                .astimezone(
                    timezone.utc
                )
            )

            if fecha > ultimo:
                nuevos.append(item)

        if nuevos:
            candidatos = nuevos

        else:
            print(
                "SINARAME todavia no publico "
                "un RMA10 nuevo."
            )

            print(
                f"Ultimo frame aceptado: "
                f"{ultimo.isoformat()}"
            )

            print(
                "Se conservara el frame actual "
                "sin duplicarlo en el historial."
            )

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

        dt += timedelta(hours=3)

        return (
            dt.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
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

        source_timestamp = archivo.stem

        utc_timestamp = (
            source_timestamp_to_utc_timestamp(
                source_timestamp
            )
        )

        destino = (
            HIST /
            f"{utc_timestamp}.png"
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


def escribir_estado(texto):
    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    CAPTURE_STATUS.write_text(
        texto,
        encoding="utf-8"
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
                            texto = select.inner_text()

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
                            "No se encontro selector "
                            "RMA10 en SINARAME"
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
                            "No se encontro Bahia Blanca "
                            "RMA10 en SINARAME"
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
                            "ninguna URL RMA10 PNG"
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

                    ahora = datetime.now(
                        ARGENTINA_TZ
                    )

                    candidatos_validos = []

                    for (
                        timestamp,
                        url,
                        response
                    ) in candidatos:

                        fecha_candidato = (
                            timestamp_source_to_datetime(
                                timestamp
                            )
                        )

                        diferencia_minutos = (
                            fecha_candidato - ahora
                        ).total_seconds() / 60.0

                        if (
                            diferencia_minutos
                            > MAX_FUTURE_SKEW_MINUTES
                        ):

                            print(
                                "DESCARTADO: "
                                "frame futuro: "
                                f"{timestamp} "
                                f"(+{diferencia_minutos:.1f} min)"
                            )

                            continue

                        candidatos_validos.append(
                            (
                                timestamp,
                                url,
                                response
                            )
                        )

                    if not candidatos_validos:

                        raise RuntimeError(
                            "SINARAME solo entrego "
                            "frames futuros "
                            "o invalidos."
                        )

                    for (
                        timestamp,
                        url,
                        _
                    ) in candidatos_validos[:10]:

                        fecha_candidato = (
                            timestamp_source_to_datetime(
                                timestamp
                            )
                        )

                        edad_candidato = (
                            ahora - fecha_candidato
                        ).total_seconds() / 60.0

                        print(
                            f"RMA10 candidato valido: "
                            f"{timestamp} | "
                            f"edad "
                            f"{edad_candidato:.1f} min | "
                            f"{url}"
                        )

                    seleccionado = (
                        seleccionar_mas_reciente(
                            candidatos_validos
                        )
                    )

                    if seleccionado is None:

                        raise RuntimeError(
                            "No se pudo seleccionar "
                            "un frame RMA10."
                        )

                    timestamp, image_url, response = (
                        seleccionado
                    )

                    fecha_seleccionada = (
                        timestamp_source_to_datetime(
                            timestamp
                        )
                    )

                    edad = (
                        ahora
                        - fecha_seleccionada
                    ).total_seconds() / 60.0

                    print(
                        f"Hora fuente RMA10: "
                        f"{timestamp} "
                        f"(Argentina)"
                    )

                    print(
                        f"Edad real del frame: "
                        f"{edad:.1f} minutos"
                    )

                    if edad < -MAX_FUTURE_SKEW_MINUTES:

                        raise RuntimeError(
                            "Timestamp RMA10 "
                            "futuro o invalido: "
                            f"{timestamp}"
                        )

                    if edad > MAX_RADAR_AGE_MINUTES:

                        raise RuntimeError(
                            "SINARAME NO TIENE "
                            "UN RMA10 ACTUALIZADO: "
                            f"el frame mas reciente "
                            f"valido encontrado es de "
                            f"{edad:.1f} minutos. "
                            f"URL: {image_url}"
                        )

                    ultimo = timestamp_persistido()

                    seleccionado_utc = (
                        fecha_seleccionada
                        .astimezone(timezone.utc)
                    )

                    # -----------------------------------------
                    # CASO IMPORTANTE:
                    # SINARAME sigue mostrando el mismo frame.
                    # NO lo volvemos a guardar.
                    # NO hacemos fallar la Action.
                    # -----------------------------------------

                    if (
                        ultimo is not None
                        and seleccionado_utc <= ultimo
                    ):

                        escribir_estado(
                            "SIN_CAMBIO\n"
                            f"frame={timestamp}\n"
                            f"ultimo_aceptado={ultimo.isoformat()}\n"
                            f"edad_minutos={edad:.1f}\n"
                        )

                        print(
                            "SIN CAMBIO: SINARAME "
                            "todavia no publico un frame "
                            "RMA10 posterior."
                        )

                        print(
                            "Se conserva actual.png "
                            "y no se duplica el historial."
                        )

                        return

                    # -----------------------------------------
                    # FRAME NUEVO
                    # -----------------------------------------

                    if response is not None:

                        try:

                            data = response.body()

                            ok, detail = (
                                validar_png(
                                    data
                                )
                            )

                        except Exception:

                            ok = False
                            detail = (
                                "respuesta no disponible"
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

                    escribir_estado(
                        "NUEVO\n"
                        f"frame={timestamp}\n"
                        f"utc={iso}\n"
                        f"edad_minutos={edad:.1f}\n"
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

                    if intento < MAX_RETRIES:

                        time.sleep(4)

                finally:

                    try:
                        page.remove_listener(
                            "response",
                            capturar_respuesta
                        )
                    except Exception:
                        pass

            try:

                page.screenshot(
                    path=str(
                        OUT /
                        "error_rma10.png"
                    ),
                    full_page=True
                )

            except Exception:
                pass

            raise RuntimeError(
                "No se pudo obtener "
                "un RMA10 desde SINARAME. "
                f"Ultimo error: "
                f"{ultimo_error}"
            )

        finally:

            browser.close()


if __name__ == "__main__":
    main()
