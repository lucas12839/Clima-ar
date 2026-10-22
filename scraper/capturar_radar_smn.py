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

MIGRATION_MARKER = OUT / ".timestamp_rma10_utc_v2"

MAX_RETRIES = 5

MAX_RADAR_AGE_MINUTES = 45

MAX_FUTURE_SKEW_MINUTES = 2

ARGENTINA_TZ = ZoneInfo(
    "America/Argentina/Buenos_Aires"
)


# ============================================================
# VALIDACIÓN PNG
# ============================================================

def validar_png(data):

    if not data or len(data) < 1000:
        return False, "archivo vacio/pequeno"

    try:

        im = Image.open(
            io.BytesIO(data)
        ).convert("RGBA")

        if im.width < 100 or im.height < 100:
            return (
                False,
                f"resolucion {im.width}x{im.height}"
            )

        alpha = im.getchannel("A")

        if alpha.getextrema()[1] < 3:
            return (
                False,
                "imagen completamente transparente"
            )

        visible = sum(
            1
            for a in alpha.getdata()
            if a > 2
        )

        if (
            visible /
            (im.width * im.height)
            < 0.01
        ):
            return (
                False,
                "menos de 1% de pixeles visibles"
            )

        return (
            True,
            f"{im.width}x{im.height}, {len(data)} bytes"
        )

    except Exception as exc:

        return (
            False,
            f"PNG invalido: {exc}"
        )


# ============================================================
# TIMESTAMP DESDE URL
# ============================================================

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


# ============================================================
# TIMESTAMP DEL ORIGEN
# ============================================================

def timestamp_source_to_datetime(timestamp):

    return datetime.strptime(
        timestamp,
        "%Y%m%d%H%M%S"
    ).replace(
        tzinfo=ARGENTINA_TZ
    )


# ============================================================
# NORMALIZAR URL
# ============================================================

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


# ============================================================
# EXTRAER URLs RMA10
# ============================================================

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


# ============================================================
# LEER TIMESTAMP UTC
# ============================================================

def leer_timestamp_utc(value):

    if not value:
        return None

    try:

        texto = str(
            value
        ).strip()

        if texto.endswith("Z"):

            texto = (
                texto[:-1]
                + "+00:00"
            )

        dt = datetime.fromisoformat(
            texto
        )

        if dt.tzinfo is None:

            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt.astimezone(
            timezone.utc
        )

    except Exception:

        return None


# ============================================================
# OBTENER ÚLTIMO TIMESTAMP ACEPTADO
# ============================================================

def timestamp_persistido():

    candidatos = []

    ahora = datetime.now(
        timezone.utc
    )

    limite_futuro = (
        ahora
        + timedelta(
            minutes=MAX_FUTURE_SKEW_MINUTES
        )
    )

    for path in (
        PERSISTENT_STATE,
        STATE
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

            dt = leer_timestamp_utc(
                value
            )

            if (
                dt is not None
                and
                dt <= limite_futuro
            ):

                candidatos.append(
                    dt
                )

        except Exception:
            pass

    if TSFILE.exists():

        try:

            texto = TSFILE.read_text(
                encoding="utf-8"
            ).strip()

            dt = leer_timestamp_utc(
                texto
            )

            if (
                dt is not None
                and
                dt <= limite_futuro
            ):

                candidatos.append(
                    dt
                )

        except Exception:
            pass

    if not candidatos:
        return None

    return max(
        candidatos
    )


# ============================================================
# SELECCIONAR FRAME MÁS RECIENTE
# ============================================================

def seleccionar_mas_reciente(
    candidatos
):

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

                nuevos.append(
                    item
                )

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
                "sin duplicarlo."
            )

            return None

    return max(
        candidatos,
        key=lambda item: item[0]
    )


# ============================================================
# DESCARGAR IMAGEN
# ============================================================

def descargar_imagen(
    page,
    url
):

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

    return (
        data,
        detail
    )


# ============================================================
# MIGRACIÓN SEGURA
# ============================================================

def migrar_historico_legacy():

    """
    No modifica timestamps internos.

    No suma ni resta 3 horas.

    No modifica JSON existentes.

    Solo crea un marcador para impedir
    migraciones peligrosas posteriores.
    """

    if MIGRATION_MARKER.exists():
        return

    HIST.mkdir(
        parents=True,
        exist_ok=True
    )

    MIGRATION_MARKER.write_text(
        "RMA10 timestamps internos en UTC\n",
        encoding="utf-8"
    )


# ============================================================
# ESTADO DE CAPTURA
# ============================================================

def escribir_estado(
    texto
):

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    CAPTURE_STATUS.write_text(
        texto,
        encoding="utf-8"
    )


# ============================================================
# GUARDAR ESTADO PERSISTENTE
# ============================================================

def guardar_estado_persistente(
    timestamp_iso
):

    PERSISTENT_STATE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data = {}

    if PERSISTENT_STATE.exists():

        try:

            data = json.loads(
                PERSISTENT_STATE.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:

            data = {}

    data[
        "ultimo_radar_timestamp"
    ] = timestamp_iso

    PERSISTENT_STATE.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


# ============================================================
# GUARDAR ESTADO LOCAL
# ============================================================

def guardar_estado_local(
    timestamp_iso
):

    STATE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data = {}

    if STATE.exists():

        try:

            data = json.loads(
                STATE.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:

            data = {}

    data[
        "ultimo_radar_timestamp"
    ] = timestamp_iso

    STATE.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


# ============================================================
# ACEPTAR FRAME
# ============================================================

def aceptar_frame(
    timestamp_source,
    data,
    detail
):

    fecha_utc = (
        timestamp_source_to_datetime(
            timestamp_source
        )
        .astimezone(
            timezone.utc
        )
    )

    ahora = datetime.now(
        timezone.utc
    )

    if fecha_utc > (
        ahora
        + timedelta(
            minutes=MAX_FUTURE_SKEW_MINUTES
        )
    ):

        raise RuntimeError(
            "El frame RMA10 tiene "
            "timestamp futuro."
        )

    edad = (
        ahora - fecha_utc
    ).total_seconds() / 60.0

    if edad < 0:
        edad = 0

    if edad > MAX_RADAR_AGE_MINUTES:

        # Es un frame real y mas nuevo que el ultimo aceptado: se
        # publica igualmente. Descartarlo dejaba el radar congelado.
        print(
            "ADVERTENCIA: el frame RMA10 llego atrasado "
            f"({edad:.1f} min); se acepta por ser el mas nuevo."
        )

    utc_iso = (
        fecha_utc.isoformat()
    )

    nombre = fecha_utc.strftime(
        "%Y%m%d%H%M%S"
    )

    HIST.mkdir(
        parents=True,
        exist_ok=True
    )

    tmp_actual = (
        OUT / "actual.png.tmp"
    )

    tmp_actual.write_bytes(
        data
    )

    tmp_actual.replace(
        ACTUAL
    )

    destino = (
        HIST /
        f"{nombre}.png"
    )

    if not destino.exists():

        destino.write_bytes(
            data
        )

    TSFILE.write_text(
        utc_iso,
        encoding="utf-8"
    )

    guardar_estado_persistente(
        utc_iso
    )

    guardar_estado_local(
        utc_iso
    )

    escribir_estado(
        "NUEVO\n"
        f"frame={utc_iso}\n"
        f"edad_minutos={edad:.1f}\n"
        f"detalle={detail}\n"
    )

    print(
        "FRAME NUEVO ACEPTADO"
    )

    print(
        "Timestamp UTC:",
        utc_iso
    )

    print(
        "Edad:",
        f"{edad:.1f} min"
    )

    print(
        "Archivo:",
        destino
    )


# ============================================================
# MAIN
# ============================================================

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

    escribir_estado(
        "INICIANDO\n"
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

                respuestas_radar = []

                def capturar_respuesta(
                    response
                ):

                    url = response.url

                    if (
                        "/cache/RMA10/" in url
                        and
                        ".png" in url.lower()
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
                                    url
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
                            "RMA10"
                            in texto.upper()
                            or
                            "BAHÍA BLANCA"
                            in texto.upper()
                            or
                            "BAHIA BLANCA"
                            in texto.upper()
                        ):

                            value = valor

                            print(
                                "Radar seleccionado:",
                                texto,
                                value
                            )

                            break

                    if value is None:

                        raise RuntimeError(
                            "No se encontro opcion RMA10."
                        )

                    selector.select_option(
                        value
                    )

                    page.wait_for_timeout(
                        12000
                    )

                    candidatos = list(
                        respuestas_radar
                    )

                    candidatos.extend(
                        extraer_urls_rma10(
                            page
                        )
                    )

                    unicos = {}

                    for timestamp, url in candidatos:

                        unicos[
                            (timestamp, url)
                        ] = (
                            timestamp,
                            url
                        )

                    candidatos = list(
                        unicos.values()
                    )

                    print(
                        "Candidatos RMA10 encontrados:",
                        len(candidatos)
                    )

                    elegido = (
                        seleccionar_mas_reciente(
                            candidatos
                        )
                    )

                    if elegido is None:

                        ultimo = (
                            TSFILE.read_text(
                                encoding="utf-8"
                            ).strip()
                            if TSFILE.exists()
                            else "desconocido"
                        )

                        escribir_estado(
                            "SIN_CAMBIO\n"
                            f"ultimo_aceptado={ultimo}\n"
                        )

                        print(
                            "SIN_CAMBIO: "
                            "se conserva el frame actual."
                        )

                        return 0

                    timestamp, url = elegido

                    print(
                        "Frame seleccionado:",
                        timestamp,
                        url
                    )

                    data, detail = (
                        descargar_imagen(
                            page,
                            url
                        )
                    )

                    aceptar_frame(
                        timestamp,
                        data,
                        detail
                    )

                    return 0

                except Exception as exc:

                    ultimo_error = exc

                    print(
                        "Error en intento:",
                        exc
                    )

                    try:

                        page.screenshot(
                            path=(
                                OUT /
                                f"captura_error_{intento}.png"
                            ),
                            full_page=True
                        )

                    except Exception:
                        pass

                    if intento < MAX_RETRIES:

                        time.sleep(
                            3
                        )

                finally:

                    try:

                        page.remove_listener(
                            "response",
                            capturar_respuesta
                        )

                    except Exception:
                        pass

            escribir_estado(
                "ERROR\n"
                f"mensaje={ultimo_error}\n"
            )

            raise RuntimeError(
                "No se pudo obtener un RMA10 valido. "
                f"Ultimo error: {ultimo_error}"
            )

        finally:

            browser.close()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
