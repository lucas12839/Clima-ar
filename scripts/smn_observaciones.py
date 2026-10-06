import csv
import gzip
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


SAZB_URL = (
    "https://aviationweather.gov/api/data/metar"
    "?ids=SAZB&format=json"
)

SAZB_CACHE_URL = (
    "https://aviationweather.gov/data/cache/"
    "metars.cache.csv.gz"
)

DIRECT_TIMEOUT = 10
CACHE_TIMEOUT = 20
DIRECT_ATTEMPTS = 2
MAX_AGE_MINUTES = 120

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "sazb"
STATUS_FILE = DATA_DIR / "status.json"
LOG_FILE = DATA_DIR / "ingesta_log.jsonl"

USER_AGENT = (
    "ClimaAR/1.2 "
    "meteorological observation client"
)


def utc_now():
    return datetime.now(timezone.utc)


def parse_obstime(value):
    if value is None:
        return None

    try:
        return datetime.fromtimestamp(
            float(value),
            tz=timezone.utc
        )
    except Exception:
        return None


def parse_iso(value):
    if not value:
        return None

    try:
        text = str(value).strip()

        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        dt = datetime.fromisoformat(text)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.astimezone(timezone.utc)

    except Exception:
        return None


def parse_cache_time(value):
    if not value:
        return None

    text = str(value).strip()

    for candidate in (
        text,
        text.replace("Z", "+00:00"),
    ):
        try:
            dt = datetime.fromisoformat(candidate)

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            return dt.astimezone(timezone.utc)

        except Exception:
            pass

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ):
        try:
            return datetime.strptime(
                text,
                fmt
            ).replace(tzinfo=timezone.utc)

        except Exception:
            pass

    return None


def numero(value):
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.upper() in {
        "M",
        "NA",
        "NULL",
        "NONE",
    }:
        return None

    try:
        return float(text)

    except Exception:
        return None


def fahrenheit_a_celsius(value):
    number = numero(value)

    if number is None:
        return None

    return round(
        (number - 32.0) * 5.0 / 9.0,
        1
    )


def pulgadas_a_hpa(value):
    number = numero(value)

    if number is None:
        return None

    return round(
        number * 33.8638866667,
        1
    )


def validar_observacion(data):

    if not isinstance(data, list):
        return False, "SAZB no devolvio una lista"

    if not data:
        return False, "SAZB no devolvio observaciones"

    obs = data[0]

    if obs.get("icaoId") != "SAZB":
        return False, "La observacion no corresponde a SAZB"

    obstime = parse_obstime(
        obs.get("obsTime")
    )

    if obstime is None:
        return False, "SAZB no tiene obsTime valido"

    ahora = utc_now()

    edad_min = (
        ahora - obstime
    ).total_seconds() / 60

    if edad_min < -5:
        return False, "Timestamp futuro invalido"

    if edad_min > MAX_AGE_MINUTES:
        return False, (
            "Observacion demasiado vieja: "
            f"{edad_min:.1f} minutos"
        )

    variables = [
        obs.get("temp"),
        obs.get("dewp"),
        obs.get("wspd"),
        obs.get("wdir"),
        obs.get("altim"),
        obs.get("visib"),
    ]

    if not any(
        value is not None
        for value in variables
    ):
        return False, (
            "SAZB no contiene variables "
            "meteorologicas"
        )

    resultado = {
        "integrado": True,
        "observacion_valida": True,
        "estado": "ok",
        "fuente": "Aviation Weather Center",
        "estacion": "SAZB",
        "estacion_nombre": "Bahia Blanca Aero",
        "icao": "SAZB",
        "obsTime": obs.get("obsTime"),
        "obsTime_utc": obstime.isoformat(),
        "edad_minutos": round(edad_min, 1),
        "temperatura_c": obs.get("temp"),
        "punto_rocio_c": obs.get("dewp"),
        "viento_kt": obs.get("wspd"),
        "direccion_viento": obs.get("wdir"),
        "presion_hpa": obs.get("altim"),
        "visibilidad_millas": obs.get("visib"),
        "actualizado_utc": ahora.isoformat(),
        "ultimo_error": None,
    }

    return True, resultado


def observacion_desde_cache(contenido):

    if contenido[:2] == b"\x1f\x8b":
        contenido = gzip.decompress(contenido)

    texto = contenido.decode(
        "utf-8",
        errors="replace"
    )

    reader = csv.DictReader(
        io.StringIO(texto)
    )

    filas = []

    for row in reader:

        station = (
            row.get("station")
            or row.get("icao")
            or row.get("icaoId")
        )

        if str(station).strip().upper() != "SAZB":
            continue

        obstime = parse_cache_time(
            row.get("valid")
        )

        if obstime is None:
            continue

        filas.append(
            (obstime, row)
        )

    if not filas:
        raise RuntimeError(
            "SAZB no aparece en el cache METAR de AWC"
        )

    obstime, row = max(
        filas,
        key=lambda item: item[0]
    )

    ahora = utc_now()

    edad_min = (
        ahora - obstime
    ).total_seconds() / 60

    if edad_min < -5:
        raise RuntimeError(
            "SAZB del cache tiene timestamp futuro"
        )

    if edad_min > MAX_AGE_MINUTES:
        raise RuntimeError(
            "SAZB del cache esta demasiado viejo: "
            f"{edad_min:.1f} minutos"
        )

    temp_c = fahrenheit_a_celsius(
        row.get("tmpf")
    )

    dewp_c = fahrenheit_a_celsius(
        row.get("dwpf")
    )

    wspd = numero(
        row.get("sknt")
    )

    wdir = numero(
        row.get("drct")
    )

    altim_hpa = pulgadas_a_hpa(
        row.get("alti")
    )

    visib = row.get("vsby")

    variables = [
        temp_c,
        dewp_c,
        wspd,
        wdir,
        altim_hpa,
        visib,
    ]

    if not any(
        value is not None and value != ""
        for value in variables
    ):
        raise RuntimeError(
            "El cache SAZB no contiene "
            "variables meteorologicas"
        )

    return {
        "integrado": True,
        "observacion_valida": True,
        "estado": "ok",
        "fuente": (
            "Aviation Weather Center - "
            "cache METAR"
        ),
        "estacion": "SAZB",
        "estacion_nombre": "Bahia Blanca Aero",
        "icao": "SAZB",
        "obsTime": int(
            obstime.timestamp()
        ),
        "obsTime_utc": obstime.isoformat(),
        "edad_minutos": round(
            edad_min,
            1
        ),
        "temperatura_c": temp_c,
        "punto_rocio_c": dewp_c,
        "viento_kt": wspd,
        "direccion_viento": wdir,
        "presion_hpa": altim_hpa,
        "visibilidad_millas": visib,
        "actualizado_utc": ahora.isoformat(),
        "ultimo_error": None,
    }


def obtener_directo():

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

    ultimo_error = None

    for intento in range(
        1,
        DIRECT_ATTEMPTS + 1
    ):

        try:

            response = requests.get(
                SAZB_URL,
                headers=headers,
                timeout=DIRECT_TIMEOUT,
            )

            response.raise_for_status()

            data = response.json()

            valido, resultado = (
                validar_observacion(data)
            )

            if not valido:
                raise RuntimeError(
                    resultado
                )

            resultado[
                "intento_fuente"
            ] = f"directo_{intento}"

            return resultado

        except Exception as exc:

            ultimo_error = exc

            print(
                "Aviso: fallo AWC directo "
                f"intento {intento}: {exc}"
            )

    raise RuntimeError(
        "AWC directo fallo despues de "
        f"{DIRECT_ATTEMPTS} intentos: "
        f"{ultimo_error}"
    )


def obtener_cache():

    headers = {
        "User-Agent": USER_AGENT,
    }

    response = requests.get(
        SAZB_CACHE_URL,
        headers=headers,
        timeout=CACHE_TIMEOUT,
    )

    response.raise_for_status()

    resultado = observacion_desde_cache(
        response.content
    )

    resultado[
        "intento_fuente"
    ] = "cache_metars"

    return resultado


def cargar_status_actual():

    if not STATUS_FILE.exists():
        return None

    try:

        return json.loads(
            STATUS_FILE.read_text(
                encoding="utf-8"
            )
        )

    except Exception:

        return None


def guardar_status(data):

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    temp = STATUS_FILE.with_suffix(
        ".json.tmp"
    )

    temp.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False
        ),
        encoding="utf-8"
    )

    temp.replace(
        STATUS_FILE
    )


def guardar_log(registro):

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with LOG_FILE.open(
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                registro,
                ensure_ascii=False
            ) + "\n"
        )


def conservar_ultima_observacion(
    error
):

    anterior = cargar_status_actual()

    ahora = utc_now()

    if (
        not isinstance(anterior, dict)
        or not anterior.get("obsTime_utc")
    ):

        status = {
            "integrado": False,
            "observacion_valida": False,
            "estado": "sin_datos",
            "fuente": (
                "Aviation Weather Center"
            ),
            "estacion": "SAZB",
            "error": error,
            "actualizado_utc": (
                ahora.isoformat()
            ),
        }

        guardar_status(status)
        guardar_log(status)

        return status

    obstime = parse_iso(
        anterior.get("obsTime_utc")
    )

    if obstime is None:

        status = {
            "integrado": False,
            "observacion_valida": False,
            "estado": "sin_datos",
            "fuente": (
                "Aviation Weather Center"
            ),
            "estacion": "SAZB",
            "error": error,
            "actualizado_utc": (
                ahora.isoformat()
            ),
        }

        guardar_status(status)
        guardar_log(status)

        return status

    edad_min = (
        ahora - obstime
    ).total_seconds() / 60

    usable = (
        edad_min <= MAX_AGE_MINUTES
    )

    status = dict(anterior)

    status["integrado"] = usable
    status[
        "observacion_valida"
    ] = usable

    status["estado"] = (
        "degradado"
        if usable
        else "stale"
    )

    status["edad_minutos"] = round(
        edad_min,
        1
    )

    status[
        "actualizado_utc"
    ] = ahora.isoformat()

    status[
        "ultimo_error"
    ] = error

    guardar_status(status)

    guardar_log({
        "tipo": "error_ingesta",
        "actualizado_utc": (
            ahora.isoformat()
        ),
        "error": error,
        "obsTime_utc": (
            status.get("obsTime_utc")
        ),
        "edad_minutos": (
            status.get("edad_minutos")
        ),
    })

    return status


def main():

    errores = []

    try:

        resultado = obtener_directo()

        guardar_status(resultado)
        guardar_log(resultado)

        print(
            "OK: SAZB actualizado "
            "desde API AWC"
        )

        print(
            json.dumps(
                resultado,
                indent=2,
                ensure_ascii=False
            )
        )

        return 0

    except Exception as exc:

        errores.append(
            f"API AWC: {exc}"
        )

        print(
            f"Aviso: {errores[-1]}"
        )

    try:

        resultado = obtener_cache()

        guardar_status(resultado)
        guardar_log(resultado)

        print(
            "OK: SAZB actualizado "
            "desde cache METAR AWC"
        )

        print(
            json.dumps(
                resultado,
                indent=2,
                ensure_ascii=False
            )
        )

        return 0

    except Exception as exc:

        errores.append(
            f"Cache AWC: {exc}"
        )

        print(
            f"Aviso: {errores[-1]}"
        )

    error_final = " | ".join(
        errores
    )

    status = conservar_ultima_observacion(
        error_final
    )

    print(
        "AVISO: no se pudo actualizar SAZB; "
        "se conserva la ultima observacion util."
    )

    print(
        json.dumps(
            status,
            indent=2,
            ensure_ascii=False
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
