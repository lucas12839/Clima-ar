import json
from datetime import datetime, timezone
from pathlib import Path

import requests


SAZB_URL = (
    "https://aviationweather.gov/api/data/metar"
    "?ids=SAZB&format=json"
)

TIMEOUT = 20

MAX_AGE_MINUTES = 120


BASE_DIR = Path(
    __file__
).resolve().parents[1]


DATA_DIR = (
    BASE_DIR
    / "data"
    / "sazb"
)


STATUS_FILE = (
    DATA_DIR
    / "status.json"
)


LOG_FILE = (
    DATA_DIR
    / "ingesta_log.jsonl"
)


# ============================================================
# TIEMPO
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    )


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

        text = str(
            value
        ).strip()


        if text.endswith("Z"):

            text = (
                text[:-1]
                + "+00:00"
            )


        dt = datetime.fromisoformat(
            text
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
# VALIDACION
# ============================================================

def validar_observacion(data):

    if (
        not isinstance(data, list)
        or
        not data
    ):

        return (
            False,
            "SAZB no devolvio observaciones"
        )


    obs = data[0]


    if obs.get(
        "icaoId"
    ) != "SAZB":

        return (
            False,
            "La observacion no corresponde a SAZB"
        )


    obstime = parse_obstime(
        obs.get(
            "obsTime"
        )
    )


    if obstime is None:

        return (
            False,
            "SAZB no tiene obsTime valido"
        )


    ahora = utc_now()


    edad_min = (
        ahora - obstime
    ).total_seconds() / 60


    if edad_min < -5:

        return (
            False,
            "Timestamp futuro invalido"
        )


    if (
        edad_min
        >
        MAX_AGE_MINUTES
    ):

        return (
            False,
            (
                "Observacion demasiado vieja: "
                f"{edad_min:.1f} minutos"
            )
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

        return (
            False,
            "SAZB no contiene variables meteorologicas"
        )


    resultado = {

        "integrado": True,

        "observacion_valida": True,

        "estado": "ok",

        "fuente":
            "Aviation Weather Center",

        "estacion":
            "SAZB",

        "estacion_nombre":
            "Bahia Blanca Aero",

        "icao":
            "SAZB",

        "obsTime":
            obs.get("obsTime"),

        "obsTime_utc":
            obstime.isoformat(),

        "edad_minutos":
            round(
                edad_min,
                1
            ),

        "temperatura_c":
            obs.get("temp"),

        "punto_rocio_c":
            obs.get("dewp"),

        "viento_kt":
            obs.get("wspd"),

        "direccion_viento":
            obs.get("wdir"),

        "presion_hpa":
            obs.get("altim"),

        "visibilidad_millas":
            obs.get("visib"),

        "actualizado_utc":
            ahora.isoformat(),

        "ultimo_error":
            None,

    }


    return (
        True,
        resultado
    )


# ============================================================
# CARGA ESTADO ANTERIOR
# ============================================================

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


# ============================================================
# GUARDADO SEGURO
# ============================================================

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


# ============================================================
# LOG
# ============================================================

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
            )
            + "\n"
        )


# ============================================================
# CONSERVAR ULTIMA OBSERVACION UTIL
# ============================================================

def conservar_ultima_observacion(
    error
):

    anterior = (
        cargar_status_actual()
    )


    ahora = utc_now()


    # --------------------------------------------------------
    # No existe observacion anterior
    # --------------------------------------------------------

    if (
        not isinstance(
            anterior,
            dict
        )
        or
        not anterior.get(
            "obsTime_utc"
        )
    ):

        status = {

            "integrado": False,

            "observacion_valida": False,

            "estado": "sin_datos",

            "fuente":
                "Aviation Weather Center",

            "estacion":
                "SAZB",

            "error":
                error,

            "actualizado_utc":
                ahora.isoformat(),

        }


        guardar_status(
            status
        )


        guardar_log(
            status
        )


        return status


    # --------------------------------------------------------
    # Recuperar timestamp anterior
    # --------------------------------------------------------

    obstime = parse_iso(
        anterior.get(
            "obsTime_utc"
        )
    )


    if obstime is None:

        status = {

            "integrado": False,

            "observacion_valida": False,

            "estado": "sin_datos",

            "fuente":
                "Aviation Weather Center",

            "estacion":
                "SAZB",

            "error":
                error,

            "actualizado_utc":
                ahora.isoformat(),

        }


        guardar_status(
            status
        )


        guardar_log(
            status
        )


        return status


    # --------------------------------------------------------
    # Calcular edad real
    # --------------------------------------------------------

    edad_min = (
        ahora - obstime
    ).total_seconds() / 60


    usable = (
        edad_min
        <=
        MAX_AGE_MINUTES
    )


    # --------------------------------------------------------
    # Conservar datos meteorologicos
    # --------------------------------------------------------

    status = dict(
        anterior
    )


    status[
        "integrado"
    ] = usable


    status[
        "observacion_valida"
    ] = usable


    status[
        "estado"
    ] = (
        "degradado"
        if usable
        else
        "stale"
    )


    status[
        "edad_minutos"
    ] = round(
        edad_min,
        1
    )


    status[
        "actualizado_utc"
    ] = ahora.isoformat()


    status[
        "ultimo_error"
    ] = error


    guardar_status(
        status
    )


    guardar_log({

        "tipo":
            "error_ingesta",

        "actualizado_utc":
            ahora.isoformat(),

        "error":
            error,

        "obsTime_utc":
            status.get(
                "obsTime_utc"
            ),

        "edad_minutos":
            status.get(
                "edad_minutos"
            ),

    })


    return status


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        headers = {

            "User-Agent":
                (
                    "ClimaAR/1.1 "
                    "meteorological observation client"
                )

        }


        response = requests.get(

            SAZB_URL,

            headers=headers,

            timeout=TIMEOUT,

        )


        response.raise_for_status()


        data = response.json()


        valido, resultado = (
            validar_observacion(
                data
            )
        )


        if not valido:

            raise RuntimeError(
                resultado
            )


        guardar_status(
            resultado
        )


        guardar_log(
            resultado
        )


        print(
            "OK: observacion SAZB integrada"
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

        status = (
            conservar_ultima_observacion(
                str(exc)
            )
        )


        print(
            "AVISO: no se pudo actualizar "
            "SAZB; se conserva la ultima "
            "observacion util."
        )


        print(
            json.dumps(
                status,
                indent=2,
                ensure_ascii=False
            )
        )


        # La falla de SAZB no debe detener
        # toda la inteligencia de ClimaAR.

        return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
