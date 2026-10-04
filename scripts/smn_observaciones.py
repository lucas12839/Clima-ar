import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests


SAZB_URL = "https://aviationweather.gov/api/data/metar?ids=SAZB&format=json"
TIMEOUT = 20
MAX_AGE_MINUTES = 120

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "sazb"
STATUS_FILE = DATA_DIR / "status.json"
LOG_FILE = DATA_DIR / "ingesta_log.jsonl"


def utc_now():
    return datetime.now(timezone.utc)


def parse_obstime(value):
    if value is None:
        return None

    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except Exception:
        return None


def validar_observacion(data):
    if not isinstance(data, list) or not data:
        return False, "SAZB no devolvio observaciones"

    obs = data[0]

    if obs.get("icaoId") != "SAZB":
        return False, "La observacion no corresponde a SAZB"

    obstime = parse_obstime(obs.get("obsTime"))

    if obstime is None:
        return False, "SAZB no tiene obsTime valido"

    ahora = utc_now()
    edad_min = (ahora - obstime).total_seconds() / 60

    if edad_min < -5:
        return False, "Timestamp futuro invalido"

    if edad_min > MAX_AGE_MINUTES:
        return False, f"Observacion demasiado vieja: {edad_min:.1f} minutos"

    variables = [
        obs.get("temp"),
        obs.get("dewp"),
        obs.get("wspd"),
        obs.get("wdir"),
        obs.get("altim"),
        obs.get("visib"),
    ]

    if not any(v is not None for v in variables):
        return False, "SAZB no contiene variables meteorologicas"

    resultado = {
        "integrado": True,
        "observacion_valida": True,
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
    }

    return True, resultado


def guardar_log(registro):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def main():
    try:
        headers = {
            "User-Agent": "ClimaAR/1.0 meteorological observation client"
        }

        response = requests.get(
            SAZB_URL,
            headers=headers,
            timeout=TIMEOUT,
        )

        response.raise_for_status()
        data = response.json()

        valido, resultado = validar_observacion(data)

        if not valido:
            status = {
                "integrado": False,
                "observacion_valida": False,
                "fuente": "Aviation Weather Center",
                "estacion": "SAZB",
                "error": resultado,
                "actualizado_utc": utc_now().isoformat(),
            }

            DATA_DIR.mkdir(parents=True, exist_ok=True)

            STATUS_FILE.write_text(
                json.dumps(status, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            guardar_log(status)

            raise RuntimeError(resultado)

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        STATUS_FILE.write_text(
            json.dumps(resultado, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        guardar_log(resultado)

        print("OK: observacion SAZB integrada")
        print(json.dumps(resultado, indent=2, ensure_ascii=False))

    except Exception as e:
        status = {
            "integrado": False,
            "observacion_valida": False,
            "fuente": "Aviation Weather Center",
            "estacion": "SAZB",
            "error": str(e),
            "actualizado_utc": utc_now().isoformat(),
        }

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        STATUS_FILE.write_text(
            json.dumps(status, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        guardar_log(status)

        print(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
