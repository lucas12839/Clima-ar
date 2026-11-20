"""Edad máxima de un dato según cada cuánto mide cada red de estaciones.

Una sola regla para todo ClimaAR (auditoría del 20/11/2026): antes todas las
redes usaban 15 min, y las estaciones del SMN (miden una vez por hora y el
dato llega 20-25 min tarde) nunca entraban, ni siquiera Bahía Blanca Aero.

- bcp:  miden cada minuto y publican al instante -> 15 min.
- cwop: transmiten cada ~10 min -> 20 min.
- smn:  parte horario; llega 20-25 min después de la hora y algunas
        estaciones son cada 3 h -> 90 min.
- sazb: informe del aeropuerto cada hora (más los especiales) -> 75 min.

La edad SIEMPRE se calcula de nuevo con la hora de la medición en el momento
de usar el dato: nunca se confía en una marca "fresca" guardada antes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

LIMITE_EDAD_MIN: dict[str, float] = {"bcp": 15.0, "cwop": 20.0, "smn": 90.0, "sazb": 75.0}
LIMITE_POR_DEFECTO_MIN = 15.0
# Más de un día sin datos: la estación está fuera de servicio (se informa aparte).
SIN_DATOS_MIN = 24 * 60.0


def limite_min(red: str | None) -> float:
    return LIMITE_EDAD_MIN.get(str(red or "").lower(), LIMITE_POR_DEFECTO_MIN)


def parse_utc(valor: Any) -> datetime | None:
    if not isinstance(valor, str) or not valor.strip():
        return None
    texto = valor.strip()
    if texto.endswith("Z"):
        texto = texto[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(texto)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc)


def edad_min(hora_medicion: Any, ahora: datetime | None = None) -> float | None:
    dt = hora_medicion if isinstance(hora_medicion, datetime) else parse_utc(hora_medicion)
    if dt is None:
        return None
    ahora = ahora or datetime.now(timezone.utc)
    return (ahora - dt).total_seconds() / 60.0


def es_fresca(hora_medicion: Any, red: str | None, ahora: datetime | None = None) -> bool:
    edad = edad_min(hora_medicion, ahora)
    return edad is not None and 0.0 <= edad <= limite_min(red)


def sin_datos_hace_un_dia(edad: float | None) -> bool:
    return edad is not None and edad > SIN_DATOS_MIN
