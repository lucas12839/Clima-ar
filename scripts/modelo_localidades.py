#!/usr/bin/env python3
"""Modelo europeo (ECMWF) en el punto de cada localidad (24/11/2026).

Antes la cadena usaba el modelo de Bahía Blanca para las 7 localidades (el
viento de Tornquist y Médanos salía de Bahía) y la página pedía por su cuenta
otro modelo (el "mejor disponible" de Open-Meteo) aunque decía "modelo
europeo". Ahora hay una sola fuente:

- meteorologia_fusion.py lo pide UNA vez cada vez que renueva el ECMWF
  (cada 30 min, no en cada ciclo) y lo guarda en estado_meteorologico.json
  ("modelo_localidades"); es el único que lo escribe.
- El motor de decisión (viento de cada localidad) y la página leen de ahí con
  valor_actual(): el valor de la hora más cercana a ahora.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCALIDADES = ROOT / "config" / "localidades.json"
ECMWF_API = "https://api.open-meteo.com/v1/ecmwf"
VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "pressure_msl",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "cloud_cover",
    "precipitation", "weather_code",
]
VARIABLES_MINIMAS = VARIABLES[:-1]       # por si el modelo no da weather_code
TIMEOUT_S = 10
MAX_HORAS_DESDE_HORA = 1.0               # el valor tiene que ser de una hora cercana


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def localidades(path: Path = LOCALIDADES) -> list[dict[str, Any]]:
    try:
        return [l for l in json.loads(path.read_text(encoding="utf-8")).get("localidades") or []
                if l.get("id") and _num(l.get("lat")) is not None and _num(l.get("lon")) is not None]
    except Exception:
        return []


def consultar(ahora: datetime | None = None, get=None) -> dict[str, Any] | None:
    """Un solo pedido para las 7 localidades. None si falla (no corta nada)."""
    import requests

    get = get or requests.get
    locs = localidades()
    if not locs:
        return None
    for variables in (VARIABLES, VARIABLES_MINIMAS):
        params = {
            "latitude": ",".join(f"{l['lat']:.4f}" for l in locs),
            "longitude": ",".join(f"{l['lon']:.4f}" for l in locs),
            "hourly": ",".join(variables),
            "past_hours": 1, "forecast_hours": 6, "timezone": "UTC",
            "wind_speed_unit": "kmh", "cell_selection": "land",
        }
        try:
            r = get(ECMWF_API, params=params, timeout=TIMEOUT_S)
            if getattr(r, "status_code", 200) == 400 and variables is VARIABLES:
                continue
            r.raise_for_status()
            datos = r.json()
        except Exception as exc:
            print(f"AVISO modelo por localidad: {type(exc).__name__}: {exc}"[:200])
            return None
        lista = datos if isinstance(datos, list) else [datos]
        if len(lista) != len(locs):
            print("AVISO modelo por localidad: cantidad de puntos inesperada")
            return None
        salida = {}
        for loc, d in zip(locs, lista):
            h = (d or {}).get("hourly") or {}
            if h.get("time"):
                salida[loc["id"]] = {"horas": h["time"], **{v: h.get(v) for v in variables}}
        if not salida:
            return None
        return {"modelo": "ECMWF IFS", "actualizado_utc": (ahora or datetime.now(timezone.utc)).isoformat(),
                "localidades": salida}
    return None


def valor_actual(modelo: dict[str, Any] | None, loc_id: str,
                 ahora: datetime | None = None) -> dict[str, Any] | None:
    """Los valores de la hora más cercana a ahora (a 1 h o menos), o None."""
    ahora = ahora or datetime.now(timezone.utc)
    loc = ((modelo or {}).get("localidades") or {}).get(loc_id)
    if not isinstance(loc, dict):
        return None
    mejor, dif = None, None
    for i, t in enumerate(loc.get("horas") or []):
        try:
            dt = datetime.fromisoformat(str(t).replace("Z", "")).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        d = abs((dt - ahora).total_seconds()) / 3600.0
        if dif is None or d < dif:
            mejor, dif = i, d
    if mejor is None or dif > MAX_HORAS_DESDE_HORA:
        return None

    def v(nombre):
        serie = loc.get(nombre) or []
        return _num(serie[mejor]) if mejor < len(serie) else None

    return {
        "temperatura": v("temperature_2m"), "humedad": v("relative_humidity_2m"),
        "punto_rocio": v("dew_point_2m"), "presion": v("pressure_msl"),
        "viento_vel": v("wind_speed_10m"), "viento_dir": v("wind_direction_10m"),
        "rafagas": v("wind_gusts_10m"), "nubosidad": v("cloud_cover"),
        "precipitacion": v("precipitation"), "codigo_tiempo": v("weather_code"),
        "hora_utc": loc["horas"][mejor],
    }


def vigente(modelo: dict[str, Any] | None, ahora: datetime | None = None, max_horas: float = 5.0) -> bool:
    """El pedido cubre 6 h: después de 5 h ya no sirve."""
    try:
        dt = datetime.fromisoformat(str((modelo or {}).get("actualizado_utc")).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    return (ahora or datetime.now(timezone.utc)) - dt <= timedelta(hours=max_horas)
