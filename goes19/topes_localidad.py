"""Topes de nubes (canal 13, infrarrojo de 10,3 µm) sobre cada localidad.

Una tormenta que crece rápido enfría sus topes minutos antes de que el radar
muestre el núcleo intenso: ese enfriamiento y los topes muy fríos son de las
mejores señales satelitales de granizo y ráfagas. Además permiten seguir la
convección cuando el radar no llega.

Por localidad (config/localidades.json), a <= 15 km y a <= 30 km:
- temperatura mínima de topes y percentil 10 (°C);
- enfriamiento respecto de la imagen anterior (°C cada 10 min).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCALIDADES = ROOT / "config" / "localidades.json"
CERCA_KM = 30.0
SOBRE_KM = 15.0
K = 273.15


def _dist(lat1, lon1, lat2, lon2):
    r = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _parse(texto):
    try:
        return datetime.fromisoformat(str(texto).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


# Desde la versión 1.6 los píxeles se ubican con las fórmulas oficiales de
# NOAA. Los estados anteriores tenían los topes de otra zona: no se comparan.
VERSION_MINIMA_PREVIA = (1, 6)


def _version(texto) -> tuple[int, ...]:
    try:
        return tuple(int(p) for p in str(texto).split("."))
    except ValueError:
        return (0,)


def _anterior(path: Path) -> tuple[dict, datetime | None]:
    try:
        previo = json.loads(path.read_text(encoding="utf-8"))
        if _version(previo.get("version")) < VERSION_MINIMA_PREVIA:
            return {}, None
        c13 = (previo.get("products") or {}).get("abi_c13") or {}
        return c13.get("por_localidad") or {}, _parse((c13.get("source") or {}).get("end_time_utc"))
    except Exception:
        return {}, None


def por_localidad(bt_k, lat, lon, fin_utc: str | None, estado_previo: Path,
                  localidades: list[dict] | None = None) -> dict[str, Any]:
    if localidades is None:
        try:
            localidades = json.loads(LOCALIDADES.read_text(encoding="utf-8")).get("localidades") or []
        except Exception:
            localidades = []
    bt = np.asarray(bt_k, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ok = np.isfinite(bt) & np.isfinite(lat) & np.isfinite(lon)
    bt, lat, lon = bt[ok], lat[ok], lon[ok]
    previos, t_prev = _anterior(estado_previo)
    t_act = _parse(fin_utc)
    minutos = (t_act - t_prev).total_seconds() / 60.0 if t_act and t_prev else None
    salida = {}
    for loc in localidades:
        d = _dist(loc["lat"], loc["lon"], lat, lon)
        item = {}
        for nombre, radio in (("15km", SOBRE_KM), ("30km", CERCA_KM)):
            v = bt[d <= radio]
            if v.size:
                item[f"min_c_{nombre}"] = round(float(v.min()) - K, 1)
                item[f"p10_c_{nombre}"] = round(float(np.percentile(v, 10)) - K, 1)
        prev = previos.get(loc["id"]) or {}
        if minutos and 5 <= minutos <= 30 and "p10_c_30km" in item and "p10_c_30km" in prev:
            item["enfriamiento_c_10min"] = round((prev["p10_c_30km"] - item["p10_c_30km"]) * 10.0 / minutos, 1)
        salida[loc["id"]] = item
    return salida


def estadisticas(bt_k, lat, lon, localidades: list[dict]) -> dict[str, dict[str, float]]:
    """Mínimo y percentil 10 del tope a <= 15 y <= 30 km de cada localidad."""
    bt = np.asarray(bt_k, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ok = np.isfinite(bt) & np.isfinite(lat) & np.isfinite(lon)
    bt, lat, lon = bt[ok], lat[ok], lon[ok]
    salida = {}
    for loc in localidades:
        d = _dist(loc["lat"], loc["lon"], lat, lon)
        item = {}
        for nombre, radio in (("15km", SOBRE_KM), ("30km", CERCA_KM)):
            v = bt[d <= radio]
            if v.size:
                item[f"min_c_{nombre}"] = round(float(v.min()) - K, 1)
                item[f"p10_c_{nombre}"] = round(float(np.percentile(v, 10)) - K, 1)
        salida[loc["id"]] = item
    return salida


def con_enfriamiento(actual: dict, previo: dict, minutos: float | None) -> dict[str, dict[str, float]]:
    """Agrega el enfriamiento (°C cada 10 min, p10 a 30 km) entre dos imágenes."""
    salida = {}
    for loc_id, item in actual.items():
        item = dict(item)
        prev = previo.get(loc_id) or {}
        if minutos and 5 <= minutos <= 30 and "p10_c_30km" in item and "p10_c_30km" in prev:
            item["enfriamiento_c_10min"] = round((prev["p10_c_30km"] - item["p10_c_30km"]) * 10.0 / minutos, 1)
        salida[loc_id] = item
    return salida
