"""Rayos GOES-19 (GLM) de los últimos 20 minutos, por localidad.

Cada archivo GLM-L2-LCFA cubre solo 20 segundos: contar rayos en un único
archivo casi nunca ve nada. Acá se juntan todos los archivos de los últimos
20 min (60 archivos, ~200 KB cada uno) y se calcula, para Bahía Blanca y
cada localidad de config/localidades.json:

- rayos por minuto a <= 30 km en los últimos 10 min y en los 10 min previos
  (el salto brusco de rayos anticipa granizo y ráfagas 10-20 min antes);
- rayos a <= 15 km ("sobre la localidad") y el rayo más cercano.

La ventana termina en el archivo más reciente publicado (no en la hora del
radar): con el radar atrasado los rayos siguen siendo de ahora.
"""
from __future__ import annotations

import json
import math
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .goes19_engine import PRODUCT_SPECS, _enrich_object_metadata, _list_candidate_objects, _parse_utc

ROOT = Path(__file__).resolve().parents[1]
LOCALIDADES = ROOT / "config" / "localidades.json"
VENTANA_MIN = 20
MITAD_MIN = 10
CERCA_KM = 30.0
SOBRE_KM = 15.0
MAX_ARCHIVOS = 66
HILOS = 8


def distancia_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def cargar_localidades(path: Path = LOCALIDADES) -> list[dict]:
    try:
        return list(json.loads(path.read_text(encoding="utf-8")).get("localidades") or [])
    except Exception:
        return []


def resumir(lat: np.ndarray, lon: np.ndarray, minutos_antes: np.ndarray,
            localidades: list[dict], cobertura_min: float) -> dict[str, Any]:
    """Estadística pura (sin red): rayos con su antigüedad en minutos (0 = fin de ventana)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    edad = np.asarray(minutos_antes, dtype=float)
    reciente = edad < MITAD_MIN
    previa = (edad >= MITAD_MIN) & (edad < 2 * MITAD_MIN)
    min_rec = min(MITAD_MIN, cobertura_min)
    min_prev = max(0.0, min(MITAD_MIN, cobertura_min - MITAD_MIN))
    por_localidad = {}
    for loc in localidades:
        if lat.size:
            d = distancia_km(loc["lat"], loc["lon"], lat, lon)
        else:
            d = np.zeros(0)
        c_rec = int(np.sum((d <= CERCA_KM) & reciente))
        c_prev = int(np.sum((d <= CERCA_KM) & previa))
        sobre = int(np.sum((d <= SOBRE_KM) & reciente))
        cercano = float(d[reciente].min()) if np.any(reciente) else None
        por_localidad[loc["id"]] = {
            "tasa_30km_por_min": round(c_rec / min_rec, 2) if min_rec > 0 else None,
            "tasa_30km_previa_por_min": round(c_prev / min_prev, 2) if min_prev >= 5 else None,
            "rayos_30km_10min": c_rec,
            "rayos_15km_10min": sobre,
            "distancia_km": round(cercano, 1) if cercano is not None else None,
        }
    bb = por_localidad.get("bahia_blanca", {})
    return {
        "disponible": cobertura_min >= 5,
        "cobertura_min": round(cobertura_min, 1),
        "rayos_dominio_10min": int(np.sum(reciente)),
        "rayos_dominio_previos_10min": int(np.sum(previa)),
        "tasa_30km_por_min": bb.get("tasa_30km_por_min"),
        "tasa_30km_previa_por_min": bb.get("tasa_30km_previa_por_min"),
        "por_localidad": por_localidad,
    }


def _archivos(ahora: datetime) -> list[dict]:
    prefijo, _ = PRODUCT_SPECS["glm"]
    vistos = {}
    for horas in (0, 1):
        for obj in _list_candidate_objects(prefijo, ahora - timedelta(hours=horas)):
            meta = _enrich_object_metadata(obj)
            if meta:
                vistos[meta["key"]] = meta
    fin = max((_parse_utc(m["end_time_utc"]) for m in vistos.values()), default=None)
    if fin is None:
        return []
    desde = fin - timedelta(minutes=VENTANA_MIN)
    elegidos = [m for m in vistos.values() if _parse_utc(m["end_time_utc"]) > desde]
    return sorted(elegidos, key=lambda m: m["end_time_utc"])[-MAX_ARCHIVOS:]


def _leer(meta: dict, carpeta: Path, centro: tuple[float, float], radio_km: float):
    import requests
    import xarray as xr

    from .s3_client import S3_ENDPOINT

    url = meta.get("url") or f"{S3_ENDPOINT}/{meta['key']}"
    destino = carpeta / Path(meta["key"]).name
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    destino.write_bytes(resp.content)
    with xr.open_dataset(destino, engine="netcdf4", mask_and_scale=True) as ds:
        lat = np.asarray(ds["flash_lat"].values, dtype=float)
        lon = np.asarray(ds["flash_lon"].values, dtype=float)
    destino.unlink(missing_ok=True)
    ok = np.isfinite(lat) & np.isfinite(lon)
    lat, lon = lat[ok], lon[ok]
    if lat.size:
        dentro = distancia_km(centro[0], centro[1], lat, lon) <= radio_km
        lat, lon = lat[dentro], lon[dentro]
    return _parse_utc(meta["end_time_utc"]), lat, lon


def ventana(ahora: datetime | None = None, centro=(-38.7342, -62.1654), radio_km: float = 240.0) -> dict[str, Any]:
    ahora = ahora or datetime.now(timezone.utc)
    archivos = _archivos(ahora)
    if not archivos:
        return {"disponible": False, "motivo": "SIN_ARCHIVOS_GLM"}
    with tempfile.TemporaryDirectory(prefix="climaar_glm_") as tmp:
        with ThreadPoolExecutor(max_workers=HILOS) as pool:
            res = []
            for fut in [pool.submit(_leer, m, Path(tmp), centro, radio_km) for m in archivos]:
                try:
                    res.append(fut.result())
                except Exception:
                    continue
    if not res:
        return {"disponible": False, "motivo": "NO_SE_PUDO_LEER_GLM"}
    fin = max(t for t, _, _ in res)
    inicio = min(t for t, _, _ in res) - timedelta(seconds=20)
    lat = np.concatenate([a for _, a, _ in res]) if res else np.zeros(0)
    lon = np.concatenate([b for _, _, b in res]) if res else np.zeros(0)
    edades = np.concatenate([np.full(a.size, (fin - t).total_seconds() / 60.0) for t, a, _ in res])
    # Cobertura real: si faltan archivos, la tasa se calcula sobre lo leído.
    cobertura = min(VENTANA_MIN, len(res) / 3.0, (fin - inicio).total_seconds() / 60.0)
    salida = resumir(lat, lon, edades, cargar_localidades(), cobertura)
    salida.update({
        "fin_ventana_utc": fin.isoformat().replace("+00:00", "Z"),
        "archivos_leidos": len(res),
        "archivos_esperados": len(archivos),
    })
    return salida
