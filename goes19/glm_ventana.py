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
MAX_RAYOS_POSICION = 1500
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


def _listar(horas_utc: list[datetime]) -> dict[str, dict]:
    prefijo, _ = PRODUCT_SPECS["glm"]
    vistos: dict[str, dict] = {}
    for cuando in horas_utc:
        for obj in _list_candidate_objects(prefijo, cuando):
            meta = _enrich_object_metadata(obj)
            if meta:
                vistos[meta["key"]] = meta
    return vistos


def _elegir(vistos: dict[str, dict], desde: datetime, hasta: datetime | None = None) -> list[dict]:
    elegidos = [
        m for m in vistos.values()
        if _parse_utc(m["end_time_utc"]) > desde
        and (hasta is None or _parse_utc(m["end_time_utc"]) <= hasta)
    ]
    return sorted(elegidos, key=lambda m: m["end_time_utc"])


def _archivos(ahora: datetime) -> list[dict]:
    vistos = _listar([ahora, ahora - timedelta(hours=1)])
    fin = max((_parse_utc(m["end_time_utc"]) for m in vistos.values()), default=None)
    if fin is None:
        return []
    return _elegir(vistos, fin - timedelta(minutes=VENTANA_MIN))[-MAX_ARCHIVOS:]


def _bajar(meta: dict, carpeta: Path) -> Path:
    """Descarga (en hilos: solo red, sin NetCDF)."""
    import requests

    from .s3_client import S3_ENDPOINT

    url = meta.get("url") or f"{S3_ENDPOINT}/{meta['key']}"
    destino = carpeta / Path(meta["key"]).name
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    destino.write_bytes(resp.content)
    return destino


def _leer(meta: dict, destino: Path, centro: tuple[float, float], radio_km: float) -> dict:
    """Lectura NetCDF de a uno: la librería HDF5 no es segura con hilos.
    Devuelve los rayos del archivo dentro del radio, con su área (km²) y
    energía (fJ): rayos chicos y muchos = corriente ascendente fuerte."""
    import xarray as xr

    with xr.open_dataset(destino, engine="netcdf4", mask_and_scale=True) as ds:
        lat = np.asarray(ds["flash_lat"].values, dtype=float)
        lon = np.asarray(ds["flash_lon"].values, dtype=float)
        area = (np.asarray(ds["flash_area"].values, dtype=float)
                if "flash_area" in ds else np.full(lat.shape, np.nan))
        energia = (np.asarray(ds["flash_energy"].values, dtype=float)
                   if "flash_energy" in ds else np.full(lat.shape, np.nan))
    destino.unlink(missing_ok=True)
    ok = np.isfinite(lat) & np.isfinite(lon)
    lat, lon, area, energia = lat[ok], lon[ok], area[ok], energia[ok]
    if lat.size:
        dentro = distancia_km(centro[0], centro[1], lat, lon) <= radio_km
        lat, lon, area, energia = lat[dentro], lon[dentro], area[dentro], energia[dentro]
    return {
        "fin": _parse_utc(meta["end_time_utc"]),
        "lat": lat,
        "lon": lon,
        "area_km2": area / 1e6,       # GLM da m²
        "energia_fj": energia * 1e15,  # GLM da J
    }


def _bajar_y_leer(archivos: list[dict], centro, radio_km) -> list[tuple]:
    res = []
    with tempfile.TemporaryDirectory(prefix="climaar_glm_") as tmp:
        with ThreadPoolExecutor(max_workers=HILOS) as pool:
            futuros = [(m, pool.submit(_bajar, m, Path(tmp))) for m in archivos]
        for meta, fut in futuros:
            try:
                res.append(_leer(meta, fut.result(), centro, radio_km))
            except Exception:
                continue
    return res


def _resumen_ventana(res: list[dict]) -> dict[str, Any]:
    if not res:
        return {"disponible": False, "motivo": "NO_SE_PUDO_LEER_GLM"}
    fin = max(r["fin"] for r in res)
    inicio = min(r["fin"] for r in res) - timedelta(seconds=20)
    lat = np.concatenate([r["lat"] for r in res])
    lon = np.concatenate([r["lon"] for r in res])
    edades = np.concatenate([np.full(r["lat"].size, (fin - r["fin"]).total_seconds() / 60.0) for r in res])
    # Cobertura real: si faltan archivos, la tasa se calcula sobre lo leído.
    cobertura = min(VENTANA_MIN, len(res) / 3.0, (fin - inicio).total_seconds() / 60.0)
    salida = resumir(lat, lon, edades, cargar_localidades(), cobertura)
    salida.update({
        "fin_ventana_utc": fin.isoformat().replace("+00:00", "Z"),
        "archivos_leidos": len(res),
    })
    # Posición de los rayos de los últimos 10 min (para el corrector del nowcast en capas).
    recientes = edades <= 10.0
    orden = np.argsort(edades[recientes])[:MAX_RAYOS_POSICION]
    # [latitud, longitud, minutos antes del fin de la ventana] (la edad la usa el
    # seguimiento de celdas con rayos, paso 2; los que leen solo [0] y [1] siguen igual).
    salida["rayos_10min"] = [[round(float(a), 3), round(float(b), 3), round(float(c), 2)] for a, b, c in
                             zip(lat[recientes][orden], lon[recientes][orden], edades[recientes][orden])]
    salida["rayos_10min_total"] = int(recientes.sum())
    return salida


def _rayos_radar(res: list[dict], frame: datetime, minutos: float) -> dict[str, Any]:
    """Rayos de los `minutos` que terminan en la imagen de radar, con su edad
    respecto de esa imagen (para atribuirlos a cada tormenta)."""
    desde = frame - timedelta(minutes=minutos)
    elegidos = [r for r in res if desde < r["fin"] <= frame]
    vacio = np.zeros(0)
    if not elegidos:
        return {"cobertura_min": 0.0, "ventana_min": minutos, "minutos": vacio, "latitude": vacio,
                "longitude": vacio, "area_km2": vacio, "energia_fj": vacio, "archivos": 0}
    return {
        "cobertura_min": min(minutos, len(elegidos) / 3.0),
        "ventana_min": minutos,
        "minutos": np.concatenate([np.full(r["lat"].size, (frame - r["fin"]).total_seconds() / 60.0)
                                   for r in elegidos]),
        "latitude": np.concatenate([r["lat"] for r in elegidos]),
        "longitude": np.concatenate([r["lon"] for r in elegidos]),
        "area_km2": np.concatenate([r["area_km2"] for r in elegidos]),
        "energia_fj": np.concatenate([r["energia_fj"] for r in elegidos]),
        "archivos": len(elegidos),
    }


def ventana_y_rayos_radar(ahora: datetime | None = None, centro=(-38.7342, -62.1654), radio_km: float = 240.0,
                          frame_radar: datetime | None = None,
                          minutos_radar: float = 14.0) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Una sola descarga para dos usos:

    1. ventana: últimos 20 min hasta el archivo más reciente (por localidad,
       como siempre: con el radar atrasado los rayos siguen siendo de ahora);
    2. rayos_radar: los 14 min que terminan en la imagen de radar (lo que
       pide el algoritmo de salto de rayos), para atribuir rayos a cada
       tormenta en el mismo instante que el radar.
    """
    ahora = ahora or datetime.now(timezone.utc)
    horas = [ahora, ahora - timedelta(hours=1)]
    if frame_radar is not None:
        horas += [frame_radar, frame_radar - timedelta(hours=1)]
    vistos = _listar(sorted({h.replace(minute=0, second=0, microsecond=0) for h in horas}, reverse=True))
    fin = max((_parse_utc(m["end_time_utc"]) for m in vistos.values()), default=None)
    if fin is None:
        return {"disponible": False, "motivo": "SIN_ARCHIVOS_GLM"}, None
    recientes = _elegir(vistos, fin - timedelta(minutes=VENTANA_MIN))[-MAX_ARCHIVOS:]
    del_radar = []
    if frame_radar is not None:
        del_radar = _elegir(vistos, frame_radar - timedelta(minutes=minutos_radar), frame_radar)
    union = {m["key"]: m for m in recientes + del_radar}
    res = _bajar_y_leer(sorted(union.values(), key=lambda m: m["end_time_utc"]), centro, radio_km)
    fines_recientes = {_parse_utc(m["end_time_utc"]) for m in recientes}
    salida = _resumen_ventana([r for r in res if r["fin"] in fines_recientes])
    if "archivos_leidos" in salida:
        salida["archivos_esperados"] = len(recientes)
    rayos = _rayos_radar(res, frame_radar, minutos_radar) if frame_radar is not None else None
    # Para el mapa de la app: cada rayo leído con su hora (lo guarda state.py
    # en data/goes19/rayos_mapa.json y lo saca del estado antes de guardarlo).
    salida["_posiciones_mapa"] = [
        (float(la), float(lo), int(r["fin"].timestamp()))
        for r in res
        for la, lo in zip(np.asarray(r["lat"]), np.asarray(r["lon"]))
        if np.isfinite(la) and np.isfinite(lo)
    ]
    return salida, rayos


def ventana(ahora: datetime | None = None, centro=(-38.7342, -62.1654), radio_km: float = 240.0) -> dict[str, Any]:
    salida, _ = ventana_y_rayos_radar(ahora, centro, radio_km)
    return salida
