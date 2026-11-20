#!/usr/bin/env python3
"""Punto 7: modelos dentro de los 240 km, por localidad (en paralelo, no cambia lo que se ve).

1. WRF del SMN (4 km, abierto en AWS `smn-ar-wrf`): lluvia cada 10 min y, cada hora,
   temperatura, humedad, presión, viento y nivel de congelamiento. Solo se leen los
   puntos de las 7 localidades (máximo y media en un cuadrado de ~10 km).
   La corrida se baja UNA vez (cada 6 h aparece una nueva) y se guarda lo de las
   localidades en data/ia/modelos/wrf_localidades.json; cada ciclo solo lee ese archivo.
2. Escenarios de varios modelos (Open-Meteo ensambles: ECMWF 51, GFS 31, ICON 40):
   probabilidad de inestabilidad, lluvia y ráfagas fuertes en las próximas 12 h por
   localidad. Se renueva cada 60 min.

Salida: data/ia/modelos_region.json. Falla segura: nunca corta el ciclo; si una fuente
falla se dice y se sigue con la otra (y con lo guardado de antes).
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

VERSION = "modelos-region-1.0"
LOCALIDADES = ROOT / "config" / "localidades.json"
SALIDA = ROOT / "data" / "ia" / "modelos_region.json"
CACHE_WRF = ROOT / "data" / "ia" / "modelos" / "wrf_localidades.json"
CACHE_ENS = ROOT / "data" / "ia" / "modelos" / "ensambles_localidades.json"
BUCKET = "https://smn-ar-wrf.s3.us-west-2.amazonaws.com"
ENSAMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
MODELOS_ENS = ("ecmwf_ifs025", "gfs025", "icon_seamless")
HORAS_WRF = 18              # horas de pronóstico que se guardan de cada corrida
CAJA_KM = 5.0               # medio lado del cuadrado alrededor de cada localidad
ENS_RENOVAR_MIN = 60.0
TIEMPO_MAX_WRF_S = 200.0    # tope para bajar una corrida nueva (si no alcanza, sigue la próxima vez)
UMBRALES = {"cape_jkg": (500, 1000, 2000), "lluvia_mm_h": (1.0, 5.0, 10.0), "rafaga_kmh": (60.0, 90.0)}
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if dt else None


def parse(texto) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(texto).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def leer_json(path: Path) -> dict:
    try:
        v = json.loads(path.read_text(encoding="utf-8"))
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def escribir_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as h:
            json.dump(data, h, ensure_ascii=False, indent=1, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def localidades() -> list[dict]:
    return list(leer_json(LOCALIDADES).get("localidades") or [])


def redondear(x, n=1):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


# ------------------------------------------------------------
# WRF del SMN
# ------------------------------------------------------------

def listar(prefijo: str) -> list[str]:
    import requests

    r = requests.get(BUCKET, params={"list-type": "2", "prefix": prefijo, "max-keys": 1000}, timeout=30)
    r.raise_for_status()
    raiz = ET.fromstring(r.text)
    return [c.findtext("s3:Key", namespaces=NS) for c in raiz.findall("s3:Contents", NS)]


def ultima_corrida(ahora: datetime) -> tuple[datetime, list[str]] | None:
    """Última corrida con al menos HORAS_WRF horas de los archivos 10M y 01H."""
    for horas in range(0, 30, 6):
        t = ahora - timedelta(hours=horas)
        init = t.replace(hour=(t.hour // 6) * 6, minute=0, second=0, microsecond=0)
        claves = listar(f"DATA/WRF/DET/{init:%Y/%m/%d/%H}/")
        n10 = sum("_10M_" in k for k in claves)
        n01 = sum("_01H_" in k for k in claves)
        if n10 >= HORAS_WRF and n01 >= HORAS_WRF:
            return init, claves
    return None


def indices_localidades(lat: np.ndarray, lon: np.ndarray, locs: list[dict]) -> dict[str, tuple]:
    """Para cada localidad: rebanada (y, x) del cuadrado de ~10 km en la grilla del WRF."""
    salida = {}
    for loc in locs:
        d = (lat - loc["lat"]) ** 2 + ((lon - loc["lon"]) * math.cos(math.radians(loc["lat"]))) ** 2
        iy, ix = np.unravel_index(int(np.argmin(d)), d.shape)
        n = max(1, int(round(CAJA_KM / 4.0)))  # grilla de 4 km
        salida[loc["id"]] = (slice(max(iy - n, 0), iy + n + 1), slice(max(ix - n, 0), ix + n + 1))
    return salida


def bajar(url: str, destino: Path) -> Path:
    import requests

    with requests.get(url, timeout=120, stream=True) as r:
        r.raise_for_status()
        with open(destino, "wb") as h:
            for parte in r.iter_content(1 << 20):
                h.write(parte)
    return destino


def resumir_campo(arr: np.ndarray, sl: tuple) -> tuple[float, float]:
    sub = np.asarray(arr[sl], dtype=float)
    return float(np.nanmax(sub)), float(np.nanmean(sub))


def procesar_corrida(init: datetime, claves: list[str], locs: list[dict], ahora: datetime) -> dict[str, Any]:
    """Baja las horas que interesan de una corrida y guarda solo las localidades."""
    import xarray as xr

    inicio = time.time()
    # Desde la hora actual (menos 1 h) hasta HORAS_WRF horas de pronóstico.
    h0 = max(0, int((ahora - init).total_seconds() // 3600) - 1)
    horas = [h for h in range(h0, h0 + 9)]
    salida = {"init_utc": iso(init), "localidades": {l["id"]: {"lluvia_10min": [], "horario": []} for l in locs},
              "horas_leidas": [], "completa": True}
    idx = None
    with tempfile.TemporaryDirectory() as tmp:
        for h in horas:
            if time.time() - inicio > TIEMPO_MAX_WRF_S:
                salida["completa"] = False
                break
            for tipo in ("10M", "01H"):
                clave = next((k for k in claves if f"_{tipo}_" in k and k.endswith(f"_{h:03d}.nc")), None)
                if clave is None:
                    continue
                p = bajar(f"{BUCKET}/{clave}", Path(tmp) / f"{tipo}_{h}.nc")
                with xr.open_dataset(p) as ds:
                    if idx is None:
                        idx = indices_localidades(np.asarray(ds["lat"]), np.asarray(ds["lon"]), locs)
                    tiempos = [datetime.fromisoformat(str(np.datetime_as_string(t, unit="s"))).replace(tzinfo=timezone.utc)
                               for t in np.asarray(ds["time"]).ravel()]
                    for k, sl in idx.items():
                        loc = salida["localidades"][k]
                        if tipo == "10M":
                            pp = np.asarray(ds["PP"])  # acumulado desde el inicio de la corrida
                            for i, t in enumerate(tiempos):
                                mx, md = resumir_campo(pp[i], sl)
                                loc["lluvia_10min"].append({"utc": iso(t), "acum_max_mm": redondear(mx, 2),
                                                            "acum_media_mm": redondear(md, 2)})
                        else:
                            fila = {"utc": iso(tiempos[0])}
                            for var, nombre, fac in (("T2", "temperatura_c", 1.0), ("HR2", "humedad_pct", 1.0),
                                                     ("PSFC", "presion_hpa", 1.0), ("magViento10", "viento_kmh", 3.6),
                                                     ("Freezing_level", "nivel_congelamiento_m", 1.0)):
                                if var in ds:
                                    _, md = resumir_campo(np.asarray(ds[var])[0], sl)
                                    fila[nombre] = redondear(md * fac, 1)
                            loc["horario"].append(fila)
            salida["horas_leidas"].append(h)
    # Lluvia cada 10 min = diferencia del acumulado.
    for loc in salida["localidades"].values():
        serie = sorted({r["utc"]: r for r in loc["lluvia_10min"]}.values(), key=lambda r: r["utc"])
        prev = None
        for r in serie:
            if prev is not None and r["acum_max_mm"] is not None and prev["acum_max_mm"] is not None:
                r["mm_10min_max"] = round(max(r["acum_max_mm"] - prev["acum_max_mm"], 0.0), 2)
                r["mm_10min_media"] = round(max(r["acum_media_mm"] - prev["acum_media_mm"], 0.0), 2)
            prev = r
        loc["lluvia_10min"] = serie
    salida["segundos"] = round(time.time() - inicio, 1)
    return salida


def wrf(ahora: datetime, locs: list[dict]) -> dict[str, Any]:
    cache = leer_json(CACHE_WRF)
    try:
        ult = ultima_corrida(ahora)
    except Exception as exc:
        ult = None
        cache["error_listado"] = f"{type(exc).__name__}: {str(exc)[:150]}"
    if ult is not None:
        init, claves = ult
        h_actual = int((ahora - init).total_seconds() // 3600)
        leidas = cache.get("horas_leidas") or []
        # Corrida nueva, o la guardada ya no cubre las próximas horas.
        if cache.get("init_utc") != iso(init) or not cache.get("completa", False) or \
                not leidas or max(leidas) < h_actual + 5:
            try:
                cache = procesar_corrida(init, claves, locs, ahora)
                escribir_json(CACHE_WRF, cache)
            except Exception as exc:
                cache["error"] = f"{type(exc).__name__}: {str(exc)[:150]}"
    return cache


def resumen_wrf(cache: dict, ahora: datetime) -> dict[str, Any]:
    """Próximas 3 h por localidad: lluvia total, máximo en 10 min y ambiente horario."""
    if not cache.get("localidades"):
        return {"estado": "sin_datos", "error": cache.get("error") or cache.get("error_listado")}
    fin = ahora + timedelta(hours=3)
    salida = {"estado": "ok", "corrida_utc": cache.get("init_utc"),
              "edad_corrida_h": redondear((ahora - parse(cache["init_utc"])).total_seconds() / 3600.0),
              "localidades": {}}
    for k, loc in cache["localidades"].items():
        lluvia = [r for r in loc.get("lluvia_10min") or [] if ahora <= parse(r["utc"]) <= fin and "mm_10min_max" in r]
        hor = [r for r in loc.get("horario") or [] if ahora - timedelta(minutes=30) <= parse(r["utc"]) <= fin]
        salida["localidades"][k] = {
            "lluvia_3h_media_mm": round(sum(r["mm_10min_media"] for r in lluvia), 1) if lluvia else None,
            "lluvia_10min_max_mm": max((r["mm_10min_max"] for r in lluvia), default=None),
            "hora_lluvia_max_utc": max(lluvia, key=lambda r: r["mm_10min_max"])["utc"] if lluvia and
            max(r["mm_10min_max"] for r in lluvia) > 0 else None,
            "proxima_hora": hor[0] if hor else None,
        }
    if salida["edad_corrida_h"] is not None and salida["edad_corrida_h"] > 18:
        salida["estado"] = "corrida_vieja"
    return salida


# ------------------------------------------------------------
# Ensambles (varios modelos)
# ------------------------------------------------------------

def ensambles(ahora: datetime, locs: list[dict]) -> dict[str, Any]:
    cache = leer_json(CACHE_ENS)
    t = parse(cache.get("bajado_utc"))
    if t is not None and (ahora - t).total_seconds() / 60.0 < ENS_RENOVAR_MIN and cache.get("localidades"):
        return cache
    import requests

    try:
        r = requests.get(ENSAMBLE_URL, params={
            "latitude": ",".join(str(l["lat"]) for l in locs),
            "longitude": ",".join(str(l["lon"]) for l in locs),
            "hourly": "cape,precipitation,wind_gusts_10m", "models": ",".join(MODELOS_ENS),
            "forecast_hours": 12, "timezone": "UTC"}, timeout=40)
        r.raise_for_status()
        datos = r.json()
        datos = datos if isinstance(datos, list) else [datos]
    except Exception as exc:
        cache["error"] = f"{type(exc).__name__}: {str(exc)[:150]}"
        return cache
    salida = {"bajado_utc": iso(ahora), "modelos": list(MODELOS_ENS), "localidades": {}}
    for loc, d in zip(locs, datos):
        h = d.get("hourly") or {}
        tiempos = h.get("time") or []
        item = {"horas": tiempos}
        for var, nombre in (("cape", "cape_jkg"), ("precipitation", "lluvia_mm_h"), ("wind_gusts_10m", "rafaga_kmh")):
            # Control y miembros de cada modelo: cape_<modelo>_ensemble, cape_memberNN_<modelo>_ensemble.
            miembros = [np.asarray([np.nan if x is None else x for x in v], dtype=float)
                        for k, v in h.items() if k.startswith(var + "_") and isinstance(v, list)]
            if not miembros:
                continue
            m = np.vstack(miembros)  # escenarios x horas
            validos = np.isfinite(m)
            item[nombre] = {"escenarios": int(m.shape[0]), "por_umbral": {}}
            for u in UMBRALES[nombre]:
                # Probabilidad de que pase en ALGUNA de las próximas 12 h, y hora por hora.
                alguna = np.any(np.where(validos, m >= u, False), axis=1)
                item[nombre]["por_umbral"][str(u)] = {
                    "prob_12h_pct": round(100.0 * float(alguna.mean()), 0),
                    "prob_por_hora_pct": [round(100.0 * float(x), 0) for x in
                                          np.where(validos, m >= u, False).mean(axis=0)],
                }
            item[nombre]["mediana_max_12h"] = redondear(float(np.nanmedian(np.nanmax(m, axis=1))))
        salida["localidades"][loc["id"]] = item
    escribir_json(CACHE_ENS, salida)
    return salida


def resumen_ensambles(cache: dict, ahora: datetime) -> dict[str, Any]:
    if not cache.get("localidades"):
        return {"estado": "sin_datos", "error": cache.get("error")}
    edad = (ahora - parse(cache["bajado_utc"])).total_seconds() / 60.0
    salida = {"estado": "ok" if edad <= 3 * ENS_RENOVAR_MIN else "viejo", "bajado_utc": cache["bajado_utc"],
              "edad_min": round(edad, 1), "modelos": cache.get("modelos"), "localidades": {}}
    for k, item in cache["localidades"].items():
        salida["localidades"][k] = {
            nombre: {"escenarios": v["escenarios"], "mediana_max_12h": v.get("mediana_max_12h"),
                     **{f"prob_ge_{u}_pct": v["por_umbral"][u]["prob_12h_pct"] for u in v["por_umbral"]}}
            for nombre, v in item.items() if isinstance(v, dict)
        }
    return salida


def ejecutar(ahora: datetime | None = None) -> dict[str, Any]:
    ahora = ahora or datetime.now(timezone.utc)
    locs = localidades()
    salida = {"version": VERSION, "en_paralelo": True, "generado_utc": iso(ahora),
              "alcance": "solo las 7 localidades (dentro de los 240 km)"}
    try:
        salida["wrf_smn"] = resumen_wrf(wrf(ahora, locs), ahora)
    except Exception as exc:
        salida["wrf_smn"] = {"estado": "error", "error": f"{type(exc).__name__}: {str(exc)[:150]}"}
    try:
        salida["ensambles"] = resumen_ensambles(ensambles(ahora, locs), ahora)
    except Exception as exc:
        salida["ensambles"] = {"estado": "error", "error": f"{type(exc).__name__}: {str(exc)[:150]}"}
    return salida


def main(argv=None) -> int:
    ahora = parse(os.environ["CLIMAAR_AHORA_UTC"]) if os.environ.get("CLIMAAR_AHORA_UTC") else None
    try:
        salida = ejecutar(ahora)
    except Exception as exc:  # nunca corta el ciclo
        salida = {"version": VERSION, "en_paralelo": True, "generado_utc": iso(datetime.now(timezone.utc)),
                  "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    escribir_json(SALIDA, salida)
    print(json.dumps({"wrf": (salida.get("wrf_smn") or {}).get("estado"),
                      "corrida": (salida.get("wrf_smn") or {}).get("corrida_utc"),
                      "ensambles": (salida.get("ensambles") or {}).get("estado")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
