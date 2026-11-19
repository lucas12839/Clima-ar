#!/usr/bin/env python3
"""Inspección de fuentes: qué hay disponible de verdad sobre Bahía Blanca.

Solo diagnóstico (no escribe en data/). Imprime en pantalla:
1. GOES-19: cada producto L2 (último archivo, tamaño, atraso, variables y
   valores a <= 240 km del radar).
2. GOES-19 mesoescala (1 min): dónde estuvieron los sectores M1/M2.
3. GLM: variables de un archivo (área y energía de rayos, grupos).
4. WRF del SMN en AWS: estructura del bucket y último pronóstico.
5. Open-Meteo: modelos y variables convectivas para Bahía Blanca, ensambles.
6. Página de radares SINARAME: qué productos e imágenes publica.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import requests  # noqa: E402
import xarray as xr  # noqa: E402

LAT, LON = -38.7342, -62.1654
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def titulo(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


def listar(bucket, prefijo, region="us-east-1", maximo=1000, delim=None):
    url = f"https://{bucket}.s3.{region}.amazonaws.com"
    params = {"list-type": "2", "prefix": prefijo, "max-keys": maximo}
    if delim:
        params["delimiter"] = delim
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    objs = [{"key": c.findtext("s3:Key", namespaces=NS), "size": int(c.findtext("s3:Size", namespaces=NS) or 0),
             "mod": c.findtext("s3:LastModified", namespaces=NS)} for c in root.findall("s3:Contents", NS)]
    pref = [p.findtext("s3:Prefix", namespaces=NS) for p in root.findall("s3:CommonPrefixes", NS)]
    return objs, pref, url


def dist(lat, lon):
    r = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, (LAT, LON, lat, lon))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def ultimo_goes(prefijo, filtro=None, horas=3):
    ahora = datetime.now(timezone.utc)
    for h in range(horas):
        t = ahora - timedelta(hours=h)
        objs, _, url = listar("noaa-goes19", f"{prefijo}/{t:%Y/%j/%H}/")
        if filtro:
            objs = [o for o in objs if filtro in o["key"]]
        if objs:
            o = max(objs, key=lambda x: x["key"])
            o["url"] = f"{url}/{o['key']}"
            return o
    return None


def goes_productos():
    titulo("1. GOES-19: productos L2 de disco completo (cada 10 min)")
    from goes19.geospatial import prepare_goes_domain

    productos = {
        "ABI-L2-CMIPF": ["M6C02", "M6C07", "M6C08", "M6C09", "M6C10", "M6C11", "M6C13", "M6C14", "M6C15"],
        "ABI-L2-ACHAF": [None], "ABI-L2-ACHTF": [None], "ABI-L2-CTPF": [None], "ABI-L2-ACTPF": [None],
        "ABI-L2-ACMF": [None], "ABI-L2-CODF": [None], "ABI-L2-CPSF": [None], "ABI-L2-DSIF": [None],
        "ABI-L2-TPWF": [None], "ABI-L2-LVTPF": [None], "ABI-L2-LVMPF": [None], "ABI-L2-RRQPEF": [None],
        "ABI-L2-LSTF": [None], "ABI-L2-DMWF": ["M6C14"], "ABI-L2-DMWVF": [None], "ABI-L2-ADPF": [None],
        "ABI-L2-FDCF": [None], "GLM-L2-LCFA": [None],
    }
    leer = {"ABI-L2-DSIF": ["CAPE", "LI", "KI", "TT", "SI"], "ABI-L2-TPWF": ["TPW"], "ABI-L2-RRQPEF": ["RRQPE"],
            "ABI-L2-ACTPF": ["Phase"], "ABI-L2-CPSF": ["PSD"], "ABI-L2-CODF": ["COD"], "ABI-L2-ACMF": ["BCM"],
            "ABI-L2-LVTPF": ["LVT"], "ABI-L2-LVMPF": ["LVM"]}
    ahora = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory() as tmp:
        for prefijo, filtros in productos.items():
            for f in filtros:
                nombre = prefijo + (f" {f}" if f else "")
                try:
                    o = ultimo_goes(prefijo, f)
                except Exception as e:
                    print(f"{nombre}: ERROR listando {e}")
                    continue
                if not o:
                    print(f"{nombre}: SIN ARCHIVOS en 3 h")
                    continue
                m = re.search(r"_e(\d{4})(\d{3})(\d{2})(\d{2})", o["key"])
                fin = datetime(int(m[1]), 1, 1, int(m[3]), int(m[4]), tzinfo=timezone.utc) + timedelta(days=int(m[2]) - 1)
                atraso = (ahora - fin).total_seconds() / 60
                print(f"{nombre}: {o['size']/1e6:.1f} MB, termina {fin:%H:%M} UTC, atraso {atraso:.0f} min")
                if prefijo not in leer:
                    continue
                try:
                    p = Path(tmp) / "x.nc"
                    p.write_bytes(requests.get(o["url"], timeout=180).content)
                    with xr.open_dataset(p) as ds:
                        vs = [v for v in ds.data_vars if ds[v].ndim >= 2]
                        print(f"   variables 2D: {vs[:25]}")
                        if "pressure" in ds.dims or any("pres" in d.lower() for d in ds.dims):
                            print(f"   dims: {dict(ds.sizes)}")
                    for var in leer[prefijo]:
                        try:
                            d = prepare_goes_domain(p, variable=var)
                            v = np.asarray(d["data"], dtype=float)
                            ok = np.isfinite(v)
                            txt = (f"min {np.nanmin(v):.1f} p50 {np.nanmedian(v):.1f} max {np.nanmax(v):.1f}"
                                   if ok.any() else "todo vacío")
                            print(f"   {var}: {ok.sum()}/{v.size} píxeles con dato a ≤240 km; {txt}")
                        except Exception as e:
                            print(f"   {var}: no se pudo leer ({e})")
                    p.unlink(missing_ok=True)
                except Exception as e:
                    print(f"   ERROR leyendo: {e}")


def goes_mesoescala():
    titulo("2. GOES-19 mesoescala M1/M2 (imagen cada 1 min): ¿cubrió Bahía Blanca?")
    ahora = datetime.now(timezone.utc)
    visto = {}
    with tempfile.TemporaryDirectory() as tmp:
        for h in range(0, 48, 3):
            t = ahora - timedelta(hours=h)
            try:
                objs, _, url = listar("noaa-goes19", f"ABI-L2-CMIPM/{t:%Y/%j/%H}/")
            except Exception as e:
                print(f"error {e}")
                continue
            for sector in ("M1", "M2"):
                cand = [o for o in objs if f"-{sector}-" in o["key"] or f"CMIP{sector}" in o["key"]]
                cand = [o for o in cand if "C13" in o["key"]]
                if not cand:
                    continue
                o = cand[0]
                p = Path(tmp) / "m.nc"
                try:
                    p.write_bytes(requests.get(f"{url}/{o['key']}", timeout=60).content)
                    with xr.open_dataset(p) as ds:
                        ext = ds["geospatial_lat_lon_extent"].attrs
                        latc, lonc = ext.get("geospatial_lat_center"), ext.get("geospatial_lon_center")
                    visto[f"{t:%d/%m %H} UTC {sector}"] = (round(float(latc), 1), round(float(lonc), 1))
                except Exception as e:
                    visto[f"{t:%d/%m %H} {sector}"] = str(e)[:60]
    for k, v in visto.items():
        print(f"{k}: centro {v}")


def goes_glm():
    titulo("3. GLM: variables de un archivo")
    o = ultimo_goes("GLM-L2-LCFA")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "g.nc"
        p.write_bytes(requests.get(o["url"], timeout=60).content)
        with xr.open_dataset(p) as ds:
            print([v for v in ds.data_vars][:60])


def wrf_smn():
    titulo("4. WRF del SMN en AWS (smn-ar-wrf)")
    try:
        _, pref, _ = listar("smn-ar-wrf", "", delim="/")
        print("raíz:", pref)
        for p in pref[:6]:
            _, sub, _ = listar("smn-ar-wrf", p, delim="/")
            print(f"  {p} -> {sub[:12]}")
        ahora = datetime.now(timezone.utc)
        for dias in range(0, 3):
            t = ahora - timedelta(days=dias)
            for base in pref:
                for patron in (f"{base}{t:%Y/%m/%d}/", f"{base}{t:%Y/%m/%d}/00/", f"{base}{t:%Y/%m/%d}/12/"):
                    objs, sub, _ = listar("smn-ar-wrf", patron, maximo=50, delim="/")
                    if objs or sub:
                        print(f"  {patron}: subcarpetas {sub[:10]}; archivos:")
                        for ob in objs[:12]:
                            print(f"     {ob['key']}  {ob['size']/1e6:.1f} MB  {ob['mod']}")
            break
    except Exception as e:
        print("ERROR:", e)


def open_meteo():
    titulo("5. Open-Meteo: modelos y variables convectivas para Bahía Blanca")
    variables = ["cape", "lifted_index", "convective_inhibition", "freezing_level_height", "lightning_potential",
                 "updraft", "wind_gusts_10m", "precipitation", "showers", "boundary_layer_height"]
    modelos = ["ecmwf_ifs025", "ecmwf_aifs025_single", "icon_global", "gfs_global", "gfs_seamless", "gem_global",
               "ukmo_global_deterministic_10km", "jma_gsm", "meteofrance_arpege_world", "cma_grapes_global",
               "bom_access_global"]
    for m in modelos:
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": LAT, "longitude": LON, "hourly": ",".join(variables), "models": m,
                "forecast_hours": 6, "timezone": "UTC"}, timeout=30)
            j = r.json()
            if "hourly" not in j:
                print(f"{m}: {j.get('reason', j)}")
                continue
            h = j["hourly"]
            con = [v for v in variables if v in h and any(x is not None for x in h[v])]
            print(f"{m}: con dato {con}")
        except Exception as e:
            print(f"{m}: ERROR {e}")
    try:
        r = requests.get("https://ensemble-api.open-meteo.com/v1/ensemble", params={
            "latitude": LAT, "longitude": LON, "hourly": "cape,precipitation,wind_gusts_10m",
            "models": "ecmwf_ifs025,gfs025,icon_global_eps", "forecast_hours": 12}, timeout=40)
        j = r.json()
        h = j.get("hourly", {})
        miembros = [k for k in h if k.startswith("cape")]
        print(f"Ensambles: {len(miembros)} series de CAPE; ejemplo {miembros[:4]} ... razón: {j.get('reason')}")
    except Exception as e:
        print("Ensambles ERROR", e)


def sinarame():
    titulo("6. Página de radares SINARAME: qué publica")
    try:
        r = requests.get("https://radares.hidricosargentina.gob.ar/", timeout=30)
        html = r.text
        print("tamaño", len(html))
        imgs = sorted(set(re.findall(r"[\w/\-\.:]+\.(?:png|gif|jpg|json|kml|nc|h5)", html)))
        print("recursos:", imgs[:80])
        for palabra in ("VRAD", "ZDR", "KDP", "RHOHV", "COLMAX", "ZH", "CMAX", "precip", "acum", "RMA10", "api"):
            print(f"  '{palabra}': {len(re.findall(palabra, html, flags=re.I))} veces")
        scripts = re.findall(r"<script[^>]+src=\"([^\"]+)\"", html)
        print("scripts:", scripts[:20])
        for s in scripts[:8]:
            url = s if s.startswith("http") else "https://radares.hidricosargentina.gob.ar/" + s.lstrip("/")
            try:
                js = requests.get(url, timeout=20).text
                urls = sorted(set(re.findall(r"https?://[\w\.\-/]+", js)))
                prods = sorted(set(re.findall(r"\b(?:VRAD|ZDR|KDP|RHOHV|COLMAX|ZH|TH|CM|VIL|ECHOTOP)\w*", js)))
                print(f"  {s}: urls {urls[:15]} productos {prods[:30]}")
            except Exception as e:
                print(f"  {s}: {e}")
    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    t0 = time.time()
    for paso in (goes_productos, goes_mesoescala, goes_glm, wrf_smn, open_meteo, sinarame):
        try:
            paso()
        except Exception as e:
            print(f"ERROR en {paso.__name__}: {e}")
    print(f"\nTotal {time.time() - t0:.0f} s")
