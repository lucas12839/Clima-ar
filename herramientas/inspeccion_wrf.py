#!/usr/bin/env python3
"""Punto 7: inspección del WRF del SMN (smn-ar-wrf) y de los ensambles de Open-Meteo.

Imprime qué archivos hay en la última corrida, sus variables, la grilla y cuánto
tarda leer solo la zona de 240 km. Corre en GitHub (este entorno no tiene red).
"""
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import requests  # noqa: E402
import xarray as xr  # noqa: E402

from herramientas.inspeccion_fuentes import listar  # noqa: E402

RADAR = (-38.7342, -62.1654)


def ultima_corrida():
    ahora = datetime.now(timezone.utc)
    for horas in range(0, 48, 6):
        t = ahora - timedelta(hours=horas)
        ciclo = f"{(t.hour // 6) * 6:02d}"
        pre = f"DATA/WRF/DET/{t:%Y/%m/%d}/{ciclo}/"
        objs, _, _ = listar("smn-ar-wrf", pre, region="us-west-2")
        if objs:
            return pre, objs
    return None, []


def main():
    pre, objs = ultima_corrida()
    print("corrida:", pre, "archivos:", len(objs))
    tipos = {}
    for o in objs:
        m = re.search(r"WRFDETAR_(\w+?)_", o["key"])
        tipos.setdefault(m[1] if m else "?", []).append(o)
    for k, v in tipos.items():
        v = sorted(v, key=lambda x: x["key"])
        print(f"  {k}: {len(v)} archivos, {v[0]['size'] / 1e6:.1f} MB; {v[0]['key']} ... {v[-1]['key']} (mod {v[-1]['mod']})")
    for k in ("10M", "01H"):
        if k not in tipos:
            continue
        o = sorted(tipos[k], key=lambda x: x["key"])[3]
        url = f"https://smn-ar-wrf.s3.us-west-2.amazonaws.com/{o['key']}"
        t0 = time.time()
        datos = requests.get(url, timeout=300).content
        print(f"\n{k}: {o['key']} bajado en {time.time() - t0:.1f} s ({len(datos) / 1e6:.1f} MB)")
        p = Path(f"/tmp/wrf_{k}.nc")
        p.write_bytes(datos)
        with xr.open_dataset(p) as ds:
            print("  dims:", dict(ds.sizes))
            print("  coords:", list(ds.coords))
            for v in ds.data_vars:
                a = ds[v]
                print(f"  {v}: {a.attrs.get('description', a.attrs.get('long_name', ''))} [{a.attrs.get('units', '')}] {a.dims} {a.dtype}")
            for c in ("lat", "lon", "XLAT", "XLONG", "latitude", "longitude"):
                if c in ds:
                    arr = np.asarray(ds[c])
                    print(f"  {c}: forma {arr.shape} min {np.nanmin(arr):.3f} max {np.nanmax(arr):.3f}")
            print("  atributos:", {a: str(ds.attrs[a])[:80] for a in list(ds.attrs)[:25]})
            if "Time" in ds or "time" in ds:
                print("  tiempo:", np.asarray(ds.get("Time", ds.get("time"))).ravel()[:5])

    t0 = time.time()
    lats = [-38.7183, -38.88, -38.10, -38.8256, -38.1378]
    lons = [-62.2663, -62.075, -62.2233, -62.6958, -61.7936]
    r = requests.get("https://ensemble-api.open-meteo.com/v1/ensemble", params={
        "latitude": ",".join(map(str, lats)), "longitude": ",".join(map(str, lons)),
        "hourly": "cape,precipitation,wind_gusts_10m", "models": "ecmwf_ifs025,gfs025,icon_seamless",
        "forecast_hours": 12, "timezone": "UTC"}, timeout=60)
    print(f"\nensambles Open-Meteo: HTTP {r.status_code} en {time.time() - t0:.1f} s, {len(r.content) / 1e3:.0f} kB")
    j = r.json()
    j = j if isinstance(j, list) else [j]
    h = j[0].get("hourly", {})
    claves = list(h)
    print("  claves:", len(claves), claves[:6], "...")
    print("  tiempos:", h.get("time", [])[:3])


if __name__ == "__main__":
    main()
