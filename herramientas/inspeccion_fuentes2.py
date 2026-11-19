#!/usr/bin/env python3
"""Segunda inspección: WRF SMN (us-west-2), lista completa de productos GOES-19 y lectura parcial."""
import sys, time, tempfile, re
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, requests, xarray as xr
from herramientas.inspeccion_fuentes import listar, titulo, ultimo_goes, LAT, LON

def goes_raiz():
    titulo("A. Todos los productos del bucket noaa-goes19")
    _, pref, _ = listar("noaa-goes19", "", delim="/")
    print(pref)
    from datetime import datetime, timezone
    for anio in ("2026", "2025"):
        _, dias, _ = listar("noaa-goes19", f"ABI-L2-RRQPEF/{anio}/", delim="/")
        print(f"RRQPEF {anio}: {len(dias)} días; últimos {dias[-3:]}")

def wrf():
    titulo("B. WRF del SMN (smn-ar-wrf, us-west-2)")
    ahora = datetime.now(timezone.utc)
    for dias in range(0, 2):
        t = ahora - timedelta(days=dias)
        for ciclo in ("18", "12", "06", "00"):
            pre = f"DATA/WRF/DET/{t:%Y/%m/%d}/{ciclo}/"
            objs, sub, _ = listar("smn-ar-wrf", pre, region="us-west-2")
            if not objs:
                continue
            tipos = {}
            for o in objs:
                m = re.search(r"WRFDETAR_(\w+?)_", o["key"])
                k = m[1] if m else "?"
                tipos.setdefault(k, []).append(o)
            print(f"{pre}: {len(objs)} archivos")
            for k, v in tipos.items():
                v = sorted(v, key=lambda x: x["key"])
                print(f"  {k}: {len(v)} archivos, {v[0]['size']/1e6:.0f} MB c/u, primero {v[0]['mod']} último {v[-1]['mod']}")
                print(f"     ej: {v[0]['key']}")
            for k in ("10M", "01H"):
                if k not in tipos:
                    continue
                o = sorted(tipos[k], key=lambda x: x["key"])[min(6, len(tipos[k]) - 1)]
                url = f"https://smn-ar-wrf.s3.us-west-2.amazonaws.com/{o['key']}"
                with tempfile.TemporaryDirectory() as tmp:
                    p = Path(tmp) / "w.nc"
                    t0 = time.time()
                    p.write_bytes(requests.get(url, timeout=300).content)
                    print(f"  {k} bajado en {time.time()-t0:.0f} s ({p.stat().st_size/1e6:.0f} MB)")
                    with xr.open_dataset(p) as ds:
                        for v in ds.data_vars:
                            print(f"     {v}: {ds[v].attrs.get('description', ds[v].attrs.get('long_name',''))} [{ds[v].attrs.get('units','')}] dims {ds[v].dims}")
            return

def lectura_parcial():
    titulo("C. ¿Se puede leer solo la zona de Bahía sin bajar el disco entero?")
    try:
        import fsspec, h5netcdf  # noqa
    except ImportError:
        print("sin fsspec/h5netcdf"); return
    o = ultimo_goes("ABI-L2-CMIPF", "M6C13")
    t0 = time.time()
    fs = fsspec.filesystem("http", block_size=2**20)
    with fs.open(o["url"], "rb", cache_type="blockcache") as f:
        with xr.open_dataset(f, engine="h5netcdf") as ds:
            cmi = ds["CMI"]
            print("chunks:", ds["CMI"].encoding.get("chunksizes"), "forma", cmi.shape)
            sub = cmi[3500:3900, 2700:3100].values
            print(f"subzona 400x400 leída: min {np.nanmin(sub):.1f} K; {time.time()-t0:.1f} s")
        try:
            print("bytes pedidos aprox:", len(f.cache.cache) * 2**20 / 1e6, "MB")
        except Exception as e:
            print("cache:", e)
    print(f"tamaño completo {o['size']/1e6:.1f} MB")

if __name__ == "__main__":
    for paso in (goes_raiz, lectura_parcial):
        try:
            paso()
        except Exception as e:
            print("ERROR", paso.__name__, e)
