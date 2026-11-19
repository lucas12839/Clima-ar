#!/usr/bin/env python3
"""Diagnóstico de red de GOES-19: qué productos hay y qué variables traen.

Uso: python herramientas/sondear_goes.py
Baja el archivo más reciente de cada producto, lista sus variables y muestra
valores cerca de Bahía Blanca. También corre la ventana de rayos GLM.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import requests  # noqa: E402
import xarray as xr  # noqa: E402

from goes19 import goes19_engine as ge  # noqa: E402
from goes19.geospatial import prepare_goes_domain  # noqa: E402

PRODUCTOS = {
    "rrqpe": ("ABI-L2-RRQPEF", None, "RRQPE"),
    "dsi": ("ABI-L2-DSIF", None, "CAPE"),
    "tpw": ("ABI-L2-TPWF", None, "TPW"),
    "acht": ("ABI-L2-ACHTF", None, "TEMP"),
    "c13": ("ABI-L2-CMIPF", "M6C13", "CMI"),
    "acm": ("ABI-L2-ACMF", None, "BCM"),
}


def main() -> int:
    ahora = datetime.now(timezone.utc)
    for nombre, (prefijo, filtro, var) in PRODUCTOS.items():
        ge.PRODUCT_SPECS[nombre] = (prefijo, filtro)
        t0 = time.time()
        try:
            meta = ge.latest_product_object(nombre, now=ahora, hours_back=3)
            if not meta:
                print(f"{nombre}: SIN ARCHIVOS")
                continue
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "x.nc"
                p.write_bytes(requests.get(meta["url"], timeout=120).content)
                with xr.open_dataset(p, engine="netcdf4") as ds:
                    print(f"{nombre}: {meta['key']} fin {meta['end_time_utc']} {meta.get('size')} bytes")
                    print("   variables:", sorted(ds.data_vars)[:40])
                try:
                    dom = prepare_goes_domain(p, variable=var)
                    datos = np.asarray(dom["data"], dtype=float)
                    ok = datos[np.isfinite(datos)]
                    print(f"   {var} en 240 km: n={ok.size} min={ok.min() if ok.size else None} "
                          f"max={ok.max() if ok.size else None} claves={list(dom)}")
                except Exception as exc:
                    print(f"   {var}: no se pudo recortar: {exc}")
        except Exception as exc:
            print(f"{nombre}: ERROR {exc}")
        print(f"   ({time.time() - t0:.1f} s)")
    from goes19.glm_ventana import ventana
    t0 = time.time()
    print("GLM ventana:", json.dumps(ventana(), ensure_ascii=False)[:1500], f"({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
