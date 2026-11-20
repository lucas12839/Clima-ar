#!/usr/bin/env python3
"""Punto 8: qué traen los productos de AMBIENTE de GOES-19 sobre los 240 km.

Uso: python herramientas/sondear_ambiente_goes.py
Para DSIF (inestabilidad), TPWF (agua precipitable), LVTPF y LVMPF (perfiles):
archivos por hora, atraso de publicación, tamaño, variables con dimensiones y
unidades, niveles de presión, y cuántos píxeles con dato hay a <= 240 km del
radar (solo hay dato con cielo despejado). Lee solo la ventana (lectura parcial).
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from goes19 import goes19_engine as ge  # noqa: E402
from goes19 import geospatial as geo  # noqa: E402

PRODUCTOS = {
    "dsi": "ABI-L2-DSIF",
    "tpw": "ABI-L2-TPWF",
    "lvt": "ABI-L2-LVTPF",
    "lvm": "ABI-L2-LVMPF",
}


def main() -> int:
    ahora = datetime.now(timezone.utc)
    for nombre, prefijo in PRODUCTOS.items():
        ge.PRODUCT_SPECS[nombre] = (prefijo, None)
        print(f"\n===== {nombre}: {prefijo}")
        # archivos de las últimas 3 horas
        claves = []
        for h in range(3):
            t = ahora - timedelta(hours=h)
            try:
                claves += [o["key"] for o in ge._list_candidate_objects(prefijo, t)]
            except Exception as exc:
                print("  error listando", exc)
        fines = sorted({ge._enrich_object_metadata({"key": k, "size": 0, "last_modified": ""}) ["end_time_utc"]
                        for k in claves if ge._enrich_object_metadata({"key": k, "size": 0, "last_modified": ""})})
        print(f"  archivos en 3 h: {len(claves)}; fines: {[f[11:16] for f in fines[-8:]]}")
        meta = ge.latest_product_object(nombre, now=ahora, hours_back=3)
        if not meta:
            print("  SIN ARCHIVOS")
            continue
        fin = ge._parse_utc(meta["end_time_utc"])
        print(f"  último: {meta['key'].split('/')[-1]}  {meta.get('size')} bytes  "
              f"atraso {(ahora - fin).total_seconds() / 60:.1f} min")
        t0 = time.time()
        try:
            with geo.abrir_goes(meta["url"]) as ds:
                print("  dims:", dict(ds.sizes))
                for v in ds.data_vars:
                    var = ds[v]
                    if var.ndim >= 1 and var.size > 1 or v in ("pressure",):
                        print(f"   var {v} {var.dims} unidades={var.attrs.get('units')} "
                              f"nombre={str(var.attrs.get('long_name'))[:70]}")
                for v in ds.data_vars:
                    if v.startswith("DQF"):
                        print(f"   {v}: valores={ds[v].attrs.get('flag_values')} "
                              f"significado={ds[v].attrs.get('flag_meanings')}")
                for c in ds.coords:
                    if ds[c].ndim == 1 and ds[c].size < 200 and c not in ("x", "y"):
                        print(f"   coord {c}: {np.asarray(ds[c].values)[:5]}...{np.asarray(ds[c].values)[-3:]} "
                              f"unidades={ds[c].attrs.get('units')}")
                lam, req, rpol, h = geo.get_projection(ds)
                xs, ys = geo.select_spatial_window(ds, radio_km=240.0)
                xv = np.asarray(ds["x"].values[xs]); yv = np.asarray(ds["y"].values[ys])
                xg, yg = np.meshgrid(xv, yv)
                lat, lon = geo.fixed_grid_to_latlon(xg, yg, lambda_0=lam, r_eq=req, r_pol=rpol, h=h)
                dist = geo.haversine_km(lat, lon, geo.RADAR_LAT, geo.RADAR_LON)
                mask = np.isfinite(dist) & (dist <= 240)
                print(f"  ventana {yv.size}x{xv.size}, píxeles a <=240 km: {int(mask.sum())}")
                for v in ds.data_vars:
                    var = ds[v]
                    if "y" not in var.dims or "x" not in var.dims:
                        continue
                    datos = np.asarray(var.isel(y=ys, x=xs).values, dtype=float)
                    if datos.ndim == 2:
                        d = datos[mask]
                        ok = np.isfinite(d)
                        txt = (f"min {np.nanmin(d):.2f} p50 {np.nanmedian(d):.2f} max {np.nanmax(d):.2f}"
                               if ok.any() else "sin dato")
                        print(f"   {v}: {int(ok.sum())}/{d.size} con dato; {txt}")
                        if v in ("CAPE", "TPW"):
                            dqf = np.asarray(ds["DQF_Overall"].isel(y=ys, x=xs).values)[mask]
                            for q in np.unique(dqf):
                                sel = (dqf == q) & ok
                                if sel.any():
                                    print(f"     DQF {q}: {int(sel.sum())} px, p50 {np.nanmedian(d[sel]):.1f}")
                    elif datos.ndim == 3:
                        orden = [var.dims.index("y"), var.dims.index("x"), var.dims.index("pressure")]
                        cubo = np.transpose(datos, orden)  # (y, x, nivel)
                        d = cubo[mask]  # (columnas, nivel)
                        ok = np.isfinite(d)
                        print(f"   {v} 3D: columnas con dato {int(ok.any(axis=1).sum())}/{d.shape[0]}")
                        pres = np.asarray(ds["pressure"].values, dtype=float)
                        cols = np.where(ok.any(axis=1))[0]
                        for nivel in (925, 850, 700, 500, 300):
                            k = int(np.argmin(np.abs(pres - nivel)))
                            col = d[:, k]
                            if np.isfinite(col).any():
                                print(f"     {pres[k]:.0f} hPa: p10 {np.nanpercentile(col, 10):.1f} "
                                      f"p50 {np.nanmedian(col):.1f} p90 {np.nanpercentile(col, 90):.1f}")
                        if cols.size:
                            print(f"     columna ejemplo (cada 10 niveles): {np.round(d[cols[0], ::10], 1)}")
                        dqf = np.asarray(ds["DQF_Overall"].isel(y=ys, x=xs).values)[mask]
                        print(f"     DQF_Overall en columnas con dato: {np.unique(dqf[ok.any(axis=1)], return_counts=True)}")
        except Exception as exc:
            print(f"  ERROR leyendo: {type(exc).__name__}: {exc}")
        print(f"  lectura {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
