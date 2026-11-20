#!/usr/bin/env python3
"""Punto 7: baja el historial del WRF del SMN de un día (corridas 00, 06, 12 y 18 UTC) y guarda
solo la lluvia acumulada de las 7 localidades, para medir si el WRF acierta la lluvia por
localidad comparándolo con el radar guardado (herramientas/medir_wrf.py).
Corre en GitHub (necesita red hacia AWS `smn-ar-wrf`).

Uso: python herramientas/historial_wrf.py 20261113 salida_dir
Salida: wrf_<dia>.json  {corrida_utc: {loc: [[utc, acum_max_mm, acum_media_mm], ...]}}
(acumulado desde el inicio de la corrida, cada 10 min, plazos 0 a HORAS h).
"""
from __future__ import annotations

import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import modelos_region as mr  # noqa: E402

HORAS = 12


def una_corrida(init: datetime, locs: list[dict]) -> dict | None:
    import xarray as xr

    claves = mr.listar(f"DATA/WRF/DET/{init:%Y/%m/%d/%H}/")
    archivos = {h: next((k for k in claves if "_10M_" in k and k.endswith(f"_{h:03d}.nc")), None)
                for h in range(0, HORAS + 1)}
    archivos = {h: k for h, k in archivos.items() if k}
    if not archivos:
        return None
    salida = {l["id"]: [] for l in locs}
    idx = None
    with tempfile.TemporaryDirectory() as tmp:
        def bajar(item):
            h, k = item
            return h, mr.bajar(f"{mr.BUCKET}/{k}", Path(tmp) / f"{h}.nc")

        with ThreadPoolExecutor(4) as ex:
            rutas = dict(ex.map(bajar, archivos.items()))
        for h in sorted(rutas):
            with xr.open_dataset(rutas[h]) as ds:
                if idx is None:
                    idx = mr.indices_localidades(np.asarray(ds["lat"]), np.asarray(ds["lon"]), locs)
                pp = np.asarray(ds["PP"])
                tiempos = [str(np.datetime_as_string(t, unit="s")) + "Z" for t in np.asarray(ds["time"]).ravel()]
                for k, sl in idx.items():
                    for i, t in enumerate(tiempos):
                        mx, md = mr.resumir_campo(pp[i], sl)
                        salida[k].append([t, round(mx, 2), round(md, 2)])
            rutas[h].unlink()
    for k in salida:
        salida[k] = sorted({r[0]: r for r in salida[k]}.values())
    return salida


def main() -> int:
    dia, carpeta = sys.argv[1], Path(sys.argv[2])
    carpeta.mkdir(parents=True, exist_ok=True)
    locs = mr.localidades()
    base = datetime.strptime(dia, "%Y%m%d").replace(tzinfo=timezone.utc)
    res = {}
    for hh in (0, 6, 12, 18):
        init = base + timedelta(hours=hh)
        try:
            r = una_corrida(init, locs)
        except Exception as exc:  # una corrida que falta no corta el día
            print(f"{init:%Y-%m-%d %H} UTC: {type(exc).__name__}: {exc}")
            r = None
        if r:
            res[mr.iso(init)] = r
            print(f"{init:%Y-%m-%d %H} UTC: {len(next(iter(r.values())))} tiempos")
    (carpeta / f"wrf_{dia}.json").write_text(json.dumps(res, separators=(",", ":")) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
