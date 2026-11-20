#!/usr/bin/env python3
"""Baja el historial de GOES-19 (rayos GLM y canal 13) de un día, solo en las horas con
lluvia en el radar guardado y solo dentro de los 240 km. Para calibrar el corrector
con rayos y satélite (herramientas/simular_huecos.py). Corre en GitHub (necesita red).

Uso: python herramientas/historial_goes.py 20261113 salida_dir
Salida: goes_<dia>.npz con
  c13_inicio (str), c13_valores (float16, barridos x píxeles), lat, lon (píxeles del dominio),
  glm_fin (str por rayo), glm_lat, glm_lon.
"""
from __future__ import annotations

import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import nowcast_flujo_optico as nf  # noqa: E402
from goes19 import glm_ventana as gv  # noqa: E402
from goes19.goes19_engine import PRODUCT_SPECS, _enrich_object_metadata, _list_candidate_objects  # noqa: E402
from goes19.state import leer_dominio  # noqa: E402


def horas_con_lluvia(dia: str) -> list[datetime]:
    frames = sorted(nf.HISTORICO.glob(f"{dia}*.png"))
    if not frames:
        return []
    clutter = nf.cargar_mascara_clutter(nf.cargar_dbz(frames[0]).shape)
    horas = set()
    for p in frames:
        d = nf.cargar_dbz(p)
        dt = nf.timestamp_de_nombre(p)
        if d is None or dt is None:
            continue
        if int((nf.campo_limpio(d, clutter) >= nf.UMBRAL_ECO_DBZ).sum()) >= 300:
            base = dt.replace(minute=0, second=0, microsecond=0)
            for h in range(-1, 3):  # 1 h antes (tendencia) y 2 h después (huecos de hasta 90 min)
                horas.add(base + timedelta(hours=h))
    return sorted(horas)


def main() -> int:
    dia, salida = sys.argv[1], Path(sys.argv[2])
    salida.mkdir(parents=True, exist_ok=True)
    horas = horas_con_lluvia(dia)
    print(f"{dia}: {len(horas)} horas con lluvia (con margen)")
    c13_inicio, c13_val, lat, lon = [], [], None, None
    glm_fin, glm_lat, glm_lon = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, hora in enumerate(horas):
            # Canal 13: todos los barridos de la hora, leyendo solo la zona.
            prefijo, filtro = PRODUCT_SPECS["abi_c13"]
            metas = [m for m in (_enrich_object_metadata(o) for o in _list_candidate_objects(prefijo, hora)) if m]
            metas = [m for m in metas if filtro in m["key"]]
            for m in sorted(metas, key=lambda x: x["start_time_utc"]):
                try:
                    dom, _ = leer_dominio(m, ["CMI"], Path(tmp), "c13")
                except Exception as exc:
                    print("  c13 error", m["key"][-40:], exc)
                    continue
                v = np.asarray(dom["datos"]["CMI"], dtype=np.float32) - 273.15
                if lat is None:
                    lat, lon = np.asarray(dom["latitude"], np.float32), np.asarray(dom["longitude"], np.float32)
                if v.shape != lat.shape:
                    continue
                c13_inicio.append(m["start_time_utc"])
                c13_val.append(v.astype(np.float16))
            # Rayos GLM de la hora (archivos de 20 s).
            vistos = gv._listar([hora])
            archivos = gv._elegir(vistos, hora, hora + timedelta(hours=1))
            for r in gv._bajar_y_leer(archivos, (-38.7342, -62.1654), 240.0):
                n = r["lat"].size
                if n:
                    glm_fin += [r["fin"].isoformat()] * n
                    glm_lat.append(r["lat"].astype(np.float32))
                    glm_lon.append(r["lon"].astype(np.float32))
            print(f"  {hora:%H} UTC: c13 {len(c13_inicio)} barridos, rayos {len(glm_fin)}", flush=True)
    np.savez_compressed(
        salida / f"goes_{dia}.npz",
        c13_inicio=np.array(c13_inicio), c13_valores=np.array(c13_val) if c13_val else np.zeros((0, 0), np.float16),
        lat=lat if lat is not None else np.zeros(0), lon=lon if lon is not None else np.zeros(0),
        glm_fin=np.array(glm_fin), glm_lat=np.concatenate(glm_lat) if glm_lat else np.zeros(0, np.float32),
        glm_lon=np.concatenate(glm_lon) if glm_lon else np.zeros(0, np.float32),
    )
    print("listo", (salida / f"goes_{dia}.npz").stat().st_size / 1e6, "MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
