#!/usr/bin/env python3
"""Calibra el pronóstico de llegada por localidad con los frames reales.

Para cada frame T del histórico (con un frame previo a 5-30 min) se corre el
mismo flujo óptico de producción y se anota, para cada localidad, si
pronosticó la llegada de un eco >=35 dBZ en <= 60 min. Después se compara con
lo que el radar vio de verdad:

- llegó:        hubo eco >=35 dBZ sobre la localidad entre T y T+eta+20 min.
- se desvió:    no llegó, pero hubo eco >=35 dBZ a <= 20 km (pasó al costado).
- se deshizo:   no llegó y tampoco quedó eco >=35 dBZ a <= 20 km.

Además cuenta las llegadas que no se pronosticaron (se formaron o se
intensificaron en el lugar). El resultado (config/calibracion_llegadas.json)
dice, por localidad, plazo e intensidad pronosticada, cuántas veces llegó de
verdad: ClimaAR lo informa como "en casos parecidos llegó X de cada 10".

Uso: python herramientas/calibrar_llegadas.py [--desde 20261012] [--hasta 20261120]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import localidades_radar as lr  # noqa: E402
import nowcast_flujo_optico as nf  # noqa: E402
from scripts.rma10_features import latlon_to_pixel  # noqa: E402

SALIDA = ROOT / "config" / "calibracion_llegadas.json"
TOLERANCIA_MIN = 20
CERCA_KM = 20.0
PLAZOS = ((0, 20), (20, 40), (40, 61))
INTENSIDAD = ((35, 40), (40, 45), (45, 99))


def balde(valor, rangos):
    for a, b in rangos:
        if a <= valor < b:
            return f"{a}-{b if b < 61 else 60}" if rangos is PLAZOS else (f"{a}-{b}" if b < 99 else f">={a}")
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default="20261001")
    ap.add_argument("--hasta", default="20991231")
    ap.add_argument("--salida", default=str(SALIDA))
    a = ap.parse_args(argv)

    frames = [p for p in sorted(nf.HISTORICO.glob("20*.png")) if a.desde <= p.stem[:8] <= a.hasta]
    locs = lr.cargar_localidades()
    obs = {}        # t -> {loc: (pico_zona, max_20km)}
    pron = []       # (t, loc, eta, dbz)
    previo = None
    clutter = None
    for i, p in enumerate(frames):
        t = nf.timestamp_de_nombre(p)
        d = nf.cargar_dbz(p)
        if d is None:
            continue
        if clutter is None:
            clutter = nf.cargar_mascara_clutter(d.shape)
            h, w = d.shape
            km = nf.km_por_pixel(w)
            zonas = lr._zonas(d.shape, locs)
            cerca = {}
            for loc in locs:
                cx, cy = latlon_to_pixel(loc["lat"], loc["lon"], w, h)
                cerca[loc["id"]] = nf.disco((h, w), cx, cy, CERCA_KM / km)
        c = nf.campo_limpio(d, clutter)
        obs[t] = {k: (lr._pico(c[z]), lr._pico(c[cerca[k]])) for k, (z, _, _) in zonas.items()}
        if previo is not None and 5 <= (t - previo[0]).total_seconds() / 60 <= 30 and (c >= 35).sum() >= 30:
            gap = (t - previo[0]).total_seconds() / 60
            flujo = nf.estimar_flujo(previo[1], c, gap, km)
            if flujo["valido"]:
                for k, l in lr.llegadas(c, flujo, locs).items():
                    if l["eta_min"] is not None:
                        pron.append((t, k, l["eta_min"], l["dbz_al_llegar"]))
        previo = (t, c)
        if i % 300 == 0:
            print(f"{i}/{len(frames)} {t:%d/%m %H:%M} pronósticos {len(pron)}", flush=True)

    tiempos = sorted(obs)

    def ventana(t0, t1):
        return [obs[t] for t in tiempos if t0 < t <= t1]

    tabla = {}
    for t, k, eta, dbz in pron:
        v = ventana(t, t + timedelta(minutes=eta + TOLERANCIA_MIN))
        if not v:
            continue
        llego = any(o[k][0] >= 35 for o in v)
        desvio = (not llego) and any(o[k][1] >= 35 for o in v)
        clave = (k, balde(eta, PLAZOS), balde(dbz, INTENSIDAD))
        fila = tabla.setdefault(clave, {"casos": 0, "llego": 0, "se_desvio": 0, "se_deshizo": 0})
        fila["casos"] += 1
        fila["llego" if llego else ("se_desvio" if desvio else "se_deshizo")] += 1

    # Llegadas sin pronóstico: comienzo de eco >=35 sobre la localidad sin
    # ningún pronóstico para esa localidad en los 60 min previos.
    sorpresas = {}
    pron_por = {}
    for t, k, eta, _ in pron:
        pron_por.setdefault(k, []).append(t)
    for k in zonas:
        antes = 0.0
        for t in tiempos:
            ahora = obs[t][k][0]
            if ahora >= 35 and antes < 35:
                tot = sorpresas.setdefault(k, {"llegadas": 0, "sin_pronostico": 0})
                tot["llegadas"] += 1
                if not any(t - timedelta(minutes=60) <= tp < t for tp in pron_por.get(k, [])):
                    tot["sin_pronostico"] += 1
            antes = ahora

    salida = {
        "version": "1.0.0",
        "periodo": {"desde": tiempos[0].isoformat() if tiempos else None,
                    "hasta": tiempos[-1].isoformat() if tiempos else None, "frames": len(tiempos)},
        "criterio": {"eco_dbz": 35, "tolerancia_min": TOLERANCIA_MIN, "cerca_km": CERCA_KM},
        "por_localidad": {},
        "global": {},
        "llegadas_sin_pronostico": sorpresas,
    }
    for (k, plazo, inten), fila in sorted(tabla.items()):
        salida["por_localidad"].setdefault(k, {}).setdefault(plazo, {})[inten] = fila
        g = salida["global"].setdefault(plazo, {}).setdefault(inten, {"casos": 0, "llego": 0, "se_desvio": 0, "se_deshizo": 0})
        for c in g:
            g[c] += fila[c]
    Path(a.salida).write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(salida["global"], ensure_ascii=False))
    print(json.dumps(sorpresas, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
