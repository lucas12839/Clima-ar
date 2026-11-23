#!/usr/bin/env python3
"""¿Por qué sobran avisos de "llega una tormenta a la zona de mi localidad (10 km)"?

Mismos casos y mismo método que lo que se ve (25 caminos, "probable" = 2 de 25,
corrector con satélite y rayos; herramientas/simular_envolvente.py). Para cada
aviso de más mira qué pasó de verdad en la hora siguiente:

- casi: en la zona hubo eco de 30 a 35 dBZ (llegó, pero un poco más débil);
- de_costado: la tormenta pasó a 10-25 km (no entró en la zona);
- se_deshizo: hubo tormenta cerca al principio y se debilitó antes de llegar;
- nada: nada cerca (eco que no era tormenta, o se armó y desarmó lejos).

Además prueba filtros que usan solo lo que ya se sabe a la hora del aviso, y
dice cuánto bajan las falsas alarmas y cuánto se pierde de detección.

Uso: python herramientas/analizar_falsas_alarmas.py --goes /tmp/goes [--dias ...] [--paso 2] [--atraso 20]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts", ROOT / "herramientas"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import corrector_tiempo_real as ct  # noqa: E402
import nowcast_capas as ncap  # noqa: E402
import nowcast_capas_nucleo as nc  # noqa: E402
import nowcast_flujo_optico as nf  # noqa: E402
import simular_envolvente as se  # noqa: E402
import simular_huecos as sh  # noqa: E402
import simular_llegadas as sl  # noqa: E402

TORMENTA = 35.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goes", required=True)
    ap.add_argument("--paso", type=int, default=2)
    ap.add_argument("--dias", default=None)
    ap.add_argument("--atraso", type=int, default=20)
    ap.add_argument("--salida", default=None)
    a = ap.parse_args(argv)
    goes = sh.cargar_goes(Path(a.goes))
    cfg = {**ncap.CONFIG_DEFECTO, **json.loads((ROOT / "config" / "nowcast_capas.json").read_text(encoding="utf-8"))}
    frames = {nf.timestamp_de_nombre(p): p for p in nf.HISTORICO.glob("20*.png")}
    frames.pop(None, None)
    dias = set(a.dias.split(",")) if a.dias else None
    shape = nf.cargar_dbz(next(iter(frames.values()))).shape
    clutter = nc.cargar_clutter_total(shape)
    km_px = nf.km_por_pixel(shape[1])
    z10, z25 = sl.zonas(shape, 10.0), sl.zonas(shape, 25.0)
    cache = {}

    def campo(dt):
        if dt not in frames:
            return None
        if dt not in cache:
            d = nf.cargar_dbz(frames[dt])
            cache[dt] = None if d is None else nc.campo_limpio(d, clutter)
            if len(cache) > 60:
                cache.pop(next(iter(cache)))
        return cache[dt]

    def verdad(dt):
        c = campo(dt)
        if c is None:
            return None
        sat = sh.satelite_cercano(goes, dt, shape)
        return ct.quitar_ecos_sin_nube(c, sat) if sat is not None else c

    causas = Counter()
    # filtros[nombre] = [aciertos, falsas, omisiones]
    filtros = defaultdict(lambda: [0, 0, 0])
    casos = 0
    detalle = []
    for i, t0 in enumerate(sorted(frames)):
        if i % a.paso or (dias and t0.strftime("%Y%m%d") not in dias):
            continue
        act, prev, pprev = campo(t0), campo(t0 - timedelta(minutes=10)), campo(t0 - timedelta(minutes=20))
        if act is None or prev is None or int((act >= 20).sum()) < sh.MIN_PIX_ECO:
            continue
        f1 = nf.estimar_flujo(prev, act, 10.0, km_px)
        if not f1["valido"]:
            continue
        f2 = nf.estimar_flujo(pprev, prev, 10.0, km_px) if pprev is not None else None
        u, v = sh.flujo_promediado(f1, f2)
        sat_t0 = sh.satelite_cercano(goes, t0, shape)
        act_cc = ct.quitar_ecos_sin_nube(act, sat_t0) if sat_t0 is not None else act
        prev_cc = prev
        ahora = t0 + timedelta(minutes=a.atraso)
        reales = {p: verdad(ahora + timedelta(minutes=p)) for p in sl.PLAZOS}
        if sum(r is not None for r in reales.values()) < 6:
            continue
        casos += 1
        tiempos = [float(a.atraso + p) for p in sl.PLAZOS]
        series = [nc.serie(act_cc, *nc.rotar(u, v, g, f), tiempos) for f, g in se.ESC25]
        pila = {t: np.sort(np.stack([s[t] for s in series]), axis=0) for t in tiempos}
        sat, rlat, rlon = sh.datos_reales(goes, ahora, shape)
        datos = {"satelite": sat, "rayos_lat": list(rlat), "rayos_lon": list(rlon), "info": {}}
        campos = {t: pila[t][-2] for t in tiempos}
        corr = ncap.con_corrector(cfg, dict(campos), float(a.atraso), {"u": u, "v": v}, datos, None)
        picos = {p: nc.picos_por_zona(corr[float(a.atraso + p)], z10) for p in sl.PLAZOS}
        obs10 = {p: nc.picos_por_zona(r, z10) for p, r in reales.items() if r is not None}
        obs25 = {p: nc.picos_por_zona(r, z25) for p, r in reales.items() if r is not None}
        # Lo que ya se sabe a la hora del aviso: cuánto creció o se debilitó la tormenta que
        # trae el aviso (pico en la zona de 25 km, imagen actual contra la anterior) y si
        # el satélite muestra nube fría encima.
        pico_act = nc.picos_por_zona(act_cc, z25)
        pico_prev = nc.picos_por_zona(prev_cc, z25)
        for loc in z10:
            fp = sl.primera(picos, loc, TORMENTA)
            fo = sl.primera(obs10, loc, TORMENTA)
            si_p, si_o = fp is not None, fo is not None
            if not si_p and not si_o:
                continue
            tendencia = pico_act[loc] - pico_prev[loc]
            # ¿cuántos caminos (de 25) la traen? (solo con el campo central de los caminos)
            n_caminos = 0
            for s in series:
                if any(nc.picos_por_zona(s[float(a.atraso + p)], {loc: z10[loc]})[loc] >= TORMENTA for p in sl.PLAZOS):
                    n_caminos += 1
            variantes = {
                "actual (2 de 25)": True,
                "3 de 25": n_caminos >= 3,
                "4 de 25": n_caminos >= 4,
                "sin debilitarse >5 dBZ": tendencia > -5.0,
                "sin debilitarse >8 dBZ": tendencia > -8.0,
                "3 de 25 o creciendo": n_caminos >= 3 or tendencia >= 0,
            }
            for nombre, pasa in variantes.items():
                aviso = si_p and pasa
                acc = filtros[nombre]
                if aviso and si_o:
                    acc[0] += 1
                elif aviso:
                    acc[1] += 1
                elif si_o:
                    acc[2] += 1
            if si_p and not si_o:
                maximo10 = max((v[loc] for v in obs10.values()), default=0.0)
                maximo25 = max((v[loc] for v in obs25.values()), default=0.0)
                if maximo10 >= 30.0:
                    c = "casi (30-35 dBZ en la zona)"
                elif maximo25 >= TORMENTA:
                    c = "de costado (pasó a 10-25 km)"
                elif pico_act[loc] >= TORMENTA:
                    c = "se deshizo antes de llegar"
                else:
                    c = "nada cerca"
                causas[c] += 1
                detalle.append({"t0": t0.isoformat(), "loc": loc, "causa": c, "tendencia_dbz": round(tendencia, 1),
                                "caminos": n_caminos})
        if casos % 25 == 0:
            print(f"... {casos} casos", file=sys.stderr, flush=True)

    def met(acc):
        h, f, m = acc
        return {"detectadas_pct": round(100 * h / (h + m), 1) if h + m else None,
                "falsas_alarmas_pct": round(100 * f / (h + f), 1) if h + f else None, "aciertos": h,
                "falsas": f, "omisiones": m}

    total_f = sum(causas.values())
    salida = {"casos": casos, "atraso_min": a.atraso,
              "causas_falsas_alarmas": {k: f"{v} ({round(100 * v / total_f)} %)" for k, v in causas.most_common()},
              "filtros": {k: met(v) for k, v in filtros.items()}}
    print(json.dumps(salida, ensure_ascii=False, indent=1))
    if a.salida:
        Path(a.salida).write_text(json.dumps({**salida, "detalle": detalle}, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
