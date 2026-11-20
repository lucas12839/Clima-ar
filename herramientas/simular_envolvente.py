#!/usr/bin/env python3
"""Punto 5b: "¿llega a mi localidad en la próxima hora?" con la ENVOLVENTE de escenarios.

Hallazgo de `analizar_omisiones.py`: 2 de cada 3 tormentas que se escapaban YA estaban en el
radar viejo a menos de 30 km, casi siempre del lado de donde viene el movimiento. El nowcast
las tenía, pero en el campo "3 de 9 escenarios" no llegaban a la localidad (la tormenta se
desvía, acelera o se estira y el escenario central no la trae). Las que nacen de la nada son
pocas (~5 %) y los rayos casi no ayudan (hay rayos cerca en el 5 % de los casos).

Esta simulación prueba, con los mismos casos que `simular_llegadas.py`, cuántos escenarios
tienen que traer la tormenta para decir "llega": k de 9 (los de hoy) y k de 25 (velocidad
0,6-1,4 y dirección ±40°). En todos se aplica el mismo corrector con satélite y rayos.

Uso: python herramientas/simular_envolvente.py --goes /tmp/goes [--paso 2] [--salida x.json]
"""
from __future__ import annotations

import argparse
import json
import sys
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
import simular_huecos as sh  # noqa: E402
import simular_llegadas as sl  # noqa: E402

ESC9 = nc.ESCENARIOS
ESC25 = [(f, g) for f in (0.6, 0.8, 1.0, 1.2, 1.4) for g in (-40.0, -20.0, 0.0, 20.0, 40.0)]
VARIANTES = {"actual": None, "9_k2": (ESC9, 2), "9_k1": (ESC9, 1),
             "25_k5": (ESC25, 5), "25_k3": (ESC25, 3), "25_k2": (ESC25, 2), "25_k1": (ESC25, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goes", required=True)
    ap.add_argument("--paso", type=int, default=2)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--atrasos", default=None, help="ej.: 40,60,90 (por defecto 10,20,30)")
    args = ap.parse_args(argv)
    if args.atrasos:
        sl.ATRASOS = tuple(int(a) for a in args.atrasos.split(","))
    goes = sh.cargar_goes(Path(args.goes))
    cfg = {**ncap.CONFIG_DEFECTO, **json.loads((ROOT / "config" / "nowcast_capas.json").read_text(encoding="utf-8"))}
    frames = {nf.timestamp_de_nombre(p): p for p in nf.HISTORICO.glob("20*.png")}
    frames.pop(None, None)
    shape = nf.cargar_dbz(next(iter(frames.values()))).shape
    clutter = nc.cargar_clutter_total(shape)
    km_px = nf.km_por_pixel(shape[1])
    zonas = {"localidad": sl.zonas(shape, 0.0), "margen_10km": sl.zonas(shape, 10.0)}
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

    # tabla[variante][zona][umbral] = [si-si, si-no, no-si, no-no, hora_ok, hora_total]
    tabla = {k: {z: {u: [0] * 6 for u in sl.UMBRALES} for z in zonas} for k in VARIANTES}
    casos = 0
    for i, t0 in enumerate(sorted(frames)):
        if i % args.paso:
            continue
        act, prev, pprev = campo(t0), campo(t0 - timedelta(minutes=10)), campo(t0 - timedelta(minutes=20))
        if act is None or prev is None or int((act >= 20).sum()) < sh.MIN_PIX_ECO:
            continue
        f1 = nf.estimar_flujo(prev, act, 10.0, km_px)
        if not f1["valido"]:
            continue
        f2 = nf.estimar_flujo(pprev, prev, 10.0, km_px) if pprev is not None else None
        u, v = sh.flujo_promediado(f1, f2)
        flujo = {"u": u, "v": v}
        sat_t0 = sh.satelite_cercano(goes, t0, shape)
        act_cc = ct.quitar_ecos_sin_nube(act, sat_t0) if sat_t0 is not None else act
        casos += 1
        # Los 25 escenarios se calculan una sola vez por caso (los 9 de hoy están dentro de los 25).
        todos_t = sorted({float(a + p) for a in sl.ATRASOS for p in sl.PLAZOS})
        series = [nc.serie(act_cc, *nc.rotar(u, v, g, f), todos_t) for f, g in ESC25]
        idx9 = [ESC25.index(e) for e in ESC9]
        pila25 = {t: np.sort(np.stack([s[t] for s in series]), axis=0) for t in todos_t}
        pila9 = {t: np.sort(np.stack([series[j][t] for j in idx9]), axis=0) for t in todos_t}
        for atraso in sl.ATRASOS:
            ahora = t0 + timedelta(minutes=atraso)
            reales = {p: verdad(ahora + timedelta(minutes=p)) for p in sl.PLAZOS}
            if sum(r is not None for r in reales.values()) < 6:
                continue
            tiempos = [float(atraso + p) for p in sl.PLAZOS]
            sat, rlat, rlon = sh.datos_reales(goes, ahora, shape)
            datos = {"satelite": sat, "rayos_lat": list(rlat), "rayos_lon": list(rlon), "info": {}}
            pron = {}
            for var, spec in VARIANTES.items():
                if spec is None:   # lo de hoy: 3 de 9 hasta 40 min desde la imagen, 2 de 9 después
                    campos = {t: nc.campo_escenarios(pila9[t], t) for t in tiempos}
                else:
                    esc, k = spec
                    pila = pila9 if esc is ESC9 else pila25
                    campos = {t: pila[t][-k] for t in tiempos}
                corr = ncap.con_corrector(cfg, dict(campos), float(atraso), flujo, datos, None)
                pron[var] = {p: corr[float(atraso + p)] for p in sl.PLAZOS}
            for nz, zz in zonas.items():
                obs = {p: nc.picos_por_zona(r, zz) for p, r in reales.items() if r is not None}
                for var, serie_p in pron.items():
                    picos = {p: nc.picos_por_zona(c, zz) for p, c in serie_p.items()}
                    for nu, umbral in sl.UMBRALES.items():
                        acc = tabla[var][nz][nu]
                        for loc in zz:
                            fp, fo = sl.primera(picos, loc, umbral), sl.primera(obs, loc, umbral)
                            si_p, si_o = fp is not None, fo is not None
                            acc[0 if si_p and si_o else 1 if si_p else 2 if si_o else 3] += 1
                            if si_p and si_o:
                                acc[5] += 1
                                acc[4] += abs(fp - fo) <= sl.TOLERANCIA_HORA_MIN
        if casos % 25 == 0:
            print(f"... {casos} casos", file=sys.stderr, flush=True)

    def met(acc):
        a, b, c, d, hok, ht = acc
        return {"llego_y_aviso": a, "aviso_y_no_llego": b, "llego_sin_aviso": c, "ni_aviso_ni_llego": d,
                "deteccion_pct": round(100 * a / (a + c), 1) if a + c else None,
                "falsas_alarmas_pct": round(100 * b / (a + b), 1) if a + b else None,
                "csi_pct": round(100 * a / (a + b + c), 1) if a + b + c else None,
                "hora_acertada_pct": round(100 * hok / ht, 1) if ht else None}

    res = {"casos": casos, "resultado": {v: {z: {u: met(tabla[v][z][u]) for u in sl.UMBRALES} for z in zonas}
                                         for v in VARIANTES}}
    if args.salida:
        Path(args.salida).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for z in zonas:
        for u in sl.UMBRALES:
            print(f"\n{u}, zona {z}: detección / falsas alarmas / CSI / hora ±15")
            for v in VARIANTES:
                r = res["resultado"][v][z][u]
                print(f"  {v:8s} {r['deteccion_pct']}/{r['falsas_alarmas_pct']}/{r['csi_pct']}/{r['hora_acertada_pct']}")
    print("casos:", casos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
