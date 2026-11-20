#!/usr/bin/env python3
"""Capa 6 del punto 5b: validación simulada de los huecos del radar.

Con el radar guardado (data/radar/historico) se simula que SINARAME se atrasa:
se toma la imagen de la hora T0 (y la anterior, T0-10) y se pronostica qué hay
en T0 + 10 ... T0 + 90 min, como si esa fuera la hora actual y no hubiera
llegado ninguna imagen nueva. Se compara con la imagen real que llegó en esa hora.

Métodos comparados (cada capa entra solo si mejora):
- hoy:        lo que se mostraba como "ahora": la imagen vieja tal como llegó.
- capa1:      la imagen vieja movida a la hora actual (flujo de 2 imágenes).
- capa1_mov3: igual, con el movimiento promediado de 3 imágenes (punto 6).
- capa1_2_mov3: capa 1 con 3 imágenes + tendencia de crecimiento/debilitamiento del radar.
- esc_mov3_N: capa 5, 9 escenarios (velocidad ±20 %, dirección ±20°): hay eco si
  al menos N escenarios lo dan.
- capas:      el criterio final de scripts/nowcast_capas.py (N=3 hasta 40 min, después N=2).

Dónde se mide: las 7 localidades y una grilla de puntos de la región (cada
25 km, de 20 a 200 km del radar, círculo de 5 km), para tener más casos.
Umbrales: 20 dBZ (lluvia) y 35 dBZ (tormenta).

Uso: python herramientas/simular_huecos.py [--desde 20261027] [--hasta 20261120]
         [--paso 2] [--salida resultado.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import localidades_radar as lr  # noqa: E402
import nowcast_capas_nucleo as nc  # noqa: E402
import nowcast_flujo_optico as nf  # noqa: E402

HUECOS = (10, 20, 30, 40, 60, 90)
UMBRALES = (20.0, 35.0)
METODOS = ("hoy", "capa1", "capa1_mov3", "capa1_2_mov3", "esc_mov3_22", "esc_mov3_33", "capas")
ESC_UMBRAL = {"esc_mov3_22": 2, "esc_mov3_33": 3}  # escenarios (de 9) que tienen que dar el eco
MIN_PIX_ECO = 300


def zonas_region(shape) -> dict[str, np.ndarray]:
    h, w = shape
    km_px = nf.km_por_pixel(w)
    c = (w - 1) / 2.0
    zonas = {}
    for dx in range(-200, 201, 25):
        for dy in range(-200, 201, 25):
            d = math.hypot(dx, dy)
            if 20 <= d <= 200:
                zonas[f"p{dx}_{dy}"] = nf.disco((h, w), c + dx / km_px, c + dy / km_px, 5.0 / km_px)
    return zonas


def zonas_localidades(shape) -> dict[str, np.ndarray]:
    return {k: z for k, (z, _, _) in lr._zonas(shape, lr.cargar_localidades()).items()}


def nueva_tabla():
    return {m: {g: {u: [0, 0, 0, 0] for u in UMBRALES} for g in HUECOS} for m in METODOS}


def sumar(tabla, metodo, g, pron: dict, obs: dict):
    for u in UMBRALES:
        acc = tabla[metodo][g][u]
        for k in obs:
            p, o = pron[k] >= u, obs[k] >= u
            acc[0 if p and o else 1 if p else 2 if o else 3] += 1


def metricas(acc):
    a, b, c, _ = acc
    pod = a / (a + c) if a + c else None
    far = b / (a + b) if a + b else None
    csi = a / (a + b + c) if a + b + c else None
    return {"aciertos": a, "falsas_alarmas": b, "omisiones": c,
            "deteccion": None if pod is None else round(pod, 3),
            "falsas_alarmas_pct": None if far is None else round(far, 3),
            "csi": None if csi is None else round(csi, 3)}


def flujo_promediado(f1: dict, f2: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """Movimiento de 3 imágenes: promedio de los dos pares (pesa más el último)."""
    if f2 is None or not f2.get("valido"):
        return f1["u"], f1["v"]
    return (0.6 * f1["u"] + 0.4 * f2["u"]).astype(np.float32), (0.6 * f1["v"] + 0.4 * f2["v"]).astype(np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default="2026")
    ap.add_argument("--hasta", default="2100")
    ap.add_argument("--paso", type=int, default=1, help="usar 1 de cada N imágenes de partida")
    ap.add_argument("--salida", default=None)
    ap.add_argument("--sin-clutter-intermitente", action="store_true",
                    help="usar solo la máscara de clutter vieja (para comparar)")
    args = ap.parse_args(argv)

    frames = {}
    for p in nf.HISTORICO.glob("20*.png"):
        dt = nf.timestamp_de_nombre(p)
        if dt is not None and args.desde <= p.stem[:8] <= args.hasta:
            frames[dt] = p
    cache: dict = {}

    def campo(dt):
        if dt not in frames:
            return None
        if dt not in cache:
            d = nf.cargar_dbz(frames[dt])
            cache[dt] = None if d is None else nc.campo_limpio(d, clutter)
            if len(cache) > 30:
                cache.pop(next(iter(cache)))
        return cache[dt]

    primero = nf.cargar_dbz(next(iter(frames.values())))
    shape = primero.shape
    clutter = nc.cargar_clutter_total(shape)
    if args.sin_clutter_intermitente:
        nc._INTERMITENTE[shape] = None
    km_px = nf.km_por_pixel(shape[1])
    zl, zr = zonas_localidades(shape), zonas_region(shape)
    tab_loc, tab_reg = nueva_tabla(), nueva_tabla()
    mae = {m: {g: [0.0, 0] for g in HUECOS} for m in METODOS}
    casos, dias, t_ini = 0, set(), time.time()
    por_loc: dict = {}

    for i, t0 in enumerate(sorted(frames)):
        if i % args.paso:
            continue
        tp, tpp = t0 - timedelta(minutes=10), t0 - timedelta(minutes=20)
        act, prev = campo(t0), campo(tp)
        if act is None or prev is None or int((act >= nf.UMBRAL_ECO_DBZ).sum()) < MIN_PIX_ECO:
            continue
        objetivos = {g: t0 + timedelta(minutes=g) for g in HUECOS if (t0 + timedelta(minutes=g)) in frames}
        if not objetivos:
            continue
        f1 = nf.estimar_flujo(prev, act, 10.0, km_px)
        if not f1["valido"]:
            continue
        pprev = campo(tpp)
        f2 = nf.estimar_flujo(pprev, prev, 10.0, km_px) if pprev is not None else None
        u3, v3 = flujo_promediado(f1, f2)
        tasa3 = nc.tendencia(prev, act, u3, v3, 10.0)
        gs = sorted(objetivos)
        s1 = nc.serie(act, f1["u"], f1["v"], gs)
        s3 = nc.serie(act, u3, v3, gs)
        s32 = nc.serie(act, u3, v3, gs, tasa3)
        esc = [nc.serie(act, *nc.rotar(u3, v3, d, f), gs) for f, d in nc.ESCENARIOS]
        casos += 1
        dias.add(t0.strftime("%Y-%m-%d"))
        for g, tg in objetivos.items():
            obs = campo(tg)
            if obs is None:
                continue
            g = float(g)
            pron = {"hoy": act, "capa1": s1[g], "capa1_mov3": s3[g], "capa1_2_mov3": s32[g]}
            pila = np.stack([e[g] for e in esc])
            for nombre, n in ESC_UMBRAL.items():
                # n-ésimo valor más alto entre escenarios: "al menos n escenarios dan >= x".
                pron[nombre] = np.sort(pila, axis=0)[-n]
            # "capas" = lo que usa scripts/nowcast_capas.py (mismo criterio).
            pron["capas"] = nc.campo_escenarios(pila, g)
            g = int(g)
            o_l, o_r = nc.picos_por_zona(obs, zl), nc.picos_por_zona(obs, zr)
            for m, c in pron.items():
                p_l, p_r = nc.picos_por_zona(c, zl), nc.picos_por_zona(c, zr)
                sumar(tab_loc, m, g, p_l, o_l)
                for k in o_l:
                    sumar(por_loc.setdefault(k, nueva_tabla()), m, g, {k: p_l[k]}, {k: o_l[k]})
                sumar(tab_reg, m, g, p_r, o_r)
                for k in o_r:
                    if o_r[k] >= 20 or p_r[k] >= 20:
                        mae[m][g][0] += abs(p_r[k] - o_r[k])
                        mae[m][g][1] += 1
        if casos % 100 == 0:
            print(f"... {casos} casos ({time.time() - t_ini:.0f} s)", file=sys.stderr)

    def armar(tabla):
        return {m: {f"{g}min": {f"{int(u)}dbz": metricas(tabla[m][g][u]) for u in UMBRALES} for g in HUECOS}
                for m in METODOS}

    resumen = {
        "casos": casos, "dias": sorted(dias),
        "localidades": armar(tab_loc), "region": armar(tab_reg),
        "por_localidad": {k: armar(t) for k, t in por_loc.items()},
        "error_medio_dbz_region": {m: {f"{g}min": round(mae[m][g][0] / mae[m][g][1], 2) if mae[m][g][1] else None
                                       for g in HUECOS} for m in METODOS},
    }
    texto = json.dumps(resumen, ensure_ascii=False, indent=1)
    if args.salida:
        Path(args.salida).write_text(texto + "\n", encoding="utf-8")
    # Tabla corta: CSI por método y hueco.
    for nombre in ("localidades", "region"):
        for u in UMBRALES:
            print(f"\n{nombre} {int(u)} dBZ (CSI)   " + "  ".join(f"{g:>5}" for g in HUECOS))
            for m in METODOS:
                print(f"{m:14s}" + "  ".join(f"{(resumen[nombre][m][f'{g}min'][f'{int(u)}dbz']['csi'] or 0):5.3f}" for g in HUECOS))
    print("\nerror medio dBZ región " + "  ".join(f"{g:>5}" for g in HUECOS))
    for m in METODOS:
        print(f"{m:14s}" + "  ".join(f"{(resumen['error_medio_dbz_region'][m][f'{g}min'] or 0):5.2f}" for g in HUECOS))
    print(f"\ncasos: {casos}, días: {len(dias)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
