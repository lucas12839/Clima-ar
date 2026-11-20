#!/usr/bin/env python3
"""Punto 6: ¿pysteps (biblioteca abierta de nowcast, sin entrenamiento) mejora
nuestro nowcast en capas? Misma prueba de huecos que simular_huecos.py.

Compara, con las mismas imágenes de partida:
- capas:        nuestro nowcast en capas (flujo Farnebäck de 3 imágenes + escenarios).
- pysteps_lk:   movimiento Lucas-Kanade de pysteps (3 imágenes) + extrapolación.
- pysteps_sprog: S-PROG (extrapolación que suaviza las escalas chicas a medida que
                 se vuelven impredecibles).
- pysteps_anvil: ANVIL (extrapolación con crecimiento y debilitamiento).
- capas_y_anvil: el mayor de los dos en cada píxel.
También mide el tiempo de cálculo (el ciclo tiene que seguir siendo rápido).

Uso: python herramientas/comparar_pysteps.py [--paso 6] [--salida archivo.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts", ROOT / "herramientas"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import nowcast_capas_nucleo as nc  # noqa: E402
import nowcast_flujo_optico as nf  # noqa: E402
import simular_huecos as sh  # noqa: E402

HUECOS = (10, 20, 30, 40, 60)
METODOS = ("hoy", "capas", "pysteps_lk", "pysteps_sprog", "pysteps_anvil", "capas_y_anvil",
           "capas_y_anvil_mismo_movimiento")


def a_lluvia(dbz):
    r = (10 ** (np.nan_to_num(dbz) / 10.0) / 200.0) ** (1 / 1.6)
    r[np.nan_to_num(dbz) < nf.UMBRAL_ECO_DBZ - 5] = 0.0
    return r


def a_dbz(r):
    r = np.nan_to_num(r, nan=0.0)
    z = 10 * np.log10(np.maximum(200.0 * np.maximum(r, 1e-6) ** 1.6, 1e-6))
    z[r < 0.05] = 0.0
    return z.astype(np.float32)


def main(argv=None) -> int:
    from pysteps import motion, nowcasts

    ap = argparse.ArgumentParser()
    ap.add_argument("--paso", type=int, default=6)
    ap.add_argument("--salida", default=None)
    args = ap.parse_args(argv)
    sh.HUECOS = HUECOS
    sh.METODOS = METODOS
    frames = {nf.timestamp_de_nombre(p): p for p in nf.HISTORICO.glob("20*.png")}
    frames.pop(None, None)
    primero = nf.cargar_dbz(next(iter(frames.values())))
    shape = primero.shape
    clutter = nc.cargar_clutter_total(shape)
    km_px = nf.km_por_pixel(shape[1])
    zl, zr = sh.zonas_localidades(shape), sh.zonas_region(shape)
    tab_loc, tab_reg = sh.nueva_tabla(), sh.nueva_tabla()
    tiempos = {m: 0.0 for m in METODOS}
    cache = {}

    def campo(dt):
        if dt not in frames:
            return None
        if dt not in cache:
            d = nf.cargar_dbz(frames[dt])
            cache[dt] = None if d is None else nf.campo_limpio(d, clutter)
            if len(cache) > 30:
                cache.pop(next(iter(cache)))
        return cache[dt]

    lk = motion.get_method("LK")
    extrap = nowcasts.get_method("extrapolation")
    sprog = nowcasts.get_method("sprog")
    anvil = nowcasts.get_method("anvil")
    casos = 0
    for i, t0 in enumerate(sorted(frames)):
        if i % args.paso:
            continue
        act, prev, pprev = campo(t0), campo(t0 - timedelta(minutes=10)), campo(t0 - timedelta(minutes=20))
        if act is None or prev is None or pprev is None or int((act >= 20).sum()) < sh.MIN_PIX_ECO:
            continue
        obj = {g: t0 + timedelta(minutes=g) for g in HUECOS if (t0 + timedelta(minutes=g)) in frames}
        if not obj:
            continue
        t = time.time()
        f1 = nf.estimar_flujo(prev, act, 10.0, km_px)
        f2 = nf.estimar_flujo(pprev, prev, 10.0, km_px)
        if not f1["valido"]:
            continue
        u3, v3 = sh.flujo_promediado(f1, f2)
        capas = nc.campos_escenarios(act, u3, v3, sorted(obj))
        tiempos["capas"] += time.time() - t
        # pysteps: pasos de 10 min (una imagen = un paso).
        t = time.time()
        r3 = np.stack([a_lluvia(pprev), a_lluvia(prev), a_lluvia(act)])
        z3 = np.stack([pprev, prev, act]).astype(float)
        vel = lk(z3)
        n_pasos = max(obj) // 10
        ext = extrap(act.astype(float), vel, n_pasos)
        tiempos["pysteps_lk"] += time.time() - t
        t = time.time()
        try:
            rdb = 10 * np.log10(np.maximum(r3, 0.1))
            sp = sprog(rdb, vel, n_pasos, n_cascade_levels=6, precip_thr=-10.0 + 1e-3)
            sp = a_dbz(10 ** (sp / 10.0))
        except Exception as exc:
            print("sprog:", exc, file=sys.stderr)
            sp = None
        tiempos["pysteps_sprog"] += time.time() - t
        t = time.time()
        try:
            an = anvil(r3, vel, n_pasos, ar_window_radius=25, ar_order=1)
            an = np.stack([a_dbz(x) for x in an])
        except Exception as exc:
            print("anvil:", exc, file=sys.stderr)
            an = None
        tiempos["pysteps_anvil"] += time.time() - t
        # ANVIL con NUESTRO movimiento (3 imágenes, con control de velocidad): un solo campo de movimiento.
        t = time.time()
        try:
            an2 = anvil(r3, np.stack([u3 * 10.0, v3 * 10.0]), n_pasos, ar_window_radius=25, ar_order=1)
            an2 = np.stack([a_dbz(x) for x in an2])
        except Exception as exc:
            print("anvil2:", exc, file=sys.stderr)
            an2 = None
        tiempos["capas_y_anvil_mismo_movimiento"] += time.time() - t
        casos += 1
        for g, tg in obj.items():
            obs = campo(tg)
            if obs is None:
                continue
            k = g // 10 - 1
            pron = {"hoy": act, "capas": capas[float(g)], "pysteps_lk": np.nan_to_num(ext[k]).astype(np.float32)}
            if sp is not None:
                pron["pysteps_sprog"] = sp[k]
            if an is not None:
                pron["pysteps_anvil"] = an[k]
                # Combinación: hay eco si lo dan los escenarios o ANVIL (crecimiento).
                pron["capas_y_anvil"] = np.maximum(capas[float(g)], an[k])
            if an2 is not None:
                pron["capas_y_anvil_mismo_movimiento"] = np.maximum(capas[float(g)], an2[k])
            o_l, o_r = nc.picos_por_zona(obs, zl), nc.picos_por_zona(obs, zr)
            for m, c in pron.items():
                sh.sumar(tab_loc, m, g, nc.picos_por_zona(c, zl), o_l)
                sh.sumar(tab_reg, m, g, nc.picos_por_zona(c, zr), o_r)
        if casos % 25 == 0:
            print(f"... {casos} casos", file=sys.stderr)

    def armar(tabla):
        return {m: {f"{g}min": {f"{int(u)}dbz": sh.metricas(tabla[m][g][u]) for u in sh.UMBRALES} for g in HUECOS}
                for m in METODOS}

    res = {"casos": casos, "segundos_por_caso": {m: round(v / max(casos, 1), 2) for m, v in tiempos.items()},
           "localidades": armar(tab_loc), "region": armar(tab_reg)}
    if args.salida:
        Path(args.salida).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for nombre in ("localidades", "region"):
        for u in sh.UMBRALES:
            print(f"\n{nombre} {int(u)} dBZ (CSI)   " + "  ".join(f"{g:>5}" for g in HUECOS))
            for m in METODOS:
                print(f"{m:14s}" + "  ".join(f"{(res[nombre][m][f'{g}min'][f'{int(u)}dbz']['csi'] or 0):5.3f}" for g in HUECOS))
    print("\nsegundos por caso:", res["segundos_por_caso"], "casos:", casos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
