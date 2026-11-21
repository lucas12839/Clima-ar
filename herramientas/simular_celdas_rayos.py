#!/usr/bin/env python3
"""Paso 2: ¿la tormenta pasa SOBRE mi localidad en los próximos 30 min?

Compara, con los días guardados (radar + historial real de rayos GLM y satélite):
- "5b": lo de hoy para la localidad exacta (2 de 25 caminos + corrector, como en producción).
- "rayos": celdas seguidas con sus rayos (scripts/celdas_rayos.py), con varias opciones.
- combinaciones de los dos.

Como en vivo: el radar llega con 10, 20 o 30 min de atraso; a la hora "ahora" solo hay
la imagen vieja, el satélite ya publicado y los rayos de los últimos 10 min.
Verdad: imágenes reales de ahora, +10, +20 y +30 min (limpias de ecos falsos con el
satélite): tormenta = 35 dBZ o más dentro de la localidad.
"Nuevas" = la tormenta no estaba sobre la localidad al empezar (lo difícil).

Uso: python herramientas/simular_celdas_rayos.py --goes /tmp/goeshist [--paso 2] [--salida x.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts", ROOT / "herramientas"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import celdas_rayos as cr  # noqa: E402
import corrector_tiempo_real as ct  # noqa: E402
import localidades_radar as lr  # noqa: E402
import nowcast_capas as ncap  # noqa: E402
import nowcast_capas_nucleo as nc  # noqa: E402
import nowcast_flujo_optico as nf  # noqa: E402
import simular_envolvente as se  # noqa: E402
import simular_huecos as sh  # noqa: E402
import simular_llegadas as sl  # noqa: E402

ATRASOS = (10, 20, 30)
PLAZOS = (0, 10, 20, 30)
UMBRAL = 35.0
VARIANTES_RAYOS = {
    "rayos": {},
    "rayos_margen0": {"margen_km": 0.0},
    "rayos_margen6": {"margen_km": 6.0},
    "rayos_mas_error": {"error_velocidad": 0.35, "error_direccion_grados": 30.0, "error_lugar_km": 5.0},
    "rayos_solo_5b": {"peso_movimiento_rayos": 0.0},
}
VARIANTES_CELDAS = {
    "celdas": {},
    "celdas_margen0": {"margen_km": 0.0},
    "celdas_margen6": {"margen_km": 6.0},
    "celdas_mas_error": {"error_por_minuto_km": 0.4, "error_velocidad": 0.35, "error_direccion_grados": 30.0},
    "celdas_sin_rayos": {"_sin_rayos": True},
}
UMBRALES_P = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goes", required=True)
    ap.add_argument("--paso", type=int, default=2)
    ap.add_argument("--salida", default=None)
    args = ap.parse_args(argv)
    goes = sh.cargar_goes(Path(args.goes))
    cfg = {**ncap.CONFIG_DEFECTO, **json.loads((ROOT / "config" / "nowcast_capas.json").read_text(encoding="utf-8"))}
    frames = {nf.timestamp_de_nombre(p): p for p in nf.HISTORICO.glob("20*.png")}
    frames.pop(None, None)
    shape = nf.cargar_dbz(next(iter(frames.values()))).shape
    h, w = shape
    clutter = nc.cargar_clutter_total(shape)
    km_px = nf.km_por_pixel(w)
    zonas = sl.zonas(shape, 0.0)
    locs_px = {}
    for loc in lr.cargar_localidades():
        x, y = nf.latlon_to_pixel(loc["lat"], loc["lon"], w, h)
        locs_px[loc["id"]] = (x, y, float(loc.get("radio_km", 4.0)) / km_px)
    glm_t = goes["glm_fin"]
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

    registros = []
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
        todos_t = sorted({float(a + p) for a in ATRASOS for p in PLAZOS})
        series = [nc.serie(act_cc, *nc.rotar(u, v, g, f), todos_t) for f, g in se.ESC25]
        pila = {t: np.sort(np.stack([s[t] for s in series]), axis=0) for t in todos_t}
        casos += 1
        for atraso in ATRASOS:
            ahora = t0 + timedelta(minutes=atraso)
            reales = {p: verdad(ahora + timedelta(minutes=p)) for p in PLAZOS}
            if sum(r is not None for r in reales.values()) < 3:
                continue
            obs = {p: nc.picos_por_zona(r, zonas) for p, r in reales.items() if r is not None}
            sat, rlat, rlon = sh.datos_reales(goes, ahora, shape)
            datos = {"satelite": sat, "rayos_lat": list(rlat), "rayos_lon": list(rlon), "info": {}}
            campos = {float(atraso + p): pila[float(atraso + p)][-2] for p in PLAZOS}
            corr = ncap.con_corrector(cfg, dict(campos), float(atraso), flujo, datos, None)
            base = {p: nc.picos_por_zona(corr[float(atraso + p)], zonas) for p in PLAZOS}
            # Rayos de los últimos 10 min con su edad.
            hh = np.datetime64(ahora.replace(tzinfo=None))
            sel = (glm_t > hh - np.timedelta64(10, "m")) & (glm_t <= hh)
            edad = (hh - glm_t[sel]).astype("timedelta64[s]").astype(float) / 60.0
            fx, fy = ct.a_pixel(goes["glm_lat"][sel], goes["glm_lon"][sel], w, h)
            lista = cr.celdas(fx, fy, edad, shape, km_px)
            probs = {var: cr.probabilidades(lista, u, v, km_px, locs_px, par)
                     for var, par in VARIANTES_RAYOS.items()}
            for var, par in VARIANTES_CELDAS.items():
                par = dict(par)
                sin_rayos = par.pop("_sin_rayos", False)
                probs[var] = cr.probabilidades_celdas(act_cc, u, v, km_px, float(atraso),
                                                      [] if sin_rayos else lista, locs_px, par)
            for loc in locs_px:
                prim_obs = next((p for p in PLAZOS if p in obs and obs[p][loc] >= UMBRAL), None)
                prim_base = next((p for p in PLAZOS if base[p][loc] >= UMBRAL), None)
                registros.append({
                    "obs": prim_obs is not None, "obs_ya": 0 in obs and obs[0][loc] >= UMBRAL,
                    "obs_min": prim_obs, "base": prim_base is not None, "base_min": prim_base,
                    "p": {var: probs[var][loc]["probabilidad"] for var in probs},
                    "min_celdas": probs["celdas"][loc]["minutos"],
                    "min": probs["rayos"][loc]["minutos"], "atraso": atraso,
                })
        if casos % 25 == 0:
            print(f"... {casos} casos, {len(registros)} registros", file=sys.stderr, flush=True)

    def metricas(pred, sub):
        a = b = c = d = 0
        for r in sub:
            pr, ob = pred(r), r["obs"]
            a += pr and ob
            b += pr and not ob
            c += (not pr) and ob
            d += (not pr) and (not ob)
        return {"detectadas": a, "falsas_alarmas": b, "omitidas": c,
                "deteccion_pct": round(100 * a / (a + c), 1) if a + c else None,
                "falsas_alarmas_pct": round(100 * b / (a + b), 1) if a + b else None,
                "csi_pct": round(100 * a / (a + b + c), 1) if a + b + c else None}

    subconjuntos = {"todas": registros, "nuevas": [r for r in registros if not r["obs_ya"]]}
    res = {"casos": casos, "registros": len(registros), "resultado": {}}
    for nombre, sub in subconjuntos.items():
        out = {"5b": metricas(lambda r: r["base"], sub)}
        for var in list(VARIANTES_RAYOS) + list(VARIANTES_CELDAS):
            for u_ in UMBRALES_P:
                out[f"{var}>={u_}"] = metricas(lambda r, var=var, u_=u_: r["p"][var] >= u_, sub)
        for u_ in UMBRALES_P:
            out[f"5b_o_rayos>={u_}"] = metricas(lambda r, u_=u_: r["base"] or r["p"]["rayos"] >= u_, sub)
            out[f"5b_y_rayos>={u_}"] = metricas(lambda r, u_=u_: r["base"] and r["p"]["rayos"] >= u_, sub)
            out[f"5b_y_celdas>={u_}"] = metricas(lambda r, u_=u_: r["base"] and r["p"]["celdas"] >= u_, sub)
            out[f"rayos>=0.5_o_5b_y_celdas>={u_}"] = metricas(
                lambda r, u_=u_: r["p"]["rayos"] >= 0.5 or (r["base"] and r["p"]["celdas"] >= u_), sub)
        res["resultado"][nombre] = out
    # Calibración: de los casos con probabilidad X, en cuántos pasó de verdad.
    cal = defaultdict(lambda: [0, 0])
    for r in subconjuntos["nuevas"]:
        b = min(int(r["p"]["rayos"] * 10), 9)
        if r["p"]["rayos"] > 0:
            cal[b][0] += 1
            cal[b][1] += r["obs"]
    res["calibracion_nuevas"] = {f"{10 * k}-{10 * k + 10}%": {"casos": n, "paso_pct": round(100 * s / n, 1)}
                                 for k, (n, s) in sorted(cal.items())}
    # Minuto de llegada: error con la primera imagen real con tormenta (nuevas, aciertos).
    errs = [abs(r["min"] - r["obs_min"]) for r in subconjuntos["nuevas"]
            if r["obs"] and r["p"]["rayos"] >= 0.3 and r["min"] is not None]
    res["error_minuto_llegada"] = {"casos": len(errs), "mediana_min": float(np.median(errs)) if errs else None}
    errs_c = [abs(r["min_celdas"] - r["obs_min"]) for r in subconjuntos["nuevas"]
              if r["obs"] and r["p"]["celdas"] >= 0.3 and r["min_celdas"] is not None]
    res["error_minuto_llegada_celdas"] = {"casos": len(errs_c),
                                          "mediana_min": float(np.median(errs_c)) if errs_c else None}
    cal_c = defaultdict(lambda: [0, 0])
    for r in subconjuntos["nuevas"]:
        if r["p"]["celdas"] > 0:
            b = min(int(r["p"]["celdas"] * 10), 9)
            cal_c[b][0] += 1
            cal_c[b][1] += r["obs"]
    res["calibracion_nuevas_celdas"] = {f"{10 * k}-{10 * k + 10}%": {"casos": n, "paso_pct": round(100 * s / n, 1)}
                                        for k, (n, s) in sorted(cal_c.items())}
    if args.salida:
        Path(args.salida).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        Path(args.salida).with_suffix(".registros.json").write_text(json.dumps(registros), encoding="utf-8")
    for nombre in subconjuntos:
        print(f"\n== {nombre}: detección / falsas alarmas / CSI")
        for k, m in res["resultado"][nombre].items():
            print(f"  {k:24s} {m['deteccion_pct']}/{m['falsas_alarmas_pct']}/{m['csi_pct']}  ({m['detectadas']}+{m['omitidas']})")
    print("\ncalibración (nuevas):", json.dumps(res["calibracion_nuevas"], ensure_ascii=False))
    print("calibración celdas (nuevas):", json.dumps(res["calibracion_nuevas_celdas"], ensure_ascii=False))
    print("minuto de llegada:", res["error_minuto_llegada"], res["error_minuto_llegada_celdas"])
    print("casos:", casos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
