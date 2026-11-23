#!/usr/bin/env python3
"""Idea 2: aviso de la localidad EXACTA solo para tormentas fuertes o con rayos.

Pregunta: ¿en los próximos 30 min pasa SOBRE la localidad una tormenta fuerte
(45 dBZ o más dentro de la localidad) o con rayos (rayos dentro de la localidad)?
Lo demás (lluvia, tormentas débiles sin rayos) lo sigue cubriendo la zona de 10 km.

Como en vivo: el radar llega con 10, 20 o 30 min de atraso; a la hora "ahora" solo
hay la imagen vieja, el satélite ya publicado y los rayos de los últimos 10 min.
Verdad: imágenes reales de ahora a +30 min (limpias de ecos falsos con el satélite)
y los rayos reales de ahora a +30 min.

Métodos comparados:
- "5b": el 5b en la localidad exacta (2 de 25 caminos + corrector), cualquier tormenta.
- "fuertes": celdas del radar de >= 35 dBZ que son fuertes (pico >= 45) o tienen rayos
  encima ahora, más las celdas de rayos nuevas (scripts/celdas_rayos.py, solo_fuertes_dbz).
- "rayos": solo las celdas de rayos.
- combinaciones.

Uso: python herramientas/simular_aviso_fuerte.py --goes /tmp/goeshist [--paso 2] [--salida x.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
UMBRAL_TORMENTA = 35.0
UMBRAL_FUERTE = 45.0
MARGEN_RAYO_KM = 4.0          # error de ubicación de un rayo del GLM (~8 km por píxel)
VARIANTES = {
    "fuertes": {"solo_fuertes_dbz": 45.0},
    "fuertes_margen6": {"solo_fuertes_dbz": 45.0, "margen_km": 6.0},
    "fuertes_mas_error": {"solo_fuertes_dbz": 45.0, "error_por_minuto_km": 0.4, "error_velocidad": 0.35,
                          "error_direccion_grados": 30.0},
    "fuertes_40": {"solo_fuertes_dbz": 40.0},
    "todas": {},
}
UMBRALES_P = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def metricas(pred, sub):
    a = b = c = 0
    for r in sub:
        pr, ob = pred(r), r["obs"]
        a += pr and ob
        b += pr and not ob
        c += (not pr) and ob
    return {"detectadas": a, "falsas_alarmas": b, "omitidas": c,
            "deteccion_pct": round(100 * a / (a + c), 1) if a + c else None,
            "falsas_alarmas_pct": round(100 * b / (a + b), 1) if a + b else None,
            "csi_pct": round(100 * a / (a + b + c), 1) if a + b + c else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goes", required=True)
    ap.add_argument("--paso", type=int, default=2)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--dias", default=None, help="solo imágenes de partida de estos días, ej.: 20261113,20261121")
    ap.add_argument("--maximo", type=int, default=0, help="cortar después de N casos (prueba)")
    args = ap.parse_args(argv)
    t_ini = time.time()
    goes = sh.cargar_goes(Path(args.goes))
    cfg = {**ncap.CONFIG_DEFECTO, **json.loads((ROOT / "config" / "nowcast_capas.json").read_text(encoding="utf-8"))}
    frames = {nf.timestamp_de_nombre(p): p for p in nf.HISTORICO.glob("20*.png")}
    frames.pop(None, None)
    dias = set(args.dias.split(",")) if args.dias else None
    shape = nf.cargar_dbz(next(iter(frames.values()))).shape
    h, w = shape
    clutter = nc.cargar_clutter_total(shape)
    km_px = nf.km_por_pixel(w)
    zonas = sl.zonas(shape, 0.0)
    locs_px, radios_km = {}, {}
    for loc in lr.cargar_localidades():
        x, y = nf.latlon_to_pixel(loc["lat"], loc["lon"], w, h)
        locs_px[loc["id"]] = (x, y, float(loc.get("radio_km", 4.0)) / km_px)
        radios_km[loc["id"]] = float(loc.get("radio_km", 4.0))
    glm_t = goes["glm_fin"]
    glm_x, glm_y = ct.a_pixel(goes["glm_lat"], goes["glm_lon"], w, h)
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

    def rayos_en_localidades(desde, hasta):
        a, b = np.datetime64(desde.replace(tzinfo=None)), np.datetime64(hasta.replace(tzinfo=None))
        sel = (glm_t > a) & (glm_t <= b)
        x, y = glm_x[sel], glm_y[sel]
        return {k: int((np.hypot(x - lx, y - ly) * km_px <= radios_km[k] + MARGEN_RAYO_KM).sum())
                for k, (lx, ly, _) in locs_px.items()}

    registros = []
    casos = 0
    for i, t0 in enumerate(sorted(frames)):
        if i % args.paso or (dias and t0.strftime("%Y%m%d") not in dias):
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
            rayos_fut = rayos_en_localidades(ahora - timedelta(minutes=1), ahora + timedelta(minutes=30))
            rayos_ya = rayos_en_localidades(ahora - timedelta(minutes=5), ahora)
            sat, rlat, rlon = sh.datos_reales(goes, ahora, shape)
            datos = {"satelite": sat, "rayos_lat": list(rlat), "rayos_lon": list(rlon), "info": {}}
            campos = {float(atraso + p): pila[float(atraso + p)][-2] for p in PLAZOS}
            corr = ncap.con_corrector(cfg, dict(campos), float(atraso), flujo, datos, None)
            base = {p: nc.picos_por_zona(corr[float(atraso + p)], zonas) for p in PLAZOS}
            hh = np.datetime64(ahora.replace(tzinfo=None))
            sel = (glm_t > hh - np.timedelta64(10, "m")) & (glm_t <= hh)
            edad = (hh - glm_t[sel]).astype("timedelta64[s]").astype(float) / 60.0
            lista = cr.celdas(glm_x[sel], glm_y[sel], edad, shape, km_px)
            probs = {var: cr.probabilidades_celdas(act_cc, u, v, km_px, float(atraso), lista, locs_px, par)
                     for var, par in VARIANTES.items()}
            probs["rayos"] = cr.probabilidades(lista, u, v, km_px, locs_px)
            for loc in locs_px:
                fuerte = any(o[loc] >= UMBRAL_FUERTE for o in obs.values())
                tormenta = any(o[loc] >= UMBRAL_TORMENTA for o in obs.values())
                electrica = rayos_fut[loc] > 0
                ya = (0 in obs and obs[0][loc] >= UMBRAL_FUERTE) or rayos_ya[loc] > 0
                registros.append({
                    "obs": fuerte or electrica, "obs_fuerte": fuerte, "obs_rayos": electrica,
                    "obs_tormenta": tormenta, "obs_ya": ya, "atraso": atraso, "loc": loc,
                    "base": any(base[p][loc] >= UMBRAL_TORMENTA for p in PLAZOS),
                    "p": {var: probs[var][loc]["probabilidad"] for var in probs},
                    "min": probs["fuertes"][loc]["minutos"],
                })
        if casos % 25 == 0:
            print(f"... {casos} casos, {len(registros)} registros, {time.time() - t_ini:.0f} s",
                  file=sys.stderr, flush=True)
        if args.maximo and casos >= args.maximo:
            break

    subconjuntos = {"todas": registros, "nuevas": [r for r in registros if not r["obs_ya"]]}
    res = {"casos": casos, "registros": len(registros),
           "eventos": {"fuerte_o_rayos": sum(r["obs"] for r in registros),
                       "fuerte": sum(r["obs_fuerte"] for r in registros),
                       "rayos": sum(r["obs_rayos"] for r in registros),
                       "tormenta_35": sum(r["obs_tormenta"] for r in registros),
                       "tormenta_35_sin_fuerte_ni_rayos": sum(r["obs_tormenta"] and not r["obs"] for r in registros)},
           "resultado": {}}
    for nombre, sub in subconjuntos.items():
        out = {"5b": metricas(lambda r: r["base"], sub)}
        for var in list(VARIANTES) + ["rayos"]:
            for u_ in UMBRALES_P:
                out[f"{var}>={u_}"] = metricas(lambda r, var=var, u_=u_: r["p"][var] >= u_, sub)
        for u_ in UMBRALES_P:
            out[f"5b_y_fuertes>={u_}"] = metricas(lambda r, u_=u_: r["base"] and r["p"]["fuertes"] >= u_, sub)
            out[f"5b_o_fuertes>={u_}"] = metricas(lambda r, u_=u_: r["base"] or r["p"]["fuertes"] >= u_, sub)
            out[f"fuertes>={u_}_o_rayos>=0.5"] = metricas(
                lambda r, u_=u_: r["p"]["fuertes"] >= u_ or r["p"]["rayos"] >= 0.5, sub)
        res["resultado"][nombre] = out
    for atraso in ATRASOS:
        sub = [r for r in registros if r["atraso"] == atraso]
        res.setdefault("por_atraso", {})[str(atraso)] = {
            k: metricas(f, sub) for k, f in (("fuertes>=0.3", lambda r: r["p"]["fuertes"] >= 0.3),
                                             ("fuertes>=0.5", lambda r: r["p"]["fuertes"] >= 0.5),
                                             ("5b", lambda r: r["base"]))}
    cal = defaultdict(lambda: [0, 0])
    for r in registros:
        if r["p"]["fuertes"] > 0:
            b = min(int(r["p"]["fuertes"] * 10), 9)
            cal[b][0] += 1
            cal[b][1] += r["obs"]
    res["calibracion_fuertes"] = {f"{10 * k}-{10 * k + 10}%": {"casos": n, "paso_pct": round(100 * s / n, 1)}
                                  for k, (n, s) in sorted(cal.items())}
    res["segundos"] = round(time.time() - t_ini)
    if args.salida:
        Path(args.salida).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        Path(args.salida).with_suffix(".registros.json").write_text(json.dumps(registros), encoding="utf-8")
    print("eventos:", res["eventos"])
    for nombre in subconjuntos:
        print(f"\n== {nombre}: detección / falsas alarmas / CSI")
        for k, m in res["resultado"][nombre].items():
            print(f"  {k:28s} {m['deteccion_pct']}/{m['falsas_alarmas_pct']}/{m['csi_pct']}  ({m['detectadas']}+{m['omitidas']})")
    print("\npor atraso:", json.dumps(res["por_atraso"], ensure_ascii=False))
    print("calibración:", json.dumps(res["calibracion_fuertes"], ensure_ascii=False))
    print("casos:", casos, "segundos:", res["segundos"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
