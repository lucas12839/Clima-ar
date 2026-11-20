#!/usr/bin/env python3
"""La pregunta útil: "¿llega (o hay) lluvia / tormenta en mi localidad en la próxima hora?".

Simula, con los 26 días guardados, que el radar tiene `atraso` minutos (10, 20 o 30): a la
hora actual solo se tiene la imagen vieja, el satélite ya publicado y los rayos de los
últimos 10 min. Se pronostica para cada localidad si en los próximos 60 min habrá eco
(>= 20 dBZ lluvia, >= 35 dBZ tormenta) y cuándo empieza. Se compara con lo que mostraron
las imágenes reales de esa hora (limpias de ecos falsos con el satélite).

Dos zonas: la localidad (círculo de cada una) y "con margen" (hasta 10 km alrededor).
Hora: acierto si la primera llegada pronosticada está a 15 min o menos de la real.

Uso: python herramientas/simular_llegadas.py --goes /tmp/goes [--paso 2] [--salida x.json]
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
import localidades_radar as lr  # noqa: E402
import nowcast_capas as ncap  # noqa: E402
import nowcast_capas_nucleo as nc  # noqa: E402
import nowcast_flujo_optico as nf  # noqa: E402
import simular_huecos as sh  # noqa: E402

ATRASOS = (10, 20, 30)
PLAZOS = (0, 10, 20, 30, 40, 50, 60)    # minutos desde ahora
UMBRALES = {"lluvia": 20.0, "tormenta": 35.0}
METODOS = ("hoy", "capas", "capas_corrector")
TOLERANCIA_HORA_MIN = 15


def zonas(shape, margen_km: float) -> dict[str, np.ndarray]:
    km_px = nf.km_por_pixel(shape[1])
    salida = {}
    for loc in lr.cargar_localidades():
        x, y = nf.latlon_to_pixel(loc["lat"], loc["lon"], shape[1], shape[0])
        radio = float(loc.get("radio_km", 4.0)) + margen_km
        salida[loc["id"]] = nf.disco(shape, x, y, radio / km_px)
    return salida


def primera(serie: dict[int, dict[str, float]], loc: str, umbral: float):
    for p in PLAZOS:
        if p in serie and serie[p][loc] >= umbral:
            return p
    return None


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
    clutter = nc.cargar_clutter_total(shape)
    km_px = nf.km_por_pixel(shape[1])
    z_loc, z_margen = zonas(shape, 0.0), zonas(shape, 10.0)
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

    # tabla[metodo][atraso][zona][umbral] = [si-si, si-no, no-si, no-no, hora_ok, hora_total]
    tabla = {m: {a: {z: {u: [0] * 6 for u in UMBRALES} for z in ("localidad", "margen_10km")}
                 for a in ATRASOS} for m in METODOS}
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
        for atraso in ATRASOS:
            ahora = t0 + timedelta(minutes=atraso)
            reales = {p: verdad(ahora + timedelta(minutes=p)) for p in PLAZOS}
            if sum(r is not None for r in reales.values()) < 6:
                continue
            tiempos = [atraso + p for p in PLAZOS]
            esc = nc.campos_escenarios(act_cc, u, v, tiempos)
            sat, rlat, rlon = sh.datos_reales(goes, ahora, shape)
            datos = {"satelite": sat, "rayos_lat": list(rlat), "rayos_lon": list(rlon), "info": {}}
            corr = ncap.con_corrector(cfg, dict(esc), float(atraso), flujo, datos, None)
            pron = {"hoy": {p: act for p in PLAZOS},
                    "capas": {p: esc[float(atraso + p)] for p in PLAZOS},
                    "capas_corrector": {p: corr[float(atraso + p)] for p in PLAZOS}}
            for nombre_z, zz in (("localidad", z_loc), ("margen_10km", z_margen)):
                obs = {p: nc.picos_por_zona(r, zz) for p, r in reales.items() if r is not None}
                for m, serie in pron.items():
                    picos = {p: nc.picos_por_zona(c, zz) for p, c in serie.items()}
                    for nombre_u, umbral in UMBRALES.items():
                        acc = tabla[m][atraso][nombre_z][nombre_u]
                        for loc in zz:
                            fp, fo = primera(picos, loc, umbral), primera(obs, loc, umbral)
                            si_p, si_o = fp is not None, fo is not None
                            acc[0 if si_p and si_o else 1 if si_p else 2 if si_o else 3] += 1
                            if si_p and si_o:
                                acc[5] += 1
                                acc[4] += abs(fp - fo) <= TOLERANCIA_HORA_MIN
        if casos % 50 == 0:
            print(f"... {casos} casos", file=sys.stderr, flush=True)

    def met(acc):
        a, b, c, d, hok, ht = acc
        return {"llego_y_aviso": a, "aviso_y_no_llego": b, "llego_sin_aviso": c, "ni_aviso_ni_llego": d,
                "deteccion_pct": round(100 * a / (a + c), 1) if a + c else None,
                "falsas_alarmas_pct": round(100 * b / (a + b), 1) if a + b else None,
                "acierto_total_pct": round(100 * (a + d) / (a + b + c + d), 1) if a + b + c + d else None,
                "csi_pct": round(100 * a / (a + b + c), 1) if a + b + c else None,
                "hora_acertada_pct": round(100 * hok / ht, 1) if ht else None}

    res = {"casos": casos, "resultado": {m: {f"atraso_{a}min": {z: {u: met(tabla[m][a][z][u]) for u in UMBRALES}
                                                                 for z in ("localidad", "margen_10km")}
                                             for a in ATRASOS} for m in METODOS}}
    if args.salida:
        Path(args.salida).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for z in ("localidad", "margen_10km"):
        for u in UMBRALES:
            print(f"\n{u} en la próxima hora, zona {z}:  detección / falsas alarmas / CSI / acierto total / hora ±15")
            for m in METODOS:
                fila = []
                for a in ATRASOS:
                    r = res["resultado"][m][f"atraso_{a}min"][z][u]
                    fila.append(f"{a}min: {r['deteccion_pct']}/{r['falsas_alarmas_pct']}/{r['csi_pct']}/"
                                f"{r['acierto_total_pct']}/{r['hora_acertada_pct']}")
                print(f"  {m:16s} " + "   ".join(fila))
    print("casos:", casos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
