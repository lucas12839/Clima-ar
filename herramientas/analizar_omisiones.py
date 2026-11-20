#!/usr/bin/env python3
"""Punto 5b: ¿por qué se escapan las tormentas de la próxima hora?

Repite la simulación de `simular_llegadas.py` (26 días, radar atrasado 10/20/30 min, satélite
ya publicado, rayos de los últimos 10 min) y guarda, para cada localidad y cada caso, lo que se
sabía en ese momento:

- radar viejo: dBZ máximo a 15, 30 y 50 km de la localidad y a 30 km "de donde viene el viento"
  (30 min antes en la trayectoria);
- satélite: tope más frío a 15/30/50 km, y enfriamiento "a prueba de movimiento" (UW-CI: cada
  píxel contra el más frío que había a 12 km en la imagen de 10 min antes y a 20 km en la de 20);
- rayos: cantidad a 15/30/50 km en los últimos 10 min y en los 10 min anteriores (salto);
- lo que pronosticó el nowcast en capas con corrector y lo que pasó (primera llegada).

Salida: un .npz con una fila por (caso, atraso, localidad). Sirve para clasificar las omisiones
(¿nacieron cerca o venían de lejos?) y para probar reglas nuevas sin repetir la simulación.

Uso: python herramientas/analizar_omisiones.py --goes /tmp/goes --salida /tmp/filas.npz [--paso 2]
"""
from __future__ import annotations

import argparse
import json
import math
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
import simular_llegadas as sl  # noqa: E402

RADIOS = (15, 30, 50)


def maximo_cerca(grilla: np.ndarray, radio_km: float) -> np.ndarray:
    return -ct.mas_frio_cerca(-np.nan_to_num(grilla, nan=-1e9), radio_km)


def enfriamiento(sat: np.ndarray | None, sat10: np.ndarray | None, sat20: np.ndarray | None) -> np.ndarray | None:
    """°C que el tope se enfrió de verdad (no por moverse), solo donde el tope ya está bajo 0 °C."""
    if sat is None or sat10 is None:
        return None
    e = ct.mas_frio_cerca(sat10, 12.0) - sat
    if sat20 is not None:
        # con 20 min: el más frío a 20 km; vale el mayor de los dos (crecimiento sostenido o rápido)
        e = np.fmax(e, ct.mas_frio_cerca(sat20, 20.0) - sat)
    e = np.where(np.isfinite(e) & (sat <= 0.0), e, 0.0)
    return np.clip(e, 0.0, 40.0).astype(np.float32)


def rayos_cerca(lat, lon, plat, plon, radio_km: float) -> int:
    if not len(lat):
        return 0
    dx = (np.asarray(lon) - plon) * 111.320 * math.cos(math.radians(plat))
    dy = (np.asarray(lat) - plat) * 111.195
    return int((dx * dx + dy * dy <= radio_km * radio_km).sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goes", required=True)
    ap.add_argument("--paso", type=int, default=2)
    ap.add_argument("--salida", required=True)
    args = ap.parse_args(argv)
    goes = sh.cargar_goes(Path(args.goes))
    cfg = {**ncap.CONFIG_DEFECTO, **json.loads((ROOT / "config" / "nowcast_capas.json").read_text(encoding="utf-8"))}
    frames = {nf.timestamp_de_nombre(p): p for p in nf.HISTORICO.glob("20*.png")}
    frames.pop(None, None)
    shape = nf.cargar_dbz(next(iter(frames.values()))).shape
    h, w = shape
    clutter = nc.cargar_clutter_total(shape)
    km_px = nf.km_por_pixel(w)
    locs = lr.cargar_localidades()
    pix = {l["id"]: nf.latlon_to_pixel(l["lat"], l["lon"], w, h) for l in locs}
    z_margen = sl.zonas(shape, 10.0)
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

    filas: dict[str, list] = {}

    def agregar(**kw):
        for k, v in kw.items():
            filas.setdefault(k, []).append(v)

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
        rad_max = {r: maximo_cerca(act_cc, r) for r in RADIOS}
        casos += 1
        for atraso in sl.ATRASOS:
            ahora = t0 + timedelta(minutes=atraso)
            reales = {p: verdad(ahora + timedelta(minutes=p)) for p in sl.PLAZOS}
            if sum(r is not None for r in reales.values()) < 6:
                continue
            tiempos = [atraso + p for p in sl.PLAZOS]
            esc = nc.campos_escenarios(act_cc, u, v, tiempos)
            sat, rlat, rlon = sh.datos_reales(goes, ahora, shape)
            sat10, plat, plon = sh.datos_reales(goes, ahora - timedelta(minutes=10), shape)
            sat20, _, _ = sh.datos_reales(goes, ahora - timedelta(minutes=20), shape)
            if sat10 is sat:
                sat10 = None
            datos = {"satelite": sat, "rayos_lat": list(rlat), "rayos_lon": list(rlon), "info": {}}
            corr = ncap.con_corrector(cfg, dict(esc), float(atraso), flujo, datos, None)
            enf = enfriamiento(sat, sat10, sat20)
            sat_min = {r: (ct.mas_frio_cerca(sat, r) if sat is not None else None) for r in RADIOS}
            enf_max = {r: (maximo_cerca(enf, r) if enf is not None else None) for r in RADIOS}
            obs = {p: nc.picos_por_zona(r, z_margen) for p, r in reales.items() if r is not None}
            picos = {p: nc.picos_por_zona(corr[float(atraso + p)], z_margen) for p in sl.PLAZOS}
            for loc in locs:
                lid = loc["id"]
                x, y = pix[lid]
                xi, yi = int(round(x)), int(round(y))
                uu, vv = float(u[yi, xi]), float(v[yi, xi])
                xu = int(np.clip(round(x - uu * 30), 0, w - 1))
                yu = int(np.clip(round(y - vv * 30), 0, h - 1))
                fila = {"t0": str(t0), "atraso": atraso, "loc": lid, "hora_local": (ahora.hour - 3) % 24,
                        "vel_kmh": math.hypot(uu, vv) * km_px * 60,
                        "obs_t": sl.primera(obs, lid, 35.0), "obs_l": sl.primera(obs, lid, 20.0),
                        "pro_t": sl.primera(picos, lid, 35.0), "pro_l": sl.primera(picos, lid, 20.0),
                        "obs_max": max(o[lid] for o in obs.values()),
                        "rad_up30": float(rad_max[30][yu, xu])}
                for r in RADIOS:
                    fila[f"rad{r}"] = float(rad_max[r][yi, xi])
                    fila[f"sat{r}"] = float(sat_min[r][yi, xi]) if sat_min[r] is not None else np.nan
                    fila[f"enf{r}"] = float(enf_max[r][yi, xi]) if enf_max[r] is not None else np.nan
                    fila[f"ray{r}"] = rayos_cerca(rlat, rlon, loc["lat"], loc["lon"], r)
                    fila[f"rayp{r}"] = rayos_cerca(plat, plon, loc["lat"], loc["lon"], r)
                fila["sat_up30"] = float(sat_min[30][yu, xu]) if sat_min[30] is not None else np.nan
                agregar(**{k: (-1 if v is None else v) for k, v in fila.items()})
        if casos % 50 == 0:
            print(f"... {casos} casos", file=sys.stderr, flush=True)
    np.savez_compressed(args.salida, **{k: np.array(v) for k, v in filas.items()})
    print("casos:", casos, "filas:", len(filas.get("loc", [])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
