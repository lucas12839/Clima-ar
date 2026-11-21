#!/usr/bin/env python3
"""Etapa A: ¿el satélite (y las estaciones) compensan el atraso del radar?

Como en vivo: a la hora "ahora" la última imagen de radar es de `atraso` minutos antes
(10, 20 o 30). Con eso y con lo que YA estaría publicado a esa hora (barridos del canal
13 de GOES-19 que se publican ~10 min después de empezar, rayos de los últimos 20 min,
estaciones) se pronostica qué hay ahora y dentro de 10, 20, 30 y 60 min. Se compara con
la imagen de radar real de cada hora (limpia de ecos falsos con el satélite, como en
simular_huecos.py).

Métodos:
- produccion: lo que corre hoy (escenarios + corrector con rayos y satélite).
- variantes de scripts/corrector_satelite.py (posición, movimiento, crecimiento) y del
  corrector con estaciones (scripts/corrector_estaciones.py), siempre con el mismo
  corrector de rayos y satélite al final.

Uso: python herramientas/simular_etapa_a.py --goes CARPETA [--paso 2] [--procesos 2]
         [--variantes JSON] [--estaciones lluvia.json] [--salida x.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts", ROOT / "herramientas"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import corrector_satelite as cs  # noqa: E402
import corrector_tiempo_real as ct  # noqa: E402
import nowcast_capas as capas  # noqa: E402
import nowcast_capas_nucleo as nc  # noqa: E402
import nowcast_flujo_optico as nf  # noqa: E402
import simular_huecos as sh  # noqa: E402

ATRASOS = (10, 20, 30)
PLAZOS = (0, 10, 20, 30, 60)
UMBRALES = (20.0, 35.0)
MIN_PIX_ECO = 300

VARIANTES_DEFECTO = {
    "posicion": {"peso_posicion": 1.0},
    "posicion_05": {"peso_posicion": 0.5},
    "posicion_40": {"peso_posicion": 1.0, "tope_c": -40.0},
    "movimiento": {"peso_posicion": 0.0, "peso_movimiento": 0.5},
    "pos_mov": {"peso_posicion": 1.0, "peso_movimiento": 0.5},
    "crece_05": {"peso_posicion": 0.0, "dbz_por_c": 0.5},
    "crece_1": {"peso_posicion": 0.0, "dbz_por_c": 1.0},
    "rayos_2": {"peso_posicion": 0.0, "dbz_por_rayo_min": 2.0},
}

G: dict = {}


def dt64(x: datetime):
    return np.datetime64(x.replace(tzinfo=None))


def barrido(goes, inicio_aprox, tolerancia_min=4.0):
    """Índice del barrido que empieza más cerca de `inicio_aprox` (o None)."""
    ini = goes["inicios"]
    i = int(np.argmin(np.abs(ini - dt64(inicio_aprox))))
    if abs((ini[i] - dt64(inicio_aprox)) / np.timedelta64(1, "s")) > tolerancia_min * 60:
        return None
    return i


def grilla(goes, i, shape):
    if i is None:
        return None
    c = goes["cache"]
    if i not in c:
        if len(c) > 30:
            c.pop(next(iter(c)))
        c[i] = ct.grilla_satelite(goes["lat"], goes["lon"], goes["valores"][i].astype(np.float32), shape)
    return c[i]


def satelites_para(goes, t0: datetime, ahora: datetime, shape):
    """Barrido de la hora del radar, el más nuevo publicado a `ahora` y el anterior a ese."""
    ini = goes["inicios"]
    i_rad = barrido(goes, t0 - timedelta(minutes=cs.VE_ZONA_MIN))
    lim = dt64(ahora - timedelta(minutes=cs.PUBLICADO_MIN))
    i_new = int(np.searchsorted(ini, lim, side="right")) - 1
    if i_new < 0 or (dt64(ahora) - ini[i_new]) > np.timedelta64(30, "m"):
        return None
    t_new = datetime.fromisoformat(str(ini[i_new])[:19])
    i_prev = barrido(goes, t_new - timedelta(minutes=10), 2.0)
    tramo = (t_new + timedelta(minutes=cs.VE_ZONA_MIN) - t0.replace(tzinfo=None)).total_seconds() / 60.0
    entre = None if i_prev is None else (ini[i_new] - ini[i_prev]) / np.timedelta64(1, "s") / 60.0
    return {"radar": grilla(goes, i_rad, shape), "nuevo": grilla(goes, i_new, shape),
            "previo": grilla(goes, i_prev, shape), "tramo": tramo, "entre": entre}


def rayos_en(goes, desde: datetime, hasta: datetime):
    f = goes["glm_fin"]
    sel = (f > dt64(desde)) & (f <= dt64(hasta))
    return goes["glm_lat"][sel], goes["glm_lon"][sel]


def nueva_tabla(metodos):
    return {m: {a: {p: {u: [0, 0, 0, 0] for u in UMBRALES} for p in PLAZOS} for a in ATRASOS} for m in metodos}


def sumar(acc_u, pron: dict, obs: dict):
    for u, acc in acc_u.items():
        for k in obs:
            a, b = pron[k] >= u, obs[k] >= u
            acc[0 if a and b else 1 if a else 2 if b else 3] += 1


def procesar_dia(dia: str) -> dict:
    goes, frames, variantes, cfg, est = G["goes"], G["frames"], G["variantes"], G["cfg"], G.get("estaciones")
    metodos = ["hoy", "produccion", *variantes]
    tab_l, tab_r = nueva_tabla(metodos), nueva_tabla(metodos)
    clutter = G["clutter"]
    cache: dict = {}

    def campo(t):
        if t not in frames:
            return None
        if t not in cache:
            d = nf.cargar_dbz(frames[t])
            cache[t] = None if d is None else nc.campo_limpio(d, clutter)
            if len(cache) > 20:
                cache.pop(next(iter(cache)))
        return cache[t]

    zl, zr = G["zl"], G["zr"]
    km_px = G["km_px"]
    casos = 0
    usos = {m: 0 for m in variantes}
    t0s = sorted(t for t in frames if t.strftime("%Y%m%d") == dia)
    for i, t0 in enumerate(t0s):
        if i % G["paso"]:
            continue
        act, prev = campo(t0), campo(t0 - timedelta(minutes=10))
        if act is None or prev is None or int((act >= nf.UMBRAL_ECO_DBZ).sum()) < MIN_PIX_ECO:
            continue
        f1 = nf.estimar_flujo(prev, act, 10.0, km_px)
        if not f1["valido"]:
            continue
        pp = campo(t0 - timedelta(minutes=20))
        f2 = nf.estimar_flujo(pp, prev, 10.0, km_px) if pp is not None else None
        u, v = sh.flujo_promediado(f1, f2)
        flujo = {"u": u, "v": v}
        casos += 1
        for atraso in ATRASOS:
            ahora = t0 + timedelta(minutes=atraso)
            sats = satelites_para(goes, t0, ahora, act.shape)
            if sats is None:
                continue
            rlat, rlon = rayos_en(goes, ahora - timedelta(minutes=10), ahora)
            rayos_prev = rayos_en(goes, ahora - timedelta(minutes=20), ahora - timedelta(minutes=10))
            datos = {"satelite": sats["nuevo"], "rayos_lat": list(rlat), "rayos_lon": list(rlon), "info": {}}
            objetivos = {p: ahora + timedelta(minutes=p) for p in PLAZOS if (ahora + timedelta(minutes=p)) in frames}
            if not objetivos:
                continue
            leads = [float(atraso + p) for p in objetivos]
            pron = {"hoy": {L: act for L in leads},
                    "produccion": nc.campos_escenarios(act, u, v, leads)}
            cambio_r = None
            for nombre, par in variantes.items():
                if par.get("sin_movimiento_radar"):
                    # Caso real: falta la imagen anterior (SINARAME saltea imágenes) y el radar
                    # no da movimiento. Hoy: persistencia. Con satélite: movimiento de los topes.
                    fs = None
                    if par.get("sin_movimiento_radar") == "satelite":
                        pp = {**cs.PARAMETROS_DEFECTO, **par}
                        fs = cs.flujo_satelite(sats["previo"], sats["nuevo"], sats["entre"], km_px, pp)
                        if fs is None:
                            fs = cs.flujo_satelite(sats["radar"], sats["nuevo"], sats["tramo"], km_px, pp)
                    pron[nombre] = nc.campos_escenarios(act, fs["u"], fs["v"], leads) if fs is not None \
                        else {L: act for L in leads}
                    usos[nombre] += fs is not None
                    continue
                if par.get("estaciones"):
                    import corrector_estaciones as ce
                    base = pron["produccion"]
                    ie: dict = {}
                    pron[nombre] = ce.corregir_campos(base, est, ahora, float(atraso), flujo, act.shape, par, act, ie) \
                        if est is not None else base
                    usos[nombre] += len(ie.get("agregadas") or []) + len(ie.get("bajadas") or [])
                    continue
                if par.get("dbz_por_rayo_min") and cambio_r is None:
                    cambio_r = cs.cambio_rayos((rlat, rlon), rayos_prev, act.shape)
                r_arg = cambio_r
                if par.get("usar_rayos"):
                    l0 = rayos_en(goes, t0 - timedelta(minutes=5), t0 + timedelta(minutes=5))
                    r_arg = {"lat0": l0[0], "lon0": l0[1], "lat1": rlat, "lon1": rlon, "minutos": atraso - 5.0}
                info: dict = {}
                pron[nombre] = cs.pronostico(act, u, v, leads, sats["radar"], sats["nuevo"], sats["previo"],
                                             sats["tramo"], sats["entre"], km_px, par, r_arg, True, info)
                usos[nombre] += bool(info.get("posicion") or info.get("movimiento") or info.get("crecimiento"))
            for m in metodos:
                if m != "hoy":
                    pron[m] = capas.con_corrector(cfg, pron[m], float(atraso), flujo, datos, None)
            for p, tg in objetivos.items():
                obs = campo(tg)
                if obs is None:
                    continue
                s_obs = sh.satelite_cercano(goes, tg, obs.shape)
                if s_obs is not None:
                    obs = ct.quitar_ecos_sin_nube(obs, s_obs)
                o_l, o_r = nc.picos_por_zona(obs, zl), nc.picos_por_zona(obs, zr)
                L = float(atraso + p)
                for m in metodos:
                    c = pron[m][L]
                    sumar(tab_l[m][atraso][p], nc.picos_por_zona(c, zl), o_l)
                    sumar(tab_r[m][atraso][p], nc.picos_por_zona(c, zr), o_r)
    return {"dia": dia, "casos": casos, "loc": tab_l, "reg": tab_r, "usos": usos}


def iniciar(args_dict):
    G.update(args_dict)
    G["goes"] = sh.cargar_goes(Path(G["goes_dir"]))
    frames = {}
    for p in nf.HISTORICO.glob("20*.png"):
        dt = nf.timestamp_de_nombre(p)
        if dt is not None:
            frames[dt] = p
    G["frames"] = frames
    shape = nf.cargar_dbz(next(iter(frames.values()))).shape
    G["clutter"] = nc.cargar_clutter_total(shape)
    G["km_px"] = nf.km_por_pixel(shape[1])
    G["zl"], G["zr"] = sh.zonas_localidades(shape), sh.zonas_region(shape)
    G["cfg"] = {**capas.CONFIG_DEFECTO, **capas.leer_json(capas.CONFIG)}
    if G.get("estaciones_path"):
        G["estaciones"] = json.loads(Path(G["estaciones_path"]).read_text(encoding="utf-8"))


def csi(acc):
    a, b, c, _ = acc
    return a / (a + b + c) if a + b + c else None


def imprimir(tot: dict, metodos, salida: dict) -> None:
    for z in ("loc", "reg"):
        for u in UMBRALES:
            print(f"\n{z} {int(u)} dBZ CSI  (atraso/plazo)")
            cab = [f"{at}/{p}" for at in ATRASOS for p in PLAZOS]
            print(f"{'':14s}" + " ".join(f"{c:>6}" for c in cab))
            for m in metodos:
                fila = [csi(tot[z][m][at][p][u]) for at in ATRASOS for p in PLAZOS]
                salida["csi"].setdefault(z, {}).setdefault(str(int(u)), {})[m] = \
                    {c: (None if x is None else round(x, 3)) for c, x in zip(cab, fila)}
                print(f"{m[:14]:14s}" + " ".join(f"{(x or 0):6.3f}" for x in fila))


def sumar_salidas(archivos, destino) -> int:
    tot, casos, usos, metodos = None, 0, {}, None
    for f in archivos:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if not d.get("tablas"):
            continue
        casos += d["casos"]
        for m, n in d["usos"].items():
            usos[m] = usos.get(m, 0) + n
        t = d["tablas"]
        if tot is None:
            tot, metodos = t, list(t["loc"])
            continue
        for z in ("loc", "reg"):
            for m in metodos:
                for at in t[z][m]:
                    for p in t[z][m][at]:
                        for u in t[z][m][at][p]:
                            for k in range(4):
                                tot[z][m][at][p][u][k] += t[z][m][at][p][u][k]
    # claves de JSON (texto) -> números, como en la corrida
    tot = {z: {m: {int(at): {int(p): {float(u): acc for u, acc in pp.items()} for p, pp in aa.items()}
                   for at, aa in mm.items()} for m, mm in zz.items()} for z, zz in tot.items()}
    salida = {"casos": casos, "usos": usos, "csi": {}}
    imprimir(tot, metodos, salida)
    print(f"\ncasos {casos}, usos {usos}")
    if destino:
        Path(destino).write_text(json.dumps(salida, ensure_ascii=False), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goes", required=True)
    ap.add_argument("--paso", type=int, default=2)
    ap.add_argument("--procesos", type=int, default=2)
    ap.add_argument("--dias", default=None, help="AAAAMMDD,AAAAMMDD (por defecto: los días con satélite)")
    ap.add_argument("--variantes", default=None)
    ap.add_argument("--estaciones", default=None)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--sumar", nargs="*", default=None, help="junta salidas (--salida) de varias corridas")
    a = ap.parse_args(argv)
    if a.sumar:
        return sumar_salidas(a.sumar, a.salida)
    variantes = json.loads(a.variantes) if a.variantes else VARIANTES_DEFECTO
    dias = a.dias.split(",") if a.dias else sorted(p.stem.split("_")[1] for p in Path(a.goes).rglob("goes_*.npz"))
    conf = {"goes_dir": a.goes, "variantes": variantes, "paso": a.paso, "estaciones_path": a.estaciones}
    t = time.time()
    if a.procesos > 1:
        with Pool(a.procesos, initializer=iniciar, initargs=(conf,)) as pool:
            res = pool.map(procesar_dia, dias)
    else:
        iniciar(conf)
        res = [procesar_dia(d) for d in dias]
    metodos = ["hoy", "produccion", *variantes]
    tot = {z: nueva_tabla(metodos) for z in ("loc", "reg")}
    casos = 0
    usos = {m: 0 for m in variantes}
    for r in res:
        casos += r["casos"]
        for m in variantes:
            usos[m] += r["usos"][m]
        for z in ("loc", "reg"):
            for m in metodos:
                for at in ATRASOS:
                    for p in PLAZOS:
                        for u in UMBRALES:
                            acc = tot[z][m][at][p][u]
                            src = r[z][m][at][p][u]
                            for k in range(4):
                                acc[k] += src[k]
    salida = {"casos": casos, "dias": dias, "segundos": round(time.time() - t), "usos": usos, "csi": {}}
    imprimir(tot, metodos, salida)
    salida["tablas"] = tot if a.salida else None
    print(f"\ncasos {casos}, {salida['segundos']} s, usos {usos}")
    if a.salida:
        Path(a.salida).write_text(json.dumps(salida, ensure_ascii=False, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
