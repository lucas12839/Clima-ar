#!/usr/bin/env python3
"""Punto 10 del plan (etapa A, punto 5): ¿qué muestra ClimaAR si el radar se corta?

Con tormentas reales: en cada imagen de radar se hace de cuenta que el radar dejó de llegar
hace `corte` minutos (40, 60, 90) y se arma lo que ClimaAR mostraría con lo que SÍ estaría
publicado a esa hora: el último barrido de GOES-19 (topes por localidad, como
goes19/topes_localidad.py) y los rayos de los últimos 20 min (como goes19/glm_ventana.py).
Se compara con lo que mostraba el radar real a esa hora (la verdad).

Métodos para la lluvia de cada localidad:
- hoy: reglas actuales sin radar (scripts/indices_fenomenos.py: tope <= -50 °C = "probable").
- radar_viejo: la última imagen movida con el nowcast (+ corrector con rayos y satélite).
- variantes de reglas por satélite y rayos (para mejorar las de hoy).

Uso: python herramientas/simular_sin_radar.py --goes CARPETA [--paso 1] [--procesos 2]
         [--dias A,B] [--salida x.json] [--sumar a.json b.json ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts", ROOT / "herramientas"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import corrector_tiempo_real as ct  # noqa: E402
import indices_fenomenos as fen  # noqa: E402
import localidades_radar as lr  # noqa: E402
import nowcast_capas as capas  # noqa: E402
import nowcast_capas_nucleo as nc  # noqa: E402
import nowcast_flujo_optico as nf  # noqa: E402
import simular_huecos as sh  # noqa: E402
from goes19 import glm_ventana, topes_localidad  # noqa: E402

CORTES = (40, 60, 90)
PUBLICADO_MIN = 10.0
K = 273.15
G: dict = {}


def dt64(x):
    return np.datetime64(x.replace(tzinfo=None))


def satelite_para(goes, ahora, locs):
    """Topes por localidad del barrido más nuevo publicado a `ahora` (y enfriamiento)."""
    ini = goes["inicios"]
    i = int(np.searchsorted(ini, dt64(ahora - timedelta(minutes=PUBLICADO_MIN)), side="right")) - 1
    if i < 0 or (dt64(ahora) - ini[i]) > np.timedelta64(30, "m"):
        return None
    act = topes_localidad.estadisticas(goes["valores"][i].astype(np.float64) + K, goes["lat"], goes["lon"], locs)
    j = i - 1
    if j >= 0:
        minutos = (ini[i] - ini[j]) / np.timedelta64(1, "s") / 60.0
        prev = topes_localidad.estadisticas(goes["valores"][j].astype(np.float64) + K, goes["lat"], goes["lon"], locs)
        act = topes_localidad.con_enfriamiento(act, prev, minutos)
    return act


def rayos_para(goes, ahora, locs):
    f = goes["glm_fin"]
    sel = (f > dt64(ahora - timedelta(minutes=20))) & (f <= dt64(ahora))
    edad = (dt64(ahora) - f[sel]) / np.timedelta64(1, "s") / 60.0
    return glm_ventana.resumir(goes["glm_lat"][sel], goes["glm_lon"][sel], edad, locs, 20.0)["por_localidad"]


def grado_regla(sat: dict, ray: dict, regla: dict) -> int:
    """Lluvia sin radar con una regla de satélite y rayos (0 nada ... 3 fuerte)."""
    tope = sat.get("min_c_15km")
    p10 = sat.get("p10_c_15km")
    tasa = ray.get("tasa_30km_por_min") or 0.0
    r15 = ray.get("rayos_15km_10min") or 0
    g = 0
    if regla.get("tope_debil") is not None and p10 is not None and p10 <= regla["tope_debil"]:
        g = 1
    if tope is not None and tope <= regla.get("tope_probable", -50.0):
        g = max(g, regla.get("grado_probable", 2))
    if regla.get("rayos_moderada") is not None and r15 >= regla["rayos_moderada"]:
        g = max(g, 2)
    if regla.get("rayos_fuerte") is not None and tasa >= regla["rayos_fuerte"] and r15 >= 1:
        g = max(g, 3)
    return g


REGLAS = {
    "tope50_r15": {"tope_probable": -50.0, "rayos_moderada": 1},
    "tope40": {"tope_probable": -40.0},
    "tope40_debil30": {"tope_probable": -40.0, "tope_debil": -30.0},
    "tope45_debil35_rayos": {"tope_probable": -45.0, "tope_debil": -35.0, "rayos_moderada": 1, "rayos_fuerte": 2.0},
    "debil30_tope50_rayos": {"tope_probable": -50.0, "tope_debil": -30.0, "rayos_moderada": 1, "rayos_fuerte": 2.0},
    "debil40_tope50_r15": {"tope_probable": -50.0, "tope_debil": -40.0, "rayos_moderada": 1},
    "tope55_r15": {"tope_probable": -55.0, "rayos_moderada": 1},
    "tope50_r30": {"tope_probable": -50.0, "rayos_moderada": 3},
    "tope55": {"tope_probable": -55.0},
    "tope60_r15": {"tope_probable": -60.0, "rayos_moderada": 1},
    "tope55_r15_debil35": {"tope_probable": -55.0, "rayos_moderada": 1, "tope_debil": -35.0},
    "tope55_r15_debil45": {"tope_probable": -55.0, "rayos_moderada": 1, "tope_debil": -45.0},
}
COMBINADOS = ("viejo_o_tope50_r15", "viejo_o_tope55_r15", "viejo_o_tope55", "viejo_o_tope60_r15")


def nueva():
    return {m: {c: {"hay": [0, 0, 0, 0], "moderada": [0, 0, 0, 0], "grado": np.zeros((5, 5), int).tolist()}
                for c in CORTES} for m in ["hoy", "radar_viejo", *REGLAS, *COMBINADOS]}


def sumar(acc, pred: int, obs: int):
    for k, u in (("hay", 1), ("moderada", 2)):
        a, b = pred >= u, obs >= u
        acc[k][0 if a and b else 1 if a else 2 if b else 3] += 1
    acc["grado"][min(obs, 4)][min(pred, 4)] += 1


def procesar_dia(dia: str) -> dict:
    goes, frames, locs, clutter, cfg = G["goes"], G["frames"], G["locs"], G["clutter"], G["cfg"]
    km_px = G["km_px"]
    tab = nueva()
    cache: dict = {}

    def campo(t):
        if t not in frames:
            return None
        if t not in cache:
            d = nf.cargar_dbz(frames[t])
            cache[t] = None if d is None else nc.campo_limpio(d, clutter)
            if len(cache) > 30:
                cache.pop(next(iter(cache)))
        return cache[t]

    casos = 0
    for i, t in enumerate(sorted(x for x in frames if x.strftime("%Y%m%d") == dia)):
        if i % G["paso"]:
            continue
        obs = campo(t)
        if obs is None:
            continue
        s_obs = sh.satelite_cercano(goes, t, obs.shape)
        if s_obs is not None:
            obs = ct.quitar_ecos_sin_nube(obs, s_obs)
        verdad = {k: fen.lluvia_por_radar(v["dbz_p75"], mm_h_area=v["mm_h_medio"])["grado"]
                  for k, v in lr.observar(obs, locs).items()}
        sat = satelite_para(goes, t, locs)
        if sat is None:
            continue
        ray = rayos_para(goes, t, locs)
        casos += 1
        for corte in CORTES:
            # Lo de hoy: reglas sin radar (topes y rayos).
            pred = {}
            for loc in locs:
                lid = loc["id"]
                ev = fen.evaluar(radar={"disponible": False}, rayos=ray.get(lid) or {}, satelite=sat.get(lid) or {})
                pred.setdefault("hoy", {})[lid] = ev["ahora"]["lluvia"]["grado"]
                for nombre, regla in REGLAS.items():
                    pred.setdefault(nombre, {})[lid] = grado_regla(sat.get(lid) or {}, ray.get(lid) or {}, regla)
            # La última imagen de radar, movida.
            t0 = t - timedelta(minutes=corte)
            act, prev = campo(t0), campo(t0 - timedelta(minutes=10))
            pv = None
            if act is not None and prev is not None:
                f1 = nf.estimar_flujo(prev, act, 10.0, km_px)
                if f1["valido"]:
                    pp = campo(t0 - timedelta(minutes=20))
                    f2 = nf.estimar_flujo(pp, prev, 10.0, km_px) if pp is not None else None
                    u, v = sh.flujo_promediado(f1, f2)
                    c = nc.campos_escenarios(act, u, v, [float(corte)])
                    gi = int(np.searchsorted(goes["inicios"], dt64(t - timedelta(minutes=PUBLICADO_MIN)), side="right")) - 1
                    sgr = ct.grilla_satelite(goes["lat"], goes["lon"], goes["valores"][gi].astype(np.float32), act.shape)
                    f = goes["glm_fin"]
                    sel = (f > dt64(t - timedelta(minutes=10))) & (f <= dt64(t))
                    datos = {"satelite": sgr, "rayos_lat": list(goes["glm_lat"][sel]),
                             "rayos_lon": list(goes["glm_lon"][sel]), "info": {}}
                    c = capas.con_corrector(cfg, c, float(corte), {"u": u, "v": v}, datos, None)
                    pv = c[float(corte)]
            if pv is None and act is not None:
                pv = act
            if pv is not None:
                pred["radar_viejo"] = {k: fen.lluvia_por_radar(v["dbz_p75"], mm_h_area=v["mm_h_medio"])["grado"]
                                       for k, v in lr.observar(pv, locs).items()}
            if "radar_viejo" in pred:
                for comb in COMBINADOS:
                    regla = comb.split("_o_", 1)[1]
                    pred[comb] = {k: max(g, pred[regla][k]) for k, g in pred["radar_viejo"].items()}
            for m, p in pred.items():
                for lid, g in p.items():
                    sumar(tab[m][corte], g, verdad.get(lid, 0))
    return {"dia": dia, "casos": casos, "tabla": tab}


def iniciar(conf):
    G.update(conf)
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
    G["locs"] = lr.cargar_localidades()
    G["cfg"] = {**capas.CONFIG_DEFECTO, **capas.leer_json(capas.CONFIG)}


def csi(a):
    return a[0] / (a[0] + a[1] + a[2]) if a[0] + a[1] + a[2] else 0.0


def informe(tab, casos):
    print(f"casos (imagen x 7 localidades): {casos}")
    for k in ("hay", "moderada"):
        print(f"\n{'Hay lluvia' if k == 'hay' else 'Lluvia moderada o más'}: acierto (CSI) / detecta / falsas")
        print(f"{'':22s}" + "".join(f"{f'corte {c}':>24s}" for c in CORTES))
        for m in tab:
            fila = ""
            for c in CORTES:
                a = tab[m][str(c)][k] if str(c) in tab[m] else tab[m][c][k]
                pod = a[0] / (a[0] + a[2]) if a[0] + a[2] else 0
                far = a[1] / (a[0] + a[1]) if a[0] + a[1] else 0
                fila += f"{csi(a):8.3f} {pod:7.2f} {far:7.2f}"
            print(f"{m:22s}{fila}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goes", default=None)
    ap.add_argument("--paso", type=int, default=1)
    ap.add_argument("--procesos", type=int, default=2)
    ap.add_argument("--dias", default=None)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--sumar", nargs="*", default=None)
    a = ap.parse_args(argv)
    if a.sumar:
        tot, casos = None, 0
        for f in a.sumar:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            casos += d["casos"]
            if tot is None:
                tot = d["tabla"]
                continue
            for m in tot:
                for c in tot[m]:
                    for k in ("hay", "moderada"):
                        tot[m][c][k] = [x + y for x, y in zip(tot[m][c][k], d["tabla"][m][c][k])]
                    tot[m][c]["grado"] = (np.array(tot[m][c]["grado"]) + np.array(d["tabla"][m][c]["grado"])).tolist()
        informe(tot, casos)
        if a.salida:
            Path(a.salida).write_text(json.dumps({"casos": casos, "tabla": tot}), encoding="utf-8")
        return 0
    dias = a.dias.split(",") if a.dias else sorted(p.stem.split("_")[1] for p in Path(a.goes).rglob("goes_*.npz"))
    conf = {"goes_dir": a.goes, "paso": a.paso}
    if a.procesos > 1:
        with Pool(a.procesos, initializer=iniciar, initargs=(conf,)) as pool:
            res = pool.map(procesar_dia, dias)
    else:
        iniciar(conf)
        res = [procesar_dia(d) for d in dias]
    tot = nueva()
    casos = 0
    for r in res:
        casos += r["casos"]
        for m in tot:
            for c in CORTES:
                for k in ("hay", "moderada"):
                    tot[m][c][k] = [x + y for x, y in zip(tot[m][c][k], r["tabla"][m][c][k])]
                tot[m][c]["grado"] = (np.array(tot[m][c]["grado"]) + np.array(r["tabla"][m][c]["grado"])).tolist()
    informe(tot, casos)
    if a.salida:
        Path(a.salida).write_text(json.dumps({"casos": casos, "tabla": tot}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
