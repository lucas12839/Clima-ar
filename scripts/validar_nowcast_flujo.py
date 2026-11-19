#!/usr/bin/env python3
"""Validación retrospectiva (hindcast) del nowcast por flujo óptico.

Para cada frame T del histórico de SINARAME con T-10 y frames futuros
disponibles, pronostica con el flujo óptico y compara contra lo que el
radar observó después. Referencia: persistencia (suponer que nada se mueve).

Métricas:
- CSI a 35 dBZ por plazo (10/20/30/60 min), sin clutter.
- Bahía Blanca: ¿llega un eco ≥35 dBZ al radio de 10 km en los próximos
  60 min? Aciertos, falsas alarmas y omisiones.

Uso: python scripts/validar_nowcast_flujo.py [--desde 20261020] [--salida archivo.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import nowcast_flujo_optico as nf  # noqa: E402
from scripts.rma10_features import BAHIA_LAT, BAHIA_LON, latlon_to_pixel  # noqa: E402

UMBRAL = nf.UMBRAL_TORMENTA_DBZ
MIN_PIXELES_CONVECCION = 150


def csi(pron: np.ndarray, obs: np.ndarray) -> tuple[int, int, int]:
    a = int(np.sum(pron & obs))
    b = int(np.sum(pron & ~obs))
    c = int(np.sum(~pron & obs))
    return a, b, c


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--desde", default="2026")
    parser.add_argument("--hasta", default="2100")
    parser.add_argument("--salida", default=None)
    args = parser.parse_args(argv)

    frames = {}
    for p in nf.HISTORICO.glob("20*.png"):
        dt = nf.timestamp_de_nombre(p)
        if dt is not None and args.desde <= p.stem <= args.hasta:
            frames[dt] = p
    if not frames:
        print("Sin frames.")
        return 1

    cache: dict = {}

    def campo(dt):
        if dt not in cache:
            d = nf.cargar_dbz(frames[dt]) if dt in frames else None
            cache[dt] = d
            if len(cache) > 40:
                cache.pop(next(iter(cache)))
        return cache[dt]

    shape = None
    clutter = None
    leads = nf.LEADS_MIN
    tot = {t: {"flujo": [0, 0, 0], "persistencia": [0, 0, 0], "casos": 0} for t in leads}
    ciudad = {"flujo": {"acierto": 0, "falsa_alarma": 0, "omision": 0, "correcto_negativo": 0},
              "persistencia": {"acierto": 0, "falsa_alarma": 0, "omision": 0, "correcto_negativo": 0}}
    eta_errores = []
    detalle_ciudad = []
    casos = 0

    for t0 in sorted(frames):
        tprev = t0 - timedelta(minutes=10)
        if tprev not in frames:
            continue
        futuros = {t: t0 + timedelta(minutes=t) for t in leads}
        if not any(f in frames for f in futuros.values()):
            continue
        actual_dbz = campo(t0)
        previo_dbz = campo(tprev)
        if actual_dbz is None or previo_dbz is None:
            continue
        if shape is None:
            shape = actual_dbz.shape
            clutter = nf.cargar_mascara_clutter(shape)
        actual_l = nf.campo_limpio(actual_dbz, clutter)
        if int((actual_l >= UMBRAL).sum()) < MIN_PIXELES_CONVECCION:
            continue
        res = nf.pronosticar(previo_dbz, actual_dbz, 10.0, clutter)
        if not res["flujo"]["valido"]:
            continue
        casos += 1
        for t, ft in futuros.items():
            if ft not in frames:
                continue
            obs_dbz = campo(ft)
            if obs_dbz is None:
                continue
            obs = nf.campo_limpio(obs_dbz, clutter) >= UMBRAL
            pf = res["campos"][t] >= UMBRAL
            pp = actual_l >= UMBRAL
            for nombre, pr in (("flujo", pf), ("persistencia", pp)):
                a, b, c = csi(pr, obs)
                acc = tot[t][nombre]
                acc[0] += a
                acc[1] += b
                acc[2] += c
            tot[t]["casos"] += 1

        # Bahía Blanca, próximos 60 min (pasos de 10 con frames observados).
        h, w = shape
        cx, cy = latlon_to_pixel(BAHIA_LAT, BAHIA_LON, w, h)
        zona = nf.disco(shape, cx, cy, nf.RADIO_CIUDAD_KM / nf.km_por_pixel(w))
        obs_llega = None
        obs_eta = None
        completos = True
        for t in range(10, 61, 10):
            ft = t0 + timedelta(minutes=t)
            if ft not in frames or campo(ft) is None:
                completos = False
                continue
            vals = nf.campo_limpio(campo(ft), clutter)[zona]
            m = float(np.partition(vals, -nf.MIN_PIXELES_CIUDAD)[-nf.MIN_PIXELES_CIUDAD])
            if m >= UMBRAL and obs_eta is None:
                obs_eta = t
        if not completos:
            continue
        vals0 = actual_l[zona]
        ya_en_ciudad = float(np.partition(vals0, -nf.MIN_PIXELES_CIUDAD)[-nf.MIN_PIXELES_CIUDAD]) >= UMBRAL
        if ya_en_ciudad:
            continue  # solo cuenta llegadas nuevas
        obs_llega = obs_eta is not None
        imp = nf.impacto_ciudad(actual_l, res["flujo"])
        pred_llega = imp["eta_tormenta_min"] is not None and imp["eta_tormenta_min"] > 0
        detalle_ciudad.append({
            "t0": t0.isoformat(),
            "pred_eta": imp["eta_tormenta_min"],
            "dbz_al_llegar": imp["dbz_al_llegar"],
            "dbz_ahora": imp["dbz_max_ahora"],
            "coherencia": res["flujo"]["coherencia"],
            "velocidad_kmh": res["flujo"]["velocidad_media_kmh"],
            "obs_eta": obs_eta,
        })
        for nombre, pred in (("flujo", pred_llega), ("persistencia", False)):
            if pred and obs_llega:
                ciudad[nombre]["acierto"] += 1
            elif pred and not obs_llega:
                ciudad[nombre]["falsa_alarma"] += 1
            elif not pred and obs_llega:
                ciudad[nombre]["omision"] += 1
            else:
                ciudad[nombre]["correcto_negativo"] += 1
        if pred_llega and obs_llega:
            eta_errores.append(imp["eta_tormenta_min"] - obs_eta)

    resumen = {"casos_convectivos": casos, "umbral_dbz": UMBRAL, "csi": {}, "bahia_blanca": ciudad}
    for t in leads:
        fila = {"casos": tot[t]["casos"]}
        for nombre in ("flujo", "persistencia"):
            a, b, c = tot[t][nombre]
            fila[nombre] = round(a / (a + b + c), 3) if (a + b + c) else None
        resumen["csi"][f"{t}min"] = fila
    f = ciudad["flujo"]
    pod = f["acierto"] / (f["acierto"] + f["omision"]) if (f["acierto"] + f["omision"]) else None
    far = f["falsa_alarma"] / (f["acierto"] + f["falsa_alarma"]) if (f["acierto"] + f["falsa_alarma"]) else None
    resumen["bahia_blanca_metricas_flujo"] = {
        "probabilidad_deteccion": round(pod, 3) if pod is not None else None,
        "tasa_falsas_alarmas": round(far, 3) if far is not None else None,
        "error_medio_eta_min": round(float(np.mean(eta_errores)), 1) if eta_errores else None,
        "error_abs_medio_eta_min": round(float(np.mean(np.abs(eta_errores))), 1) if eta_errores else None,
    }
    resumen["detalle_ciudad"] = detalle_ciudad
    texto = json.dumps(resumen, ensure_ascii=False, indent=2)
    print(texto)
    if args.salida:
        Path(args.salida).write_text(texto + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
