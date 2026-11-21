#!/usr/bin/env python3
"""Idea 3: reproduce la alarma de "ya llega" por estaciones con días reales.

Usa el historial de estaciones guardado (data/ia/estaciones/historial_superficie.json
de distintos momentos, juntados en un archivo) y el radar guardado. Como en vivo:
cada 5 min, con el radar atrasado 15 min (movimiento y tormenta cerca de cada
estación salen de esa imagen vieja). La alarma se verifica con las imágenes reales
que llegan después (y con las firmas en las estaciones de la localidad).

Uso: python herramientas/simular_alarma_estaciones.py historial.json DESDE HASTA [--atraso 15]
     (DESDE/HASTA en UTC, ej. 2026-11-20T15:00)
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts", ROOT / "scripts" / "estaciones"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import alarma_llegada as al  # noqa: E402
import nowcast_capas_nucleo as nc  # noqa: E402
import nowcast_flujo_optico as nf  # noqa: E402
import senales_superficie as ss  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("historial")
    ap.add_argument("desde")
    ap.add_argument("hasta")
    ap.add_argument("--atraso", type=float, default=15.0)
    args = ap.parse_args(argv)
    hist = json.loads(Path(args.historial).read_text(encoding="utf-8"))
    desde = datetime.fromisoformat(args.desde).replace(tzinfo=timezone.utc)
    hasta = datetime.fromisoformat(args.hasta).replace(tzinfo=timezone.utc)
    frames = {nf.timestamp_de_nombre(p): p for p in nf.HISTORICO.glob("20*.png")}
    frames.pop(None, None)
    frames = {t: p for t, p in frames.items() if desde - timedelta(minutes=40) <= t <= hasta + timedelta(hours=2)}
    if not frames:
        print("sin imágenes de radar en ese rango")
        return 1
    clutter = None
    campos = {}
    for t, p in sorted(frames.items()):
        d = nf.cargar_dbz(p)
        if d is None:
            continue
        clutter = nc.cargar_clutter_total(d.shape) if clutter is None else clutter
        campos[t] = nc.campo_limpio(d, clutter)
    h, w = next(iter(campos.values())).shape
    km_px = nf.km_por_pixel(w)
    al.SALIDA = Path(tempfile.mkdtemp()) / "alarma.json"
    t = desde
    while t <= hasta:
        llegados = sorted(x for x in campos if x <= t - timedelta(minutes=args.atraso))
        mov, contexto = None, (lambda a, b: False, lambda a, b: False)
        if len(llegados) >= 2:
            act, prev = campos[llegados[-1]], campos[llegados[-2]]
            gap = (llegados[-1] - llegados[-2]).total_seconds() / 60.0
            f = nf.estimar_flujo(prev, act, gap, km_px)
            if f["valido"]:
                mov = {"velocidad_kmh": f["velocidad_media_kmh"], "direccion_hacia_grados": f["direccion_hacia_grados"],
                       "radar_utc": al.iso(llegados[-1])}
            ys, xs = np.nonzero(np.nan_to_num(act) >= al.UMBRAL_ECO_DBZ)

            def eco(lat, lon, xs=xs, ys=ys):
                if xs.size == 0:
                    return False
                x, y = nf.latlon_to_pixel(lat, lon, w, h)
                return bool((np.hypot(xs - x, ys - y) * km_px <= al.RADIO_ECO_KM).any())
            contexto = (eco, lambda a, b: False)
        corte = {k: [r for r in v if (al.parse_utc(r["utc"]) or t) <= t] for k, v in hist.items()}
        visibles = {x: campos[x] for x in llegados}
        r = al.calcular(t, corte, mov, contexto, campos_verificacion=visibles)
        al.escribir(al.SALIDA, r)
        firmas = [(e["nombre"], e["senales"], "tormenta cerca" if e["tormenta_cerca"] else "SIN tormenta cerca")
                  for e in r["estaciones_con_firma"]]
        locs = {k: (v["estado"], v["minutos"], [a["estacion"] for a in v["avisos"]])
                for k, v in r["localidades"].items() if v["estado"]}
        if firmas or locs:
            hl = (t - timedelta(hours=3)).strftime("%H:%M")
            m = f"{mov['velocidad_kmh']:.0f} km/h hacia {mov['direccion_hacia_grados']:.0f}°" if mov else "sin movimiento"
            print(f"{hl} (hora Arg.) [{m}] firmas={firmas} localidades={locs}")
        t += timedelta(minutes=5)
    # Cerrar la verificación con todo el radar.
    r = al.calcular(hasta + timedelta(hours=2), {}, None, None, campos_verificacion=campos)
    for u in r.get("ultimas_verificadas", []):
        print("  verificada:", u)
    print("\nresumen:", json.dumps(r["resumen"], ensure_ascii=False), "acumulado:", r["acumulado"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
