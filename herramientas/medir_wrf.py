#!/usr/bin/env python3
"""Punto 7: ¿el WRF del SMN acierta la lluvia por localidad?

Compara la lluvia del WRF (historial bajado con `historial_wrf.py`) con la lluvia que midió
el radar guardado en cada localidad (dBZ -> mm/h con Marshall-Palmer, limpio de clutter y de
ecos falsos con el satélite cuando hay historial de GOES-19).

Se usa la corrida que estaría disponible en vivo: la última que salió al menos 4 h antes
(el SMN la publica con ~4 h de demora), así que el plazo va de 4 a 10 h.

Mide, por localidad, por hora y por bloque de 3 h (como lo resume `modelos_region.py`):
detección (llovió y el WRF lo dio), falsas alarmas (el WRF lo dio y no llovió), CSI y
"con ±1 h" (acierta si el WRF da la lluvia una hora antes o después).

Uso: python herramientas/medir_wrf.py --wrf /tmp/wrf [--goes /tmp/goes] [--salida x.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts", ROOT / "herramientas"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import localidades_radar as lr  # noqa: E402
import nowcast_capas_nucleo as nc  # noqa: E402
import nowcast_flujo_optico as nf  # noqa: E402

DEMORA_H = 4
UMBRALES_MM_H = (0.5, 1.0, 5.0)
MIN_IMAGENES_HORA = 4


def lluvia_radar(goes_dir: str | None) -> dict[str, dict[datetime, float]]:
    """mm por hora (media en el círculo de cada localidad), solo horas con >= 4 imágenes."""
    import corrector_tiempo_real as ct
    import simular_huecos as sh

    goes = sh.cargar_goes(Path(goes_dir)) if goes_dir else None
    frames = sorted((nf.timestamp_de_nombre(p), p) for p in nf.HISTORICO.glob("20*.png"))
    frames = [(t, p) for t, p in frames if t is not None]
    shape = nf.cargar_dbz(frames[0][1]).shape
    clutter = nc.cargar_clutter_total(shape)
    zonas = {}
    for loc in lr.cargar_localidades():
        x, y = nf.latlon_to_pixel(loc["lat"], loc["lon"], shape[1], shape[0])
        zonas[loc["id"]] = nf.disco(shape, x, y, float(loc.get("radio_km", 4.0)) / nf.km_por_pixel(shape[1]))
    suma = defaultdict(lambda: defaultdict(float))
    n = defaultdict(int)
    for t, p in frames:
        d = nf.cargar_dbz(p)
        if d is None:
            continue
        c = nc.campo_limpio(d, clutter)
        if goes is not None:
            sat = sh.satelite_cercano(goes, t, shape)
            if sat is not None:
                c = ct.quitar_ecos_sin_nube(c, sat)
        r = nc.a_lluvia(c)
        hora = t.replace(minute=0, second=0, microsecond=0, tzinfo=None)
        n[hora] += 1
        for k, z in zonas.items():
            suma[k][hora] += float(r[z].mean())
    salida = {}
    for k, horas in suma.items():
        # mm en la hora = media de mm/h de las imágenes (cada imagen representa 10 min)
        salida[k] = {h: v / n[h] for h, v in horas.items() if n[h] >= MIN_IMAGENES_HORA}
    return salida


def redondear_10min(t: datetime) -> datetime:
    """El WRF del SMN guarda algunas horas como 14:09:59: se llevan a 14:10."""
    t = t.replace(tzinfo=None) + timedelta(seconds=30)
    return t.replace(second=0, microsecond=0, minute=(t.minute // 10) * 10)


def lluvia_wrf(carpeta: Path) -> dict[str, dict[datetime, dict[str, tuple[float, float]]]]:
    """mm por hora de cada corrida: {loc: {hora: {init_iso: (max, media)}}}.
    En los archivos 10M del WRF del SMN, PP es la lluvia de esos 10 min (no el acumulado
    desde el inicio de la corrida): la hora H suma los valores de (H, H+1h]."""
    salida: dict = defaultdict(lambda: defaultdict(dict))
    for p in sorted(carpeta.glob("wrf_*.json")):
        for init, locs in json.loads(p.read_text(encoding="utf-8")).items():
            for k, serie in locs.items():
                horas: dict = defaultdict(lambda: [0.0, 0.0, 0])
                for t, mx, md in serie:
                    tt = redondear_10min(datetime.fromisoformat(t.replace("Z", "")))
                    h = (tt - timedelta(minutes=10)).replace(minute=0)
                    horas[h][0] += mx
                    horas[h][1] += md
                    horas[h][2] += 1
                for h, (mx, md, n) in horas.items():
                    if n == 6:
                        salida[k][h][init] = (mx, md)
    return salida


def corrida_en_vivo(hora: datetime) -> str:
    base = hora - timedelta(hours=DEMORA_H)
    init = base.replace(hour=(base.hour // 6) * 6, minute=0, second=0, microsecond=0)
    return init.strftime("%Y-%m-%dT%H:00:00Z")


def metricas(a, b, c, d):
    return {"llovio_y_dio": a, "dio_y_no_llovio": b, "llovio_y_no_dio": c, "nada": d,
            "deteccion_pct": round(100 * a / (a + c), 1) if a + c else None,
            "falsas_alarmas_pct": round(100 * b / (a + b), 1) if a + b else None,
            "csi_pct": round(100 * a / (a + b + c), 1) if a + b + c else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrf", required=True)
    ap.add_argument("--goes", default=None)
    ap.add_argument("--salida", default=None)
    args = ap.parse_args(argv)
    radar = lluvia_radar(args.goes)
    wrf = lluvia_wrf(Path(args.wrf))
    res: dict = {"demora_corrida_h": DEMORA_H, "por_umbral": {}}
    for umbral in UMBRALES_MM_H:
        tot = defaultdict(lambda: [0, 0, 0, 0])
        for k, horas in radar.items():
            w = {}
            for hu in horas:
                v = wrf.get(k, {}).get(hu, {}).get(corrida_en_vivo(hu))
                if v is not None:
                    w[hu] = v
            comunes = sorted(hu for hu in w)
            obs = {hu: horas[hu] for hu in comunes}
            for campo, idx in (("max_caja", 0), ("media_caja", 1)):
                for hu in comunes:
                    o, p = obs[hu] >= umbral, w[hu][idx] >= umbral
                    # con ±1 h: vale si el WRF lo da en la hora anterior o siguiente
                    p1 = p or any(w.get(hu + timedelta(hours=s), (0, 0))[idx] >= umbral for s in (-1, 1))
                    o1 = o or any((obs.get(hu + timedelta(hours=s)) or 0) >= umbral for s in (-1, 1))
                    for clave, (oo, pp) in ((f"{campo}|hora", (o, p)), (f"{campo}|hora_tol1h", (o, p1 if o else p and not o1))):
                        acc = tot[(k, clave)]
                        acc[0 if oo and pp else 1 if pp else 2 if oo else 3] += 1
                        acc2 = tot[("todas", clave)]
                        acc2[0 if oo and pp else 1 if pp else 2 if oo else 3] += 1
                # bloques de 3 h (como "lluvia_3h" de modelos_region)
                bloques = defaultdict(lambda: [0.0, 0.0, 0])
                for hu in comunes:
                    b = hu.replace(hour=(hu.hour // 3) * 3)
                    bloques[b][0] += obs[hu]
                    bloques[b][1] += w[hu][idx]
                    bloques[b][2] += 1
                for b, (o3, p3, nh) in bloques.items():
                    if nh < 3:
                        continue
                    oo, pp = o3 >= umbral, p3 >= umbral
                    for kk in (k, "todas"):
                        acc = tot[(kk, f"{campo}|bloque_3h")]
                        acc[0 if oo and pp else 1 if pp else 2 if oo else 3] += 1
        res["por_umbral"][f"{umbral}_mm"] = {f"{k}|{c}": metricas(*v) for (k, c), v in sorted(tot.items())}
    res["horas_comparadas"] = sum(1 for k in radar for _ in radar[k]) // max(len(radar), 1)
    if args.salida:
        Path(args.salida).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for u, tabla in res["por_umbral"].items():
        print(f"\nLluvia >= {u} (det / FA / CSI):")
        for clave, m in tabla.items():
            if clave.startswith("todas"):
                print(f"  {clave:32s} {m['deteccion_pct']}/{m['falsas_alarmas_pct']}/{m['csi_pct']}  (llovió {m['llovio_y_dio'] + m['llovio_y_no_dio']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
