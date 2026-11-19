#!/usr/bin/env python3
"""Corre los índices por fenómeno sobre todos los frames guardados (solo radar).

Para cada frame del histórico: análisis por localidad (lluvia media, acumulados,
llegada, velocidad local) y índices por fenómeno, como en producción pero sin
estaciones, rayos ni satélite (no hay historial de esos datos). Cuenta, por
localidad y fenómeno, cuántos días y frames llegaron a cada grado: sirve para
ver falsas alarmas y para calibrar con los días que Lucas recuerde.

Uso: python herramientas/revisar_30_dias.py [--desde 20261020] [--salida x.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import indices_fenomenos as fen  # noqa: E402
import localidades_radar as lr  # noqa: E402
import nowcast_flujo_optico as nf  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default="20261001")
    ap.add_argument("--hasta", default="20991231")
    ap.add_argument("--salida", default=None)
    a = ap.parse_args(argv)
    frames = [p for p in sorted(nf.HISTORICO.glob("20*.png")) if a.desde <= p.stem[:8] <= a.hasta]
    locs = lr.cargar_localidades()
    historial = Path(tempfile.mkdtemp()) / "historial.json"
    previo, clutter, prev_vel = None, None, {}
    dias = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))   # loc -> fen -> grado -> días
    frames_g = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    detalle = defaultdict(list)
    for i, p in enumerate(frames):
        t = nf.timestamp_de_nombre(p)
        d = nf.cargar_dbz(p)
        if d is None:
            continue
        if clutter is None:
            clutter = nf.cargar_mascara_clutter(d.shape)
            km = nf.km_por_pixel(d.shape[1])
        c = nf.campo_limpio(d, clutter)
        flujo = None
        if previo is not None and 5 <= (t - previo[0]).total_seconds() / 60 <= 30 and (c >= 35).sum() >= 30:
            f = nf.estimar_flujo(previo[1], c, (t - previo[0]).total_seconds() / 60, km)
            flujo = f if f["valido"] else None
        res = lr.analizar(t, c, flujo, 0.0, None, locs, historial)
        dia_local = (t - timedelta(hours=3)).strftime("%Y-%m-%d")
        for k, x in res.items():
            lleg = x.get("llegada") or {}
            ev = fen.evaluar(radar={
                "disponible": True, "dbz_ciudad": x["dbz_p75"], "dbz_pico": x["dbz_pico"],
                "mm_h_area": x["mm_h_medio"],
                "dbz_30km": lleg.get("dbz_entorno_20km") or x["dbz_pico"],
                "dbz_llegada": lleg.get("dbz_al_llegar"), "eta_llegada_min": lleg.get("eta_desde_ahora_min"),
                "velocidad_kmh": lleg.get("velocidad_local_kmh"),
                "velocidad_previa_kmh": prev_vel.get(k),
                "coherencia": (flujo or {}).get("coherencia"),
                "minutos_lluvia_intensa": x["minutos_lluvia_intensa"],
                "lluvia_1h_mm": x["lluvia_1h_mm"], "lluvia_3h_mm": x["lluvia_3h_mm"],
                "lluvia_6h_mm": x["lluvia_6h_mm"],
            })
            prev_vel[k] = lleg.get("velocidad_local_kmh")
            for fn, e in ev["ahora"].items():
                g = e["grado"]
                if g:
                    dias[k][fn][g].add(dia_local)
                    frames_g[k][fn][g] += 1
                if g >= 4:
                    detalle[k].append({"utc": t.isoformat(), "fenomeno": fn, "razones": e["razones"][:2]})
        previo = (t, c)
        if i % 300 == 0:
            print(f"{i}/{len(frames)} {t:%d/%m %H:%M}", flush=True)
    salida = {
        "frames": len(frames),
        "dias_por_grado": {k: {fn: {str(g): sorted(s) for g, s in sorted(v.items())} for fn, v in d.items()}
                            for k, d in dias.items()},
        "frames_por_grado": {k: {fn: dict(v) for fn, v in d.items()} for k, d in frames_g.items()},
        "detalle_severo": detalle,
    }
    texto = json.dumps(salida, ensure_ascii=False, indent=1)
    if a.salida:
        Path(a.salida).write_text(texto, encoding="utf-8")
    for k, d in salida["dias_por_grado"].items():
        print(k, {fn: {g: len(s) for g, s in v.items()} for fn, v in d.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
