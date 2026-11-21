#!/usr/bin/env python3
"""¿Qué trayectoria acierta más: el predictor viejo o el movimiento del 5b?

Vuelve a correr días guardados imagen por imagen, como en producción (en una
copia aparte del repo): seguimiento -> predictor viejo (prediccion_evolucion.py)
-> trayectorias del 5b (trayectorias_capas.py). Cada predicción de una
tormenta a 10, 20 y 30 min se compara con dónde estuvo DE VERDAD esa misma
tormenta (mismo número del seguimiento) en la imagen de esa hora.

Uso: python herramientas/comparar_trayectorias.py [--dias 20261028,20261112]
     [--desde-hora 15] [--horas 12] [--salida x.json]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAS_DEFECTO = "20261027,20261028,20261101,20261103,20261105,20261112,20261113,20261120"
KM_PX = 240.0 / 255.5
HORIZONTES = (10, 20, 30)

DRIVER = r'''
import json, sys, subprocess
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import trayectorias_capas as tc
frame = sys.argv[1]
radar_dt = datetime.strptime(frame, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
estado = json.loads(Path("data/radar/seguimiento/estado.json").read_text())
tracks = tc.tracks_actuales(estado, radar_dt.isoformat())
nueva = tc.calcular(radar_dt, tracks)
vieja = {}
try:
    pe = json.loads(Path("data/radar/seguimiento/prediccion_evolucion.json").read_text())
    for n in pe.get("nucleos") or []:
        if n.get("pronostico_disponible") is True:
            vieja[str(n["track_id"])] = {str(int(h["horizonte_minutos"])): [h["x_estimado"], h["y_estimado"]]
                                         for h in n.get("horizontes") or []}
except Exception:
    pass
obs = {str(t["track_id"]): [t["x"], t["y"]] for t in tracks}
nuevas = {k: {str(h["minutos"]): [h["x"], h["y"]] for h in v["horizontes"]} for k, v in (nueva.get("tracks") or {}).items()}
salida = {"obs": obs, "vieja": vieja, "nueva": nuevas, "disponible": nueva.get("disponible")}
# Etapa A: misma trayectoria con lo que ya mide el satélite a la hora de llegada del radar.
import os
if os.path.exists("sat.npz"):
    import numpy as np
    z = np.load("sat.npz")
    sat = {k: (None if not z[k].size else z[k]) for k in ("radar", "nuevo", "previo")}
    sat["tramo"], sat["entre"] = float(z["tramo"]), (None if float(z["entre"]) < 0 else float(z["entre"]))
    for nombre, par in json.loads(Path("variantes.json").read_text()).items():
        r = tc.calcular(radar_dt, tracks, satelite=sat, parametros_satelite=par)
        salida[nombre] = {k: {str(h["minutos"]): [h["x"], h["y"]] for h in v["horizontes"]}
                          for k, v in (r.get("tracks") or {}).items()}
print(json.dumps(salida))
'''


def preparar_copia() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="trayectorias_"))
    for nombre in ("scripts", "goes19", "config", "seguimiento_temporal.py"):
        origen = ROOT / nombre
        (shutil.copytree if origen.is_dir() else shutil.copy)(origen, tmp / nombre)
    radar = tmp / "data" / "radar"
    (radar / "historico").mkdir(parents=True)
    (radar / "seguimiento").mkdir()
    for p in (ROOT / "data" / "radar").glob("mascara_ecos_estaticos*"):
        shutil.copy(p, radar / p.name)
    (tmp / "driver.py").write_text(DRIVER, encoding="utf-8")
    return tmp


def correr(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stdout + r.stderr)[-400:])
    return r.stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dias", default=DIAS_DEFECTO)
    ap.add_argument("--desde-hora", type=int, default=15)
    ap.add_argument("--horas", type=int, default=12)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--goes", default=None, help="historial de GOES-19 (etapa A: corrección por satélite)")
    ap.add_argument("--llegada", type=float, default=20.0,
                    help="minutos entre la hora de la imagen y cuando se usa (atraso de SINARAME + cadena)")
    ap.add_argument("--variantes", default='{"satelite": {"peso_posicion": 1.0}}')
    a = ap.parse_args(argv)
    variantes = json.loads(a.variantes) if a.goes else {}
    metodos = ("vieja", "nueva", *variantes)
    errores = {m: defaultdict(list) for m in metodos}
    goes = None
    if a.goes:
        sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "herramientas"), str(ROOT)]
        import numpy as np
        import simular_etapa_a as sea
        import simular_huecos as sh
        goes = sh.cargar_goes(Path(a.goes))
    pares = 0
    for dia in a.dias.split(","):
        tmp = preparar_copia()
        radar = tmp / "data" / "radar"
        ini = datetime.strptime(dia, "%Y%m%d").replace(hour=a.desde_hora, tzinfo=timezone.utc)
        fin = ini + timedelta(hours=a.horas)
        frames = [p for p in sorted((ROOT / "data/radar/historico").glob("20*.png"))
                  if ini <= datetime.strptime(p.stem, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc) <= fin]
        pred = {m: {} for m in metodos}   # (track, hora objetivo, plazo) -> (x, y)
        if variantes:
            (tmp / "variantes.json").write_text(json.dumps(variantes))
        n_ok = 0
        for p in frames:
            t = datetime.strptime(p.stem, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            shutil.copy(p, radar / "historico" / p.name)
            shutil.copy(p, radar / "actual.png")
            (radar / "radar_timestamp.txt").write_text(t.isoformat())
            if goes is not None:
                sats = sea.satelites_para(goes, t, t + timedelta(minutes=a.llegada), (512, 512))
                if sats is None:
                    (tmp / "sat.npz").unlink(missing_ok=True)
                else:
                    vacio = np.zeros(0, np.float32)
                    np.savez(tmp / "sat.npz", **{k: (vacio if sats[k] is None else sats[k]) for k in ("radar", "nuevo", "previo")},
                             tramo=sats["tramo"], entre=-1.0 if sats["entre"] is None else sats["entre"])
            try:
                correr([sys.executable, "seguimiento_temporal.py"], tmp)
                correr([sys.executable, "scripts/prediccion_evolucion.py"], tmp)
                r = json.loads(correr([sys.executable, "driver.py", p.stem], tmp).strip().splitlines()[-1])
            except Exception as exc:
                print(f"  {p.stem}: {exc}"[:200])
                continue
            n_ok += 1
            # Comparar lo que se predijo para esta hora con lo observado ahora.
            for metodo in metodos:
                for (tid, objetivo, plazo), (x, y) in list(pred[metodo].items()):
                    if abs((objetivo - t).total_seconds()) > 60:
                        continue
                    if tid in r["obs"]:
                        ox, oy = r["obs"][tid]
                        errores[metodo][plazo].append(math.hypot(ox - x, oy - y) * KM_PX)
                        pares += metodo == "nueva"
                    del pred[metodo][(tid, objetivo, plazo)]
            # Solo tormentas con predicción de los DOS métodos (comparación justa).
            comunes = set(r["vieja"]) & set(r["nueva"])
            for m in variantes:   # sin satélite a esa hora: la variante es igual a "nueva"
                r.setdefault(m, r["nueva"])
            for tid in comunes:
                for h in HORIZONTES:
                    k = str(h)
                    if all(tid in r[m] and k in r[m][tid] for m in metodos):
                        obj = t + timedelta(minutes=h)
                        for m in metodos:
                            pred[m][(tid, obj, h)] = tuple(r[m][tid][k])
        print(f"{dia}: {n_ok}/{len(frames)} imágenes")
        shutil.rmtree(tmp, ignore_errors=True)
    resumen = {}
    for metodo, por_plazo in errores.items():
        resumen[metodo] = {}
        for h in HORIZONTES:
            e = por_plazo.get(h) or []
            if not e:
                continue
            resumen[metodo][f"{h}_min"] = {
                "casos": len(e),
                "error_mediano_km": round(statistics.median(e), 1),
                "error_p75_km": round(sorted(e)[int(0.75 * (len(e) - 1))], 1),
                "dentro_5km_pct": round(100 * sum(x <= 5 for x in e) / len(e), 0),
                "dentro_10km_pct": round(100 * sum(x <= 10 for x in e) / len(e), 0),
            }
    print(json.dumps(resumen, ensure_ascii=False, indent=1))
    if a.salida:
        Path(a.salida).write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
