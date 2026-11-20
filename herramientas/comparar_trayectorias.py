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
print(json.dumps({"obs": obs, "vieja": vieja, "nueva": nuevas, "disponible": nueva.get("disponible")}))
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
    a = ap.parse_args(argv)
    errores = {m: defaultdict(list) for m in ("vieja", "nueva")}
    pares = 0
    for dia in a.dias.split(","):
        tmp = preparar_copia()
        radar = tmp / "data" / "radar"
        ini = datetime.strptime(dia, "%Y%m%d").replace(hour=a.desde_hora, tzinfo=timezone.utc)
        fin = ini + timedelta(hours=a.horas)
        frames = [p for p in sorted((ROOT / "data/radar/historico").glob("20*.png"))
                  if ini <= datetime.strptime(p.stem, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc) <= fin]
        pred = {"vieja": {}, "nueva": {}}   # (track, hora objetivo, plazo) -> (x, y)
        n_ok = 0
        for p in frames:
            t = datetime.strptime(p.stem, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            shutil.copy(p, radar / "historico" / p.name)
            shutil.copy(p, radar / "actual.png")
            (radar / "radar_timestamp.txt").write_text(t.isoformat())
            try:
                correr([sys.executable, "seguimiento_temporal.py"], tmp)
                correr([sys.executable, "scripts/prediccion_evolucion.py"], tmp)
                r = json.loads(correr([sys.executable, "driver.py", p.stem], tmp).strip().splitlines()[-1])
            except Exception as exc:
                print(f"  {p.stem}: {exc}"[:200])
                continue
            n_ok += 1
            # Comparar lo que se predijo para esta hora con lo observado ahora.
            for metodo in ("vieja", "nueva"):
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
            for tid in comunes:
                for h in HORIZONTES:
                    k = str(h)
                    if k in r["vieja"][tid] and k in r["nueva"][tid]:
                        obj = t + timedelta(minutes=h)
                        pred["vieja"][(tid, obj, h)] = tuple(r["vieja"][tid][k])
                        pred["nueva"][(tid, obj, h)] = tuple(r["nueva"][tid][k])
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
