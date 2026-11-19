#!/usr/bin/env python3
"""Reproduce frames del histórico RMA10 por la cadena real de radar + nowcast.

Para cada frame (en orden): lo copia como actual.png, corre procesador_radar,
seguimiento_temporal, nowcast_flujo_optico, adaptar_rma10_intelligence y
georreferenciar_rma10, y registra la velocidad del principal y de cada
núcleo. Sirve para medir velocidades imposibles y comparar versiones.

Uso (sobre una COPIA del repo):
  python herramientas/reproducir_frames.py <copia> <desde YYYYmmddHHMMSS> <hasta> [--salida x.json]
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PASOS = (
    "scripts/procesador_radar.py",
    "seguimiento_temporal.py",
    "scripts/nowcast_flujo_optico.py",
    "scripts/adaptar_rma10_intelligence.py",
    "scripts/georreferenciar_rma10.py",
)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    salida = None
    if "--salida" in argv:
        i = argv.index("--salida")
        salida = argv[i + 1]
        del argv[i:i + 2]
    copia, desde, hasta = Path(argv[0]), argv[1], argv[2]
    hist = copia / "data/radar/historico"
    frames = sorted(p for p in hist.glob("20*.png") if desde <= p.stem <= hasta)
    # Estado limpio para que el seguimiento arranque en el primer frame.
    for f in ("data/radar/seguimiento/estado.json", ".climaar/estado_rma10.json"):
        (copia / f).unlink(missing_ok=True)
    registros = []
    for path in frames:
        dt = datetime.strptime(path.stem, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        (copia / "data/radar/actual.png").write_bytes(path.read_bytes())
        (copia / "data/radar/radar_timestamp.txt").write_text(dt.isoformat(), encoding="utf-8")
        ok = True
        import os
        from datetime import timedelta
        env = dict(os.environ, CLIMAAR_AHORA_UTC=(dt + timedelta(minutes=7)).isoformat())
        for paso in PASOS:
            r = subprocess.run([sys.executable, paso], cwd=copia, capture_output=True, text=True, env=env)
            if r.returncode != 0:
                ok = False
                registros.append({"frame": path.stem, "error": paso, "detalle": (r.stdout + r.stderr)[-300:]})
                break
        if not ok:
            continue
        nc = json.loads((copia / "data/radar/radar_nowcast.json").read_text(encoding="utf-8"))["nowcast"]
        nucleos = []
        for n in nc.get("nucleos_georreferenciados") or []:
            mov = n.get("movimiento_bahia") or {}
            nucleos.append({
                "track_id": n.get("track_id"),
                "velocidad_kmh": mov.get("velocidad_kmh"),
                "fuente": (n.get("movimiento") or {}).get("fuente_movimiento"),
            })
        flujo_path = copia / "data/radar/nowcast_flujo.json"
        flujo = json.loads(flujo_path.read_text(encoding="utf-8")) if flujo_path.exists() else {}
        registros.append({
            "frame": path.stem,
            "flujo_kmh": (flujo.get("movimiento") or {}).get("velocidad_media_kmh"),
            "flujo_coherencia": (flujo.get("movimiento") or {}).get("coherencia"),
            "actividad": nc.get("actividad"),
            "principal": nc.get("nucleo_principal_id"),
            "velocidad_principal_kmh": nc.get("velocidad_kmh"),
            "eta_min": nc.get("eta_minutos"),
            "nucleos": nucleos,
        })
    texto = json.dumps(registros, ensure_ascii=False, indent=1)
    if salida:
        Path(salida).write_text(texto, encoding="utf-8")
    vel = [r["velocidad_principal_kmh"] for r in registros if r.get("velocidad_principal_kmh")]
    todas = [n["velocidad_kmh"] for r in registros for n in r.get("nucleos", []) if n.get("velocidad_kmh")]
    errores = [r for r in registros if "error" in r]
    print(json.dumps({
        "frames": len(frames),
        "errores": len(errores),
        "principal_con_velocidad": len(vel),
        "principal_mayor_120": sum(v > 120 for v in vel),
        "nucleos_con_velocidad": len(todas),
        "nucleos_mayor_120": sum(v > 120 for v in todas),
        "mediana_principal": sorted(vel)[len(vel) // 2] if vel else None,
        "primer_error": errores[0] if errores else None,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
