#!/usr/bin/env python3
"""Pasa por el seguimiento los frames recuperados antes del frame nuevo.

capturar_radar_smn.py deja en data/radar/intermedios_pendientes.json los
frames que SINARAME publicó pero la página no mostró. Para que cada celda
mantenga su trayectoria, se procesan en orden (procesador + seguimiento)
antes que el frame nuevo; al terminar se restaura el frame nuevo como
radar activo. Nunca corta el ciclo: si algo falla, sigue con el frame nuevo.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RADAR = BASE / "data" / "radar"
PENDIENTES = RADAR / "intermedios_pendientes.json"
ACTUAL = RADAR / "actual.png"
TIMESTAMP = RADAR / "radar_timestamp.txt"
PASOS = ("scripts/procesador_radar.py", "seguimiento_temporal.py")


def parse_utc(texto):
    try:
        dt = datetime.fromisoformat(str(texto).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def leer_pendientes(actual_dt):
    try:
        datos = json.loads(PENDIENTES.read_text(encoding="utf-8"))
    except Exception:
        return []
    salida = []
    for item in datos if isinstance(datos, list) else []:
        dt = parse_utc((item or {}).get("utc"))
        archivo = Path((item or {}).get("archivo", ""))
        if dt is not None and actual_dt is not None and dt < actual_dt and archivo.is_file():
            salida.append((dt, archivo))
    return sorted(salida)


def main() -> int:
    if not ACTUAL.exists() or not TIMESTAMP.exists():
        print("Sin radar activo: nada que hacer.")
        return 0
    actual_ts = TIMESTAMP.read_text(encoding="utf-8").strip()
    actual_dt = parse_utc(actual_ts)
    pendientes = leer_pendientes(actual_dt)
    if not pendientes:
        print("Sin frames intermedios pendientes.")
        PENDIENTES.unlink(missing_ok=True)
        return 0

    respaldo_png = RADAR / ".actual_nuevo.png"
    shutil.copyfile(ACTUAL, respaldo_png)
    procesados = 0
    try:
        for dt, archivo in pendientes:
            shutil.copyfile(archivo, ACTUAL)
            TIMESTAMP.write_text(dt.isoformat(), encoding="utf-8")
            ok = True
            for paso in PASOS:
                r = subprocess.run([sys.executable, paso], cwd=BASE, capture_output=True, text=True)
                if r.returncode != 0:
                    ok = False
                    print(f"AVISO: {paso} falló con el intermedio {dt.isoformat()}: {(r.stdout + r.stderr)[-300:]}")
                    break
            if ok:
                procesados += 1
                print("Intermedio procesado:", dt.isoformat())
    finally:
        shutil.copyfile(respaldo_png, ACTUAL)
        TIMESTAMP.write_text(actual_ts, encoding="utf-8")
        respaldo_png.unlink(missing_ok=True)
        PENDIENTES.unlink(missing_ok=True)
    print(f"Frames intermedios procesados: {procesados}/{len(pendientes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
