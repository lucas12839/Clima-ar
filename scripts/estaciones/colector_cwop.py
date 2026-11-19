#!/usr/bin/env python3
"""Colector de estaciones CWOP (red abierta de estaciones ciudadanas).

Se conecta a la red APRS-IS en modo solo escucha (clave -1, permitido para
recibir), filtra las estaciones de config/estaciones_cwop.json y guarda la
última transmisión de cada una en data/ia/estaciones/observaciones/cwop_actual.json
con el mismo formato que la BCP. Termina apenas recibió todas o al llegar al
tiempo máximo (las estaciones transmiten cada ~10 min).

Si no llega nada nuevo, conserva la última observación: su edad crece y el
control de calidad la marca como atrasada.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[2]
CONFIG = BASE / "config/estaciones_cwop.json"
SALIDA = BASE / "data/ia/estaciones/observaciones/cwop_actual.json"
EDAD_VALIDA_MIN = 20.0
MPH_A_KMH = 1.609344
CENT_PULG_A_MM = 0.254

PATRON = re.compile(
    r"^(?P<call>[A-Z0-9-]+)>[^:]*:[@/](?P<d>\d{2})(?P<h>\d{2})(?P<m>\d{2})z"
    r"(?P<lat>\d{4}\.\d{2})(?P<ns>[NS]).(?P<lon>\d{5}\.\d{2})(?P<ew>[EW])_"
    r"(?P<dir>[\d.]{3})/(?P<vel>[\d.]{3})(?P<resto>.*)$"
)


def _campo(resto: str, letra: str, largo: int) -> float | None:
    m = re.search(letra + r"(-?[\d.]{%d})" % largo, resto)
    if not m or "." in m.group(1).replace("-", ""):
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parsear(linea: str, ahora: datetime | None = None) -> dict[str, Any] | None:
    """Paquete meteorológico APRS (formato con posición y hora) -> variables."""
    m = PATRON.match(linea.strip())
    if not m:
        return None
    ahora = ahora or datetime.now(timezone.utc)
    try:
        t = ahora.replace(day=int(m["d"]), hour=int(m["h"]), minute=int(m["m"]), second=0, microsecond=0)
    except ValueError:
        return None
    if t > ahora:  # transmisión del mes anterior (cambio de mes)
        return None
    lat = int(m["lat"][:2]) + float(m["lat"][2:]) / 60.0
    lon = int(m["lon"][:3]) + float(m["lon"][3:]) / 60.0
    lat = -lat if m["ns"] == "S" else lat
    lon = -lon if m["ew"] == "W" else lon
    resto = m["resto"]

    def num(txt):
        return None if "." in txt else float(txt)

    temp_f = _campo(resto, "t", 3)
    rafaga = _campo(resto, "g", 3)
    hum = _campo(resto, "h", 2)
    presion = _campo(resto, "b", 5)
    lluvia_h = _campo(resto, "r", 3)
    lluvia_dia = _campo(resto, "P", 3)
    vel = num(m["vel"])
    return {
        "indicativo": m["call"],
        "observation_time": t.isoformat(),
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "variables": {
            "temperatura_c": round((temp_f - 32) * 5 / 9, 1) if temp_f is not None else None,
            "viento_kmh": round(vel * MPH_A_KMH, 1) if vel is not None else None,
            "direccion_viento_grados": num(m["dir"]),
            "rafaga_15min_kmh": round(rafaga * MPH_A_KMH, 1) if rafaga is not None else None,
            "humedad_relativa_pct": (100.0 if hum == 0 else hum) if hum is not None else None,
            "presion_hpa": round(presion / 10.0, 1) if presion is not None else None,
            "lluvia_1h_mm": round(lluvia_h * CENT_PULG_A_MM, 1) if lluvia_h is not None else None,
            "precipitacion_dia_mm": round(lluvia_dia * CENT_PULG_A_MM, 1) if lluvia_dia is not None else None,
        },
    }


def escuchar(indicativos: list[str], servidor: str, puerto: int, max_seg: float) -> dict[str, dict]:
    recibidas: dict[str, dict] = {}
    try:
        s = socket.create_connection((servidor, puerto), timeout=30)
    except OSError as exc:
        print(f"AVISO: no se pudo conectar a APRS-IS: {exc}")
        return recibidas
    s.sendall(("user CLIMAAR pass -1 vers ClimaAR 1.0 filter b/" + "/".join(indicativos) + "\r\n").encode())
    s.settimeout(20)
    fin = time.time() + max_seg
    buf = b""
    try:
        while time.time() < fin and len(recibidas) < len(indicativos):
            try:
                datos = s.recv(4096)
            except socket.timeout:
                continue
            if not datos:
                break
            buf += datos
            while b"\n" in buf:
                linea, buf = buf.split(b"\n", 1)
                obs = parsear(linea.decode("latin-1"))
                if obs and obs["indicativo"] in indicativos:
                    recibidas[obs["indicativo"]] = obs
                    print(f"recibida {obs['indicativo']} {obs['observation_time']}")
    finally:
        s.close()
    return recibidas


def escribir(path: Path, datos) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(datos, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def armar_salida(cfg: dict, recibidas: dict, previa: dict, ahora: datetime) -> dict:
    previas = {s.get("indicativo"): s for s in (previa.get("stations") or [])}
    estaciones = []
    for e in cfg.get("estaciones") or []:
        obs = recibidas.get(e["indicativo"]) or previas.get(e["indicativo"])
        if not obs:
            estaciones.append({"ok": False, "indicativo": e["indicativo"], "nombre": e["nombre"],
                               "estado": "sin_datos", "fresh": False})
            continue
        t = datetime.fromisoformat(obs["observation_time"])
        edad = (ahora - t).total_seconds() / 60.0
        valida = edad <= EDAD_VALIDA_MIN
        estaciones.append({
            "ok": True, "indicativo": e["indicativo"], "nombre": e["nombre"],
            "observation_time": obs["observation_time"], "edad_minutos": round(edad, 1),
            "fresh": valida, "estado": "valida" if valida else "atrasada",
            "lat": e["lat"], "lon": e["lon"], "variables": obs["variables"],
        })
    return {"ok": True, "collector": "cwop", "version": "1.0.0", "generated_utc": ahora.isoformat(),
            "max_age_minutes": EDAD_VALIDA_MIN, "stations_total": len(estaciones),
            "stations_validas": sum(1 for s in estaciones if s.get("estado") == "valida"),
            "stations": estaciones}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-segundos", type=float, default=630.0)
    a = ap.parse_args(argv)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    indicativos = [e["indicativo"] for e in cfg.get("estaciones") or []]
    try:
        previa = json.loads(SALIDA.read_text(encoding="utf-8")) if SALIDA.exists() else {}
    except Exception:
        previa = {}
    recibidas = escuchar(indicativos, cfg.get("servidor", "rotate.aprs2.net"), int(cfg.get("puerto", 14580)),
                         a.max_segundos) if indicativos else {}
    salida = armar_salida(cfg, recibidas, previa, datetime.now(timezone.utc))
    escribir(SALIDA, salida)
    print(json.dumps({"validas": salida["stations_validas"], "total": salida["stations_total"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
