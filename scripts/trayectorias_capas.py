#!/usr/bin/env python3
"""Trayectoria de cada tormenta con el movimiento del nowcast en capas (5b).

Reemplaza a la trayectoria del predictor viejo (prediccion_evolucion.py) para
lo que se ve. Cada tormenta del seguimiento se mueve siguiendo el campo de
movimiento del 5b (flujo óptico con 3 imágenes y control de velocidades
imposibles) en su propio lugar: no todas a la misma velocidad.

Escribe data/radar/trayectorias_capas.json para la imagen de radar del ciclo.
Lo usa conectar_evolucion_trayectoria.py (proyecciones 5/10/20/30 de la
tormenta principal). Si falta, ese paso sigue con el predictor viejo.
Nunca corta el ciclo.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import nowcast_capas as capas  # noqa: E402
import nowcast_capas_nucleo as nc  # noqa: E402
import nowcast_flujo_optico as nf  # noqa: E402

VERSION = "trayectorias-capas-1.0"
SALIDA = ROOT / "data" / "radar" / "trayectorias_capas.json"
ESTADO = ROOT / "data" / "radar" / "seguimiento" / "estado.json"
HORIZONTES_MIN = (5, 10, 20, 30, 60)
PASO_MIN = 2.5
# Incertidumbre del lugar (km) según el plazo: mediana del error medida con
# herramientas/comparar_trayectorias.py (8 días de tormentas, ~3000 casos por
# plazo, 20/11/2026). 5 y 60 min no se midieron: se estiman de los demás.
INCERTIDUMBRE_KM = {5: 2.5, 10: 4.5, 20: 9.5, 30: 14.8, 60: 30.0}


def parse_utc(texto: Any) -> datetime | None:
    if not texto:
        return None
    try:
        v = datetime.fromisoformat(str(texto).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return (v if v.tzinfo else v.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def iso(v: datetime) -> str:
    return v.isoformat().replace("+00:00", "Z")


def muestrear(campo: np.ndarray, x: float, y: float) -> float:
    """Valor bilineal del campo en (x, y) (bordes replicados)."""
    h, w = campo.shape
    x = min(max(x, 0.0), w - 1.001)
    y = min(max(y, 0.0), h - 1.001)
    x0, y0 = int(x), int(y)
    fx, fy = x - x0, y - y0
    a = campo[y0, x0] * (1 - fx) + campo[y0, x0 + 1] * fx
    b = campo[y0 + 1, x0] * (1 - fx) + campo[y0 + 1, x0 + 1] * fx
    return float(a * (1 - fy) + b * fy)


def velocidad_celda(u: np.ndarray, v: np.ndarray, x: float, y: float, radio_px: float) -> tuple[float, float]:
    """Movimiento medio de la celda (disco de su tamaño), en px/min."""
    disco = nf.disco(u.shape, x, y, max(2.0, radio_px))
    if not disco.any():
        return muestrear(u, x, y), muestrear(v, x, y)
    return float(np.mean(u[disco])), float(np.mean(v[disco]))


def avanzar(u: np.ndarray, v: np.ndarray, x: float, y: float, radio_px: float,
            minutos: float, paso: float = PASO_MIN) -> tuple[float, float]:
    """Sigue el campo de movimiento hacia adelante (punto medio, pasos de 2,5 min)."""
    t = 0.0
    while t < minutos - 1e-9:
        dt = min(paso, minutos - t)
        ux, vy = velocidad_celda(u, v, x, y, radio_px)
        xm, ym = x + ux * dt / 2.0, y + vy * dt / 2.0
        ux, vy = velocidad_celda(u, v, xm, ym, radio_px)
        x, y = x + ux * dt, y + vy * dt
        t += dt
    return x, y


def trayectoria(u: np.ndarray, v: np.ndarray, km_px: float, track: dict, radar_dt: datetime) -> dict[str, Any]:
    x0, y0 = float(track["x"]), float(track["y"])
    radio_px = math.sqrt(max(float(track.get("pixeles") or 1.0), 1.0) / math.pi)
    ux, vy = velocidad_celda(u, v, x0, y0, radio_px)
    vel_kmh = math.hypot(ux, vy) * km_px * 60.0
    rumbo = (math.degrees(math.atan2(ux, -vy)) + 360.0) % 360.0
    horizontes = []
    x, y, t_prev = x0, y0, 0.0
    for h in HORIZONTES_MIN:
        x, y = avanzar(u, v, x, y, radio_px, h - t_prev)
        t_prev = h
        horizontes.append({
            "minutos": h,
            "timestamp_utc": iso(radar_dt + timedelta(minutes=h)),
            "x": round(x, 2), "y": round(y, 2),
            "incertidumbre_km": INCERTIDUMBRE_KM.get(h),
            "incertidumbre_pixeles": round(INCERTIDUMBRE_KM.get(h, 0.0) / km_px, 2),
        })
    return {
        "track_id": track.get("track_id"),
        "x": round(x0, 2), "y": round(y0, 2),
        "velocidad_kmh": round(vel_kmh, 1),
        "rumbo_hacia_grados": round(rumbo, 1),
        "velocidad_pixeles_min": round(math.hypot(ux, vy), 4),
        "horizontes": horizontes,
    }


def calcular(radar_dt: datetime, tracks: list[dict], ahora: datetime | None = None) -> dict[str, Any]:
    ahora = ahora or datetime.now(timezone.utc)
    base = {"version": VERSION, "radar_timestamp": iso(radar_dt), "generado_utc": iso(ahora),
            "fuente": "movimiento del nowcast en capas (5b)", "horizontes_minutos": list(HORIZONTES_MIN)}
    actual = nf.cargar_dbz(nf.ACTUAL) if nf.ACTUAL.exists() else None
    if actual is None:
        return {**base, "disponible": False, "motivo": "SIN_RADAR_ACTUAL", "tracks": {}}
    cfg = {**capas.CONFIG_DEFECTO, **capas.leer_json(capas.CONFIG)}
    clutter = nc.cargar_clutter_total(actual.shape)
    campo = nc.campo_limpio(actual, clutter)
    flujo = capas.movimiento(radar_dt, campo, actual, clutter, cfg)
    if flujo is None:
        return {**base, "disponible": False, "motivo": "SIN_MOVIMIENTO_VALIDO", "tracks": {}}
    salida = {
        **base, "disponible": True, "motivo": None,
        "movimiento": {k: flujo[k] for k in ("velocidad_kmh", "direccion_hacia_grados", "coherencia",
                                              "intervalo_min")},
        "tracks": {},
    }
    for t in tracks:
        if t.get("x") is None or t.get("y") is None:
            continue
        salida["tracks"][str(t.get("track_id"))] = trayectoria(flujo["u"], flujo["v"], flujo["km_px"],
                                                              t, radar_dt)
    return salida


def tracks_actuales(estado: dict, radar_ts: str) -> list[dict]:
    """Tormentas presentes en la imagen del ciclo (el seguimiento ya la procesó)."""
    if parse_utc(estado.get("ultimo_radar_timestamp")) != parse_utc(radar_ts):
        raise RuntimeError("el seguimiento no corresponde a la imagen de radar del ciclo")
    return [t for t in estado.get("tracks") or []
            if isinstance(t, dict) and t.get("activo") and t.get("presente_en_ultimo_frame")]


def main() -> int:
    try:
        radar_ts = nf.TIMESTAMP_FILE.read_text(encoding="utf-8").strip()
        radar_dt = parse_utc(radar_ts)
        if radar_dt is None:
            raise RuntimeError("sin imagen de radar")
        estado = json.loads(ESTADO.read_text(encoding="utf-8"))
        salida = calcular(radar_dt, tracks_actuales(estado, radar_ts))
    except Exception as exc:  # nunca corta el ciclo: sigue el predictor viejo
        salida = {"version": VERSION, "disponible": False, "motivo": f"{type(exc).__name__}: {exc}"[:200],
                  "generado_utc": iso(datetime.now(timezone.utc)), "tracks": {}}
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    tmp = SALIDA.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(salida, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(SALIDA)
    print(f"Trayectorias 5b: disponible={salida.get('disponible')} tormentas={len(salida.get('tracks') or {})} "
          f"{salida.get('motivo') or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
