#!/usr/bin/env python3
"""Radar por localidad: Bahía Blanca y las localidades cercanas por separado.

Una tormenta puede pasar al costado de Bahía Blanca y golpear Punta Alta o
Ingeniero White (13/11/2026: en Bahía llovió poco y en Punta Alta más de
70 mm). Por eso cada localidad de config/localidades.json tiene su propia
observación, su propio pronóstico de llegada y su propio acumulado.

- Observación: lluvia MEDIA sobre el área (no el píxel más intenso: el 13/11
  el píxel más intenso daba 175 mm en Bahía Blanca y el promedio del área 7 mm,
  que es lo que se vio), intensidad típica (percentil 75) y núcleo más
  intenso (para granizo).
- Llegada: el campo advectado por el flujo óptico cada 5 min hasta 60 min.
- Acumulados: historial de 8 h por localidad (data/radar/historial_localidades.json).
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

import nowcast_flujo_optico as nf
from scripts.rma10_features import latlon_to_pixel

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "localidades.json"
HISTORIAL = ROOT / "data" / "radar" / "historial_localidades.json"
HISTORIAL_HORAS = 8
RADIO_MIN_KM = 3.0          # el radar tiene ~0,94 km/píxel: menos no es representativo
MIN_PIXELES = 3
LLUVIA_INTENSA_DBZ = 40.0
DBZ_TOPE_LLUVIA = 53.0       # por encima suele ser granizo: no suma lluvia
ECO_LLEGADA_DBZ = nf.UMBRAL_TORMENTA_DBZ
ECO_FUERTE_DBZ = nf.UMBRAL_FUERTE_DBZ


def cargar_localidades(path: Path = CONFIG) -> list[dict[str, Any]]:
    try:
        return list(json.loads(path.read_text(encoding="utf-8")).get("localidades") or [])
    except Exception:
        return []


def _zonas(shape, localidades):
    h, w = shape
    km_px = nf.km_por_pixel(w)
    zonas = {}
    for loc in localidades:
        cx, cy = latlon_to_pixel(loc["lat"], loc["lon"], w, h)
        radio = max(float(loc.get("radio_km", 5.0)), RADIO_MIN_KM)
        zonas[loc["id"]] = (nf.disco((h, w), cx, cy, radio / km_px), cx, cy)
    return zonas


def mm_h(dbz: np.ndarray) -> np.ndarray:
    """Marshall-Palmer vectorizado con tope de granizo."""
    d = np.nan_to_num(np.asarray(dbz, dtype=np.float64), nan=0.0)
    d = np.minimum(d, DBZ_TOPE_LLUVIA)
    r = (10 ** (d / 10.0) / 200.0) ** (1 / 1.6)
    return np.where(d >= 15.0, r, 0.0)


def _pico(valores: np.ndarray) -> float:
    valores = np.nan_to_num(valores, nan=0.0)
    if valores.size < MIN_PIXELES:
        return 0.0
    return float(np.partition(valores, -MIN_PIXELES)[-MIN_PIXELES])


def observar(campo: np.ndarray, localidades: list[dict]) -> dict[str, dict[str, Any]]:
    salida = {}
    for loc_id, (zona, _, _) in _zonas(campo.shape, localidades).items():
        vals = np.nan_to_num(campo[zona], nan=0.0)
        salida[loc_id] = {
            "dbz_pico": round(_pico(vals), 1),
            "dbz_p75": round(float(np.percentile(vals, 75)) if vals.size else 0.0, 1),
            "mm_h_medio": round(float(mm_h(vals).mean()) if vals.size else 0.0, 2),
            "cobertura_eco_pct": round(float((vals >= nf.UMBRAL_ECO_DBZ).mean() * 100.0) if vals.size else 0.0, 1),
        }
    return salida


def llegadas(actual: np.ndarray, flujo: dict[str, Any], localidades: list[dict],
             campos: dict[int, np.ndarray] | None = None, max_min: int = 60) -> dict[str, dict[str, Any]]:
    """Primera llegada de eco >=35 dBZ (y >=45) a cada localidad en <= 60 min."""
    h, w = actual.shape
    km_px = nf.km_por_pixel(w)
    zonas = _zonas(actual.shape, localidades)
    campos = dict(campos or {})
    res = {k: {"eta_min": None, "dbz_al_llegar": None, "eta_fuerte_min": None, "dbz_max_60min": 0.0}
           for k in zonas}
    for t in range(0, max_min + 1, nf.PASO_ETA_MIN):
        if t == 0:
            campo = actual
        else:
            if t not in campos:
                campos[t] = nf.advectar(actual, flujo["u"], flujo["v"], float(t))
            campo = campos[t]
        for loc_id, (zona, _, _) in zonas.items():
            m = _pico(campo[zona])
            r = res[loc_id]
            r["dbz_max_60min"] = max(r["dbz_max_60min"], round(m, 1))
            if t == 0:
                r["ya_sobre_la_localidad"] = m >= ECO_LLEGADA_DBZ
                continue
            if r["eta_min"] is None and not r["ya_sobre_la_localidad"] and m >= ECO_LLEGADA_DBZ:
                r["eta_min"] = t
                r["dbz_al_llegar"] = round(m, 1)
            if r["eta_fuerte_min"] is None and m >= ECO_FUERTE_DBZ:
                r["eta_fuerte_min"] = t
    # Velocidad del eco que llega, medida donde está ahora (aguas arriba).
    for loc_id, (zona, cx, cy) in zonas.items():
        eta = res[loc_id]["eta_min"]
        if eta is None:
            continue
        x, y = cx, cy
        for _ in range(max(1, eta // 2)):
            ix = int(np.clip(round(x), 0, w - 1))
            iy = int(np.clip(round(y), 0, h - 1))
            x -= float(flujo["u"][iy, ix]) * 2.0
            y -= float(flujo["v"][iy, ix]) * 2.0
        ix = int(np.clip(round(x), 0, w - 1))
        iy = int(np.clip(round(y), 0, h - 1))
        u = float(flujo["u"][iy, ix])
        v = float(flujo["v"][iy, ix])
        res[loc_id]["velocidad_eco_kmh"] = round(math.hypot(u, v) * km_px * 60.0, 1)
        res[loc_id]["distancia_eco_km"] = round(math.hypot(x - cx, y - cy) * km_px, 1)
    return res


def _parse(texto):
    try:
        return datetime.fromisoformat(str(texto).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def acumulados(dt: datetime, obs: dict[str, dict], path: Path = HISTORIAL) -> dict[str, dict[str, float]]:
    """Guarda la lluvia media por localidad y devuelve acumulados y rachas."""
    try:
        datos = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        datos = []
    registros = {r["utc"]: r["loc"] for r in datos if isinstance(r, dict) and "utc" in r and "loc" in r}
    registros[nf.iso(dt)] = {k: [v["mm_h_medio"], v["dbz_p75"]] for k, v in obs.items()}
    limite = dt - timedelta(hours=HISTORIAL_HORAS)
    ordenados = sorted((t, v) for t, v in ((_parse(k), v) for k, v in registros.items()) if t is not None)
    ordenados = [(t, v) for t, v in ordenados if limite <= t <= dt]
    try:
        nf.escribir_json_atomico_lista(path, [{"utc": nf.iso(t), "loc": v} for t, v in ordenados])
    except Exception as exc:
        print(f"AVISO: no se pudo guardar el historial por localidad: {exc}")

    salida = {}
    for loc_id in obs:
        serie = [(t, v[loc_id]) for t, v in ordenados if loc_id in v]

        def suma(horas):
            total, anterior = 0.0, None
            for t, (r, _) in serie:
                minutos = 10.0 if anterior is None else min((t - anterior).total_seconds() / 60.0, 20.0)
                anterior = t
                if dt - timedelta(hours=horas) < t <= dt:
                    total += r * minutos / 60.0
            return round(total, 1)

        racha, fin, previo = 0.0, None, None
        for t, (_, p75) in reversed(serie):
            if previo is not None and (previo - t).total_seconds() / 60.0 > 40.0:
                break
            if p75 < LLUVIA_INTENSA_DBZ:
                break
            fin = fin or t
            racha = (fin - t).total_seconds() / 60.0 + 10.0
            previo = t
        salida[loc_id] = {"lluvia_1h_mm": suma(1), "lluvia_3h_mm": suma(3), "lluvia_6h_mm": suma(6),
                          "minutos_lluvia_intensa": round(racha, 1)}
    return salida


def analizar(dt: datetime, campo: np.ndarray, flujo: dict[str, Any] | None,
             edad_min: float = 0.0, campos: dict[int, np.ndarray] | None = None,
             localidades: list[dict] | None = None,
             historial: Path = HISTORIAL) -> dict[str, dict[str, Any]]:
    """Todo lo del radar para cada localidad (con o sin movimiento calculable)."""
    localidades = localidades if localidades is not None else cargar_localidades()
    if not localidades:
        return {}
    obs = observar(campo, localidades)
    acum = acumulados(dt, obs, historial)
    lleg = llegadas(campo, flujo, localidades, campos) if flujo and flujo.get("valido") else {}
    salida = {}
    for loc in localidades:
        k = loc["id"]
        item = {"nombre": loc["nombre"], "lat": loc["lat"], "lon": loc["lon"], **obs[k], **acum[k]}
        if k in lleg:
            l = dict(lleg[k])
            if l.get("eta_min") is not None:
                l["eta_desde_ahora_min"] = round(max(l["eta_min"] - edad_min, 0.0), 1)
            item["llegada"] = l
        salida[k] = item
    return salida
