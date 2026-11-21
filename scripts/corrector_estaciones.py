#!/usr/bin/env python3
"""Etapa A, punto 1: corrector del pronóstico con la lluvia MEDIDA por las estaciones.

Las estaciones BCP miden cada minuto y publican cada ~5 min (CWOP cada ~10): es lo más
fresco que hay en el suelo mientras el radar llega 16 min tarde. Con eso:

- AGREGAR: donde una estación midió lluvia fuerte en los últimos 10 min y el pronóstico
  no la tiene, se pone al menos la intensidad medida (convertida a dBZ con Marshall-Palmer)
  en un disco chico alrededor de la estación.
- BAJAR: donde una estación con datos frescos no midió NADA en los últimos 15 min y el
  pronóstico dice que ahí llueve fuerte desde hace 10 min o más, se saca esa lluvia en
  un disco chico (un pluviómetro que no marcó en 15 min no puede estar bajo 30 dBZ).
- Para los plazos que vienen, la corrección se mueve con la tormenta (mismo movimiento
  del radar), como el corrector de rayos y satélite.

La lluvia en una ventana sale del acumulado diario (BCP se reinicia a la medianoche local,
03 UTC). Una bajada que NO cruza la medianoche es un error de lectura (caso La Vitícola
21/11 02:47: 6,0 -> 0,0 -> 6,0) y se ignora.

Se enciende solo si mejora en la simulación (herramientas/simular_etapa_a.py).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

import nowcast_capas_nucleo as nc
import nowcast_flujo_optico as nf

HUSO_LOCAL = timezone(timedelta(hours=-3))
PARAMETROS_DEFECTO: dict[str, Any] = {
    "agregar": True, "lluvia_agregar_mm_10min": 0.6, "radio_agregar_km": 4.0, "dbz_max": 50.0,
    "bajar": True, "ventana_bajar_min": 15.0, "dbz_bajar": 30.0, "radio_bajar_km": 3.0,
    "edad_max_min": 15.0, "puntos_min": 3,
}


def parse_utc(texto: Any) -> datetime | None:
    if isinstance(texto, datetime):
        return texto if texto.tzinfo else texto.replace(tzinfo=timezone.utc)
    if not texto:
        return None
    try:
        v = datetime.fromisoformat(str(texto).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (v if v.tzinfo else v.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def lluvia_ventana(puntos: list[tuple[datetime, float]], desde: datetime, hasta: datetime
                   ) -> tuple[float | None, float]:
    """(mm caídos entre `desde` y `hasta`, minutos cubiertos). Usa el último dato antes de
    `desde` como referencia. Bajadas sin cruzar la medianoche local = error, se ignoran."""
    pts = sorted((t, v) for t, v in puntos if t is not None and v is not None and t <= hasta)
    previos = [p for p in pts if p[0] <= desde]
    dentro = [p for p in pts if p[0] > desde]
    serie = ([previos[-1]] if previos else []) + dentro
    if len(serie) < 2:
        return None, 0.0
    total, ref = 0.0, serie[0]
    for t, v in serie[1:]:
        if v >= ref[1]:
            total += v - ref[1]
            ref = (t, v)
        elif t.astimezone(HUSO_LOCAL).date() != ref[0].astimezone(HUSO_LOCAL).date():
            total += v          # reinicio diario: cuenta lo nuevo
            ref = (t, v)
        # si no: lectura errónea, se mantiene la referencia
    cubierto = (serie[-1][0] - max(serie[0][0], desde - timedelta(minutes=10))).total_seconds() / 60.0
    return round(total, 2), cubierto


def dbz_de_lluvia(mm_h: float) -> float:
    """Marshall-Palmer: Z = 200 R^1.6."""
    return 10.0 * math.log10(200.0 * max(mm_h, 1e-3) ** 1.6)


def mediciones(est: dict, ahora: datetime, p: dict) -> list[dict[str, Any]]:
    """Por estación: lluvia de los últimos 10 min y de la ventana para bajar, con su posición."""
    salida = []
    dudosas = set(est.get("dudosas") or [])
    for nombre, serie in (est.get("series") or {}).items():
        if nombre in dudosas or nombre not in (est.get("coords") or {}):
            continue
        pts = [(parse_utc(t), v) for t, v in serie]
        pts = [(t, float(v)) for t, v in pts if t is not None and v is not None and t <= ahora]
        if len(pts) < p["puntos_min"]:
            continue
        ultima = max(t for t, _ in pts)
        if (ahora - ultima).total_seconds() / 60.0 > p["edad_max_min"]:
            continue
        l10, cub10 = lluvia_ventana(pts, ultima - timedelta(minutes=10), ultima)
        lb, cubb = lluvia_ventana(pts, ultima - timedelta(minutes=p["ventana_bajar_min"]), ultima)
        n_b = sum(1 for t, _ in pts if t > ultima - timedelta(minutes=p["ventana_bajar_min"]))
        lat, lon = est["coords"][nombre]
        salida.append({"nombre": nombre, "lat": lat, "lon": lon, "ultima": ultima,
                       "edad_min": (ahora - ultima).total_seconds() / 60.0,
                       "lluvia_10min_mm": l10, "lluvia_bajar_mm": lb,
                       "puntos_bajar": n_b, "minutos_bajar": cubb})
    return salida


def a_pixel(lat: float, lon: float, shape) -> tuple[float, float]:
    import corrector_tiempo_real as ct

    x, y = ct.a_pixel(np.array([lat]), np.array([lon]), shape[1], shape[0])
    return float(x[0]), float(y[0])


def corregir_campos(campos: dict[float, np.ndarray], est: dict | None, ahora: datetime, edad_radar: float,
                    flujo: dict | None, shape, parametros: dict | None = None,
                    campo_radar: np.ndarray | None = None, info: dict | None = None
                    ) -> dict[float, np.ndarray]:
    """`campos` = {minutos desde la imagen de radar: dBZ}. Devuelve los campos corregidos.
    `campo_radar` = la imagen de radar limpia: con ella se mira si el pronóstico ya decía
    lluvia fuerte en la estación 10 min antes de ahora (regla de bajar)."""
    p = {**PARAMETROS_DEFECTO, **{k: v for k, v in (parametros or {}).items() if k in PARAMETROS_DEFECTO}}
    info = info if info is not None else {}
    if not est or not campos:
        info["usado"] = False
        return campos
    med = mediciones(est, ahora, p)
    info.update({"usado": True, "estaciones": len(med), "agregadas": [], "bajadas": []})
    if not med:
        return campos
    km_px = nf.km_por_pixel(shape[1])
    acciones = []   # (x, y, radio_px, "agregar"/"bajar", dBZ)
    hace10 = None
    if campo_radar is not None:
        hace10 = campo_radar if (edad_radar <= 10 or flujo is None) else \
            nc.serie(campo_radar, flujo["u"], flujo["v"], [float(edad_radar) - 10.0])[float(edad_radar) - 10.0]
    for m in med:
        x, y = a_pixel(m["lat"], m["lon"], shape)
        if not (0 <= x < shape[1] and 0 <= y < shape[0]):
            continue
        if p["agregar"] and m["lluvia_10min_mm"] is not None and m["lluvia_10min_mm"] >= p["lluvia_agregar_mm_10min"]:
            dbz = min(p["dbz_max"], dbz_de_lluvia(m["lluvia_10min_mm"] * 6.0))
            acciones.append((x, y, p["radio_agregar_km"] / km_px, "agregar", dbz))
            info["agregadas"].append({"estacion": m["nombre"], "mm_10min": m["lluvia_10min_mm"], "dbz": round(dbz, 1)})
        elif (p["bajar"] and m["lluvia_bajar_mm"] == 0.0 and m["puntos_bajar"] >= 2
              and m["minutos_bajar"] >= p["ventana_bajar_min"] - 2):
            ahora_c = campos.get(float(edad_radar))
            if ahora_c is None:
                continue
            disco = nf.disco(shape, x, y, p["radio_bajar_km"] / km_px)
            fuerte_ahora = float(np.max(ahora_c[disco])) >= p["dbz_bajar"] if disco.any() else False
            fuerte_antes = True if hace10 is None else (float(np.max(hace10[disco])) >= p["dbz_bajar"]
                                                         if disco.any() else False)
            if fuerte_ahora and fuerte_antes:
                acciones.append((x, y, p["radio_bajar_km"] / km_px, "bajar", 0.0))
                info["bajadas"].append({"estacion": m["nombre"], "minutos_sin_lluvia": round(m["minutos_bajar"], 1)})
    if not acciones:
        return campos
    salida = {}
    for t, c in campos.items():
        plazo = float(t) - float(edad_radar)   # minutos desde ahora (la corrección se mueve)
        c2 = c.astype(np.float32).copy()
        for x, y, r, tipo, dbz in acciones:
            xx, yy = x, y
            if plazo > 0.5 and flujo is not None:
                xi = int(min(max(round(x), 0), shape[1] - 1))
                yi = int(min(max(round(y), 0), shape[0] - 1))
                xx, yy = x + float(flujo["u"][yi, xi]) * plazo, y + float(flujo["v"][yi, xi]) * plazo
            elif plazo < -0.5:
                continue   # antes de la medición no se toca
            d = nf.disco(shape, xx, yy, r)
            if tipo == "agregar":
                c2[d] = np.maximum(c2[d], dbz)
            else:
                c2[d] = 0.0
        salida[t] = c2
    return salida


def desde_historial(historial: dict, coords: dict, dudosas=()) -> dict:
    """Arma la entrada del corrector con data/ia/estaciones/historial_superficie.json."""
    series = {}
    for nombre, filas in (historial or {}).items():
        s = [(r.get("utc"), r.get("lluvia_dia")) for r in filas or [] if r.get("lluvia_dia") is not None]
        if s:
            series[nombre] = s
    return {"series": series, "coords": coords, "dudosas": list(dudosas)}
