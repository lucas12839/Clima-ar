#!/usr/bin/env python3
"""Paso 2: cada celda de tormenta seguida con sus rayos, minuto a minuto.

El radar llega con 20-30 min de atraso; los rayos del satélite (GLM), con
~1 min. Para las tormentas con actividad eléctrica:

1. Celdas AHORA: los rayos de los últimos 10 min se agrupan (a <= 6 km unos de
   otros); cada grupo es una celda con su centro (pesando más los rayos más
   nuevos), su tamaño (dónde cayó el 90 % de sus rayos) y su actividad.
2. Movimiento de cada celda: el del 5b en su lugar (radar, 3 imágenes); si sus
   propios rayos muestran otro desplazamiento creíble, se mezcla.
3. Probabilidad de que pase SOBRE cada localidad en los próximos 30 min:
   muchos caminos posibles (lugar ± error, velocidad ± 25 %, dirección ± 20°);
   el porcentaje es la fracción de caminos que la tocan. También el minuto
   más probable de llegada.

Sin entrenamiento: física (movimiento) y estadística (caminos). Las celdas sin
rayos (solo lluvia) no entran: para eso sigue el 5b por zona de 10 km.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

VENTANA_MIN = 10.0
RADIO_UNION_KM = 6.0
MIN_RAYOS = 3
VIDA_MEDIA_PESO_MIN = 4.0
VELOCIDAD_MAX_KMH = 120.0

PARAMETROS_DEFECTO = {
    "horizonte_min": 30.0,
    "paso_min": 2.0,
    "caminos": 200,
    "margen_km": 3.0,          # la lluvia fuerte se extiende un poco más allá de los rayos
    "radio_min_km": 4.0,
    "error_lugar_km": 3.0,     # error del centro de los rayos
    "error_velocidad": 0.25,   # fracción
    "error_direccion_grados": 20.0,
    "peso_movimiento_rayos": 0.3,
}


def _agrupar(x: np.ndarray, y: np.ndarray, shape, km_px: float, radio_union_km: float) -> np.ndarray:
    """Etiqueta (1..n) de cada rayo: juntos si caen a <= radio_union_km."""
    import cv2

    h, w = shape
    xi, yi = np.rint(x).astype(int).clip(0, w - 1), np.rint(y).astype(int).clip(0, h - 1)
    mascara = np.zeros(shape, dtype=np.uint8)
    mascara[yi, xi] = 1
    r = max(1, int(round(radio_union_km / 2.0 / km_px)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    _, etiquetas = cv2.connectedComponents(cv2.dilate(mascara, k))
    return etiquetas[yi, xi]


def _celda(cx_: np.ndarray, cy_: np.ndarray, e: np.ndarray, km_px: float) -> dict[str, Any]:
    peso = np.exp(-e / VIDA_MEDIA_PESO_MIN)
    cx, cy = float(np.average(cx_, weights=peso)), float(np.average(cy_, weights=peso))
    d = np.hypot(cx_ - cx, cy_ - cy) * km_px
    return {"x": cx, "y": cy, "rayos": int(e.size), "rayos_5min": int((e <= 5.0).sum()),
            "radio_km": float(np.percentile(d, 90)) if d.size > 1 else 0.0,
            "edad_ultimo_rayo_min": float(e.min())}


def celdas(x: np.ndarray, y: np.ndarray, edad_min: np.ndarray, shape, km_px: float,
           ventana_min: float = VENTANA_MIN, radio_union_km: float = RADIO_UNION_KM,
           min_rayos: int = MIN_RAYOS) -> list[dict[str, Any]]:
    """Agrupa los rayos (posición en píxeles del radar, edad en minutos) en celdas.

    Las celdas se arman con los rayos de los últimos 5 min (dónde está AHORA). Los
    rayos de 5 a 10 min se suman a la celda más cercana (hasta 15 km: una tormenta
    rápida se corre ~10 km en 10 min) y dan su movimiento propio; los que no tienen
    celda nueva cerca forman celdas que se están apagando."""
    x, y, edad = (np.asarray(a, dtype=float) for a in (x, y, edad_min))
    h, w = shape
    sel = (edad >= 0) & (edad <= ventana_min) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x, y, edad = x[sel], y[sel], edad[sel]
    if x.size < min_rayos:
        return []
    nuevos = edad <= 5.0
    salida: list[dict[str, Any]] = []
    asignados = np.zeros(x.size, dtype=bool)
    if nuevos.sum() >= min_rayos:
        xn, yn, en = x[nuevos], y[nuevos], edad[nuevos]
        lab = _agrupar(xn, yn, shape, km_px, radio_union_km)
        idx_n = np.where(nuevos)[0]
        for i in np.unique(lab):
            m = lab == i
            if m.sum() < min_rayos:
                continue
            c = _celda(xn[m], yn[m], en[m], km_px)
            c["_idx"] = list(idx_n[m])
            salida.append(c)
            asignados[idx_n[m]] = True
    # Rayos de 5 a 10 min: a la celda nueva más cercana (<= 15 km).
    viejos_idx = np.where(~nuevos)[0]
    if salida and viejos_idx.size:
        centros = np.array([[c["x"], c["y"]] for c in salida])
        d = np.hypot(x[viejos_idx, None] - centros[None, :, 0], y[viejos_idx, None] - centros[None, :, 1]) * km_px
        cerca = d.min(axis=1) <= 15.0
        cual = d.argmin(axis=1)
        for k, c in enumerate(salida):
            mios = viejos_idx[cerca & (cual == k)]
            asignados[mios] = True
            if mios.size >= 5 and len(c["_idx"]) >= 5:
                nn = np.asarray(c["_idx"])
                dt = float(np.mean(edad[mios]) - np.mean(edad[nn]))
                if dt > 2.0:
                    vx = (float(np.mean(x[nn])) - float(np.mean(x[mios]))) / dt
                    vy = (float(np.mean(y[nn])) - float(np.mean(y[mios]))) / dt
                    if math.hypot(vx, vy) * km_px * 60.0 <= VELOCIDAD_MAX_KMH:
                        c["movimiento_rayos_px_min"] = (vx, vy)
            c["rayos"] += int(mios.size)
    for c in salida:
        c.pop("_idx", None)
    # Celdas que se apagan: rayos viejos sin celda nueva cerca.
    resto = np.where(~asignados)[0]
    if resto.size >= min_rayos:
        lab = _agrupar(x[resto], y[resto], shape, km_px, radio_union_km)
        for i in np.unique(lab):
            m = lab == i
            if m.sum() >= min_rayos:
                c = _celda(x[resto][m], y[resto][m], edad[resto][m], km_px)
                c["apagandose"] = True
                salida.append(c)
    return salida


def movimiento_celda(celda: dict, u: np.ndarray | None, v: np.ndarray | None, km_px: float,
                     peso_rayos: float) -> tuple[float, float]:
    """px/min: el del 5b alrededor de la celda, mezclado con el de sus rayos."""
    if u is None or v is None:
        base = (0.0, 0.0)
    else:
        h, w = u.shape
        r = max(2, int(round(max(celda["radio_km"], 4.0) / km_px)))
        x0, y0 = int(round(celda["x"])), int(round(celda["y"]))
        ys, xs = slice(max(0, y0 - r), min(h, y0 + r + 1)), slice(max(0, x0 - r), min(w, x0 + r + 1))
        base = (float(np.mean(u[ys, xs])), float(np.mean(v[ys, xs]))) if u[ys, xs].size else (0.0, 0.0)
    propio = celda.get("movimiento_rayos_px_min")
    if propio is None or peso_rayos <= 0:
        return base
    return ((1 - peso_rayos) * base[0] + peso_rayos * propio[0],
            (1 - peso_rayos) * base[1] + peso_rayos * propio[1])


def probabilidades(lista: list[dict], u: np.ndarray | None, v: np.ndarray | None, km_px: float,
                   localidades: dict[str, tuple[float, float, float]], parametros: dict | None = None,
                   semilla: int = 7) -> dict[str, dict[str, Any]]:
    """Para cada localidad {id: (x, y, radio_px)}: probabilidad de que alguna celda
    con rayos pase sobre ella en el horizonte, y minuto más probable de llegada."""
    p = {**PARAMETROS_DEFECTO, **(parametros or {})}
    rng = np.random.default_rng(semilla)
    n = int(p["caminos"])
    tiempos = np.arange(0.0, p["horizonte_min"] + 1e-6, p["paso_min"])
    resultado = {k: {"probabilidad": 0.0, "minutos": None, "celdas": 0} for k in localidades}
    no_toca = {k: 1.0 for k in localidades}
    llegadas: dict[str, list[tuple[float, float]]] = {k: [] for k in localidades}
    for celda in lista:
        vx, vy = movimiento_celda(celda, u, v, km_px, p["peso_movimiento_rayos"])
        rapidez, angulo = math.hypot(vx, vy), math.atan2(vy, vx)
        f = 1.0 + p["error_velocidad"] * rng.standard_normal(n)
        a = angulo + math.radians(p["error_direccion_grados"]) * rng.standard_normal(n)
        sx = celda["x"] + p["error_lugar_km"] / km_px * rng.standard_normal(n)
        sy = celda["y"] + p["error_lugar_km"] / km_px * rng.standard_normal(n)
        vxs, vys = rapidez * f * np.cos(a), rapidez * f * np.sin(a)
        px = sx[:, None] + vxs[:, None] * tiempos[None, :]
        py = sy[:, None] + vys[:, None] * tiempos[None, :]
        radio_px = (max(celda["radio_km"], p["radio_min_km"]) + p["margen_km"]) / km_px
        for loc_id, (lx, ly, lr) in localidades.items():
            toca = np.hypot(px - lx, py - ly) <= radio_px + lr          # (caminos, tiempos)
            alguna = toca.any(axis=1)
            frac = float(alguna.mean())
            if frac <= 0:
                continue
            primero = tiempos[np.argmax(toca, axis=1)][alguna]
            no_toca[loc_id] *= (1.0 - frac)
            llegadas[loc_id].append((frac, float(np.median(primero))))
            resultado[loc_id]["celdas"] += 1
    for loc_id in localidades:
        prob = 1.0 - no_toca[loc_id]
        resultado[loc_id]["probabilidad"] = round(prob, 3)
        if llegadas[loc_id]:
            # minuto de la celda que más probablemente la toca
            resultado[loc_id]["minutos"] = round(max(llegadas[loc_id])[1], 0)
    return resultado


# ---------------------------------------------------------------------------
# Celdas del radar (todas, con o sin rayos), corregidas con los rayos de ahora
# ---------------------------------------------------------------------------

PARAMETROS_RADAR = {
    "umbral_dbz": 35.0,
    "min_pixeles": 4,
    "distancia_union_rayos_km": 15.0,   # celda del radar movida a "ahora" <-> celda de rayos
    "error_por_minuto_km": 0.25,        # error del lugar que crece con el atraso del radar
    "error_lugar_con_rayos_km": 3.0,
    # Idea 2 (aviso de la localidad exacta): si se da, solo cuentan las celdas del radar
    # con un pico de al menos este valor (tormenta fuerte) o con rayos encima ahora.
    "solo_fuertes_dbz": None,
}


def celdas_radar(campo: np.ndarray, u: np.ndarray | None, v: np.ndarray | None, km_px: float,
                 umbral_dbz: float = 35.0, min_pixeles: int = 4) -> list[dict[str, Any]]:
    """Cada núcleo de >= umbral dBZ de la imagen del radar con su forma y su movimiento."""
    import cv2

    mascara = (np.nan_to_num(campo) >= umbral_dbz).astype(np.uint8)
    n, etiquetas, stats, centros = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    salida = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_pixeles:
            continue
        m = etiquetas == i
        vx = float(np.mean(u[m])) if u is not None else 0.0
        vy = float(np.mean(v[m])) if v is not None else 0.0
        # Distancia (px) de cada punto de la grilla a la celda: sirve para ver si la
        # celda, corrida, toca una localidad (sin suponer que es redonda).
        dist = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)
        salida.append({"x": float(centros[i][0]), "y": float(centros[i][1]),
                       "pixeles": int(stats[i, cv2.CC_STAT_AREA]), "vx": vx, "vy": vy, "dist": dist,
                       "max_dbz": float(np.nanmax(campo[m]))})
    return salida


def _distancia_en(dist: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = dist.shape
    xi = np.rint(x).astype(int)
    yi = np.rint(y).astype(int)
    fuera = (xi < 0) | (xi >= w) | (yi < 0) | (yi >= h)
    d = dist[yi.clip(0, h - 1), xi.clip(0, w - 1)]
    return np.where(fuera, np.inf, d)


def probabilidades_celdas(campo: np.ndarray, u: np.ndarray | None, v: np.ndarray | None, km_px: float,
                          atraso_min: float, rayos: list[dict], localidades: dict[str, tuple[float, float, float]],
                          parametros: dict | None = None, semilla: int = 7) -> dict[str, dict[str, Any]]:
    """Probabilidad de que una celda pase SOBRE cada localidad en el horizonte, desde ahora.

    Celdas del radar: se mueven desde la imagen (vieja) hasta ahora y después; si una
    celda de rayos de ahora coincide, el lugar de ahora sale de los rayos (corrige el
    atraso del radar). Celdas de rayos sin celda en el radar (nacieron después): discos.
    """
    p = {**PARAMETROS_DEFECTO, **PARAMETROS_RADAR, **(parametros or {})}
    rng = np.random.default_rng(semilla)
    n = int(p["caminos"])
    tiempos = np.arange(0.0, p["horizonte_min"] + 1e-6, p["paso_min"])
    no_toca = {k: 1.0 for k in localidades}
    llegadas: dict[str, list[tuple[float, float]]] = {k: [] for k in localidades}
    usadas = set()
    for c in celdas_radar(campo, u, v, km_px, p["umbral_dbz"], int(p["min_pixeles"])):
        # Dónde estaría ahora según el radar.
        ax, ay = c["x"] + c["vx"] * atraso_min, c["y"] + c["vy"] * atraso_min
        mejor, dmin = None, p["distancia_union_rayos_km"] / km_px
        for j, r in enumerate(rayos):
            d = math.hypot(r["x"] - ax, r["y"] - ay)
            if d <= dmin:
                mejor, dmin = j, d
        if mejor is not None:
            usadas.add(mejor)
        fuerte = p["solo_fuertes_dbz"]
        if fuerte is not None and mejor is None and c["max_dbz"] < float(fuerte):
            continue  # Idea 2: ni fuerte ni con rayos -> la cubre la zona de 10 km
        rapidez, angulo = math.hypot(c["vx"], c["vy"]), math.atan2(c["vy"], c["vx"])
        f = 1.0 + p["error_velocidad"] * rng.standard_normal(n)
        a = angulo + math.radians(p["error_direccion_grados"]) * rng.standard_normal(n)
        vxs, vys = rapidez * f * np.cos(a), rapidez * f * np.sin(a)
        if mejor is not None:
            r = rayos[mejor]
            sigma = p["error_lugar_con_rayos_km"] / km_px
            ox = (r["x"] - c["x"]) + sigma * rng.standard_normal(n)
            oy = (r["y"] - c["y"]) + sigma * rng.standard_normal(n)
        else:
            sigma = p["error_por_minuto_km"] * atraso_min / km_px
            ox = vxs * atraso_min + sigma * rng.standard_normal(n)
            oy = vys * atraso_min + sigma * rng.standard_normal(n)
        dx = ox[:, None] + vxs[:, None] * tiempos[None, :]
        dy = oy[:, None] + vys[:, None] * tiempos[None, :]
        for loc_id, (lx, ly, lr) in localidades.items():
            toca = _distancia_en(c["dist"], lx - dx, ly - dy) <= lr + p["margen_km"] / km_px
            alguna = toca.any(axis=1)
            frac = float(alguna.mean())
            if frac <= 0:
                continue
            no_toca[loc_id] *= (1.0 - frac)
            llegadas[loc_id].append((frac, float(np.median(tiempos[np.argmax(toca, axis=1)][alguna]))))
    nuevas = [r for j, r in enumerate(rayos) if j not in usadas]
    if nuevas:
        extra = probabilidades(nuevas, u, v, km_px, localidades, p, semilla + 1)
        for loc_id, e in extra.items():
            if e["probabilidad"] > 0:
                no_toca[loc_id] *= (1.0 - e["probabilidad"])
                llegadas[loc_id].append((e["probabilidad"], e["minutos"] or 0.0))
    return {loc_id: {"probabilidad": round(1.0 - no_toca[loc_id], 3),
                     "minutos": round(max(llegadas[loc_id])[1], 0) if llegadas[loc_id] else None,
                     "celdas": len(llegadas[loc_id]),
                     "celdas_corregidas_con_rayos": len(usadas)}
            for loc_id in localidades}
