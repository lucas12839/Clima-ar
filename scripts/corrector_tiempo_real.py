#!/usr/bin/env python3
"""Corrector del pronóstico con datos reales mientras falta el radar (punto 5b+).

El pronóstico del radar (movido, con escenarios y ANVIL) es una suposición. Los rayos
(cada 20 s) y el satélite (cada 10 min) son datos MEDIDOS de ahora: este corrector los
usa para corregir el pronóstico en las dos direcciones, agregar y quitar:

- QUITAR con satélite: donde el pronóstico pone lluvia pero el satélite no ve nube alta
  ni media (tope tibio en ~9 km a la redonda), no puede estar lloviendo: se borra.
  Esto corta, por ejemplo, una lluvia que ANVIL prolongó y que ya paró.
- BAJAR con rayos y satélite: donde el pronóstico pone tormenta (>= 35 dBZ) pero no hay
  rayos cerca y el tope no es frío, se baja a lluvia.
- AGREGAR con rayos: los rayos solos dicen TORMENTA ELÉCTRICA, no lluvia ni viento (puede
  haber rayos sin lluvia). Por eso: (a) los rayos marcan siempre la zona de tormenta
  eléctrica (`zona_electrica`), aparte de la lluvia; (b) solo se agrega LLUVIA de tormenta
  al pronóstico si además el satélite ve un tope frío encima (nube de tormenta que llueve).

Cada regla se encendió solo si mejora en la simulación con el historial real
(herramientas/simular_huecos.py --goes). Todo en la grilla del radar (512 x 512, 240 km).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np

import nowcast_flujo_optico as nf
from scripts.rma10_features import RADAR_LAT, RADAR_LON, RADAR_RADIUS_KM


def a_pixel(lat, lon, w: int, h: int):
    """Como rma10_features.latlon_to_pixel, pero para muchos puntos a la vez."""
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    x = cx + (np.asarray(lon, float) - RADAR_LON) * 111.320 * math.cos(math.radians(RADAR_LAT)) / RADAR_RADIUS_KM * cx
    y = cy - (np.asarray(lat, float) - RADAR_LAT) * 110.574 / RADAR_RADIUS_KM * cy
    return x, y

# GOES-19 (posición fija) para corregir el paralaje (la nube se ve corrida hacia afuera).
PROYECCION_GOES19 = (math.radians(-75.2), 6378137.0, 6356752.31414, 35786023.0 + 6378137.0)
TEMP_SUELO_C = 15.0
GRADIENTE_C_KM = 6.5

REGLAS_DEFECTO = {
    "quitar_sin_nube": True, "tope_tibio_c": -5.0, "radio_tibio_km": 9.0,
    "bajar_sin_rayos": False, "tope_no_frio_c": -30.0, "radio_rayos_bajar_km": 20.0,
    "agregar_con_rayos": True, "rayos_min": 2, "radio_rayos_km": 6.0, "dbz_rayos": 40.0,
    "tope_frio_rayos_c": -35.0,
}


def desplazamiento_paralaje_km_por_km() -> tuple[float, float]:
    """Cuánto se corre (en km de la grilla del radar, x este / y sur) la nube vista por el
    satélite por cada km de altura del tope, en la zona del radar."""
    from goes19.geospatial import posicion_aparente

    lat0, lon0 = RADAR_LAT, RADAR_LON
    la, lo = posicion_aparente(lat0, lon0, 10.0, PROYECCION_GOES19)
    dx = (lo - lon0) * 111.320 * math.cos(math.radians(lat0)) / 10.0
    dy = -(la - lat0) * 111.195 / 10.0   # fila crece hacia el sur
    return dx, dy


_PARALAJE: tuple[float, float] | None = None


def grilla_satelite(lat: np.ndarray, lon: np.ndarray, temp_c: np.ndarray, shape) -> np.ndarray:
    """Temperatura del tope (°C) en la grilla del radar, con el paralaje corregido."""
    import cv2

    global _PARALAJE
    if _PARALAJE is None:
        try:
            _PARALAJE = desplazamiento_paralaje_km_por_km()
        except Exception:
            _PARALAJE = (0.0, 0.0)
    h, w = shape
    km_px = nf.km_por_pixel(w)
    t = np.asarray(temp_c, dtype=np.float32)
    ok = np.isfinite(t) & np.isfinite(lat) & np.isfinite(lon)
    lat, lon, t = np.asarray(lat)[ok], np.asarray(lon)[ok], t[ok]
    x, y = a_pixel(lat, lon, w, h)
    altura = np.clip((TEMP_SUELO_C - t) / GRADIENTE_C_KM, 0.0, 16.0)
    x = np.asarray(x, dtype=np.float32) - altura * _PARALAJE[0] / km_px
    y = np.asarray(y, dtype=np.float32) - altura * _PARALAJE[1] / km_px
    xi, yi = np.rint(x).astype(int), np.rint(y).astype(int)
    dentro = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    grilla = np.full(shape, np.inf, dtype=np.float32)
    np.minimum.at(grilla, (yi[dentro], xi[dentro]), t[dentro])
    # Rellena los huecos entre píxeles del satélite (2 km) con el más frío cercano.
    relleno = cv2.erode(grilla, np.ones((5, 5), np.uint8))
    grilla = np.where(np.isfinite(grilla), grilla, relleno)
    grilla[~np.isfinite(grilla)] = np.nan
    return grilla


def mas_frio_cerca(grilla: np.ndarray, radio_km: float) -> np.ndarray:
    import cv2

    km_px = nf.km_por_pixel(grilla.shape[1])
    r = max(1, int(round(radio_km / km_px)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    g = np.where(np.isfinite(grilla), grilla, np.inf).astype(np.float32)
    return cv2.erode(g, k)


def densidad_rayos(lat: np.ndarray, lon: np.ndarray, shape, radio_km: float) -> np.ndarray:
    """Cantidad de rayos a <= radio_km de cada píxel."""
    import cv2

    h, w = shape
    cuenta = np.zeros(shape, dtype=np.float32)
    if len(lat):
        x, y = a_pixel(lat, lon, w, h)
        xi, yi = np.rint(np.asarray(x)).astype(int), np.rint(np.asarray(y)).astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        np.add.at(cuenta, (yi[ok], xi[ok]), 1.0)
    r = max(1, int(round(radio_km / nf.km_por_pixel(w))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)).astype(np.float32)
    return cv2.filter2D(cuenta, -1, k, borderType=cv2.BORDER_CONSTANT)


def corregir(campo: np.ndarray, satelite: np.ndarray | None, rayos_lat, rayos_lon,
             reglas: dict | None = None) -> tuple[np.ndarray, dict[str, int]]:
    """Corrige el campo pronosticado (dBZ) con el satélite (grilla °C, puede ser None) y los
    rayos de los últimos 10 min. Devuelve (campo, píxeles cambiados por regla)."""
    r = {**REGLAS_DEFECTO, **(reglas or {})}
    c = campo.astype(np.float32).copy()
    cambios = {}
    hay_sat = satelite is not None and np.isfinite(satelite).mean() > 0.5
    rayos_lat = np.asarray(rayos_lat if rayos_lat is not None else [], dtype=float)
    rayos_lon = np.asarray(rayos_lon if rayos_lon is not None else [], dtype=float)
    if hay_sat and r["quitar_sin_nube"]:
        frio = mas_frio_cerca(satelite, r["radio_tibio_km"])
        quitar = (c >= nf.UMBRAL_ECO_DBZ) & (frio > r["tope_tibio_c"]) & np.isfinite(frio)
        c[quitar] = 0.0
        cambios["quitado_sin_nube"] = int(quitar.sum())
    if r["bajar_sin_rayos"] and hay_sat:
        cerca = densidad_rayos(rayos_lat, rayos_lon, c.shape, r["radio_rayos_bajar_km"])
        frio = mas_frio_cerca(satelite, 10.0)
        bajar = (c >= nf.UMBRAL_TORMENTA_DBZ) & (cerca < 0.5) & (frio > r["tope_no_frio_c"])
        c[bajar] = np.minimum(c[bajar], nf.UMBRAL_TORMENTA_DBZ - 5.0)
        cambios["bajado_sin_rayos"] = int(bajar.sum())
    if r["agregar_con_rayos"] and rayos_lat.size and hay_sat:
        dens = densidad_rayos(rayos_lat, rayos_lon, c.shape, r["radio_rayos_km"])
        frio = mas_frio_cerca(satelite, r["radio_rayos_km"])
        agregar = (dens >= r["rayos_min"]) & (c < r["dbz_rayos"]) & (frio <= r["tope_frio_rayos_c"])
        c[agregar] = r["dbz_rayos"]
        cambios["agregado_con_rayos"] = int(agregar.sum())
    return c, cambios


def quitar_ecos_sin_nube(campo: np.ndarray, satelite: np.ndarray | None, tope_tibio_c: float = -5.0,
                         radio_km: float = 9.0) -> np.ndarray:
    """Control de calidad del radar: un eco donde el satélite no ve ninguna nube alta ni
    media en ~9 km es un eco FALSO (propagación anómala de noche, mar, terreno). Probado:
    el 8/11 a las 00:10 (hora de Argentina) el radar marcaba 75 dBZ con el cielo despejado."""
    if satelite is None or np.isfinite(satelite).mean() <= 0.5:
        return campo
    frio = mas_frio_cerca(satelite, radio_km)
    c = campo.copy()
    c[(c > 0) & np.isfinite(frio) & (frio > tope_tibio_c)] = 0.0
    return c


def zona_electrica(rayos_lat, rayos_lon, shape, radio_km: float = 10.0, minimo: int = 1) -> np.ndarray:
    """Dónde hay tormenta eléctrica AHORA (rayos a <= radio_km en los últimos 10 min).
    Es un fenómeno aparte: no dice nada de lluvia ni de viento."""
    return densidad_rayos(np.asarray(rayos_lat, float), np.asarray(rayos_lon, float), shape, radio_km) >= minimo


def elegir_satelite(inicios: list[datetime], hasta: datetime, latencia_min: float = 8.0) -> int | None:
    """Índice del último barrido que ya estaría publicado a la hora `hasta`."""
    mejor = None
    for i, t in enumerate(inicios):
        if t <= hasta - timedelta(minutes=latencia_min) and (mejor is None or t > inicios[mejor]):
            mejor = i
    return mejor


def rayos_ventana(fines: list[datetime] | np.ndarray, lat, lon, hasta: datetime, minutos: float = 10.0):
    fines = np.asarray(fines)
    sel = (fines > np.datetime64(hasta - timedelta(minutes=minutos))) & (fines <= np.datetime64(hasta))
    return np.asarray(lat)[sel], np.asarray(lon)[sel]


def resumen(cambios: dict[str, int]) -> dict[str, Any]:
    return {k: v for k, v in cambios.items() if v}
