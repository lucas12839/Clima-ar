"""Rayos por tormenta (punto 3): salto de rayos y tamaño de los rayos.

1. Asignación: cada rayo (GLM) se asigna a UNA sola tormenta, la que tenía
   más cerca en el minuto del rayo (con su movimiento), siempre que caiga
   dentro de su radio. Así dos tormentas vecinas no cuentan el mismo rayo.

2. Salto de rayos, algoritmo "2σ" (Schultz y otros 2009, 2011; el estándar
   propuesto para GLM):
   - los rayos de los 14 min que terminan en la imagen de radar se agrupan en
     7 tramos de 2 min y se calcula la frecuencia (rayos por minuto) de cada uno;
   - DFRDT = cambio de frecuencia entre tramos seguidos (rayos/min²): 6 valores;
   - umbral = 2 desvíos estándar de los 5 DFRDT anteriores;
   - hay salto si el DFRDT más reciente supera el umbral y la frecuencia
     actual es de al menos 10 rayos/min (filtra tormentas comunes).
   Los estudios con datos tipo GLM sugieren que el umbral de 10 rayos/min
   puede ser alto para GLM: se informa también el nivel en desvíos para
   calibrarlo con casos reales (punto 4 y simulaciones), sin cambiar el método.

3. Tamaño de los rayos: los rayos chicos (< 300 km²) aparecen en la
   convección más intensa; el área mínima y la fracción de rayos chicos
   suben cuando la corriente ascendente se intensifica (NOAA, guía GLM).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

KM_POR_GRADO_LAT = 111.195
VENTANA_SALTO_MIN = 14.0
TRAMO_MIN = 2.0
SIGMA = 2.0
UMBRAL_ACTIVACION_POR_MIN = 10.0
VENTANA_CONTEO_MIN = 10.0
RAYO_CHICO_KM2 = 300.0
COBERTURA_MIN_SALTO = 12.0   # al menos 12 de los 14 min con archivos GLM


def _posicion(tormenta: dict, edad_min: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Posición de la tormenta `edad_min` minutos antes de la imagen de radar."""
    lat = tormenta["latitud"] - tormenta["norte_km_min"] * edad_min / KM_POR_GRADO_LAT
    lon = tormenta["longitud"] - tormenta["este_km_min"] * edad_min / (
        KM_POR_GRADO_LAT * math.cos(math.radians(tormenta["latitud"])))
    return lat, lon


def _distancia(lat1, lon1, lat2, lon2):
    r = 6371.0088
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def asignar(rayos: dict | None, tormentas: list[dict]) -> np.ndarray:
    """Índice de la tormenta de cada rayo (-1 = ninguna). Solo rayos de la
    ventana (edad 0 a 14 min, nunca posteriores a la imagen de radar)."""
    if not rayos or not tormentas:
        n = 0 if not rayos else np.asarray(rayos.get("minutos", [])).size
        return np.full(n, -1, dtype=int)
    edad = np.asarray(rayos["minutos"], dtype=float)
    lat = np.asarray(rayos["latitude"], dtype=float)
    lon = np.asarray(rayos["longitude"], dtype=float)
    mejor = np.full(edad.size, np.inf)
    asignado = np.full(edad.size, -1, dtype=int)
    valido = (edad >= 0) & (edad < VENTANA_SALTO_MIN)
    for i, t in enumerate(tormentas):
        lat_t, lon_t = _posicion(t, edad)
        relativa = _distancia(lat, lon, lat_t, lon_t) / t["radio_km"]
        mejor_aca = valido & (relativa <= 1.0) & (relativa < mejor)
        asignado[mejor_aca] = i
        mejor[mejor_aca] = relativa[mejor_aca]
    return asignado


def salto_2sigma(edades_min: np.ndarray, cobertura_min: float) -> dict[str, Any]:
    """Algoritmo 2σ sobre las edades (min antes de la imagen de radar) de los
    rayos de UNA tormenta."""
    if cobertura_min < COBERTURA_MIN_SALTO:
        return {"estado": "cobertura_insuficiente", "cobertura_min": round(cobertura_min, 1)}
    edades = np.asarray(edades_min, dtype=float)
    n_tramos = int(VENTANA_SALTO_MIN / TRAMO_MIN)
    # Tramo 0 = el más viejo (12-14 min antes), tramo 6 = el más reciente (0-2 min).
    tasas = []
    for k in range(n_tramos):
        hasta = VENTANA_SALTO_MIN - k * TRAMO_MIN
        desde = hasta - TRAMO_MIN
        tasas.append(float(np.sum((edades >= desde) & (edades < hasta))) / TRAMO_MIN)
    dfrdt = [(b - a) / TRAMO_MIN for a, b in zip(tasas, tasas[1:])]
    previos, actual = dfrdt[:-1], dfrdt[-1]
    sigma = float(np.std(previos))
    umbral = SIGMA * sigma
    tasa_actual = tasas[-1]
    nivel = (actual / sigma) if sigma > 0 else None
    salto = bool(tasa_actual >= UMBRAL_ACTIVACION_POR_MIN and actual > umbral and actual > 0)
    return {
        "estado": "ok",
        "metodo": "2σ (Schultz 2009, 2011): tramos de 2 min en 14 min",
        "tasas_por_min": [round(x, 2) for x in tasas],
        "dfrdt_por_min2": [round(x, 2) for x in dfrdt],
        "dfrdt_actual": round(actual, 2),
        "umbral_2sigma": round(umbral, 2),
        "nivel_sigma": round(nivel, 2) if nivel is not None else None,
        "tasa_actual_por_min": round(tasa_actual, 2),
        "umbral_activacion_por_min": UMBRAL_ACTIVACION_POR_MIN,
        "salto": salto,
    }


def tamano(area_km2: np.ndarray, energia_fj: np.ndarray) -> dict[str, Any]:
    area = np.asarray(area_km2, dtype=float)
    area = area[np.isfinite(area)]
    energia = np.asarray(energia_fj, dtype=float)
    energia = energia[np.isfinite(energia)]
    if area.size == 0:
        return {"cantidad": 0}
    return {
        "cantidad": int(area.size),
        "area_minima_km2": round(float(area.min()), 1),
        "area_mediana_km2": round(float(np.median(area)), 1),
        "area_media_km2": round(float(area.mean()), 1),
        "fraccion_chicos": round(float(np.mean(area < RAYO_CHICO_KM2)), 3),
        "umbral_chico_km2": RAYO_CHICO_KM2,
        "energia_total_fj": round(float(energia.sum()), 1) if energia.size else None,
    }


def por_tormenta(rayos: dict | None, asignado: np.ndarray, indice: int, frame_iso: str | None) -> dict[str, Any]:
    """Rayos de una tormenta: conteo de 10 min (como antes), salto 2σ y tamaño."""
    if not rayos or rayos.get("cobertura_min", 0) < 5:
        return {"estado": "no_disponible", "ventana_min": VENTANA_CONTEO_MIN, "fin_utc": frame_iso}
    edad = np.asarray(rayos["minutos"], dtype=float)
    mios = asignado == indice
    recientes = mios & (edad >= 0) & (edad < VENTANA_CONTEO_MIN)
    cobertura_total = float(rayos["cobertura_min"])
    # Cobertura de los últimos 10 min: la ventana completa es de 14 min.
    ventana_total = float(rayos.get("ventana_min", VENTANA_CONTEO_MIN))
    cobertura = min(VENTANA_CONTEO_MIN, cobertura_total * VENTANA_CONTEO_MIN / max(ventana_total, 1e-9))
    cantidad = int(np.sum(recientes))
    salida = {
        "estado": "ok",
        "ventana_min": VENTANA_CONTEO_MIN,
        "fin_utc": frame_iso,
        "cobertura_min": round(cobertura, 1),
        "cantidad": cantidad,
        "tasa_por_min": round(cantidad / cobertura, 2) if cobertura > 0 else None,
        "asignacion": "exclusiva (tormenta más cercana en el minuto del rayo)",
    }
    salida["salto_rayos"] = salto_2sigma(edad[mios], cobertura_total if ventana_total >= VENTANA_SALTO_MIN else 0.0)
    area = np.asarray(rayos.get("area_km2", np.full(edad.size, np.nan)), dtype=float)
    energia = np.asarray(rayos.get("energia_fj", np.full(edad.size, np.nan)), dtype=float)
    salida["tamano_rayos"] = tamano(area[recientes], energia[recientes])
    return salida
