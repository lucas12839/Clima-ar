#!/usr/bin/env python3
"""Potencial del ambiente para tormentas severas (modelo europeo ECMWF).

No dice que haya tormenta: dice qué puede dar una tormenta si se forma o si
llega. Se calcula con el viento y la temperatura en altura:

- cizalladura 850-500 hPa (cambio del viento entre ~1,5 y ~5,5 km): con
  mucha cizalladura las tormentas se organizan (líneas, supercélulas) y duran;
- gradiente térmico 850-500 hPa: cuánto se enfría el aire con la altura;
- aire seco en niveles medios (700/500 hPa): cuando la lluvia cae en aire
  seco lo enfría y genera ráfagas descendentes fuertes, aunque llueva poco;
- CAPE: energía disponible para la tormenta.

Comprobado con el ambiente de los temporales (data/eventos_severos):
16/12/2023 (viento destructivo) CAPE 750-1100 J/kg y cizalladura 62-67 km/h
-> favorable a tormentas severas; 7/3/2025 (inundación) CAPE < 650 J/kg ->
no favorable a viento/granizo (fue lluvia persistente).
"""
from __future__ import annotations

import math
from typing import Any

CAPE_MODERADO = 500.0
CAPE_ALTO = 1000.0
CAPE_MUY_ALTO = 2000.0
CIZALLADURA_ORGANIZADA_KMH = 55.0    # ~15 m/s
CIZALLADURA_FUERTE_KMH = 75.0        # ~20 m/s
GRADIENTE_PRONUNCIADO = 6.5          # °C/km
HUMEDAD_SECA_MEDIA = 45.0            # % en 700 o 500 hPa
ESPESOR_850_500_KM = 4.0             # aproximado para el gradiente


def numero(valor: Any) -> float | None:
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def componentes(velocidad: float, direccion_desde: float) -> tuple[float, float]:
    r = math.radians(direccion_desde)
    return -velocidad * math.sin(r), -velocidad * math.cos(r)


def cizalladura_kmh(amb: dict[str, Any]) -> float | None:
    v8, d8 = numero(amb.get("wind_speed_850hPa")), numero(amb.get("wind_direction_850hPa"))
    v5, d5 = numero(amb.get("wind_speed_500hPa")), numero(amb.get("wind_direction_500hPa"))
    if None in (v8, d8, v5, d5):
        return None
    u8, w8 = componentes(v8, d8)
    u5, w5 = componentes(v5, d5)
    return math.hypot(u5 - u8, w5 - w8)


def evaluar(amb: dict[str, Any] | None) -> dict[str, Any]:
    amb = amb or {}
    cape = numero(amb.get("cape"))
    shear = cizalladura_kmh(amb)
    t8, t5 = numero(amb.get("temperature_850hPa")), numero(amb.get("temperature_500hPa"))
    gradiente = (t8 - t5) / ESPESOR_850_500_KM if t8 is not None and t5 is not None else None
    hum = [h for h in (numero(amb.get("relative_humidity_700hPa")), numero(amb.get("relative_humidity_500hPa")))
           if h is not None]
    seco = min(hum) if hum else None

    razones = []
    severo = False
    rafagas = False
    if cape is not None and shear is not None:
        if cape >= CAPE_ALTO and shear >= CIZALLADURA_ORGANIZADA_KMH:
            severo = True
            razones.append(f"energía alta (CAPE {cape:.0f} J/kg) y cizalladura {shear:.0f} km/h: "
                           "las tormentas pueden organizarse y ser severas")
        elif cape >= CAPE_MODERADO and shear >= CIZALLADURA_FUERTE_KMH:
            severo = True
            razones.append(f"cizalladura muy fuerte ({shear:.0f} km/h) con CAPE {cape:.0f} J/kg: "
                           "líneas de tormenta con viento")
    elif cape is not None and cape >= CAPE_MUY_ALTO:
        severo = True
        razones.append(f"energía muy alta (CAPE {cape:.0f} J/kg)")
    if (cape is not None and cape >= CAPE_MODERADO and seco is not None and seco <= HUMEDAD_SECA_MEDIA
            and (gradiente is None or gradiente >= GRADIENTE_PRONUNCIADO)):
        rafagas = True
        razones.append(f"aire seco en altura ({seco:.0f} % de humedad): favorece ráfagas descendentes")
    if severo or rafagas:
        nivel = "favorable_severas"
        etiqueta = "Ambiente favorable a tormentas severas"
    elif cape is not None and cape >= CAPE_MODERADO:
        nivel = "favorable_tormentas"
        etiqueta = "Ambiente favorable a tormentas"
    elif cape is None and shear is None:
        nivel = "sin_datos"
        etiqueta = "Sin datos del ambiente"
    else:
        nivel = "poco_favorable"
        etiqueta = "Ambiente poco favorable a tormentas"
    return {
        "nivel": nivel,
        "etiqueta": etiqueta,
        "favorable_severas": severo,
        "favorable_rafagas_descendentes": rafagas,
        "cape": cape,
        "cizalladura_850_500_kmh": round(shear, 1) if shear is not None else None,
        "gradiente_850_500_c_km": round(gradiente, 2) if gradiente is not None else None,
        "humedad_media_minima_pct": seco,
        "razones": razones,
    }
