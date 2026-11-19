"""Señales de tormenta severa por satélite (GOES-19), por tormenta.

1. Topes sobresalientes: una corriente ascendente muy fuerte perfora el yunque
   y forma una "burbuja" más fría que el yunque que la rodea. Método de textura
   del infrarrojo (Bedka y otros, 2010), sin entrenamiento:
   - píxel del canal 13 a <= -58 °C (215 K) y el más frío a 4 km a la redonda;
   - el yunque a ~8 km (anillo de 6 a 10 km) es frío (<= -48 °C, 225 K) en al
     menos la mitad de sus píxeles;
   - el píxel está >= 6,5 °C más frío que el promedio del yunque.
   Son de las mejores señales de granizo grande y ráfagas dañinas.
2. Diferencia vapor de agua - infrarrojo (canal 8 - canal 13): positiva cuando
   el tope llega a la estratosfera (vapor inyectado por encima del yunque).
3. Altura del tope (detalle 2 km) por encima del nivel de congelamiento del
   modelo: cuanto más alto sube la tormenta por encima de 0 °C, más espacio
   para formar granizo.
4. Fase del tope: hielo, mixta o agua sobreenfriada (el congelamiento del tope
   marca una tormenta madura o que se está formando).

Estadística pura: las funciones reciben los campos ya recortados a la zona.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

K = 273.15
OT_MAX_K = 215.0            # -58 °C
OT_MIN_DIF_K = 6.5          # más frío que el yunque
YUNQUE_MAX_K = 225.0        # -48 °C
YUNQUE_FRACCION_MIN = 0.5
RADIO_MINIMO_LOCAL_KM = 4.0
ANILLO_KM = (6.0, 10.0)

FASES = {0: "despejado", 1: "agua", 2: "sobreenfriada", 3: "mixta", 4: "hielo", 5: "desconocida"}


def _xy_km(lat, lon, lat0, lon0):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = (lon - lon0) * 111.320 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 111.195
    return x, y


def detectar_topes_sobresalientes(bt_k, lat, lon) -> list[dict]:
    """Topes sobresalientes en un campo del canal 13 (posiciones como las ve
    el satélite). Devuelve [{latitud, longitud, bt_c, yunque_c, diferencia_c}]."""
    bt = np.asarray(bt_k, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ok = np.isfinite(bt) & np.isfinite(lat) & np.isfinite(lon)
    bt, lat, lon = bt[ok], lat[ok], lon[ok]
    candidatos = np.where(bt <= OT_MAX_K)[0]
    if candidatos.size == 0:
        return []
    from scipy.spatial import cKDTree

    lat0, lon0 = float(np.mean(lat)), float(np.mean(lon))
    x, y = _xy_km(lat, lon, lat0, lon0)
    arbol = cKDTree(np.column_stack([x, y]))
    hallados = []
    for i in candidatos:
        cerca = arbol.query_ball_point([x[i], y[i]], RADIO_MINIMO_LOCAL_KM)
        if bt[i] > np.min(bt[cerca]):
            continue  # no es el más frío del lugar
        anillo = np.asarray(arbol.query_ball_point([x[i], y[i]], ANILLO_KM[1]))
        d = np.hypot(x[anillo] - x[i], y[anillo] - y[i])
        anillo = anillo[d >= ANILLO_KM[0]]
        if anillo.size == 0:
            continue
        frios = bt[anillo] <= YUNQUE_MAX_K
        if np.mean(frios) < YUNQUE_FRACCION_MIN:
            continue
        yunque = float(np.mean(bt[anillo][frios]))
        dif = yunque - float(bt[i])
        if dif < OT_MIN_DIF_K:
            continue
        hallados.append({
            "latitud": round(float(lat[i]), 4),
            "longitud": round(float(lon[i]), 4),
            "bt_c": round(float(bt[i]) - K, 1),
            "yunque_c": round(yunque - K, 1),
            "diferencia_c": round(dif, 1),
        })
    # Dos píxeles iguales muy juntos son el mismo tope.
    unicos: list[dict] = []
    for ot in sorted(hallados, key=lambda o: o["bt_c"]):
        if all(math.hypot(*(_xy_km(ot["latitud"], ot["longitud"], u["latitud"], u["longitud"]))) > RADIO_MINIMO_LOCAL_KM
               for u in unicos):
            unicos.append(ot)
    return unicos


def diferencia_vapor_infrarrojo(campo_c08: dict | None, campo_c13: dict | None) -> dict | None:
    """Campo canal 8 - canal 13 (K). Solo si son del mismo barrido y la misma
    grilla (píxel a píxel)."""
    if not campo_c08 or not campo_c13:
        return None
    h08, h13 = campo_c08.get("hora_satelite"), campo_c13.get("hora_satelite")
    # Mismo barrido: la hora de paso por Bahía coincide (difiere en ~1 s).
    if h08 is None or h13 is None or abs((h08 - h13).total_seconds()) > 5:
        return None
    a, b = np.asarray(campo_c08["latitude"]), np.asarray(campo_c13["latitude"])
    if a.shape != b.shape or not np.allclose(a, b, equal_nan=True):
        return None
    return {
        "valores": np.asarray(campo_c08["valores"], dtype=float) - np.asarray(campo_c13["valores"], dtype=float),
        "latitude": campo_c13["latitude"],
        "longitude": campo_c13["longitude"],
        "proyeccion": campo_c13.get("proyeccion"),
        "hora_satelite": campo_c13.get("hora_satelite"),
    }


def resumen_fase(valores: np.ndarray) -> dict:
    v = np.asarray(valores, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"pixeles": 0}
    salida: dict[str, Any] = {"pixeles": int(v.size)}
    for codigo, nombre in FASES.items():
        salida[f"fraccion_{nombre}"] = round(float(np.mean(v == codigo)), 3)
    nubes = v[(v >= 1) & (v <= 4)]
    salida["fraccion_hielo_en_nubes"] = round(float(np.mean(nubes == 4)), 3) if nubes.size else None
    return salida


def nivel_congelamiento_para(ambiente: dict | None, lat: float, lon: float, frame) -> dict:
    """Nivel de congelamiento del modelo en el punto de grilla más cercano a
    la tormenta y la hora más cercana a la imagen de radar."""
    puntos = ((ambiente or {}).get("ambiente_futuro") or {}).get("puntos") or []
    mejor = None
    for punto in puntos:
        plat, plon = punto.get("latitud"), punto.get("longitud")
        if plat is None or plon is None:
            continue
        d = math.hypot(*_xy_km(plat, plon, lat, lon))
        for registro in punto.get("datos") or []:
            valor = registro.get("freezing_level_height")
            if valor is None:
                continue
            try:
                from datetime import datetime, timezone

                t = datetime.fromisoformat(str(registro.get("tiempo_utc")).replace("Z", "+00:00"))
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                horas = abs((t - frame).total_seconds()) / 3600.0
            except (TypeError, ValueError):
                continue
            if horas > 1.5:
                continue
            clave = (round(d), horas)
            if mejor is None or clave < mejor[0]:
                mejor = (clave, float(valor), punto.get("punto"), registro.get("tiempo_utc"), d)
    if mejor is None:
        return {"estado": "no_disponible"}
    return {
        "estado": "ok",
        "metros": round(mejor[1]),
        "punto": mejor[2],
        "hora_modelo_utc": mejor[3],
        "distancia_punto_km": round(mejor[4], 1),
    }
