"""Punto 5 del plan: nubes que se están convirtiendo en tormenta, vistas por
GOES-19 antes de que el radar tenga eco.

Junta lo mejor de los métodos que se usan en el mundo, todos sin modelos
entrenados:
- RDT (Météo-France / NWCSAF): objetos de nube fría seguidos en el tiempo,
  con su enfriamiento y su fase.
- UW-CI (Universidad de Wisconsin, Sieglaff 2011): enfriamiento del tope
  medido de forma que el simple movimiento de la nube no cuente como
  crecimiento, más el paso del tope a hielo.
- SATCAST / GOES-R CI (Mecikalski y Bedka 2006): "campos de interés":
  tope que baja de 0 °C, enfriamiento en 10 y 20 min, vapor de agua menos
  infrarrojo.

Enfriamiento "a prueba de movimiento": cada píxel de ahora se compara con
el píxel MÁS FRÍO que había a su alrededor (12 km) en la imagen de 10 min
antes (20 km para la de 20 min antes). Si la nube solo se movió, ese mínimo
ya era igual de frío y no hay enfriamiento. Si el píxel está más frío que
todo lo que había cerca, la nube creció de verdad. El yunque que se
expande tampoco cuenta (nunca es más frío que la torre que lo formó), y si
al lado del objeto hay nube igual de fría que no creció, es el borde de un
yunque o de una banda de nubes altas que avanza con el viento: se descarta.

Este módulo es estadística pura (sin red). Solo usa satélite y rayos; el
control con el radar, el ambiente, el seguimiento entre ciclos y los grados
de aviso están en scripts/aviso_formacion.py.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np

from .geospatial import RADAR_LAT, RADAR_LON, RADAR_RANGE_KM, haversine_km, posicion_aparente

K = 273.15
CERO_K = 273.15               # tope bajo 0 °C: ya hay nube con hielo posible
ENFRIAMIENTO_MIN_C = 3.0      # en 10 min (UW-CI: 4 K en 15 min ≈ 2,7 K en 10 min)
RADIO_MOV_10_KM = 12.0        # movimiento de hasta ~70 km/h en 10 min
RADIO_MOV_20_KM = 20.0
MIN_PIXELES = 2               # objetos de al menos 2 píxeles (~10 km²)
UNION_KM = 4.5                # píxeles vecinos (2 km de resolución + deformación)
ALTURA_DEFECTO_KM = 6.0       # nube en crecimiento, si no hay altura medida
MAX_OBJETOS = 40
SINCRONIA_S = 5.0             # productos del mismo barrido
BORDE_KM = 8.0                # nube más fría al lado = borde de yunque o de banda alta
BORDE_TOLERANCIA_C = 1.0


def iso(valor: datetime | None) -> str | None:
    return valor.isoformat().replace("+00:00", "Z") if valor else None


def xy_km(lat, lon) -> tuple[np.ndarray, np.ndarray]:
    """Plano local centrado en el radar (km al este y al norte)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = (lon - RADAR_LON) * 111.320 * math.cos(math.radians(RADAR_LAT))
    y = (lat - RADAR_LAT) * 111.195
    return x, y


def posicion_real(lat_ap: float, lon_ap: float, altura_km: float, proyeccion) -> tuple[float, float]:
    """Inversa del paralaje: dónde está en el suelo una nube que el satélite
    ve en (lat_ap, lon_ap) con el tope a altura_km."""
    if proyeccion is None or altura_km <= 0:
        return lat_ap, lon_ap
    lat, lon = lat_ap, lon_ap
    for _ in range(4):
        la, lo = posicion_aparente(lat, lon, altura_km, proyeccion)
        lat, lon = lat + (lat_ap - la), lon + (lon_ap - lo)
    return lat, lon


def minimo_vecino(xy_prev: np.ndarray, t_prev: np.ndarray, xy: np.ndarray, radio_km: float,
                  area_px_km2: float) -> np.ndarray:
    """Para cada punto de `xy`, el valor mínimo de `t_prev` a <= radio_km
    (NaN si no hay datos cerca)."""
    from scipy.spatial import cKDTree

    salida = np.full(len(xy), np.nan)
    validos = np.isfinite(t_prev)
    if len(xy) == 0 or not np.any(validos):
        return salida
    xy_v, t_v = xy_prev[validos], t_prev[validos]
    arbol = cKDTree(xy_v)
    k = int(min(len(t_v), math.pi * radio_km ** 2 / max(area_px_km2, 1.0) * 1.6 + 12))
    k = max(k, 1)
    relleno = np.append(t_v, np.inf)
    for inicio in range(0, len(xy), 4000):   # por partes, para no usar mucha memoria
        parte = xy[inicio:inicio + 4000]
        dist, idx = arbol.query(parte, k=k, distance_upper_bound=radio_km)
        if k == 1:
            dist, idx = dist[:, None], idx[:, None]
        valores = relleno[np.minimum(idx, len(t_v))]
        valores[~np.isfinite(dist)] = np.inf
        minimo = valores.min(axis=1)
        minimo[~np.isfinite(minimo)] = np.nan
        salida[inicio:inicio + len(parte)] = minimo
    return salida


def _campo_xy(campo: dict) -> np.ndarray:
    x, y = xy_km(campo["latitude"], campo["longitude"])
    return np.column_stack([x, y])


def _mismo_barrido(campo: dict | None, hora: datetime | None) -> bool:
    if campo is None or hora is None or campo.get("hora_satelite") is None:
        return False
    return abs((campo["hora_satelite"] - hora).total_seconds()) <= SINCRONIA_S


def valores_en(campo: dict | None, xy: np.ndarray, radio_km: float) -> np.ndarray | None:
    """Valores de otro producto del mismo barrido en los puntos `xy` (vecino
    más cercano a <= radio_km)."""
    if campo is None or len(xy) == 0:
        return None
    from scipy.spatial import cKDTree

    xy_c = _campo_xy(campo)
    v = np.asarray(campo["valores"], dtype=float)
    dist, idx = cKDTree(xy_c).query(xy, k=1, distance_upper_bound=radio_km)
    salida = np.full(len(xy), np.nan)
    ok = np.isfinite(dist)
    salida[ok] = v[idx[ok]]
    return salida


def _grupos(xy: np.ndarray) -> np.ndarray:
    """Etiqueta de grupo de cada píxel candidato (píxeles vecinos juntos)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(xy)
    if n == 0:
        return np.zeros(0, dtype=int)
    pares = np.array(list(cKDTree(xy).query_pairs(UNION_KM)), dtype=int).reshape(-1, 2)
    m = coo_matrix((np.ones(len(pares)), (pares[:, 0], pares[:, 1])), shape=(n, n))
    _, etiquetas = connected_components(m, directed=False)
    return etiquetas


def _r(v, dec=1):
    return None if v is None or not np.isfinite(v) else round(float(v), dec)


def detectar(c13: dict | None, c13_10: dict | None, c13_20: dict | None,
             campos: dict[str, dict | None] | None = None, rayos: dict | None = None,
             frame: datetime | None = None) -> dict[str, Any]:
    """Objetos de nube en crecimiento vistos por el satélite.

    c13, c13_10, c13_20: canal 13 de ahora, 10 y 20 min antes, cada uno
        {"valores" (K), "latitude", "longitude", "proyeccion", "hora_satelite"}.
    campos: {"fase", "altura_topes_2km", "altura_topes", "c08"} del mismo barrido.
    rayos: rayos de los 14 min que terminan en la imagen de radar (GLM).
    """
    campos = campos or {}
    base: dict[str, Any] = {"metodo": "RDT + UW-CI + SATCAST, enfriamiento a prueba de movimiento",
                            "objetos": []}
    if c13 is None or c13.get("hora_satelite") is None:
        return {**base, "estado": "sin_satelite"}
    hora = c13["hora_satelite"]
    base["hora_satelite_utc"] = iso(hora)

    previos = {}
    for nombre, campo, esperado in (("10", c13_10, 10.0), ("20", c13_20, 20.0)):
        if campo is None or campo.get("hora_satelite") is None:
            continue
        minutos = (hora - campo["hora_satelite"]).total_seconds() / 60.0
        if abs(minutos - esperado) <= 3.0:
            previos[nombre] = (campo, minutos)
    base["imagenes_previas"] = {k: {"hora_satelite_utc": iso(c["hora_satelite"]), "minutos": round(m, 1)}
                                for k, (c, m) in previos.items()}
    if "10" not in previos:
        return {**base, "estado": "sin_imagen_previa"}

    t = np.asarray(c13["valores"], dtype=float)
    xy_all = _campo_xy(c13)
    dist_radar = haversine_km(np.asarray(c13["latitude"]), np.asarray(c13["longitude"]), RADAR_LAT, RADAR_LON)
    n_validos = int(np.sum(np.isfinite(t)))
    area_px = math.pi * float(np.nanmax(dist_radar)) ** 2 / max(n_validos, 1)

    # Solo nubes con el tope bajo 0 °C (las candidatas).
    cand = np.where(np.isfinite(t) & (t <= CERO_K))[0]
    base["area_pixel_km2"] = round(area_px, 2)
    if cand.size == 0:
        return {**base, "estado": "sin_nubes_frias"}
    xy = xy_all[cand]
    campo10, min10 = previos["10"]
    prev10 = minimo_vecino(_campo_xy(campo10), np.asarray(campo10["valores"], dtype=float), xy,
                           RADIO_MOV_10_KM, area_px)
    d10 = (t[cand] - prev10) * 10.0 / min10            # °C cada 10 min (negativo = enfría)
    if "20" in previos:
        campo20, min20 = previos["20"]
        prev20 = minimo_vecino(_campo_xy(campo20), np.asarray(campo20["valores"], dtype=float), xy,
                               RADIO_MOV_20_KM, area_px)
        d20 = (t[cand] - prev20) * 20.0 / min20        # °C cada 20 min
    else:
        prev20 = np.full(cand.size, np.nan)
        d20 = np.full(cand.size, np.nan)

    crece = np.isfinite(d10) & (d10 <= -ENFRIAMIENTO_MIN_C)
    if not np.any(crece):
        return {**base, "estado": "sin_crecimiento", "candidatos": int(cand.size)}
    sel = np.where(crece)[0]
    etiquetas = _grupos(xy[sel])

    fase = campos.get("fase") if _mismo_barrido(campos.get("fase"), hora) else None
    alt2 = campos.get("altura_topes_2km") if _mismo_barrido(campos.get("altura_topes_2km"), hora) else None
    alt10 = campos.get("altura_topes") if _mismo_barrido(campos.get("altura_topes"), hora) else None
    c08 = campos.get("c08") if _mismo_barrido(campos.get("c08"), hora) else None
    proy = c13.get("proyeccion")

    from scipy.spatial import cKDTree

    finitos = np.isfinite(t)
    arbol_actual = cKDTree(xy_all[finitos])
    t_validos = t[finitos]
    posicion_valida = np.cumsum(finitos) - 1     # índice de cada píxel dentro de los válidos
    objetos = []
    descartados_borde = 0
    for g in np.unique(etiquetas):
        miembros = sel[etiquetas == g]
        if miembros.size < MIN_PIXELES:
            continue
        idx = cand[miembros]
        pxy = xy[miembros]
        tt = t[idx]
        # La torre que crece es lo más frío del lugar. Si al lado (en la
        # imagen de ahora) hay nube igual o más fría que NO creció, esto es el
        # borde de un yunque o de una banda de nubes altas que avanza con el
        # viento de altura, no una tormenta nueva.
        vecinos = np.unique(np.concatenate(arbol_actual.query_ball_point(pxy, BORDE_KM)))
        vecinos = np.setdiff1d(vecinos, posicion_valida[idx])
        if vecinos.size and int(np.sum(t_validos[vecinos] <= float(np.min(tt)) + BORDE_TOLERANCIA_C)) >= 3:
            descartados_borde += 1
            continue
        lat_ap = float(np.mean(np.asarray(c13["latitude"])[idx]))
        lon_ap = float(np.mean(np.asarray(c13["longitude"])[idx]))
        area = miembros.size * area_px

        altura = None
        origen_altura = None
        for campo_alt, radio, nombre in ((alt2, 3.0, "2km"), (alt10, 7.0, "10km")):
            v = valores_en(campo_alt, pxy, radio)
            if v is not None and np.any(np.isfinite(v)):
                altura = float(np.nanpercentile(v, 90)) / 1000.0
                origen_altura = nombre
                break
        altura_paralaje = float(np.clip(altura if altura is not None else ALTURA_DEFECTO_KM, 1.0, 16.0))
        lat, lon = posicion_real(lat_ap, lon_ap, altura_paralaje, proy)
        dist = float(haversine_km(np.array([lat]), np.array([lon]), RADAR_LAT, RADAR_LON)[0])
        if dist > RADAR_RANGE_KM:
            continue

        hielo = None
        vf = valores_en(fase, pxy, 3.0)
        if vf is not None:
            nubes = vf[np.isfinite(vf) & (vf >= 1) & (vf <= 4)]
            if nubes.size:
                hielo = float(np.mean(nubes == 4))
                sobreenfriada = float(np.mean((nubes == 2) | (nubes == 3)))
            else:
                sobreenfriada = None
        else:
            sobreenfriada = None

        vapor_ir = None
        v8 = valores_en(c08, pxy, 1.5)
        if v8 is not None and np.any(np.isfinite(v8)):
            vapor_ir = float(np.nanmedian(v8 - tt))

        p10 = prev10[miembros]
        p20 = prev20[miembros]
        cruce0 = bool(np.any(np.isfinite(p20) & (p20 > CERO_K)) or np.any(np.isfinite(p10) & (p10 > CERO_K)))

        radio_eq = math.sqrt(area / math.pi)
        rayos_obj = _rayos_en(rayos, lat, lon, radio_eq + 5.0)

        objetos.append({
            "latitud": round(lat, 4),
            "longitud": round(lon, 4),
            "latitud_satelite": round(lat_ap, 4),
            "longitud_satelite": round(lon_ap, 4),
            "distancia_radar_km": round(dist, 1),
            "pixeles": int(miembros.size),
            "area_km2": round(area, 1),
            "radio_km": round(radio_eq, 1),
            "tope_min_c": _r(float(np.min(tt)) - K),
            "tope_p10_c": _r(float(np.percentile(tt, 10)) - K),
            "enfriamiento_10min_c": _r(-float(np.median(d10[miembros]))),
            "enfriamiento_10min_max_c": _r(-float(np.min(d10[miembros]))),
            "enfriamiento_20min_c": _r(-float(np.nanmedian(d20[miembros])))
            if np.any(np.isfinite(d20[miembros])) else None,
            "cruce_0c": cruce0,
            "fraccion_hielo": _r(hielo, 2),
            "fraccion_sobreenfriada": _r(sobreenfriada, 2),
            "altura_tope_km": _r(altura, 2),
            "origen_altura": origen_altura,
            "altura_paralaje_km": round(altura_paralaje, 1),
            "vapor_menos_infrarrojo_c": _r(vapor_ir),
            "rayos": rayos_obj,
        })

    objetos.sort(key=lambda o: -(o["enfriamiento_10min_c"] or 0.0) * o["pixeles"] ** 0.5)
    return {**base, "estado": "ok" if objetos else "sin_crecimiento",
            "candidatos": int(cand.size), "pixeles_en_crecimiento": int(sel.size),
            "descartados_borde_de_nube_fria": descartados_borde,
            "objetos": objetos[:MAX_OBJETOS]}


def _rayos_en(rayos: dict | None, lat: float, lon: float, radio_km: float) -> dict[str, Any]:
    if not rayos or float(rayos.get("cobertura_min") or 0.0) < 5.0:
        return {"estado": "no_disponible"}
    la = np.asarray(rayos.get("latitude", []), dtype=float)
    lo = np.asarray(rayos.get("longitude", []), dtype=float)
    edad = np.asarray(rayos.get("minutos", []), dtype=float)
    if la.size == 0:
        return {"estado": "ok", "cantidad": 0, "cobertura_min": round(float(rayos["cobertura_min"]), 1)}
    d = haversine_km(la, lo, lat, lon)
    dentro = (d <= radio_km) & (edad >= 0)
    return {
        "estado": "ok",
        "cantidad": int(np.sum(dentro)),
        "ultimos_5min": int(np.sum(dentro & (edad < 5.0))),
        "cobertura_min": round(float(rayos["cobertura_min"]), 1),
    }
