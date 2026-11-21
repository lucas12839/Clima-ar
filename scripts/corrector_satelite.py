#!/usr/bin/env python3
"""Etapa A (compensar el atraso del radar), puntos 2, 3 y 4: lo que mide GOES-19 después
de la última imagen de radar. Funciones puras, en la grilla del radar (512 x 512, 240 km).

SINARAME llega con ~16 min de atraso; la imagen del canal 13 de GOES-19 se publica a los
~10 min de empezar el barrido y ve la zona a los ~8 min. Por eso, cuando se arma el
pronóstico, el satélite casi siempre es 8-20 min MÁS NUEVO que el radar. Eso es dato
medido, no una suposición:

- Punto 2, posición de ahora: entre el barrido de la hora del radar y el más nuevo, los
  topes fríos se movieron lo que de verdad se movió cada tormenta. Ese corrimiento medido
  reemplaza (en la zona con topes fríos) al que suponía el movimiento del radar en ese
  tramo. Así se corrige dónde está cada tormenta ahora, también las que no tienen rayos.
- Punto 3, movimiento: el movimiento de los topes entre los dos barridos más nuevos
  controla el del radar para lo que viene (mezcla, solo donde hay topes fríos).
- Punto 4, crece o se debilita: siguiendo a cada tope, cuánto se enfrió o se calentó
  entre los dos barridos (medido) y si los rayos subieron o bajaron cerca de la tormenta.
  Sube o baja la intensidad que dice el radar viejo. Solo lo medido: no se prolonga la
  tendencia hacia adelante.

Cada parte se enciende solo si mejora en la simulación con el historial real
(herramientas/simular_etapa_a.py). Falla segura: si falta algo, devuelve lo del radar.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

import nowcast_flujo_optico as nf
import nowcast_capas_nucleo as nc

# El canal 13 ve la zona del radar ~8 min después de empezar el barrido y se publica
# ~10 min después (medido en vivo: barrido 03:30:21, zona 03:38:27, disponible 03:40).
VE_ZONA_MIN = 8.1
PUBLICADO_MIN = 10.0

PARAMETROS_DEFECTO: dict[str, Any] = {
    # topes que se siguen: más fríos que esto (°C)
    "tope_c": -20.0,
    # punto 2: cuánto del corrimiento medido se usa (1 = todo)
    "peso_posicion": 1.0,
    # tramo mínimo entre radar y satélite para corregir la posición (min)
    "tramo_min": 4.0,
    # punto 3: cuánto del movimiento medido por el satélite se usa para lo que viene
    "peso_movimiento": 0.0,
    # diferencia máxima satélite - radar que se acepta (km/h); más es un error
    "diferencia_max_kmh": 60.0,
    "coherencia_min": 0.3,
    # punto 4: dBZ por °C de enfriamiento medido (0 = apagado) y tope del cambio
    "dbz_por_c": 0.0,
    "cambio_max_dbz": 10.0,
    # punto 4: dBZ por cada rayo/min de aumento cerca de la tormenta (0 = apagado)
    "dbz_por_rayo_min": 0.0,
    # punto 4: usar solo lo que baja (calentamiento) o solo lo que sube (enfriamiento)
    "crecimiento_sentido": "ambos",
    # puntos 2 y 3 por TORMENTA (no por flujo de nubes): "flujo" o "nucleos"
    "metodo": "flujo",
    "nucleo_c": -40.0,          # el núcleo frío de la tormenta tiene que ser más frío que esto
    "nucleo_margen_c": 4.0,     # píxeles hasta 4 °C más tibios que el mínimo = núcleo
    "busqueda_km": 12.0,        # se busca el núcleo nuevo a esta distancia de donde lo pone el radar
    "corrimiento_max_km": 15.0, # corrección máxima de posición
    "area_min_px": 30,
    "usar_rayos": False,        # posición también con el centro de los rayos
    "rayos_min": 5,
}


def _centro(mascara: np.ndarray, peso: np.ndarray | None = None) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mascara)
    if xs.size == 0:
        return None
    w = np.ones(xs.size) if peso is None else np.maximum(peso[ys, xs], 1e-3)
    return float(np.average(xs, weights=w)), float(np.average(ys, weights=w))


def nucleo(sat: np.ndarray, zona: np.ndarray, p: dict) -> tuple[tuple[float, float], float] | None:
    """Centro del núcleo frío (lo más frío de la zona, hasta `nucleo_margen_c` más tibio)."""
    vals = np.where(zona & np.isfinite(sat), sat, np.inf)
    tmin = float(vals.min())
    if not math.isfinite(tmin) or tmin > p["nucleo_c"]:
        return None
    c = _centro(vals <= tmin + p["nucleo_margen_c"], -(vals - tmin - p["nucleo_margen_c"] - 1.0))
    return (c, tmin) if c else None


def corrimientos_por_tormenta(campo: np.ndarray, u: np.ndarray, v: np.ndarray, sat_radar, sat_nuevo,
                              tramo: float, km_px: float, p: dict, rayos: dict | None = None
                              ) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Puntos 2 por tormenta: dónde está cada tormenta a la hora del barrido nuevo según su
    NÚCLEO frío (y/o el centro de sus rayos), contra donde la pone el movimiento del radar.
    La diferencia (acotada) se aplica corriendo esa tormenta en la imagen de radar; después
    todo se mueve como siempre. Devuelve (campo corrido, lista de correcciones)."""
    import cv2

    eco = (campo >= nf.UMBRAL_ECO_DBZ).astype(np.uint8)
    n, lab = cv2.connectedComponents(cv2.dilate(eco, np.ones((3, 3), np.uint8)), connectivity=8)
    salida = campo.copy()
    lista = []
    r_zona = max(1, int(round(8.0 / km_px)))
    k_zona = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_zona + 1, 2 * r_zona + 1))
    lim = p["corrimiento_max_km"] / km_px
    h, w = campo.shape
    for i in range(1, n):
        m = (lab == i) & (eco > 0)
        if int(m.sum()) < p["area_min_px"] or float(campo[m].max()) < 30.0:
            continue
        ur, vr = float(u[m].mean()), float(v[m].mean())
        medidas = []
        if sat_radar is not None and sat_nuevo is not None and tramo >= p["tramo_min"]:
            zona0 = cv2.dilate(m.astype(np.uint8), k_zona) > 0
            n0 = nucleo(sat_radar, zona0, p)
            if n0 is not None:
                (x0, y0), t0 = n0
                xe, ye = x0 + ur * tramo, y0 + vr * tramo
                zona1 = nf.disco(campo.shape, xe, ye, p["busqueda_km"] / km_px)
                n1 = nucleo(sat_nuevo, zona1, p)
                if n1 is not None and abs(n1[1] - t0) <= 25.0:
                    medidas.append((n1[0][0] - xe, n1[0][1] - ye, "nucleo_satelite"))
        if p["usar_rayos"] and rayos is not None and rayos.get("minutos", 0) >= p["tramo_min"]:
            import corrector_tiempo_real as ct

            zona0 = cv2.dilate(m.astype(np.uint8), k_zona) > 0
            x0s, y0s = ct.a_pixel(rayos["lat0"], rayos["lon0"], w, h)
            sel0 = [(a, b) for a, b in zip(np.atleast_1d(x0s), np.atleast_1d(y0s))
                    if 0 <= a < w and 0 <= b < h and zona0[int(b), int(a)]]
            if len(sel0) >= p["rayos_min"]:
                x0, y0 = float(np.mean([a for a, _ in sel0])), float(np.mean([b for _, b in sel0]))
                xe, ye = x0 + ur * rayos["minutos"], y0 + vr * rayos["minutos"]
                x1s, y1s = ct.a_pixel(rayos["lat1"], rayos["lon1"], w, h)
                r1 = p["busqueda_km"] / km_px
                sel1 = [(a, b) for a, b in zip(np.atleast_1d(x1s), np.atleast_1d(y1s)) if math.hypot(a - xe, b - ye) <= r1]
                if len(sel1) >= p["rayos_min"]:
                    medidas.append((float(np.mean([a for a, _ in sel1])) - xe,
                                    float(np.mean([b for _, b in sel1])) - ye, "rayos"))
        if not medidas:
            continue
        dx = float(np.mean([d[0] for d in medidas])) * p["peso_posicion"]
        dy = float(np.mean([d[1] for d in medidas])) * p["peso_posicion"]
        norma = math.hypot(dx, dy)
        if norma > lim:
            dx, dy = dx * lim / norma, dy * lim / norma
        if math.hypot(dx, dy) < 0.5:
            continue
        objeto = np.where(m, campo, 0.0).astype(np.float32)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        movido = cv2.warpAffine(objeto, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0.0)
        salida[m] = 0.0
        salida = np.maximum(salida, movido)
        lista.append({"x": round(float(np.mean(np.nonzero(m)[1])), 1), "y": round(float(np.mean(np.nonzero(m)[0])), 1),
                      "corrimiento_km": round(math.hypot(dx, dy) * km_px, 1),
                      "fuentes": sorted({d[2] for d in medidas})})
    return salida.astype(np.float32), lista


def frio(grilla: np.ndarray | None, tope_c: float) -> np.ndarray | None:
    """Topes fríos como 'pseudo dBZ': 20 en `tope_c`, sube 1 por cada °C más frío.
    Así el mismo flujo óptico del radar (umbral 20) sigue a los topes."""
    if grilla is None:
        return None
    t = np.nan_to_num(np.asarray(grilla, dtype=np.float32), nan=30.0)
    return np.clip(20.0 + (tope_c - t), 0.0, 70.0).astype(np.float32)


def flujo_satelite(sat_a: np.ndarray | None, sat_b: np.ndarray | None, minutos: float,
                   km_px: float, p: dict) -> dict[str, Any] | None:
    """Movimiento de los topes fríos entre dos barridos (px/min), con la zona donde vale."""
    import cv2

    if sat_a is None or sat_b is None or minutos is None or float(minutos) < 3.0:
        return None
    minutos = float(minutos)
    fa, fb = frio(sat_a, p["tope_c"]), frio(sat_b, p["tope_c"])
    f = nf.estimar_flujo(fa, fb, minutos, km_px)
    if not f["valido"]:
        return None
    # Tope casi quieto: la coherencia (dirección) no está definida y no es un error.
    if f["coherencia"] < p["coherencia_min"] and f["velocidad_media_kmh"] >= 5.0:
        return None
    # El flujo de a->b queda en el lugar de la imagen vieja; el de b->a (al revés) en el de la
    # nueva. El promedio cubre todo el camino (si no, el corrimiento sale corto).
    r = nf.estimar_flujo(fb, fa, minutos, km_px)
    if r["valido"]:
        f["u"] = (0.5 * (f["u"] - r["u"])).astype(np.float32)
        f["v"] = (0.5 * (f["v"] - r["v"])).astype(np.float32)
    zona = ((fa >= 20.0) | (fb >= 20.0)).astype(np.float32)
    peso = np.clip(cv2.GaussianBlur(zona, (0, 0), 6) * 3.0, 0.0, 1.0)
    return {"u": f["u"], "v": f["v"], "peso": peso.astype(np.float32), "minutos": minutos,
            "velocidad_kmh": f["velocidad_media_kmh"], "direccion_hacia_grados": f["direccion_hacia_grados"],
            "coherencia": f["coherencia"]}


def mezclar(u: np.ndarray, v: np.ndarray, fs: dict | None, peso: float, km_px: float,
            dif_max_kmh: float) -> tuple[np.ndarray, np.ndarray]:
    """u + peso·w·(u_sat - u): solo donde hay topes fríos; diferencias imposibles no entran."""
    if fs is None or peso <= 0:
        return u, v
    du, dv = fs["u"] - u, fs["v"] - v
    tope = dif_max_kmh / 60.0 / km_px
    dif = np.hypot(du, dv)
    escala = np.where(dif > tope, tope / np.maximum(dif, 1e-6), 1.0)
    w = peso * fs["peso"]
    return (u + w * du * escala).astype(np.float32), (v + w * dv * escala).astype(np.float32)


def cambio_medido(sat_a: np.ndarray | None, sat_b: np.ndarray | None, fs: dict | None,
                  p: dict) -> np.ndarray | None:
    """°C de enfriamiento siguiendo a cada tope entre dos barridos (positivo = se enfrió,
    la tormenta crece). Suavizado a escala de celda; 0 donde no hay topes fríos."""
    import cv2

    if sat_a is None or sat_b is None or fs is None:
        return None
    fa, fb = frio(sat_a, p["tope_c"]), frio(sat_b, p["tope_c"])
    movido = nf.advectar(fa, fs["u"], fs["v"], fs["minutos"])
    zona = (fb >= 20.0) | (movido >= 20.0)
    dif = np.where(zona, fb - movido, 0.0).astype(np.float32)
    peso = cv2.GaussianBlur(zona.astype(np.float32), (0, 0), 4)
    suave = cv2.GaussianBlur(dif, (0, 0), 4)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(peso > 0.05, suave / np.maximum(peso, 1e-3), 0.0).astype(np.float32)


def cambio_rayos(rayos_10: tuple, rayos_previos: tuple, shape, radio_km: float = 15.0) -> np.ndarray:
    """Rayos/min de aumento cerca de cada punto: últimos 10 min menos los 10 min anteriores."""
    import corrector_tiempo_real as ct

    a = ct.densidad_rayos(np.asarray(rayos_10[0]), np.asarray(rayos_10[1]), shape, radio_km)
    b = ct.densidad_rayos(np.asarray(rayos_previos[0]), np.asarray(rayos_previos[1]), shape, radio_km)
    return ((a - b) / 10.0).astype(np.float32)


def aplicar_cambio(campo: np.ndarray, enfriamiento: np.ndarray | None, rayos: np.ndarray | None,
                   p: dict) -> np.ndarray:
    """Punto 4: sube o baja los ecos según lo medido. Nunca crea eco donde no hay."""
    total = np.zeros(campo.shape, dtype=np.float32)
    if enfriamiento is not None and p["dbz_por_c"]:
        total += p["dbz_por_c"] * enfriamiento
    if rayos is not None and p["dbz_por_rayo_min"]:
        total += p["dbz_por_rayo_min"] * rayos
    if not total.any():
        return campo
    total = np.clip(total, -p["cambio_max_dbz"], p["cambio_max_dbz"])
    eco = campo >= nf.UMBRAL_ECO_DBZ
    salida = np.where(eco, campo + total, campo).astype(np.float32)
    salida[eco & (salida < nf.UMBRAL_ECO_DBZ - 5.0)] = 0.0
    return salida


def preparar(campo: np.ndarray, u: np.ndarray, v: np.ndarray, sat_radar: np.ndarray | None,
             sat_nuevo: np.ndarray | None, sat_nuevo_previo: np.ndarray | None,
             tramo_sat_min: float | None, entre_barridos_min: float | None, km_px: float,
             parametros: dict | None = None, rayos: np.ndarray | None = None,
             info: dict | None = None) -> dict[str, Any]:
    """Lo que mide el satélite después de la imagen de radar, listo para pronosticar.

    Devuelve {base, tramo, ua, va, ub, vb}: `base` = el radar llevado hasta la hora que ve
    el barrido más nuevo (`tramo` min después de la imagen) con el movimiento del tramo 1
    (ua, va: corregido con el corrimiento medido de los topes, punto 2) y subido o bajado
    con lo medido (punto 4). Desde ahí sigue con (ub, vb): el movimiento del radar
    controlado por el de los topes entre los dos barridos más nuevos (punto 3)."""
    p = {**PARAMETROS_DEFECTO, **(parametros or {})}
    info = info if info is not None else {}
    tramo = float(tramo_sat_min or 0.0)
    if p["metodo"] == "nucleos":
        return _preparar_nucleos(campo, u, v, sat_radar, sat_nuevo, sat_nuevo_previo, tramo,
                                 entre_barridos_min, km_px, p, rayos, info)
    usar_pos = p["peso_posicion"] > 0 and tramo >= p["tramo_min"]
    fs_pos = flujo_satelite(sat_radar, sat_nuevo, tramo, km_px, p) if usar_pos else None
    fs_mov = None
    if (p["peso_movimiento"] > 0 or (p["dbz_por_c"] and fs_pos is None)) and entre_barridos_min:
        fs_mov = flujo_satelite(sat_nuevo_previo, sat_nuevo, entre_barridos_min, km_px, p)
    info.update({"posicion": fs_pos is not None, "movimiento": fs_mov is not None and p["peso_movimiento"] > 0,
                 "tramo_satelite_min": round(tramo, 1)})
    ua, va = u, v
    base = campo
    if fs_pos is not None:
        ua, va = mezclar(u, v, fs_pos, p["peso_posicion"], km_px, p["diferencia_max_kmh"])
        base = nc.serie(campo, ua, va, [tramo])[tramo]
        info["topes_velocidad_kmh"] = fs_pos["velocidad_kmh"]
        info["topes_direccion_hacia_grados"] = fs_pos["direccion_hacia_grados"]
    elif not (p["dbz_por_c"] or p["dbz_por_rayo_min"]):
        tramo = 0.0
    else:
        base = nc.serie(campo, u, v, [tramo])[tramo] if tramo > 0.5 else campo
    if p["dbz_por_c"] or p["dbz_por_rayo_min"]:
        enf = None
        if p["dbz_por_c"]:
            enf = cambio_medido(sat_radar, sat_nuevo, fs_pos, p) if fs_pos is not None else \
                cambio_medido(sat_nuevo_previo, sat_nuevo, fs_mov, p)
        antes = base
        base = aplicar_cambio(base, enf, rayos if p["dbz_por_rayo_min"] else None, p)
        info["crecimiento"] = base is not antes
    ub, vb = mezclar(u, v, fs_mov, p["peso_movimiento"], km_px, p["diferencia_max_kmh"])
    if fs_mov is not None and p["peso_movimiento"] > 0:
        info["movimiento_topes_kmh"] = fs_mov["velocidad_kmh"]
    return {"campo": campo, "base": base, "tramo": tramo, "ua": ua, "va": va, "ub": ub, "vb": vb}


def _preparar_nucleos(campo, u, v, sat_radar, sat_nuevo, sat_nuevo_previo, tramo, entre, km_px, p, rayos, info):
    """Variante por tormenta: posición con el núcleo frío (y los rayos), crecimiento medido."""
    rayos_pos = rayos if isinstance(rayos, dict) else None
    corrido, lista = (campo, [])
    if p["peso_posicion"] > 0:
        corrido, lista = corrimientos_por_tormenta(campo, u, v, sat_radar, sat_nuevo, tramo, km_px, p, rayos_pos)
    info.update({"posicion": bool(lista), "movimiento": False, "tramo_satelite_min": round(tramo, 1),
                 "tormentas_corridas": lista[:20]})
    base, t_base = corrido, 0.0
    if p["dbz_por_c"] and sat_radar is not None and sat_nuevo is not None and tramo >= p["tramo_min"]:
        fs = flujo_satelite(sat_radar, sat_nuevo, tramo, km_px, p)
        enf = cambio_medido(sat_radar, sat_nuevo, fs, p)
        if enf is not None:
            if p["crecimiento_sentido"] == "baja":
                enf = np.minimum(enf, 0.0)
            elif p["crecimiento_sentido"] == "sube":
                enf = np.maximum(enf, 0.0)
            base = nc.serie(corrido, u, v, [tramo])[tramo]
            antes = base
            base = aplicar_cambio(base, enf, None, p)
            t_base = tramo
            info["crecimiento"] = not np.array_equal(base, antes)
    return {"campo": corrido, "base": base, "tramo": t_base, "ua": u, "va": v, "ub": u, "vb": v}


def campos(prep: dict, tiempos, modo: str = "escenarios", k=None) -> dict:
    """Campos a `tiempos` minutos desde la imagen de radar, con la preparación de `preparar`.
    modo: "escenarios" (9, como siempre), "serie" (uno solo) o "envolvente" (25 caminos, k)."""
    tiempos = sorted(float(t) for t in tiempos)
    tramo = prep["tramo"]
    antes = [t for t in tiempos if t < tramo - 1e-6]
    despues = [t for t in tiempos if t >= tramo - 1e-6]
    resto = [t - tramo for t in despues]
    salida: dict = {}
    if modo == "envolvente":
        pilas = nc.campos_envolvente(prep["base"], prep["ub"], prep["vb"], resto, k)
        salida = {kk: {t: c[r] for t, r in zip(despues, resto)} for kk, c in pilas.items()}
        if antes:   # antes del barrido nuevo: tramo 1 (pocos casos, sin corrimiento extra)
            pre = nc.campos_envolvente(prep["campo"], prep["ua"], prep["va"], antes, k)
            for kk in salida:
                salida[kk].update(pre[kk])
        return salida
    if modo == "escenarios":
        series = [nc.serie(prep["base"], *nc.rotar(prep["ub"], prep["vb"], d, f), resto) for f, d in nc.ESCENARIOS]
        salida = {t: nc.campo_escenarios(np.stack([s[r] for s in series]), t) for t, r in zip(despues, resto)}
        if antes:
            salida.update(nc.campos_escenarios(prep["campo"], prep["ua"], prep["va"], antes))
    else:
        s = nc.serie(prep["base"], prep["ub"], prep["vb"], resto)
        salida = {t: s[r] for t, r in zip(despues, resto)}
        if antes:
            salida.update(nc.serie(prep["campo"], prep["ua"], prep["va"], antes))
    return salida


def pronostico(campo: np.ndarray, u: np.ndarray, v: np.ndarray, tiempos, sat_radar: np.ndarray | None,
               sat_nuevo: np.ndarray | None, sat_nuevo_previo: np.ndarray | None,
               tramo_sat_min: float | None, entre_barridos_min: float | None, km_px: float,
               parametros: dict | None = None, rayos: np.ndarray | None = None,
               escenarios: bool = True, info: dict | None = None) -> dict[float, np.ndarray]:
    """Atajo: preparar + campos."""
    prep = preparar(campo, u, v, sat_radar, sat_nuevo, sat_nuevo_previo, tramo_sat_min, entre_barridos_min,
                    km_px, parametros, rayos, info)
    return campos(prep, tiempos, "escenarios" if escenarios else "serie")


def mover_track(x: float, y: float, prep: dict, minutos: float) -> tuple[float, float]:
    """Posición de una tormenta `minutos` después de la imagen: tramo 1 y después tramo 2."""
    t1 = min(minutos, prep["tramo"])
    if t1 > 0:
        x, y = mover_punto(x, y, prep["ua"], prep["va"], t1)
    if minutos > t1:
        x, y = mover_punto(x, y, prep["ub"], prep["vb"], minutos - t1)
    return x, y


def mover_punto(x: float, y: float, u: np.ndarray, v: np.ndarray, minutos: float,
                paso: float = 2.5) -> tuple[float, float]:
    """Sigue el campo de movimiento desde (x, y) (punto medio, pasos de 2,5 min)."""
    h, w = u.shape

    def vel(xx, yy):
        xi = int(min(max(round(xx), 0), w - 1))
        yi = int(min(max(round(yy), 0), h - 1))
        return float(u[yi, xi]), float(v[yi, xi])

    t = 0.0
    while t < minutos - 1e-9:
        dt = min(paso, minutos - t)
        a, b = vel(x, y)
        a, b = vel(x + a * dt / 2, y + b * dt / 2)
        x, y, t = x + a * dt, y + b * dt, t + dt
    return x, y


def km_entre(a: tuple[float, float], b: tuple[float, float], km_px: float) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1]) * km_px
