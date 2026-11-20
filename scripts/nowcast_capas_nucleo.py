#!/usr/bin/env python3
"""Núcleo del nowcast en capas (punto 5b): funciones puras, sin lectura de archivos.

No es otro nowcast: usa el mismo flujo óptico y la misma advección de
`nowcast_flujo_optico.py` y le agrega lo que le faltaba para no quedar a ciegas
cuando el radar se atrasa o falta una imagen:

- Capa 1, movimiento: la última imagen se mueve hasta la HORA ACTUAL (el "ahora"
  ya no es la imagen vieja tal como llegó) y los plazos se cuentan desde ahora.
- Capa 2, crecimiento y debilitamiento (parte del radar): tendencia de la
  reflectividad siguiendo a cada eco (diferencia entre la imagen actual y la
  anterior movida), que se aplica amortiguada porque una tendencia no dura.
- Capa 5, incertidumbre: escenarios con la velocidad y la dirección corridas;
  el porcentaje de llegada es la fracción de escenarios en que llega.

La validación (capa 6) decide qué se usa: `herramientas/simular_huecos.py`.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

import nowcast_flujo_optico as nf

# Capa 2: tendencia de la reflectividad.
TENDENCIA_SIGMA_PX = 5.0          # ~5 km: escala de una celda, no del píxel
TENDENCIA_TAU_MIN = 20.0          # la tendencia se apaga con esta escala de tiempo
TENDENCIA_MAX_DBZ = 8.0           # nunca cambia más de esto en total
# Capa 5: escenarios (velocidad x factor, dirección + grados).
ESCENARIOS = [(f, d) for f in (0.8, 1.0, 1.2) for d in (-20.0, 0.0, 20.0)]
# Confianza que baja de a poco con la edad del radar (sin cortar a los 40 min).
EDAD_MEDIA_VIDA_MIN = 30.0


CLUTTER_TRANQUILO_FRACCION = 0.05
CLUTTER_TRANQUILO_MAX_PIXELES_ECO = 300


def mascara_clutter_tranquilo(campos_limpios) -> np.ndarray | None:
    """Ecos fijos intermitentes: píxeles con >= 20 dBZ en más del 5 % de las
    imágenes TRANQUILAS (menos de 300 píxeles con eco en todo el radar).
    Con lluvia real no se cuenta, así que una tormenta no queda enmascarada.
    Cerca del radar (Bahía Blanca, Punta Alta, Médanos) estos píxeles daban
    "lluvia" en el 37 % de las imágenes sin lluvia."""
    acc, n = None, 0
    for c in campos_limpios:
        eco = c >= nf.UMBRAL_ECO_DBZ
        if int(eco.sum()) >= CLUTTER_TRANQUILO_MAX_PIXELES_ECO:
            continue
        acc = eco.astype(np.int32) if acc is None else acc + eco
        n += 1
    if acc is None or n < 50:
        return None
    return (acc / n) > CLUTTER_TRANQUILO_FRACCION


CLUTTER_INTERMITENTE = nf.ROOT / "config" / "clutter_intermitente_rma10.png"


MANCHA_FUERA_MIN_PX = 8         # una mancha real tiene al menos 8 píxeles (~7 km²) fuera del clutter
MANCHA_FUERA_FRACCION = 0.5     # y la mitad o más de su superficie fuera del clutter
_INTERMITENTE: dict = {}


def cargar_clutter_total(shape: tuple[int, int]) -> np.ndarray:
    """Clutter FIJO (config/clutter_rma10.png). El intermitente se aplica aparte y con
    cuidado en `campo_limpio` (solo saca ecos sueltos, nunca una tormenta que pasa)."""
    return nf.cargar_mascara_clutter(shape)


def cargar_intermitente(shape: tuple[int, int]) -> np.ndarray | None:
    if shape not in _INTERMITENTE:
        extra = None
        if CLUTTER_INTERMITENTE.exists():
            from PIL import Image

            with Image.open(CLUTTER_INTERMITENTE) as img:
                extra = np.asarray(img.convert("L")) > 127
            if extra.shape != shape:
                extra = None
        _INTERMITENTE[shape] = extra
    return _INTERMITENTE[shape]


def campo_limpio(dbz: np.ndarray, clutter: np.ndarray) -> np.ndarray:
    """Como nf.campo_limpio, más los ecos fijos intermitentes, pero solo si están SUELTOS.

    Un punto del clutter intermitente se borra solo si su mancha de eco está casi toda
    sobre puntos de clutter (eco del terreno). Si una lluvia o una tormenta pasa por
    encima, la mancha sigue fuera del clutter y se deja entera: no pierde superficie."""
    import cv2

    campo = nf.campo_limpio(dbz, clutter)
    inter = cargar_intermitente(campo.shape)
    if inter is None or not inter.any():
        return campo
    # Cada mancha de eco (conectada) que toca el clutter intermitente: si la mayor parte
    # está FUERA de los puntos de clutter, es lluvia o tormenta real y se deja entera.
    eco = (campo >= nf.UMBRAL_ECO_DBZ).astype(np.uint8)
    toca = inter & (eco > 0)
    if not toca.any():
        return campo
    dil = cv2.dilate(eco, np.ones((3, 3), np.uint8))  # une ecos separados por 1 píxel
    n, etiquetas = cv2.connectedComponents(dil, connectivity=8)
    total = np.bincount(etiquetas[eco > 0], minlength=n)
    en_clutter = np.bincount(etiquetas[toca], minlength=n)
    fuera = total - en_clutter
    real = (fuera >= MANCHA_FUERA_MIN_PX) & (fuera >= MANCHA_FUERA_FRACCION * total)
    real[0] = False
    borrar = toca & ~real[etiquetas]
    campo[borrar] = 0.0
    return campo


def confianza_por_edad(edad_min: float | None) -> float:
    """1.0 hasta 10 min; después se reduce a la mitad cada 30 min (nunca 0)."""
    if edad_min is None or edad_min <= 10.0:
        return 1.0
    return round(0.5 ** ((edad_min - 10.0) / EDAD_MEDIA_VIDA_MIN), 3)


def rotar(u: np.ndarray, v: np.ndarray, grados: float, factor: float = 1.0):
    """Gira el campo de movimiento (grados, sentido horario en la imagen) y lo escala."""
    a = math.radians(grados)
    c, s = math.cos(a), math.sin(a)
    return ((u * c - v * s) * factor).astype(np.float32), ((u * s + v * c) * factor).astype(np.float32)


def tendencia(previo: np.ndarray, actual: np.ndarray, u: np.ndarray, v: np.ndarray,
              gap_min: float) -> np.ndarray:
    """dBZ por minuto siguiendo al eco (lagrangiano), suavizado a escala de celda.

    Fuera de los ecos vale 0 (no se inventa lluvia donde no hay)."""
    import cv2

    movido = nf.advectar(previo, u, v, float(gap_min))
    eco = (actual >= nf.UMBRAL_ECO_DBZ) | (movido >= nf.UMBRAL_ECO_DBZ)
    dif = np.where(eco, actual - movido, 0.0).astype(np.float32)
    peso = cv2.GaussianBlur(eco.astype(np.float32), (0, 0), TENDENCIA_SIGMA_PX)
    suave = cv2.GaussianBlur(dif, (0, 0), TENDENCIA_SIGMA_PX)
    with np.errstate(invalid="ignore", divide="ignore"):
        tasa = np.where(peso > 0.05, suave / np.maximum(peso, 1e-3), 0.0) / max(gap_min, 1.0)
    return np.where(actual >= nf.UMBRAL_ECO_DBZ, tasa, 0.0).astype(np.float32)


def cambio_total(tasa: np.ndarray, minutos: float, tau: float = TENDENCIA_TAU_MIN) -> np.ndarray:
    """Cambio acumulado con la tendencia amortiguada: tasa·tau·(1 - e^(-t/tau))."""
    return np.clip(tasa * tau * (1.0 - math.exp(-minutos / tau)), -TENDENCIA_MAX_DBZ, TENDENCIA_MAX_DBZ)


def pronostico(actual: np.ndarray, u: np.ndarray, v: np.ndarray, minutos: float,
               tasa: np.ndarray | None = None) -> np.ndarray:
    """Campo a `minutos` desde la imagen: con tendencia (capa 2) si se pasa `tasa`."""
    campo = actual
    if tasa is not None and minutos > 0:
        campo = actual + cambio_total(tasa, minutos)
        campo = np.where(actual >= nf.UMBRAL_ECO_DBZ, campo, actual).astype(np.float32)
        campo[campo < nf.UMBRAL_ECO_DBZ - 5.0] = 0.0
    if minutos <= 0:
        return campo.astype(np.float32)
    return nf.advectar(campo.astype(np.float32), u, v, float(minutos))


def serie(actual: np.ndarray, u: np.ndarray, v: np.ndarray, tiempos, tasa: np.ndarray | None = None,
          paso: float = 2.5) -> dict[float, np.ndarray]:
    """Campos a varios minutos desde la imagen, en una sola pasada (más rápido
    que advectar cada plazo desde cero; mismo esquema semi-lagrangiano)."""
    import cv2

    h, w = actual.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x, y, t_act = gx.copy(), gy.copy(), 0.0
    salida: dict[float, np.ndarray] = {}

    def fuente(t):
        if tasa is None or t <= 0:
            return actual
        c = actual + cambio_total(tasa, t)
        c = np.where(actual >= nf.UMBRAL_ECO_DBZ, c, actual).astype(np.float32)
        c[c < nf.UMBRAL_ECO_DBZ - 5.0] = 0.0
        return c

    for t in sorted(float(t) for t in tiempos):
        while t_act + paso <= t + 1e-6:
            uu = cv2.remap(u, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            vv = cv2.remap(v, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            x, y, t_act = x - uu * paso, y - vv * paso, t_act + paso
        resto = t - t_act
        xs, ys = x, y
        if resto > 1e-6:
            uu = cv2.remap(u, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            vv = cv2.remap(v, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            xs, ys = x - uu * resto, y - vv * resto
        salida[t] = cv2.remap(fuente(t).astype(np.float32), xs, ys, cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return salida


ESCENARIOS_MINIMOS = ((40.0, 3), (1e9, 2))  # hasta 40 min desde la imagen: 3 de 9; después: 2 de 9


def campo_escenarios(pila: np.ndarray, minutos_desde_imagen: float) -> np.ndarray:
    """Campo de la capa 5: en cada píxel, el valor que dan al menos N escenarios.
    Validado con herramientas/simular_huecos.py (26 días): mejor que la imagen
    vieja y que el pronóstico de un solo escenario en todos los huecos."""
    n = next(k for limite, k in ESCENARIOS_MINIMOS if minutos_desde_imagen <= limite)
    n = min(n, pila.shape[0])
    return np.sort(pila, axis=0)[-n]


def campos_escenarios(actual: np.ndarray, u: np.ndarray, v: np.ndarray, tiempos,
                      tasa: np.ndarray | None = None) -> dict[float, np.ndarray]:
    """Campos de la capa 5 a varios minutos desde la imagen."""
    series = [serie(actual, *rotar(u, v, d, f), tiempos, tasa) for f, d in ESCENARIOS]
    return {float(t): campo_escenarios(np.stack([s[float(t)] for s in series]), float(t)) for t in tiempos}


# Punto 5b, "¿llega a mi localidad en la próxima hora?": ENVOLVENTE de caminos posibles.
# herramientas/analizar_omisiones.py (26 días, 9.597 casos): 2 de cada 3 tormentas que se
# escapaban ya estaban en el radar a menos de 30 km, del lado de donde venía el movimiento;
# no llegaban en el campo "3 de 9" porque se desviaron, aceleraron o se estiraron. Con 25
# caminos (velocidad 0,6-1,4 y dirección ±40°) y "llega si lo dan k de 25" se decide el
# grado "posible"; el grado "probable" sigue siendo el campo de hoy. Validado con
# herramientas/simular_envolvente.py (k en config/nowcast_capas.json).
ESCENARIOS_AMPLIOS = [(f, g) for f in (0.6, 0.8, 1.0, 1.2, 1.4) for g in (-40.0, -20.0, 0.0, 20.0, 40.0)]


def campos_envolvente(actual: np.ndarray, u: np.ndarray, v: np.ndarray, tiempos, k,
                      tasa: np.ndarray | None = None, escenarios=ESCENARIOS_AMPLIOS):
    """En cada píxel, el valor que dan al menos `k` de los caminos (k=1: el máximo).
    Con `k` lista/tupla devuelve {k: campos} calculando los caminos una sola vez."""
    tiempos = [float(t) for t in tiempos]
    series = [serie(actual, *rotar(u, v, g, f), tiempos, tasa) for f, g in escenarios]
    pilas = {t: np.sort(np.stack([s[t] for s in series]), axis=0) for t in tiempos}
    ks = list(k) if isinstance(k, (list, tuple)) else [k]
    salida = {kk: {t: pilas[t][-max(1, min(int(kk), len(series)))] for t in tiempos} for kk in ks}
    return salida if isinstance(k, (list, tuple)) else salida[ks[0]]


# Punto 6: crecimiento y debilitamiento con ANVIL (pysteps; física y estadística,
# sin entrenamiento). Validado con herramientas/comparar_pysteps.py (2 muestras de
# ~270 casos): combinado con los escenarios ("hay eco si lo dan los escenarios o
# ANVIL") mejora desde 30 min; a 10 y 20 min no aporta y no se usa. ANVIL va con su
# propio movimiento (Lucas-Kanade de pysteps): con el nuestro rindió menos.
ANVIL_DESDE_MIN = 20.0


def a_lluvia(dbz: np.ndarray) -> np.ndarray:
    """dBZ -> mm/h (Marshall-Palmer), 0 debajo del umbral de eco."""
    d = np.nan_to_num(dbz, nan=0.0)
    r = (10 ** (d / 10.0) / 200.0) ** (1 / 1.6)
    r[d < nf.UMBRAL_ECO_DBZ - 5.0] = 0.0
    return r


def a_dbz(r: np.ndarray) -> np.ndarray:
    r = np.nan_to_num(r, nan=0.0)
    z = 10 * np.log10(np.maximum(200.0 * np.maximum(r, 1e-6) ** 1.6, 1e-6))
    z[r < 0.05] = 0.0
    return z.astype(np.float32)


def anvil(campos_3: list[np.ndarray], tiempos) -> dict[float, np.ndarray]:
    """ANVIL de pysteps. `campos_3` = [hace 20 min, hace 10 min, actual] en dBZ limpios.
    Devuelve dBZ a los minutos pedidos (desde la imagen)."""
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):  # pysteps escribe mucho en pantalla
        from pysteps import motion, nowcasts

    validos = sorted(float(t) for t in tiempos if float(t) > ANVIL_DESDE_MIN)
    if not validos:
        return {}
    with contextlib.redirect_stdout(io.StringIO()):
        vel = motion.get_method("LK")(np.stack([np.nan_to_num(c) for c in campos_3]).astype(float))
        r3 = np.stack([a_lluvia(c) for c in campos_3])
        salida = nowcasts.get_method("anvil")(r3, vel, [t / 10.0 for t in validos],
                                              ar_window_radius=25, ar_order=1)
    return {t: a_dbz(c) for t, c in zip(validos, salida)}


def combinar_anvil(escenarios: dict[float, np.ndarray], anv: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    """Hay eco si lo dan los escenarios o ANVIL (desde ANVIL_DESDE_MIN de la imagen)."""
    salida = {}
    for t, c in escenarios.items():
        a = anv.get(float(t))
        salida[t] = np.maximum(c, a) if (a is not None and float(t) > ANVIL_DESDE_MIN) else c
    return salida


def picos_por_zona(campo: np.ndarray, zonas: dict[str, np.ndarray], min_pixeles: int = 3) -> dict[str, float]:
    """dBZ que cubre al menos `min_pixeles` píxeles de cada zona."""
    salida = {}
    for k, zona in zonas.items():
        vals = np.nan_to_num(campo[zona], nan=0.0)
        salida[k] = 0.0 if vals.size < min_pixeles else float(np.partition(vals, -min_pixeles)[-min_pixeles])
    return salida


def escenarios_llegada(actual: np.ndarray, u: np.ndarray, v: np.ndarray, zonas: dict[str, np.ndarray],
                       desde_min: float, hasta_min: float, umbral: float,
                       tasa: np.ndarray | None = None, paso: float = 10.0,
                       escenarios=ESCENARIOS) -> dict[str, dict[str, Any]]:
    """Porcentaje de escenarios en que llega un eco >= umbral a cada zona entre
    `desde_min` y `hasta_min` (minutos contados desde la imagen) y la primera
    llegada típica (mediana de los escenarios que llegan), contada desde la imagen."""
    llegadas: dict[str, list[float | None]] = {k: [] for k in zonas}
    tiempos = list(np.arange(max(desde_min, 0.0), hasta_min + 0.01, paso))
    for factor, grados in escenarios:
        uu, vv = rotar(u, v, grados, factor)
        primera = {k: None for k in zonas}
        campos = serie(actual, uu, vv, tiempos, tasa)
        for t in tiempos:
            picos = picos_por_zona(campos[float(t)], zonas)
            for k, m in picos.items():
                if primera[k] is None and m >= umbral:
                    primera[k] = float(t)
        for k in zonas:
            llegadas[k].append(primera[k])
    salida = {}
    for k, lista in llegadas.items():
        ok = [t for t in lista if t is not None]
        salida[k] = {
            "porcentaje": round(100.0 * len(ok) / len(lista), 0) if lista else 0.0,
            "minutos_desde_imagen": float(np.median(ok)) if ok else None,
            "primera_min": min(ok) if ok else None,
            "ultima_min": max(ok) if ok else None,
        }
    return salida
