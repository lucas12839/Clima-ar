"""Radar para la gente: imagen limpia y suave del RMA10 para el mapa.

Solo es para mirar. La estructura (detector, seguimiento, nowcast) sigue
usando el PNG crudo de SINARAME sin ningún cambio.

Pasos:
  1. Índice de la paleta RMA10 -> dBZ (misma fórmula que
     scripts/decodificador_rma10_dbz.py: índices 64..207 = -15..75 dBZ).
  2. Se sacan los falsos ecos fijos (config/clutter_rma10.png), los ecos
     por debajo de la lluvia débil y los píxeles sueltos.
  3. Se agranda al doble con suavizado leve: los núcleos fuertes no se
     achican ni se corren (el suavizado se aplica al borde, no al máximo).
  4. Se pinta con una escala de colores clásica y transparencia.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

INDICE_MIN = 64
INDICE_MAX = 207
DBZ_MIN = -15.0
DBZ_MAX = 75.0

# Anclas de la paleta RMA10 validada (decodificador_rma10_dbz.py).
ANCLAS = {
    64: (57, 113, 163),
    104: (77, 225, 51),
    167: (203, 0, 27),
    207: (255, 255, 255),
}

UMBRAL_VISIBLE_DBZ = 10.0   # desde lluvia débil / llovizna
ESCALA = 2                  # 512 -> 1024 px

# dBZ, (R, G, B, opacidad 0-255)
COLORES = [
    (10.0, (130, 210, 255, 110)),
    (20.0, (40, 140, 255, 165)),
    (27.0, (25, 190, 110, 195)),
    (35.0, (40, 200, 40, 210)),
    (40.0, (255, 230, 40, 220)),
    (45.0, (255, 160, 20, 228)),
    (50.0, (245, 50, 30, 236)),
    (57.0, (190, 20, 60, 240)),
    (62.0, (210, 40, 220, 245)),
    (68.0, (255, 255, 255, 250)),
]


def _paleta_valida(img: Image.Image) -> bool:
    pal = img.getpalette()
    if img.mode != "P" or not pal:
        return False
    for i, rgb in ANCLAS.items():
        if tuple(pal[i * 3:i * 3 + 3]) != rgb:
            return False
    return True


def decodificar(img: Image.Image) -> np.ndarray | None:
    if not _paleta_valida(img):
        return None
    idx = np.asarray(img, dtype=np.int16)
    tr = img.info.get("transparency")
    dbz = DBZ_MIN + (np.clip(idx, INDICE_MIN, INDICE_MAX) - INDICE_MIN) / (
        INDICE_MAX - INDICE_MIN
    ) * (DBZ_MAX - DBZ_MIN)
    sin_eco = idx < INDICE_MIN
    if isinstance(tr, int):
        sin_eco |= idx == tr
    dbz = dbz.astype(np.float32)
    dbz[sin_eco] = np.nan
    return dbz


def _mascara_clutter(path: Path | None, shape) -> np.ndarray:
    if path is None or not path.is_file():
        return np.zeros(shape, dtype=bool)
    with Image.open(path) as m:
        mask = np.asarray(m.convert("L")) > 127
    return mask if mask.shape == shape else np.zeros(shape, dtype=bool)


def _vecinos(mask: np.ndarray) -> np.ndarray:
    p = np.pad(mask.astype(np.int8), 1)
    h, w = mask.shape
    total = np.zeros((h, w), dtype=np.int8)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            total += p[dy:dy + h, dx:dx + w]
    return total


def limpiar(dbz: np.ndarray, clutter: np.ndarray) -> np.ndarray:
    campo = np.nan_to_num(dbz, nan=0.0)
    campo[clutter] = 0.0
    campo[campo < UMBRAL_VISIBLE_DBZ] = 0.0
    eco = campo > 0
    # Píxel suelto (sin ningún vecino con eco): ruido, no lluvia.
    campo[eco & (_vecinos(eco) == 0)] = 0.0
    return campo


def _colorear(campo: np.ndarray) -> np.ndarray:
    xs = np.array([c[0] for c in COLORES], dtype=np.float32)
    rgba = np.zeros(campo.shape + (4,), dtype=np.float32)
    for k in range(4):
        ys = np.array([c[1][k] for c in COLORES], dtype=np.float32)
        rgba[..., k] = np.interp(campo, xs, ys)
    rgba[campo < UMBRAL_VISIBLE_DBZ - 2.0] = 0.0
    # Borde suave entre 8 y 10 dBZ en vez de un corte en escalera.
    borde = (campo >= UMBRAL_VISIBLE_DBZ - 2.0) & (campo < UMBRAL_VISIBLE_DBZ)
    rgba[..., 3][borde] *= (campo[borde] - (UMBRAL_VISIBLE_DBZ - 2.0)) / 2.0
    return np.clip(rgba, 0, 255).astype(np.uint8)


def suavizar(campo: np.ndarray) -> np.ndarray:
    """Agranda y suaviza promediando solo entre píxeles con eco.

    El radar crudo viene salpicado (huecos sin dato entre píxeles con
    lluvia). Un promedio común mezclaría la lluvia con esos huecos y la
    haría desaparecer; acá cada punto toma el promedio de los ecos que
    tiene alrededor, y se muestra solo donde hay suficiente eco cerca.
    """
    import cv2

    h, w = campo.shape
    eco = (campo > 0).astype(np.float32)
    sigma = 1.3
    suma = cv2.GaussianBlur(campo, (0, 0), sigma)
    peso = cv2.GaussianBlur(eco, (0, 0), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        prom = np.where(peso > 1e-3, suma / peso, 0.0).astype(np.float32)
    # Cobertura: cuánto eco hay alrededor (0 a 1). Con poca cobertura
    # el borde se desvanece en vez de cortarse en escalera.
    grande = cv2.resize(prom, (w * ESCALA, h * ESCALA), interpolation=cv2.INTER_LINEAR)
    cob = cv2.resize(peso, (w * ESCALA, h * ESCALA), interpolation=cv2.INTER_LINEAR)
    # Los núcleos fuertes conservan su valor máximo real.
    crudo = cv2.resize(campo, (w * ESCALA, h * ESCALA), interpolation=cv2.INTER_NEAREST)
    fuerte = crudo >= 35.0
    grande[fuerte] = np.maximum(grande[fuerte], crudo[fuerte])
    grande[cob < 0.22] = 0.0
    return grande


def imagen_mapa(png: Path, clutter_png: Path | None = None) -> bytes | None:
    """PNG RGBA listo para el mapa, o None si la paleta no es la RMA10."""
    with Image.open(png) as img:
        img.load()
        dbz = decodificar(img)
    if dbz is None:
        return None
    campo = limpiar(dbz, _mascara_clutter(clutter_png, dbz.shape))
    rgba = _colorear(suavizar(campo))
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()
