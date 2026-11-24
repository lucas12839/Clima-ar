"""Satélite para mirar: nubes de GOES-19 (canal 13) sobre el mapa del radar.

Solo es para la vista Meteorólogos. La estructura sigue leyendo el satélite
como siempre; esto no guarda nada ni cambia ningún dato.

El canal 13 mide la temperatura del tope de la nube: cuanto más fría, más
alta y más desarrollada la nube. Se pasa a la grilla del radar (512 x 512,
240 km, con el paralaje corregido, igual que el control de falsos ecos) y se
pinta con la escala infrarroja realzada de uso habitual.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parent
ESCALA = 2  # 512 -> 1024 px, igual que el radar limpio

# °C, (R, G, B, opacidad). Lo más tibio (suelo o nube baja) casi no se ve.
COLORES = [
    (20.0, (0, 0, 0, 0)),
    (8.0, (120, 120, 120, 40)),
    (0.0, (160, 160, 160, 110)),
    (-15.0, (205, 205, 205, 170)),
    (-30.0, (240, 240, 240, 200)),
    (-35.0, (90, 200, 240, 215)),
    (-45.0, (30, 90, 230, 225)),
    (-55.0, (40, 190, 70, 230)),
    (-62.0, (250, 230, 40, 235)),
    (-70.0, (240, 60, 30, 240)),
    (-78.0, (200, 30, 200, 245)),
    (-90.0, (255, 255, 255, 250)),
]


def _grilla(lat, lon, temp_c) -> np.ndarray:
    scripts = str(RAIZ / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import corrector_tiempo_real as ct

    return ct.grilla_satelite(lat, lon, temp_c, (512, 512))


def _colorear(temp: np.ndarray) -> np.ndarray:
    xs = np.array([c[0] for c in COLORES][::-1], dtype=np.float32)
    rgba = np.zeros(temp.shape + (4,), dtype=np.float32)
    t = np.where(np.isfinite(temp), temp, 99.0)
    for k in range(4):
        ys = np.array([c[1][k] for c in COLORES][::-1], dtype=np.float32)
        rgba[..., k] = np.interp(t, xs, ys)
    rgba[~np.isfinite(temp)] = 0.0
    return np.clip(rgba, 0, 255).astype(np.uint8)


def imagen_nubes(lat, lon, temp_c) -> bytes:
    import cv2

    grilla = _grilla(np.asarray(lat), np.asarray(lon), np.asarray(temp_c, np.float32))
    valida = np.isfinite(grilla)
    base = np.where(valida, grilla, 30.0).astype(np.float32)
    grande = cv2.resize(base, (512 * ESCALA, 512 * ESCALA), interpolation=cv2.INTER_LINEAR)
    grande = cv2.GaussianBlur(grande, (0, 0), 1.0)
    mascara = cv2.resize(valida.astype(np.uint8), grande.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    grande[~mascara] = np.nan
    buf = io.BytesIO()
    # 64 colores alcanzan para la escala y la imagen pesa varias veces menos.
    img = Image.fromarray(_colorear(grande), "RGBA").quantize(64, method=Image.Quantize.FASTOCTREE)
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
