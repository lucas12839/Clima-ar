#!/usr/bin/env python3
"""Decodifica las imágenes públicas del SMN "Bahía Blanca-SINARAME ZH MAX" a dBZ.

Las imágenes del sitio del SMN (805x802 px, con título, escala de colores,
círculo de 240 km y mapa) no son el PNG de paleta que usa ClimaAR, pero son
el mismo radar RMA10 y el mismo producto ZH MAX. Este módulo:

1. Lee la hora del título (OCR con tesseract; hora local + 3 h = UTC).
2. Construye la tabla color -> dBZ a partir de la barra de colores.
3. Convierte cada píxel del mapa a dBZ (descarta etiquetas, rutas, círculo
   y polígonos de aviso, que no son colores de la escala).
4. Remuestrea a la grilla de ClimaAR (512x512, 240 km de radio, radar al
   centro) para usar el mismo nowcast y la misma geometría.

Se usa para validar eventos históricos (herramientas/validar_eventos.py).
"""
from __future__ import annotations

import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

# Geometría medida en las imágenes (círculo de 240 km ajustado por elipse,
# idéntico en los tres eventos 2023-2025).
CENTRO_X = 328.5
CENTRO_Y = 417.8
RADIO_PX = 310.0
MAPA = (8, 92, 657, 737)  # x0, y0, x1, y1

# Barra de colores: columna x=729, etiquetas 75 dBZ en y=101 y -15 dBZ en y=710.
BARRA_X = 729
BARRA_Y0, BARRA_Y1 = 94, 736
DBZ_Y75, DBZ_YM15 = 101.0, 710.0

DISTANCIA_COLOR_MAX = 22.0
GRILLA = 512


def dbz_de_fila(y: float) -> float:
    return 75.0 - (y - DBZ_Y75) * (90.0 / (DBZ_YM15 - DBZ_Y75))


def tabla_colores(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    filas = np.arange(BARRA_Y0, BARRA_Y1 + 1)
    colores = img_rgb[filas, BARRA_X].astype(np.float32)
    dbz = np.array([dbz_de_fila(y) for y in filas], dtype=np.float32)
    # Fuera de la escala útil: blanco puro (59-61 dBZ, igual a etiquetas y
    # rutas) y el extremo inferior (< 0 dBZ, ruido).
    blanco = colores.min(axis=1) >= 245
    util = ~blanco & (dbz >= 0.0)
    return colores[util], dbz[util]


def leer_hora_utc(path: Path) -> datetime | None:
    with Image.open(path) as img:
        titulo = img.convert("L").crop((0, 0, 805, 26)).resize((805 * 3, 78))
        tmp = Path("/tmp/_titulo_smn.png")
        titulo.save(tmp)
    texto = subprocess.run(
        ["tesseract", str(tmp), "-", "--psm", "7"], capture_output=True, text=True
    ).stdout
    texto = texto.replace("Q", "0").replace("O", "0").replace(": ", ":")
    m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2})H0A", texto)
    if not m:
        return None
    local = datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)), int(m.group(4)), int(m.group(5)))
    return (local + timedelta(hours=3)).replace(tzinfo=timezone.utc)


def decodificar(path: Path) -> np.ndarray:
    """dBZ en la imagen original (NaN = sin eco o no es color de la escala)."""
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    colores, valores = tabla_colores(rgb)
    x0, y0, x1, y1 = MAPA
    mapa = rgb[y0:y1, x0:x1]
    h, w, _ = mapa.shape
    plano = mapa.reshape(-1, 3)
    salida = np.full(plano.shape[0], np.nan, dtype=np.float32)
    # Solo píxeles con color (descarta fondo gris oscuro y grises del mapa).
    sat = plano.max(axis=1) - plano.min(axis=1)
    candidatos = np.where(sat > 25)[0]
    for ini in range(0, len(candidatos), 20000):
        idx = candidatos[ini:ini + 20000]
        d = np.linalg.norm(plano[idx, None, :] - colores[None, :, :], axis=2)
        j = d.argmin(axis=1)
        ok = d[np.arange(len(idx)), j] <= DISTANCIA_COLOR_MAX
        salida[idx[ok]] = valores[j[ok]]
    campo = salida.reshape(h, w)
    completo = np.full(rgb.shape[:2], np.nan, dtype=np.float32)
    completo[y0:y1, x0:x1] = campo
    return completo


def a_grilla_climaar(dbz_smn: np.ndarray) -> np.ndarray:
    """Remuestrea a 512x512 con el radar al centro y 240 km de radio."""
    c = (GRILLA - 1) / 2.0
    gy, gx = np.mgrid[0:GRILLA, 0:GRILLA].astype(np.float32)
    escala = RADIO_PX / c
    sx = np.rint(CENTRO_X + (gx - c) * escala).astype(int)
    sy = np.rint(CENTRO_Y + (gy - c) * escala).astype(int)
    h, w = dbz_smn.shape
    dentro = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h) & (np.hypot(gx - c, gy - c) <= c)
    out = np.full((GRILLA, GRILLA), np.nan, dtype=np.float32)
    out[dentro] = dbz_smn[sy[dentro], sx[dentro]]
    return out


def limpiar_lineas(dbz: np.ndarray) -> np.ndarray:
    """Quita restos finos (bordes de polígonos, texto) con una apertura 3x3."""
    import cv2

    eco = np.isfinite(dbz).astype(np.uint8)
    abierto = cv2.morphologyEx(eco, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    out = dbz.copy()
    out[(eco == 1) & (abierto == 0)] = np.nan
    return out


def cargar_evento(carpeta: Path) -> list[tuple[datetime, np.ndarray]]:
    frames = []
    for path in sorted(carpeta.rglob("*.png")):
        hora = leer_hora_utc(path)
        if hora is None:
            continue
        grilla = limpiar_lineas(a_grilla_climaar(decodificar(path)))
        frames.append((hora, grilla))
    frames.sort(key=lambda x: x[0])
    return frames
