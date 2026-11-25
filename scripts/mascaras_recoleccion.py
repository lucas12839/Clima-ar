#!/usr/bin/env python3
"""Máscara de ecos fijos de cada radar del recolector (Córdoba, Resistencia, Tostado, Formosa, Salta).

Un eco que aparece en el mismo lugar en la mayoría de las imágenes (sierras,
edificios, antenas) no es lluvia: es un eco fijo. Con las imágenes que juntó
scripts/recoleccion_radares.py se arma, para cada radar, una máscara con esos
píxeles, igual que la de Bahía (scripts/generar_mascara_estatica.py).

- Usa las últimas imágenes guardadas (hasta 2 días). Hace falta al menos 1 día.
- Se rehace una vez por día (si la máscara tiene menos de 24 h, no hace nada).
- Guarda en la carpeta de cada radar: mascara_ecos_fijos.png (blanco = eco fijo)
  y mascara_ecos_fijos.json (cuántas imágenes, cuántos píxeles, fecha).

Lo llama el recolector una vez por día con el Python del proyecto (necesita
numpy y Pillow). Uso a mano:
  .venv-ia/bin/python scripts/mascaras_recoleccion.py --destino /var/lib/climaar-recoleccion
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PERSISTENCIA_MINIMA = 0.65
MIN_IMAGENES = 144          # 1 día cada 10 min
MAX_IMAGENES = 288          # 2 días
RADIO_TOLERANCIA_PX = 2
REHACER_CADA_H = 24.0
NOMBRE = "mascara_ecos_fijos"


def imagenes_radar(carpeta: Path) -> list[Path]:
    archivos = [p for p in carpeta.glob("*/*.png") if p.stem.isdigit() and len(p.stem) == 14]
    archivos.sort(key=lambda p: p.stem)
    return archivos[-MAX_IMAGENES:]


def hace_falta(carpeta: Path, ahora: datetime) -> bool:
    info = carpeta / f"{NOMBRE}.json"
    try:
        previo = json.loads(info.read_text(encoding="utf-8"))
        hecha = datetime.fromisoformat(previo["generada_utc"].replace("Z", "+00:00"))
    except (OSError, ValueError, KeyError):
        return True
    return (ahora - hecha).total_seconds() / 3600.0 >= REHACER_CADA_H


def construir(archivos: list[Path]):
    """(máscara bool, imágenes usadas, tamaño) con el eco presente en >= 65 %."""
    import numpy as np
    from PIL import Image
    from scipy import ndimage

    yy, xx = np.ogrid[-RADIO_TOLERANCIA_PX:RADIO_TOLERANCIA_PX + 1, -RADIO_TOLERANCIA_PX:RADIO_TOLERANCIA_PX + 1]
    disco = (xx * xx + yy * yy) <= RADIO_TOLERANCIA_PX ** 2
    soporte = None
    usadas = 0
    tam = None
    for ruta in archivos:
        try:
            with Image.open(ruta) as img:
                rgba = np.asarray(img.convert("RGBA"))
        except Exception:
            continue  # imagen dañada: se saltea
        if tam is None:
            tam = rgba.shape[:2]
            soporte = np.zeros(tam, dtype=np.uint16)
        if rgba.shape[:2] != tam:
            continue
        eco = rgba[..., 3] > 0
        soporte += ndimage.binary_dilation(eco, structure=disco).astype(np.uint16)
        usadas += 1
    if usadas == 0:
        return None, 0, None
    return soporte >= int(np.ceil(usadas * PERSISTENCIA_MINIMA)), usadas, tam


def procesar(destino: Path, ahora: datetime | None = None, forzar: bool = False) -> dict:
    ahora = ahora or datetime.now(timezone.utc)
    resumen = {}
    for carpeta in sorted(p for p in destino.iterdir() if p.is_dir()):
        archivos = imagenes_radar(carpeta)
        if len(archivos) < MIN_IMAGENES:
            resumen[carpeta.name] = f"faltan imágenes ({len(archivos)} de {MIN_IMAGENES})"
            continue
        if not forzar and not hace_falta(carpeta, ahora):
            resumen[carpeta.name] = "al día"
            continue
        mascara, usadas, tam = construir(archivos)
        if mascara is None or usadas < MIN_IMAGENES:
            resumen[carpeta.name] = f"imágenes válidas insuficientes ({usadas})"
            continue
        from PIL import Image
        import numpy as np

        png = carpeta / f"{NOMBRE}.png"
        tmp = carpeta / f".{NOMBRE}.png.tmp"
        Image.fromarray(mascara.astype(np.uint8) * 255, mode="L").save(tmp, format="PNG")
        os.replace(tmp, png)
        info = {
            "generada_utc": ahora.isoformat().replace("+00:00", "Z"),
            "imagenes": usadas,
            "desde": archivos[0].stem,
            "hasta": archivos[-1].stem,
            "tamano": list(tam),
            "pixeles_eco_fijo": int(mascara.sum()),
            "fraccion_eco_fijo": round(float(mascara.mean()), 5),
            "persistencia_minima": PERSISTENCIA_MINIMA,
        }
        (carpeta / f"{NOMBRE}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        resumen[carpeta.name] = f"máscara nueva: {info['pixeles_eco_fijo']} píxeles fijos con {usadas} imágenes"
    return resumen


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--destino", default="/var/lib/climaar-recoleccion")
    p.add_argument("--forzar", action="store_true", help="rehacer aunque tenga menos de 24 h")
    a = p.parse_args(argv)
    destino = Path(a.destino)
    if not destino.is_dir():
        print(f"mascaras: no existe {destino}")
        return 0
    for radar, texto in procesar(destino, forzar=a.forzar).items():
        print(f"mascaras: {radar}: {texto}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
