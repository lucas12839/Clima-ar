#!/usr/bin/env python3
"""
Decoder cuantitativo independiente para RMA10 ZH_MAX.

IMPORTANTE:
- No modifica el procesador actual.
- No modifica resultado_procesamiento.json.
- No modifica el seguimiento ni el ML.
- En esta primera etapa SOLO analiza la paleta real del PNG.
- No asigna dBZ inventados.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


def analizar_png(ruta: Path) -> dict:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe: {ruta}")

    with Image.open(ruta) as img:
        imagen = img.convert("RGBA")
        ancho, alto = imagen.size

        pixeles = list(imagen.getdata())

    colores = Counter(pixeles)

    # Transparencia separada de los colores visibles.
    visibles = Counter(
        color for color, cantidad in colores.items()
        if color[3] > 0
    )

    transparentes = sum(
        cantidad for color, cantidad in colores.items()
        if color[3] == 0
    )

    resultado = {
        "archivo": str(ruta),
        "ancho": ancho,
        "alto": alto,
        "modo_original": Image.open(ruta).mode,
        "pixeles_totales": len(pixeles),
        "pixeles_transparentes": transparentes,
        "colores_totales": len(colores),
        "colores_visibles": len(visibles),
        "paleta_real": [
            {
                "r": color[0],
                "g": color[1],
                "b": color[2],
                "a": color[3],
                "pixeles": cantidad,
            }
            for color, cantidad in visibles.most_common()
        ],
        "dbz_decodificado": False,
        "motivo": (
            "La correspondencia color -> dBZ todavía no se fuerza "
            "hasta validar la escala oficial del producto."
        ),
    }

    return resultado


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Uso: python scripts/decodificador_rma10_dbz.py "
            "<imagen_RMA10.png>"
        )
        return 2

    ruta = Path(sys.argv[1])

    try:
        resultado = analizar_png(ruta)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    salida = ruta.with_name("rma10_analisis_paleta.json")
    salida.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"OK: {ruta}")
    print(f"Resolución: {resultado['ancho']}x{resultado['alto']}")
    print(f"Colores visibles: {resultado['colores_visibles']}")
    print(f"Resultado: {salida}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
