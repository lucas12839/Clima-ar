#!/usr/bin/env python3
"""
Decoder cuantitativo independiente para RMA10 ZH_MAX.

Esta etapa NO integra el dBZ al procesador operativo.
Solo decodifica la reflectividad del PNG RMA10 y genera un resumen
cuantitativo seguro para validacion.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

DBZ_MIN = -15.0
DBZ_MAX = 75.0
PALETTE_INDEX_MIN = 64
PALETTE_INDEX_MAX = 207

# Anclas observadas en la paleta real RMA10 del artifact validado.
EXPECTED_ANCHORS = {
    64: (57, 113, 163),
    88: (40, 151, 198),
    104: (77, 225, 51),
    136: (225, 228, 57),
    167: (203, 0, 27),
    191: (234, 0, 234),
    207: (255, 255, 255),
}


def _palette_rgb(img: Image.Image, index: int) -> tuple[int, int, int]:
    palette = img.getpalette()
    if palette is None or len(palette) < (index + 1) * 3:
        raise ValueError("La imagen no contiene una paleta P valida.")
    start = index * 3
    return tuple(palette[start : start + 3])  # type: ignore[return-value]


def validar_paleta(img: Image.Image) -> None:
    if img.mode != "P":
        raise ValueError(f"Modo RMA10 inesperado: {img.mode}. Se esperaba P.")

    transparencia = img.info.get("transparency")
    if not isinstance(transparencia, int):
        raise ValueError("No se encontro el indice de transparencia esperado.")

    for index, esperado in EXPECTED_ANCHORS.items():
        obtenido = _palette_rgb(img, index)
        if obtenido != esperado:
            raise ValueError(
                f"Paleta RMA10 no reconocida en indice {index}: "
                f"{obtenido} != {esperado}. Decodificacion cancelada."
            )


def indice_a_dbz(index: int) -> float:
    if index <= PALETTE_INDEX_MIN:
        return DBZ_MIN
    if index >= PALETTE_INDEX_MAX:
        return DBZ_MAX
    proporcion = (index - PALETTE_INDEX_MIN) / (
        PALETTE_INDEX_MAX - PALETTE_INDEX_MIN
    )
    return DBZ_MIN + proporcion * (DBZ_MAX - DBZ_MIN)


def analizar_png(ruta: Path) -> dict:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe: {ruta}")

    with Image.open(ruta) as img:
        validar_paleta(img)
        ancho, alto = img.size
        transparencia = img.info["transparency"]
        indices = list(img.get_flattened_data())
        paleta = img.getpalette()

    conteo = Counter(indices)
    transparentes = conteo.get(transparencia, 0)
    visibles = [
        index
        for index in indices
        if index != transparencia
    ]

    if not visibles:
        raise ValueError("La imagen no contiene pixeles de radar visibles.")

    valores_dbz = [indice_a_dbz(index) for index in visibles]
    valores_ordenados = sorted(valores_dbz)

    def percentil(p: float) -> float:
        pos = (len(valores_ordenados) - 1) * p
        bajo = int(pos)
        alto = min(bajo + 1, len(valores_ordenados) - 1)
        fraccion = pos - bajo
        return valores_ordenados[bajo] + (
            valores_ordenados[alto] - valores_ordenados[bajo]
        ) * fraccion

    histograma_5dbz = {}
    for inicio in range(-15, 75, 5):
        fin = inicio + 5
        histograma_5dbz[f"{inicio}:{fin}"] = sum(
            inicio <= valor < fin for valor in valores_dbz
        )
    histograma_5dbz["75:mas"] = sum(valor >= 75 for valor in valores_dbz)

    colores_visibles = Counter(
        (
            paleta[index * 3],
            paleta[index * 3 + 1],
            paleta[index * 3 + 2],
        )
        for index in visibles
    )

    return {
        "archivo": str(ruta),
        "ancho": ancho,
        "alto": alto,
        "modo_original": "P",
        "indice_transparencia": transparencia,
        "pixeles_totales": len(indices),
        "pixeles_transparentes": transparentes,
        "pixeles_radar": len(visibles),
        "colores_visibles": len(colores_visibles),
        "dbz_decodificado": True,
        "escala_dbz": {
            "min": DBZ_MIN,
            "max": DBZ_MAX,
            "indice_min": PALETTE_INDEX_MIN,
            "indice_max": PALETTE_INDEX_MAX,
            "metodo": "interpolacion_lineal_paleta_rma10_validada",
        },
        "dbz": {
            "min": min(valores_dbz),
            "max": max(valores_dbz),
            "media": sum(valores_dbz) / len(valores_dbz),
            "p95": percentil(0.95),
            "p99": percentil(0.99),
        },
        "histograma_5dbz": histograma_5dbz,
        "pixeles_por_indice": dict(sorted(conteo.items())),
        "paleta_validada": {
            str(index): {
                "rgb": tuple(paleta[index * 3 : index * 3 + 3]),
                "dbz": indice_a_dbz(index),
            }
            for index in sorted(conteo)
            if index != transparencia
        },
    }


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

    salida = ruta.with_name("rma10_analisis_dbz.json")
    salida.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"OK: {ruta}")
    print(f"Resolucion: {resultado['ancho']}x{resultado['alto']}")
    print(f"Pixeles radar: {resultado['pixeles_radar']}")
    print(f"dBZ min/max: {resultado['dbz']['min']:.2f} / {resultado['dbz']['max']:.2f}")
    print(f"dBZ medio: {resultado['dbz']['media']:.2f}")
    print(f"Resultado: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
