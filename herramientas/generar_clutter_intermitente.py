#!/usr/bin/env python3
"""Genera config/clutter_intermitente_rma10.png (ecos fijos que aparecen y desaparecen).

Complementa a config/clutter_rma10.png: usa solo imágenes sin lluvia, donde
cualquier eco repetido es del terreno o de propagación anómala.
Uso: python herramientas/generar_clutter_intermitente.py [--cada 2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import nowcast_capas_nucleo as nc  # noqa: E402
import nowcast_flujo_optico as nf  # noqa: E402

SALIDA = ROOT / "config" / "clutter_intermitente_rma10.png"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cada", type=int, default=2)
    args = ap.parse_args(argv)
    frames = sorted(nf.HISTORICO.glob("20*.png"))[:: args.cada]
    clutter = None

    def campos():
        nonlocal clutter
        for p in frames:
            d = nf.cargar_dbz(p)
            if d is None:
                continue
            if clutter is None:
                clutter = nf.cargar_mascara_clutter(d.shape)
            yield nf.campo_limpio(d, clutter)

    mask = nc.mascara_clutter_tranquilo(campos())
    if mask is None:
        print("No hay suficientes imágenes tranquilas.")
        return 1
    from PIL import Image

    Image.fromarray((mask * 255).astype(np.uint8)).save(SALIDA)
    print(f"Clutter intermitente: {int(mask.sum())} píxeles -> {SALIDA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
