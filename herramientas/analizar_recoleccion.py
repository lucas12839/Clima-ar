#!/usr/bin/env python3
"""Qué fenómenos juntó el recolector de radares (Córdoba, Resistencia, Tostado).

Solo LEE las imágenes de /var/lib/climaar-recoleccion (no escribe nada ahí ni
en el proyecto). Por cada imagen: intensidad (dBZ) con la paleta de SINARAME,
sin los ecos fijos (máscara del recolector). Por cada día y radar cuenta las
imágenes con:

  lluvia            >= 20 dBZ en 20 km² o más
  tormenta          >= 45 dBZ en 10 km² o más
  tormenta fuerte   >= 50 dBZ en 10 km² o más
  posible granizo   >= 55 dBZ en 5 km² o más  (núcleo muy intenso)

Cada fenómeno cuenta solo si es un ÁREA CONTINUA (una celda), no píxeles
sueltos (ecos del terreno que la máscara no tapa), y fuera de los primeros 5 km
alrededor del radar. Además, un lugar con eco en más del 20 % de todas las
imágenes del período es terreno (una tormenta no se queda tanto en un lugar).

Es solo lo que ve el radar (sin rayos, satélite ni estaciones de esas zonas):
"posible granizo" es una señal del radar, no una confirmación.

Uso en el VPS:
  /opt/Clima-ar/.venv-ia/bin/python /opt/Clima-ar/herramientas/analizar_recoleccion.py
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
from decodificador_rma10_dbz import indice_a_dbz, validar_paleta  # noqa: E402

RADIO_KM = 240.0
CENTRO_EXCLUIDO_KM = 5.0
PERSISTENCIA_TERRENO = 0.20   # eco en más del 20 % de las imágenes del período = terreno
MUESTRA_PERSISTENCIA = 400
NOMBRES = {"RMA1": "Córdoba", "RMA4": "Resistencia", "RMA21": "Tostado"}
CLASES = [  # (nombre, dBZ mínimo, km² mínimos)
    ("lluvia", 20.0, 20.0),
    ("tormenta", 45.0, 10.0),
    ("tormenta fuerte", 50.0, 10.0),
    ("posible granizo", 55.0, 5.0),
]
TABLA_DBZ = np.array([indice_a_dbz(i) for i in range(256)], dtype=np.float32)


def leer(ruta: Path, mascara: np.ndarray | None):
    with Image.open(ruta) as img:
        validar_paleta(img)
        idx = np.asarray(img)
        transp = img.info["transparency"]
    dbz = TABLA_DBZ[idx]
    dbz[idx == transp] = np.nan
    if mascara is not None and mascara.shape == dbz.shape:
        dbz[mascara] = np.nan
    return dbz


def area_mayor(binaria: np.ndarray) -> int:
    """Píxeles de la celda continua más grande."""
    from scipy import ndimage

    etiquetas, n = ndimage.label(binaria)
    if n == 0:
        return 0
    return int(np.bincount(etiquetas.ravel())[1:].max())


def celda_maxima(dbz: np.ndarray, km2px: float) -> float:
    """dBZ más alto que forma una celda de 5 km² o más (no un píxel suelto)."""
    for umbral in np.arange(70.0, 10.0, -2.5):
        if area_mayor(dbz >= umbral) * km2px >= 5.0:
            return float(umbral)
    return -99.0


def analizar(destino: Path, radares: list[str]) -> int:
    for radar in radares:
        carpeta = destino / radar
        archivos = sorted(p for p in carpeta.glob("*/*.png") if p.stem.isdigit() and len(p.stem) == 14)
        nombre = NOMBRES.get(radar, radar)
        if not archivos:
            print(f"\n== {nombre} ({radar}): sin imágenes")
            continue
        mascara = None
        mpng = carpeta / "mascara_ecos_fijos.png"
        if mpng.exists():
            with Image.open(mpng) as m:
                mascara = np.asarray(m.convert("L")) > 127
        # Terreno: lo que tiene eco en más del 20 % de las imágenes (muestra pareja).
        paso = max(1, len(archivos) // MUESTRA_PERSISTENCIA)
        cuenta, n_muestra = None, 0
        for ruta in archivos[::paso]:
            try:
                d0 = leer(ruta, mascara)
            except Exception:
                continue
            eco = np.nan_to_num(d0, nan=-99.0) >= 20.0
            cuenta = eco.astype(np.uint16) if cuenta is None else cuenta + eco
            n_muestra += 1
        if cuenta is not None and n_muestra >= 20:
            terreno = cuenta > PERSISTENCIA_TERRENO * n_muestra
            mascara = terreno if mascara is None else (mascara | terreno)
        por_dia: dict[str, dict] = defaultdict(lambda: {"imagenes": 0, "max": -99.0, **{c[0]: 0 for c in CLASES},
                                                        "hora_max": ""})
        malas = 0
        km2px = None
        for ruta in archivos:
            try:
                dbz = leer(ruta, mascara)
            except Exception:
                malas += 1
                continue
            if km2px is None:
                kmpx = RADIO_KM / ((dbz.shape[1] - 1) / 2.0)
                km2px = kmpx * kmpx
                yy, xx = np.indices(dbz.shape)
                cy, cx = (dbz.shape[0] - 1) / 2.0, (dbz.shape[1] - 1) / 2.0
                centro = np.hypot(yy - cy, xx - cx) * kmpx <= CENTRO_EXCLUIDO_KM
            dbz[centro] = np.nan
            d = por_dia[ruta.stem[:8]]
            d["imagenes"] += 1
            mx = celda_maxima(dbz, km2px)
            if mx > d["max"]:
                d["max"], d["hora_max"] = mx, f"{ruta.stem[8:10]}:{ruta.stem[10:12]}"
            for clase, umbral, km2 in CLASES:
                if area_mayor(dbz >= umbral) * km2px >= km2:
                    d[clase] += 1
        print(f"\n== {nombre} ({radar}): {len(archivos)} imágenes"
              f"{f', {malas} no se pudieron leer' if malas else ''}"
              f"{'' if mascara is not None else ' (todavía sin máscara de ecos fijos: puede contar ecos del terreno)'}")
        print("día        imágenes  lluvia  tormenta  fuerte  pos.granizo  celda más intensa dBZ (hora)")
        tot = defaultdict(int)
        for dia in sorted(por_dia):
            d = por_dia[dia]
            print(f"{dia[6:8]}/{dia[4:6]}/{dia[:4]}  {d['imagenes']:8d}  {d['lluvia']:6d}  {d['tormenta']:8d}"
                  f"  {d['tormenta fuerte']:6d}  {d['posible granizo']:11d}  {d['max']:5.1f} ({d['hora_max']})")
            for k in ("imagenes", *[c[0] for c in CLASES]):
                tot[k] += d[k]
        print(f"TOTAL      {tot['imagenes']:8d}  {tot['lluvia']:6d}  {tot['tormenta']:8d}  {tot['tormenta fuerte']:6d}"
              f"  {tot['posible granizo']:11d}   (cada imagen = 10 min)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--destino", default="/var/lib/climaar-recoleccion")
    p.add_argument("--radares", default="RMA1,RMA4,RMA21")
    a = p.parse_args()
    return analizar(Path(a.destino), [r.strip() for r in a.radares.split(",") if r.strip()])


if __name__ == "__main__":
    sys.exit(main())
