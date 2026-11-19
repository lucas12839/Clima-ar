#!/usr/bin/env python3
"""Valida el nowcast de ClimaAR contra eventos históricos (imágenes del SMN).

Para cada evento (carpeta con imágenes "Bahía Blanca-SINARAME ZH MAX"):
- Línea de tiempo observada sobre Bahía Blanca (radio 10 km, >=5 km² de eco).
- En cada frame, pronóstico por flujo óptico con el frame anterior
  (5-30 min antes), igual que en producción.
- Llegada observada de eco >=35 y >=45 dBZ, y con cuánta anticipación lo
  avisó el nowcast (ETA pronosticada coherente con la llegada real ±20 min).
- Velocidad y dirección del sistema según el flujo óptico.

Uso: python herramientas/validar_eventos.py <carpeta_evento> [<carpeta> ...] [--salida archivo.json]
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT / "herramientas"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import decodificar_imagen_smn as dec  # noqa: E402
import nowcast_flujo_optico as nf  # noqa: E402
from scripts.rma10_features import BAHIA_LAT, BAHIA_LON, latlon_to_pixel  # noqa: E402

TOLERANCIA_ETA_MIN = 20


def categoria(dbz: float) -> str:
    if dbz < 20:
        return "sin_lluvia_o_llovizna"
    if dbz < 30:
        return "lluvia_debil"
    if dbz < 40:
        return "lluvia_moderada"
    if dbz < 45:
        return "lluvia_fuerte"
    if dbz < 50:
        return "tormenta"
    if dbz < 55:
        return "tormenta_fuerte"
    return "tormenta_muy_fuerte"


def analizar(frames, nombre):
    shape = frames[0][1].shape
    # Artefactos fijos del mapa: >=55 dBZ en más del 30 % de los frames.
    alto = np.mean([np.nan_to_num(g, nan=-99) >= 55 for _, g in frames], axis=0)
    artefactos = alto > 0.3
    sin_clutter = np.zeros(shape, dtype=bool)

    cx, cy = latlon_to_pixel(BAHIA_LAT, BAHIA_LON, shape[1], shape[0])
    zona = nf.disco(shape, cx, cy, nf.RADIO_CIUDAD_KM / nf.km_por_pixel(shape[1]))

    def limpio(g):
        g = g.copy()
        g[artefactos] = np.nan
        return g

    def dbz_ciudad(campo):
        vals = np.sort(campo[zona])
        return float(vals[-nf.MIN_PIXELES_CIUDAD]) if vals.size >= nf.MIN_PIXELES_CIUDAD else 0.0

    serie = []
    for t, g in frames:
        c = nf.campo_limpio(limpio(g), sin_clutter)
        serie.append({"utc": t.isoformat(), "dbz_ciudad": round(dbz_ciudad(c), 1),
                      "categoria": categoria(dbz_ciudad(c))})

    def primera(umbral):
        for s in serie:
            if s["dbz_ciudad"] >= umbral:
                return s["utc"]
        return None

    llegada = {"35": primera(35), "45": primera(45)}

    pron = []
    for i in range(1, len(frames)):
        t, g = frames[i]
        previo = None
        for j in range(i - 1, -1, -1):
            gap = (t - frames[j][0]).total_seconds() / 60
            if gap > nf.GAP_MAX_MIN:
                break
            if gap >= nf.GAP_MIN_MIN:
                previo = (frames[j][0], frames[j][1], gap)
                break
        if previo is None:
            continue
        res = nf.pronosticar(limpio(previo[1]), limpio(g), previo[2], sin_clutter)
        if not res["flujo"]["valido"]:
            continue
        imp = nf.impacto_ciudad(res["actual"], res["flujo"])
        pron.append({
            "utc": t.isoformat(),
            "intervalo_min": round(previo[2], 1),
            "velocidad_kmh": res["flujo"]["velocidad_media_kmh"],
            "direccion_hacia": res["flujo"]["direccion_hacia_grados"],
            "coherencia": res["flujo"]["coherencia"],
            "eta_35_min": imp["eta_tormenta_min"],
            "dbz_al_llegar": imp["dbz_al_llegar"],
            "eta_45_min": imp["eta_tormenta_fuerte_min"],
            "dbz_ciudad_ahora": imp["dbz_max_ahora"],
        })

    def anticipacion(clave_eta, umbral):
        real = llegada[umbral]
        if real is None:
            return None
        from datetime import datetime
        real_dt = datetime.fromisoformat(real)
        mejor = None
        for p in pron:
            t = datetime.fromisoformat(p["utc"])
            if t >= real_dt:
                break
            eta = p[clave_eta]
            if eta is None or eta == 0:
                continue
            pred = t + timedelta(minutes=eta)
            if abs((pred - real_dt).total_seconds()) / 60 <= TOLERANCIA_ETA_MIN:
                lead = (real_dt - t).total_seconds() / 60
                mejor = max(mejor or 0, lead)
        return round(mejor, 1) if mejor is not None else 0.0

    vel = [p["velocidad_kmh"] for p in pron if p["coherencia"] >= 0.6]
    maximo = max(s["dbz_ciudad"] for s in serie)
    return {
        "evento": nombre,
        "frames": len(frames),
        "desde_utc": frames[0][0].isoformat(),
        "hasta_utc": frames[-1][0].isoformat(),
        "artefactos_filtrados_px": int(artefactos.sum()),
        "llegada_observada": llegada,
        "anticipacion_min": {
            "eco_35dbz": anticipacion("eta_35_min", "35"),
            "eco_45dbz": anticipacion("eta_45_min", "45"),
        },
        "maximo_sobre_ciudad_dbz": maximo,
        "categoria_maxima": categoria(maximo),
        "velocidad_flujo_kmh": {
            "mediana": round(float(np.median(vel)), 1) if vel else None,
            "p90": round(float(np.percentile(vel, 90)), 1) if vel else None,
        },
        "serie_ciudad": serie,
        "pronosticos": pron,
    }


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    salida = None
    if "--salida" in argv:
        i = argv.index("--salida")
        salida = argv[i + 1]
        del argv[i:i + 2]
    resultados = []
    for carpeta in argv:
        frames = dec.cargar_evento(Path(carpeta))
        r = analizar(frames, Path(carpeta).name)
        resultados.append(r)
        resumen = {k: v for k, v in r.items() if k not in ("serie_ciudad", "pronosticos")}
        print(json.dumps(resumen, ensure_ascii=False, indent=1))
    if salida:
        Path(salida).write_text(json.dumps(resultados, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
