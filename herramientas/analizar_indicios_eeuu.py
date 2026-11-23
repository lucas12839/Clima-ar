#!/usr/bin/env python3
"""Etapa G: mide los indicios contra la verdad oficial (salida de simular_indicios_eeuu.py).

Para cada indicio y cada corte posible:
- de cada 100 veces que el indicio se activa, en cuántas la tormenta dio un reporte
  oficial severo en la hora siguiente (acierto = "precisión");
- comparado con lo mismo para TODAS las tormentas con núcleo fuerte (lo que ya se sabe
  sin el indicio): si no lo supera claramente, el indicio no aporta;
- de las tormentas que dieron algún reporte, cuántas fueron marcadas ANTES (anticipación).

Uso: python herramientas/analizar_indicios_eeuu.py resultados/*.json [--salida x.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def severo(rep: list, tipo: str | None = None, desde: float = 0.0) -> bool:
    """Reporte oficial que toca la tormenta entre `desde` y +60 min. Tipos:
    viento (ráfaga >= 93 km/h), granizo (>= 2,5 cm), granizo_grande (>= 5 cm), tornado."""
    for t, mag, medido, dt, d in rep:
        if dt < desde:
            continue
        if tipo is None or t == tipo:
            return True
        if tipo == "granizo_grande" and t == "granizo" and (mag or 0) >= 2.0:
            return True
        if tipo == "supercelda" and (t == "tornado" or (t == "granizo" and (mag or 0) >= 2.0)):
            return True
    return False


def tabla(registros: list[dict], activo, tipo: str | None) -> dict:
    a = b = 0
    for r in registros:
        if activo(r):
            if severo(r["reportes"], tipo):
                a += 1
            else:
                b += 1
    return {"activaciones": a + b, "acierto_pct": round(100 * a / (a + b), 1) if a + b else None}


def anticipacion(registros_por_tray: dict, activo, tipo: str | None) -> dict:
    """De las tormentas con reporte, ¿cuántas se marcaron antes del primer reporte?"""
    con, antes, minutos = 0, 0, []
    for regs in registros_por_tray.values():
        regs = sorted(regs, key=lambda r: r["min"])
        primero = None
        for r in regs:
            for t, mag, med, dt, d in r["reportes"]:
                if severo([[t, mag, med, dt, d]], tipo):
                    tr = r["min"] + dt
                    primero = tr if primero is None else min(primero, tr)
        if primero is None:
            continue
        con += 1
        marcas = [r["min"] for r in regs if activo(r) and r["min"] <= primero]
        if marcas:
            antes += 1
            minutos.append(primero - min(marcas))
    minutos.sort()
    return {"tormentas_con_reporte": con, "marcadas_antes_pct": round(100 * antes / con, 1) if con else None,
            "anticipacion_mediana_min": minutos[len(minutos) // 2] if minutos else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archivos", nargs="+")
    ap.add_argument("--salida", default=None)
    a = ap.parse_args(argv)
    registros, por_tray = [], defaultdict(list)
    casos = 0
    for p in a.archivos:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        casos += 1
        clave = d["caso"]["dia"] + d["caso"]["centro"]
        for r in d["registros"]:
            r["caso"] = clave
            registros.append(r)
            por_tray[(clave, r["id"])].append(r)
    nucleo = [r for r in registros if r["pico_dbz"] >= 45]
    res = {"casos": casos, "registros": len(registros), "con_nucleo_45": len(nucleo), "indicios": {}}
    base = {}
    for tipo in (None, "viento", "granizo_grande", "tornado", "supercelda"):
        base[str(tipo)] = tabla(nucleo, lambda r: True, tipo)
    res["base_todas_las_tormentas_45dbz"] = base
    for p in (50, 55, 60, 65):
        res["indicios"][f"pico>={p}"] = {str(t): tabla(nucleo, lambda r, p=p: r["pico_dbz"] >= p, t)
                                         for t in (None, "granizo_grande")}
    pruebas = {}
    for c in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20):
        for largo in (30, 40, 60, 80):
            pruebas[f"arco c>={c} largo>={largo}"] = (
                lambda r, c=c, largo=largo: (r.get("curvatura") or 0) >= c and (r.get("largo_km") or 0) >= largo, "viento")
    for dmin in (10, 15, 20, 25, 30, 40):
        pruebas[f"desvio>={dmin}"] = (lambda r, dmin=dmin: (r.get("desvio_kmh") or -999) >= dmin, "supercelda")
    for dur in (30, 45, 60, 90, 120):
        pruebas[f"duracion>={dur}"] = (lambda r, dur=dur: r["duracion_min"] >= dur, None)
    for nombre, (f, tipo) in pruebas.items():
        res["indicios"][nombre] = {"tipo": str(tipo), **tabla(nucleo, f, tipo), **tabla_extra(nucleo, f),
                                   **anticipacion(por_tray, f, tipo)}
    texto = json.dumps(res, ensure_ascii=False, indent=1)
    if a.salida:
        Path(a.salida).write_text(texto, encoding="utf-8")
    print(texto)
    return 0


def tabla_extra(registros, activo) -> dict:
    """Acierto contra CUALQUIER reporte severo (para comparar con la base)."""
    t = tabla(registros, activo, None)
    return {"acierto_cualquier_severo_pct": t["acierto_pct"]}


if __name__ == "__main__":
    sys.exit(main())
