#!/usr/bin/env python3
"""¿Cuánto confunden al seguimiento las tormentas que se dividen o se unen?

Vuelve a correr el seguimiento (seguimiento_temporal.procesar_frame_secuencia)
sobre las imágenes guardadas en data/radar/historico, día por día y en orden,
con la misma detección de núcleos que en producción. No escribe nada del repo.

Mide, en cada división (split) y unión (merge):
- la velocidad que le queda a la tormenta en ese ciclo (km/h);
- el cambio de tamaño que informa (lo que lee el motor de decisión para
  "crece" o "se debilita");
- si la tormenta hija arranca con dirección o sin movimiento.

Y para todas las tormentas: el error de la posición a 10, 20 y 30 min usando
el historial de cada una (recta por sus últimas posiciones, como el predictor
de evolución), comparado con dónde estuvo de verdad esa misma tormenta.

Uso: python herramientas/medir_division_union.py [--dias 20261027,20261112] [--salida x.json]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import seguimiento_temporal as st  # noqa: E402

DIAS_DEFECTO = "20261027,20261028,20261101,20261103,20261105,20261112,20261113,20261120"
KM_PX = 240.0 / 255.5
HORIZONTES = (10, 20, 30)


def velocidad_historial(historial, puntos=4):
    """Recta por las últimas posiciones (px/min), como prediccion_evolucion."""
    h = [p for p in historial if p.get("x") is not None][-puntos:]
    if len(h) < 3:
        return None
    t0 = st.parse_iso(h[0]["timestamp"])
    ts = [(st.parse_iso(p["timestamp"]) - t0).total_seconds() / 60.0 for p in h]
    tm = statistics.mean(ts)
    den = sum((t - tm) ** 2 for t in ts)
    if den <= 1e-9:
        return None
    vx = sum((t - tm) * (p["x"] - statistics.mean(q["x"] for q in h)) for t, p in zip(ts, h)) / den
    vy = sum((t - tm) * (p["y"] - statistics.mean(q["y"] for q in h)) for t, p in zip(ts, h)) / den
    return vx, vy


def salto(previo, t):
    """Cambio de velocidad respecto del ciclo anterior (km/h), si había movimiento previo."""
    p = previo.get(t["track_id"])
    if p is None or p[3] < 2:
        return None
    return math.hypot(t.get("velocidad_x_pixeles_min", 0) - p[0],
                      t.get("velocidad_y_pixeles_min", 0) - p[1]) * KM_PX * 60


def correr_dia(dia, detectores):
    detectar, cargar_mascara, analizar, confianza, aplicar = detectores
    mascara = cargar_mascara()
    frames = sorted((ROOT / "data/radar/historico").glob(f"{dia}*.png"))
    estado = st.estado_nuevo()
    anterior = None
    eventos = []
    pronosticos = []  # (track_id, objetivo_dt, horizonte, x, y)
    posiciones = {}  # (track_id, iso) -> (x, y)
    for frame in frames:
        dt = st.parse_timestamp_filename(frame.stem)
        if dt is None:
            continue
        ts = st.iso(dt)
        nucleos = st.nucleos_desde_imagen(frame, mascara, detectar, analizar, confianza, aplicar)
        if anterior is not None and (dt - st.parse_iso(anterior)).total_seconds() / 60.0 > st.MAX_INTERVALO:
            estado = st.estado_nuevo()
            anterior = None
        previo = {t["track_id"]: (t.get("velocidad_x_pixeles_min", 0), t.get("velocidad_y_pixeles_min", 0),
                                  t.get("pixeles", 0), len(t.get("historial") or []))
                  for t in estado["tracks"] if t.get("activo")}
        salida = st.procesar_frame_secuencia(estado, nucleos, ts, anterior)
        anterior = ts
        por_id = {t["track_id"]: t for t in estado["tracks"]}
        for ev in salida.get("eventos_split") or []:
            padre = por_id.get(ev["track_padre"])
            hijo = next((t for t in estado["tracks"] if t.get("parent_track_id") == ev["track_padre"]
                         and t.get("primer_frame") == ts), None)
            if padre is None:
                continue
            eventos.append({
                "tipo": "division", "utc": ts, "salto_kmh": salto(previo, padre),
                "fraccion": (hijo.get("pixeles", 0) / max(1, previo.get(padre["track_id"], (0, 0, 1))[2])) if hijo else 0,
                "kmh": padre.get("velocidad_pixeles_min", 0) * KM_PX * 60,
                "cambio_area": padre.get("cambio_area_porcentaje", 0),
                "hijo_con_direccion": bool(hijo and hijo.get("rumbo_grados") is not None),
                "hijo_kmh": (hijo.get("velocidad_pixeles_min", 0) * KM_PX * 60) if hijo else None,
            })
        for ev in salida.get("eventos_merge") or []:
            principal = por_id.get(ev["track_principal"])
            if principal is None:
                continue
            eventos.append({
                "tipo": "union", "utc": ts, "salto_kmh": salto(previo, principal),
                "fraccion": previo.get(ev["track_fusionado"], (0, 0, 0))[2]
                / max(1, previo.get(principal["track_id"], (0, 0, 1))[2]),
                "kmh": principal.get("velocidad_pixeles_min", 0) * KM_PX * 60,
                "cambio_area": principal.get("cambio_area_porcentaje", 0),
            })
        presentes = [t for t in estado["tracks"] if t.get("activo") and t.get("presente_en_ultimo_frame")]
        con_evento = {e.get("track_padre") for e in salida.get("eventos_split") or []}
        con_evento |= {e.get("track_principal") for e in salida.get("eventos_merge") or []}
        for t in presentes:
            if t["track_id"] in con_evento or len(t.get("historial") or []) < 2:
                continue
            eventos.append({"tipo": "normal", "utc": ts, "salto_kmh": salto(previo, t), "fraccion": 0,
                            "kmh": t.get("velocidad_pixeles_min", 0) * KM_PX * 60,
                            "cambio_area": t.get("cambio_area_porcentaje", 0)})
        for t in presentes:
            posiciones[(t["track_id"], ts)] = (t["x"], t["y"])
            v = velocidad_historial(t.get("historial") or [])
            if v is None:
                continue
            for h in HORIZONTES:
                pronosticos.append((t["track_id"], dt + timedelta(minutes=h), h,
                                    t["x"] + v[0] * h, t["y"] + v[1] * h))
    errores = defaultdict(list)
    tiempos = sorted({st.parse_iso(k[1]) for k in posiciones})
    for tid, objetivo, h, x, y in pronosticos:
        cerca = [t for t in tiempos if abs((t - objetivo).total_seconds()) <= 300]
        if not cerca:
            continue
        real = posiciones.get((tid, st.iso(min(cerca, key=lambda t: abs((t - objetivo).total_seconds())))))
        if real is None:
            continue
        errores[h].append(math.hypot(real[0] - x, real[1] - y) * KM_PX)
    return eventos, errores


def resumen(eventos, errores):
    def med(vals):
        return round(statistics.median(vals), 1) if vals else None

    def p90(vals):
        return round(sorted(vals)[int(0.9 * (len(vals) - 1))], 1) if vals else None

    salida = {}
    grupos = {
        "division": lambda e: e["tipo"] == "division",
        "division_grande": lambda e: e["tipo"] == "division" and e["fraccion"] >= 0.3,
        "union": lambda e: e["tipo"] == "union",
        "union_grande": lambda e: e["tipo"] == "union" and e["fraccion"] >= 0.3,
        "normal": lambda e: e["tipo"] == "normal",
    }
    for tipo, cond in grupos.items():
        ev = [e for e in eventos if cond(e)]
        saltos = [e["salto_kmh"] for e in ev if e.get("salto_kmh") is not None]
        cambios = [e["cambio_area"] for e in ev]
        salida[tipo] = {
            "cantidad": len(ev),
            "velocidad_kmh_mediana": med([e["kmh"] for e in ev]),
            "velocidad_kmh_p90": p90([e["kmh"] for e in ev]),
            "cambio_area_mediano_pct": med(cambios),
            "salto_velocidad_kmh_mediana": med(saltos),
            "salto_velocidad_kmh_p90": p90(saltos),
            "parecen_debilitarse_pct": round(100 * sum(c <= -15 for c in cambios) / len(ev), 1) if ev else None,
            "parecen_crecer_pct": round(100 * sum(c >= 15 for c in cambios) / len(ev), 1) if ev else None,
        }
        if tipo.startswith("division"):
            salida[tipo]["hijas_con_direccion_pct"] = (
                round(100 * sum(e["hijo_con_direccion"] for e in ev) / len(ev), 1) if ev else None)
    salida["error_posicion_km"] = {
        str(h): {"mediana": med(errores[h]), "p90": p90(errores[h]), "casos": len(errores[h])}
        for h in HORIZONTES
    }
    return salida


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dias", default=DIAS_DEFECTO)
    ap.add_argument("--salida")
    args = ap.parse_args(argv)
    from scripts.procesador_radar import (analizar_mascara, aplicar_mascara_estatica, calcular_confianza,
                                          cargar_mascara, detectar_nucleos)
    detectores = (detectar_nucleos, cargar_mascara, analizar_mascara, calcular_confianza, aplicar_mascara_estatica)
    eventos, errores = [], defaultdict(list)
    for dia in args.dias.split(","):
        ev, er = correr_dia(dia.strip(), detectores)
        eventos += ev
        for h, v in er.items():
            errores[h] += v
        print(f"{dia}: {sum(e['tipo'] == 'division' for e in ev)} divisiones, "
              f"{sum(e['tipo'] == 'union' for e in ev)} uniones", flush=True)
    res = resumen(eventos, errores)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.salida:
        Path(args.salida).write_text(json.dumps({"resumen": res, "eventos": eventos}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
