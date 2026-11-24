#!/usr/bin/env python3
"""¿Qué velocidades informa el seguimiento y cuánto se equivoca al proyectarlas?

Vuelve a correr el seguimiento (seguimiento_temporal.procesar_frame_secuencia)
sobre las imágenes guardadas en data/radar/historico, día por día y en orden,
con la misma detección de núcleos que en producción. No escribe nada del repo.

Mide:
- velocidad de cada tramo (cada vez que una tormenta con pasado se vuelve a
  ver): cuántos pasan de 80 y 100 km/h y el máximo;
- cuántos tramos quedan marcados como dudosos;
- error de la posición a 10, 20 y 30 min proyectando la velocidad guardada
  de cada tormenta, contra dónde estuvo de verdad esa misma tormenta;
- continuidad: tormentas creadas y vida media (para ver que el arreglo no
  rompa la identidad de las tormentas).

Uso: python herramientas/medir_velocidades.py [--dias 20261027,20261113] [--salida x.json]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import seguimiento_temporal as st  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import prediccion_evolucion as pe  # noqa: E402

DIAS_DEFECTO = "20261027,20261028,20261101,20261103,20261105,20261112,20261113,20261120,20261121,20261122"
KM_PX = 240.0 / 255.5
HORIZONTES = (10, 20, 30)


def correr_dia(dia, detectores):
    detectar, cargar_mascara, analizar, confianza, aplicar = detectores
    mascara = cargar_mascara()
    frames = sorted((ROOT / "data/radar/historico").glob(f"{dia}*.png"))
    estado = st.estado_nuevo()
    anterior = None
    tramos = []
    pronosticos = []
    posiciones = {}
    vistos = defaultdict(int)
    for frame in frames:
        dt = st.parse_timestamp_filename(frame.stem)
        if dt is None:
            continue
        ts = st.iso(dt)
        nucleos = st.nucleos_desde_imagen(frame, mascara, detectar, analizar, confianza, aplicar)
        if anterior is not None and (dt - st.parse_iso(anterior)).total_seconds() / 60.0 > st.MAX_INTERVALO:
            estado = st.estado_nuevo()
            anterior = None
        st.procesar_frame_secuencia(estado, nucleos, ts, anterior, frame_actual=frame)
        anterior = ts
        presentes = [t for t in estado["tracks"] if t.get("activo") and t.get("presente_en_ultimo_frame")]
        for t in presentes:
            vistos[t["track_id"]] += 1
            posiciones[(t["track_id"], ts)] = (t["x"], t["y"])
            if t.get("ultimo_frame") != ts or len(t.get("historial") or []) < 2:
                continue
            tramos.append({
                "utc": ts, "track": t["track_id"],
                "kmh": t.get("velocidad_pixeles_min", 0) * KM_PX * 60,
                "kmh_centro": t.get("velocidad_centro_pixeles_min", t.get("velocidad_pixeles_min", 0)) * KM_PX * 60,
                "dudoso": bool(t.get("movimiento_dudoso")),
                "pixeles": t.get("pixeles", 0),
            })
            vx, vy = t.get("velocidad_x_pixeles_min", 0), t.get("velocidad_y_pixeles_min", 0)
            for h in HORIZONTES:
                pronosticos.append((t["track_id"], dt + timedelta(minutes=h), h, t["x"] + vx * h, t["y"] + vy * h,
                                    t.get("pixeles", 0), "seguimiento"))
            # Lo que proyecta la predicción de evolución (la que se dibuja en la app).
            m = pe.estimate_motion(t)
            if m is not None:
                for h in HORIZONTES:
                    pronosticos.append((t["track_id"], dt + timedelta(minutes=h), h, t["x"] + m["vx"] * h,
                                        t["y"] + m["vy"] * h, t.get("pixeles", 0), "evolucion"))
    errores = defaultdict(list)
    errores_grandes = defaultdict(list)
    tiempos = sorted({st.parse_iso(k[1]) for k in posiciones})
    for tid, objetivo, h, x, y, px, fuente in pronosticos:
        cerca = [t for t in tiempos if abs((t - objetivo).total_seconds()) <= 300]
        if not cerca:
            continue
        real = posiciones.get((tid, st.iso(min(cerca, key=lambda t: abs((t - objetivo).total_seconds())))))
        if real is None:
            continue
        e = math.hypot(real[0] - x, real[1] - y) * KM_PX
        if fuente == "evolucion":
            errores[f"evolucion_{h}"].append(e)
            continue
        errores[h].append(e)
        if px >= 150:
            errores_grandes[h].append(e)
    return tramos, errores, errores_grandes, dict(vistos)


def resumen(tramos, errores, errores_grandes, vistos):
    def med(vals):
        return round(statistics.median(vals), 1) if vals else None

    def p90(vals):
        return round(sorted(vals)[int(0.9 * (len(vals) - 1))], 1) if vals else None

    v = [t["kmh"] for t in tramos]
    vc = [t["kmh_centro"] for t in tramos]
    vidas = list(vistos.values())
    return {
        "tramos": len(tramos),
        "velocidad_kmh_mediana": med(v),
        "velocidad_kmh_p90": p90(v),
        "velocidad_kmh_max": round(max(v), 1) if v else None,
        "mas_de_80_pct": round(100 * sum(x > 80 for x in v) / len(v), 1) if v else None,
        "mas_de_100_pct": round(100 * sum(x > 100 for x in v) / len(v), 1) if v else None,
        "centro_mas_de_80_pct": round(100 * sum(x > 80 for x in vc) / len(vc), 1) if vc else None,
        "dudosos_pct": round(100 * sum(t["dudoso"] for t in tramos) / len(tramos), 1) if tramos else None,
        "tormentas": len(vidas),
        "vida_media_frames": round(statistics.mean(vidas), 2) if vidas else None,
        "tormentas_3_o_mas_frames": sum(x >= 3 for x in vidas),
        "error_posicion_km": {
            str(h): {"mediana": med(errores[h]), "p90": p90(errores[h]), "casos": len(errores[h])}
            for h in HORIZONTES
        },
        "error_posicion_km_prediccion_evolucion": {
            str(h): {"mediana": med(errores[f"evolucion_{h}"]), "p90": p90(errores[f"evolucion_{h}"]),
                     "casos": len(errores[f"evolucion_{h}"])}
            for h in HORIZONTES
        },
        "error_posicion_km_tormentas_grandes": {
            str(h): {"mediana": med(errores_grandes[h]), "p90": p90(errores_grandes[h]),
                     "casos": len(errores_grandes[h])}
            for h in HORIZONTES
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dias", default=DIAS_DEFECTO)
    ap.add_argument("--salida")
    args = ap.parse_args(argv)
    from scripts.procesador_radar import (analizar_mascara, aplicar_mascara_estatica, calcular_confianza,
                                          cargar_mascara, detectar_nucleos)
    detectores = (detectar_nucleos, cargar_mascara, analizar_mascara, calcular_confianza, aplicar_mascara_estatica)
    tramos, errores, grandes, vistos = [], defaultdict(list), defaultdict(list), {}
    for dia in args.dias.split(","):
        dia = dia.strip()
        tr, er, eg, vi = correr_dia(dia, detectores)
        tramos += tr
        for h, vals in er.items():
            errores[h] += vals
        for h in HORIZONTES:
            grandes[h] += eg[h]
        vistos.update({f"{dia}-{k}": n for k, n in vi.items()})
        r = resumen(tr, er, eg, vi)
        print(f"{dia}: {r['tramos']} tramos, >80 km/h {r['mas_de_80_pct']} %, max {r['velocidad_kmh_max']}", flush=True)
    res = resumen(tramos, errores, grandes, vistos)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.salida:
        Path(args.salida).write_text(json.dumps({"resumen": res, "tramos": tramos}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
