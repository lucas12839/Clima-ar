#!/usr/bin/env python3
"""
ClimaAR - Predicción de evolución y relleno de huecos radar.

Implementa:
- extrapolación de movimiento por track;
- predicciones a 5/10/20/30 min;
- detección de huecos temporales;
- puntos intermedios estimados dentro de un hueco;
- persistencia de predicciones pendientes;
- validación automática contra el siguiente radar real;
- confianza e incertidumbre crecientes con el horizonte.

No presenta una estimación como radar real.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

INPUT = Path("data/radar/seguimiento/seguimiento_temporal.json")
OUTPUT = Path("data/radar/seguimiento/prediccion_evolucion.json")
PENDING = Path("data/radar/seguimiento/predicciones_pendientes.json")

HORIZONTES_MIN = (5, 10, 20, 30)
PASO_HUECO_MIN = 5
INTERVALO_ESPERADO_MIN = 10
TOLERANCIA_HUECO_MIN = 12
MAX_PENDIENTES = 500


def clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, float(v)))


def parse_ts(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def sector(deg):
    if deg is None:
        return None
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
    return names[int((deg % 360) / 22.5 + 0.5) % 16]


def angle_diff(a, b):
    if a is None or b is None:
        return None
    return abs((a - b + 180) % 360 - 180)


def recent(track, n=6):
    h = [
        x for x in track.get("historial", [])
        if x.get("timestamp") and x.get("x") is not None and x.get("y") is not None
    ]
    h.sort(key=lambda x: x["timestamp"])
    return h[-n:]


def movements(hist):
    out = []
    for a, b in zip(hist, hist[1:]):
        ta, tb = parse_ts(a["timestamp"]), parse_ts(b["timestamp"])
        if not ta or not tb:
            continue
        dt = (tb - ta).total_seconds() / 60
        if not 2 <= dt <= 60:
            continue

        dx = float(b["x"]) - float(a["x"])
        dy = float(b["y"]) - float(a["y"])
        speed = math.hypot(dx, dy) / dt
        bearing = math.degrees(math.atan2(dx, -dy)) % 360 if speed > 0.001 else None

        out.append({
            "dt": dt,
            "vx": dx / dt,
            "vy": dy / dt,
            "speed": speed,
            "bearing": bearing,
            "from": a["timestamp"],
            "to": b["timestamp"],
        })
    return out


def estimate_motion(track):
    hist = recent(track)
    mv = movements(hist)
    if not hist or not mv:
        return None

    # Más peso a los movimientos recientes, sin descartar los anteriores.
    weights = list(range(1, len(mv) + 1))
    tw = sum(weights)
    vx = sum(m["vx"] * w for m, w in zip(mv, weights)) / tw
    vy = sum(m["vy"] * w for m, w in zip(mv, weights)) / tw
    speed = math.hypot(vx, vy)
    bearing = math.degrees(math.atan2(vx, -vy)) % 360 if speed > 0.001 else None

    acc = track.get("aceleracion_pixeles_min2")
    if acc is None and len(mv) >= 2:
        # Aceleración de velocidad entre los dos últimos desplazamientos.
        a = mv[-2]
        b = mv[-1]
        acc = (b["speed"] - a["speed"]) / max((a["dt"] + b["dt"]) / 2, 1)

    turns = [
        angle_diff(mv[i]["bearing"], mv[i - 1]["bearing"])
        for i in range(1, len(mv))
    ]
    turns = [x for x in turns if x is not None]
    consistency = clamp(100 - (mean(turns) * 0.25 if turns else 0))

    movement_conf = float(track.get("confianza_movimiento") or 50)
    confidence = clamp(
        0.60 * movement_conf +
        0.25 * consistency +
        15 * min(len(mv), 3) / 3
    )

    area_changes = [
        float(x["cambio_area_porcentaje"])
        for x in hist[1:]
        if x.get("cambio_area_porcentaje") is not None
    ]
    area_trend = mean(area_changes[-4:]) if area_changes else None

    # Índice heurístico de intensificación/debilitamiento.
    intens = 50.0
    if area_trend is not None:
        intens += max(-25, min(25, area_trend * 0.8))
    if acc is not None:
        intens += max(-15, min(15, float(acc) * 300))
    intens = clamp(intens)

    return {
        "vx": vx,
        "vy": vy,
        "speed": speed,
        "bearing": bearing,
        "acc": acc,
        "confidence": confidence,
        "area_trend": area_trend,
        "intensification": intens,
        "weakening": 100 - intens,
        "history_points": len(hist),
        "last": hist[-1],
    }


def project(motion, x0, y0, horizon):
    vx, vy = motion["vx"], motion["vy"]
    speed = motion["speed"]
    acc = motion["acc"]

    # Aplicamos aceleración sólo si es pequeña/razonable. Evita explosiones
    # cuando hay un salto temporal o ruido de seguimiento.
    vxh, vyh = vx, vy
    if acc is not None and abs(float(acc)) < 0.15 and speed > 0.01:
        ux, uy = vx / speed, vy / speed
        delta_v = float(acc) * horizon
        delta_v = max(-speed * 0.5, min(speed * 0.5, delta_v))
        vxh += ux * delta_v
        vyh += uy * delta_v

    xf = float(x0) + vxh * horizon
    yf = float(y0) + vyh * horizon
    fs = math.hypot(vxh, vyh)
    bearing = math.degrees(math.atan2(vxh, -vyh)) % 360 if fs > 0.001 else None

    # La incertidumbre crece con el horizonte y baja con la confianza del track.
    base_error = max(2.0, (100 - motion["confidence"]) * 0.12)
    uncertainty = base_error + horizon * max(0.10, 0.35 - motion["confidence"] / 400)

    return {
        "x_estimado": round(xf, 2),
        "y_estimado": round(yf, 2),
        "velocidad_estimada_pixeles_min": round(fs, 4),
        "rumbo_estimado_grados": round(bearing, 1) if bearing is not None else None,
        "direccion_estimada": sector(bearing),
        "incertidumbre_pixeles": round(uncertainty, 2),
    }


def load_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def validate_pending(current_ts, current_tracks, pending):
    """
    Valida una predicción SOLO contra una observación radar real con el
    mismo timestamp objetivo. Nunca usa la posición actual si el objetivo
    pertenece a un frame anterior, porque eso produciría un error falso.
    """
    observations = {}
    for track in current_tracks:
        tid = track.get("id")
        if tid is None:
            continue
        for obs in track.get("historial", []):
            ts = obs.get("timestamp")
            if ts and obs.get("x") is not None and obs.get("y") is not None:
                observations[(str(tid), str(ts))] = obs

    validated = []
    remaining = []
    now = parse_ts(current_ts)

    for p in pending:
        target_text = p.get("timestamp_objetivo")
        target = parse_ts(target_text)

        if not target or not now or target > now:
            remaining.append(p)
            continue

        tid = str(p.get("track_id"))
        actual = observations.get((tid, str(target_text)))

        # Si todavía no tenemos el frame real exacto, NO validamos.
        # Puede aparecer en el historial cuando llegue el próximo ciclo.
        if actual is None:
            remaining.append(p)
            continue

        dx = float(actual["x"]) - float(p["x_estimado"])
        dy = float(actual["y"]) - float(p["y_estimado"])
        error = math.hypot(dx, dy)

        p["validacion"] = {
            "estado": "VALIDADA",
            "timestamp_real": str(actual["timestamp"]),
            "x_real": round(float(actual["x"]), 2),
            "y_real": round(float(actual["y"]), 2),
            "error_pixeles": round(error, 2),
            "error_x_pixeles": round(dx, 2),
            "error_y_pixeles": round(dy, 2),
        }
        validated.append(p)

    return remaining, validated


def make_gap_predictions(track, motion, previous_ts, current_ts):
    """
    Si entre dos observaciones reales hay un hueco mayor a la tolerancia,
    crea puntos intermedios cada PASO_HUECO_MIN.
    """
    ta, tb = parse_ts(previous_ts), parse_ts(current_ts)
    if not ta or not tb:
        return []

    gap = (tb - ta).total_seconds() / 60
    if gap <= TOLERANCIA_HUECO_MIN:
        return []

    last = motion["last"]
    x0, y0 = float(last["x"]), float(last["y"])
    result = []

    t = PASO_HUECO_MIN
    while t < gap:
        proj = project(motion, x0, y0, t)
        target = ta.timestamp() + t * 60
        target_dt = datetime.fromtimestamp(target, tz=timezone.utc)

        result.append({
            "track_id": track.get("id"),
            "timestamp_origen": previous_ts,
            "timestamp_objetivo": iso(target_dt),
            "horizonte_desde_origen_minutos": t,
            "tipo": "ESTIMACION_HUECO",
            **proj,
            "confianza_prediccion": round(
                clamp(motion["confidence"] - max(0, t - 5) * 1.2), 1
            ),
        })
        t += PASO_HUECO_MIN

    # El extremo del hueco coincide con un radar REAL. Este punto sí puede
    # validarse inmediatamente contra la observación que cerró el hueco.
    proj = project(motion, x0, y0, gap)
    result.append({
        "track_id": track.get("id"),
        "timestamp_origen": previous_ts,
        "timestamp_objetivo": current_ts,
        "horizonte_desde_origen_minutos": round(gap, 2),
        "tipo": "PREDICCION_CIERRE_HUECO",
        **proj,
        "confianza_prediccion": round(
            clamp(motion["confidence"] - max(0, gap - 5) * 1.2), 1
        ),
    })

    return result


def predict_track(track, radar_ts):
    motion = estimate_motion(track)
    if not motion:
        return {
            "track_id": track.get("id"),
            "pronostico_disponible": False,
            "motivo": "SIN_MOVIMIENTO_VALIDO"
        }, []

    cur = motion["last"]
    forecasts = []
    for horizon in HORIZONTES_MIN:
        forecasts.append({
            "horizonte_minutos": horizon,
            **project(motion, cur["x"], cur["y"], horizon),
            "confianza_prediccion": round(
                clamp(motion["confidence"] - max(0, horizon - 5) * 1.2), 1
            ),
        })

    # Sólo detectar un hueco ya ocurrido dentro del historial.
    gap_predictions = []
    hist = recent(track, 8)
    for a, b in zip(hist, hist[1:]):
        ta, tb = parse_ts(a["timestamp"]), parse_ts(b["timestamp"])
        if not ta or not tb:
            continue
        gap = (tb - ta).total_seconds() / 60
        if gap > TOLERANCIA_HUECO_MIN:
            # Usamos el movimiento disponible antes del hueco. Si el track
            # sólo conserva la observación posterior, no inventamos.
            before = {
                **track,
                "historial": [h for h in hist if parse_ts(h["timestamp"]) <= ta]
            }
            before_motion = estimate_motion(before)
            if before_motion:
                gap_points = make_gap_predictions(
                    track, before_motion, a["timestamp"], b["timestamp"]
                )
                gap_predictions.extend(gap_points)

                # Validación del punto que cierra el hueco: b es radar REAL.
                cierre = next(
                    (x for x in gap_points if x["tipo"] == "PREDICCION_CIERRE_HUECO"),
                    None
                )
                if cierre:
                    error = math.hypot(
                        float(b["x"]) - float(cierre["x_estimado"]),
                        float(b["y"]) - float(cierre["y_estimado"])
                    )
                    cierre["validacion"] = {
                        "estado": "VALIDADA",
                        "x_real": round(float(b["x"]), 2),
                        "y_real": round(float(b["y"]), 2),
                        "error_pixeles": round(error, 2),
                        "error_x_pixeles": round(float(b["x"]) - float(cierre["x_estimado"]), 2),
                        "error_y_pixeles": round(float(b["y"]) - float(cierre["y_estimado"]), 2),
                    }

    return {
        "track_id": track.get("id"),
        "pronostico_disponible": True,
        "posicion_actual": {
            "x": round(float(cur["x"]), 2),
            "y": round(float(cur["y"]), 2),
        },
        "velocidad_actual_pixeles_min": round(motion["speed"], 4),
        "rumbo_actual_grados": round(motion["bearing"], 1) if motion["bearing"] is not None else None,
        "direccion_actual": sector(motion["bearing"]),
        "aceleracion_pixeles_min2": round(float(motion["acc"]), 5) if motion["acc"] is not None else None,
        "tendencia_area_porcentaje": round(motion["area_trend"], 2) if motion["area_trend"] is not None else None,
        "probabilidad_intensificacion": round(motion["intensification"], 1),
        "probabilidad_debilitamiento": round(motion["weakening"], 1),
        "confianza_prediccion": round(motion["confidence"], 1),
        "horizontes": forecasts,
        "huecos_detectados": len(gap_predictions),
        "estimaciones_hueco": gap_predictions,
    }, gap_predictions


def main():
    data = load_json(INPUT, {})
    radar_ts = data.get("radar_timestamp")
    tracks = data.get("tracks", [])
    intervalo = float(data.get("intervalo_minutos") or 0)

    pending = load_json(PENDING, {})
    pending_items = pending.get("pendientes", []) if isinstance(pending, dict) else []

    pending_items, validated_now = validate_pending(
        radar_ts,
        tracks,
        pending_items
    )

    predictions = []
    new_pending = []
    gap_count = 0

    existing_pending_keys = {
        (str(x.get("track_id")), x.get("timestamp_objetivo"))
        for x in pending_items
    }

    for track in tracks:
        if not track.get("activo", True):
            continue

        prediction, gap_preds = predict_track(track, radar_ts)
        predictions.append(prediction)

        for gp in gap_preds:
            gap_count += 1
            # El cierre del hueco ya tiene un radar REAL para validarlo;
            # no lo dejamos como pendiente.
            if gp.get("validacion", {}).get("estado") == "VALIDADA":
                continue

            key = (str(gp["track_id"]), gp["timestamp_objetivo"])
            if key not in existing_pending_keys:
                new_pending.append(gp)
                existing_pending_keys.add(key)

    pending_items.extend(new_pending)
    pending_items = pending_items[-MAX_PENDIENTES:]

    resumen = {
        "nucleos_con_prediccion": sum(
            1 for p in predictions if p.get("pronostico_disponible")
        ),
        "nucleos_sin_prediccion": sum(
            1 for p in predictions if not p.get("pronostico_disponible")
        ),
        "promedio_intensificacion": None,
        "promedio_debilitamiento": None,
        "promedio_confianza": None,
        "huecos_estimados": gap_count,
        "predicciones_validadas_en_este_frame": len(validated_now),
        "error_medio_pixeles_validado": None,
    }

    ok = [p for p in predictions if p.get("pronostico_disponible")]
    if ok:
        resumen["promedio_intensificacion"] = round(
            mean(p["probabilidad_intensificacion"] for p in ok), 1
        )
        resumen["promedio_debilitamiento"] = round(
            mean(p["probabilidad_debilitamiento"] for p in ok), 1
        )
        resumen["promedio_confianza"] = round(
            mean(p["confianza_prediccion"] for p in ok), 1
        )

    errors = [
        float(p["validacion"]["error_pixeles"])
        for p in validated_now
        if p.get("validacion", {}).get("estado") == "VALIDADA"
        and p.get("validacion", {}).get("error_pixeles") is not None
    ]
    for pred in predictions:
        for gp in pred.get("estimaciones_hueco", []):
            if gp.get("validacion", {}).get("estado") == "VALIDADA":
                errors.append(float(gp["validacion"]["error_pixeles"]))
    if errors:
        resumen["error_medio_pixeles_validado"] = round(mean(errors), 2)

    result = {
        "schema_version": 2,
        "radar_timestamp": radar_ts,
        "intervalo_minutos": intervalo,
        "tipo": "PREDICCION_MOVIMIENTO_CON_HUECOS",
        "prediccion_global_disponible": bool(tracks),
        "nucleos": predictions,
        "resumen": resumen,
        "validaciones": validated_now[-100:],
        "nota": (
            "Las posiciones estimadas son interpolaciones/extrapolaciones del "
            "seguimiento temporal y nunca se etiquetan como radar observado."
        ),
    }

    save_json(OUTPUT, result)
    save_json(
        PENDING,
        {
            "schema_version": 1,
            "actualizado": radar_ts,
            "pendientes": pending_items,
        },
    )

    print(f"OK: {OUTPUT}")
    print(f"Predicciones: {len(predictions)}")
    print(f"Huecos estimados: {gap_count}")
    print(f"Predicciones validadas: {len(validated_now)}")


if __name__ == "__main__":
    main()
