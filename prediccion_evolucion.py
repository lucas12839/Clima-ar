#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean, pstdev
import json
import math

STATE = Path("data/radar/seguimiento/estado.json")
OUTPUT = Path("data/radar/seguimiento/prediccion_evolucion.json")

HORIZONTES = (10, 20, 30)
MIN_DT = 2.0
MAX_DT = 30.0

SECTORES = [
    "N","NNE","NE","ENE","E","ESE","SE","SSE",
    "S","SSO","SO","OSO","O","ONO","NO","NNO"
]


def num(v):
    try:
        return float(v)
    except Exception:
        return None


def dt(v):
    if not isinstance(v, str):
        return None
    try:
        d = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except Exception:
        return None


def clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, float(v)))


def sector(grados):
    if grados is None:
        return None
    return SECTORES[int((grados + 11.25) / 22.5) % 16]


def bearing(dx, dy):
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    return math.degrees(math.atan2(dx, -dy)) % 360


def clean_history(track, radar_timestamp):
    limite = dt(radar_timestamp)
    out = {}

    for h in track.get("historial", []):
        if not isinstance(h, dict):
            continue
        fecha = dt(h.get("timestamp"))
        if not fecha or h.get("x") is None or h.get("y") is None:
            continue
        if limite and fecha > limite:
            continue
        item = dict(h)
        item["timestamp"] = fecha.isoformat().replace("+00:00", "Z")
        out[item["timestamp"]] = item

    return sorted(out.values(), key=lambda x: dt(x["timestamp"]))


def intensity(h):
    mapa = {
        "MUY_DEBIL": 1, "DEBIL": 2, "MODERADA": 3,
        "FUERTE": 4, "MUY_FUERTE": 5, "SEVERA": 6
    }
    v = h.get("nivel_intensidad")
    if isinstance(v, str):
        return mapa.get(v)
    return num(v)


def movements(hist):
    out = []
    for a, b in zip(hist, hist[1:]):
        t1, t2 = dt(a["timestamp"]), dt(b["timestamp"])
        if not t1 or not t2:
            continue
        minutos = (t2 - t1).total_seconds() / 60
        if minutos < MIN_DT or minutos > MAX_DT:
            continue

        dx = float(b["x"]) - float(a["x"])
        dy = float(b["y"]) - float(a["y"])
        vx, vy = dx / minutos, dy / minutos
        speed = math.hypot(vx, vy)

        out.append({
            "dt": minutos, "dx": dx, "dy": dy,
            "vx": vx, "vy": vy, "speed": speed,
            "bearing": bearing(dx, dy)
        })
    return out


def evolution(hist):
    areas = [num(h.get("pixeles")) for h in hist]
    areas = [x for x in areas if x and x > 0]

    cambios = []
    for a, b in zip(areas, areas[1:]):
        cambios.append((b - a) / a * 100)

    tendencia_area = mean(cambios[-3:]) if cambios else 0.0

    intensidades = [intensity(h) for h in hist]
    intensidades = [x for x in intensidades if x is not None]
    tendencia_int = (
        intensidades[-1] - intensidades[0]
        if len(intensidades) >= 2 else 0
    )

    score = 50 + clamp(tendencia_area, -25, 25) * 0.9
    score += max(-18, min(18, tendencia_int * 9))

    return clamp(score), clamp(100 - score), tendencia_area, tendencia_int


def predict(track, radar_timestamp):
    hist = clean_history(track, radar_timestamp)
    if not hist:
        return None

    moves = movements(hist[-6:])
    actual = hist[-1]
    intens, weaken, area_trend, int_trend = evolution(hist[-6:])

    result = {
        "track_id": track.get("id"),
        "posicion_actual": {
            "x": round(float(actual["x"]), 2),
            "y": round(float(actual["y"]), 2)
        },
        "frames_utilizados": len(hist),
        "intervalos_movimiento_validos": len(moves),
        "probabilidad_intensificacion": round(intens, 1),
        "probabilidad_debilitamiento": round(weaken, 1),
        "pronostico_disponible": False,
        "horizontes": []
    }

    if not moves:
        result["motivo"] = "SIN_MOVIMIENTO_VALIDO"
        result["confianza_prediccion"] = 0.0
        return result

    pesos = range(1, len(moves) + 1)
    total = sum(pesos)

    vx = sum(m["vx"] * p for m, p in zip(moves, pesos)) / total
    vy = sum(m["vy"] * p for m, p in zip(moves, pesos)) / total
    velocidad = math.hypot(vx, vy)
    rumbo = bearing(vx, vy)

    velocidades = [m["speed"] for m in moves]
    variacion = pstdev(velocidades) if len(velocidades) > 1 else 0
    confianza_mov = num(track.get("confianza_movimiento")) or 50

    consistencia = clamp(
        100 - (variacion / max(velocidad, 0.01)) * 20
    )

    confianza_datos = 35 if len(moves) >= 3 else 22 if len(moves) == 2 else 8
    confianza = clamp(
        0.45 * confianza_mov + 0.30 * consistencia + confianza_datos
    )

    aceleracion = None
    if len(moves) >= 2:
        aceleracion = (
            moves[-1]["speed"] - moves[-2]["speed"]
        ) / max(moves[-1]["dt"], 1)

    x0, y0 = float(actual["x"]), float(actual["y"])

    for horizonte in HORIZONTES:
        vx2, vy2 = vx, vy

        if aceleracion is not None and velocidad > 0.01:
            ux, uy = vx / velocidad, vy / velocidad
            acc = max(-0.15, min(0.15, aceleracion))
            vx2 = vx + ux * acc * horizonte
            vy2 = vy + uy * acc * horizonte

        x = x0 + vx2 * horizonte
        y = y0 + vy2 * horizonte
        speed2 = math.hypot(vx2, vy2)
        rumbo2 = bearing(vx2, vy2)

        incertidumbre = max(
            2.0,
            horizonte * max(0.05, velocidad * 0.35)
        )

        result["horizontes"].append({
            "horizonte_minutos": horizonte,
            "x_estimado": round(x, 2),
            "y_estimado": round(y, 2),
            "velocidad_estimada_pixeles_min": round(speed2, 4),
            "rumbo_estimado_grados": round(rumbo2, 1) if rumbo2 is not None else None,
            "direccion_estimada": sector(rumbo2),
            "incertidumbre_pixeles": round(incertidumbre, 1)
        })

    result.update({
        "pronostico_disponible": len(moves) >= 2,
        "velocidad_actual_pixeles_min": round(velocidad, 4),
        "rumbo_actual_grados": round(rumbo, 1) if rumbo is not None else None,
        "direccion_actual": sector(rumbo),
        "aceleracion_pixeles_min2": round(aceleracion, 5) if aceleracion is not None else None,
        "tendencia_area_porcentaje": round(area_trend, 2),
        "tendencia_intensidad_niveles": int_trend,
        "consistencia_movimiento": round(consistencia, 1),
        "confianza_prediccion": round(confianza if len(moves) >= 2 else min(confianza, 49), 1),
        "motivo": None if len(moves) >= 2 else "SOLO_UN_INTERVALO_VALIDO"
    })
    return result


def main():
    if not STATE.exists():
        raise FileNotFoundError(STATE)

    estado = json.loads(STATE.read_text(encoding="utf-8"))
    radar_timestamp = estado.get("ultimo_radar_timestamp")

    # SOLO tracks activos y NO clasificados como eco estático.
    tracks = []
    for track in estado.get("tracks", []):
        if not track.get("activo", False):
            continue
        estado_actual = str(track.get("estado", "")).upper()
        if "ESTATICO" in estado_actual:
            continue

        tracks.append(track)

    resultado = {
        "version": 3,
        "radar_timestamp": radar_timestamp,
        "tipo": "PREDICCION_HEURISTICA_MULTIFRAME",
        "nucleos": [],
        "resumen": {}
    }

    intervalo = None
    try:
        intervalo = (
            estado.get("intervalo_minutos")
            or estado.get("ultimo_intervalo_minutos")
        )
    except Exception:
        pass

    if intervalo is not None:
        intervalo = num(intervalo)

    intervalo_valido = intervalo is not None and MIN_DT <= intervalo <= MAX_DT

    if intervalo_valido:
        for track in tracks:
            p = predict(track, radar_timestamp)
            if p:
                resultado["nucleos"].append(p)
    else:
        resultado["motivo"] = "INTERVALO_TEMPORAL_NO_VALIDO"

    disponibles = [
        p for p in resultado["nucleos"]
        if p.get("pronostico_disponible")
    ]

    resultado["resumen"] = {
        "nucleos_activos_no_estaticos": len(tracks),
        "nucleos_evaluados": len(resultado["nucleos"]),
        "nucleos_con_prediccion": len(disponibles),
        "nucleos_sin_prediccion": len(resultado["nucleos"]) - len(disponibles),
        "promedio_intensificacion": round(
            mean(p["probabilidad_intensificacion"] for p in disponibles), 1
        ) if disponibles else None,
        "promedio_debilitamiento": round(
            mean(p["probabilidad_debilitamiento"] for p in disponibles), 1
        ) if disponibles else None,
        "promedio_confianza": round(
            mean(p["confianza_prediccion"] for p in disponibles), 1
        ) if disponibles else None
    }

    if disponibles:
        xs, ys = [], []
        for p in disponibles:
            v = p.get("velocidad_actual_pixeles_min") or 0
            r = p.get("rumbo_actual_grados")
            if r is None:
                continue
            rad = math.radians(r)
            xs.append(v * math.sin(rad))
            ys.append(-v * math.cos(rad))

        if xs:
            vx, vy = mean(xs), mean(ys)
            vs = math.hypot(vx, vy)
            rs = bearing(vx, vy)
            resultado["movimiento_sistema"] = {
                "velocidad_media_vectorial_pixeles_min": round(vs, 4),
                "rumbo_grados": round(rs, 1) if rs is not None else None,
                "direccion": sector(rs),
                "nucleos_utilizados": len(xs)
            }

    resultado["limitacion"] = (
        "Predicción heurística. Velocidad y posición permanecen en píxeles "
        "hasta georreferenciar RMA10."
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("OK:", OUTPUT)
    print("Núcleos evaluados:", len(resultado["nucleos"]))
    print("Con predicción:", len(disponibles))


if __name__ == "__main__":
    main()
