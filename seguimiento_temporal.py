#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
import json
import math
import re

RADAR_DIR = Path("data/radar")
RESULTADO_FILE = RADAR_DIR / "resultado_procesamiento.json"
SEGUIMIENTO_DIR = RADAR_DIR / "seguimiento"
ESTADO_FILE = SEGUIMIENTO_DIR / "estado.json"
RESULTADO_SEGUIMIENTO = SEGUIMIENTO_DIR / "seguimiento_temporal.json"
HISTORICO_DIR = RADAR_DIR / "historico"
RADAR_TIMESTAMP_FILE = RADAR_DIR / "radar_timestamp.txt"

DISTANCIA_BASE = 25.0
DISTANCIA_MAXIMA = 90.0
CAMBIO_AREA_CRECE = 15.0
CAMBIO_AREA_DEBILITA = -15.0
MOVIMIENTO_MINIMO = 5.0
MAX_HISTORIAL = 12
MAX_MISSED_FRAMES = 2

# Nunca calcular velocidad/aceleracion usando un estado demasiado viejo.
# El radar normalmente entrega frames de ~10 min; 30 min permite un salto
# ocasional sin convertirlo en una falsa velocidad.
MAX_INTERVALO_MOVIMIENTO_MIN = 30.0
MIN_INTERVALO_MOVIMIENTO_MIN = 2.0


def ahora_utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def cargar_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"No se pudo leer {path}: {e}")
        return None


def guardar_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _parsear_timestamp_nombre(nombre):
    m = re.search(r"(\d{14})", nombre)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    m = re.search(r"(?:^|_)(\d{10})(?:\D|$)", nombre)
    if m:
        try:
            return datetime.fromtimestamp(int(m.group(1)), timezone.utc)
        except (ValueError, OSError, OverflowError):
            pass

    return None


def obtener_timestamp_radar():
    """Timestamp del frame descargado en ESTA ejecucion."""
    if RADAR_TIMESTAMP_FILE.exists():
        try:
            texto = RADAR_TIMESTAMP_FILE.read_text(encoding="utf-8").strip()
            dt = datetime.fromisoformat(texto.replace("Z", "+00:00"))
            return dt.astimezone(timezone.utc)
        except Exception:
            pass

    candidatos = []
    if HISTORICO_DIR.exists():
        for p in HISTORICO_DIR.glob("*.png"):
            try:
                if p.stat().st_size < 1000:
                    continue
            except OSError:
                continue
            dt = _parsear_timestamp_nombre(p.stem)
            if dt:
                candidatos.append((dt, p))

    return max(candidatos, key=lambda x: x[0])[0] if candidatos else None


def obtener_nucleos(resultado):
    if not isinstance(resultado, dict):
        return []
    for key in ("nucleos", "núcleos", "cells", "ecos", "detected_cells"):
        valor = resultado.get(key)
        if isinstance(valor, list):
            return valor
    return []


def normalizar_nucleo(nucleo):
    if not isinstance(nucleo, dict):
        return None
    try:
        return {
            "x": float(nucleo.get("centro_x", nucleo.get("x", 0))),
            "y": float(nucleo.get("centro_y", nucleo.get("y", 0))),
            "pixeles": max(1, int(nucleo.get("pixeles", nucleo.get("area", 0)))),
            "ancho": max(1.0, float(nucleo.get("ancho", nucleo.get("width", 1)))),
            "alto": max(1.0, float(nucleo.get("alto", nucleo.get("height", 1)))),
            "intensidad": nucleo.get("intensidad_maxima", nucleo.get("intensity", "DESCONOCIDA")),
            "nivel_intensidad": nucleo.get("nivel_intensidad", "DESCONOCIDO"),
            "estado": nucleo.get("estado", "DESCONOCIDO"),
            "confianza": float(nucleo.get("confianza", 0)),
        }
    except Exception:
        return None


def distancia(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def iou(a, b):
    al, ar = a["x"] - a["ancho"] / 2, a["x"] + a["ancho"] / 2
    at, ab = a["y"] - a["alto"] / 2, a["y"] + a["alto"] / 2
    bl, br = b["x"] - b["ancho"] / 2, b["x"] + b["ancho"] / 2
    bt, bb = b["y"] - b["alto"] / 2, b["y"] + b["alto"] / 2
    inter = max(0.0, min(ar, br) - max(al, bl)) * max(0.0, min(ab, bb) - max(at, bt))
    union = a["ancho"] * a["alto"] + b["ancho"] * b["alto"] - inter
    return inter / union if union > 0 else 0.0


def umbral_distancia(minutos):
    return min(DISTANCIA_MAXIMA, max(DISTANCIA_BASE, DISTANCIA_BASE + minutos * 5.0))


def compatibilidad(actual, anterior, minutos):
    dist = distancia(actual, anterior)
    limite = umbral_distancia(minutos)
    if dist > limite:
        return None

    area_a = max(1.0, actual["pixeles"])
    area_b = max(1.0, anterior["pixeles"])
    ratio_area = max(area_a, area_b) / min(area_a, area_b)
    solapamiento = iou(actual, anterior)
    ratio_ancho = max(actual["ancho"], anterior["ancho"]) / max(1.0, min(actual["ancho"], anterior["ancho"]))
    ratio_alto = max(actual["alto"], anterior["alto"]) / max(1.0, min(actual["alto"], anterior["alto"]))

    if ratio_area > 2.5 and solapamiento < 0.15:
        return None
    if ratio_ancho > 2.8 and ratio_alto > 2.0:
        return None
    if ratio_ancho > 3.5 or ratio_alto > 3.5:
        return None

    if dist > max(35.0, limite * 0.65):
        if solapamiento < 0.08 or ratio_area > 2.0:
            return None

    cercania = max(0.0, 1.0 - dist / limite)
    similitud_area = max(0.0, 1.0 - abs(math.log(area_a / area_b)) / math.log(2.5))
    similitud_forma = max(0.0, 1.0 - (abs(math.log(ratio_ancho)) + abs(math.log(ratio_alto))) / math.log(3.0))
    puntaje = 35.0 * cercania + 35.0 * solapamiento + 20.0 * similitud_area + 10.0 * similitud_forma

    if solapamiento < 0.02:
        if dist > 20.0 or ratio_area > 1.5 or ratio_ancho > 1.6 or ratio_alto > 1.6 or puntaje < 55.0:
            return None

    return {
        "puntaje": puntaje,
        "distancia": dist,
        "iou": solapamiento,
        "cambio_area": (area_a - area_b) / area_b * 100.0,
        "ratio_area": ratio_area,
        "ratio_ancho": ratio_ancho,
        "ratio_alto": ratio_alto,
        "similitud_forma": similitud_forma,
    }


def siguiente_id(tracks):
    ids = []
    for t in tracks:
        try:
            ids.append(int(t.get("id", 0)))
        except Exception:
            pass
    return max(ids, default=0) + 1


def timestamp_iso(dt):
    return dt.isoformat().replace("+00:00", "Z") if dt else None


def crear_track(track_id, actual, timestamp):
    return {
        "id": track_id,
        "primer_frame": timestamp,
        "ultimo_frame": timestamp,
        "frames": 1,
        "missed_frames": 0,
        "activo": True,
        "x": actual["x"],
        "y": actual["y"],
        "pixeles": actual["pixeles"],
        "intensidad": actual["intensidad"],
        "nivel_intensidad": actual["nivel_intensidad"],
        "estado": "NUEVO",
        "historial": [{
            "timestamp": timestamp,
            "x": actual["x"],
            "y": actual["y"],
            "pixeles": actual["pixeles"],
            "intensidad": actual["intensidad"],
            "estado": "NUEVO",
        }],
    }


def rumbo_meteorologico(dx, dy):
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None, "SIN_MOVIMIENTO"
    # Supuesto del radar: norte arriba, +X este y +Y hacia el sur.
    rumbo = math.degrees(math.atan2(dx, -dy)) % 360.0
    sectores = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
    return rumbo, sectores[int((rumbo + 11.25) // 22.5) % 16]


def calcular_confianza_movimiento(match, desplazamiento, minutos):
    score = float(match.get("puntaje", 0.0))
    iou_val = float(match.get("iou", 0.0))
    dist = float(match.get("distancia", 0.0))
    confianza = 45.0 + score * 0.45
    if iou_val >= 0.20:
        confianza += 8.0
    elif iou_val <= 0.02:
        confianza -= 5.0
    limite = umbral_distancia(minutos)
    if limite > 0:
        confianza -= max(0.0, (dist / limite) * 8.0)
    if desplazamiento < MOVIMIENTO_MINIMO:
        confianza -= 5.0
    return round(max(0.0, min(99.0, confianza)), 1)


def actualizar_track(track, actual, anterior, match, minutos, timestamp):
    dx = actual["x"] - anterior["x"]
    dy = actual["y"] - anterior["y"]
    desplazamiento = math.hypot(dx, dy)
    velocidad = desplazamiento / max(minutos, 0.1)
    cambio_area = match["cambio_area"]

    if cambio_area >= CAMBIO_AREA_CRECE:
        evolucion = "CRECIENDO"
    elif cambio_area <= CAMBIO_AREA_DEBILITA:
        evolucion = "DEBILITANDOSE"
    else:
        evolucion = "ESTABLE"

    rumbo, direccion = rumbo_meteorologico(dx, dy)
    historial = track.setdefault("historial", [])
    rumbo_anterior = next((float(item["rumbo_grados"]) for item in reversed(historial) if isinstance(item.get("rumbo_grados"), (int, float))), None)
    cambio_rumbo = None
    if rumbo is not None and rumbo_anterior is not None:
        diferencia = abs(rumbo - rumbo_anterior) % 360.0
        cambio_rumbo = min(diferencia, 360.0 - diferencia)

    velocidad_anterior = track.get("velocidad_pixeles_min")
    aceleracion = None
    if isinstance(velocidad_anterior, (int, float)):
        aceleracion = (velocidad - float(velocidad_anterior)) / max(minutos, 0.1)

    if aceleracion is not None and aceleracion > 0.02:
        tendencia_velocidad = "ACELERANDO"
    elif aceleracion is not None and aceleracion < -0.02:
        tendencia_velocidad = "FRENANDO"
    else:
        tendencia_velocidad = "ESTABLE"

    confianza = calcular_confianza_movimiento(match, desplazamiento, minutos)
    calidad = "ALTA" if confianza >= 80 else "MEDIA" if confianza >= 65 else "BAJA"

    track.update({
        "ultimo_frame": timestamp,
        "frames": int(track.get("frames", 0)) + 1,
        "missed_frames": 0,
        "activo": True,
        "x": actual["x"],
        "y": actual["y"],
        "pixeles": actual["pixeles"],
        "intensidad": actual["intensidad"],
        "nivel_intensidad": actual["nivel_intensidad"],
        "estado": evolucion,
        "dx_pixeles": round(dx, 2),
        "dy_pixeles": round(dy, 2),
        "desplazamiento_pixeles": round(desplazamiento, 2),
        "velocidad_pixeles_min": round(velocidad, 3),
        "rumbo_grados": round(rumbo, 1) if rumbo is not None else None,
        "direccion": direccion,
        "cambio_rumbo_grados": round(cambio_rumbo, 1) if cambio_rumbo is not None else None,
        "aceleracion_pixeles_min2": round(aceleracion, 4) if aceleracion is not None else None,
        "tendencia_velocidad": tendencia_velocidad,
        "confianza_movimiento": confianza,
        "calidad_movimiento": calidad,
        "cambio_area_porcentaje": round(cambio_area, 1),
        "iou": round(match["iou"], 3),
        "ratio_area": round(match["ratio_area"], 3),
        "ratio_ancho": round(match["ratio_ancho"], 3),
        "ratio_alto": round(match["ratio_alto"], 3),
    })

    historial.append({
        "timestamp": timestamp,
        "x": actual["x"],
        "y": actual["y"],
        "pixeles": actual["pixeles"],
        "intensidad": actual["intensidad"],
        "estado": evolucion,
        "dx_pixeles": round(dx, 2),
        "dy_pixeles": round(dy, 2),
        "desplazamiento_pixeles": round(desplazamiento, 2),
        "velocidad_pixeles_min": round(velocidad, 3),
        "rumbo_grados": round(rumbo, 1) if rumbo is not None else None,
        "direccion": direccion,
        "cambio_rumbo_grados": round(cambio_rumbo, 1) if cambio_rumbo is not None else None,
        "aceleracion_pixeles_min2": round(aceleracion, 4) if aceleracion is not None else None,
        "tendencia_velocidad": tendencia_velocidad,
        "confianza_movimiento": confianza,
        "calidad_movimiento": calidad,
        "cambio_area_porcentaje": round(cambio_area, 1),
    })
    track["historial"] = historial[-MAX_HISTORIAL:]
    return evolucion, desplazamiento


def guardar_frame_inicial(actuales, radar_ts, motivo):
    tracks = []
    estado_nucleos = []
    next_id = 1
    ts = radar_ts or ahora_utc()
    for actual in actuales:
        track = crear_track(next_id, actual, ts)
        tracks.append(track)
        estado_nucleos.append({**actual, "track_id": next_id})
        next_id += 1

    guardar_json(ESTADO_FILE, {
        "timestamp": ahora_utc(),
        "radar_timestamp": radar_ts,
        "nucleos": estado_nucleos,
        "tracks": tracks,
    })
    guardar_json(RESULTADO_SEGUIMIENTO, {
        "timestamp": ahora_utc(),
        "radar_timestamp": radar_ts,
        "intervalo_minutos": None,
        "frame_anterior_disponible": False,
        "calidad_intervalo": "SIN_COMPARACION",
        "motivo": motivo,
        "nucleos_actuales": len(actuales),
        "nucleos_nuevos": len(actuales),
        "nucleos_seguimiento": 0,
        "nucleos_creciendo": 0,
        "nucleos_debilitandose": 0,
        "nucleos_movimiento": 0,
        "nucleos_estables": 0,
        "nucleos_desaparecidos": 0,
        "asociaciones": [],
        "tracks": tracks,
    })
    return tracks


def analizar():
    print("======================================")
    print("ClimaAR - SEGUIMIENTO TEMPORAL FINAL")
    print("======================================")

    resultado = cargar_json(RESULTADO_FILE)
    if resultado is None:
        raise RuntimeError("No existe resultado_procesamiento.json")

    actuales = [n for raw in obtener_nucleos(resultado) if (n := normalizar_nucleo(raw))]
    radar_dt = obtener_timestamp_radar()
    radar_ts = timestamp_iso(radar_dt)
    print(f"Núcleos actuales: {len(actuales)}")
    print(f"Radar timestamp: {radar_ts or 'NO DISPONIBLE'}")

    estado_anterior = cargar_json(ESTADO_FILE)
    if not estado_anterior:
        guardar_frame_inicial(actuales, radar_ts, "PRIMER_FRAME")
        print("Primer frame guardado. Ejecutar nuevamente para comparar.")
        return

    prev_dt = None
    prev_text = estado_anterior.get("radar_timestamp")
    if prev_text:
        try:
            prev_dt = datetime.fromisoformat(prev_text.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            prev_dt = None

    # Sem timestamp confiavel, nao calcular movimento.
    if radar_dt is None or prev_dt is None:
        guardar_frame_inicial(actuales, radar_ts, "TIMESTAMP_NO_CONFIABLE")
        print("Timestamp no confiable: se reinicia el estado sin calcular movimiento.")
        return

    delta_min = (radar_dt - prev_dt).total_seconds() / 60.0
    print(f"Intervalo entre frames: {delta_min:.1f} minutos")

    # Mesmo frame o frame atrasado: nao tocar no estado anterior.
    if delta_min <= 0:
        guardar_json(RESULTADO_SEGUIMIENTO, {
            "timestamp": ahora_utc(),
            "radar_timestamp": radar_ts,
            "intervalo_minutos": round(delta_min, 2),
            "frame_anterior_disponible": True,
            "calidad_intervalo": "FRAME_REPETIDO_O_ATRASADO",
            "motivo": "El frame actual no es posterior al ultimo estado procesado.",
            "nucleos_actuales": len(actuales),
            "nucleos_nuevos": 0,
            "nucleos_seguimiento": 0,
            "nucleos_creciendo": 0,
            "nucleos_debilitandose": 0,
            "nucleos_movimiento": 0,
            "nucleos_estables": 0,
            "nucleos_desaparecidos": 0,
            "asociaciones": [],
            "tracks": estado_anterior.get("tracks", []),
        })
        print("Frame repetido o atrasado: no se calcula movimiento.")
        return

    # Salto demasiado grande: no inventar velocidad/aceleracion.
    # El frame actual pasa a ser la nueva base para la proxima ejecucion.
    if delta_min < MIN_INTERVALO_MOVIMIENTO_MIN or delta_min > MAX_INTERVALO_MOVIMIENTO_MIN:
        motivo = "INTERVALO_DEMASIADO_CORTO" if delta_min < MIN_INTERVALO_MOVIMIENTO_MIN else "SALTO_TEMPORAL_DEMASIADO_GRANDE"
        guardar_frame_inicial(actuales, radar_ts, motivo)
        print(f"{motivo}: se reinicia el seguimiento y NO se calcula velocidad/aceleracion.")
        return

    minutos = delta_min
    anteriores = []
    for raw in estado_anterior.get("nucleos", []):
        n = normalizar_nucleo(raw)
        if n:
            n["track_id"] = raw.get("track_id")
            anteriores.append(n)

    tracks = [dict(t) for t in estado_anterior.get("tracks", []) if isinstance(t, dict)]
    candidatos = []
    for ia, actual in enumerate(actuales):
        for ip, anterior in enumerate(anteriores):
            match = compatibilidad(actual, anterior, minutos)
            if match is not None:
                candidatos.append((match["puntaje"], ia, ip, match))
    candidatos.sort(reverse=True, key=lambda x: x[0])

    asignaciones_actual = {}
    usados_prev = set()
    for _, ia, ip, match in candidatos:
        if ia in asignaciones_actual or ip in usados_prev:
            continue
        asignaciones_actual[ia] = (ip, match)
        usados_prev.add(ip)

    track_by_id = {}
    for t in tracks:
        try:
            track_by_id[int(t["id"])] = t
        except Exception:
            continue

    next_id = siguiente_id(tracks)
    estado_nucleos = []
    asociaciones = []
    nuevos = creciendo = debilitando = movimiento = estables = 0
    ts_actual = radar_ts or ahora_utc()

    for ia, actual in enumerate(actuales):
        if ia not in asignaciones_actual:
            track = crear_track(next_id, actual, ts_actual)
            tracks.append(track)
            track_by_id[next_id] = track
            estado_nucleos.append({**actual, "track_id": next_id})
            asociaciones.append({"track_id": next_id, "tipo": "NUEVO"})
            nuevos += 1
            next_id += 1
            continue

        ip, match = asignaciones_actual[ia]
        anterior = anteriores[ip]
        try:
            track_id = int(anterior.get("track_id"))
        except Exception:
            track_id = next_id
            next_id += 1

        track = track_by_id.get(track_id)
        if track is None:
            track = crear_track(track_id, anterior, estado_anterior.get("radar_timestamp", ts_actual))
            tracks.append(track)
            track_by_id[track_id] = track

        evolucion, desplazamiento = actualizar_track(track, actual, anterior, match, minutos, ts_actual)
        if evolucion == "CRECIENDO":
            creciendo += 1
        elif evolucion == "DEBILITANDOSE":
            debilitando += 1
        else:
            estables += 1
        if desplazamiento >= MOVIMIENTO_MINIMO:
            movimiento += 1

        estado_nucleos.append({**actual, "track_id": track_id})
        asociaciones.append({
            "track_id": track_id,
            "tipo": "SEGUIMIENTO",
            "distancia_pixeles": round(match["distancia"], 2),
            "iou": round(match["iou"], 3),
            "ratio_area": round(match["ratio_area"], 3),
            "ratio_ancho": round(match["ratio_ancho"], 3),
            "ratio_alto": round(match["ratio_alto"], 3),
            "cambio_area_porcentaje": round(match["cambio_area"], 1),
            "puntaje": round(match["puntaje"], 2),
            "dx_pixeles": track.get("dx_pixeles"),
            "dy_pixeles": track.get("dy_pixeles"),
            "desplazamiento_pixeles": track.get("desplazamiento_pixeles"),
            "velocidad_pixeles_min": track.get("velocidad_pixeles_min"),
            "rumbo_grados": track.get("rumbo_grados"),
            "direccion": track.get("direccion"),
            "cambio_rumbo_grados": track.get("cambio_rumbo_grados"),
            "aceleracion_pixeles_min2": track.get("aceleracion_pixeles_min2"),
            "tendencia_velocidad": track.get("tendencia_velocidad"),
            "confianza_movimiento": track.get("confianza_movimiento"),
            "calidad_movimiento": track.get("calidad_movimiento"),
        })

    desaparecidos = 0
    activos_ids = {n.get("track_id") for n in estado_nucleos}
    for ip, anterior in enumerate(anteriores):
        if ip in usados_prev:
            continue
        try:
            tid = int(anterior.get("track_id"))
        except Exception:
            continue
        track = track_by_id.get(tid)
        if track:
            track["missed_frames"] = int(track.get("missed_frames", 0)) + 1
            track["estado"] = "DESAPARECIDO"
            if track["missed_frames"] > MAX_MISSED_FRAMES:
                track["activo"] = False
        desaparecidos += 1

    for t in tracks:
        t.setdefault("activo", True)
        if t.get("id") in activos_ids:
            t["activo"] = True

    tracks = [t for t in tracks if t.get("activo", True)]

    seguimiento = {
        "timestamp": ahora_utc(),
        "radar_timestamp": radar_ts,
        "timestamp_anterior": timestamp_iso(prev_dt),
        "intervalo_minutos": round(minutos, 2),
        "frame_anterior_disponible": True,
        "calidad_intervalo": "VALIDO",
        "nucleos_actuales": len(actuales),
        "nucleos_nuevos": nuevos,
        "nucleos_seguimiento": len(asignaciones_actual),
        "nucleos_creciendo": creciendo,
        "nucleos_debilitandose": debilitando,
        "nucleos_movimiento": movimiento,
        "nucleos_estables": estables,
        "nucleos_desaparecidos": desaparecidos,
        "asociaciones": asociaciones,
        "tracks": tracks,
    }
    guardar_json(RESULTADO_SEGUIMIENTO, seguimiento)
    guardar_json(ESTADO_FILE, {
        "timestamp": ahora_utc(),
        "radar_timestamp": radar_ts,
        "nucleos": estado_nucleos,
        "tracks": tracks,
    })

    print("======================================")
    print("SEGUIMIENTO TEMPORAL FINAL")
    print("======================================")
    print(f"Actuales: {len(actuales)}")
    print(f"Seguimiento: {len(asignaciones_actual)}")
    print(f"Nuevos: {nuevos}")
    print(f"Creciendo: {creciendo}")
    print(f"Debilitándose: {debilitando}")
    print(f"Moviéndose: {movimiento}")
    print(f"Estables: {estables}")
    print(f"Desaparecidos: {desaparecidos}")


if __name__ == "__main__":
    analizar()
