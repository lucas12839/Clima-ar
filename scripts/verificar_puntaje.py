#!/usr/bin/env python3
"""Etapa B, punto 9: el puntaje de gravedad contra lo que se midió después.

El puntaje (scripts/puntaje_gravedad.py, EN PARALELO) da a cada tormenta un
grado de 0 a 4 de ráfagas, lluvia intensa y granizo. Acá se mira qué pasó de
verdad debajo de cada tormenta, para ajustar los cortes más adelante (etapa F):

- Cada imagen de radar guardada en data/ia/puntaje/AAAA-MM-DD.jsonl, cuando ya
  pasaron 45 min, se compara con las estaciones que estaban a <= 10 km de cada
  tormenta: viento máximo (ráfaga o viento medio, lo que haya) y lluvia medida
  desde 10 min antes hasta 40 min después de la imagen.
- Rayos: los que tuvo la misma tormenta en las 3 imágenes siguientes
  (data/goes19/historial/), si esas imágenes están.
- Granizo: no hay ninguna medición de granizo, así que no se verifica (se
  cuenta igual cuántas veces el puntaje lo marcó).

Estaciones con lluvia o datos dudosos (calidad_estaciones.json) no se usan.
Lee y nada más: puntaje, historial de estaciones (últimas 6 h), catálogo,
calidad, historial de GOES-19. Escribe solo lo suyo:
- data/ia/verificacion_puntaje.json (acumulado por mes, por fenómeno y grado);
- data/ia/verificacion_puntaje/AAAA-MM.jsonl (cada caso, para recalcular).

EN PARALELO: no cambia lo que se ve. Falla segura: nunca corta el ciclo.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[1]
for _p in (BASE / "scripts", BASE / "scripts" / "estaciones"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PUNTAJE_DIR = BASE / "data" / "ia" / "puntaje"
GOES_DIR = BASE / "data" / "goes19" / "historial"
HISTORIAL_SUP = BASE / "data" / "ia" / "estaciones" / "historial_superficie.json"
CALIDAD = BASE / "data" / "ia" / "calibracion" / "calidad_estaciones.json"
SALIDA = BASE / "data" / "ia" / "verificacion_puntaje.json"
CASOS_DIR = BASE / "data" / "ia" / "verificacion_puntaje"

VERSION = "1.0.0"
ESPERA_MIN = 45.0          # 40 min de ventana + margen para que entren los datos
VENTANA_ANTES_MIN = 10.0
VENTANA_DESPUES_MIN = 40.0
DISTANCIA_KM = 10.0
HISTORIA_HORAS = 5.5       # el historial de estaciones guarda 6 h
FRAMES_RAYOS = 3
UMBRALES = {"viento_kmh": (60.0, 90.0), "lluvia_mm": (10.0, 20.0), "rayos": (1, 10)}
FENOMENOS = (("rafagas", "viento_kmh"), ("lluvia", "lluvia_mm"), ("granizo", None))


def leer(path: Path, defecto):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return defecto


def escribir(path: Path, datos) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(datos, fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def parse(texto) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(texto).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def dist_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, a)))


def leer_jsonl(path: Path) -> list[dict]:
    """Líneas JSON de un archivo (también si ya se comprimió: .jsonl.gz)."""
    salida = []
    try:
        texto = ""
        gz = path.with_name(path.name + ".gz")
        if gz.exists():
            import gzip
            with gzip.open(gz, "rt", encoding="utf-8") as f:
                texto = f.read()
        if path.exists():
            texto += "\n" + path.read_text(encoding="utf-8")
        for linea in texto.splitlines():
            try:
                salida.append(json.loads(linea))
            except ValueError:
                continue
    except OSError:
        pass
    return salida


def dias(desde: datetime, hasta: datetime) -> list[str]:
    d, salida = desde.date(), []
    while d <= hasta.date():
        salida.append(d.isoformat())
        d += timedelta(days=1)
    return salida


# --------------------------------------------------------------------------
# Lo medido (funciones puras, tests/test_verificar_puntaje.py)
# --------------------------------------------------------------------------
def viento_max(serie: list[dict], desde: datetime, hasta: datetime) -> float | None:
    valores = [max(v for v in (r.get("rafaga"), r.get("viento")) if isinstance(v, (int, float)))
               for r in serie
               if (t := parse(r.get("utc"))) is not None and desde <= t <= hasta
               and any(isinstance(v, (int, float)) for v in (r.get("rafaga"), r.get("viento")))]
    return round(max(valores), 1) if valores else None


def cubre(serie: list[dict], desde: datetime, hasta: datetime) -> bool:
    """La estación tiene datos al principio y al final de la ventana."""
    tiempos = [t for r in serie if (t := parse(r.get("utc"))) is not None]
    return (any(desde - timedelta(minutes=10) <= t <= desde + timedelta(minutes=10) for t in tiempos)
            and any(hasta - timedelta(minutes=10) <= t <= hasta + timedelta(minutes=10) for t in tiempos))


def rayos_despues(goes_por_frame: dict[str, dict], frame: datetime, track_id) -> int | None:
    """Rayos de la misma tormenta en las 3 imágenes siguientes (None si faltan imágenes)."""
    siguientes = [goes_por_frame.get(iso(frame + timedelta(minutes=10 * i))) for i in range(1, FRAMES_RAYOS + 1)]
    if any(g is None for g in siguientes):
        return None
    total = 0
    for g in siguientes:
        for t in g.get("tormentas") or []:
            if t.get("track_id") == track_id and isinstance(t.get("rayos_10min"), (int, float)):
                total += int(t["rayos_10min"])
    return total


def sumar(acc: dict, caso: dict) -> None:
    """Acumula un caso: por fenómeno y grado, cuántas veces se superó cada umbral."""
    med = caso["medido"]
    for fen, campo in FENOMENOS:
        grado = str(caso["grados"].get(fen, 0))
        fila = acc.setdefault(fen, {}).setdefault(grado, {"casos": 0})
        fila["casos"] += 1
        for nombre in ([campo] if campo else []) + ["rayos"]:
            v = med.get(nombre)
            if v is None:
                continue
            bajo, alto = UMBRALES[nombre]
            fila[f"{nombre}_casos"] = fila.get(f"{nombre}_casos", 0) + 1
            fila[f"{nombre}_suma"] = round(fila.get(f"{nombre}_suma", 0.0) + float(v), 1)
            fila[f"{nombre}_max"] = max(fila.get(f"{nombre}_max", 0.0), float(v))
            fila[f"{nombre}_ge_{bajo:g}"] = fila.get(f"{nombre}_ge_{bajo:g}", 0) + (v >= bajo)
            fila[f"{nombre}_ge_{alto:g}"] = fila.get(f"{nombre}_ge_{alto:g}", 0) + (v >= alto)


def resumen(acc: dict) -> dict:
    """Para cada grado: en qué % de los casos se midió algo fuerte o severo."""
    salida = {}
    for fen, campo in FENOMENOS:
        for grado, fila in sorted((acc.get(fen) or {}).items()):
            item = {"casos": fila["casos"]}
            for nombre in ([campo] if campo else []) + ["rayos"]:
                n = fila.get(f"{nombre}_casos", 0)
                if not n:
                    continue
                bajo, alto = UMBRALES[nombre]
                item[nombre] = {"casos": n, "promedio": round(fila[f"{nombre}_suma"] / n, 1),
                                "maximo": fila[f"{nombre}_max"],
                                f"pct_ge_{bajo:g}": round(100 * fila[f"{nombre}_ge_{bajo:g}"] / n, 1),
                                f"pct_ge_{alto:g}": round(100 * fila[f"{nombre}_ge_{alto:g}"] / n, 1)}
            salida.setdefault(fen, {})[grado] = item
    return salida


def casos_de_frame(registro: dict, estaciones: list[dict], historial: dict, dudosas: set[str],
                   goes_por_frame: dict[str, dict]) -> list[dict]:
    frame = parse(registro.get("frame_radar_utc"))
    if frame is None:
        return []
    desde = frame - timedelta(minutes=VENTANA_ANTES_MIN)
    hasta = frame + timedelta(minutes=VENTANA_DESPUES_MIN)
    import calidad_y_calibracion as cc
    salida = []
    for t in registro.get("tormentas") or []:
        if t.get("lat") is None or t.get("lon") is None:
            continue
        rayos = rayos_despues(goes_por_frame, frame, t.get("track_id"))
        for e in estaciones:
            nombre = e["nombre"]
            serie = historial.get(nombre) or []
            if nombre in dudosas or not serie or not cubre(serie, desde, hasta):
                continue
            d = dist_km(t["lat"], t["lon"], e["lat"], e["lon"])
            if d > DISTANCIA_KM:
                continue
            salida.append({
                "frame_radar_utc": iso(frame), "track_id": t.get("track_id"), "estacion": nombre,
                "distancia_km": round(d, 1), "sospecha": bool(t.get("sospecha")),
                "potencial": {f: (t.get(f) or [None])[0] for f, _ in FENOMENOS},
                "grados": {f: int((t.get(f) or [0, 0])[1] or 0) for f, _ in FENOMENOS},
                "medido": {"viento_kmh": viento_max(serie, desde, hasta),
                           "lluvia_mm": cc.lluvia_entre(serie, desde, hasta),
                           "rayos": rayos},
            })
    return salida


def main(ahora: datetime | None = None) -> int:
    try:
        import localidades_radar
        ahora = ahora or datetime.now(timezone.utc)
        previo = leer(SALIDA, {})
        ultimo = parse(previo.get("ultimo_frame_verificado"))
        limite = ahora - timedelta(minutes=ESPERA_MIN)
        inicio = max(ahora - timedelta(hours=HISTORIA_HORAS), ultimo or datetime.min.replace(tzinfo=timezone.utc))
        historial = leer(HISTORIAL_SUP, {})
        calidad = (leer(CALIDAD, {}).get("estaciones") or {})
        dudosas = {k for k, v in calidad.items() if v.get("estado") == "dudosa" or v.get("estado_lluvia") == "dudosa"}
        estaciones = localidades_radar.cargar_puntos_control()
        registros, goes = [], {}
        for dia in dias(inicio - timedelta(hours=1), ahora):
            registros += leer_jsonl(PUNTAJE_DIR / f"{dia}.jsonl")
            for g in leer_jsonl(GOES_DIR / f"{dia}.jsonl"):
                if g.get("frame_radar_utc"):
                    goes[iso(parse(g["frame_radar_utc"]))] = g
        vistos, nuevos = set(), []
        for r in sorted(registros, key=lambda x: x.get("frame_radar_utc") or ""):
            f = parse(r.get("frame_radar_utc"))
            if f is None or f in vistos or not (inicio < f <= limite):
                continue
            vistos.add(f)   # refrescos repiten la misma imagen: se cuenta una sola vez
            nuevos += casos_de_frame(r, estaciones, historial, dudosas, goes)
        acumulado = previo.get("acumulado_por_mes") or {}
        for c in nuevos:
            sumar(acumulado.setdefault(c["frame_radar_utc"][:7], {}), c)
        por_mes: dict[str, list[dict]] = {}
        for c in nuevos:
            por_mes.setdefault(c["frame_radar_utc"][:7], []).append(c)
        for mes, casos in por_mes.items():
            CASOS_DIR.mkdir(parents=True, exist_ok=True)
            with (CASOS_DIR / f"{mes}.jsonl").open("a", encoding="utf-8") as fh:
                for c in casos:
                    fh.write(json.dumps(c, ensure_ascii=False, separators=(",", ":")) + "\n")
        ultimo_nuevo = max(vistos) if vistos else ultimo
        salida = {
            "version": VERSION, "en_paralelo": True, "generado_utc": iso(ahora),
            "ultimo_frame_verificado": iso(ultimo_nuevo) if ultimo_nuevo else None,
            "criterio": {"distancia_km": DISTANCIA_KM, "ventana_min": [-VENTANA_ANTES_MIN, VENTANA_DESPUES_MIN],
                         "umbrales": UMBRALES, "granizo": "sin medición: no se verifica"},
            "casos_este_ciclo": len(nuevos),
            "acumulado_por_mes": acumulado,
            "resumen_por_mes": {mes: resumen(acc) for mes, acc in sorted(acumulado.items())},
        }
        escribir(SALIDA, salida)
        print(f"Verificación del puntaje: {len(vistos)} imágenes nuevas, {len(nuevos)} casos tormenta-estación.")
    except Exception as exc:  # nunca corta el ciclo
        print(f"::warning::No se pudo verificar el puntaje de gravedad: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
