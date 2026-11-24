"""Casos guardados (regla 7 del plan): datos de satélite y rayos de cada
tormenta, una línea por imagen de radar.

El estado GOES-19 (`goes19_state.json`) se pisa en cada ciclo. Este registro
queda para siempre y permite volver a correr cualquier tormenta real con los
datos del satélite tal como los vio ClimaAR (simulaciones, medición de
aciertos, calibración del puntaje del punto 4).

Archivos: data/goes19/historial/AAAA-MM-DD.jsonl (día en hora UTC, igual que
la hora del radar). Solo los números que hacen falta: ~0,3 KB por tormenta.
Se guarda solo cuando hay tormentas y una sola vez por imagen de radar.
Se conservan 400 días. Los días que ya cerraron (más de 2 días) se comprimen
(AAAA-MM-DD.jsonl.gz, ~10 veces más chico): 400 días sin comprimir eran
~300 MB dentro del repositorio (24/11/2026).
"""
from __future__ import annotations

import gzip
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
CARPETA = BASE_DIR / "data" / "goes19" / "historial"
DIAS_CONSERVAR = 400
DIAS_SIN_COMPRIMIR = 2
VERSION = 1


def _r(valor, dec=2):
    if isinstance(valor, (int, float)) and not isinstance(valor, bool):
        return round(float(valor), dec)
    return valor


def _tormenta(t: dict) -> dict[str, Any]:
    c13 = t.get("c13") or {}
    c08 = t.get("c08") or {}
    alt = t.get("altura_topes") or {}
    pres = t.get("presion_topes") or {}
    rayos = t.get("rayos") or {}
    sev = t.get("senales_severas") or {}
    ot = sev.get("topes_sobresalientes") or {}
    vi = sev.get("vapor_menos_infrarrojo") or {}
    a2 = sev.get("altura_topes_2km") or {}
    fase = sev.get("fase") or {}
    congel = sev.get("congelamiento") or {}
    mov = t.get("movimiento_km_min") or {}
    return {
        "track_id": t.get("track_id"),
        "lat": _r(t.get("latitud"), 4),
        "lon": _r(t.get("longitud"), 4),
        "radio_km": _r(t.get("radio_km"), 1),
        "dbz_max": _r(t.get("dbz_max"), 1),
        "mov_este_km_min": _r(mov.get("este"), 3),
        "mov_norte_km_min": _r(mov.get("norte"), 3),
        "c13_estado": c13.get("estado"),
        "c13_min_c": c13.get("min_c"),
        "c13_p10_c": c13.get("p10_c"),
        "c13_enfriamiento_c_10min": c13.get("enfriamiento_c_10min"),
        "c13_desfase_min": (c13.get("sincronizacion") or {}).get("desfase_min"),
        "c08_min_c": c08.get("min_c"),
        "altura_max_km": alt.get("max_km"),
        "altura_2km_max_km": a2.get("max_km"),
        "altura_2km_p90_km": a2.get("p90_km"),
        "presion_min_hpa": pres.get("min_hpa"),
        "rayos_10min": rayos.get("cantidad"),
        "rayos_cobertura_min": rayos.get("cobertura_min"),
        "rayos_tasas_2min": (rayos.get("salto_rayos") or {}).get("tasas_por_min"),
        "rayos_salto": (rayos.get("salto_rayos") or {}).get("salto"),
        "rayos_nivel_sigma": (rayos.get("salto_rayos") or {}).get("nivel_sigma"),
        "rayos_area_min_km2": (rayos.get("tamano_rayos") or {}).get("area_minima_km2"),
        "rayos_fraccion_chicos": (rayos.get("tamano_rayos") or {}).get("fraccion_chicos"),
        "ot_cantidad": ot.get("cantidad"),
        "ot_mas_frio_c": ot.get("mas_frio_c"),
        "ot_dif_yunque_c": ot.get("max_diferencia_yunque_c"),
        "vapor_ir_max_c": vi.get("max_c"),
        "hielo_en_nubes": fase.get("fraccion_hielo_en_nubes"),
        "nivel_0c_m": (congel.get("nivel") or {}).get("metros"),
        "tope_sobre_0c_km": congel.get("tope_sobre_congelamiento_km"),
        "altura_paralaje_km": t.get("altura_paralaje_km"),
    }


def registro(estado: dict) -> dict[str, Any] | None:
    """Línea del historial para un estado GOES-19, o None si no hay tormentas."""
    pt = estado.get("por_tormenta") or {}
    if pt.get("estado") != "ok" or not pt.get("tormentas"):
        return None
    sinc = (estado.get("sincronizacion") or {}).get("productos") or {}
    return {
        "v": VERSION,
        "frame_radar_utc": pt.get("frame_radar_utc"),
        "goes_version": estado.get("version"),
        "barrido_inicio_utc": (estado.get("sincronizacion") or {}).get("inicio_barrido_referencia_utc"),
        "productos": {nombre: (item or {}).get("estado") for nombre, item in sinc.items()},
        "topes_sobresalientes_zona": pt.get("topes_sobresalientes_zona"),
        "saltos_de_rayos": pt.get("saltos_de_rayos"),
        "rayos_cobertura_min": ((estado.get("glm") or {}).get("rayos_frame_radar") or {}).get("cobertura_min"),
        "tormentas": [_tormenta(t) for t in pt["tormentas"]],
    }


def _archivo(frame: datetime, carpeta: Path) -> Path:
    return carpeta / f"{frame:%Y-%m-%d}.jsonl"


def _completitud(caso: dict) -> int:
    """Cuántos productos del satélite llegaron sincronizados."""
    return sum(1 for v in (caso.get("productos") or {}).values() if v == "sincronizado")


def _dia(p: Path):
    nombre = p.name.split(".", 1)[0]
    try:
        return datetime.strptime(nombre, "%Y-%m-%d").date()
    except ValueError:
        return None


def _comprimir(p: Path) -> None:
    gz = p.with_name(p.name + ".gz")
    tmp = gz.with_name(gz.name + ".tmp")
    with p.open("rb") as origen, gzip.open(tmp, "wb", compresslevel=9) as destino:
        destino.write(origen.read())
    os.replace(tmp, gz)
    p.unlink(missing_ok=True)


def _limpiar(carpeta: Path, hoy: datetime) -> None:
    limite = (hoy - timedelta(days=DIAS_CONSERVAR)).date()
    cerrado = (hoy - timedelta(days=DIAS_SIN_COMPRIMIR)).date()
    for p in list(carpeta.glob("*.jsonl")) + list(carpeta.glob("*.jsonl.gz")):
        dia = _dia(p)
        if dia is None:
            continue
        if dia < limite:
            p.unlink(missing_ok=True)
        elif p.suffix == ".jsonl" and dia < cerrado:
            try:
                _comprimir(p)
            except OSError:
                pass  # se reintenta la próxima vez


def leer_renglones(path: Path) -> list[str]:
    """Renglones de un día, esté comprimido o no (AAAA-MM-DD.jsonl[.gz])."""
    renglones = []
    gz = path.with_name(path.name + ".gz")
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            renglones += f.read().splitlines()
    if path.exists():
        renglones += path.read_text(encoding="utf-8").splitlines()
    return renglones


def guardar(estado: dict, carpeta: Path = CARPETA) -> str:
    """Agrega el caso de esta imagen de radar. Devuelve qué hizo (para el log).
    Nunca corta el ciclo: cualquier error se informa y se sigue."""
    try:
        linea = registro(estado)
        if linea is None:
            return "sin tormentas: no se guarda"
        texto = str(linea["frame_radar_utc"])
        frame = datetime.fromisoformat(texto.replace("Z", "+00:00")).astimezone(timezone.utc)
        carpeta.mkdir(parents=True, exist_ok=True)
        path = _archivo(frame, carpeta)
        nueva = json.dumps(linea, ensure_ascii=False, separators=(", ", ": "))
        renglones = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        for i, renglon in enumerate(renglones):
            if f'"frame_radar_utc": "{texto}"' not in renglon:
                continue
            try:
                previo = json.loads(renglon)
            except ValueError:
                previo = {}
            # La misma imagen de radar se guarda una sola vez; solo se
            # reemplaza si ahora llegaron más productos del satélite.
            if _completitud(linea) > _completitud(previo):
                renglones[i] = nueva
                path.write_text("\n".join(renglones) + "\n", encoding="utf-8")
                return f"reemplazado con más datos del satélite ({texto})"
            return f"ya guardado ({texto})"
        with path.open("a", encoding="utf-8") as f:
            f.write(nueva + "\n")
        _limpiar(carpeta, frame)
        return f"guardado: {len(linea['tormentas'])} tormentas en {path.name}"
    except Exception as exc:  # falla segura
        return f"ERROR (no corta el ciclo): {type(exc).__name__}: {exc}"


def leer(desde: datetime, hasta: datetime, carpeta: Path = CARPETA) -> list[dict]:
    """Casos guardados entre dos horas UTC (para simulaciones)."""
    salida = []
    dia = desde.date()
    while dia <= hasta.date():
        for renglon in leer_renglones(carpeta / f"{dia:%Y-%m-%d}.jsonl"):
            try:
                caso = json.loads(renglon)
                t = datetime.fromisoformat(caso["frame_radar_utc"].replace("Z", "+00:00"))
            except (ValueError, KeyError):
                continue
            if desde <= t <= hasta:
                salida.append(caso)
        dia += timedelta(days=1)
    return salida
