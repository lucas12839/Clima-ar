#!/usr/bin/env python3
"""Control de salud de la cadena (auditoría del 20/11/2026).

Revisa en cada ciclo las cosas que se habían pasado por alto, para que no
vuelvan a quedar escondidas:

- estaciones: cuántas entran por red; el SMN con 0 válidas durante horas
  indicaba un límite de edad mal puesto; estaciones sin datos hace un día;
- reloj CWOP: si su último dato es viejo, la cadena encadenada se cortó;
- latido del detector del VPS: si está apagado, se está funcionando con el
  respaldo de GitHub;
- informe del aeropuerto (SAZB) y edad del radar;
- numeración de tormentas: nunca debe volver atrás;
- predicción por tormenta: un error típico enorme a 10 min indica que se
  comparan tormentas distintas;
- tamaño de los archivos que se guardan en cada ciclo.

Escribe data/ia/salud_cadena.json y avisa cada problema en Actions
(::warning::). En paralelo: nunca corta el ciclo ni cambia lo que se ve.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))

from estaciones import frescura  # noqa: E402

DATA = BASE / "data"
SALIDA = DATA / "ia" / "salud_cadena.json"
VERSION = "1.0.0"

KM_POR_PX = 240.0 / 255.5
LIMITE_CWOP_MIN = 30.0         # transmite cada 10 min; la vuelta del reloj dura ~10-12
LIMITE_LATIDO_MIN = 10.0
LIMITE_RADAR_MIN = 40.0
LIMITE_ERROR_10MIN_KM = 15.0   # una tormenta se mueve ~10 km en 10 min
MIN_MUESTRAS_ERROR = 20
LIMITE_ARCHIVO_KB = 800.0
ARCHIVOS_CICLO = (
    "ia/resultado_final.json",
    "ia/inteligencia_tormenta.json",
    "ia/estado_meteorologico.json",
    "ia/goes19_intelligence.json",
    "radar/radar_nowcast.json",
)


def leer(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def revisar(ahora: datetime | None = None, latido: str | None = None,
            previo: dict | None = None) -> dict[str, Any]:
    ahora = ahora or datetime.now(timezone.utc)
    alertas: list[str] = []
    datos: dict[str, Any] = {}

    # Estaciones por red ---------------------------------------------------
    red = leer(DATA / "ia/estaciones/red_observaciones_estado.json") or {}
    smn = red.get("smn") or {}
    datos["estaciones"] = {
        "validas": red.get("valid_count"), "total": red.get("station_count"),
        "bcp": red.get("bcp"), "smn": smn,
        "sin_datos_24h": red.get("estaciones_sin_datos_24h") or [],
    }
    if smn.get("total") and not smn.get("valid"):
        alertas.append("Ninguna estación del SMN entra (¿límite de edad o fuente caída?).")
    if (red.get("bcp") or {}).get("total") and not (red.get("bcp") or {}).get("valid"):
        alertas.append("Ninguna estación BCP entra.")
    if datos["estaciones"]["sin_datos_24h"]:
        alertas.append("Estaciones sin datos hace más de un día: "
                       + ", ".join(datos["estaciones"]["sin_datos_24h"]) + ".")

    # Reloj CWOP -----------------------------------------------------------
    cwop = leer(DATA / "ia/estaciones/observaciones/cwop_actual.json") or {}
    edades = [frescura.edad_min(s.get("observation_time"), ahora) for s in cwop.get("stations") or []]
    edades = [e for e in edades if e is not None]
    edad_cwop = min(edades) if edades else None
    datos["cwop_edad_min"] = round(edad_cwop, 1) if edad_cwop is not None else None
    if edad_cwop is None or edad_cwop > LIMITE_CWOP_MIN:
        alertas.append("La estación CWOP no se actualiza: el reloj encadenado (climaar-cwop.yml) "
                       "puede haberse cortado.")

    # Latido del detector del VPS --------------------------------------------
    edad_latido = frescura.edad_min(latido, ahora) if latido else None
    datos["latido_detector_edad_min"] = round(edad_latido, 1) if edad_latido is not None else None
    if latido is not None and (edad_latido is None or edad_latido > LIMITE_LATIDO_MIN):
        alertas.append("El detector del VPS no da latido: se está funcionando con el respaldo de GitHub.")

    # SAZB y radar ---------------------------------------------------------
    sazb = leer(DATA / "sazb/status.json") or {}
    edad_sazb = frescura.edad_min(sazb.get("obsTime_utc"), ahora)
    datos["sazb_edad_min"] = round(edad_sazb, 1) if edad_sazb is not None else None
    if edad_sazb is None or edad_sazb > frescura.limite_min("sazb"):
        alertas.append("El informe del aeropuerto (SAZB) está vencido.")
    ts = (DATA / "radar/radar_timestamp.txt")
    edad_radar = frescura.edad_min(ts.read_text(encoding="utf-8").strip(), ahora) if ts.exists() else None
    datos["radar_edad_min"] = round(edad_radar, 1) if edad_radar is not None else None
    if edad_radar is None or edad_radar > LIMITE_RADAR_MIN:
        alertas.append("La imagen de radar tiene más de 40 min.")

    # Numeración de tormentas ------------------------------------------------
    estado = leer(DATA / "radar/seguimiento/estado.json") or {}
    siguiente = estado.get("siguiente_track_id")
    datos["siguiente_numero_tormenta"] = siguiente
    anterior = (previo or {}).get("siguiente_numero_tormenta")
    if isinstance(siguiente, int) and isinstance(anterior, int) and siguiente < anterior:
        alertas.append(f"La numeración de tormentas volvió atrás ({anterior} -> {siguiente}).")

    # Seguimiento: reconstruir desde cero en cada imagen cambia los números de
    # las tormentas (pasaba en casi todos los frames hasta el 20/11/2026).
    seg = leer(DATA / "radar/seguimiento/seguimiento_temporal.json") or {}
    modo = seg.get("modo_seguimiento")
    frame = estado.get("ultimo_radar_timestamp")
    seguidas = int((previo or {}).get("reconstrucciones_seguidas") or 0)
    if frame != (previo or {}).get("frame_revisado"):  # solo cuenta imágenes nuevas
        seguidas = seguidas + 1 if modo == "RECONSTRUCCION_HISTORICA" else 0
    datos["modo_seguimiento"] = modo
    datos["frame_revisado"] = frame
    datos["reconstrucciones_seguidas"] = seguidas
    if seguidas >= 3:
        alertas.append(f"El seguimiento se reconstruyó desde cero en {seguidas} imágenes seguidas: "
                       "las tormentas cambian de número.")

    # Error de la predicción por tormenta a 10 min -----------------------------
    cal = leer(DATA / "radar/seguimiento/calibracion_prediccion.json") or {}
    errores = [r.get("error_pixeles") for r in cal.get("registros") or []
               if abs((r.get("horizonte_minutos") or 0) - 10) < 1 and r.get("error_pixeles") is not None]
    if errores:
        mediana_km = statistics.median(errores) * KM_POR_PX
        datos["prediccion_error_10min_km"] = {"mediana": round(mediana_km, 1), "muestras": len(errores)}
        if len(errores) >= MIN_MUESTRAS_ERROR and mediana_km > LIMITE_ERROR_10MIN_KM:
            alertas.append(f"La predicción por tormenta erra {mediana_km:.0f} km a 10 min "
                           "(¿se comparan tormentas distintas?).")

    # Lo que se ve sale del 5b (desde el 20/11/2026) ------------------------------
    final = leer(DATA / "ia/resultado_final.json") or {}
    fuente_lleg = ((final.get("tipo_tiempo") or {}).get("fuente_llegadas")) if isinstance(final, dict) else None
    meta = (((leer(DATA / "radar/radar_nowcast.json") or {}).get("nowcast") or {}).get("proyecciones_metadata") or {})
    datos["fuente_llegadas"] = fuente_lleg
    datos["fuente_trayectorias"] = meta.get("fuente")
    if fuente_lleg == "flujo_optico":
        alertas.append("Las llegadas que se ven salieron del flujo óptico (respaldo): el 5b no estuvo disponible.")
    if meta.get("pronostico_disponible") and meta.get("fuente") == "prediccion_evolucion":
        alertas.append("Las trayectorias salieron del predictor viejo (respaldo): el 5b no estuvo disponible.")

    # Ambiente medido por el satélite (punto 8) --------------------------------
    amb = leer(DATA / "ia/ambiente_satelite.json")
    if isinstance(amb, dict):
        datos["ambiente_satelite"] = {"disponible": amb.get("disponible"),
                                      "hora_satelite_utc": amb.get("hora_satelite_utc")}
        if amb.get("disponible") is False:
            alertas.append(f"El ambiente del satélite no se pudo leer: {amb.get('motivo')}.")

    # Tamaño de lo que se guarda en cada ciclo -----------------------------------
    tamanos = {}
    for rel in ARCHIVOS_CICLO:
        p = DATA / rel
        if p.exists():
            tamanos[rel] = round(p.stat().st_size / 1024, 1)
    datos["tamano_kb"] = tamanos
    grandes = [k for k, v in tamanos.items() if v > LIMITE_ARCHIVO_KB]
    if grandes:
        alertas.append("Archivos de cada ciclo muy pesados: " + ", ".join(grandes) + ".")

    return {
        "version": VERSION,
        "en_paralelo": True,
        "generado_utc": ahora.isoformat(),
        "estado": "ok" if not alertas else "con_alertas",
        "alertas": alertas,
        **datos,
    }


def main() -> int:
    try:
        latido = os.environ.get("CLIMAAR_DETECTOR_LATIDO")
        latido = latido.strip() if latido and latido.strip() else None
        salida = revisar(latido=latido, previo=leer(SALIDA))
        SALIDA.parent.mkdir(parents=True, exist_ok=True)
        tmp = SALIDA.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(SALIDA)
        for alerta in salida["alertas"]:
            print(f"::warning::Salud de la cadena: {alerta}")
        print(f"Salud de la cadena: {salida['estado']} ({len(salida['alertas'])} alertas)")
    except Exception as exc:  # nunca corta el ciclo
        print(f"::warning::No se pudo revisar la salud de la cadena: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
