#!/usr/bin/env python3
"""Control de salud de la cadena (auditoría del 20/11/2026).

Revisa en cada ciclo las cosas que se habían pasado por alto, para que no
vuelvan a quedar escondidas:

- estaciones: cuántas entran por red; el SMN con 0 válidas durante horas
  indicaba un límite de edad mal puesto; estaciones sin datos hace un día;
- reloj CWOP: si su último dato es viejo, la cadena encadenada se cortó;
- latido del detector del VPS: si está apagado, se está funcionando con el
  respaldo de GitHub;
- clave de GitHub del VPS por vencer y actualización automática del VPS
  (etapa C; el aviso al celular lo da scripts/vigilancia.py);
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
ERROR_5B_MEDIDO_10MIN_KM = 4.5  # herramientas/comparar_trayectorias.py (20/11/2026)
LIMITE_ERROR_5B_10MIN_KM = 9.0  # el doble de lo medido: algo se rompió
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


# Resultados de cada ciclo y dónde dicen de qué imagen de radar son. Los pasos
# "en paralelo" nunca cortan el ciclo: si uno falla, su archivo queda de una
# imagen vieja y antes nadie se enteraba (re-auditoría 23/11/2026).
# Atrasado = no se volvió a escribir en este ciclo (marca de generación vieja) o
# dice ser de otra imagen. Sin hora de radar (p. ej. sin ecos) no es atraso.
SINCRONIA_CICLO = (
    ("ia/ambiente_satelite.json", ("frame_radar_utc",), "generado_utc"),
    ("ia/celdas_localidad.json", ("radar_timestamp",), "generado_utc"),
    ("ia/estaciones/alarma_llegada.json", ("movimiento", "radar_utc"), "generado_utc"),
    ("ia/formacion_tormentas.json", ("frame_radar_utc",), "generado_utc"),
    ("ia/indicios_severos.json", ("radar_timestamp",), "generado_utc"),
    ("ia/puntaje_tormentas.json", ("frame_radar_utc",), "generado_utc"),
    ("ia/goes19_intelligence.json", ("reference_frame", "frame_actual_utc"), "generated_at_utc"),
    ("radar/nowcast_capas.json", ("radar_timestamp",), "generado_utc"),
    ("radar/nowcast_flujo.json", ("radar_timestamp",), "generado_utc"),
    ("radar/trayectorias_capas.json", ("radar_timestamp",), "generado_utc"),
)
GENERADO_MAX_MIN = 15.0  # el ciclo de la IA dura ~2 min
# Una alerta que sigue igual este tiempo pasa a la vigilancia (aviso al celular).
ALERTA_PERSISTENTE_MIN = 20.0


def _clave_alerta(texto: str) -> str:
    """Misma alerta aunque cambien los números (edades, minutos)."""
    import re
    return re.sub(r"[0-9]+(?:[.,][0-9]+)?", "N", texto)[:90]


def _dato(d: Any, ruta: tuple[str, ...]) -> Any:
    for k in ruta:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def revisar(ahora: datetime | None = None, latido: str | None = None,
            previo: dict | None = None, info_detector: dict | None = None) -> dict[str, Any]:
    ahora = ahora or datetime.now(timezone.utc)
    alertas: list[str] = []
    datos: dict[str, Any] = {}

    # Estaciones por red ---------------------------------------------------
    red = leer(DATA / "ia/estaciones/red_observaciones_estado.json") or {}
    smn = red.get("smn") or {}
    datos["estaciones"] = {
        "validas": red.get("valid_count"),
        # Solo las activas: las dadas de baja (Cabildo, San Eloy, Pasman, SMN) no cuentan.
        "total": red.get("activas", red.get("station_count")),
        "bcp": red.get("bcp"), "smn": smn,
        "sin_datos_24h": red.get("estaciones_sin_datos_24h") or [],
    }
    if smn.get("total") and not smn.get("valid"):
        edades_smn = [x.get("age_minutes") for x in red.get("stations") or []
                      if x.get("network") == "smn" and isinstance(x.get("age_minutes"), (int, float))]
        if edades_smn:
            alertas.append(f"Ninguna estación del SMN entra: su dato más nuevo es de hace "
                           f"{min(edades_smn) / 60:.1f} h (el SMN las publica con atraso; se usan las BCP y CWOP).")
        else:
            alertas.append("Ninguna estación del SMN entra: la fuente del SMN no devuelve datos.")
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

    # Clave y actualización del VPS (etapa C, 23/11/2026) ------------------------
    if isinstance(info_detector, dict):
        import vigilancia
        datos["detector"] = {k: info_detector.get(k) for k in ("utc", "commit", "clave_vence", "actualizacion")}
        for p in vigilancia.revisar(ahora, latido or (ahora.isoformat()), info_detector, ahora.isoformat(), [], [
                {"created_at": ahora.isoformat(), "status": "in_progress"}]):
            if p["clave"].startswith(("clave_", "actualizacion_")):
                alertas.append(p["texto"])

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

    # Error de la trayectoria que se ve (5b) a 10 min, medido en vivo ------------
    # (23/11/2026: antes se miraba solo el predictor viejo, que ya no se ve).
    ver = ((leer(DATA / "ia/verificacion_trayectorias.json") or {}).get("resumen") or {})
    datos["trayectoria_5b_error_km"] = ver or None
    diez = ver.get("10") or {}
    if (diez.get("muestras") or 0) >= MIN_MUESTRAS_ERROR and (diez.get("mediana_km") or 0) > LIMITE_ERROR_5B_10MIN_KM:
        alertas.append(f"La trayectoria que se ve erra {diez['mediana_km']:.0f} km a 10 min en vivo "
                       f"(en las pruebas: {ERROR_5B_MEDIDO_10MIN_KM} km).")

    # Error del predictor viejo (respaldo) a 10 min -----------------------------
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

    # Pronóstico europeo (Open-Meteo): se baja cada 30 min; si no responde se usa
    # el anterior hasta 9 h. Pasado eso se corta todo el ciclo: avisar antes.
    amb = leer(DATA / "meteorologia/ambiente_futuro.json")
    if isinstance(amb, dict):
        edad_ecmwf = frescura.edad_min(amb.get("actualizado_utc"), ahora)
        datos["ecmwf_edad_min"] = round(edad_ecmwf, 1) if edad_ecmwf is not None else None
        if edad_ecmwf is None or edad_ecmwf > 90.0:
            alertas.append(f"El pronóstico europeo (Open-Meteo) no se actualiza hace "
                           f"{(edad_ecmwf or 0) / 60.0:.1f} h: se usa el anterior (sirve hasta 9 h).")

    # Control de calidad de las estaciones (24/11/2026) -------------------------
    # Un dato que no pasa el control no se usa; acá se avisa para que se vea
    # (por ejemplo, un anemómetro roto) antes de que lo note otra persona.
    sen = leer(DATA / "ia/estaciones/senales_superficie.json")
    if isinstance(sen, dict):
        descartes = sen.get("datos_descartados") or []
        datos["datos_descartados"] = descartes
        if descartes:
            NOMBRE_DATO = {"temperatura_c": "temperatura", "viento_kmh": "viento", "rafaga_kmh": "ráfaga",
                           "humedad_pct": "humedad", "presion_hpa": "presión"}
            alertas.append("Datos de estaciones que no pasan el control de calidad (no se usan): " + "; ".join(
                f"{d.get('estacion')} {NOMBRE_DATO.get(d.get('dato'), d.get('dato'))}: {d.get('motivo')}"
                for d in descartes[:8]) + ".")
    meteo = leer(DATA / "ia/estado_meteorologico.json")
    if isinstance(meteo, dict):
        ml = meteo.get("modelo_localidades") or {}
        edad_ml = frescura.edad_min(ml.get("actualizado_utc"), ahora)
        datos["modelo_localidades_edad_min"] = round(edad_ml, 1) if edad_ml is not None else None
        if edad_ml is None or edad_ml > 5 * 60:
            alertas.append("No hay modelo europeo por localidad al día: la página usa su respaldo "
                           "y el viento del índice usa el de Bahía.")

    # Calibración de la lluvia con el SMN (corre 2 veces por día) ---------------
    cal_smn = leer(DATA / "ia/calibracion/factor_lluvia_smn.json")
    if isinstance(cal_smn, dict):
        edad_cal = frescura.edad_min(cal_smn.get("actualizado_utc"), ahora)
        datos["calibracion_smn_edad_h"] = round(edad_cal / 60.0, 1) if edad_cal is not None else None
        if edad_cal is None or edad_cal > 3 * 24 * 60:
            alertas.append("La calibración de la lluvia con el SMN no corre hace más de 3 días "
                           "(climaar-calibracion.yml).")

    # Todos los resultados del ciclo son de la misma imagen de radar -------------
    frame_dt = frescura.parse_utc(frame) if frame else None
    if frame_dt is not None:
        atrasados = []
        for rel, ruta, campo_gen in SINCRONIA_CICLO:
            contenido = leer(DATA / rel)
            if not isinstance(contenido, dict):
                continue  # todavía no existe (aparece con la primera tormenta)
            ts = frescura.parse_utc(_dato(contenido, ruta))
            gen = frescura.parse_utc(contenido.get(campo_gen))
            viejo = gen is None or (ahora - gen).total_seconds() / 60.0 > GENERADO_MAX_MIN
            if viejo or (ts is not None and ts != frame_dt):
                atrasados.append(rel.split("/")[-1])
        datos["sin_actualizar_en_el_ciclo"] = atrasados
        if atrasados:
            alertas.append("No se actualizaron con la imagen de radar del ciclo: " + ", ".join(atrasados) + ".")

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

    # Desde cuándo sigue cada alerta (la vigilancia avisa las que persisten).
    previas = (previo or {}).get("alertas_desde") or {}
    desde = {}
    for a in alertas:
        clave = _clave_alerta(a)
        desde[clave] = previas.get(clave) or ahora.isoformat()

    return {
        "version": VERSION,
        "en_paralelo": True,
        "generado_utc": ahora.isoformat(),
        "alertas_desde": desde,
        "estado": "ok" if not alertas else "con_alertas",
        "alertas": alertas,
        **datos,
    }


def main() -> int:
    try:
        latido = os.environ.get("CLIMAAR_DETECTOR_LATIDO")
        latido = latido.strip() if latido and latido.strip() else None
        try:
            info = json.loads(os.environ.get("CLIMAAR_DETECTOR_INFO") or "null")
        except ValueError:
            info = None
        salida = revisar(latido=latido, previo=leer(SALIDA), info_detector=info)
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
