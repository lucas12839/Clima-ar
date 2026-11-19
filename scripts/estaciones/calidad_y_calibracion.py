#!/usr/bin/env python3
"""Control de calidad de las estaciones y calibración de la lluvia del radar.

Ni el radar ni las estaciones se creen a ciegas: se controlan entre sí.

1. Calidad de cada estación (data/ia/calibracion/calidad_estaciones.json)
   - atrasada: el último dato tiene más de 30 min;
   - valores imposibles (temperatura, viento, lluvia fuera de rango físico);
   - trabada: la temperatura no cambió en 3 h mientras otras estaciones sí;
   - acumulado inconsistente: la lluvia del día baja sin que cambie el día;
   - desacuerdo con el radar: el radar ve >= 5 mm en la última hora justo
     encima y la estación marca 0, o la estación marca >= 10 mm y el radar
     no ve nada. Cada desacuerdo es una "falta"; con 2 faltas en 14 días la
     lluvia de esa estación queda "dudosa".
   Una estación dudosa no se usa para calibrar ni para los índices.

2. Registro de calibración (data/ia/calibracion/registro_lluvia.json)
   Una fila por hora y estación confiable a <= 100 km del radar: lluvia
   medida en la última hora y lluvia que estimó el radar sobre la estación
   (sin corregir). Se guardan 120 días.

3. Factor automático (data/ia/calibracion/factor_lluvia.json)
   lluvia real / lluvia del radar, con las horas en que llovió (>= 1 mm en
   la estación o en el radar). Se usa recién con >= 20 horas de >= 3 días
   distintos; mientras tanto rige el inicial (config/calibracion_lluvia.json,
   Punta Alta 13/11/2026).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[2]
EST = BASE / "data/ia/estaciones"
SENALES = EST / "senales_superficie.json"
HISTORIAL_SUP = EST / "historial_superficie.json"
NOWCAST = BASE / "data/radar/nowcast_flujo.json"
CAL = BASE / "data/ia/calibracion"
CALIDAD = CAL / "calidad_estaciones.json"
REGISTRO = CAL / "registro_lluvia.json"
FACTOR = CAL / "factor_lluvia.json"

EDAD_MAX_MIN = 30.0
DIAS_FALTAS = 14
FALTAS_DUDOSA = 2
TRABADA_HORAS = 3.0
RADAR_LLUEVE_1H_MM = 5.0
ESTACION_LLUEVE_1H_MM = 10.0
RADAR_SECO_1H_MM = 0.2
CALIBRAR_MAX_KM = 100.0
LLUVIA_MIN_CALIBRAR_MM = 1.0
HORAS_MIN = 20
DIAS_MIN = 3
REGISTRO_DIAS = 120
LIMITES = {"temperatura_c": (-20.0, 50.0), "viento_kmh": (0.0, 200.0), "rafaga_kmh": (0.0, 250.0),
           "lluvia_1h_mm": (0.0, 200.0)}


def leer(path: Path, defecto):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return defecto


def escribir(path: Path, datos) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(datos, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def parse(texto):
    try:
        t = datetime.fromisoformat(str(texto).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def dia_local(t: datetime) -> str:
    return (t - timedelta(hours=3)).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------
# Controles (funciones puras, probadas en tests/test_calidad_estaciones.py)
# --------------------------------------------------------------------------
def imposibles(est: dict[str, Any]) -> list[str]:
    problemas = []
    for campo, (lo, hi) in LIMITES.items():
        v = est.get(campo)
        if v is not None and not (lo <= float(v) <= hi):
            problemas.append(f"{campo} imposible ({v})")
    return problemas


def trabada(serie: list[dict], otras_varian: bool, ahora: datetime) -> bool:
    """Temperatura idéntica durante >= 3 h mientras otras estaciones cambian."""
    if not otras_varian:
        return False
    desde = ahora - timedelta(hours=TRABADA_HORAS)
    temps = [r.get("temp") for r in serie if (parse(r.get("utc")) or ahora) >= desde and r.get("temp") is not None]
    return len(temps) >= 12 and len(set(temps)) == 1


def acumulado_inconsistente(serie: list[dict]) -> bool:
    """La lluvia del día no puede bajar dentro del mismo día local."""
    previo = None
    for r in serie:
        t, v = parse(r.get("utc")), r.get("lluvia_dia")
        if t is None or v is None:
            continue
        if previo and previo[0] == dia_local(t) and v + 0.05 < previo[1]:
            return True
        previo = (dia_local(t), v)
    return False


def desacuerdo_lluvia(est_1h: float | None, radar_1h_sin_corregir: float | None,
                      radar_entorno_dbz: float | None, factor: float) -> str | None:
    """Falta de lluvia: radar y estación no pueden estar tan en desacuerdo."""
    if est_1h is None or radar_1h_sin_corregir is None:
        return None
    radar_1h = radar_1h_sin_corregir * factor
    if radar_1h >= RADAR_LLUEVE_1H_MM and est_1h <= 0.0:
        return f"el radar ve ~{radar_1h:.0f} mm en la última hora encima y la estación marca 0"
    if est_1h >= ESTACION_LLUEVE_1H_MM and radar_1h <= RADAR_SECO_1H_MM and (radar_entorno_dbz or 0) < 20:
        return f"la estación marca {est_1h:.0f} mm y el radar no ve lluvia cerca"
    return None


def calcular_factor(registro: list[dict]) -> dict[str, Any]:
    filas = [r for r in registro
             if r.get("confiable") and r.get("distancia_radar_km", 999) <= CALIBRAR_MAX_KM
             and max(r.get("estacion_mm", 0), r.get("radar_mm", 0)) >= LLUVIA_MIN_CALIBRAR_MM]
    dias = {r["dia_local"] for r in filas}
    total_est = sum(r["estacion_mm"] for r in filas)
    total_rad = sum(r["radar_mm"] for r in filas)
    salida = {"horas": len(filas), "dias": sorted(dias), "estacion_mm": round(total_est, 1),
              "radar_mm": round(total_rad, 1), "valido": False, "factor": None,
              "minimo_requerido": {"horas": HORAS_MIN, "dias": DIAS_MIN}}
    if len(filas) >= HORAS_MIN and len(dias) >= DIAS_MIN and total_rad > 0:
        salida["factor"] = round(total_est / total_rad, 3)
        salida["valido"] = True
    return salida


# --------------------------------------------------------------------------
def main(ahora: datetime | None = None) -> int:
    ahora = ahora or datetime.now(timezone.utc)
    senales = leer(SENALES, {})
    historial = leer(HISTORIAL_SUP, {})
    nowcast = leer(NOWCAST, {})
    puntos = nowcast.get("puntos_control") or {}
    calidad = leer(CALIDAD, {"estaciones": {}})
    registro = leer(REGISTRO, [])
    sys.path.insert(0, str(BASE / "scripts"))
    sys.path.insert(0, str(BASE))
    try:
        import localidades_radar
        factor = localidades_radar.factor_lluvia()["factor"]
    except Exception:
        factor = 1.0

    estaciones = senales.get("estaciones") or []
    # ¿Cambian las temperaturas de las otras estaciones? (para "trabada")
    variaciones = {}
    for nombre, serie in (historial or {}).items():
        temps = [r.get("temp") for r in serie[-36:] if r.get("temp") is not None]
        variaciones[nombre] = len(set(temps)) > 1
    limite = ahora - timedelta(days=DIAS_FALTAS)
    hora = ahora.replace(minute=0, second=0, microsecond=0)
    estados = calidad.get("estaciones") or {}

    for e in estaciones:
        nombre = e.get("nombre")
        if not nombre:
            continue
        previo = estados.get(nombre) or {}
        faltas = [f for f in previo.get("faltas_lluvia", []) if (parse(f.get("utc")) or ahora) >= limite]
        problemas = []
        t_obs = parse(e.get("utc"))
        if t_obs is None or (ahora - t_obs).total_seconds() / 60.0 > EDAD_MAX_MIN:
            problemas.append("dato atrasado")
        problemas += imposibles(e)
        serie = historial.get(nombre) or []
        otras = any(v for k, v in variaciones.items() if k != nombre)
        if trabada(serie, otras, ahora):
            problemas.append("temperatura trabada 3 h")
        lluvia_problemas = []
        if acumulado_inconsistente(serie):
            lluvia_problemas.append("la lluvia del día bajó sin cambiar el día")
        punto = next((p for k, p in puntos.items() if p.get("nombre") == nombre), None)
        if punto and "dato atrasado" not in problemas:
            motivo = desacuerdo_lluvia(e.get("lluvia_1h_mm"), punto.get("lluvia_1h_sin_corregir_mm"),
                                       punto.get("dbz_pico"), factor)
            if motivo and not any(f.get("hora") == iso(hora) for f in faltas):
                faltas.append({"utc": iso(ahora), "hora": iso(hora), "motivo": motivo})
        if len(faltas) >= FALTAS_DUDOSA:
            lluvia_problemas.append(f"{len(faltas)} desacuerdos con el radar en {DIAS_FALTAS} días")
        estado = "dudosa" if any(p != "dato atrasado" for p in problemas) else (
            "atrasada" if problemas else "confiable")
        estado_lluvia = "dudosa" if (lluvia_problemas or estado == "dudosa") else (
            "atrasada" if estado == "atrasada" else "confiable")
        estados[nombre] = {
            "estado": estado,
            "estado_lluvia": estado_lluvia,
            "problemas": problemas + lluvia_problemas,
            "faltas_lluvia": faltas,
            "revisado_utc": iso(ahora),
        }
        # Registro de calibración: una fila por hora y estación.
        if punto and e.get("lluvia_1h_mm") is not None and punto.get("lluvia_1h_sin_corregir_mm") is not None:
            clave = (nombre, iso(hora))
            if not any((r.get("estacion"), r.get("hora")) == clave for r in registro):
                registro.append({
                    "estacion": nombre, "hora": iso(hora), "dia_local": dia_local(hora),
                    "estacion_mm": round(float(e["lluvia_1h_mm"]), 1),
                    "radar_mm": round(float(punto["lluvia_1h_sin_corregir_mm"]), 1),
                    "distancia_radar_km": punto.get("distancia_radar_km"),
                    "confiable": estado_lluvia == "confiable",
                })

    corte = ahora - timedelta(days=REGISTRO_DIAS)
    registro = [r for r in registro if (parse(r.get("hora")) or ahora) >= corte]
    factor_auto = calcular_factor(registro)
    factor_auto.update({"actualizado_utc": iso(ahora), "factor_en_uso": factor})

    escribir(CALIDAD, {"version": "1.0.0", "actualizado_utc": iso(ahora), "estaciones": estados})
    escribir(REGISTRO, registro)
    escribir(FACTOR, factor_auto)
    dudosas = [k for k, v in estados.items() if v["estado"] == "dudosa" or v["estado_lluvia"] == "dudosa"]
    print(f"Calidad: {len(estados)} estaciones, dudosas: {dudosas or 'ninguna'}. "
          f"Calibración: {factor_auto['horas']} horas de lluvia, factor automático "
          f"{factor_auto['factor'] if factor_auto['valido'] else 'todavía no'} (en uso {factor}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
