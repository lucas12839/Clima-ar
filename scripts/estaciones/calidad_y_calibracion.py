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
   Una fila por hora y estación: lluvia medida y lluvia que estimó el radar
   sobre la estación (sin corregir), las dos en LA MISMA hora (la que termina
   en la primera imagen de radar de cada hora: la de en punto o hasta y 20). Antes la estación se medía en
   la hora que terminaba "ahora" y el radar en la que terminaba en la imagen,
   que llega 10 a 30 min tarde: las ventanas no coincidían (corregido el
   23/11/2026, esas filas viejas quedan con "ventana": "desfasada" y no se
   usan). Solo se registra si el radar tuvo la hora completa (>= 5 imágenes,
   sin huecos de más de 20 min) y la estación datos al principio y al final.
   Se guardan 120 días.

3. Factor automático (data/ia/calibracion/factor_lluvia.json)
   lluvia real / lluvia del radar, con las horas en que llovió (>= 1 mm en
   la estación o en el radar) en estaciones confiables a <= 100 km del radar.
   Se usa recién con >= 20 horas de >= 3 días distintos y >= 30 mm medidos;
   mientras tanto rige el inicial (config/calibracion_lluvia.json, Punta Alta
   13/11/2026). Se recalcula solo en cada ciclo a medida que llueve.
   También se guarda la relación por distancia al radar (informativa: lejos,
   el haz pasa por arriba de la lluvia y el radar ve menos).
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
HISTORIAL_PUNTOS = BASE / "data/radar/historial_puntos_control.json"
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
MM_MIN = 30.0
IMAGENES_HORA_MIN = 5
HUECO_RADAR_MAX_MIN = 20.0
BORDE_ESTACION_MIN = 10.0
BANDAS_KM = ((0, 50), (50, 100), (100, 160), (160, 250))
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


def escribir_compacto(path: Path, datos) -> None:
    """Registro largo (120 días): una fila por línea, sin sangría."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("[\n" + ",\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in datos)
                 + ("\n" if datos else "") + "]\n")
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


def _llovio(r: dict) -> bool:
    return (r.get("confiable") and r.get("ventana") == "alineada"
            and max(r.get("estacion_mm", 0), r.get("radar_mm", 0)) >= LLUVIA_MIN_CALIBRAR_MM)


def calcular_factor(registro: list[dict]) -> dict[str, Any]:
    filas = [r for r in registro if _llovio(r) and r.get("distancia_radar_km", 999) <= CALIBRAR_MAX_KM]
    dias = {r["dia_local"] for r in filas}
    total_est = sum(r["estacion_mm"] for r in filas)
    total_rad = sum(r["radar_mm"] for r in filas)
    salida = {"horas": len(filas), "dias": sorted(dias), "estacion_mm": round(total_est, 1),
              "radar_mm": round(total_rad, 1), "valido": False, "factor": None,
              "minimo_requerido": {"horas": HORAS_MIN, "dias": DIAS_MIN, "estacion_mm": MM_MIN}}
    if len(filas) >= HORAS_MIN and len(dias) >= DIAS_MIN and total_est >= MM_MIN and total_rad > 0:
        salida["factor"] = round(total_est / total_rad, 3)
        salida["valido"] = True
    bandas = {}
    for lo, hi in BANDAS_KM:
        f = [r for r in registro if _llovio(r) and lo <= r.get("distancia_radar_km", 999) < hi]
        e, a = sum(r["estacion_mm"] for r in f), sum(r["radar_mm"] for r in f)
        bandas[f"{lo}_{hi}_km"] = {"horas": len(f), "estacion_mm": round(e, 1), "radar_mm": round(a, 1),
                                   "relacion": round(e / a, 2) if a > 0 else None}
    salida["por_distancia"] = bandas
    return salida


def lluvia_entre(serie: list[dict], desde: datetime, hasta: datetime) -> float | None:
    """Lluvia medida por la estación entre dos horas (acumulado diario que se
    reinicia a la medianoche local). None si la estación no tiene datos cerca
    del principio y del final de la ventana."""
    margen = timedelta(minutes=BORDE_ESTACION_MIN)
    puntos = sorted((t, r.get("lluvia_dia")) for r in serie
                    if r.get("lluvia_dia") is not None and (t := parse(r.get("utc"))) is not None
                    and desde - margen <= t <= hasta + margen)
    antes = [p for p in puntos if p[0] <= desde]
    despues = [p for p in puntos if p[0] >= hasta - margen]
    if not antes or not despues:
        return None
    puntos = [antes[-1]] + [p for p in puntos if desde < p[0] <= hasta]
    if len(puntos) < 2:
        return None
    total, (ref_t, ref_v) = 0.0, puntos[0]
    for t, v in puntos[1:]:
        if v >= ref_v:
            total += v - ref_v
        elif dia_local(t) != dia_local(ref_t):
            total += v       # reinicio diario
        else:
            continue         # bajada sin medianoche = lectura errónea
        ref_t, ref_v = t, v
    return round(total, 1)


def radar_hora_completa(historial_puntos: list[dict], fin: datetime) -> bool:
    """El radar cubrió la hora (fin - 60 min, fin]: >= 5 imágenes y ningún hueco
    de más de 20 min (con huecos la suma del radar queda corta)."""
    tiempos = sorted(t for r in historial_puntos if isinstance(r, dict)
                     and (t := parse(r.get("utc"))) is not None and t <= fin)
    inicio = fin - timedelta(hours=1)
    dentro = [t for t in tiempos if t > inicio]
    if len(dentro) < IMAGENES_HORA_MIN or not dentro or dentro[-1] != fin:
        return False
    previas = [t for t in tiempos if t <= inicio]
    marcas = ([previas[-1]] if previas else [inicio]) + dentro
    return all((b - a).total_seconds() / 60.0 <= HUECO_RADAR_MAX_MIN for a, b in zip(marcas, marcas[1:]))


# --------------------------------------------------------------------------
def main(ahora: datetime | None = None) -> int:
    ahora = ahora or datetime.now(timezone.utc)
    senales = leer(SENALES, {})
    historial = leer(HISTORIAL_SUP, {})
    nowcast = leer(NOWCAST, {})
    puntos = nowcast.get("puntos_control") or {}
    calidad = leer(CALIDAD, {"estaciones": {}})
    registro = leer(REGISTRO, [])
    for r in registro:
        r.setdefault("ventana", "desfasada")   # filas de antes del 23/11/2026
    frame = parse(nowcast.get("radar_timestamp"))
    # La primera imagen de cada hora (la de en punto o, si esa se salteó, una de
    # hasta y 20) cierra una hora del registro: así las horas casi no se pisan.
    hora_frame = frame if frame is not None and frame.minute <= 20 else None
    hora_completa = hora_frame is not None and radar_hora_completa(leer(HISTORIAL_PUNTOS, []), hora_frame)
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
        # Lluvia de la estación en la misma hora que suma el radar (termina en la imagen).
        est_1h = lluvia_entre(serie, frame - timedelta(hours=1), frame) if frame else None
        if punto and "dato atrasado" not in problemas:
            motivo = desacuerdo_lluvia(est_1h if est_1h is not None else e.get("lluvia_1h_mm"),
                                       punto.get("lluvia_1h_sin_corregir_mm"), punto.get("dbz_pico"), factor)
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
        # Registro de calibración: una fila por hora y estación, estación y radar en la
        # misma hora (la que termina en la imagen de la hora en punto).
        if (hora_completa and punto and est_1h is not None
                and punto.get("lluvia_1h_sin_corregir_mm") is not None):
            h_fin = hora_frame.replace(minute=0, second=0, microsecond=0)
            clave = (nombre, iso(h_fin))
            if not any((r.get("estacion"), r.get("hora")) == clave for r in registro):
                registro.append({
                    "estacion": nombre, "hora": iso(h_fin), "dia_local": dia_local(h_fin - timedelta(minutes=30)),
                    "estacion_mm": est_1h,
                    "radar_mm": round(float(punto["lluvia_1h_sin_corregir_mm"]), 1),
                    "distancia_radar_km": punto.get("distancia_radar_km"),
                    "confiable": estado_lluvia == "confiable",
                    "ventana": "alineada",
                })

    corte = ahora - timedelta(days=REGISTRO_DIAS)
    registro = [r for r in registro if (parse(r.get("hora")) or ahora) >= corte]
    factor_auto = calcular_factor(registro)
    factor_auto.update({"actualizado_utc": iso(ahora), "factor_en_uso": factor})

    escribir(CALIDAD, {"version": "1.0.0", "actualizado_utc": iso(ahora), "estaciones": estados})
    escribir_compacto(REGISTRO, registro)
    escribir(FACTOR, factor_auto)
    dudosas = [k for k, v in estados.items() if v["estado"] == "dudosa" or v["estado_lluvia"] == "dudosa"]
    print(f"Calidad: {len(estados)} estaciones, dudosas: {dudosas or 'ninguna'}. "
          f"Calibración: {factor_auto['horas']} horas de lluvia, factor automático "
          f"{factor_auto['factor'] if factor_auto['valido'] else 'todavía no'} (en uso {factor}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
