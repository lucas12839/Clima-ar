#!/usr/bin/env python3
"""Etapa B, punto 9: informe mensual de aciertos, automático.

Junta en un solo lugar, mes por mes, lo que ClimaAR ya verifica solo en cada
ciclo (no recalcula nada, solo lee lo guardado):

- nowcast (data/ia/verificacion_nowcast.json): acierto CSI con lluvia (20 dBZ)
  y tormenta (35 dBZ) por plazo, contra quedarse con la imagen sin mover; el
  corrector con estaciones de la etapa A (en paralelo); el modo sin radar
  (pronósticos hechos con el radar de 40 min o más);
- llegadas por localidad en la próxima hora (detección y falsas alarmas);
- alarma "ya llega" por estaciones (data/ia/estaciones/alarma_llegada.json);
- puntaje de gravedad contra lo medido (data/ia/verificacion_puntaje.json);
- calibración de la lluvia del radar (data/ia/calibracion/factor_lluvia.json).

Los contadores de esos archivos se acumulan desde que empezaron. Para sacar el
mes, al empezar cada mes (hora de Argentina) se guarda una foto de los
contadores (data/ia/informe/base_mensual.json) y el informe es lo de ahora
menos esa foto. El primer mes cuenta desde el inicio de las mediciones.

Escribe solo lo suyo: data/ia/informe/base_mensual.json, AAAA-MM.json y
AAAA-MM.md (para leer en el celular, en GitHub). El informe del mes en curso se
actualiza en cada ciclo; al cambiar el mes queda cerrado el anterior.
EN PARALELO: no cambia lo que se ve. Falla segura: nunca corta el ciclo.
"""
from __future__ import annotations

import copy
import json
import os
import statistics
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[1]
IA = BASE / "data" / "ia"
VERIF_NOWCAST = IA / "verificacion_nowcast.json"
ALARMA = IA / "estaciones" / "alarma_llegada.json"
VERIF_PUNTAJE = IA / "verificacion_puntaje.json"
FACTOR = IA / "calibracion" / "factor_lluvia.json"
DIR = IA / "informe"
BASE_MENSUAL = DIR / "base_mensual.json"
VERSION = "1.0.0"
ARG = timezone(timedelta(hours=-3))
METODOS = {"capas": "nowcast (lo que se ve)", "hoy": "imagen sin mover (para comparar)",
           "estaciones": "nowcast + corrector con estaciones (en paralelo, empezó después: menos casos)"}
PLAZOS = ("hasta_10min", "hasta_20min", "hasta_30min", "hasta_40min", "hasta_60min", "hasta_90min",
          "sin_radar_40min")
NOMBRE_PLAZO = {"sin_radar_40min": "sin radar (radar de 40 min o más)"}


def leer(path: Path, defecto):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return defecto


def escribir(path: Path, texto: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(texto)
    os.replace(tmp, path)


def escribir_json(path: Path, datos) -> None:
    escribir(path, json.dumps(datos, ensure_ascii=False, indent=1))


# --------------------------------------------------------------------------
# Contadores (funciones puras, tests/test_informe_aciertos.py)
# --------------------------------------------------------------------------
def contadores() -> dict[str, Any]:
    """Foto de todos los contadores acumulados que se usan en el informe."""
    v = leer(VERIF_NOWCAST, {})
    a = leer(ALARMA, {}).get("acumulado") or {}
    return {
        "nowcast": copy.deepcopy(v.get("acumulado") or {}),
        "llegadas": copy.deepcopy(v.get("llegadas_acumulado") or {}),
        "alarma": {"acierto": int(a.get("acierto") or 0), "falsa_alarma": int(a.get("falsa_alarma") or 0),
                   "errores_min": list(a.get("errores_min") or [])},
    }


def _restar_lista(a: list, b: list) -> list:
    return [x - y for x, y in zip(a, b + [0] * (len(a) - len(b)))]


def restar(actual: dict, base: dict) -> dict:
    """Lo del mes = lo acumulado ahora - la foto del principio del mes. Si un
    contador volvió atrás (se reinició el archivo), se toma desde cero."""
    salida = {"nowcast": {}, "llegadas": {}, "alarma": {}}
    for clave, acc in actual.get("nowcast", {}).items():
        b = base.get("nowcast", {}).get(clave) or {}
        fila = {}
        for u in ("20", "35"):
            d = _restar_lista(acc.get(u, [0] * 4), b.get(u, [0] * 4))
            fila[u] = d if min(d) >= 0 else list(acc.get(u, [0] * 4))
        n = acc.get("n", 0) - b.get("n", 0)
        err = acc.get("error_abs", 0.0) - b.get("error_abs", 0.0)
        fila["n"], fila["error_abs"] = (n, err) if n >= 0 and err >= 0 else (acc.get("n", 0), acc.get("error_abs", 0.0))
        salida["nowcast"][clave] = fila
    for clave, acc in actual.get("llegadas", {}).items():
        d = _restar_lista(acc, base.get("llegadas", {}).get(clave) or [0, 0, 0, 0])
        salida["llegadas"][clave] = d if min(d) >= 0 else list(acc)
    al, bl = actual.get("alarma", {}), base.get("alarma", {})
    ac, fa = al.get("acierto", 0) - bl.get("acierto", 0), al.get("falsa_alarma", 0) - bl.get("falsa_alarma", 0)
    if ac < 0 or fa < 0:
        ac, fa = al.get("acierto", 0), al.get("falsa_alarma", 0)
    errores = al.get("errores_min") or []
    salida["alarma"] = {"acierto": ac, "falsa_alarma": fa, "errores_min": errores[-ac:] if ac else []}
    return salida


def csi(t: list[int]) -> float | None:
    a, b, c, _ = t
    return round(a / (a + b + c), 3) if a + b + c else None


def armar(mes: str, delta: dict, puntaje: dict, factor: dict, desde: str, ahora: datetime) -> dict:
    nowcast = {}
    for clave, acc in delta["nowcast"].items():
        metodo, plazo = clave.split("|", 1)
        nowcast.setdefault(metodo, {})[plazo] = {
            "csi_lluvia_20dbz": csi(acc["20"]), "casos_lluvia": sum(acc["20"][:3]),
            "csi_tormenta_35dbz": csi(acc["35"]), "casos_tormenta": sum(acc["35"][:3]),
            "error_medio_dbz": round(acc["error_abs"] / acc["n"], 1) if acc.get("n") else None}
    llegadas = {}
    for clave, (a, b, c, _) in sorted(delta["llegadas"].items()):
        llegadas[clave] = {"deteccion_pct": round(100 * a / (a + c), 1) if a + c else None,
                           "falsas_alarmas_pct": round(100 * b / (a + b), 1) if a + b else None,
                           "llegadas": a + c, "avisos": a + b}
    al = delta["alarma"]
    n = al["acierto"] + al["falsa_alarma"]
    alarma = {"verificadas": n, "aciertos_pct": round(100 * al["acierto"] / n, 1) if n else None,
              "falsas_alarmas_pct": round(100 * al["falsa_alarma"] / n, 1) if n else None,
              "error_mediano_min": statistics.median(al["errores_min"]) if al["errores_min"] else None}
    return {
        "version": VERSION, "mes": mes, "desde": desde, "actualizado_utc": ahora.isoformat(),
        "nowcast": nowcast, "llegadas": llegadas, "alarma_ya_llega": alarma,
        "puntaje_gravedad": (puntaje.get("resumen_por_mes") or {}).get(mes) or {},
        "calibracion_lluvia": {k: factor.get(k) for k in ("factor_en_uso", "valido", "factor", "horas", "dias",
                                                          "estacion_mm", "radar_mm", "minimo_requerido",
                                                          "por_distancia")},
    }


def _n(v, suf="") -> str:
    """Número con coma decimal (como se escribe en Argentina)."""
    if v is None:
        return "—"
    texto = f"{v:.2f}" if isinstance(v, float) and v < 1.5 and suf == "" else f"{v:g}"
    return texto.replace(".", ",") + suf


def markdown(inf: dict) -> str:
    """El informe para leer en el celular (en español, horarios de Argentina)."""
    act = datetime.fromisoformat(inf["actualizado_utc"]).astimezone(ARG).strftime("%d/%m/%Y %H:%M")
    l = [f"# ClimaAR: informe de aciertos de {inf['mes']}", "",
         f"Actualizado: {act} (hora de Argentina). Cuenta desde: {inf['desde']}.",
         "Todo esto corre **en paralelo**: se mide, no cambia lo que se ve.", "",
         "## 1. Nowcast (¿dónde llueve y dónde hay tormenta?)", "",
         "Acierto CSI: 1 = perfecto, 0 = nada. Plazo = minutos desde la imagen de radar.", "",
         "| Plazo | Método | Lluvia | Tormenta | Casos tormenta |", "|---|---|---|---|---|"]
    for plazo in PLAZOS:
        for metodo in ("capas", "hoy", "estaciones"):
            f = (inf["nowcast"].get(metodo) or {}).get(plazo)
            if f and (f["casos_lluvia"] or f["casos_tormenta"]):
                l.append(f"| {NOMBRE_PLAZO.get(plazo, plazo.replace('hasta_', 'hasta ').replace('min', ' min'))} "
                         f"| {METODOS[metodo]} | {_n(f['csi_lluvia_20dbz'])} | {_n(f['csi_tormenta_35dbz'])} "
                         f"| {f['casos_tormenta']} |")
    l += ["", "## 2. Llegada a cada localidad en la próxima hora", "",
          "| Qué | Grado | Zona | Detectadas | Falsas alarmas | Llegadas |", "|---|---|---|---|---|---|"]
    for clave, f in inf["llegadas"].items():
        fen, grado, zona = clave.split("|")
        l.append(f"| {fen} | {grado} | {'localidad' if zona == 'localidad' else 'hasta 10 km'} "
                 f"| {_n(f['deteccion_pct'], ' %')} | {_n(f['falsas_alarmas_pct'], ' %')} | {f['llegadas']} |")
    a = inf["alarma_ya_llega"]
    l += ["", "## 3. Alarma \"ya llega\" por estaciones", "",
          f"Verificadas: {a['verificadas']}. Aciertos: {_n(a['aciertos_pct'], ' %')}. "
          f"Falsas alarmas: {_n(a['falsas_alarmas_pct'], ' %')}. "
          f"Error mediano de la hora de llegada: {_n(a['error_mediano_min'], ' min')}.", "",
          "## 4. Puntaje de gravedad contra lo medido", "",
          "Estaciones a 10 km o menos de cada tormenta, desde 10 min antes hasta 40 min después de la imagen.",
          "El granizo no se puede verificar: no hay mediciones de granizo.", "",
          "| Fenómeno | Grado | Casos | Medido | Promedio | ≥ fuerte | ≥ severo |", "|---|---|---|---|---|---|---|"]
    nombres = {"viento_kmh": ("viento (km/h)", "60", "90"), "lluvia_mm": ("lluvia (mm)", "10", "20"),
               "rayos": ("rayos en 30 min", "1", "10")}
    for fen, grados in (inf["puntaje_gravedad"] or {}).items():
        for grado, f in grados.items():
            for campo, (nom, bajo, alto) in nombres.items():
                m = f.get(campo)
                if m:
                    l.append(f"| {fen} | {grado} | {m['casos']} | {nom} | {_n(m['promedio'])} "
                             f"| {_n(m.get('pct_ge_' + bajo), ' %')} (≥{bajo}) | {_n(m.get('pct_ge_' + alto), ' %')} (≥{alto}) |")
    if not inf["puntaje_gravedad"]:
        l.append("| — | — | 0 | todavía no hubo tormentas encima de una estación | | | |")
    c = inf["calibracion_lluvia"]
    req = c.get("minimo_requerido") or {}
    l += ["", "## 5. Calibración de la lluvia del radar", "",
          f"Factor en uso: {_n(c.get('factor_en_uso'))} "
          f"({'calculado con las estaciones' if c.get('valido') else 'el inicial, todavía faltan datos'}). "
          f"Horas con lluvia juntadas: {c.get('horas') or 0} de {req.get('horas', '?')}, "
          f"días: {len(c.get('dias') or [])} de {req.get('dias', '?')}, "
          f"lluvia medida: {_n(c.get('estacion_mm'), ' mm')} de {req.get('estacion_mm', '?')} mm.", ""]
    bandas = c.get("por_distancia") or {}
    if bandas:
        l += ["| Distancia al radar | Horas | Estación | Radar sin corregir | Relación |", "|---|---|---|---|---|"]
        for k, f in bandas.items():
            lo, hi, _ = k.split("_")
            l.append(f"| {lo} a {hi} km | {f['horas']} | {_n(f['estacion_mm'], ' mm')} "
                     f"| {_n(f['radar_mm'], ' mm')} | {_n(f['relacion'])} |")
    return "\n".join(l) + "\n"


def main(ahora: datetime | None = None) -> int:
    try:
        ahora = ahora or datetime.now(timezone.utc)
        mes = ahora.astimezone(ARG).strftime("%Y-%m")
        actual = contadores()
        base = leer(BASE_MENSUAL, {})
        if base.get("mes") != mes:
            if base.get("mes"):
                # Cerrar el mes anterior con lo juntado hasta ahora.
                previo = base["mes"]
                inf = armar(previo, restar(actual, base.get("contadores") or {}), leer(VERIF_PUNTAJE, {}),
                            leer(FACTOR, {}), base.get("desde", "?"), ahora)
                inf["cerrado"] = True
                escribir_json(DIR / f"{previo}.json", inf)
                escribir(DIR / f"{previo}.md", markdown(inf))
                base = {"mes": mes, "desde": ahora.astimezone(ARG).strftime("%d/%m/%Y %H:%M"),
                        "contadores": actual}
            else:
                base = {"mes": mes, "desde": "el inicio de las mediciones", "contadores": {}}
            escribir_json(BASE_MENSUAL, base)
        inf = armar(mes, restar(actual, base.get("contadores") or {}), leer(VERIF_PUNTAJE, {}),
                    leer(FACTOR, {}), base.get("desde", "?"), ahora)
        inf["cerrado"] = False
        escribir_json(DIR / f"{mes}.json", inf)
        escribir(DIR / f"{mes}.md", markdown(inf))
        print(f"Informe de aciertos {mes} actualizado.")
    except Exception as exc:  # nunca corta el ciclo
        print(f"::warning::No se pudo armar el informe de aciertos: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
