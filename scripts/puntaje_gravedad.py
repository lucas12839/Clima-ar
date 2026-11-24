#!/usr/bin/env python3
"""Puntaje de gravedad por tormenta (punto 4 del plan), estilo NowCastMIX.

EN PARALELO (regla 5): se calcula y se guarda en cada ciclo, pero NO cambia
lo que muestra ClimaAR. Reemplaza a la clasificación actual solo cuando las
simulaciones con casos reales demuestren que acierta más.

Para cada tormenta que sigue el radar calcula el potencial (0 a 1) de
granizo, ráfagas y lluvia intensa con reglas graduales ("lógica difusa":
cada dato suma de a poco, sin cortes de sí o no), juntando:
- radar: intensidad del núcleo (p99 de dBZ), velocidad, tendencia;
- satélite (GOES-19, sincronizado por tormenta): tope frío, tope sobresaliente,
  altura del tope sobre el 0 °C, enfriamiento;
- rayos (GLM, asignados por tormenta): frecuencia, salto 2σ, rayos chicos;
- ambiente (ECMWF, punto de grilla más cercano): CAPE, cizalladura, gradiente
  térmico, aire seco, nivel de 0 °C, agua precipitable.

Reglas de seguridad:
- Sin núcleo fuerte en el radar no hay granizo ni ráfagas fuertes (el
  potencial no puede pasar mucho del aporte del radar).
- "Severo" exige una confirmación independiente del radar (satélite o rayos).
  El radar solo llega como máximo a "probable" (caso Punta Alta, 13/11).
- Eco fuerte con tope tibio y sin rayos: probable eco no meteorológico
  (sierras, tierra, mar); se marca y se baja el potencial.
- Si falta una fuente se usa el resto y baja la confianza (falla segura).

Salida: data/ia/puntaje_tormentas.json y data/ia/puntaje/AAAA-MM-DD.jsonl
(una línea por imagen de radar, para comparar con lo que mostró ClimaAR).
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))

import ambiente_severo  # noqa: E402

RADAR_FILE = BASE / "data" / "radar" / "radar_nowcast.json"
TIMESTAMP_FILE = BASE / "data" / "radar" / "radar_timestamp.txt"
GOES_FILE = BASE / "data" / "goes19" / "goes19_state.json"
METEO_FILE = BASE / "data" / "ia" / "estado_meteorologico.json"
FINAL_FILE = BASE / "data" / "ia" / "resultado_final.json"
LOCALIDADES_FILE = BASE / "config" / "localidades.json"
SALIDA = BASE / "data" / "ia" / "puntaje_tormentas.json"
HISTORIAL = BASE / "data" / "ia" / "puntaje"

VERSION = "1.0"
KM_POR_PX = 0.93933

# Pesos de cada fuente (se reparten entre las disponibles).
PESOS = {
    "granizo": {"radar": 0.40, "satelite": 0.25, "rayos": 0.20, "ambiente": 0.15},
    "rafagas": {"radar": 0.40, "satelite": 0.15, "rayos": 0.15, "ambiente": 0.30},
    "lluvia": {"radar": 0.55, "satelite": 0.05, "rayos": 0.10, "ambiente": 0.30},
}
# Cuánto puede superar el potencial al aporte del radar.
TOPE_SOBRE_RADAR = {"granizo": 0.20, "rafagas": 0.30, "lluvia": 0.25}
# Grados (misma escala de 1 a 4 que indices_fenomenos: 1 posible, 2 probable,
# 3 fuerte, 4 severo; el 4 solo si otra fuente lo confirma).
CORTES = ((0.70, 4), (0.55, 2), (0.35, 1))
CONFIRMACION = 0.5


def numero(valor: Any) -> float | None:
    try:
        x = float(valor)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def rampa(valor: float | None, desde: float, hasta: float) -> float | None:
    """0 por debajo de `desde`, 1 por encima de `hasta`, lineal en el medio."""
    if valor is None:
        return None
    if hasta == desde:
        return 1.0 if valor >= hasta else 0.0
    return max(0.0, min(1.0, (valor - desde) / (hasta - desde)))


def _media(*pares: tuple[float | None, float]) -> float | None:
    usados = [(v, w) for v, w in pares if v is not None]
    if not usados:
        return None
    total = sum(w for _, w in usados)
    return sum(v * w for v, w in usados) / total


def _max(*valores: float | None) -> float | None:
    usados = [v for v in valores if v is not None]
    return max(usados) if usados else None


def cargar(path: Path) -> dict | None:
    try:
        valor = json.loads(path.read_text(encoding="utf-8"))
        return valor if isinstance(valor, dict) else None
    except Exception:
        return None


def parse_utc(texto: Any) -> datetime | None:
    if not texto:
        return None
    try:
        t = datetime.fromisoformat(str(texto).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)


def iso(t: datetime | None) -> str | None:
    return t.isoformat().replace("+00:00", "Z") if t else None


def _dist_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


# ----------------------------------------------------------------------
# Ambiente del modelo en la tormenta
# ----------------------------------------------------------------------

def ambiente_en(meteo: dict | None, lat: float, lon: float, frame: datetime) -> dict | None:
    puntos = ((meteo or {}).get("ambiente_futuro") or {}).get("puntos") or []
    mejor = None
    for punto in puntos:
        plat, plon = numero(punto.get("latitud")), numero(punto.get("longitud"))
        if plat is None or plon is None:
            continue
        d = _dist_km(lat, lon, plat, plon)
        for registro in punto.get("datos") or []:
            t = parse_utc(registro.get("tiempo_utc"))
            if t is None:
                continue
            horas = abs((t - frame).total_seconds()) / 3600.0
            if horas > 1.5:
                continue
            clave = (round(d), horas)
            if mejor is None or clave < mejor[0]:
                mejor = (clave, registro, punto.get("punto"), d)
    if mejor is None:
        actual = (meteo or {}).get("ambiente_actual")
        return {"registro": actual, "punto": "actual", "distancia_km": None} if isinstance(actual, dict) else None
    return {"registro": mejor[1], "punto": mejor[2], "distancia_km": round(mejor[3], 1)}


# ----------------------------------------------------------------------
# Componentes por fuente
# ----------------------------------------------------------------------

def componentes(nucleo: dict, goes: dict | None, amb: dict | None) -> dict[str, Any]:
    dbz = nucleo.get("dbz") or {}
    zh = numero(dbz.get("p99")) or numero(dbz.get("max"))
    mov = nucleo.get("movimiento") or {}
    vel = (numero(mov.get("velocidad_px_min")) or 0.0) * KM_POR_PX * 60.0
    tendencia = mov.get("tendencia")

    radar = {
        "dbz_p99": zh,
        "velocidad_kmh": round(vel, 1),
        "tendencia": tendencia,
        "granizo": rampa(zh, 50.0, 65.0),
        "lluvia": rampa(zh, 40.0, 55.0),
        "rafagas": _media((rampa(zh, 45.0, 60.0), 0.6), (rampa(vel, 40.0, 80.0), 0.4)),
        "lento": 1.0 - (rampa(vel, 15.0, 50.0) or 0.0),
    }

    sat: dict[str, Any] = {"disponible": False}
    rayos: dict[str, Any] = {"disponible": False}
    sospecha = False
    if goes:
        c13 = goes.get("c13") or {}
        sev = goes.get("senales_severas") or {}
        ot = sev.get("topes_sobresalientes") or {}
        congel = sev.get("congelamiento") or {}
        if c13.get("estado") == "ok":
            p10 = numero(c13.get("p10_c"))
            tope_frio = rampa(-p10 if p10 is not None else None, 40.0, 65.0)
            ot_v = None
            if ot.get("estado") == "ok":
                ot_v = rampa(numero(ot.get("max_diferencia_yunque_c")), 5.0, 15.0) if ot.get("cantidad") else 0.0
            sobre0 = rampa(numero(congel.get("tope_sobre_congelamiento_km")), 4.0, 9.0) \
                if congel.get("estado") == "ok" else None
            enfr = rampa(numero(c13.get("enfriamiento_c_10min")), 2.0, 8.0)
            sat = {
                "disponible": True,
                "tope_p10_c": p10,
                "tope_frio": tope_frio,
                "tope_sobresaliente": ot_v,
                "sobre_0c": sobre0,
                "enfriamiento": enfr,
                "granizo": _max(ot_v, _media((tope_frio, 0.5), (sobre0, 0.5))),
                "rafagas": _max(ot_v, enfr, (tope_frio or 0.0) * 0.7 if tope_frio is not None else None),
                "lluvia": tope_frio,
            }
        r = goes.get("rayos") or {}
        if r.get("estado") == "ok":
            salto = r.get("salto_rayos") or {}
            tam = r.get("tamano_rayos") or {}
            tasa = numero(r.get("tasa_por_min"))
            if salto.get("estado") == "ok":
                v_salto = 1.0 if salto.get("salto") else (
                    (rampa(numero(salto.get("nivel_sigma")), 1.0, 3.0) or 0.0)
                    * (rampa(numero(salto.get("tasa_actual_por_min")), 2.0, 10.0) or 0.0))
            else:
                v_salto = None
            chicos = rampa(numero(tam.get("fraccion_chicos")), 0.2, 0.6) if (tam.get("cantidad") or 0) >= 5 else None
            v_tasa = rampa(tasa, 1.0, 10.0)
            organizada = _media((v_tasa, 0.6), (chicos, 0.4))
            rayos = {
                "disponible": True,
                "tasa_por_min": tasa,
                "salto": salto.get("salto"),
                "salto_valor": v_salto,
                "chicos": chicos,
                "granizo": _max(v_salto, organizada),
                "rafagas": _max(v_salto, organizada),
                "lluvia": v_tasa,
                "cantidad": r.get("cantidad"),
            }
        # Eco fuerte, tope tibio y ni un rayo: probable eco no meteorológico.
        if (sat.get("disponible") and rayos.get("disponible") and zh is not None and zh >= 45.0
                and sat.get("tope_p10_c") is not None and sat["tope_p10_c"] > -20.0
                and (rayos.get("cantidad") or 0) == 0):
            sospecha = True

    ambiente: dict[str, Any] = {"disponible": False}
    if amb and isinstance(amb.get("registro"), dict):
        reg = amb["registro"]
        ev = ambiente_severo.evaluar(reg)
        cape = rampa(numero(ev.get("cape")), 300.0, 2000.0)
        ciz = rampa(numero(ev.get("cizalladura_850_500_kmh")), 30.0, 70.0)
        grad = rampa(numero(ev.get("gradiente_850_500_c_km")), 6.0, 8.0)
        hum = numero(ev.get("humedad_media_minima_pct"))
        seco = rampa(60.0 - hum, 0.0, 35.0) if hum is not None else None
        iso0 = numero(reg.get("freezing_level_height"))
        agua = rampa(numero(reg.get("total_column_integrated_water_vapour")), 20.0, 45.0)
        ambiente = {
            "disponible": True,
            "punto": amb.get("punto"),
            "cape": numero(ev.get("cape")),
            "cizalladura_kmh": numero(ev.get("cizalladura_850_500_kmh")),
            "nivel_0c_m": iso0,
            "granizo": _media((cape, 0.5), (ciz, 0.3), (grad, 0.2)),
            "rafagas": _media((seco, 0.4), (grad, 0.3), (ciz, 0.3)),
            "lluvia": _media((agua, 0.6), (cape, 0.4)),
            # Con el 0 °C muy alto el granizo chico se derrite antes de llegar.
            "derretimiento": rampa(iso0, 3500.0, 4800.0),
        }
    return {"radar": radar, "satelite": sat, "rayos": rayos, "ambiente": ambiente,
            "sospecha_eco_no_meteorologico": sospecha}


def potencial(fen: str, comp: dict[str, Any]) -> dict[str, Any]:
    pesos = PESOS[fen]
    aportes = {}
    for fuente, peso in pesos.items():
        datos = comp[fuente]
        valor = datos.get(fen) if (fuente == "radar" or datos.get("disponible")) else None
        if valor is not None:
            aportes[fuente] = (valor, peso)
    if "radar" not in aportes:
        return {"potencial": None, "grado": 0, "confianza": 0.0, "fuentes": []}
    pot = sum(v * w for v, w in aportes.values()) / sum(w for _, w in aportes.values())
    if fen == "lluvia":
        pot = 0.85 * pot + 0.15 * comp["radar"]["lento"]
    radar_v = aportes["radar"][0]
    pot = min(pot, radar_v + TOPE_SOBRE_RADAR[fen])
    if fen == "granizo" and comp["ambiente"].get("disponible"):
        pot *= 1.0 - 0.4 * (comp["ambiente"].get("derretimiento") or 0.0)
    if comp["sospecha_eco_no_meteorologico"]:
        pot *= 0.4
    confirmado = any((aportes.get(f, (0.0, 0))[0] or 0.0) >= CONFIRMACION for f in ("satelite", "rayos"))
    if fen == "rafagas":
        confirmado = confirmado or (aportes.get("ambiente", (0.0, 0))[0] or 0.0) >= 0.6
    grado = 0
    for corte, g in CORTES:
        if pot >= corte:
            grado = g
            break
    if grado == 4 and not confirmado:
        grado = 3
    confianza = sum(w for _, w in aportes.values()) / sum(pesos.values())
    return {
        "potencial": round(pot, 3),
        "grado": grado,
        "confirmado_por_otra_fuente": confirmado,
        "confianza": round(confianza, 2),
        "fuentes": sorted(aportes),
        "aportes": {f: round(v, 3) for f, (v, _) in aportes.items()},
    }


# ----------------------------------------------------------------------
# Ciclo
# ----------------------------------------------------------------------

def calcular(radar: dict | None, frame: datetime, goes_estado: dict | None, meteo: dict | None,
             final: dict | None = None, localidades: list[dict] | None = None) -> dict[str, Any]:
    base = {"version": VERSION, "frame_radar_utc": iso(frame), "modo": "en_paralelo_no_se_muestra",
            "generado_utc": iso(datetime.now(timezone.utc))}
    if not isinstance(radar, dict) or parse_utc(radar.get("actualizado")) != frame:
        return {**base, "estado": "radar_de_otro_frame", "tormentas": [], "localidades": {}}
    nucleos = (radar.get("nowcast") or {}).get("nucleos_georreferenciados") or []
    goes_ok = isinstance(goes_estado, dict) and parse_utc(goes_estado.get("reference_frame_utc")) == frame
    goes_por_track = {}
    if goes_ok:
        for t in ((goes_estado.get("por_tormenta") or {}).get("tormentas") or []):
            goes_por_track[str(t.get("track_id"))] = t
    tormentas = []
    for n in nucleos:
        geo = n.get("georreferencia") or {}
        lat, lon = numero(geo.get("latitud")), numero(geo.get("longitud"))
        if lat is None or lon is None:
            continue
        comp = componentes(n, goes_por_track.get(str(n.get("track_id"))), ambiente_en(meteo, lat, lon, frame))
        radio = math.sqrt(max(numero(n.get("pixeles")) or 0.0, 0.0) * KM_POR_PX ** 2 / math.pi) + 5.0
        tormentas.append({
            "track_id": n.get("track_id"),
            "latitud": round(lat, 4),
            "longitud": round(lon, 4),
            "radio_km": round(radio, 1),
            "granizo": potencial("granizo", comp),
            "rafagas": potencial("rafagas", comp),
            "lluvia_intensa": potencial("lluvia", comp),
            "sospecha_eco_no_meteorologico": comp["sospecha_eco_no_meteorologico"],
            "componentes": comp,
        })

    # Por localidad: tormentas que hoy cubren la localidad (comparación con
    # lo que muestra ClimaAR; la llegada futura es del punto 6).
    if localidades is None:
        localidades = (cargar(LOCALIDADES_FILE) or {}).get("localidades") or []
    mostrado = (((final or {}).get("tipo_tiempo") or {}).get("localidades")) or {}
    por_loc = {}
    for loc in localidades:
        encima = [t for t in tormentas
                  if _dist_km(loc["lat"], loc["lon"], t["latitud"], t["longitud"]) <= t["radio_km"]]
        actual = ((mostrado.get(loc["id"]) or {}).get("ahora")) or {}
        por_loc[loc["id"]] = {
            "tormentas": [t["track_id"] for t in encima],
            "granizo_grado": max((t["granizo"]["grado"] for t in encima), default=0),
            "rafagas_grado": max((t["rafagas"]["grado"] for t in encima), default=0),
            "lluvia_intensa_grado": max((t["lluvia_intensa"]["grado"] for t in encima), default=0),
            "mostrado_granizo_grado": (actual.get("granizo") or {}).get("grado"),
            "mostrado_viento_grado": (actual.get("viento") or {}).get("grado"),
            "mostrado_lluvia_grado": (actual.get("lluvia") or {}).get("grado"),
        }
    return {**base, "estado": "ok" if tormentas else "sin_tormentas",
            "goes_sincronizado": goes_ok, "tormentas": tormentas, "localidades": por_loc}


def _historial(salida: dict) -> str:
    if salida.get("estado") != "ok":
        return "sin tormentas: no se guarda"
    frame = parse_utc(salida["frame_radar_utc"])
    HISTORIAL.mkdir(parents=True, exist_ok=True)
    path = HISTORIAL / f"{frame:%Y-%m-%d}.jsonl"
    texto = f'"frame_radar_utc": "{salida["frame_radar_utc"]}"'
    if path.exists() and texto in path.read_text(encoding="utf-8")[-300_000:]:
        return "ya guardado"
    linea = {
        "frame_radar_utc": salida["frame_radar_utc"],
        "goes_sincronizado": salida.get("goes_sincronizado"),
        "tormentas": [{
            "track_id": t["track_id"], "lat": t["latitud"], "lon": t["longitud"],
            "granizo": [t["granizo"]["potencial"], t["granizo"]["grado"]],
            "rafagas": [t["rafagas"]["potencial"], t["rafagas"]["grado"]],
            "lluvia": [t["lluvia_intensa"]["potencial"], t["lluvia_intensa"]["grado"]],
            "sospecha": t["sospecha_eco_no_meteorologico"],
            "confianza": t["granizo"]["confianza"],
        } for t in salida["tormentas"]],
        "localidades": salida["localidades"],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(linea, ensure_ascii=False, separators=(", ", ": ")) + "\n")
    return f"guardado en {path.name}"


def main() -> int:
    try:
        frame = parse_utc(TIMESTAMP_FILE.read_text(encoding="utf-8").strip())
        salida = calcular(cargar(RADAR_FILE), frame, cargar(GOES_FILE), cargar(METEO_FILE), cargar(FINAL_FILE))
        SALIDA.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Puntaje de gravedad (en paralelo, no se muestra): {salida['estado']}, "
              f"{len(salida['tormentas'])} tormentas, satélite sincronizado: {salida.get('goes_sincronizado')}")
        for t in salida["tormentas"]:
            print(f"  tormenta {t['track_id']}: granizo {t['granizo']['grado']} ({t['granizo']['potencial']}), "
                  f"ráfagas {t['rafagas']['grado']} ({t['rafagas']['potencial']}), "
                  f"lluvia {t['lluvia_intensa']['grado']} ({t['lluvia_intensa']['potencial']}), "
                  f"confianza {t['granizo']['confianza']}"
                  + (" | SOSPECHA eco no meteorológico" if t["sospecha_eco_no_meteorologico"] else ""))
        print(f"Historial: {_historial(salida)}")
    except Exception as exc:  # falla segura: nunca corta el ciclo
        print(f"AVISO puntaje de gravedad: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
