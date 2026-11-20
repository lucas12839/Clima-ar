#!/usr/bin/env python3
"""Punto 8: ambiente medido por el satélite antes de que se nuble (en paralelo).

Para la imagen de radar del ciclo lee de GOES-19 (solo la zona de 240 km):
inestabilidad (CAPE, índice de elevación, K, Total Totals, Showalter), agua
precipitable y perfiles de temperatura y humedad. Con eso calcula, por
localidad y para toda la región, el ambiente MEDIDO y lo compara con el modelo
ECMWF que ya usa ClimaAR.

"Antes de que se nuble": el satélite solo mide con cielo despejado. Cuando una
localidad se nubla, se conserva su última medición despejada (hasta 6 h) con
su hora y su edad: es justamente el ambiente en el que se forman las tormentas.

Reglas de confiabilidad: mismo barrido que el canal 13 del ciclo (o el más
reciente anterior a la imagen de radar, hasta 20 min antes); nunca un dato
posterior al radar; si falta un producto se dice y se sigue con los demás.
En los refrescos entre imágenes se reutiliza lo ya leído (no se baja de nuevo).

Escribe data/ia/ambiente_satelite.json. No cambia lo que se ve y nunca corta
el ciclo.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

VERSION = "1.0.0"
SALIDA = BASE / "data" / "ia" / "ambiente_satelite.json"
RADAR_TS = BASE / "data" / "radar" / "radar_timestamp.txt"
GOES_STATE = BASE / "data" / "goes19" / "goes19_state.json"
METEO = BASE / "data" / "ia" / "estado_meteorologico.json"
LOCALIDADES = BASE / "config" / "localidades.json"

RADIO_LOCALIDAD_KM = 25.0      # 10 km por píxel: ~20 píxeles por localidad
MEMORIA_MAX_MIN = 360.0        # última medición despejada: hasta 6 h
GRUPOS = {
    "inestabilidad": ("cape_jkg", "cape_max_jkg", "indice_elevacion_c", "indice_k_c",
                      "total_totals_c", "showalter_c"),
    "agua_precipitable": ("agua_precipitable_mm",),
    "perfiles": ("perfil",),
}


def leer_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_utc(texto: Any) -> datetime | None:
    if not texto:
        return None
    try:
        v = datetime.fromisoformat(str(texto).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return (v if v.tzinfo else v.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def iso(v: datetime | None) -> str | None:
    return v.isoformat().replace("+00:00", "Z") if v else None


# ------------------------------------------------------------------
# Archivos del satélite
# ------------------------------------------------------------------

def buscar_archivos(frame: datetime, inicio_barrido: str | None) -> dict[str, dict]:
    """Archivo de cada producto: el del mismo barrido que el canal 13 del
    ciclo; si todavía no está publicado, el más reciente anterior al radar."""
    from goes19 import ambiente as amb
    from goes19 import goes19_engine as ge

    salida = {}
    for nombre, (prefijo, _vars) in amb.PRODUCTOS.items():
        clave = f"amb_{nombre}"
        ge.PRODUCT_SPECS.setdefault(clave, (prefijo, None))
        try:
            obj = ge.product_object_for_scan(clave, inicio_barrido, now=frame) if inicio_barrido else None
            mismo = obj is not None
            if obj is None:
                obj = ge.latest_product_object(clave, now=frame, hours_back=1)
            salida[nombre] = {"obj": obj, "mismo_barrido_que_c13": mismo} if obj else {"error": "sin archivo"}
        except Exception as exc:
            salida[nombre] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    return salida


def leer_productos(archivos: dict[str, dict], frame: datetime) -> tuple[dict, dict]:
    """Lee la ventana de cada producto sincronizado. Devuelve (ventanas, estado)."""
    from goes19 import ambiente as amb
    from goes19 import por_tormenta as pt

    ventanas, estado = {}, {}
    for nombre, info in archivos.items():
        if "obj" not in info:
            estado[nombre] = {"estado": "sin_datos", "motivo": info.get("error")}
            continue
        obj = info["obj"]
        hora = parse_utc(obj.get("end_time_utc"))
        sinc = pt.sincronizacion(hora, frame)
        e = {
            "archivo": obj.get("key", "").split("/")[-1],
            "inicio_barrido_utc": obj.get("start_time_utc"),
            "fin_barrido_utc": obj.get("end_time_utc"),
            "mismo_barrido_que_c13": info.get("mismo_barrido_que_c13", False),
            **{k: v for k, v in sinc.items() if k != "desfase_max_min"},
        }
        if sinc["estado"] != "sincronizado":
            e["estado"] = "no_sincronizado"
            e["motivo"] = sinc["estado"]
            estado[nombre] = e
            continue
        inicio = datetime.now(timezone.utc)
        try:
            ventanas[nombre] = amb.leer_ventana(obj["url"], amb.PRODUCTOS[nombre][1])
            e["estado"] = "ok"
            e["pixeles_despejados"] = int(amb.buenos(ventanas[nombre]).sum())
            e["pixeles_zona"] = int(ventanas[nombre]["dqf"].size)
        except Exception as exc:
            e["estado"] = "error"
            e["motivo"] = f"{type(exc).__name__}: {exc}"[:200]
        e["segundos_lectura"] = round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)
        estado[nombre] = e
    return ventanas, estado


# ------------------------------------------------------------------
# Memoria: última medición despejada
# ------------------------------------------------------------------

PRODUCTOS_GRUPO = {
    "inestabilidad": ("inestabilidad",),
    "agua_precipitable": ("agua_precipitable",),
    "perfiles": ("perfil_temperatura", "perfil_humedad"),
}


def horas_por_grupo(estado: dict, defecto: datetime) -> dict[str, datetime]:
    salida = {}
    for grupo, productos in PRODUCTOS_GRUPO.items():
        hs = [parse_utc(estado.get(p, {}).get("fin_barrido_utc")) for p in productos
              if estado.get(p, {}).get("estado") == "ok"]
        hs = [h for h in hs if h]
        salida[grupo] = max(hs) if hs else defecto
    return salida


def con_memoria(actual: dict, previo: dict | None, hora: datetime | dict, frame: datetime) -> dict:
    """Une lo medido ahora con la última medición despejada de cada grupo.
    `hora`: hora del barrido (una para todos o {grupo: hora})."""
    previo = previo or {}
    memoria_previa = previo.get("ultimo_despejado") or {}
    memoria = {}
    usados: dict[str, Any] = {"pixeles_despejados": actual.get("pixeles_despejados", {})}
    origen = {}
    for grupo, campos in GRUPOS.items():
        valores = {c: actual[c] for c in campos if actual.get(c) is not None}
        if valores:
            h_grupo = hora.get(grupo, frame) if isinstance(hora, dict) else hora
            memoria[grupo] = {"hora_utc": iso(h_grupo), "valores": valores}
        else:
            m = memoria_previa.get(grupo)
            h = parse_utc((m or {}).get("hora_utc"))
            if m and h and h <= frame and (frame - h).total_seconds() / 60.0 <= MEMORIA_MAX_MIN:
                memoria[grupo] = m
        if grupo in memoria:
            h = parse_utc(memoria[grupo]["hora_utc"])
            edad = round((frame - h).total_seconds() / 60.0, 0)
            usados.update(memoria[grupo]["valores"])
            origen[grupo] = "ahora" if valores else f"ultima_medicion_despejada_hace_{int(edad)}_min"
    for grupo in memoria:
        h = parse_utc(memoria[grupo]["hora_utc"])
        memoria[grupo] = dict(memoria[grupo], edad_min=round((frame - h).total_seconds() / 60.0, 0))
    return {"ahora": actual, "ultimo_despejado": memoria, "origen": origen, "valores": usados}


# ------------------------------------------------------------------
# Comparación con el modelo (ECMWF del ciclo, punto de Bahía Blanca)
# ------------------------------------------------------------------

def registro_modelo(meteo: dict, hora: datetime) -> dict | None:
    puntos = ((meteo or {}).get("ambiente_futuro") or {}).get("puntos") or []
    centro = next((p for p in puntos if p.get("punto") == "BB_CENTRO"), None)
    mejor, dif = None, None
    for r in (centro or {}).get("datos") or []:
        t = parse_utc(r.get("tiempo_utc"))
        if t is None:
            continue
        d = abs((t - hora).total_seconds())
        if d <= 5400 and (dif is None or d < dif):
            mejor, dif = r, d
    return mejor


def comparar(medido: dict, modelo: dict | None) -> dict[str, Any]:
    """Medido por el satélite (Bahía Blanca) contra el ECMWF a la misma hora."""
    if not modelo:
        return {"estado": "sin_modelo"}
    perfil = medido.get("perfil") or {}
    pares = {
        "cape_jkg": (medido.get("cape_jkg"), modelo.get("cape")),
        "agua_precipitable_mm": (medido.get("agua_precipitable_mm"),
                                 modelo.get("total_column_integrated_water_vapour")),
        "temperatura_850_c": (perfil.get("temperatura_850_c"), modelo.get("temperature_850hPa")),
        "temperatura_700_c": (perfil.get("temperatura_700_c"), modelo.get("temperature_700hPa")),
        "temperatura_500_c": (perfil.get("temperatura_500_c"), modelo.get("temperature_500hPa")),
        "humedad_700_pct": (perfil.get("humedad_700_pct"), modelo.get("relative_humidity_700hPa")),
        "humedad_500_pct": (perfil.get("humedad_500_pct"), modelo.get("relative_humidity_500hPa")),
        "nivel_0c_m": ((perfil.get("nivel_0c") or {}).get("altura_m"), modelo.get("freezing_level_height")),
    }
    salida: dict[str, Any] = {"estado": "ok", "hora_modelo_utc": modelo.get("tiempo_utc"), "variables": {}}
    for k, (sat, mod) in pares.items():
        if sat is None or mod is None:
            continue
        salida["variables"][k] = {"satelite": sat, "modelo": mod, "diferencia": round(float(sat) - float(mod), 1)}
    v = salida["variables"]
    lectura = []
    if "agua_precipitable_mm" in v:
        d = v["agua_precipitable_mm"]["diferencia"]
        if abs(d) >= 3:
            lectura.append(f"el satélite mide {'más' if d > 0 else 'menos'} agua en la columna que el modelo ({d:+.1f} mm)")
    if "cape_jkg" in v:
        d = v["cape_jkg"]["diferencia"]
        if abs(d) >= 300:
            lectura.append(f"el satélite mide {'más' if d > 0 else 'menos'} energía para tormentas que el modelo ({d:+.0f} J/kg)")
    if "temperatura_500_c" in v:
        d = v["temperatura_500_c"]["diferencia"]
        if abs(d) >= 2:
            lectura.append(f"en 500 hPa el aire medido es {abs(d):.1f} °C más {'cálido' if d > 0 else 'frío'} que en el modelo")
    salida["lectura"] = lectura or ["el modelo coincide con lo medido"]
    if not v:
        salida["estado"] = "sin_medicion_despejada"
    return salida


# ------------------------------------------------------------------
# Ciclo
# ------------------------------------------------------------------

def calcular(frame: datetime, inicio_barrido: str | None, previo: dict | None,
             meteo: dict | None, localidades: list[dict], ahora: datetime | None = None) -> dict:
    from goes19 import ambiente as amb

    ahora = ahora or datetime.now(timezone.utc)
    archivos = buscar_archivos(frame, inicio_barrido)
    claves = {k: (v.get("obj") or {}).get("key") for k, v in archivos.items()}
    previo = previo if isinstance(previo, dict) else {}

    # Refresco entre imágenes: mismos archivos -> no se baja nada de nuevo.
    if (previo.get("frame_radar_utc") == iso(frame) and previo.get("claves_archivos") == claves
            and all(claves.values())):
        return dict(previo, generado_utc=iso(ahora), reutilizado=True)

    ventanas, estado = leer_productos(archivos, frame)
    fines = [parse_utc(e.get("fin_barrido_utc")) for e in estado.values() if e.get("estado") == "ok"]
    hora = max([h for h in fines if h], default=None)

    prev_locs = previo.get("localidades") or {}
    if ventanas and hora:
        actual_locs = amb.por_localidades(ventanas, localidades, RADIO_LOCALIDAD_KM)
        actual_region = amb.region(ventanas)
    else:
        actual_locs = {loc["id"]: {"pixeles_despejados": {}} for loc in localidades}
        actual_region = {"pixeles_despejados": {}}
    ref = hora or frame
    horas = horas_por_grupo(estado, ref)
    locs = {}
    for loc in localidades:
        r = con_memoria(actual_locs.get(loc["id"], {}), prev_locs.get(loc["id"]), horas, frame)
        r["lectura"] = amb.calificar(r["valores"])
        r["nombre"] = loc.get("nombre")
        locs[loc["id"]] = r
    reg = con_memoria(actual_region, previo.get("region"), horas, frame)
    reg["lectura"] = amb.calificar(reg["valores"])
    for k in ("fraccion_despejada", "mas_inestable", "cape_p90_jkg", "indice_elevacion_p10_c",
              "agua_precipitable_p90_mm"):
        if k in actual_region:
            reg[k] = actual_region[k]

    bahia = locs.get("bahia_blanca", {}).get("valores", {})
    disponible = any(e.get("estado") == "ok" for e in estado.values())
    return {
        "version": VERSION,
        "en_paralelo": True,
        "disponible": disponible,
        "motivo": None if disponible else "sin productos del satélite sincronizados en este ciclo",
        "generado_utc": iso(ahora),
        "frame_radar_utc": iso(frame),
        "hora_satelite_utc": iso(hora),
        "reutilizado": False,
        "claves_archivos": claves,
        "productos": estado,
        "radio_localidad_km": RADIO_LOCALIDAD_KM,
        "region": reg,
        "localidades": locs,
        "comparacion_modelo_bahia": comparar(bahia, registro_modelo(meteo, ref)),
        "nota": ("Ambiente MEDIDO por GOES-19 (solo cielo despejado). Con nubes se usa la última "
                 "medición despejada (hasta 6 h) y se indica su edad. Lectura de reglas iniciales "
                 "EN EVALUACIÓN; no cambia lo que se ve."),
    }


def limpiar(v: Any) -> Any:
    """Sin NaN ni tipos de numpy en el JSON."""
    import math
    if isinstance(v, dict):
        return {k: limpiar(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [limpiar(x) for x in v]
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def main() -> int:
    try:
        texto = RADAR_TS.read_text(encoding="utf-8").strip() if RADAR_TS.exists() else ""
        frame = parse_utc(texto)
        if frame is None:
            raise RuntimeError("sin imagen de radar del ciclo")
        goes = leer_json(GOES_STATE) or {}
        inicio = (goes.get("sincronizacion") or {}).get("inicio_barrido_referencia_utc")
        locs = (leer_json(LOCALIDADES) or {}).get("localidades") or []
        salida = calcular(frame, inicio, leer_json(SALIDA), leer_json(METEO), locs)
    except Exception as exc:  # nunca corta el ciclo
        previo = leer_json(SALIDA) or {}
        salida = {
            "version": VERSION, "en_paralelo": True, "disponible": False,
            "motivo": f"{type(exc).__name__}: {exc}"[:300],
            "generado_utc": iso(datetime.now(timezone.utc)),
            "frame_radar_utc": previo.get("frame_radar_utc"),
            "localidades": previo.get("localidades") or {},
        }
    try:
        SALIDA.parent.mkdir(parents=True, exist_ok=True)
        tmp = SALIDA.with_suffix(".json.tmp")
        salida = limpiar(salida)
        tmp.write_text(json.dumps(salida, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
        tmp.replace(SALIDA)
    except Exception as exc:
        print(f"::warning::No se pudo guardar el ambiente del satélite: {exc}")
        return 0
    reg = (salida.get("region") or {}).get("valores") or {}
    print(f"Ambiente satélite: disponible={salida.get('disponible')} "
          f"reutilizado={salida.get('reutilizado')} CAPE región={reg.get('cape_jkg')} "
          f"agua={reg.get('agua_precipitable_mm')} mm {salida.get('motivo') or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
