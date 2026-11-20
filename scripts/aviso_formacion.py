#!/usr/bin/env python3
"""Aviso de tormenta que se forma en el lugar, antes del radar (punto 5).

EN PARALELO (regla 5): se calcula y se guarda en cada ciclo, pero NO cambia
lo que muestra ClimaAR hasta que las simulaciones con casos reales muestren
cuánto acierta.

Toma las nubes en crecimiento que encontró el satélite (goes19/formacion.py,
en goes19_state.json -> formacion_satelite) y:

1. Control con el radar (imagen actual.png, la misma del ciclo): si en ese
   lugar ya hay eco de tormenta (35 dBZ o más) o un núcleo del radar, no es
   una tormenta nueva sino una que crece (de eso se ocupa el punto 4). Si no
   hay eco, o hay eco débil (20 a 35 dBZ), es una tormenta en formación.
2. Grado (calibrado con la simulación del 13/11 y el 19/11):
   - 0: tope todavía sobre -10 °C (casi nunca llega a tormenta), o tope muy
     frío sin nada de precipitación en el radar (nube alta).
   - 1 vigilancia: nube que crece, sin confirmación.
   - 2 probable: con eco débil en el radar (20-35 dBZ) o un rayo; o, cerca
     del radar, tope de -20 a -25 °C con hielo.
   - 3 inminente: eco débil o rayo con el tope ya bajo -20 °C.
   Se guarda además un potencial con reglas graduales (estilo SATCAST:
   enfriamiento en 10 y 20 min, cruce de 0 °C, hielo, altura sobre el 0 °C,
   vapor menos infrarrojo y CAPE) para calibrarlo con más casos.
3. Reglas de seguridad: ambiente estable o nube muy extensa (frente) como
   máximo "vigilancia" (salvo eco débil o rayos); sin radar para controlar,
   como máximo "probable".
4. Seguimiento entre ciclos (estilo RDT): cada nube tiene un número; cuando
   el radar empieza a verla se guarda cuántos minutos antes la avisó el
   satélite (la ventaja). Si en 60 min nunca aparece eco, queda como
   probable falsa alarma.
5. Verificación continua: cada tormenta nueva que aparece en el radar se
   anota con aviso previo o sin aviso. Con eso se mide cuántas se anticipan
   y con cuántos minutos (para las simulaciones de la regla 6).

Salida: data/ia/formacion_tormentas.json y data/ia/formacion/AAAA-MM-DD.jsonl.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))

import ambiente_severo  # noqa: E402
from puntaje_gravedad import ambiente_en, cargar, iso, numero, parse_utc, rampa  # noqa: E402

RADAR_FILE = BASE / "data" / "radar" / "radar_nowcast.json"
RADAR_PNG = BASE / "data" / "radar" / "actual.png"
MASCARA_PNG = BASE / "data" / "radar" / "mascara_ecos_estaticos.png"
CLUTTER_PNG = BASE / "config" / "clutter_rma10.png"
TIMESTAMP_FILE = BASE / "data" / "radar" / "radar_timestamp.txt"
GOES_FILE = BASE / "data" / "goes19" / "goes19_state.json"
METEO_FILE = BASE / "data" / "ia" / "estado_meteorologico.json"
LOCALIDADES_FILE = BASE / "config" / "localidades.json"
SALIDA = BASE / "data" / "ia" / "formacion_tormentas.json"
HISTORIAL = BASE / "data" / "ia" / "formacion"

VERSION = "1.0"
RADAR_LAT, RADAR_LON = -38.7342, -62.1654
RADAR_RANGE_KM = 240.0

ECO_TORMENTA_DBZ = 35.0      # primer eco de tormenta (criterio usual de "inicio")
ECO_DEBIL_DBZ = 20.0
MIN_PIXELES_ECO = 3
MARGEN_ECO_KM = 4.0
MARGEN_NUCLEO_KM = 5.0
UNION_SEGUIMIENTO_KM = 20.0
ESPERA_ECO_MIN = 60.0
CRECE_RADIO_KM_MIN = 0.5      # cuánto se agranda la zona de búsqueda mientras se espera el eco
RADIO_ESPERA_MAX_KM = 30.0
SEGUIMIENTO_VIEJO_MIN = 90.0
VERIFICAR_HASTA_KM = 200.0
YA_ESTABA_KM = 12.0
# Una torre que ya llegó a -25 °C casi siempre tiene precipitación que el radar
# ve (hasta ~160 km el haz está a menos de 3 km de altura). Si no hay nada,
# es nube alta (altoestratos, cirros) que se espesa. Simulación del 19/11:
# 48 avisos así al mediodía, ninguno terminó en tormenta.
TOPE_FRIO_SIN_ECO_C = -25.0
RADAR_VE_BAJO_KM = 160.0
LLEGADAS_RECIENTES_MIN = 30.0
TOPE_MINIMO_AVISO_C = -10.0
TOPE_INMINENTE_C = -20.0           # movimiento de ~70 km/h en 10 min    # tormentas nuevas más lejos pueden venir de afuera del radar
CERCA_LOCALIDAD_KM = 30.0
HISTORIA_MAX = 8

PESOS = {"enfriamiento": 0.30, "persistencia": 0.15, "tope": 0.10, "hielo": 0.15,
         "sobre_0c": 0.10, "vapor_ir": 0.05, "ambiente": 0.15}
NOMBRES = {0: "sin_aviso", 1: "vigilancia", 2: "probable", 3: "inminente"}


def dist_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _d(lat1, lon1, lat2, lon2) -> float:
    return float(dist_km(lat1, lon1, lat2, lon2))


# ----------------------------------------------------------------------
# Radar
# ----------------------------------------------------------------------

def leer_radar_dbz(path: Path = RADAR_PNG) -> np.ndarray | None:
    """dBZ de cada píxel de la imagen RMA10 (NaN = sin eco)."""
    from PIL import Image

    try:
        from decodificador_rma10_dbz import DBZ_MAX, DBZ_MIN, PALETTE_INDEX_MAX, PALETTE_INDEX_MIN

        with Image.open(path) as img:
            if img.mode != "P":
                return None
            transparencia = img.info.get("transparency")
            idx = np.asarray(img, dtype=float)
        dbz = DBZ_MIN + (np.clip(idx, PALETTE_INDEX_MIN, PALETTE_INDEX_MAX) - PALETTE_INDEX_MIN) / (
            PALETTE_INDEX_MAX - PALETTE_INDEX_MIN) * (DBZ_MAX - DBZ_MIN)
        if transparencia is not None:
            dbz[idx == transparencia] = np.nan
        return dbz
    except Exception:
        return None


def leer_mascara(shape) -> np.ndarray | None:
    """Ecos fijos conocidos (sierras, tierra) + clutter persistente."""
    from PIL import Image

    salida = None
    for path, corte in ((MASCARA_PNG, 100), (CLUTTER_PNG, 127)):
        try:
            with Image.open(path) as img:
                m = np.asarray(img.convert("L")) > corte
        except Exception:
            continue
        if m.shape == shape:
            salida = m if salida is None else (salida | m)
    return salida


def pixel_de(lat: float, lon: float, ancho: int, alto: int) -> tuple[float, float]:
    """Inversa de georreferenciar_rma10.pixel_to_latlon (proyección azimutal)."""
    d = _d(RADAR_LAT, RADAR_LON, lat, lon)
    p1, p2 = math.radians(RADAR_LAT), math.radians(lat)
    dl = math.radians(lon - RADAR_LON)
    rumbo = math.atan2(math.sin(dl) * math.cos(p2),
                       math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl))
    este, norte = d * math.sin(rumbo), d * math.cos(rumbo)
    cx, cy = ancho / 2.0, alto / 2.0
    return cx + este / RADAR_RANGE_KM * (ancho / 2.0), cy - norte / RADAR_RANGE_KM * (alto / 2.0)


def eco_en(dbz: np.ndarray | None, mascara: np.ndarray | None, lat: float, lon: float,
           radio_km: float) -> dict[str, Any]:
    """Eco del radar alrededor de un punto (sin contar los ecos fijos)."""
    if dbz is None:
        return {"estado": "no_disponible"}
    if _d(RADAR_LAT, RADAR_LON, lat, lon) > RADAR_RANGE_KM:
        return {"estado": "fuera_de_alcance"}
    alto, ancho = dbz.shape
    x, y = pixel_de(lat, lon, ancho, alto)
    km_px = RADAR_RANGE_KM / (ancho / 2.0)
    r = radio_km / km_px
    yy, xx = np.ogrid[:alto, :ancho]
    zona = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    if mascara is not None:
        zona &= ~mascara
    v = dbz[zona]
    v = v[np.isfinite(v)]
    n35 = int(np.sum(v >= ECO_TORMENTA_DBZ))
    n20 = int(np.sum(v >= ECO_DEBIL_DBZ))
    clase = "eco_tormenta" if n35 >= MIN_PIXELES_ECO else ("eco_debil" if n20 >= MIN_PIXELES_ECO else "sin_eco")
    return {"estado": "ok", "clase": clase, "dbz_max": round(float(v.max()), 1) if v.size else None,
            "pixeles_35dbz": n35, "pixeles_20dbz": n20, "radio_km": round(radio_km, 1)}


def nucleos_radar(radar: dict | None) -> list[dict]:
    salida = []
    for n in ((radar or {}).get("nowcast") or {}).get("nucleos_georreferenciados") or []:
        geo = n.get("georreferencia") or {}
        lat, lon = numero(geo.get("latitud")), numero(geo.get("longitud"))
        if lat is None or lon is None:
            continue
        radio = math.sqrt(max(numero(n.get("pixeles")) or 0.0, 0.0) * 0.93933 ** 2 / math.pi)
        salida.append({"track_id": n.get("track_id"), "latitud": lat, "longitud": lon, "radio_km": radio,
                       "dbz_max": numero((n.get("dbz") or {}).get("max"))})
    return salida


def nucleo_cerca(nucleos: list[dict], lat: float, lon: float, margen_km: float) -> dict | None:
    mejor = None
    for n in nucleos:
        d = _d(lat, lon, n["latitud"], n["longitud"])
        if d <= n["radio_km"] + margen_km and (mejor is None or d < mejor[0]):
            mejor = (d, n)
    return None if mejor is None else {"track_id": mejor[1]["track_id"], "distancia_km": round(mejor[0], 1),
                                        "dbz_max": mejor[1]["dbz_max"]}


# ----------------------------------------------------------------------
# Grado de cada nube en formación
# ----------------------------------------------------------------------

def componentes(obj: dict, amb: dict | None) -> dict[str, Any]:
    c: dict[str, Any] = {}
    c["enfriamiento"] = rampa(numero(obj.get("enfriamiento_10min_c")), 3.0, 10.0)
    c["persistencia"] = rampa(numero(obj.get("enfriamiento_20min_c")), 4.0, 14.0)
    tope = numero(obj.get("tope_min_c"))
    c["tope"] = rampa(-tope if tope is not None else None, 0.0, 35.0)
    if obj.get("cruce_0c") and c["tope"] is not None:
        c["tope"] = max(c["tope"], 0.5)   # acaba de pasar los 0 °C
    c["hielo"] = rampa(numero(obj.get("fraccion_hielo")), 0.2, 0.7)
    c["vapor_ir"] = rampa(numero(obj.get("vapor_menos_infrarrojo_c")), -35.0, -10.0)
    nivel0 = cape = None
    if amb and isinstance(amb.get("registro"), dict):
        ev = ambiente_severo.evaluar(amb["registro"])
        cape = numero(ev.get("cape"))
        nivel0 = numero(amb["registro"].get("freezing_level_height"))
    c["ambiente"] = rampa(cape, 150.0, 1500.0)
    altura = numero(obj.get("altura_tope_km"))
    c["sobre_0c"] = rampa(altura - nivel0 / 1000.0, 0.5, 4.0) if altura is not None and nivel0 is not None else None
    c["cape"] = cape
    c["nivel_0c_m"] = nivel0
    return c


def calificar(obj: dict, comp: dict, eco: dict, detecciones: int) -> dict[str, Any]:
    """Grado de aviso. Reglas calibradas con la simulación de dos días reales
    (13/11 y 19/11, 327 nubes seguidas): lo que mejor anticipa una tormenta
    es una nube que crece con eco débil (20-35 dBZ) en el radar (79 % llegó a
    tormenta); una nube que crece con el tope todavía sobre -10 °C casi nunca
    llega (7 %); el enfriamiento más rápido o repetido no mejoró el acierto.
    El potencial (reglas graduales) se guarda para la calibración."""
    usados = {k: (comp[k], w) for k, w in PESOS.items() if comp.get(k) is not None}
    if "enfriamiento" not in usados:
        return {"potencial": None, "grado": 0, "nombre": NOMBRES[0], "confianza": 0.0, "limites": []}
    pot = sum(v * w for v, w in usados.values()) / sum(w for _, w in usados.values())
    rayos = (obj.get("rayos") or {})
    con_rayos = rayos.get("estado") == "ok" and (rayos.get("cantidad") or 0) > 0
    tope = numero(obj.get("tope_min_c"))
    dist = numero(obj.get("distancia_radar_km")) or 999.0
    eco_debil = eco.get("clase") == "eco_debil"
    confirmaciones = []
    if (numero(obj.get("fraccion_hielo")) or 0.0) >= 0.5:
        confirmaciones.append("tope_con_hielo")
    if con_rayos:
        confirmaciones.append("rayos")
    if eco_debil:
        confirmaciones.append("eco_debil_en_radar")

    limites = []
    if tope is None or tope > TOPE_MINIMO_AVISO_C:
        grado = 0
        limites.append("tope todavía sobre -10 °C: nube chica, casi nunca llega a tormenta")
    elif eco_debil or con_rayos:
        grado = 3 if tope <= TOPE_INMINENTE_C else 2
    elif (dist <= RADAR_VE_BAJO_KM and tope <= TOPE_INMINENTE_C and tope > TOPE_FRIO_SIN_ECO_C
          and "tope_con_hielo" in confirmaciones):
        grado = 2
    else:
        grado = 1
    if (eco.get("clase") == "sin_eco" and tope is not None and tope <= TOPE_FRIO_SIN_ECO_C
            and dist <= RADAR_VE_BAJO_KM and not con_rayos and grado > 0):
        grado = 0
        limites.append("tope muy frío sin nada de precipitación en el radar: probable nube alta, no tormenta")
    if comp.get("cape") is not None and comp["cape"] < 50.0 and grado > 1 and not (eco_debil or con_rayos):
        grado = 1
        limites.append("ambiente estable")
    if (obj.get("area_km2") or 0.0) > 3000.0 and (numero(obj.get("enfriamiento_10min_c")) or 0.0) < 6.0 and grado > 1:
        grado = 1
        limites.append("nube muy extensa que se enfría poco (probable frente)")
    if eco.get("estado") != "ok" and grado > 2:
        grado = 2
        limites.append("sin radar para controlar")
    confianza = sum(w for _, w in usados.values()) / sum(PESOS.values())
    return {"potencial": round(pot, 3), "grado": grado, "nombre": NOMBRES[grado],
            "confirmaciones": confirmaciones, "limites": limites, "confianza": round(confianza, 2),
            "detecciones": detecciones, "aportes": {k: round(v, 3) for k, (v, _) in usados.items()}}


# ----------------------------------------------------------------------
# Seguimiento entre ciclos
# ----------------------------------------------------------------------

def _minutos(a: datetime | None, b: datetime | None) -> float | None:
    return None if a is None or b is None else round((a - b).total_seconds() / 60.0, 1)


def _emparejar(tracks: list[dict], objetos: list[dict]) -> dict[int, int]:
    """Objeto -> seguimiento, el más cercano primero."""
    pares = []
    for i, o in enumerate(objetos):
        for j, t in enumerate(tracks):
            d = _d(o["latitud"], o["longitud"], t["latitud"], t["longitud"])
            if d <= max(UNION_SEGUIMIENTO_KM, o["radio_km"] + t["radio_km"] + 5.0):
                pares.append((d, i, j))
    usado_o, usado_t, salida = set(), set(), {}
    for _, i, j in sorted(pares):
        if i in usado_o or j in usado_t:
            continue
        usado_o.add(i)
        usado_t.add(j)
        salida[i] = j
    return salida


def _evento_llegada(t: dict, frame: datetime, eco: dict, nucleo: dict | None) -> dict[str, Any]:
    primer = parse_utc(t.get("primer_aviso_satelite_utc"))
    return {
        "tipo": "llego_eco",
        "id": t["id"],
        "frame_radar_utc": iso(frame),
        "inicio_satelite_utc": t.get("inicio_satelite_utc"),
        "primer_aviso_satelite_utc": t.get("primer_aviso_satelite_utc"),
        "grado_max": t.get("grado_max", 0),
        "ventaja_min": _minutos(frame, primer),
        "ventaja_desde_deteccion_min": _minutos(frame, parse_utc(t.get("inicio_satelite_utc"))),
        "track_id_radar": (nucleo or {}).get("track_id"),
        "dbz_max": eco.get("dbz_max") if eco.get("estado") == "ok" else (nucleo or {}).get("dbz_max"),
        "latitud": t["latitud"],
        "longitud": t["longitud"],
    }


def calcular(goes: dict | None, radar: dict | None, frame: datetime, meteo: dict | None,
             previo: dict | None, dbz: np.ndarray | None, mascara: np.ndarray | None,
             localidades: list[dict] | None = None) -> dict[str, Any]:
    base: dict[str, Any] = {"version": VERSION, "frame_radar_utc": iso(frame),
                            "modo": "en_paralelo_no_se_muestra",
                            "generado_utc": iso(datetime.now(timezone.utc)),
                            "grados": NOMBRES}

    # Estado anterior (si el ciclo se repite para la misma imagen, se parte
    # de lo mismo que la vez anterior: el resultado no cambia).
    entrada = {"seguimiento": [], "tracks_radar_conocidos": None, "nucleos_previos": None,
               "llegadas_recientes": [], "frame_radar_utc": None}
    if isinstance(previo, dict):
        fp = parse_utc(previo.get("frame_radar_utc"))
        if fp == frame and isinstance(previo.get("entrada"), dict):
            entrada = previo["entrada"]
        elif fp is not None and fp < frame and (frame - fp).total_seconds() / 60.0 <= SEGUIMIENTO_VIEJO_MIN:
            entrada = {"seguimiento": previo.get("seguimiento") or [],
                       "tracks_radar_conocidos": previo.get("tracks_radar_conocidos"),
                       "nucleos_previos": previo.get("nucleos_radar"),
                       "llegadas_recientes": previo.get("llegadas_recientes") or [],
                       "frame_radar_utc": previo.get("frame_radar_utc")}
    base["entrada"] = entrada

    radar_ok = isinstance(radar, dict) and parse_utc(radar.get("actualizado")) == frame
    nucleos = nucleos_radar(radar) if radar_ok else []
    if not radar_ok:
        dbz = None
    fs = (goes or {}).get("formacion_satelite") if isinstance(goes, dict) else None
    goes_ok = (isinstance(goes, dict) and parse_utc(goes.get("reference_frame_utc")) == frame
               and isinstance(fs, dict))
    hora_sat = parse_utc((fs or {}).get("hora_satelite_utc")) if goes_ok else None
    if hora_sat is not None and not (0.0 <= (frame - hora_sat).total_seconds() / 60.0 <= 20.0):
        goes_ok, hora_sat = False, None   # satélite de otro momento: no se usa
    base["fuentes"] = {
        "satelite": (fs or {}).get("estado") if goes_ok else "no_sincronizado",
        "radar": "ok" if dbz is not None else ("solo_nucleos" if radar_ok else "no_disponible"),
        "imagenes_previas": (fs or {}).get("imagenes_previas") if goes_ok else None,
    }
    base["hora_satelite_utc"] = iso(hora_sat)

    # 1. Objetos del satélite: control con el radar y grado.
    objetos = []
    for o in ((fs or {}).get("objetos") or []) if goes_ok else []:
        lat, lon = o["latitud"], o["longitud"]
        eco = eco_en(dbz, mascara, lat, lon, o["radio_km"] + MARGEN_ECO_KM)
        nucleo = nucleo_cerca(nucleos, lat, lon, MARGEN_NUCLEO_KM)
        existente = eco.get("clase") == "eco_tormenta" or nucleo is not None
        item = {**o, "eco_radar": eco, "nucleo_radar": nucleo,
                "tipo": "tormenta_existente_que_crece" if existente else "formacion"}
        item["componentes"] = componentes(o, ambiente_en(meteo, lat, lon, frame))
        objetos.append(item)

    # 2. Seguimiento.
    tracks = [dict(t) for t in (entrada.get("seguimiento") or [])]
    emparejados = _emparejar(tracks, objetos)
    eventos: list[dict] = []
    terminados: set[int] = set()
    vistos: set[int] = set()
    nuevos_tracks = []
    for i, o in enumerate(objetos):
        j = emparejados.get(i)
        if o["tipo"] == "tormenta_existente_que_crece":
            o["calificacion"] = {"potencial": None, "grado": 0, "nombre": "ya_es_tormenta_en_radar"}
            if j is not None and j not in terminados:
                eventos.append(_evento_llegada(tracks[j], frame, o["eco_radar"], o["nucleo_radar"]))
                terminados.add(j)
                o["seguimiento_id"] = tracks[j]["id"]
            continue
        detecciones = (tracks[j].get("detecciones", 0) + 1) if j is not None else 1
        o["calificacion"] = calificar(o, o["componentes"], o["eco_radar"], detecciones)
        g = o["calificacion"]["grado"]
        if j is None:
            t = {"id": f"F{hora_sat:%Y%m%d%H%M}-{len(nuevos_tracks) + 1}" if hora_sat else f"F{i}",
                 "inicio_satelite_utc": iso(hora_sat), "primer_aviso_satelite_utc": None, "grado_max": 0,
                 "detecciones": 0, "historia": []}
            nuevos_tracks.append(t)
        else:
            t = tracks[j]
            vistos.add(j)
        t.update({"latitud": o["latitud"], "longitud": o["longitud"], "radio_km": o["radio_km"],
                  "ultima_hora_satelite_utc": iso(hora_sat), "detecciones": detecciones,
                  "estado": "en_formacion", "grado_actual": g,
                  "grado_max": max(int(t.get("grado_max") or 0), g)})
        if g >= 1 and not t.get("primer_aviso_satelite_utc"):
            t["primer_aviso_satelite_utc"] = iso(hora_sat)
        t["historia"] = (t.get("historia") or [])[-(HISTORIA_MAX - 1):] + [
            [iso(hora_sat), g, o.get("enfriamiento_10min_c"), o.get("tope_min_c")]]
        o["seguimiento_id"] = t["id"]

    # Seguimientos que hoy no tienen nube en crecimiento: se espera el eco.
    for j, t in enumerate(tracks):
        if j in terminados or j in vistos:
            continue
        t["estado"] = "esperando_eco"
        t["grado_actual"] = 0
        ultima = parse_utc(t.get("ultima_hora_satelite_utc"))
        espera = _minutos(frame, ultima) or 0.0
        radio = min(RADIO_ESPERA_MAX_KM, t.get("radio_km", 5.0) + MARGEN_ECO_KM + CRECE_RADIO_KM_MIN * espera)
        eco = eco_en(dbz, mascara, t["latitud"], t["longitud"], radio)
        nucleo = nucleo_cerca(nucleos, t["latitud"], t["longitud"], radio)
        if eco.get("clase") == "eco_tormenta" or nucleo is not None:
            eventos.append(_evento_llegada(t, frame, eco, nucleo))
            terminados.add(j)
        elif espera > ESPERA_ECO_MIN:
            eventos.append({"tipo": "sin_eco", "id": t["id"], "frame_radar_utc": iso(frame),
                            "inicio_satelite_utc": t.get("inicio_satelite_utc"),
                            "grado_max": t.get("grado_max", 0),
                            "probable_falsa_alarma": (t.get("grado_max") or 0) >= 1,
                            "radar": eco.get("estado"),
                            "latitud": t["latitud"], "longitud": t["longitud"]})
            terminados.add(j)

    activos = [t for j, t in enumerate(tracks) if j not in terminados] + nuevos_tracks

    # 3. Verificación: tormentas nuevas en el radar, con o sin aviso previo.
    actuales = sorted({str(n["track_id"]) for n in nucleos
                       if n["track_id"] is not None and (n["dbz_max"] or 0.0) >= ECO_TORMENTA_DBZ})
    conocidos = entrada.get("tracks_radar_conocidos")
    if radar_ok and conocidos is not None:
        recientes = [e for e in (entrada.get("llegadas_recientes") or [])
                     if (_minutos(frame, parse_utc(e.get("frame_radar_utc"))) or 0.0) <= LLEGADAS_RECIENTES_MIN]
        todas = [e for e in eventos if e["tipo"] == "llego_eco"] + recientes
        llegadas = {str(e.get("track_id_radar")): e for e in todas if e.get("track_id_radar") is not None}
        for n in nucleos:
            tid = str(n["track_id"])
            if tid not in actuales or tid in conocidos:
                continue
            if _d(RADAR_LAT, RADAR_LON, n["latitud"], n["longitud"]) > VERIFICAR_HASTA_KM:
                continue
            # Número nuevo pero en el lugar de una tormenta que ya estaba (se
            # partió, se unió o el seguimiento le cambió el número): no es nueva.
            if any(_d(p["latitud"], p["longitud"], n["latitud"], n["longitud"])
                   <= p["radio_km"] + n["radio_km"] + YA_ESTABA_KM * p.get("imagenes_antes", 1)
                   for p in (entrada.get("nucleos_previos") or [])):
                continue
            e = llegadas.get(tid)
            if e is None:
                # Un seguimiento con aviso muy cerca también cuenta.
                cerca = [x for x in todas if _d(x["latitud"], x["longitud"], n["latitud"], n["longitud"]) <= 25.0]
                cerca += [{"id": t["id"], "grado_max": t.get("grado_max", 0),
                           "ventaja_min": _minutos(frame, parse_utc(t.get("primer_aviso_satelite_utc")))}
                          for t in activos if (t.get("grado_max") or 0) >= 1
                          and _d(t["latitud"], t["longitud"], n["latitud"], n["longitud"]) <= 25.0]
                cerca.sort(key=lambda x: -(x.get("grado_max") or 0))
                e = cerca[0] if cerca else None
            con_aviso = e is not None and (e.get("grado_max") or 0) >= 1
            eventos.append({"tipo": "tormenta_nueva_en_radar", "track_id_radar": n["track_id"],
                            "frame_radar_utc": iso(frame), "latitud": round(n["latitud"], 4),
                            "longitud": round(n["longitud"], 4), "con_aviso_previo": con_aviso,
                            "seguimiento_id": (e or {}).get("id"),
                            "ventaja_min": (e or {}).get("ventaja_min") if con_aviso else None})
    tracks_conocidos = actuales if radar_ok else conocidos

    # 4. Localidades.
    if localidades is None:
        localidades = (cargar(LOCALIDADES_FILE) or {}).get("localidades") or []
    por_loc = {}
    for loc in localidades:
        cerca = []
        for t in activos:
            if t.get("estado") != "en_formacion" or (t.get("grado_actual") or 0) < 1:
                continue
            d = _d(loc["lat"], loc["lon"], t["latitud"], t["longitud"])
            if d <= CERCA_LOCALIDAD_KM + t.get("radio_km", 0.0):
                cerca.append((d, t))
        cerca.sort(key=lambda x: x[0])
        por_loc[loc["id"]] = {
            "grado": max((t["grado_actual"] for _, t in cerca), default=0),
            "nube_mas_cercana_km": round(cerca[0][0], 1) if cerca else None,
            "seguimientos": [t["id"] for _, t in cerca],
        }

    formando = [o for o in objetos if o["tipo"] == "formacion"]
    resumen = {
        "nubes_en_formacion": len(formando),
        "tormentas_existentes_que_crecen": len(objetos) - len(formando),
        "por_grado": {NOMBRES[g]: sum(1 for o in formando if o["calificacion"]["grado"] == g) for g in (1, 2, 3)},
        "seguimientos_activos": len(activos),
        "llegadas_de_eco": [e["ventaja_min"] for e in eventos if e["tipo"] == "llego_eco"],
        "tormentas_nuevas_con_aviso": sum(1 for e in eventos if e["tipo"] == "tormenta_nueva_en_radar"
                                          and e["con_aviso_previo"]),
        "tormentas_nuevas_sin_aviso": sum(1 for e in eventos if e["tipo"] == "tormenta_nueva_en_radar"
                                          and not e["con_aviso_previo"]),
    }
    if not goes_ok or (fs or {}).get("estado") in ("sin_satelite", "error"):
        estado = "sin_satelite"
    elif (fs or {}).get("estado") == "sin_imagen_previa":
        estado = "satelite_sin_imagen_previa"
    elif formando:
        estado = "ok"
    else:
        estado = "sin_formacion"
    # Núcleos de esta imagen y de la anterior (una imagen con ecos cortados no
    # hace aparecer tormentas "nuevas" en la siguiente).
    nucleos_guardar = entrada.get("nucleos_previos")
    if radar_ok:
        nucleos_guardar = [{"latitud": round(n["latitud"], 4), "longitud": round(n["longitud"], 4),
                            "radio_km": round(n["radio_km"], 1), "imagenes_antes": 1} for n in nucleos
                           if (n["dbz_max"] or 0.0) >= ECO_TORMENTA_DBZ]
        nucleos_guardar += [{**p, "imagenes_antes": 2} for p in (entrada.get("nucleos_previos") or [])
                            if p.get("imagenes_antes", 1) == 1]
    return {**base, "estado": estado, "objetos": objetos, "seguimiento": activos, "eventos": eventos,
            "tracks_radar_conocidos": tracks_conocidos, "nucleos_radar": nucleos_guardar,
            "llegadas_recientes": [e for e in (entrada.get("llegadas_recientes") or [])
                                   if (_minutos(frame, parse_utc(e.get("frame_radar_utc"))) or 0.0)
                                   <= LLEGADAS_RECIENTES_MIN]
            + [{k: e.get(k) for k in ("id", "frame_radar_utc", "grado_max", "ventaja_min", "track_id_radar",
                                      "latitud", "longitud")} for e in eventos if e["tipo"] == "llego_eco"], "localidades": por_loc, "resumen": resumen}


# ----------------------------------------------------------------------
# Historial (regla 7) y ciclo
# ----------------------------------------------------------------------

def _historial(salida: dict, carpeta: Path = HISTORIAL) -> str:
    if not salida.get("objetos") and not salida.get("eventos"):
        return "sin nubes en formación ni eventos: no se guarda"
    frame = parse_utc(salida["frame_radar_utc"])
    carpeta.mkdir(parents=True, exist_ok=True)
    path = carpeta / f"{frame:%Y-%m-%d}.jsonl"
    linea = json.dumps({
        "frame_radar_utc": salida["frame_radar_utc"],
        "hora_satelite_utc": salida.get("hora_satelite_utc"),
        "fuentes": salida.get("fuentes"),
        "objetos": [{
            "id": o.get("seguimiento_id"), "tipo": o["tipo"], "lat": o["latitud"], "lon": o["longitud"],
            "area_km2": o["area_km2"], "tope_min_c": o.get("tope_min_c"),
            "enfr10": o.get("enfriamiento_10min_c"), "enfr20": o.get("enfriamiento_20min_c"),
            "cruce_0c": o.get("cruce_0c"), "hielo": o.get("fraccion_hielo"),
            "altura_km": o.get("altura_tope_km"), "vapor_ir": o.get("vapor_menos_infrarrojo_c"),
            "rayos": (o.get("rayos") or {}).get("cantidad"), "eco": (o.get("eco_radar") or {}).get("clase"),
            "dbz_max": (o.get("eco_radar") or {}).get("dbz_max"), "cape": (o.get("componentes") or {}).get("cape"),
            "grado": (o.get("calificacion") or {}).get("grado"),
            "potencial": (o.get("calificacion") or {}).get("potencial"),
        } for o in salida.get("objetos") or []],
        "eventos": salida.get("eventos") or [],
        "localidades": {k: v["grado"] for k, v in (salida.get("localidades") or {}).items() if v["grado"]},
    }, ensure_ascii=False, separators=(", ", ": "))
    clave = f'"frame_radar_utc": "{salida["frame_radar_utc"]}"'
    renglones = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for i, r in enumerate(renglones):
        if r.startswith("{" + clave):
            renglones[i] = linea   # misma imagen: se reemplaza (resultado idéntico al repetir)
            path.write_text("\n".join(renglones) + "\n", encoding="utf-8")
            return f"reemplazado ({path.name})"
    with path.open("a", encoding="utf-8") as f:
        f.write(linea + "\n")
    return f"guardado en {path.name}"


def main() -> int:
    try:
        frame = parse_utc(TIMESTAMP_FILE.read_text(encoding="utf-8").strip())
        dbz = leer_radar_dbz()
        mascara = leer_mascara(dbz.shape) if dbz is not None else None
        salida = calcular(cargar(GOES_FILE), cargar(RADAR_FILE), frame, cargar(METEO_FILE),
                          cargar(SALIDA), dbz, mascara)
        SALIDA.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
        r = salida["resumen"]
        print(f"Tormentas en formación (en paralelo, no se muestra): {salida['estado']}, "
              f"satélite {salida['fuentes']['satelite']}, radar {salida['fuentes']['radar']}")
        print(f"  nubes en formación: {r['nubes_en_formacion']} {r['por_grado']}, "
              f"tormentas existentes que crecen: {r['tormentas_existentes_que_crecen']}, "
              f"seguimientos activos: {r['seguimientos_activos']}")
        for o in salida["objetos"]:
            c = o.get("calificacion") or {}
            print(f"  {o.get('seguimiento_id')}: {o['tipo']}, {o['distancia_radar_km']} km del radar, "
                  f"tope {o.get('tope_min_c')} °C, enfría {o.get('enfriamiento_10min_c')} °C/10 min, "
                  f"radar {(o.get('eco_radar') or {}).get('clase')}, grado {c.get('grado')} ({c.get('nombre')})")
        for e in salida["eventos"]:
            print(f"  evento: {e}")
        print(f"Historial: {_historial(salida)}")
    except Exception as exc:  # falla segura: nunca corta el ciclo
        print(f"AVISO tormentas en formación: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
