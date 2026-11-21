#!/usr/bin/env python3
"""Nowcast en capas (punto 5b): nunca a ciegas cuando no hay imagen de radar nueva.

No es otro nowcast: toma el mismo radar, el mismo flujo óptico y la misma
advección que `nowcast_flujo_optico.py` y le suma, en capas, lo que llega en
tiempo real. Cada capa corrige a la anterior; si falta una fuente, siguen las demás.

- Capa 1, movimiento: la última imagen movida a la HORA ACTUAL; los plazos se
  cuentan desde ahora y no se corta a los 40 min (la confianza baja de a poco).
- Capa 2, crecimiento y debilitamiento: tendencia del radar siguiendo al eco
  (solo si la validación simulada la aprueba: config/nowcast_capas.json) y,
  por tormenta, satélite (tope que se enfría o se calienta) y rayos (salto o caída).
- Capa 3, relleno en tiempo real: rayos GLM (ventana que termina ahora), la
  imagen más reciente del satélite por localidad y las estaciones. Cuanto más
  viejo el radar, más pesan; lo medido manda sobre el radar viejo.
- Capa 4, qué llega al suelo: capa fría y estable abajo (convección elevada) y
  firma de las estaciones (frente de ráfagas débil o reventón).
- Capa 5, incertidumbre: escenarios con la velocidad y la dirección corridas;
  porcentaje de llegada por localidad; edad y fuente de cada dato.
- Capa 6, validación automática: cada pronóstico se guarda y, cuando llega la
  imagen real de esa hora, se compara por localidad y en la región.

EN PARALELO (regla 5): calcula y guarda, no cambia lo que se ve.
Salidas: data/radar/nowcast_capas.json y data/ia/verificacion_nowcast.json.
Falla segura: nunca corta el ciclo; si falta el radar lo dice y sigue con lo medido.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import localidades_radar as lr  # noqa: E402
import corrector_satelite as cs  # noqa: E402
import corrector_tiempo_real as ct  # noqa: E402
import nowcast_capas_nucleo as nc  # noqa: E402
import nowcast_flujo_optico as nf  # noqa: E402
from scripts.rma10_features import parse_utc  # noqa: E402

VERSION = "nowcast-capas-1.1"
SALIDA = ROOT / "data" / "radar" / "nowcast_capas.json"
VERIFICACION = ROOT / "data" / "ia" / "verificacion_nowcast.json"
CONFIG = ROOT / "config" / "nowcast_capas.json"
GOES = ROOT / "data" / "goes19" / "goes19_state.json"
SENALES = ROOT / "data" / "ia" / "estaciones" / "senales_superficie.json"
METEO = ROOT / "data" / "ia" / "estado_meteorologico.json"

PLAZOS_DESDE_AHORA = (0, 10, 20, 30, 60)
UMBRAL_LLUVIA = nf.UMBRAL_ECO_DBZ
UMBRAL_TORMENTA = nf.UMBRAL_TORMENTA_DBZ
EDAD_ESTACIONES_MAX_MIN = 20.0
EDAD_SATELITE_MAX_MIN = 20.0
PENDIENTES_MAX_MIN = 100.0
TOLERANCIA_VERIF_MIN = 2.5
REGION_PASO_KM = 30

CONFIG_DEFECTO = {
    "movimiento_3_imagenes": True,
    "tendencia_radar": False,
    "escenarios": True,
    "anvil": False,
    "corrector": False,
    "control_calidad_satelite": True,
    "reglas_corrector": {},
    "llegada_envolvente": {"activa": False, "k_probable": 2, "k_posible": 1},
    # Etapa A: satélite más nuevo que el radar (posición, movimiento, crecimiento) y lluvia
    # medida por las estaciones. "activo" = cambia lo que se pronostica; "en_paralelo" =
    # se calcula y se valida en vivo (verificacion_nowcast.json) sin cambiar nada.
    "satelite_etapa_a": {"activo": False, "en_paralelo": False, "trayectorias": False, "parametros": {}},
    "estaciones_etapa_a": {"activo": False, "en_paralelo": False, "parametros": {}},
    "nota": "Lo deciden las simulaciones de herramientas/simular_huecos.py y simular_etapa_a.py.",
}


def iso(dt: datetime | None) -> str | None:
    return nf.iso(dt)


def leer_json(path: Path) -> dict:
    try:
        valor = json.loads(path.read_text(encoding="utf-8"))
        return valor if isinstance(valor, dict) else {}
    except Exception:
        return {}


def edad_min(texto: Any, ahora: datetime) -> float | None:
    dt = parse_utc(str(texto)) if texto else None
    return None if dt is None else round((ahora - dt).total_seconds() / 60.0, 1)


def categoria(dbz: float) -> str:
    if dbz >= UMBRAL_TORMENTA:
        return "tormenta"
    if dbz >= UMBRAL_LLUVIA:
        return "lluvia"
    return "sin_eco"


# ------------------------------------------------------------
# zonas
# ------------------------------------------------------------

def zonas_localidades(shape, localidades, margen_km: float = 0.0) -> dict[str, np.ndarray]:
    """Círculo de cada localidad; con `margen_km`, el círculo agrandado (zona "hasta 10 km")."""
    if not margen_km:
        return {k: z for k, (z, _, _) in lr._zonas(shape, localidades).items()}
    km_px = nf.km_por_pixel(shape[1])
    salida = {}
    for loc in localidades:
        cx, cy = lr.latlon_to_pixel(loc["lat"], loc["lon"], shape[1], shape[0])
        radio = max(float(loc.get("radio_km", 5.0)), lr.RADIO_MIN_KM) + float(margen_km)
        salida[loc["id"]] = nf.disco(shape, cx, cy, radio / km_px)
    return salida


def zonas_region(shape) -> dict[str, np.ndarray]:
    """Puntos de la región cada 30 km (círculo de 5 km), de 20 a 200 km del radar."""
    h, w = shape
    km_px = nf.km_por_pixel(w)
    c = (w - 1) / 2.0
    zonas = {}
    for dx in range(-210, 211, REGION_PASO_KM):
        for dy in range(-210, 211, REGION_PASO_KM):
            if 20 <= math.hypot(dx, dy) <= 200:
                zonas[f"{dx}_{dy}"] = nf.disco((h, w), c + dx / km_px, c + dy / km_px, 5.0 / km_px)
    return zonas


# ------------------------------------------------------------
# capa 2 (satélite y rayos por tormenta) y capa 3 (relleno)
# ------------------------------------------------------------

def tendencia_tormentas(goes: dict) -> list[dict[str, Any]]:
    """Crecimiento o debilitamiento de cada tormenta seguida, por satélite y rayos."""
    salida = []
    for t in ((goes.get("por_tormenta") or {}).get("tormentas") or []):
        c13 = t.get("c13") or {}
        rayos = t.get("rayos") or {}
        salto = (rayos.get("salto_rayos") or {})
        enfr = c13.get("enfriamiento_c_10min")
        dfrdt = salto.get("dfrdt_actual")
        senales_crece, senales_baja = [], []
        if isinstance(enfr, (int, float)):
            if enfr >= 4.0:
                senales_crece.append(f"tope que se enfría {enfr} °C en 10 min")
            elif enfr <= -4.0:
                senales_baja.append(f"tope que se calienta {abs(enfr)} °C en 10 min")
        if salto.get("salto"):
            senales_crece.append("salto de rayos")
        elif isinstance(dfrdt, (int, float)) and (rayos.get("tasa_por_min") or 0) >= 1.0:
            if dfrdt <= -1.0:
                senales_baja.append("rayos en caída")
        estado = "creciendo" if senales_crece and not senales_baja else \
            "debilitandose" if senales_baja and not senales_crece else \
            "mixto" if senales_crece else "sin_cambio_claro"
        salida.append({"track_id": t.get("track_id"), "latitud": t.get("latitud"), "longitud": t.get("longitud"),
                       "dbz_max": t.get("dbz_max"), "estado": estado,
                       "senales": senales_crece + senales_baja})
    return salida


def relleno_localidad(loc_id: str, goes: dict, senales: dict, ahora: datetime) -> dict[str, Any]:
    """Capa 3: lo que dicen ahora los rayos, el satélite y las estaciones."""
    salida: dict[str, Any] = {}
    ventana = ((goes.get("glm") or {}).get("ventana")) or {}
    if ventana.get("disponible"):
        r = (ventana.get("por_localidad") or {}).get(loc_id) or {}
        salida["rayos"] = {
            "a_15km_10min": r.get("rayos_15km_10min"), "a_30km_10min": r.get("rayos_30km_10min"),
            "tasa_30km_por_min": r.get("tasa_30km_por_min"), "distancia_rayo_km": r.get("distancia_km"),
            "edad_min": edad_min(ventana.get("fin_ventana_utc"), ahora), "fuente": "GOES-19 GLM",
        }
    sat = goes.get("satelite_ahora") or {}
    if sat.get("estado") == "ok":
        s = (sat.get("por_localidad") or {}).get(loc_id) or {}
        salida["satelite"] = {
            "tope_p10_c_15km": s.get("p10_c_15km"), "tope_min_c_15km": s.get("min_c_15km"),
            "enfriamiento_c_10min": s.get("enfriamiento_c_10min"),
            "edad_min": edad_min(sat.get("hora_satelite_utc"), ahora), "fuente": "GOES-19 canal 13 (imagen más reciente)",
        }
    loc = (senales.get("localidades") or {}).get(loc_id)
    if loc:
        salida["estaciones"] = {
            "cantidad": loc.get("estaciones"), "lluvia_1h_mm": loc.get("lluvia_1h_mm"),
            "rafaga_max_kmh": loc.get("rafaga_max_kmh"), "frente_de_rafagas": loc.get("frente_de_rafagas"),
            "nombres": loc.get("nombres"), "edad_min": edad_min(senales.get("generado_utc"), ahora),
            "fuente": "estaciones BCP / SMN / CWOP",
        }
    return salida


def estado_ahora(dbz_radar: float | None, edad_radar: float | None, relleno: dict) -> dict[str, Any]:
    """Combina radar movido y fuentes en tiempo real. Lo medido manda sobre el radar viejo."""
    peso_radar = nc.confianza_por_edad(edad_radar) if dbz_radar is not None else 0.0
    cat_radar = categoria(dbz_radar) if dbz_radar is not None else None
    evidencias = []
    rayos = relleno.get("rayos") or {}
    sat = relleno.get("satelite") or {}
    est = relleno.get("estaciones") or {}
    rayos_ok = rayos and (rayos.get("edad_min") is None or rayos["edad_min"] <= 15)
    sat_ok = sat and sat.get("edad_min") is not None and sat["edad_min"] <= EDAD_SATELITE_MAX_MIN
    est_ok = est and est.get("edad_min") is not None and est["edad_min"] <= EDAD_ESTACIONES_MAX_MIN

    medido = None
    # Rayos = tormenta ELÉCTRICA (fenómeno aparte). Solo cuentan como tormenta con lluvia si
    # además el satélite ve un tope frío encima; si no, puede ser solo eléctrica, sin lluvia.
    electrica = bool(rayos_ok and (rayos.get("a_15km_10min") or 0) >= 1)
    tope = sat.get("tope_p10_c_15km") if sat_ok else None
    if electrica:
        evidencias.append(f"{rayos['a_15km_10min']} rayos a menos de 15 km en 10 min")
        if isinstance(tope, (int, float)) and tope <= -35.0:
            medido = "tormenta"
    if est_ok and (est.get("lluvia_1h_mm") or 0) >= 0.4:
        evidencias.append(f"lluvia medida {est['lluvia_1h_mm']} mm en 1 h")
        medido = medido or "lluvia"
    if est_ok and est.get("frente_de_rafagas"):
        evidencias.append("frente de ráfagas en estaciones")
        medido = "tormenta"
    satelite_conv = sat_ok and isinstance(sat.get("tope_p10_c_15km"), (int, float)) and sat["tope_p10_c_15km"] <= -45.0
    if satelite_conv:
        evidencias.append(f"tope de nube {sat['tope_p10_c_15km']} °C encima")
    sat_despejado = sat_ok and isinstance(sat.get("tope_p10_c_15km"), (int, float)) and sat["tope_p10_c_15km"] > -10.0
    sin_rayos = rayos_ok and (rayos.get("a_30km_10min") or 0) == 0

    orden = {"sin_eco": 0, "lluvia": 1, "tormenta": 2}
    if cat_radar is None:
        final = medido or ("posible_convectiva" if satelite_conv else "sin_datos_radar")
        manda = "medido" if medido else ("satelite" if satelite_conv else "ninguna")
    elif medido and orden[medido] > orden[cat_radar]:
        final, manda = medido, "medido"
    elif cat_radar != "sin_eco" and peso_radar < 0.6 and sat_despejado and sin_rayos and not medido:
        # Radar viejo dice algo y ahora no hay nube alta, ni rayos, ni lluvia medida.
        final, manda = "posible_ya_paso_o_se_deshizo", "medido"
        evidencias.append("radar viejo sin confirmación: sin nube alta ni rayos ahora")
    else:
        final, manda = cat_radar, "radar_movido"
    rafaga = est.get("rafaga_max_kmh") if est_ok else None
    fenomenos = {
        # Cada fenómeno por separado (índice mixto): puede haber rayos sin lluvia ni viento.
        "precipitacion": final,
        "tormenta_electrica": electrica,
        "viento_medido_kmh": rafaga,
        "lluvia_medida_1h_mm": est.get("lluvia_1h_mm") if est_ok else None,
    }
    return {"categoria": final, "manda": manda, "categoria_radar_movido": cat_radar,
            "peso_radar": round(peso_radar, 2), "evidencias": evidencias, "fenomenos": fenomenos}


# ------------------------------------------------------------
# capa 4: qué llega al suelo
# ------------------------------------------------------------

def capa_estable(meteo: dict) -> dict[str, Any]:
    """Convección elevada: aire frío y estable pegado al suelo (inversión)."""
    amb = meteo.get("ambiente_actual") or {}
    t2, t925 = amb.get("temperature_2m"), amb.get("temperature_925hPa")
    cin, pbl = amb.get("convective_inhibition"), amb.get("boundary_layer_height")
    motivos = []
    if isinstance(t2, (int, float)) and isinstance(t925, (int, float)) and t925 > t2:
        motivos.append(f"inversión: 925 hPa {t925} °C más cálido que 2 m {t2} °C")
    if isinstance(cin, (int, float)) and cin <= -100:
        motivos.append(f"inhibición alta ({cin} J/kg)")
    if isinstance(pbl, (int, float)) and pbl < 300:
        motivos.append(f"capa límite baja ({pbl} m)")
    elevada = bool(motivos) and (len(motivos) >= 2 or motivos[0].startswith("inversión"))
    return {
        "conveccion_elevada": elevada, "motivos": motivos,
        "factor_viento_superficie": 0.5 if elevada else 1.0,
        "factor_granizo_superficie": 0.7 if elevada else 1.0,
        "fuente": "modelo (estado_meteorologico.json)", "hora": amb.get("tiempo_utc"),
        "nota": ("Tormenta por encima de una capa fría: rayos y lluvia, pero el viento fuerte "
                 "y el granizo llegan debilitados al suelo (caso 19/11/2026).") if elevada else None,
    }


def firma_estacion(e: dict) -> str | None:
    """Reventón: bajón fuerte de temperatura + salto de presión + ráfaga fuerte.
    Frente de ráfagas débil: presión que sube de a poco sin bajón de temperatura."""
    dt = e.get("cambio_temperatura_30min_c")
    dp = e.get("salto_presion_30min_hpa")
    raf = e.get("rafaga_kmh") or e.get("viento_kmh") or 0
    if not isinstance(dt, (int, float)) or not isinstance(dp, (int, float)):
        return None
    if dt <= -5.0 and dp >= 2.0 and raf >= 70:
        return "reventon"
    if dt <= -3.0 and dp >= 1.0:
        return "frente_de_rafagas"
    if dt >= -1.0 and 0.5 <= dp < 2.0 and raf < 50:
        return "frente_de_rafagas_debil"
    return None


def firmas_localidades(senales: dict) -> dict[str, list[dict[str, Any]]]:
    por_nombre = {e.get("nombre"): e for e in (senales.get("estaciones") or [])}
    salida = {}
    for loc_id, loc in (senales.get("localidades") or {}).items():
        lista = []
        for nombre in loc.get("nombres") or []:
            e = por_nombre.get(nombre) or {}
            f = firma_estacion(e)
            if f:
                lista.append({"estacion": nombre, "firma": f, "utc": e.get("utc")})
        salida[loc_id] = lista
    return salida


# ------------------------------------------------------------
# "¿llega a mi localidad en la próxima hora?" (punto 5b)
# ------------------------------------------------------------

PLAZOS_LLEGADA = (0, 10, 20, 30, 40, 50, 60)
MARGEN_LLEGADA_KM = 10.0
LLEGADAS_PENDIENTES_MAX = 40


def primera_llegada(picos: dict[int, dict[str, float]], loc: str, umbral: float) -> int | None:
    for p in PLAZOS_LLEGADA:
        if p in picos and picos[p].get(loc, 0.0) >= umbral:
            return p
    return None


def llegadas(campos_por_grado: dict[str, dict[int, np.ndarray]], zonas: dict[str, dict[str, np.ndarray]]
             ) -> dict[str, dict[str, Any]]:
    """Primera llegada (minutos desde ahora, None = no llega en la hora) por localidad,
    fenómeno, grado y zona."""
    picos = {g: {z: {p: nc.picos_por_zona(c, zz) for p, c in campos.items()} for z, zz in zonas.items()}
             for g, campos in campos_por_grado.items()}
    salida: dict[str, dict[str, Any]] = {}
    locs = next(iter(zonas.values())).keys()
    for loc in locs:
        item = {}
        for fen, umbral in (("lluvia", UMBRAL_LLUVIA), ("tormenta", UMBRAL_TORMENTA)):
            item[fen] = {g: {z: primera_llegada(picos[g][z], loc, umbral) for z in zonas} for g in picos}
        salida[loc] = item
    return salida


def verificar_llegadas(verif: dict, radar_dt: datetime, obs: dict[str, dict[str, float]]) -> None:
    """Cada pronóstico de llegada guardado mira las imágenes reales de su próxima hora; al
    cerrarse la hora se cuenta si llegó y si se avisó (por fenómeno, grado y zona)."""
    tot = verif.setdefault("llegadas_acumulado", {})
    quedan = []
    for p in verif.get("llegadas_pendientes") or []:
        desde, hasta = parse_utc(p.get("desde_utc")), parse_utc(p.get("hasta_utc"))
        if desde is None or hasta is None:
            continue
        if desde - timedelta(minutes=TOLERANCIA_VERIF_MIN) <= radar_dt <= hasta + timedelta(minutes=TOLERANCIA_VERIF_MIN):
            vistos = p.setdefault("visto", {})
            for z, por_loc in obs.items():
                for loc, dbz in por_loc.items():
                    clave = f"{z}|{loc}"
                    vistos[clave] = max(vistos.get(clave, 0.0), round(float(dbz), 1))
            p["imagenes"] = p.get("imagenes", 0) + 1
        if radar_dt > hasta + timedelta(minutes=TOLERANCIA_VERIF_MIN):
            if p.get("imagenes", 0) >= 5:   # al menos 5 de las 7 imágenes de la hora
                for loc, fens in (p.get("pronostico") or {}).items():
                    for fen, grados in fens.items():
                        umbral = UMBRAL_LLUVIA if fen == "lluvia" else UMBRAL_TORMENTA
                        for g, zonas in grados.items():
                            for z, minutos in zonas.items():
                                llego = p["visto"].get(f"{z}|{loc}", 0.0) >= umbral
                                aviso = minutos is not None
                                acc = tot.setdefault(f"{fen}|{g}|{z}", [0, 0, 0, 0])
                                acc[0 if aviso and llego else 1 if aviso else 2 if llego else 3] += 1
            continue
        quedan.append(p)
    verif["llegadas_pendientes"] = quedan[-LLEGADAS_PENDIENTES_MAX:]


def resumen_llegadas(verif: dict) -> dict[str, Any]:
    salida = {}
    for clave, (a, b, c, d) in sorted((verif.get("llegadas_acumulado") or {}).items()):
        salida[clave] = {"deteccion_pct": round(100 * a / (a + c), 1) if a + c else None,
                         "falsas_alarmas_pct": round(100 * b / (a + b), 1) if a + b else None,
                         "llegadas": a + c, "avisos": a + b}
    return salida


# ------------------------------------------------------------
# capa 6: validación automática
# ------------------------------------------------------------

def verificar(verif: dict, radar_dt: datetime, obs_loc: dict, obs_reg: dict) -> list[dict]:
    """Compara los pronósticos guardados para la hora de esta imagen con lo observado."""
    hechos = []
    pendientes = []
    tot = verif.setdefault("acumulado", {})
    for p in verif.get("pendientes") or []:
        obj = parse_utc(p.get("objetivo_utc"))
        if obj is None:
            continue
        dif = abs((obj - radar_dt).total_seconds()) / 60.0
        if dif <= TOLERANCIA_VERIF_MIN:
            res = {"objetivo_utc": p["objetivo_utc"], "emitido_utc": p.get("emitido_utc"),
                   "edad_radar_al_emitir_min": p.get("edad_radar_min"), "plazo_min": p.get("plazo_min")}
            # Etapa A: "satelite"/"estaciones" = con esa corrección; "sin_satelite" = sin ella
            # (cuando está activa). Sin la clave, el pronóstico fue igual al de "capas".
            for metodo in ("hoy", "capas", "satelite", "estaciones", "sin_satelite"):
                if metodo in ("hoy", "capas") or metodo in p:
                    datos = p.get(metodo, {})
                elif metodo in p.get("etapa_a", []):
                    datos = p.get("capas", {})
                else:
                    continue
                pl, pr = datos.get("loc", {}), datos.get("reg", [])
                clave = f"{metodo}|{bucket(p.get('plazo_min'))}"
                acc = tot.setdefault(clave, {"20": [0, 0, 0, 0], "35": [0, 0, 0, 0], "error_abs": 0.0, "n": 0})
                pares = [(pl.get(k), obs_loc.get(k)) for k in obs_loc] + list(zip(pr, [obs_reg[k] for k in sorted(obs_reg)]))
                for a, b in pares:
                    if a is None or b is None:
                        continue
                    for u in ("20", "35"):
                        x, y = a >= float(u), b >= float(u)
                        acc[u][0 if x and y else 1 if x else 2 if y else 3] += 1
                    if a >= 20 or b >= 20:
                        acc["error_abs"] = round(acc["error_abs"] + abs(a - b), 1)
                        acc["n"] += 1
                if metodo in ("hoy", "capas"):
                    res[metodo] = {k: [pl.get(k), round(obs_loc[k], 1)] for k in obs_loc}
            hechos.append(res)
        elif obj > radar_dt - timedelta(minutes=TOLERANCIA_VERIF_MIN):
            pendientes.append(p)
    verif["pendientes"] = pendientes
    verif["ultimas"] = (hechos + (verif.get("ultimas") or []))[:40]
    return hechos


def bucket(plazo: Any) -> str:
    try:
        p = float(plazo)
    except (TypeError, ValueError):
        return "?"
    for b in (10, 20, 30, 40, 60, 90):
        if p <= b + 2.5:
            return f"hasta_{b}min"
    return "mas_90min"


def resumen_verificacion(verif: dict) -> dict[str, Any]:
    salida = {}
    for clave, acc in sorted((verif.get("acumulado") or {}).items()):
        fila = {}
        for u in ("20", "35"):
            a, b, c, _ = acc[u]
            fila[f"csi_{u}dbz"] = round(a / (a + b + c), 3) if a + b + c else None
            fila[f"casos_{u}dbz"] = a + b + c
        fila["error_medio_dbz"] = round(acc["error_abs"] / acc["n"], 2) if acc.get("n") else None
        salida[clave] = fila
    return salida


# ------------------------------------------------------------
# ciclo
# ------------------------------------------------------------

C13_AHORA = Path(__import__("tempfile").gettempdir()) / "climaar_c13_ahora.npz"
C13_RADAR = Path(__import__("tempfile").gettempdir()) / "climaar_c13_radar.npz"


def control_calidad(campo: np.ndarray, radar_dt: datetime, salida: dict) -> np.ndarray:
    """Ecos del radar sin ninguna nube encima (satélite de la misma hora) = ecos falsos."""
    info: dict[str, Any] = {"usado": False}
    try:
        if C13_RADAR.exists():
            z = np.load(C13_RADAR)
            hora = parse_utc(str(z["hora"]))
            dif = None if hora is None else abs((hora - radar_dt).total_seconds()) / 60.0
            info["diferencia_satelite_radar_min"] = None if dif is None else round(dif, 1)
            if dif is not None and dif <= 12.0:
                sat = ct.grilla_satelite(z["lat"], z["lon"], z["temp_c"], campo.shape)
                limpio = ct.quitar_ecos_sin_nube(campo, sat)
                eco = int((campo >= UMBRAL_LLUVIA).sum())
                falsos = int(((campo >= UMBRAL_LLUVIA) & (limpio < UMBRAL_LLUVIA)).sum())
                info.update({"usado": True, "pixeles_eco": eco, "pixeles_sin_nube": falsos,
                             "fraccion_sin_nube": round(falsos / eco, 3) if eco else 0.0})
                salida["control_calidad_satelite"] = info
                return limpio
    except Exception as exc:
        info["motivo"] = f"{type(exc).__name__}: {str(exc)[:100]}"
    salida["control_calidad_satelite"] = info
    return campo
EDAD_RAYOS_MAX_MIN = 10.0


def datos_corrector(goes: dict, ahora: datetime, shape) -> dict[str, Any]:
    """Satélite más reciente (grilla del radar) y posición de los rayos de los últimos 10 min."""
    datos: dict[str, Any] = {"satelite": None, "rayos_lat": [], "rayos_lon": [], "info": {}}
    try:
        if C13_AHORA.exists():
            z = np.load(C13_AHORA)
            hora = parse_utc(str(z["hora"]))
            edad_sat = None if hora is None else (ahora - hora).total_seconds() / 60.0
            datos["info"]["satelite_edad_min"] = None if edad_sat is None else round(edad_sat, 1)
            if edad_sat is not None and edad_sat <= EDAD_SATELITE_MAX_MIN + 5:
                datos["satelite"] = ct.grilla_satelite(z["lat"], z["lon"], z["temp_c"], shape)
    except Exception as exc:
        datos["info"]["satelite_error"] = f"{type(exc).__name__}: {str(exc)[:100]}"
    ventana = ((goes.get("glm") or {}).get("ventana")) or {}
    e = edad_min(ventana.get("fin_ventana_utc"), ahora)
    datos["info"]["rayos_edad_min"] = e
    if ventana.get("disponible") is not False and e is not None and e <= EDAD_RAYOS_MAX_MIN:
        pos = ventana.get("rayos_10min") or []
        datos["rayos_lat"] = [p[0] for p in pos]
        datos["rayos_lon"] = [p[1] for p in pos]
        datos["info"]["rayos"] = len(pos)
    return datos


C13_AHORA_PREVIA = Path(__import__("tempfile").gettempdir()) / "climaar_c13_ahora_previa.npz"
DIF_SATELITE_RADAR_MAX_MIN = 6.0


def _leer_c13(path: Path, shape):
    if not path.exists():
        return None, None
    z = np.load(path)
    return ct.grilla_satelite(z["lat"], z["lon"], z["temp_c"], shape), parse_utc(str(z["hora"]))


def satelites_etapa_a(radar_dt: datetime, ahora: datetime, shape) -> tuple[dict | None, dict[str, Any]]:
    """Barrido de la hora del radar, el más nuevo y el anterior a ese (los deja goes19/state.py
    en el mismo ciclo). Las edades se calculan AHORA con la hora de cada barrido."""
    info: dict[str, Any] = {}
    try:
        nuevo, h_n = _leer_c13(C13_AHORA, shape)
        if nuevo is None or h_n is None:
            return None, {"motivo": "sin barrido nuevo"}
        info["edad_barrido_nuevo_min"] = round((ahora - h_n).total_seconds() / 60.0, 1)
        if not (-1.0 <= info["edad_barrido_nuevo_min"] <= EDAD_SATELITE_MAX_MIN + 5):
            return None, {**info, "motivo": "barrido nuevo viejo o del futuro"}
        radar, h_r = _leer_c13(C13_RADAR, shape)
        if h_r is None or abs((h_r - radar_dt).total_seconds()) / 60.0 > DIF_SATELITE_RADAR_MAX_MIN:
            radar = None
        previo, h_p = _leer_c13(C13_AHORA_PREVIA, shape)
        entre = None if h_p is None else (h_n - h_p).total_seconds() / 60.0
        if entre is not None and not (5.0 <= entre <= 15.0):
            previo, entre = None, None
        tramo = (h_n - radar_dt).total_seconds() / 60.0
        info.update({"tramo_min": round(tramo, 1), "barrido_radar": radar is not None,
                     "entre_barridos_min": None if entre is None else round(entre, 1)})
        return {"radar": radar, "nuevo": nuevo, "previo": previo, "tramo": tramo, "entre": entre}, info
    except Exception as exc:
        return None, {"motivo": f"{type(exc).__name__}: {str(exc)[:100]}"}


def preparar_satelite(campo: np.ndarray, flujo: dict, radar_dt: datetime, ahora: datetime, sat_cfg: dict,
                      goes: dict, salida: dict) -> dict | None:
    """Etapa A, puntos 2-4: radar llevado a la hora del barrido más nuevo con lo que midió el
    satélite. Falla segura: None y el motivo en salida["satelite_etapa_a"]."""
    sats, info = satelites_etapa_a(radar_dt, ahora, campo.shape)
    info = {"activo": bool(sat_cfg.get("activo")), **info}
    prep = None
    if sats is not None:
        try:
            prep = cs.preparar(campo, flujo["u"], flujo["v"], sats["radar"], sats["nuevo"], sats["previo"],
                               sats["tramo"], sats["entre"], flujo["km_px"], sat_cfg.get("parametros"), None, info)
            if not (info.get("posicion") or info.get("movimiento") or info.get("crecimiento")):
                prep = None
                info.setdefault("motivo", "el satélite no agrega nada en este ciclo (barrido de la misma hora o sin topes fríos)")
        except Exception as exc:
            prep = None
            info["motivo"] = f"{type(exc).__name__}: {str(exc)[:100]}"
    info["usado"] = prep is not None
    salida["satelite_etapa_a"] = info
    return prep


def datos_estaciones(senales: dict) -> dict | None:
    """Lluvia de cada estación (historial de 6 h) con su posición; sin las de lluvia dudosa."""
    import corrector_estaciones as ce

    try:
        sys.path.insert(0, str(ROOT / "scripts" / "estaciones"))
        import senales_superficie as ss

        hist = leer_json(ss.HISTORIAL)
        dudosas = [e.get("nombre") for e in senales.get("estaciones") or []
                   if e.get("calidad_lluvia") == "dudosa" or e.get("calidad") == "dudosa"]
        return ce.desde_historial(hist, ss.coordenadas(), dudosas) if hist else None
    except Exception:
        return None


def con_corrector(cfg: dict, campos: dict[float, np.ndarray], edad: float, flujo: dict | None,
                  datos: dict, salida: dict | None) -> dict[float, np.ndarray]:
    """Corrige con datos medidos de AHORA. Para los plazos futuros, el satélite y los rayos
    se mueven con la tormenta (mismo movimiento del radar). Falla segura."""
    if not cfg.get("corrector") or not campos:
        return campos
    try:
        reglas = cfg.get("reglas_corrector") or {}
        sat, rlat, rlon = datos["satelite"], np.asarray(datos["rayos_lat"]), np.asarray(datos["rayos_lon"])
        frio = None if sat is None else np.where(np.isfinite(sat), np.maximum(-sat, 0.0), 0.0).astype(np.float32)
        nuevos, total = {}, {}
        for t, c in campos.items():
            p = float(t) - edad
            sat_t, lat_t, lon_t = sat, rlat, rlon
            if p > 0.5 and flujo is not None:
                if frio is not None:
                    movido = nc.serie(frio, flujo["u"], flujo["v"], [p])[p]
                    sat_t = np.where(movido > 0, -movido, 5.0).astype(np.float32)
                if rlat.size:
                    h, w = c.shape
                    x, y = ct.a_pixel(rlat, rlon, w, h)
                    xi = np.clip(np.rint(x).astype(int), 0, w - 1)
                    yi = np.clip(np.rint(y).astype(int), 0, h - 1)
                    x2, y2 = x + flujo["u"][yi, xi] * p, y + flujo["v"][yi, xi] * p
                    lat_t, lon_t = pixel_a_latlon(x2, y2, w, h)
            nuevos[t], cambios = ct.corregir(c, sat_t, lat_t, lon_t, reglas)
            for k, v in cambios.items():
                total[k] = total.get(k, 0) + v
        if salida is not None:
            salida["corrector"] = {"usado": True, "pixeles_cambiados": total, **datos["info"]}
        return nuevos
    except Exception as exc:
        if salida is not None:
            salida["corrector"] = {"usado": False, "motivo": f"{type(exc).__name__}: {str(exc)[:120]}"}
        return campos


def pixel_a_latlon(x, y, w: int, h: int):
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    lon = ct.RADAR_LON + (np.asarray(x) - cx) / cx * ct.RADAR_RADIUS_KM / (111.320 * math.cos(math.radians(ct.RADAR_LAT)))
    lat = ct.RADAR_LAT - (np.asarray(y) - cy) / cy * ct.RADAR_RADIUS_KM / 110.574
    return lat, lon


def con_anvil(cfg: dict, flujo: dict, campos: dict[float, np.ndarray], salida: dict | None):
    """Punto 6: suma ANVIL (crecimiento y debilitamiento) si está encendido. Falla segura."""
    if not cfg.get("anvil") or not flujo.get("tres"):
        if salida is not None:
            salida["anvil"] = {"usado": False, "motivo": "apagado" if not cfg.get("anvil") else "faltan 3 imágenes"}
        return campos
    try:
        cache = flujo.setdefault("anvil_cache", {})
        faltan = [t for t in campos if float(t) > nc.ANVIL_DESDE_MIN and float(t) not in cache]
        if faltan:
            cache.update(nc.anvil(flujo["tres"], faltan))
        anv = {t: cache[float(t)] for t in campos if float(t) in cache}
        if salida is not None:
            salida["anvil"] = {"usado": True, "desde_min": nc.ANVIL_DESDE_MIN}
        return nc.combinar_anvil(campos, anv)
    except Exception as exc:  # sin pysteps o error: siguen los escenarios solos
        if salida is not None:
            salida["anvil"] = {"usado": False, "motivo": f"{type(exc).__name__}: {str(exc)[:120]}"}
        return campos


def movimiento(radar_dt: datetime, campo: np.ndarray, actual_dbz: np.ndarray, clutter: np.ndarray,
               cfg: dict) -> dict[str, Any] | None:
    """Campo de movimiento del nowcast en capas (px/min): flujo óptico con la
    imagen anterior, controlado contra velocidades imposibles y mezclado con el
    par anterior (3 imágenes). Lo usan el nowcast y las trayectorias por tormenta."""
    previo = nf.frame_previo(radar_dt) if int((campo >= nf.UMBRAL_ECO_DBZ).sum()) >= 30 else None
    if previo is None:
        return None
    prev_dbz = nf.cargar_dbz(previo[0])
    if prev_dbz is None or prev_dbz.shape != actual_dbz.shape:
        return None
    gap = (radar_dt - previo[1]).total_seconds() / 60.0
    prev = nc.campo_limpio(prev_dbz, clutter)
    km_px = nf.km_por_pixel(actual_dbz.shape[1])
    f1 = nf.estimar_flujo(prev, campo, gap, km_px)
    if not f1["valido"]:
        return None
    try:
        previas = nf.velocidades_previas(previo[1], prev_dbz, clutter)
    except Exception:
        previas = []
    f1 = nf.controlar_velocidad(f1, previas)
    u, v = f1["u"], f1["v"]
    prev2 = None
    if cfg.get("movimiento_3_imagenes") or cfg.get("anvil"):
        pp = nf.frame_previo(previo[1])
        if pp is not None:
            ppd = nf.cargar_dbz(pp[0])
            if ppd is not None and ppd.shape == actual_dbz.shape:
                prev2 = nc.campo_limpio(ppd, clutter)
                f2 = nf.estimar_flujo(prev2, prev, (previo[1] - pp[1]).total_seconds() / 60.0, km_px)
                if f2["valido"] and cfg.get("movimiento_3_imagenes"):
                    u = (0.6 * u + 0.4 * f2["u"]).astype(np.float32)
                    v = (0.6 * v + 0.4 * f2["v"]).astype(np.float32)
    tasa = nc.tendencia(prev, campo, u, v, gap) if cfg.get("tendencia_radar") else None
    return {"u": u, "v": v, "tasa": tasa, "tres": [prev2, prev, campo] if prev2 is not None else None,
            "velocidad_kmh": f1["velocidad_media_kmh"], "direccion_hacia_grados": f1["direccion_hacia_grados"],
            "coherencia": f1["coherencia"], "intervalo_min": round(gap, 1), "km_px": km_px}


def ejecutar(ahora: datetime | None = None) -> dict[str, Any]:
    ahora = ahora or datetime.now(timezone.utc)
    cfg = {**CONFIG_DEFECTO, **leer_json(CONFIG)}
    goes, senales, meteo = leer_json(GOES), leer_json(SENALES), leer_json(METEO)
    localidades = lr.cargar_localidades()
    nombres = {l["id"]: l["nombre"] for l in localidades}

    radar_ts = nf.TIMESTAMP_FILE.read_text(encoding="utf-8").strip() if nf.TIMESTAMP_FILE.exists() else None
    radar_dt = parse_utc(radar_ts) if radar_ts else None
    edad = round(max(0.0, (ahora - radar_dt).total_seconds() / 60.0), 1) if radar_dt else None
    actual_dbz = nf.cargar_dbz(nf.ACTUAL) if (radar_dt and nf.ACTUAL.exists()) else None

    estable = capa_estable(meteo)
    firmas = firmas_localidades(senales)
    tormentas = tendencia_tormentas(goes)
    salida: dict[str, Any] = {
        "version": VERSION, "en_paralelo": False, "generado_utc": iso(ahora),
        # Desde el 20/11/2026 se usa para lo que se ve (motor_decision_climaar.py).
        "en_uso": ("llegadas a la zona de cada localidad (tormenta a <= 10 km, grado probable) "
                   "y movimiento de las trayectorias por tormenta"),
        "radar_timestamp": radar_ts, "edad_radar_min": edad,
        "confianza_radar_por_edad": nc.confianza_por_edad(edad) if edad is not None else 0.0,
        "configuracion": cfg,
        "fuentes": {
            "radar": {"hora": radar_ts, "edad_min": edad, "fuente": "SINARAME RMA10"},
            "rayos": {"hora": ((goes.get("glm") or {}).get("ventana") or {}).get("fin_ventana_utc"),
                      "fuente": "GOES-19 GLM"},
            "satelite": {"hora": (goes.get("satelite_ahora") or {}).get("hora_satelite_utc"),
                         "fuente": "GOES-19 canal 13"},
            "estaciones": {"hora": senales.get("generado_utc"), "fuente": "BCP / SMN / CWOP"},
            "ambiente": {"hora": (meteo.get("ambiente_actual") or {}).get("tiempo_utc"), "fuente": "modelos"},
        },
        "suelo": {"capa_estable": estable},
        "tormentas": tormentas,
    }

    flujo = None
    campo = None
    if actual_dbz is not None:
        clutter = nc.cargar_clutter_total(actual_dbz.shape)
        campo = nc.campo_limpio(actual_dbz, clutter)
        if cfg.get("control_calidad_satelite"):
            campo = control_calidad(campo, radar_dt, salida)
        flujo = movimiento(radar_dt, campo, actual_dbz, clutter, cfg)
    salida["movimiento"] = None if flujo is None else {
        k: flujo[k] for k in ("velocidad_kmh", "direccion_hacia_grados", "coherencia", "intervalo_min")}
    if campo is None:
        salida["motivo_sin_radar"] = "SIN_RADAR_ACTUAL"

    zl = zonas_localidades(campo.shape, localidades) if campo is not None else {}
    zr = zonas_region(campo.shape) if campo is not None else {}
    tiempos = [edad + p for p in PLAZOS_DESDE_AHORA] if edad is not None else []
    campos: dict[float, np.ndarray] = {}
    datos_corr = None
    # Etapa A: satélite más nuevo que el radar y lluvia de las estaciones.
    sat_cfg = cfg.get("satelite_etapa_a") or {}
    est_cfg = cfg.get("estaciones_etapa_a") or {}
    prep_sat = None
    if campo is not None and flujo is not None and (sat_cfg.get("activo") or sat_cfg.get("en_paralelo")):
        prep_sat = preparar_satelite(campo, flujo, radar_dt, ahora, sat_cfg, goes, salida)
    usar_sat = bool(prep_sat is not None and sat_cfg.get("activo") and cfg.get("escenarios"))
    est_datos = datos_estaciones(senales) if (campo is not None and (est_cfg.get("activo") or est_cfg.get("en_paralelo"))) else None
    usar_est = bool(est_datos and est_cfg.get("activo"))

    def estaciones(c: dict[float, np.ndarray], info: dict | None = None) -> dict[float, np.ndarray]:
        import corrector_estaciones as ce
        try:
            return ce.corregir_campos(c, est_datos, ahora, edad, flujo, campo.shape,
                                      est_cfg.get("parametros"), campo, info)
        except Exception as exc:
            if info is not None:
                info.update({"usado": False, "motivo": f"{type(exc).__name__}: {str(exc)[:100]}"})
            return c

    if est_datos:
        info_est: dict[str, Any] = {}
        estaciones({float(edad): campo}, info_est)   # solo para informar qué haría ahora
        salida["corrector_estaciones"] = {"activo": usar_est, **{k: info_est.get(k) for k in
                                          ("usado", "estaciones", "agregadas", "bajadas", "motivo") if k in info_est}}
    if campo is not None:
        if usar_sat:
            campos = cs.campos(prep_sat, tiempos)
        elif flujo is not None and cfg.get("escenarios"):
            campos = nc.campos_escenarios(campo, flujo["u"], flujo["v"], tiempos, flujo["tasa"])
            campos = con_anvil(cfg, flujo, campos, salida)
        elif flujo is not None:
            campos = nc.serie(campo, flujo["u"], flujo["v"], tiempos, flujo["tasa"])
        else:
            campos = {float(t): campo for t in tiempos}  # sin movimiento: persistencia (se dice)
        if usar_est:
            campos = estaciones(campos)
        # Corrector con rayos y satélite medidos ahora (agrega y quita).
        datos_corr = datos_corrector(goes, ahora, campo.shape) if cfg.get("corrector") else None
        campos = con_corrector(cfg, campos, edad, flujo, datos_corr, salida) if datos_corr else campos
    picos_loc = {p: nc.picos_por_zona(campos[float(edad + p)], zl) for p in PLAZOS_DESDE_AHORA} if campos else {}
    # ¿Llega a mi localidad en la próxima hora? "probable" = campo de hoy; "posible" = envolvente.
    llegada_loc: dict[str, dict[str, Any]] = {}
    env_cfg = cfg.get("llegada_envolvente") or {}
    if campo is not None and edad is not None:
        try:
            t_lleg = [float(edad + p) for p in PLAZOS_LLEGADA]
            if usar_sat:
                prob = cs.campos(prep_sat, t_lleg)
            elif flujo is not None and cfg.get("escenarios"):
                prob = nc.campos_escenarios(campo, flujo["u"], flujo["v"], t_lleg, flujo["tasa"])
            elif flujo is not None:
                prob = nc.serie(campo, flujo["u"], flujo["v"], t_lleg, flujo["tasa"])
            else:
                prob = {t: campo for t in t_lleg}
            grados = {"probable": prob}
            if env_cfg.get("activa") and flujo is not None:
                # Validado (herramientas/simular_envolvente.py, 684 casos): con 25 caminos,
                # "probable" = 2 de 25 y "posible" = 1 de 25 (algún camino la trae).
                kp, kq = int(env_cfg.get("k_probable", 2)), int(env_cfg.get("k_posible", 1))
                pilas = cs.campos(prep_sat, t_lleg, "envolvente", (kp, kq)) if usar_sat else \
                    nc.campos_envolvente(campo, flujo["u"], flujo["v"], t_lleg, (kp, kq), flujo["tasa"])
                env = {"probable": pilas[kp], "posible": pilas[kq]}
                grados = {"probable": {t: np.maximum(env["probable"][t], prob[t]) for t in t_lleg}}
                # "posible" nunca dice menos que "probable".
                grados["posible"] = {t: np.maximum(env["posible"][t], grados["probable"][t]) for t in t_lleg}
            if usar_est:
                grados = {g: estaciones(c) for g, c in grados.items()}
            if datos_corr:
                grados = {g: con_corrector(cfg, c, edad, flujo, datos_corr, None) for g, c in grados.items()}
            grados = {g: {p: c[float(edad + p)] for p in PLAZOS_LLEGADA} for g, c in grados.items()}
            zl10 = zonas_localidades(campo.shape, localidades, MARGEN_LLEGADA_KM)
            llegada_loc = llegadas(grados, {"localidad": zl, "hasta_10km": zl10})
            salida["llegada_proxima_hora"] = {
                "usado": True, "grados": sorted(grados),
                "caminos": len(nc.ESCENARIOS_AMPLIOS) if "posible" in grados else len(nc.ESCENARIOS),
                "k_probable": int(env_cfg.get("k_probable", 2)) if "posible" in grados else None,
                "k_posible": int(env_cfg.get("k_posible", 1)) if "posible" in grados else None,
                # Simulado con el radar atrasado 10 a 90 min; más viejo que eso no está probado.
                "fuera_de_lo_validado": bool(edad > 90.0),
                "nota": "Minutos desde ahora hasta la primera llegada (null = no llega en la próxima hora). "
                        "'probable' = 2 de 25 caminos posibles la traen (menos falsas alarmas); 'posible' = "
                        "algún camino la trae (detecta más, con más falsas alarmas). Validado con el radar "
                        "atrasado 10 a 90 min (docs/VALIDACION_NOWCAST_CAPAS.md).",
            }
        except Exception as exc:
            salida["llegada_proxima_hora"] = {"usado": False, "motivo": f"{type(exc).__name__}: {str(exc)[:120]}"}
    rayos_zona = None
    if datos_corr and datos_corr.get("rayos_lat") and campo is not None:
        try:
            ze = ct.zona_electrica(datos_corr["rayos_lat"], datos_corr["rayos_lon"], campo.shape)
            rayos_zona = {k: bool(ze[z].any()) for k, z in zl.items()}
        except Exception:
            rayos_zona = None

    llegada = {}
    if flujo is not None:
        llegada = nc.escenarios_llegada(campo, flujo["u"], flujo["v"], zl, edad, edad + 60.0,
                                        UMBRAL_TORMENTA, flujo["tasa"])

    locs = {}
    for loc in localidades:
        k = loc["id"]
        rell = relleno_localidad(k, goes, senales, ahora)
        dbz_ahora = picos_loc[0][k] if picos_loc else None
        item: dict[str, Any] = {"nombre": nombres[k], "relleno_tiempo_real": rell}
        item["radar_movido_a_ahora"] = None if dbz_ahora is None else {
            "dbz": round(dbz_ahora, 1), "categoria": categoria(dbz_ahora),
            "movido_min": edad if flujo is not None else 0.0,
            "fuente": "radar SINARAME movido a la hora actual" if flujo is not None
            else "radar SINARAME sin mover (no hay movimiento calculable)"}
        item["estado_ahora"] = estado_ahora(dbz_ahora, edad, rell)
        if picos_loc:
            item["pronostico_desde_ahora"] = {
                f"{p}min": {"dbz": round(picos_loc[p][k], 1), "categoria": categoria(picos_loc[p][k])}
                for p in PLAZOS_DESDE_AHORA if p > 0}
        if k in llegada:
            l = llegada[k]
            item["tormenta_proxima_hora"] = {
                "porcentaje_escenarios": l["porcentaje"],
                "minutos_desde_ahora": None if l["minutos_desde_imagen"] is None else round(l["minutos_desde_imagen"] - edad, 1),
                "rango_min": None if l["primera_min"] is None else [round(l["primera_min"] - edad, 1), round(l["ultima_min"] - edad, 1)],
            }
        if k in llegada_loc:
            item["llegada_proxima_hora"] = llegada_loc[k]
        if rayos_zona is not None:
            # Rayos solos = tormenta eléctrica (no dice nada de lluvia ni viento).
            item["tormenta_electrica_ahora"] = rayos_zona.get(k, False)
        item["suelo"] = {"firmas_estaciones": firmas.get(k, []),
                         "conveccion_elevada": estable["conveccion_elevada"]}
        item["confianza"] = round(nc.confianza_por_edad(edad) * (0.4 + 0.6 * flujo["coherencia"]), 3) \
            if (flujo is not None and edad is not None) else (0.3 if campo is not None else 0.0)
        locs[k] = item
    salida["localidades"] = locs

    # Capa 6: validación automática.
    verif = leer_json(VERIFICACION) or {"version": VERSION, "pendientes": [], "acumulado": {}, "ultimas": []}
    if campo is not None:
        obs_loc = nc.picos_por_zona(campo, zl)
        obs_reg = nc.picos_por_zona(campo, zr)
        if verif.get("ultima_imagen_verificada") != radar_ts:
            verificar(verif, radar_dt, obs_loc, obs_reg)
            zl10_obs = zonas_localidades(campo.shape, localidades, MARGEN_LLEGADA_KM)
            verificar_llegadas(verif, radar_dt, {"localidad": obs_loc,
                                                 "hasta_10km": nc.picos_por_zona(campo, zl10_obs)})
            verif["ultima_imagen_verificada"] = radar_ts
        # Guardar el pronóstico de llegada (uno por imagen de radar: el primero que se emite).
        if llegada_loc and edad is not None:
            emitidos = {p.get("radar_utc") for p in verif.get("llegadas_pendientes") or []}
            if radar_ts not in emitidos:
                verif.setdefault("llegadas_pendientes", []).append({
                    "emitido_utc": iso(ahora), "radar_utc": radar_ts, "edad_radar_min": edad,
                    "desde_utc": iso(ahora), "hasta_utc": iso(ahora + timedelta(minutes=60)),
                    "pronostico": llegada_loc})
        # Guardar los pronósticos para las próximas horas de imagen (cada 10 min).
        if campos:
            base = radar_dt
            emit = {(p.get("emitido_utc"), p.get("objetivo_utc")) for p in verif.get("pendientes") or []}
            objetivos = []
            for n in range(1, 10):
                obj = base + timedelta(minutes=10 * n)
                plazo = (obj - ahora).total_seconds() / 60.0
                if 0 <= plazo <= 90 and (iso(ahora), iso(obj)) not in emit:
                    objetivos.append((obj, plazo, 10.0 * n))
            t_imgs = [t for _, _, t in objetivos]
            extra: dict[str, Any] = {}
            if not t_imgs:
                futuros = {}
            elif flujo is None:
                futuros = {t: campo for t in t_imgs}
            elif cfg.get("escenarios"):
                # Sin la etapa A (base) y, aparte, con cada parte de la etapa A (activa o en paralelo).
                base_f = con_anvil(cfg, flujo, nc.campos_escenarios(campo, flujo["u"], flujo["v"], t_imgs,
                                                                    flujo["tasa"]), None)
                sat_f = cs.campos(prep_sat, t_imgs) if prep_sat is not None else None
                corr = datos_corr or (datos_corrector(goes, ahora, campo.shape) if cfg.get("corrector") else None)

                def terminar(c):
                    return con_corrector(cfg, c, edad, flujo, corr, None) if (cfg.get("corrector") and corr) else c

                futuros = terminar(estaciones(sat_f if usar_sat else base_f) if usar_est else (sat_f if usar_sat else base_f))
                if prep_sat is not None:
                    extra["satelite"] = terminar(sat_f) if not usar_sat else futuros
                    extra["sin_satelite"] = terminar(estaciones(base_f) if usar_est else base_f) if usar_sat else None
                if est_datos:
                    extra["estaciones"] = terminar(estaciones(sat_f if usar_sat else base_f)) if not usar_est else futuros
            else:
                futuros = nc.serie(campo, flujo["u"], flujo["v"], t_imgs, flujo["tasa"])
            for obj, plazo, t_img in objetivos:
                cmp_ = futuros[t_img]
                verif.setdefault("pendientes", []).append({
                    "emitido_utc": iso(ahora), "radar_utc": radar_ts, "edad_radar_min": edad,
                    "objetivo_utc": iso(obj), "plazo_min": round(t_img, 1),
                    "hoy": {"loc": {k: round(x, 1) for k, x in obs_loc.items()},
                            "reg": [int(round(obs_reg[z])) for z in sorted(obs_reg)]},
                    "capas": {"loc": {k: round(x, 1) for k, x in nc.picos_por_zona(cmp_, zl).items()},
                              "reg": [int(round(x)) for _, x in sorted(nc.picos_por_zona(cmp_, zr).items())]},
                    # Etapa A en vivo: lo mismo con y sin cada corrección (se guarda solo si cambia
                    # algo; si no, "etapa_a" dice que se calculó y fue igual a "capas").
                    "etapa_a": [m for m, c in extra.items() if c is not None],
                    **{m: {"loc": {k: round(x, 1) for k, x in nc.picos_por_zona(c[t_img], zl).items()},
                           "reg": [int(round(x)) for _, x in sorted(nc.picos_por_zona(c[t_img], zr).items())]}
                       for m, c in extra.items() if c is not None and not np.array_equal(c[t_img], cmp_)},
                })
        # Tope de tamaño: solo pendientes dentro de la ventana.
        verif["pendientes"] = [p for p in verif.get("pendientes") or []
                               if (parse_utc(p["objetivo_utc"]) - ahora).total_seconds() / 60.0 > -PENDIENTES_MAX_MIN][-60:]
    verif["actualizado_utc"] = iso(ahora)
    verif["resumen"] = resumen_verificacion(verif)
    verif["resumen_llegadas"] = resumen_llegadas(verif)
    salida["validacion_automatica"] = {
        "resumen": verif["resumen"],
        "llegadas_proxima_hora": verif["resumen_llegadas"],
        "nota": "'hoy' = imagen vieja tal como llegó; 'capas' = nowcast en capas. CSI más alto = mejor.",
    }
    salida["_verificacion"] = verif
    return salida


def escribir_compacto(path: Path, data: dict) -> None:
    """Sin espacios ni sangría: el archivo se guarda en git en cada ciclo (cuida el tamaño)."""
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as h:
            json.dump(data, h, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main(argv=None) -> int:
    ahora = parse_utc(os.environ["CLIMAAR_AHORA_UTC"]) if os.environ.get("CLIMAAR_AHORA_UTC") else None
    try:
        salida = ejecutar(ahora)
        verif = salida.pop("_verificacion", None)
        if verif is not None:
            escribir_compacto(VERIFICACION, verif)
    except Exception as exc:  # nunca corta el ciclo
        salida = {"version": VERSION, "en_paralelo": True, "disponible": False,
                  "motivo": f"ERROR: {type(exc).__name__}: {str(exc)[:200]}",
                  "generado_utc": iso(datetime.now(timezone.utc))}
    salida.setdefault("disponible", "localidades" in salida)
    nf.escribir_json_atomico(SALIDA, salida)
    resumen = {k: salida.get(k) for k in ("disponible", "radar_timestamp", "edad_radar_min", "motivo")}
    resumen["localidades"] = {k: v["estado_ahora"]["categoria"] for k, v in (salida.get("localidades") or {}).items()}
    print(json.dumps(resumen, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
