#!/usr/bin/env python3
"""Índices por fenómeno para Bahía Blanca (reglas físicas, sin entrenamiento).

Cada fenómeno se evalúa por separado y puede estar o no estar:

  lluvia     0 sin lluvia · 1 débil (llovizna) · 2 moderada · 3 fuerte · 4 severa
             (severa = lluvia fuerte que dura o acumula: riesgo de anegamientos)
  viento     0 sin viento destacable · 2 moderado · 3 fuerte · 4 severo
  granizo    0 sin señal · 2 posible · 3 probable · 4 severo
  electrica  0 sin rayos · 1 aislada · 3 frecuente · 4 severa

El grado (0-4) es común a todos para poder compararlos: la calificación
general es el grado más alto y dice de qué fenómeno se trata. El resumen
informa la combinación real ("Viento fuerte, sin lluvia").

Cada fenómeno se informa "ahora" (sobre la ciudad) y "al llegar" (lo que
trae el sistema que se acerca en <= 60 min).

Grado 4 sin confirmación de una estación se informa como "posible severo"
hasta calibrar los cortes con datos de SINARAME (decisión del 19/11/2026).
No es un sistema de avisos: los avisos oficiales son del SMN.
"""
from __future__ import annotations

import math
from typing import Any

# --------------------------------------------------------------------------
# Cortes (revisables). mm/h, km/h, dBZ.
# --------------------------------------------------------------------------
LLUVIA_ECO_DBZ = 15.0            # menos que esto no es lluvia que llegue al suelo
LLUVIA_LLOVIZNA_MM_H = 1.0
LLUVIA_MODERADA_MM_H = 3.0
LLUVIA_FUERTE_MM_H = 12.0
LLUVIA_SEVERA_MM_H = 50.0
LLUVIA_DBZ_TOPE = 53.0           # por encima suele ser granizo: no suma lluvia
LLUVIA_SEVERA_1H_MM = 30.0
LLUVIA_SEVERA_3H_MM = 50.0
LLUVIA_RADAR_3H_MM = 30.0        # el radar subestima: cortes del 7/3/2025
LLUVIA_RADAR_6H_MM = 50.0
LLUVIA_FUERTE_SOSTENIDA_MIN = 60.0

VIENTO_MODERADO_KMH = 30.0
VIENTO_FUERTE_KMH = 50.0
VIENTO_SEVERO_KMH = 90.0
VIENTO_MEDIO_FUERTE_KMH = 40.0   # viento medio (no ráfaga) que ya es fuerte
LINEA_RAPIDA_KMH = 60.0
LINEA_COHERENCIA = 0.85
LINEA_DBZ = 45.0

GRANIZO_POSIBLE_DBZ = 50.0
GRANIZO_PROBABLE_DBZ = 55.0
GRANIZO_SEVERO_DBZ = 60.0
ISOTERMA_CERO_ALTA_M = 4500.0    # granizo chico se derrite antes de llegar

RAYOS_CERCA_KM = 30.0
RAYOS_FRECUENTES_MIN = 2.0       # rayos por minuto a <= 30 km
RAYOS_SALTO_FACTOR = 3.0         # aumento brusco frente a los 10 min previos
RAYOS_SALTO_MIN = 1.0            # ...con al menos 1 rayo por minuto

LLEGADA_MAX_MIN = 60.0

# GOES-19 canal 13 (topes de nubes).
TOPE_MUY_FRIO_C = -60.0          # tormenta muy profunda: granizo más probable
TOPE_TORMENTA_C = -50.0          # nube de tormenta (sin radar: lluvia probable)
ENFRIAMIENTO_RAPIDO_C = 8.0      # °C cada 10 min: la tormenta crece rápido

NOMBRES = {
    "lluvia": "lluvia",
    "viento": "viento",
    "granizo": "granizo",
    "electrica": "tormenta eléctrica",
}

CATEGORIAS = {
    "lluvia": {0: "sin lluvia", 1: "débil", 2: "moderada", 3: "fuerte", 4: "severa"},
    "viento": {0: "sin viento destacable", 2: "moderado", 3: "fuerte", 4: "severo"},
    "granizo": {0: "sin señal", 2: "posible", 3: "probable", 4: "severo"},
    "electrica": {0: "sin rayos", 1: "aislada", 3: "frecuente", 4: "severa"},
}

GENERAL = {0: "Sin fenómenos", 1: "Débil", 2: "Moderada", 3: "Fuerte", 4: "Severa"}


def numero(valor: Any) -> float | None:
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def mm_h_desde_dbz(dbz: float | None) -> float:
    """Marshall-Palmer (Z = 200 R^1.6) con tope de 53 dBZ (granizo)."""
    if dbz is None or dbz < LLUVIA_ECO_DBZ:
        return 0.0
    dbz = min(dbz, LLUVIA_DBZ_TOPE)
    return (10 ** (dbz / 10.0) / 200.0) ** (1 / 1.6)


def _evaluacion(fenomeno: str, grado: int, fuente: str, razones: list[str],
                confirmado: bool, extra: dict | None = None) -> dict[str, Any]:
    categoria = CATEGORIAS[fenomeno].get(grado, "")
    posible = grado == 4 and not confirmado
    salida = {
        "grado": grado,
        "categoria": ("posible " + categoria) if posible else categoria,
        "confirmado": confirmado,
        "en_calibracion": posible,
        "fuente": fuente,
        "razones": razones,
    }
    if extra:
        salida.update(extra)
    return salida


def _mejor(*opciones: dict[str, Any]) -> dict[str, Any]:
    """La evaluación de mayor grado; ante igualdad, la confirmada."""
    return max(opciones, key=lambda e: (e["grado"], e["confirmado"]))


# --------------------------------------------------------------------------
# Lluvia
# --------------------------------------------------------------------------
def grado_lluvia_mm_h(mm_h: float) -> int:
    """Intensidad instantánea. Llega hasta "fuerte": la lluvia severa (la que
    anega) depende de cuánto dura y cuánto acumula, no de un instante."""
    if mm_h <= 0:
        return 0
    if mm_h < LLUVIA_MODERADA_MM_H:
        return 1
    if mm_h < LLUVIA_FUERTE_MM_H:
        return 2
    return 3


def lluvia_por_radar(dbz: float | None, *, minutos_intensa: float | None = None,
                     l3: float | None = None, l6: float | None = None,
                     mm_h_area: float | None = None, l1: float | None = None) -> dict[str, Any]:
    """Con mm_h_area (lluvia media sobre la localidad) se usa esa intensidad:
    representa mejor lo que cae que el píxel más intenso."""
    area = numero(mm_h_area)
    mm_h = area if area is not None else mm_h_desde_dbz(dbz)
    grado = grado_lluvia_mm_h(mm_h if mm_h >= 0.1 else 0.0)
    razones = []
    if grado:
        if area is not None:
            razones.append(f"~{mm_h:.1f} mm/h en promedio sobre la zona (radar)")
        else:
            razones.append(f"{dbz:.0f} dBZ en el radar (~{mm_h:.0f} mm/h)" if mm_h >= 1
                           else f"{dbz:.0f} dBZ en el radar (llovizna)")
    intensa = numero(minutos_intensa) or 0.0
    l1 = numero(l1) or 0.0
    l3 = numero(l3) or 0.0
    l6 = numero(l6) or 0.0
    if l1 >= LLUVIA_SEVERA_1H_MM:
        grado = 4
        razones.append(f"~{l1:.0f} mm en la última hora estimados por radar (riesgo de anegamientos)")
    if grado >= 3 and intensa >= LLUVIA_FUERTE_SOSTENIDA_MIN:
        grado = 4
        razones.append(f"lluvia fuerte sostenida {intensa:.0f} min (riesgo de anegamientos)")
    if l3 >= LLUVIA_RADAR_3H_MM or l6 >= LLUVIA_RADAR_6H_MM:
        grado = 4
        razones.append(f"acumulado estimado por radar {l3:.0f} mm en 3 h, {l6:.0f} mm en 6 h "
                       "(el radar suele subestimarlo; riesgo de anegamientos)")
    extra = {"mm_h": round(mm_h, 1)}
    if mm_h >= LLUVIA_SEVERA_MM_H:
        extra["detalle"] = "torrencial"
    if grado == 1 and mm_h < LLUVIA_LLOVIZNA_MM_H:
        extra["detalle"] = "llovizna"
    return _evaluacion("lluvia", grado, "radar", razones, False, extra)


def lluvia_por_estaciones(sup: dict[str, Any]) -> dict[str, Any] | None:
    l1 = numero(sup.get("lluvia_1h_mm"))
    l3 = numero(sup.get("lluvia_3h_mm"))
    if l1 is None and l3 is None:
        return None
    l1 = l1 or 0.0
    l3 = l3 or 0.0
    grado = grado_lluvia_mm_h(l1)
    razones = [f"{l1:.1f} mm en la última hora medidos en Bahía Blanca"] if l1 > 0 else []
    if l1 >= LLUVIA_SEVERA_1H_MM or l3 >= LLUVIA_SEVERA_3H_MM:
        grado = 4
        razones.append(f"medido: {l1:.0f} mm en 1 h, {l3:.0f} mm en 3 h (riesgo de anegamientos)")
    extra = {"mm_1h": round(l1, 1), "mm_3h": round(l3, 1)}
    if grado == 1 and l1 < LLUVIA_LLOVIZNA_MM_H:
        extra["detalle"] = "llovizna"
    return _evaluacion("lluvia", grado, "estaciones", razones, True, extra)


# --------------------------------------------------------------------------
# Viento
# --------------------------------------------------------------------------
def grado_viento_kmh(kmh: float | None) -> int:
    if kmh is None or kmh < VIENTO_MODERADO_KMH:
        return 0
    if kmh < VIENTO_FUERTE_KMH:
        return 2
    if kmh < VIENTO_SEVERO_KMH:
        return 3
    return 4


def viento_por_estaciones(sup: dict[str, Any]) -> dict[str, Any] | None:
    rafaga = numero(sup.get("rafaga_max_kmh"))
    medio = numero(sup.get("viento_max_kmh"))
    if rafaga is None and medio is None and not sup.get("frente_de_rafagas"):
        return None
    grado = grado_viento_kmh(rafaga)
    razones = []
    if rafaga is not None and grado:
        razones.append(f"ráfaga de {rafaga:.0f} km/h medida en Bahía Blanca")
    if medio is not None:
        g_medio = 3 if medio >= VIENTO_MEDIO_FUERTE_KMH else grado_viento_kmh(medio)
        if g_medio > grado:
            grado = g_medio
            razones.append(f"viento de {medio:.0f} km/h medido en Bahía Blanca")
    confirmado = True
    if sup.get("frente_de_rafagas") and grado < 3:
        grado = 3
        confirmado = False
        razones.append("frente de ráfagas medido: salto de presión y caída de temperatura en "
                       + ", ".join(sup.get("con_frente_de_rafagas") or ["Bahía Blanca"]))
    extra = {}
    if rafaga is not None:
        extra["rafaga_kmh"] = round(rafaga)
    if medio is not None:
        extra["viento_kmh"] = round(medio)
    return _evaluacion("viento", grado, "estaciones", razones, confirmado, extra)


def viento_por_modelo(amb: dict[str, Any]) -> dict[str, Any] | None:
    """Pronóstico ECMWF para la hora: anticipa el viento aunque no llueva."""
    rafaga = numero(amb.get("wind_gusts_10m"))
    medio = numero(amb.get("wind_speed_10m"))
    if rafaga is None and medio is None:
        return None
    grado = grado_viento_kmh(rafaga)
    if medio is not None and medio >= VIENTO_MEDIO_FUERTE_KMH:
        grado = max(grado, 3)
    # El modelo solo no llega a "severo": anticipa, no mide.
    grado = min(grado, 3)
    razones = []
    if grado:
        razones.append(f"pronóstico del modelo europeo: ráfagas de {rafaga or 0:.0f} km/h")
    return _evaluacion("viento", grado, "modelo", razones, False,
                       {"rafaga_kmh": round(rafaga) if rafaga is not None else None})


def viento_por_radar(*, velocidad: float | None, coherencia: float | None,
                     dbz_cercano: float | None) -> dict[str, Any] | None:
    """Línea rápida y ordenada con núcleos fuertes cerca: firma de viento
    destructivo en la tormenta (16/12/2023: 75-84 km/h, coherencia 0,94-0,98)."""
    vel = numero(velocidad)
    coh = numero(coherencia)
    dbz = numero(dbz_cercano) or 0.0
    if vel is None or coh is None or dbz < LINEA_DBZ:
        return None
    if vel >= LINEA_RAPIDA_KMH and coh >= LINEA_COHERENCIA:
        return _evaluacion("viento", 4, "radar", [
            f"línea de tormentas avanzando a {vel:.0f} km/h en bloque (coherencia {coh:.2f}) "
            f"con núcleos de {dbz:.0f} dBZ"], False, {"velocidad_sistema_kmh": round(vel)})
    if vel >= 45 and coh >= 0.7:
        return _evaluacion("viento", 3, "radar", [
            f"tormenta rápida ({vel:.0f} km/h) con núcleos de {dbz:.0f} dBZ: ráfagas fuertes"],
            False, {"velocidad_sistema_kmh": round(vel)})
    return None


# --------------------------------------------------------------------------
# Granizo
# --------------------------------------------------------------------------
def ajustar_por_satelite(ev: dict[str, Any], sat: dict[str, Any]) -> dict[str, Any]:
    """Topes muy fríos o enfriándose rápido sobre una tormenta con núcleo
    >= 50 dBZ suben un grado el granizo (nunca lo crean solos)."""
    if not sat or ev["grado"] < 2:
        return ev
    tope = numero(sat.get("min_c_15km"))
    enfr = numero(sat.get("enfriamiento_c_10min"))
    razones = []
    if tope is not None and tope <= TOPE_MUY_FRIO_C:
        razones.append(f"topes de nube muy fríos ({tope:.0f} °C, GOES-19)")
    if enfr is not None and enfr >= ENFRIAMIENTO_RAPIDO_C:
        razones.append(f"topes enfriándose {enfr:.0f} °C en 10 min: la tormenta crece rápido")
    if not razones or ev["grado"] >= 4:
        return dict(ev, razones=ev["razones"] + razones) if razones else ev
    nuevo = _evaluacion("granizo", min(4, ev["grado"] + 1), ev["fuente"] + "+goes19",
                        ev["razones"] + razones, False)
    return nuevo


def lluvia_por_satelite(sat: dict[str, Any]) -> dict[str, Any] | None:
    """Sin radar: nube de tormenta sobre la zona = lluvia probable (no se mide)."""
    tope = numero((sat or {}).get("min_c_15km"))
    if tope is None or tope > TOPE_TORMENTA_C:
        return None
    return _evaluacion("lluvia", 2, "goes19", [
        f"nube de tormenta sobre la zona (topes {tope:.0f} °C): lluvia probable, sin radar para medirla"],
        False, {"estimada": True})


def granizo_por_radar(dbz: float | None, amb: dict[str, Any]) -> dict[str, Any]:
    dbz = numero(dbz) or 0.0
    if dbz >= GRANIZO_SEVERO_DBZ:
        grado = 4
    elif dbz >= GRANIZO_PROBABLE_DBZ:
        grado = 3
    elif dbz >= GRANIZO_POSIBLE_DBZ:
        grado = 2
    else:
        grado = 0
    razones = [f"núcleo de {dbz:.0f} dBZ"] if grado else []
    iso0 = numero(amb.get("freezing_level_height"))
    if grado and iso0 is not None and iso0 >= ISOTERMA_CERO_ALTA_M and grado < 4:
        grado -= 1
        razones.append(f"la isoterma de 0 °C está alta ({iso0:.0f} m): el granizo chico se derrite")
    return _evaluacion("granizo", grado, "radar", razones, False)


# --------------------------------------------------------------------------
# Tormenta eléctrica
# --------------------------------------------------------------------------
def electrica(rayos: dict[str, Any]) -> dict[str, Any]:
    """Rayos de GOES-19 (GLM). Con ventana de varios minutos usa la tasa a
    <= 30 km y el salto brusco; si solo hay el último archivo, lo que haya."""
    tasa = numero(rayos.get("tasa_30km_por_min"))
    tasa_previa = numero(rayos.get("tasa_30km_previa_por_min"))
    flashes = numero(rayos.get("flashes")) or 0.0
    cerca = numero(rayos.get("distancia_km"))
    razones = []
    if tasa is None:
        # Solo el último archivo de GLM: rayo cerca = aislada; muchos = frecuente.
        if flashes <= 0 or cerca is None or cerca > RAYOS_CERCA_KM:
            grado = 0
        elif flashes >= 10:
            grado = 3
        else:
            grado = 1
        if grado:
            razones.append(f"{flashes:.0f} rayos, el más cercano a {cerca:.0f} km")
        return _evaluacion("electrica", grado, "goes19", razones, True)

    if tasa <= 0:
        grado = 0
    elif tasa < RAYOS_FRECUENTES_MIN:
        grado = 1
    else:
        grado = 3
    if grado:
        razones.append(f"{tasa:.1f} rayos por minuto a menos de {RAYOS_CERCA_KM:.0f} km")
    salto = (
        tasa >= RAYOS_SALTO_MIN
        and tasa_previa is not None
        and tasa >= RAYOS_SALTO_FACTOR * max(tasa_previa, 0.2)
    )
    if salto:
        grado = 4
        razones.append(f"salto brusco de rayos ({tasa_previa:.1f} → {tasa:.1f} por minuto): "
                       "la tormenta se intensifica, puede traer granizo o ráfagas")
    return _evaluacion("electrica", grado, "goes19", razones, True,
                       {"tasa_por_min": round(tasa, 2), "salto": bool(salto)})


# --------------------------------------------------------------------------
# Evaluación completa
# --------------------------------------------------------------------------
def frase_fenomeno(clave: str, ev: dict[str, Any]) -> str:
    if clave == "lluvia":
        if ev.get("detalle") == "llovizna":
            return "llovizna"
        if ev.get("detalle") == "torrencial" and ev["grado"] == 3:
            return "lluvia fuerte (torrencial)"
        return f"lluvia {ev['categoria']}"
    if clave == "viento":
        return f"viento {ev['categoria']}"
    if clave == "granizo":
        cat = ev["categoria"]
        return "granizo severo posible" if cat == "posible severo" else (
            "granizo severo" if cat == "severo" else f"granizo {cat}")
    if clave == "electrica":
        cat = ev["categoria"]
        return {"aislada": "rayos aislados", "frecuente": "rayos frecuentes",
                "severa": "actividad eléctrica severa"}.get(cat, f"tormenta eléctrica {cat}")
    return clave


def resumen(fenomenos: dict[str, dict[str, Any]]) -> str:
    presentes = sorted(
        ((k, v) for k, v in fenomenos.items() if v["grado"] > 0),
        key=lambda kv: (-kv[1]["grado"], list(NOMBRES).index(kv[0])),
    )
    if not presentes:
        return "Sin fenómenos"
    partes = [frase_fenomeno(k, v) for k, v in presentes]
    texto = partes[0][0].upper() + partes[0][1:]
    if len(partes) > 1:
        texto += " con " + (", ".join(partes[1:-1]) + " y " if len(partes) > 2 else "") + partes[-1]
    # Lo que NO hay, cuando importa para entender la situación.
    if fenomenos["lluvia"]["grado"] == 0:
        texto += ", sin lluvia"
    return texto


def calificacion(fenomenos: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grado = max(v["grado"] for v in fenomenos.values())
    if grado == 0:
        return {"grado": 0, "etiqueta": GENERAL[0], "por": [], "confirmada": True}
    por = [k for k in NOMBRES if fenomenos[k]["grado"] == grado]
    confirmada = any(fenomenos[k]["confirmado"] for k in por)
    base = GENERAL[grado]
    if grado == 4 and not confirmada:
        base = "Posible severa"
    return {
        "grado": grado,
        "etiqueta": f"{base} ({', '.join(NOMBRES[k] for k in por)})",
        "por": por,
        "confirmada": confirmada,
    }


def evaluar(
    *,
    radar: dict[str, Any] | None = None,
    superficie: dict[str, Any] | None = None,
    rayos: dict[str, Any] | None = None,
    ambiente: dict[str, Any] | None = None,
    satelite: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Devuelve {ahora, al_llegar, fenomenos, resumen, calificacion}.

    radar: dbz_ciudad, dbz_30km, dbz_llegada, eta_llegada_min, velocidad_kmh,
           coherencia, minutos_lluvia_intensa, lluvia_3h_mm, lluvia_6h_mm,
           disponible (bool). Por localidad además: mm_h_area (lluvia media
           sobre la zona), dbz_pico (núcleo, para granizo), lluvia_1h_mm y
           velocidad_eco_kmh (del eco que llega).
    superficie: resumen de estaciones de Bahía Blanca (senales_superficie).
    rayos: flashes, distancia_km y, si hay ventana, tasa_30km_por_min y
           tasa_30km_previa_por_min.
    ambiente: ECMWF de la hora (wind_gusts_10m, wind_speed_10m,
              freezing_level_height, ...).
    satelite: topes GOES-19 de la zona (min_c_15km, p10_c_30km,
              enfriamiento_c_10min).
    """
    sat = satelite or {}
    radar = radar or {}
    sup = superficie or {}
    rayos = rayos or {}
    amb = ambiente or {}
    hay_radar = radar.get("disponible", True) is not False

    est_lluvia = lluvia_por_estaciones(sup)
    est_viento = viento_por_estaciones(sup)
    mod_viento = viento_por_modelo(amb)
    ev_rayos = electrica(rayos)
    nada = {k: _evaluacion(k, 0, "sin_datos", [], False) for k in NOMBRES}

    # ---------------- Ahora (sobre la ciudad) ----------------
    ahora: dict[str, dict[str, Any]] = {}
    if hay_radar:
        dbz_c = numero(radar.get("dbz_ciudad"))
        ahora["lluvia"] = lluvia_por_radar(
            dbz_c,
            minutos_intensa=radar.get("minutos_lluvia_intensa"),
            l3=radar.get("lluvia_3h_mm"),
            l6=radar.get("lluvia_6h_mm"),
            mm_h_area=radar.get("mm_h_area"),
            l1=radar.get("lluvia_1h_mm"),
        )
        pico = numero(radar.get("dbz_pico"))
        ahora["granizo"] = ajustar_por_satelite(
            granizo_por_radar(pico if pico is not None else dbz_c, amb), sat)
        cercano = max(numero(radar.get("dbz_ciudad")) or 0, numero(radar.get("dbz_30km")) or 0)
        ev_vr = viento_por_radar(velocidad=radar.get("velocidad_kmh"),
                                 coherencia=radar.get("coherencia"), dbz_cercano=cercano)
    else:
        ahora["lluvia"] = lluvia_por_satelite(sat) or dict(nada["lluvia"])
        ahora["granizo"] = dict(nada["granizo"])
        ev_vr = None
    if est_lluvia is not None:
        ahora["lluvia"] = _mejor(ahora["lluvia"], est_lluvia)
    viento_opciones = [o for o in (est_viento, mod_viento, ev_vr) if o is not None]
    ahora["viento"] = _mejor(*viento_opciones) if viento_opciones else dict(nada["viento"])
    # Una ráfaga medida baja manda sobre el pronóstico: si la estación mide
    # poco viento y no hay firma de radar, no se informa el del modelo.
    if (est_viento is not None and est_viento["grado"] < ahora["viento"]["grado"]
            and ahora["viento"]["fuente"] == "modelo"):
        ahora["viento"] = dict(ahora["viento"], fuente="modelo",
                               razones=ahora["viento"]["razones"]
                               + ["las estaciones todavía no lo miden"])
    ahora["electrica"] = ev_rayos

    # ---------------- Al llegar (<= 60 min) ----------------
    al_llegar: dict[str, dict[str, Any]] | None = None
    eta = numero(radar.get("eta_llegada_min"))
    dbz_l = numero(radar.get("dbz_llegada"))
    if hay_radar and eta is not None and eta <= LLEGADA_MAX_MIN and dbz_l is not None and dbz_l >= LLUVIA_ECO_DBZ:
        al_llegar = {
            "lluvia": lluvia_por_radar(dbz_l),
            "granizo": granizo_por_radar(dbz_l, amb),
        }
        ev_vl = viento_por_radar(velocidad=radar.get("velocidad_eco_kmh") or radar.get("velocidad_kmh"),
                                 coherencia=radar.get("coherencia"), dbz_cercano=dbz_l)
        opciones = [o for o in (ev_vl, mod_viento) if o is not None]
        al_llegar["viento"] = _mejor(*opciones) if opciones else dict(nada["viento"])
        al_llegar["electrica"] = ev_rayos
        for ev in al_llegar.values():
            ev["minutos"] = round(eta, 1)

    # ---------------- Lo más alto de ahora y lo que llega ----------------
    fenomenos = {}
    for k in NOMBRES:
        candidatos = [ahora[k]] + ([al_llegar[k]] if al_llegar else [])
        mejor = _mejor(*candidatos)
        fenomenos[k] = dict(mejor, cuando="al_llegar" if (al_llegar and mejor is al_llegar[k]
                                                           and mejor["grado"] > ahora[k]["grado"])
                            else "ahora")

    return {
        "version": "1.0.0",
        "ahora": ahora,
        "resumen_ahora": resumen(ahora),
        "al_llegar": al_llegar,
        "resumen_al_llegar": resumen(al_llegar) if al_llegar else None,
        "minutos_hasta_llegada": round(eta, 1) if al_llegar else None,
        "fenomenos": fenomenos,
        "resumen": resumen(fenomenos),
        "calificacion": calificacion(fenomenos),
        "radar_disponible": hay_radar,
    }
