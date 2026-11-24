#!/usr/bin/env python3
"""Índices por fenómeno para Bahía Blanca (reglas físicas, sin entrenamiento).

Cada fenómeno se evalúa por separado y puede estar o no estar. Escala de 1 a 4,
igual para todos (decisión de Lucas, 23/11/2026): 1 débil (se nota, sin impacto),
2 moderado (molestias), 3 fuerte (precaución, daños menores), 4 severo (peligroso).
El 0 no es un nivel: quiere decir que el fenómeno no está.

  lluvia     1 débil (< 3 mm/h) · 2 moderada (3-12) · 3 fuerte (12-50) · 4 severa
             (>= 50 mm/h, o 30 mm en 1 h, o 50 mm en 3 h: riesgo de anegamientos)
  viento     ráfaga 1 débil 30-49 · 2 moderado 50-69 · 3 fuerte 70-89 · 4 severo >= 90 km/h
             (sostenido >= 45 km/h: fuerte). 3 = alerta amarilla del SMN; 4 = estándar
             internacional de tormenta severa.
  granizo    1 posible (>= 50 dBZ) · 2 probable (>= 55, el blanco) · 3 fuerte (>= 60 con
             tope frío o rayos) · 4 severo (>= 60, tope muy frío Y muchos rayos: >= 2 cm)
  electrica  1 aislada · 2 moderada (0,5-2 rayos/min) · 3 frecuente (>= 2/min) · 4 severa
             (aumento brusco de rayos)

La calificación general es el nivel más alto y dice de qué fenómeno se trata. El resumen
informa la combinación real ("Viento fuerte, sin lluvia").

Cada fenómeno se informa "ahora" (sobre la ciudad) y "al llegar" (lo que
trae el sistema que se acerca en <= 60 min).

Grado 4 sin confirmación de una estación se informa como "posible severo"
hasta calibrar los cortes con datos de SINARAME (decisión del 19/11/2026).
No es un sistema de avisos: los avisos oficiales son del SMN.
"""
from __future__ import annotations

import math
import re
from typing import Any

import ambiente_severo

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
LLUVIA_SIGUE_1H_MM = 5.0         # sigue lloviendo en serio

# Etapa F (23/11/2026): cortes del viento llevados a los del SMN. Su aviso amarillo por
# viento es de ráfagas de ~70 km/h o más, o viento medio de 40-60 km/h; con los cortes
# anteriores (ráfaga 50 = "fuerte") ClimaAR mostraba viento fuerte en toda la región casi
# todas las tardes ventosas de Bahía (20-22/11: ~1 de cada 3 imágenes).
VIENTO_DEBIL_KMH = 30.0          # escala de Lucas (23/11/2026): 30 / 50 / 70 / 90
VIENTO_MODERADO_KMH = 50.0
VIENTO_FUERTE_KMH = 70.0
VIENTO_SEVERO_KMH = 90.0
VIENTO_MEDIO_FUERTE_KMH = 45.0   # viento medio (no ráfaga) que ya es fuerte
LINEA_DBZ = 45.0                 # núcleo fuerte (ambiente favorable a ráfagas)

GRANIZO_POSIBLE_DBZ = 50.0
GRANIZO_PROBABLE_DBZ = 55.0
GRANIZO_SEVERO_DBZ = 60.0
ISOTERMA_CERO_ALTA_M = 4500.0    # granizo chico se derrite antes de llegar

RAYOS_CERCA_KM = 30.0
RAYOS_MODERADOS_MIN = 0.5        # rayos por minuto a <= 30 km
RAYOS_FRECUENTES_MIN = 2.0
RAYOS_SALTO_FACTOR = 3.0         # aumento brusco frente a los 10 min previos
RAYOS_SALTO_MIN = 1.0            # ...con al menos 1 rayo por minuto

LLEGADA_MAX_MIN = 60.0

# GOES-19 canal 13 (topes de nubes).
TOPE_MUY_FRIO_C = -60.0          # tormenta muy profunda: granizo más probable
TOPE_TORMENTA_C = -50.0          # nube de tormenta
# Sin radar (punto 10, 21/11/2026, herramientas/simular_sin_radar.py, 26 días): "tope <= -50 °C"
# solo acertaba la lluvia moderada 9 de cada 100 veces (88 % falsas). Tope <= -60 °C y al menos
# un rayo a <= 15 km acierta 33 de cada 100 (54 % falsas).
TOPE_LLUVIA_SIN_RADAR_C = -60.0
RAYOS_LLUVIA_SIN_RADAR = 1
# Radar viejo movido (40-90 min): sigue siendo lo mejor para "¿llueve?" (acierto 0,41 a 40 min
# contra 0,08 de las reglas de satélite de antes). Más viejo no está probado.
RADAR_MOVIDO_MAX_MIN = 90.0
ENFRIAMIENTO_RAPIDO_C = 8.0      # °C cada 10 min: la tormenta crece rápido

NOMBRES = {
    "lluvia": "lluvia",
    "viento": "viento",
    "granizo": "granizo",
    "electrica": "tormenta eléctrica",
}

CATEGORIAS = {
    "lluvia": {0: "sin lluvia", 1: "débil", 2: "moderada", 3: "fuerte", 4: "severa"},
    "viento": {0: "sin viento destacable", 1: "débil", 2: "moderado", 3: "fuerte", 4: "severo"},
    "granizo": {0: "sin señal", 1: "posible", 2: "probable", 3: "fuerte", 4: "severo"},
    "electrica": {0: "sin rayos", 1: "aislada", 2: "moderada", 3: "frecuente", 4: "severa"},
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
    extra = {"mm_h": round(mm_h, 1)}
    if l3 >= LLUVIA_RADAR_3H_MM or l6 >= LLUVIA_RADAR_6H_MM:
        texto = (f"acumulado estimado por radar {l3:.0f} mm en 3 h, {l6:.0f} mm en 6 h "
                 "(riesgo de anegamientos)")
        # Solo es "lluvia severa" mientras sigue lloviendo: si ya paró, se
        # informa el acumulado aparte (13/11/2026: Punta Alta seguía como
        # "severa" 3 h después de que dejó de llover).
        if grado >= 2 or l1 >= LLUVIA_SIGUE_1H_MM:
            grado = 4
            razones.append(texto)
        else:
            extra["acumulado_alto"] = True
            extra["acumulado_razon"] = texto
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
    if kmh is None or kmh < VIENTO_DEBIL_KMH:
        return 0
    if kmh < VIENTO_MODERADO_KMH:
        return 1
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
    razones = []
    # 24/11/2026: una ráfaga nunca es menor que el viento medio. Si lo es, la
    # midió otra estación (en Bahía solo la Noroeste mide ráfagas y estaba
    # reparada: 3 km/h con 33 km/h en el Centro). No se usa.
    if rafaga is not None and medio is not None and rafaga < medio:
        rafaga = None
    grado = grado_viento_kmh(rafaga)
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


def viento_por_arco(arco: dict[str, Any] | None) -> dict[str, Any] | None:
    """Eco en arco (etapa G, docs/ETAPA_G.md): línea de tormentas curvada hacia adelante,
    la firma de viento destructivo que se ve sin Doppler (16/12/2023 en Bahía). Medido con
    46 jornadas severas verificadas de EE. UU.: 6 de cada 10 dieron ráfagas severas
    (>= 93 km/h) en la hora siguiente, contra 1 de cada 10 de una tormenta cualquiera.
    Reemplaza a la "línea rápida" por velocidad, que acertaba 15 % (casi como cualquier
    tormenta). Llega como mucho a "fuerte": "severo" necesita una ráfaga medida."""
    if not arco:
        return None
    d, v = numero(arco.get("distancia_km")), numero(arco.get("velocidad_kmh"))
    donde = "encima" if (d is None or d <= 0.5) else f"a {d:.0f} km"
    razon = (f"eco en arco (línea de tormentas curvada hacia adelante) {donde}"
             + (f", avanzando a {v:.0f} km/h" if v is not None else "")
             + ": ráfagas fuertes posibles (6 de cada 10 dieron ráfagas severas)")
    return _evaluacion("viento", 3, "radar", [razon], False,
                       {"eco_en_arco": True, "velocidad_sistema_kmh": round(v) if v is not None else None})


# --------------------------------------------------------------------------
# Granizo
# --------------------------------------------------------------------------
def ajustar_por_satelite(ev: dict[str, Any], sat: dict[str, Any]) -> dict[str, Any]:
    """Topes muy fríos o enfriándose rápido sobre una tormenta con núcleo
    >= 50 dBZ suben un grado el granizo (nunca lo crean solos)."""
    if not sat or ev["grado"] < 1:
        return ev
    tope = numero(sat.get("min_c_15km"))
    enfr = numero(sat.get("enfriamiento_c_10min"))
    razones = []
    if tope is not None and tope <= TOPE_MUY_FRIO_C:
        razones.append(f"topes de nube muy fríos ({tope:.0f} °C, GOES-19)")
    if enfr is not None and enfr >= ENFRIAMIENTO_RAPIDO_C:
        razones.append(f"topes enfriándose {enfr:.0f} °C en 10 min: la tormenta crece rápido")
    # Solo sube hasta "fuerte": "severo" lo decide confirmar_granizo con todas las señales.
    if not razones or ev["grado"] >= 3:
        return dict(ev, razones=ev["razones"] + razones) if razones else ev
    nuevo = _evaluacion("granizo", min(3, ev["grado"] + 1), ev["fuente"] + "+goes19",
                        ev["razones"] + razones, False)
    return nuevo


def ajustar_por_ambiente(clave: str, ev: dict[str, Any], amb_sev: dict[str, Any],
                         dbz_tormenta: float | None) -> dict[str, Any]:
    """El ambiente no crea fenómenos: dice qué puede dar una tormenta que ya está.
    - viento: tormenta >= 45 dBZ en ambiente favorable -> ráfagas fuertes posibles;
    - granizo: núcleo >= 50 dBZ con CAPE >= 1000 y cizalladura -> un grado más (hasta 3)."""
    if not amb_sev or not (amb_sev.get("favorable_severas") or amb_sev.get("favorable_rafagas_descendentes")):
        return ev
    dbz = numero(dbz_tormenta) or 0.0
    if clave == "viento" and dbz >= LINEA_DBZ and ev["grado"] < 3:
        return _evaluacion("viento", 3, "radar+ambiente", ev["razones"] + [
            f"tormenta de {dbz:.0f} dBZ en un ambiente favorable a ráfagas fuertes"] + amb_sev["razones"][:1],
            False)
    if (clave == "granizo" and 1 <= ev["grado"] < 3 and amb_sev.get("favorable_severas")
            and (numero(amb_sev.get("cape")) or 0) >= ambiente_severo.CAPE_ALTO):
        return _evaluacion("granizo", ev["grado"] + 1, ev["fuente"] + "+ambiente",
                           ev["razones"] + amb_sev["razones"][:1], False)
    return ev


def lluvia_por_satelite(sat: dict[str, Any], rayos: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Sin radar: nube de tormenta muy fría encima Y con rayos cerca = lluvia probable (no se mide)."""
    tope = numero((sat or {}).get("min_c_15km"))
    cerca = numero((rayos or {}).get("rayos_15km_10min")) or 0
    if tope is None or tope > TOPE_LLUVIA_SIN_RADAR_C or cerca < RAYOS_LLUVIA_SIN_RADAR:
        return None
    return _evaluacion("lluvia", 2, "goes19", [
        f"nube de tormenta encima (topes {tope:.0f} °C) con {cerca:.0f} rayos a menos de 15 km: "
        "lluvia probable, sin radar para medirla"], False, {"estimada": True})


def lluvia_por_radar_movido(movido: dict[str, Any] | None) -> dict[str, Any] | None:
    """Radar viejo (40-90 min) movido con el nowcast hasta ahora: lluvia estimada."""
    if not isinstance(movido, dict):
        return None
    edad = numero(movido.get("edad_min"))
    if edad is None or edad > RADAR_MOVIDO_MAX_MIN:
        return None
    ev = lluvia_por_radar(numero(movido.get("dbz_p75")), mm_h_area=numero(movido.get("mm_h_medio")))
    if not ev["grado"]:
        return None
    ev = dict(ev, fuente="radar_movido", estimada=True)
    ev["razones"] = [f"radar de hace {edad:.0f} min movido hasta ahora: " + ", ".join(ev["razones"])]
    return ev


def granizo_por_radar(dbz: float | None, amb: dict[str, Any]) -> dict[str, Any]:
    dbz = numero(dbz) or 0.0
    if dbz >= GRANIZO_SEVERO_DBZ:
        grado = 3   # el 4 (>= 2 cm) necesita tope muy frío y muchos rayos (confirmar_granizo)
    elif dbz >= GRANIZO_PROBABLE_DBZ:
        grado = 2
    elif dbz >= GRANIZO_POSIBLE_DBZ:
        grado = 1
    else:
        grado = 0
    razones = [f"núcleo de {dbz:.0f} dBZ"] if grado else []
    iso0 = numero(amb.get("freezing_level_height"))
    if grado and iso0 is not None and iso0 >= ISOTERMA_CERO_ALTA_M:
        grado -= 1
        razones.append(f"la isoterma de 0 °C está alta ({iso0:.0f} m): el granizo chico se derrite")
    return _evaluacion("granizo", grado, "radar", razones, False)


def confirmar_granizo(ev: dict[str, Any], sat: dict[str, Any], ev_rayos: dict[str, Any],
                      amb_sev: dict[str, Any]) -> dict[str, Any]:
    """Granizo "probable" o más solo con el radar necesita otra señal, y "severo"
    (>= 2 cm) necesita todas.

    - Severo (4): núcleo >= 60 dBZ con tope muy frío (<= -60 °C) Y rayos frecuentes o en
      salto. El 13/11 Punta Alta tuvo 75 dBZ en el radar crudo y cayó granizo chico: el
      radar solo no alcanza para decir granizo grande.
    - Si el satélite y el ambiente están disponibles y ninguna fuente confirma un
      "probable" o "fuerte" (topes fríos o enfriándose rápido, rayos frecuentes o salto,
      ambiente con energía para tormentas), se baja un grado. Nunca se saca el aviso, y si
      falta un dato no se baja (falla segura). Caso 19/11 23:20: 56 dBZ con CAPE 0, tope
      -37 °C y casi sin rayos mostró "granizo probable"."""
    tope = numero((sat or {}).get("min_c_15km"))
    if tope is None:
        tope = numero((sat or {}).get("p10_c_30km"))
    muchos_rayos = (ev_rayos or {}).get("grado", 0) >= 3
    # El dBZ del núcleo es la primera razón que deja granizo_por_radar.
    m = re.match(r"núcleo de (\d+) dBZ", ev["razones"][0]) if ev["razones"] else None
    nucleo_severo = m is not None and float(m.group(1)) >= GRANIZO_SEVERO_DBZ
    if (ev["grado"] >= 3 and nucleo_severo
            and tope is not None and tope <= TOPE_MUY_FRIO_C and muchos_rayos):
        return _evaluacion("granizo", 4, ev["fuente"] + "+goes19", ev["razones"] + [
            f"tope muy frío ({tope:.0f} °C) y rayos frecuentes: granizo grande posible"], False)
    if ev["grado"] < 2 or ev["fuente"] != "radar":
        return ev   # ya lo confirmó el satélite o el ambiente
    enfr = numero((sat or {}).get("enfriamiento_c_10min"))
    nivel = (amb_sev or {}).get("nivel")
    if tope is None or nivel in (None, "sin_datos"):
        return ev
    confirma = (tope <= TOPE_TORMENTA_C or (enfr is not None and enfr >= ENFRIAMIENTO_RAPIDO_C)
                or muchos_rayos
                or nivel in ("favorable_tormentas", "favorable_severas"))
    if confirma:
        return ev
    return _evaluacion("granizo", ev["grado"] - 1, "radar", ev["razones"] + [
        f"solo lo indica el radar: topes de {tope:.0f} °C, pocos rayos y {amb_sev.get('etiqueta', 'ambiente').lower()}"],
        False)


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
        elif flashes >= 3:
            grado = 2
        else:
            grado = 1
        if grado:
            razones.append(f"{flashes:.0f} rayos, el más cercano a {cerca:.0f} km")
        return _evaluacion("electrica", grado, "goes19", razones, True)

    if tasa <= 0:
        grado = 0
    elif tasa < RAYOS_MODERADOS_MIN:
        grado = 1
    elif tasa < RAYOS_FRECUENTES_MIN:
        grado = 2
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
LLEGADA_PROBABLE = 0.5


def descripcion_llegada(conf: dict[str, Any] | None) -> dict[str, Any] | None:
    """Texto honesto de la llegada según lo que pasó en casos parecidos."""
    if not isinstance(conf, dict) or not conf.get("casos"):
        return None
    casos, llego = int(conf["casos"]), int(conf.get("llego", 0))
    fraccion = llego / casos
    de_cada_10 = round(fraccion * 10)
    texto = f"en casos parecidos llegó {de_cada_10} de cada 10"
    no_llego = casos - llego
    if no_llego and conf.get("se_desvio", 0) >= 0.6 * no_llego:
        texto += "; las demás pasaron al costado"
    elif no_llego and conf.get("se_deshizo", 0) >= 0.6 * no_llego:
        texto += "; las demás se deshicieron antes"
    return {"probable": fraccion >= LLEGADA_PROBABLE, "de_cada_10": de_cada_10,
            "casos": casos, "texto": texto,
            "verbo": "llega" if fraccion >= LLEGADA_PROBABLE else "puede llegar"}


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
        return {"aislada": "rayos aislados", "moderada": "rayos seguidos", "frecuente": "rayos frecuentes",
                "severa": "actividad eléctrica severa"}.get(cat, f"tormenta eléctrica {cat}")
    return clave


def resumen(fenomenos: dict[str, dict[str, Any]]) -> str:
    presentes = sorted(
        ((k, v) for k, v in fenomenos.items() if v["grado"] > 0),
        key=lambda kv: (-kv[1]["grado"], list(NOMBRES).index(kv[0])),
    )
    if not presentes:
        if fenomenos["lluvia"].get("acumulado_alto"):
            return "Sin lluvia ahora; lluvia acumulada alta (posibles anegamientos)"
        return "Sin fenómenos"
    partes = [frase_fenomeno(k, v) for k, v in presentes]
    texto = partes[0][0].upper() + partes[0][1:]
    if len(partes) > 1:
        texto += " con " + (", ".join(partes[1:-1]) + " y " if len(partes) > 2 else "") + partes[-1]
    # Lo que NO hay, cuando importa para entender la situación.
    if fenomenos["lluvia"]["grado"] == 0:
        texto += ", sin lluvia"
    if fenomenos["lluvia"].get("acumulado_alto"):
        texto += "; lluvia acumulada alta (posibles anegamientos)"
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
           disponible (bool). Sin radar al día: movido = {dbz_p75, mm_h_medio,
           edad_min} del radar viejo movido hasta ahora (<= 90 min).
           Por localidad además: mm_h_area (lluvia media
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
    amb_sev = ambiente_severo.evaluar(ambiente)
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
        dbz_nucleo = pico if pico is not None else dbz_c
        ahora["granizo"] = confirmar_granizo(ajustar_por_ambiente("granizo", ajustar_por_satelite(
            granizo_por_radar(dbz_nucleo, amb), sat), amb_sev, dbz_nucleo), sat, ev_rayos, amb_sev)
        arco = radar.get("arco") or None
        ev_vr = viento_por_arco(arco if arco and (numero(arco.get("minutos")) or 0) <= 0 else None)
    else:
        opciones = [o for o in (lluvia_por_satelite(sat, rayos), lluvia_por_radar_movido(radar.get("movido")))
                    if o is not None]
        ahora["lluvia"] = _mejor(*opciones) if opciones else dict(nada["lluvia"])
        ahora["granizo"] = dict(nada["granizo"])
        ev_vr = None
    if est_lluvia is not None:
        ahora["lluvia"] = _mejor(ahora["lluvia"], est_lluvia)
    viento_opciones = [o for o in (est_viento, mod_viento, ev_vr) if o is not None]
    ahora["viento"] = _mejor(*viento_opciones) if viento_opciones else dict(nada["viento"])
    if hay_radar:
        ahora["viento"] = ajustar_por_ambiente(
            "viento", ahora["viento"], amb_sev,
            max(numero(radar.get("dbz_pico")) or 0, numero(radar.get("dbz_ciudad")) or 0))
    # Una ráfaga medida baja manda sobre el pronóstico: si la estación mide
    # poco viento y no hay firma de radar, no se informa el del modelo.
    # 24/11/2026: antes solo se agregaba una nota y quedaba el número del
    # modelo (ráfagas de 38 km/h en las 7 localidades con 19 medidos).
    if (est_viento is not None and est_viento["grado"] < ahora["viento"]["grado"]
            and ahora["viento"]["fuente"] == "modelo"):
        pron = numero((mod_viento or {}).get("rafaga_kmh"))
        ahora["viento"] = dict(est_viento, razones=list(est_viento["razones"]) + (
            [f"el modelo europeo pronostica ráfagas de {pron:.0f} km/h, todavía no medidas"] if pron else []))
    ahora["electrica"] = ev_rayos

    # ---------------- Al llegar (<= 60 min) ----------------
    al_llegar: dict[str, dict[str, Any]] | None = None
    eta = numero(radar.get("eta_llegada_min"))
    dbz_l = numero(radar.get("dbz_llegada"))
    if hay_radar and eta is not None and eta <= LLEGADA_MAX_MIN and dbz_l is not None and dbz_l >= LLUVIA_ECO_DBZ:
        al_llegar = {
            "lluvia": lluvia_por_radar(dbz_l),
            "granizo": confirmar_granizo(ajustar_por_ambiente("granizo", granizo_por_radar(dbz_l, amb),
                                                              amb_sev, dbz_l), sat, ev_rayos, amb_sev),
        }
        arco_l = radar.get("arco") or None
        ev_vl = viento_por_arco(arco_l if arco_l and (numero(arco_l.get("minutos")) or 0) > 0 else None)
        opciones = [o for o in (ev_vl, mod_viento) if o is not None]
        al_llegar["viento"] = ajustar_por_ambiente(
            "viento", _mejor(*opciones) if opciones else dict(nada["viento"]), amb_sev, dbz_l)
        al_llegar["electrica"] = ev_rayos
        for ev in al_llegar.values():
            ev["minutos"] = round(eta, 1)
    llegada = descripcion_llegada(radar.get("confiabilidad_llegada")) if al_llegar else None
    # Una llegada poco confiable se muestra, pero no sube la calificación:
    # en 30 días de radar la mayoría pasó al costado (13/11/2026).
    cuenta_llegada = bool(al_llegar) and (llegada is None or llegada["probable"])

    # ---------------- Lo más alto de ahora y lo que llega ----------------
    fenomenos = {}
    for k in NOMBRES:
        candidatos = [ahora[k]] + ([al_llegar[k]] if cuenta_llegada else [])
        mejor = _mejor(*candidatos)
        fenomenos[k] = dict(mejor, cuando="al_llegar" if (cuenta_llegada and mejor is al_llegar[k]
                                                           and mejor["grado"] > ahora[k]["grado"])
                            else "ahora")

    return {
        "version": "1.0.0",
        "ahora": ahora,
        "resumen_ahora": resumen(ahora),
        "al_llegar": al_llegar,
        "resumen_al_llegar": resumen(al_llegar) if al_llegar else None,
        "minutos_hasta_llegada": round(eta, 1) if al_llegar else None,
        "llegada": llegada,
        "fenomenos": fenomenos,
        "resumen": resumen(fenomenos),
        "calificacion": calificacion(fenomenos),
        "ambiente": amb_sev,
        "radar_disponible": hay_radar,
    }
