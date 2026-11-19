#!/usr/bin/env python3
"""Clasificación del tiempo para Bahía Blanca (reglas físicas, sin colores).

Escala descriptiva, de menor a mayor:

  0 sin_precipitacion
  1 llovizna                      eco < 20 dBZ
  2 lluvia_debil                  20-30 dBZ
  3 lluvia_moderada               30-40 dBZ
  4 lluvia_fuerte                 40-45 dBZ sin señales convectivas
  5 tormenta_debil                >=40 dBZ con rayos o inestabilidad
  6 tormenta_moderada             >=45 dBZ
  7 tormenta_fuerte               >=50 dBZ, o >=45 con mucha actividad eléctrica
  8 tormenta_potencialmente_severa >=55 dBZ (granizo), línea rápida y ordenada
                                  (viento) o lluvia intensa persistente (inundación)

Las amenazas (granizo, viento, inundación) se informan aparte con sus
razones. No es un sistema de alertas oficiales: describe lo que miden el
radar, el satélite y el ambiente.
"""
from __future__ import annotations

import math
from typing import Any

TIPOS = (
    ("sin_precipitacion", "Sin precipitación"),
    ("llovizna", "Llovizna o lluvia muy débil"),
    ("lluvia_debil", "Lluvia débil"),
    ("lluvia_moderada", "Lluvia moderada"),
    ("lluvia_fuerte", "Lluvia fuerte"),
    ("tormenta_debil", "Tormenta débil"),
    ("tormenta_moderada", "Tormenta moderada"),
    ("tormenta_fuerte", "Tormenta fuerte"),
    ("tormenta_potencialmente_severa", "Tormenta potencialmente severa"),
)

ECO_MIN_DBZ = 5.0

# Amenazas que ya pueden subir el nivel a "potencialmente severa". Viento e
# inundación se validaron con imágenes del SMN (3 temporales), pero con los
# PNG crudos de SINARAME dispararon en 5 de 14 días de noviembre de 2026:
# hasta calibrarlos con datos de SINARAME se publican como indicios en
# evaluación, sin subir el nivel.
AMENAZAS_CALIBRADAS = {"granizo": True, "viento": False, "inundacion": False}
GRANIZO_DBZ = 55.0
LINEA_RAPIDA_KMH = 60.0
LINEA_COHERENCIA = 0.85
LINEA_DBZ = 45.0
INUNDACION_DBZ = 40.0
INUNDACION_MINUTOS = 60.0
# Validado con el 7/3/2025: el radar subestima mucho la lluvia eficiente y
# persistente (estimó 13 mm donde cayeron 290 mm); la duración la separa
# mejor (10,5 h seguidas de lluvia contra 5 h el 16/12/2023).
# Lluvia estimada por radar (Marshall-Palmer) sobre la ciudad. Separó el
# 7/3/2025 (33 mm en 3 h, 57 mm en 6 h) del 16/12/2023 (24 / 39) y del
# 2/2/2025 (20 / 20). Calibrar con pluviómetros cuando estén conectados.
LLUVIA_3H_MM = 30.0
LLUVIA_6H_MM = 50.0
CAPE_CONVECTIVO = 300.0
CAPE_ALTO = 1500.0
RAYOS_MUCHOS = 10
RAYO_CERCA_KM = 30.0
LLEGADA_MAX_MIN = 60.0

# Mediciones de estaciones en Bahía Blanca (no son estimaciones: suben el nivel).
RAFAGA_FUERTE_KMH = 70.0
VIENTO_FUERTE_KMH = 50.0
LLUVIA_MEDIDA_1H_MM = 30.0
LLUVIA_MEDIDA_3H_MM = 50.0


def nivel_por_lluvia_medida(mm_1h: float | None) -> int:
    """Intensidad de lluvia medida en 1 h (mm/h) -> nivel de la escala."""
    if mm_1h is None or mm_1h <= 0:
        return 0
    if mm_1h < 0.5:
        return 1
    if mm_1h < 2.5:
        return 2
    if mm_1h < 10:
        return 3
    return 4


def amenazas_medidas(superficie: dict[str, Any]) -> tuple[dict[str, dict], list[str]]:
    """Amenazas confirmadas por estaciones de Bahía Blanca."""
    amenazas, razones = {}, []
    rafaga = numero(superficie.get("rafaga_max_kmh"))
    viento = numero(superficie.get("viento_max_kmh"))
    l1 = numero(superficie.get("lluvia_1h_mm"))
    l3 = numero(superficie.get("lluvia_3h_mm"))
    if (rafaga is not None and rafaga >= RAFAGA_FUERTE_KMH) or (viento is not None and viento >= VIENTO_FUERTE_KMH):
        medido = f"ráfaga {rafaga:.0f} km/h" if rafaga is not None and rafaga >= RAFAGA_FUERTE_KMH else f"viento {viento:.0f} km/h"
        amenazas["viento"] = {"indicio": True, "razon": f"{medido} medido en Bahía Blanca", "fuente": "estaciones"}
    if (l1 is not None and l1 >= LLUVIA_MEDIDA_1H_MM) or (l3 is not None and l3 >= LLUVIA_MEDIDA_3H_MM):
        amenazas["inundacion"] = {
            "indicio": True,
            "razon": f"lluvia medida en Bahía Blanca: {l1 or 0:.0f} mm en 1 h, {l3 or 0:.0f} mm en 3 h",
            "fuente": "estaciones",
        }
    if superficie.get("frente_de_rafagas"):
        razones.append(
            "frente de ráfagas en " + ", ".join(superficie.get("con_frente_de_rafagas") or [])
            + " (salto de presión y caída de temperatura)"
        )
    if l1:
        razones.append(f"lluvia medida: {l1:.1f} mm en la última hora")
    return amenazas, razones


def numero(valor: Any, defecto: float | None = None) -> float | None:
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return defecto
    return v if math.isfinite(v) else defecto


def nivel_por_reflectividad(dbz: float | None, convectivo: bool) -> int:
    if dbz is None or dbz < ECO_MIN_DBZ:
        return 0
    if dbz < 20:
        return 1
    if dbz < 30:
        return 2
    if dbz < 40:
        return 3
    if dbz < 45:
        return 5 if convectivo else 4
    if dbz < 50:
        return 6
    if dbz < GRANIZO_DBZ:
        return 7
    return 8


def tipo(nivel: int) -> dict[str, Any]:
    nivel = max(0, min(nivel, len(TIPOS) - 1))
    clave, etiqueta = TIPOS[nivel]
    return {"nivel": nivel, "tipo": clave, "etiqueta": etiqueta}


def señales_convectivas(rayos: dict[str, Any], ambiente: dict[str, Any]) -> tuple[bool, bool, list[str]]:
    """(hay_conveccion, actividad_electrica_alta, razones)."""
    razones = []
    flashes = numero(rayos.get("flashes"), 0) or 0
    cerca = numero(rayos.get("distancia_km"))
    cape = numero(ambiente.get("cape"), 0) or 0
    li = numero(ambiente.get("lifted_index"))
    hay = False
    if flashes > 0 and (cerca is None or cerca <= 240):
        hay = True
        razones.append(f"{int(flashes)} rayos detectados por GOES-19")
    if cape >= CAPE_CONVECTIVO:
        hay = True
        razones.append(f"inestabilidad (CAPE {int(cape)} J/kg)")
    if li is not None and li <= -2:
        hay = True
        razones.append(f"índice de levantamiento {li:.1f}")
    alta = flashes >= RAYOS_MUCHOS and (cerca is not None and cerca <= RAYO_CERCA_KM)
    if alta:
        razones.append(f"actividad eléctrica alta a {cerca:.0f} km")
    return hay, alta, razones


def clasificar(
    *,
    dbz_ciudad: float | None,
    dbz_llegada: float | None,
    eta_llegada_min: float | None,
    dbz_region: float | None,
    dbz_30km: float | None = None,
    lluvia_3h_mm: float | None = None,
    lluvia_6h_mm: float | None = None,
    distancia_region_km: float | None,
    velocidad_kmh: float | None,
    coherencia: float | None,
    minutos_lluvia_intensa: float | None,
    rayos: dict[str, Any] | None = None,
    ambiente: dict[str, Any] | None = None,
    radar_disponible: bool = True,
    superficie: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rayos = rayos or {}
    ambiente = ambiente or {}
    convectivo, electrica_alta, razones_conv = señales_convectivas(rayos, ambiente)

    superficie = superficie or {}
    medidas, razones_sup = amenazas_medidas(superficie)

    if not radar_disponible:
        nivel = max(5 if convectivo else 0, nivel_por_lluvia_medida(numero(superficie.get("lluvia_1h_mm"))))
        if medidas:
            nivel = 8
        base = tipo(nivel)
        for a in medidas.values():
            a["en_calibracion"] = False
        return {
            **base,
            "alcance": "sin_radar",
            "radar_disponible": False,
            "amenazas": medidas,
            "razones": ["Sin radar reciente: satélite, estaciones y ambiente."] + razones_sup + razones_conv,
        }

    ahora = nivel_por_reflectividad(dbz_ciudad, convectivo)
    llega = 0
    if eta_llegada_min is not None and eta_llegada_min <= LLEGADA_MAX_MIN:
        llega = nivel_por_reflectividad(dbz_llegada, convectivo)

    amenazas: dict[str, dict[str, Any]] = {}
    razones: list[str] = []

    # Granizo: núcleo muy intenso sobre o hacia la ciudad.
    pico = max(dbz_ciudad or 0, dbz_llegada or 0 if llega else 0)
    if pico >= GRANIZO_DBZ:
        amenazas["granizo"] = {"indicio": True, "razon": f"núcleo de {pico:.0f} dBZ"}

    # Viento: línea rápida y ordenada con núcleos fuertes (firma de bow echo).
    vel = numero(velocidad_kmh)
    coh = numero(coherencia)
    # Solo cuenta si los núcleos fuertes están sobre la ciudad, a <=30 km o
    # llegando en <=60 min (una línea rápida lejana no afecta a Bahía Blanca).
    cercano = max(dbz_ciudad or 0, dbz_30km or 0, (dbz_llegada or 0) if llega else 0)
    if (
        vel is not None and coh is not None
        and vel >= LINEA_RAPIDA_KMH and coh >= LINEA_COHERENCIA
        and cercano >= LINEA_DBZ
    ):
        amenazas["viento"] = {
            "indicio": True,
            "razon": f"sistema avanzando a {vel:.0f} km/h en bloque (coherencia {coh:.2f})",
        }

    # Inundación: lluvia intensa persistente sobre la ciudad.
    intensa = numero(minutos_lluvia_intensa, 0) or 0
    l3 = numero(lluvia_3h_mm, 0) or 0
    l6 = numero(lluvia_6h_mm, 0) or 0
    if intensa >= INUNDACION_MINUTOS:
        amenazas["inundacion"] = {
            "indicio": True,
            "razon": f"{intensa:.0f} min de lluvia intensa (≥{INUNDACION_DBZ:.0f} dBZ) sobre la ciudad",
        }
    elif l3 >= LLUVIA_3H_MM or l6 >= LLUVIA_6H_MM:
        amenazas["inundacion"] = {
            "indicio": True,
            "razon": (
                f"lluvia acumulada estimada por radar: {l3:.0f} mm en 3 h, {l6:.0f} mm en 6 h "
                "(el radar suele subestimarla)"
            ),
        }

    for clave, amenaza in amenazas.items():
        amenaza["en_calibracion"] = not AMENAZAS_CALIBRADAS.get(clave, False)
    # Lo medido por estaciones reemplaza al indicio del radar: está confirmado.
    for clave, amenaza in medidas.items():
        amenazas[clave] = dict(amenaza, en_calibracion=False)
    if superficie.get("frente_de_rafagas") and "viento" not in amenazas:
        amenazas["viento"] = {
            "indicio": True,
            "razon": "frente de ráfagas medido (salto de presión y caída de temperatura)",
            "fuente": "estaciones",
            "en_calibracion": True,
        }
    calibradas = {k for k, v in amenazas.items() if not v["en_calibracion"]}

    nivel = max(ahora, llega)
    if nivel >= 6 and electrica_alta:
        nivel = max(nivel, 7)
    if calibradas and max(ahora, llega) >= 5:
        nivel = 8
    # Viento (línea rápida con núcleos fuertes cerca) e inundación son firmas
    # de temporal destructivo aunque sobre la ciudad todavía llueva poco.
    if calibradas & {"viento", "inundacion"}:
        nivel = max(nivel, 8)

    if ahora >= llega and ahora > 0:
        alcance = "sobre_bahia"
        razones.append(f"{dbz_ciudad:.0f} dBZ sobre la ciudad")
    elif llega > 0:
        alcance = "acercandose"
        if eta_llegada_min <= 2:
            razones.append(f"llegando ahora con {dbz_llegada:.0f} dBZ")
        else:
            razones.append(f"llega en ~{eta_llegada_min:.0f} min con {dbz_llegada:.0f} dBZ")
    else:
        alcance = "ninguno"

    region = tipo(nivel_por_reflectividad(dbz_region, convectivo))
    if distancia_region_km is not None:
        region["distancia_km"] = round(distancia_region_km, 1)

    if CAPE_ALTO <= (numero(ambiente.get("cape"), 0) or 0) and nivel >= 5:
        razones.append("ambiente muy inestable: puede intensificarse")

    salida = tipo(nivel)
    if nivel == 8 and calibradas:
        nombres = {"granizo": "granizo", "viento": "viento", "inundacion": "inundación"}
        detalle = ", ".join(nombres[a] for a in ("granizo", "viento", "inundacion") if a in calibradas)
        if max(ahora, llega) < 5 and calibradas == {"inundacion"}:
            salida["etiqueta"] = "Lluvia persistente potencialmente severa (inundación)"
        else:
            salida["etiqueta"] = f"{salida['etiqueta']} ({detalle})"
    salida.update({
        "alcance": alcance,
        "radar_disponible": True,
        "minutos_hasta_llegada": (
            round(eta_llegada_min, 1) if alcance == "acercandose" else None
        ),
        "region": region,
        "amenazas": amenazas,
        "razones": razones + [a["razon"] for a in amenazas.values()] + razones_sup + razones_conv,
    })
    return salida


MANTENER_AMENAZA_MIN = 40.0


def mantener_amenazas(actual: dict[str, Any], anterior: dict[str, Any] | None,
                      ahora_utc: str, anterior_utc: str | None) -> dict[str, Any]:
    """Una amenaza detectada se mantiene 30 min después de su última detección,
    para que el aviso no se prenda y apague entre frames."""
    from datetime import datetime

    def parse(texto):
        try:
            return datetime.fromisoformat(str(texto).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None

    ahora = parse(ahora_utc)
    resultado = dict(actual)
    amenazas = {k: dict(v, ultima_deteccion_utc=ahora_utc) for k, v in (actual.get("amenazas") or {}).items()}
    if ahora is not None and isinstance(anterior, dict) and anterior.get("radar_disponible", True):
        for clave, previa in (anterior.get("amenazas") or {}).items():
            if clave in amenazas or not isinstance(previa, dict):
                continue
            ultima = parse(previa.get("ultima_deteccion_utc") or anterior_utc)
            if ultima is not None and 0 <= (ahora - ultima).total_seconds() / 60.0 <= MANTENER_AMENAZA_MIN:
                amenazas[clave] = dict(previa, mantenida=True)
    resultado["amenazas"] = amenazas
    calibradas = {k for k, v in amenazas.items() if not v.get("en_calibracion")}
    if calibradas & {"viento", "inundacion"} or (
        calibradas and resultado.get("nivel", 0) >= 5
    ):
        if resultado.get("nivel", 0) < 8:
            base = tipo(8)
            nombres = {"granizo": "granizo", "viento": "viento", "inundacion": "inundación"}
            detalle = ", ".join(nombres[a] for a in ("granizo", "viento", "inundacion") if a in calibradas)
            resultado.update(base)
            resultado["etiqueta"] = f"{base['etiqueta']} ({detalle})"
    resultado["indicios_en_evaluacion"] = sorted(
        k for k, v in amenazas.items() if v.get("en_calibracion")
    )
    return resultado
