"""Datos de GOES-19 por cada tormenta que sigue el radar RMA10.

Para cada núcleo georreferenciado del radar (data/radar/radar_nowcast.json)
se juntan los píxeles del satélite que le corresponden, con dos
correcciones para que el dato sea de esa tormenta y de ese momento:

1. Hora: el satélite pasa por Bahía Blanca unos minutos antes que la imagen
   de radar. Se mueve la tormenta con su propio desplazamiento hasta la hora
   en que el satélite la vio.
2. Perspectiva (paralaje): GOES-19 mira la zona de costado; el tope de una
   tormenta de 10 km de alto aparece ~11 km corrido al sureste. Se busca la
   tormenta donde el satélite la ve.

Los rayos (GLM) ya vienen ubicados por NOAA a la altura típica de los rayos:
solo se corrige la hora.

Todo es estadística pura (sin red): las funciones reciben los datos ya
recortados a la zona de 240 km.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .geospatial import haversine_km, posicion_aparente

K = 273.15
RADIO_MIN_KM = 8.0
RADIO_MAX_KM = 40.0
MARGEN_RADIO_KM = 6.0          # píxel de 2 km + error de ubicación + yunque
ALTURA_TOPE_DEFECTO_KM = 10.0  # para ubicar la tormenta si no hay altura medida
ALTURA_TOPE_MIN_KM = 3.0
ALTURA_TOPE_MAX_KM = 16.0
DESFASE_MAX_MIN = 20.0         # satélite hasta 20 min antes que el radar
RAYOS_VENTANA_MIN = 10.0
KM_POR_GRADO_LAT = 111.195


def parse_utc(texto: Any) -> datetime | None:
    if not texto:
        return None
    try:
        valor = datetime.fromisoformat(str(texto).replace("Z", "+00:00"))
    except ValueError:
        return None
    if valor.tzinfo is None:
        valor = valor.replace(tzinfo=timezone.utc)
    return valor.astimezone(timezone.utc)


def iso(valor: datetime | None) -> str | None:
    return valor.isoformat().replace("+00:00", "Z") if valor else None


def _num(valor: Any) -> float | None:
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return None
    return numero if math.isfinite(numero) else None


# ------------------------------------------------------------------
# Tormentas del radar
# ------------------------------------------------------------------

def tormentas_desde_nowcast(radar_nowcast: dict) -> list[dict]:
    """Núcleos georreferenciados del radar con posición, tamaño y movimiento."""
    nowcast = (radar_nowcast or {}).get("nowcast") or {}
    km_px = _num(((radar_nowcast or {}).get("escala") or {}).get("km_por_pixel_estimado")) or 0.93933
    salida = []
    for nucleo in nowcast.get("nucleos_georreferenciados") or []:
        geo = nucleo.get("georreferencia") or {}
        lat, lon = _num(geo.get("latitud")), _num(geo.get("longitud"))
        if lat is None or lon is None:
            continue
        pixeles = _num(nucleo.get("pixeles")) or 0.0
        radio = math.sqrt(max(pixeles, 0.0) * km_px * km_px / math.pi) + MARGEN_RADIO_KM
        mov = nucleo.get("movimiento") or {}
        velocidad = (_num(mov.get("velocidad_px_min")) or 0.0) * km_px  # km/min
        rumbo = _num(mov.get("direccion_grados"))                       # hacia dónde va
        if rumbo is None:
            velocidad = 0.0
            rumbo = 0.0
        salida.append({
            "track_id": nucleo.get("track_id"),
            "nucleo_id": nucleo.get("nucleo_id"),
            "latitud": lat,
            "longitud": lon,
            "radio_km": round(float(np.clip(radio, RADIO_MIN_KM, RADIO_MAX_KM)), 1),
            "este_km_min": velocidad * math.sin(math.radians(rumbo)),
            "norte_km_min": velocidad * math.cos(math.radians(rumbo)),
            "dbz_max": _num((nucleo.get("dbz") or {}).get("max")),
        })
    return salida


def posicion_en(tormenta: dict, minutos: float) -> tuple[float, float]:
    """Posición de la tormenta `minutos` después de la imagen de radar
    (negativo = antes), según su propio movimiento."""
    lat = tormenta["latitud"] + tormenta["norte_km_min"] * minutos / KM_POR_GRADO_LAT
    lon = tormenta["longitud"] + tormenta["este_km_min"] * minutos / (
        KM_POR_GRADO_LAT * math.cos(math.radians(tormenta["latitud"]))
    )
    return lat, lon


# ------------------------------------------------------------------
# Hora del satélite sobre la zona
# ------------------------------------------------------------------

def hora_local_escaneo(inicio: datetime | None, fin: datetime | None, fraccion: float | None) -> datetime | None:
    """Hora en que el barrido del disco completo pasó por Bahía Blanca."""
    if inicio is None or fin is None:
        return fin
    if fraccion is None:
        return fin
    return inicio + (fin - inicio) * float(fraccion)


def sincronizacion(hora_satelite: datetime | None, frame_radar: datetime) -> dict:
    """Desfase entre el satélite y la imagen de radar y si sirve para esta imagen."""
    if hora_satelite is None:
        return {"estado": "sin_hora", "desfase_min": None, "hora_satelite_utc": None}
    desfase = (frame_radar - hora_satelite).total_seconds() / 60.0
    if desfase < -1.0:
        estado = "posterior_al_radar"  # nunca se usa: sería un dato del futuro
    elif desfase > DESFASE_MAX_MIN:
        estado = "desfase_excesivo"
    else:
        estado = "sincronizado"
    return {
        "estado": estado,
        "desfase_min": round(desfase, 1),
        "hora_satelite_utc": iso(hora_satelite),
        "desfase_max_min": DESFASE_MAX_MIN,
    }


# ------------------------------------------------------------------
# Estadísticas por tormenta
# ------------------------------------------------------------------

def _dentro(campo: dict, lat: float, lon: float, radio_km: float) -> np.ndarray:
    d = haversine_km(campo["latitude"], campo["longitude"], lat, lon)
    return np.isfinite(d) & (d <= radio_km)


def ubicar_en_satelite(tormenta: dict, minutos: float, altura_km: float,
                       proyeccion) -> tuple[float, float, float]:
    """Centro donde el satélite ve la tormenta y corrimiento total (km)."""
    lat_t, lon_t = posicion_en(tormenta, minutos)
    if proyeccion is not None:
        lat_s, lon_s = posicion_aparente(lat_t, lon_t, altura_km, proyeccion)
    else:
        lat_s, lon_s = lat_t, lon_t
    corrimiento = float(haversine_km(np.array([lat_s]), np.array([lon_s]),
                                     tormenta["latitud"], tormenta["longitud"])[0])
    return lat_s, lon_s, corrimiento


def altura_para_paralaje(campo_altura: dict | None, tormenta: dict, minutos: float) -> tuple[float, str]:
    """Altura del tope (km) para corregir la perspectiva: la medida por el
    satélite si hay, si no 10 km."""
    if campo_altura is None:
        return ALTURA_TOPE_DEFECTO_KM, "defecto"
    proy = campo_altura.get("proyeccion")
    lat, lon, _ = ubicar_en_satelite(tormenta, minutos, ALTURA_TOPE_DEFECTO_KM, proy)
    m = _dentro(campo_altura, lat, lon, tormenta["radio_km"])
    valores = np.asarray(campo_altura["valores"], dtype=float)[m]
    valores = valores[np.isfinite(valores)]
    if valores.size == 0:
        return ALTURA_TOPE_DEFECTO_KM, "defecto"
    km = float(np.percentile(valores, 90)) / 1000.0
    return float(np.clip(km, ALTURA_TOPE_MIN_KM, ALTURA_TOPE_MAX_KM)), "medida"


def estadistica_campo(campo: dict, tormenta: dict, frame: datetime, altura_km: float,
                      tipo: str) -> dict:
    """Estadística de un campo del satélite sobre una tormenta.

    tipo: "temperatura" (K -> °C), "altura" (m -> km), "presion" (hPa).
    """
    sinc = sincronizacion(campo.get("hora_satelite"), frame)
    salida: dict[str, Any] = {"sincronizacion": sinc}
    if sinc["estado"] != "sincronizado":
        salida["estado"] = "no_usado"
        return salida
    minutos = -float(sinc["desfase_min"])
    lat, lon, corrimiento = ubicar_en_satelite(tormenta, minutos, altura_km, campo.get("proyeccion"))
    m = _dentro(campo, lat, lon, tormenta["radio_km"])
    v = np.asarray(campo["valores"], dtype=float)[m]
    v = v[np.isfinite(v)]
    salida.update({
        "centro_satelite": {"latitud": round(lat, 4), "longitud": round(lon, 4)},
        "corrimiento_km": round(corrimiento, 1),
        "pixeles": int(v.size),
    })
    if v.size == 0:
        salida["estado"] = "sin_pixeles"
        return salida
    salida["estado"] = "ok"
    if tipo == "temperatura":
        salida.update({
            "min_c": round(float(v.min()) - K, 1),
            "p10_c": round(float(np.percentile(v, 10)) - K, 1),
            "media_c": round(float(v.mean()) - K, 1),
        })
    elif tipo == "altura":
        salida.update({
            "max_km": round(float(v.max()) / 1000.0, 2),
            "p90_km": round(float(np.percentile(v, 90)) / 1000.0, 2),
            "mediana_km": round(float(np.median(v)) / 1000.0, 2),
        })
    elif tipo == "presion":
        salida.update({
            "min_hpa": round(float(v.min()), 1),
            "p10_hpa": round(float(np.percentile(v, 10)), 1),
            "mediana_hpa": round(float(np.median(v)), 1),
        })
    return salida


def rayos_por_tormenta(rayos: dict | None, tormenta: dict, frame: datetime) -> dict:
    """Rayos GLM de la tormenta en los 10 min que terminan en la imagen de radar.

    rayos: {"minutos": edad respecto del radar (positivo = antes),
            "latitude", "longitude", "cobertura_min"}.
    Cada rayo se compara con la posición que tenía la tormenta en ese minuto.
    """
    if not rayos or rayos.get("cobertura_min", 0) < 5:
        return {"estado": "no_disponible", "ventana_min": RAYOS_VENTANA_MIN,
                "fin_utc": iso(frame)}
    edad = np.asarray(rayos["minutos"], dtype=float)
    lat = np.asarray(rayos["latitude"], dtype=float)
    lon = np.asarray(rayos["longitude"], dtype=float)
    usar = (edad >= 0) & (edad < RAYOS_VENTANA_MIN)
    edad, lat, lon = edad[usar], lat[usar], lon[usar]
    if edad.size:
        lat_t = tormenta["latitud"] - tormenta["norte_km_min"] * edad / KM_POR_GRADO_LAT
        lon_t = tormenta["longitud"] - tormenta["este_km_min"] * edad / (
            KM_POR_GRADO_LAT * math.cos(math.radians(tormenta["latitud"])))
        r = 6371.0088
        la1, lo1, la2, lo2 = map(np.radians, (lat, lon, lat_t, lon_t))
        a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
        d = 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        cantidad = int(np.sum(d <= tormenta["radio_km"]))
    else:
        cantidad = 0
    cobertura = min(RAYOS_VENTANA_MIN, float(rayos["cobertura_min"]))
    return {
        "estado": "ok",
        "ventana_min": RAYOS_VENTANA_MIN,
        "fin_utc": iso(frame),
        "cobertura_min": round(cobertura, 1),
        "cantidad": cantidad,
        "tasa_por_min": round(cantidad / cobertura, 2) if cobertura > 0 else None,
    }


def _anteriores(estado_previo: dict | None) -> dict[str, dict]:
    previo = ((estado_previo or {}).get("por_tormenta") or {}).get("tormentas") or []
    return {str(t.get("track_id")): t for t in previo if t.get("track_id") is not None}


def enfriamiento(actual: dict, previo: dict | None) -> float | None:
    """Enfriamiento del 10 % más frío del tope (°C cada 10 min) entre dos
    pasadas del satélite sobre la MISMA tormenta (mismo track)."""
    if not previo:
        return None
    a = actual.get("c13") or {}
    p = previo.get("c13") or {}
    if a.get("estado") != "ok" or p.get("estado") != "ok":
        return None
    ta = parse_utc((a.get("sincronizacion") or {}).get("hora_satelite_utc"))
    tp = parse_utc((p.get("sincronizacion") or {}).get("hora_satelite_utc"))
    if ta is None or tp is None:
        return None
    minutos = (ta - tp).total_seconds() / 60.0
    if not 5.0 <= minutos <= 30.0:
        return None
    return round((p["p10_c"] - a["p10_c"]) * 10.0 / minutos, 1)


def calcular(radar_nowcast: dict | None, frame: datetime, campos: dict[str, dict | None],
             rayos: dict | None, estado_previo: dict | None = None) -> dict:
    """Datos de satélite por tormenta.

    campos: {"c13", "c08", "altura_topes", "presion_topes"} -> cada uno
        {"valores", "latitude", "longitude", "proyeccion", "hora_satelite"}
        o None si no hay.
    """
    if not isinstance(radar_nowcast, dict):
        return {"estado": "sin_datos_radar", "frame_radar_utc": iso(frame), "tormentas": []}
    frame_nowcast = parse_utc(radar_nowcast.get("actualizado"))
    if frame_nowcast is None or abs((frame_nowcast - frame).total_seconds()) > 1:
        return {
            "estado": "radar_de_otro_frame",
            "frame_radar_utc": iso(frame),
            "frame_nowcast_utc": iso(frame_nowcast),
            "tormentas": [],
        }
    tormentas = tormentas_desde_nowcast(radar_nowcast)
    if not tormentas:
        return {"estado": "sin_tormentas", "frame_radar_utc": iso(frame), "tormentas": []}

    previos = _anteriores(estado_previo)
    salida = []
    for t in tormentas:
        campo_altura = campos.get("altura_topes")
        if campo_altura is not None:
            sinc_alt = sincronizacion(campo_altura.get("hora_satelite"), frame)
            if sinc_alt["estado"] != "sincronizado":
                campo_altura = None
        minutos_ref = 0.0
        c13 = campos.get("c13")
        if c13 is not None and c13.get("hora_satelite") is not None:
            minutos_ref = -(frame - c13["hora_satelite"]).total_seconds() / 60.0
        altura_km, origen_altura = altura_para_paralaje(campo_altura, t, minutos_ref)

        item: dict[str, Any] = {
            "track_id": t["track_id"],
            "nucleo_id": t["nucleo_id"],
            "latitud": round(t["latitud"], 5),
            "longitud": round(t["longitud"], 5),
            "radio_km": t["radio_km"],
            "dbz_max": t["dbz_max"],
            "movimiento_km_min": {
                "este": round(t["este_km_min"], 3),
                "norte": round(t["norte_km_min"], 3),
            },
            "altura_paralaje_km": round(altura_km, 1),
            "altura_paralaje_origen": origen_altura,
        }
        for nombre, tipo in (("c13", "temperatura"), ("c08", "temperatura"),
                             ("altura_topes", "altura"), ("presion_topes", "presion")):
            campo = campos.get(nombre)
            if campo is None:
                item[nombre] = {"estado": "no_disponible"}
            else:
                item[nombre] = estadistica_campo(campo, t, frame, altura_km, tipo)
        item["c13"]["enfriamiento_c_10min"] = enfriamiento(item, previos.get(str(t["track_id"])))
        item["rayos"] = rayos_por_tormenta(rayos, t, frame)
        salida.append(item)

    return {
        "estado": "ok",
        "frame_radar_utc": iso(frame),
        "metodo": "posición corregida por hora del satélite y por perspectiva (paralaje)",
        "tormentas": salida,
    }
