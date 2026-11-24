#!/usr/bin/env python3
"""Señales de superficie de las estaciones (física, sin entrenamiento).

Lee las observaciones frescas de BCP y SMN (las que ya publican los
colectores), guarda un historial de 6 h por estación y calcula:

- lluvia medida en 1 h y 3 h (diferencias del acumulado diario de BCP),
- viento y ráfaga máximos,
- salto de presión y caída de temperatura en 30 min: la firma del frente de
  ráfagas de una tormenta (aire frío que baja y se expande),

para la zona de Bahía Blanca (estaciones a <=20 km) y para la región.

Salida: data/ia/estaciones/senales_superficie.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from . import frescura
except ImportError:  # ejecutado como script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import frescura

BASE = Path(__file__).resolve().parents[2]
EST = BASE / "data/ia/estaciones"
BCP = EST / "observaciones/bcp_actual.json"
SMN = EST / "observaciones/smn_synop_actual.json"
CWOP = EST / "observaciones/cwop_actual.json"
CWOP_CONFIG = BASE / "config/estaciones_cwop.json"
CATALOGO = EST / "catalogo_estaciones.json"
HISTORIAL = EST / "historial_superficie.json"
SALIDA = EST / "senales_superficie.json"

BAHIA_LAT, BAHIA_LON = -38.7196, -62.2724
# 20 km (antes 25): la estación nueva de Punta Alta (21/11/2026) está a 24,8 km y
# su lluvia no tiene que mezclarse con la de Bahía (13/11: 70 mm en Punta Alta, poco
# en Bahía). Las estaciones de Bahía de siempre están a menos de 15 km.
RADIO_BAHIA_KM = 20.0
HISTORIAL_HORAS = 6
VENTANA_SALTO_MIN = 30.0
SALTO_PRESION_HPA = 2.0
CAIDA_TEMPERATURA_C = 4.0


def parse_utc(texto: Any) -> datetime | None:
    if not texto:
        return None
    try:
        dt = datetime.fromisoformat(str(texto).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def numero(valor: Any) -> float | None:
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def distancia_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def leer(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def escribir(path: Path, datos) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(datos, handle, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def coordenadas() -> dict[str, tuple[float, float]]:
    cat = leer(CATALOGO)
    salida = {}
    for s in cat.get("stations", []):
        lat = numero(s.get("lat", s.get("latitude")))
        lon = numero(s.get("lon", s.get("longitude")))
        if s.get("name") and lat is not None and lon is not None:
            salida[s["name"]] = (lat, lon)
    for e in leer(CWOP_CONFIG).get("estaciones", []):
        if e.get("nombre") and e.get("lat") is not None:
            salida[e["nombre"]] = (float(e["lat"]), float(e["lon"]))
    return salida


def _smn_valor(variables: dict, nombre: str) -> float | None:
    lista = variables.get(nombre) or []
    return numero(lista[0].get("value")) if lista and isinstance(lista[0], dict) else None


def observaciones_frescas(ahora: datetime | None = None) -> list[dict[str, Any]]:
    """Normaliza BCP, SMN y CWOP a {nombre, red, utc, temp, lluvia_dia, viento, rafaga, presion}.

    La edad se calcula AHORA con la hora de la medición y el límite de su red
    (frescura.py): la marca "valida" del colector puede ser de horas atrás
    (la escucha CWOP no corre en cada ciclo)."""
    ahora = ahora or datetime.now(timezone.utc)
    obs = []
    for s in leer(BCP).get("stations", []):
        if s.get("estado") != "valida":
            continue
        v = s.get("variables") or {}
        obs.append({
            "nombre": s.get("nombre"), "red": "bcp", "utc": s.get("observation_time"),
            "temp": numero(v.get("temperatura_c")),
            "lluvia_dia": numero(v.get("precipitacion_dia_mm")),
            "viento": numero(v.get("viento_kmh")),
            "rafaga": numero(v.get("rafaga_15min_kmh")),
            "presion": numero(v.get("presion_hpa")),
        })
    for s in leer(SMN).get("stations", []):
        if s.get("estado") != "valida":
            continue
        v = s.get("variables") or {}
        viento = _smn_valor(v, "wind_speed")
        rafaga = _smn_valor(v, "maximum_wind_gust_speed") or _smn_valor(v, "wind_gust")
        obs.append({
            "nombre": s.get("nombre"), "red": "smn", "utc": s.get("observation_time"),
            "temp": _smn_valor(v, "air_temperature"),
            "lluvia_dia": None,
            "viento": viento * 3.6 if viento is not None else None,
            "rafaga": rafaga * 3.6 if rafaga is not None else None,
            "presion": _smn_valor(v, "non_coordinate_pressure"),
        })
    # Red CWOP (estaciones ciudadanas validadas, config/estaciones_cwop.json).
    for s in leer(CWOP).get("stations", []):
        if s.get("estado") != "valida":
            continue
        v = s.get("variables") or {}
        obs.append({
            "nombre": s.get("nombre"), "red": "cwop", "utc": s.get("observation_time"),
            "temp": numero(v.get("temperatura_c")),
            "lluvia_dia": numero(v.get("precipitacion_dia_mm")),
            "viento": numero(v.get("viento_kmh")),
            "rafaga": numero(v.get("rafaga_15min_kmh")),
            "presion": numero(v.get("presion_hpa")),
        })
    return [o for o in obs if o["nombre"] and parse_utc(o["utc"])
            and frescura.es_fresca(o["utc"], o["red"], ahora)]


def actualizar_historial(obs: list[dict], ahora: datetime, path: Path = HISTORIAL) -> dict[str, list[dict]]:
    historial = leer(path) if path.exists() else {}
    if not isinstance(historial, dict):
        historial = {}
    for o in obs:
        serie = historial.setdefault(o["nombre"], [])
        if not any(r.get("utc") == o["utc"] for r in serie):
            serie.append({k: o[k] for k in ("utc", "temp", "lluvia_dia", "viento", "rafaga", "presion")})
    limite = ahora - timedelta(hours=HISTORIAL_HORAS)
    for nombre in list(historial):
        serie = [r for r in historial[nombre] if (parse_utc(r.get("utc")) or limite) > limite]
        serie.sort(key=lambda r: r["utc"])
        if serie:
            historial[nombre] = serie
        else:
            del historial[nombre]
    escribir(path, historial)
    return historial


def lluvia_en(serie: list[dict], ahora: datetime, horas: float) -> float | None:
    """Lluvia medida en la ventana a partir del acumulado diario (se reinicia a medianoche local)."""
    puntos = [(parse_utc(r["utc"]), r.get("lluvia_dia")) for r in serie if r.get("lluvia_dia") is not None]
    puntos = [(t, v) for t, v in puntos if t is not None and t >= ahora - timedelta(hours=horas)]
    if len(puntos) < 2:
        return None
    total = 0.0
    ref_t, ref_v = puntos[0]
    local = timezone(timedelta(hours=-3))
    for t, v in puntos[1:]:
        if v >= ref_v:
            total += v - ref_v
        elif t.astimezone(local).date() != ref_t.astimezone(local).date():
            total += v  # reinicio diario (medianoche local): cuenta lo nuevo
        else:
            continue    # bajada sin cruzar la medianoche = lectura errónea (La Vitícola 21/11 02:47)
        ref_t, ref_v = t, v
    return round(max(total, 0.0), 1)


def cambio_en(serie: list[dict], campo: str, ahora: datetime, minutos: float) -> float | None:
    """Cambio del campo entre la observación más vieja dentro de la ventana y la última."""
    puntos = [(parse_utc(r["utc"]), r.get(campo)) for r in serie if r.get(campo) is not None]
    puntos = [(t, v) for t, v in puntos if t is not None and t >= ahora - timedelta(minutes=minutos)]
    if len(puntos) < 2:
        return None
    return round(puntos[-1][1] - puntos[0][1], 1)


def resumen_estacion(nombre: str, serie: list[dict], ahora: datetime, coords) -> dict[str, Any]:
    ultima = serie[-1]
    lat, lon = coords.get(nombre, (None, None))
    salto_p = cambio_en(serie, "presion", ahora, VENTANA_SALTO_MIN)
    cambio_t = cambio_en(serie, "temp", ahora, VENTANA_SALTO_MIN)
    return {
        "nombre": nombre,
        "utc": ultima["utc"],
        "lat": lat,
        "lon": lon,
        "distancia_bahia_km": round(distancia_km(lat, lon, BAHIA_LAT, BAHIA_LON), 1) if lat is not None else None,
        "temperatura_c": ultima.get("temp"),
        "viento_kmh": ultima.get("viento"),
        "rafaga_kmh": ultima.get("rafaga"),
        "presion_hpa": ultima.get("presion"),
        "lluvia_1h_mm": lluvia_en(serie, ahora, 1),
        "lluvia_3h_mm": lluvia_en(serie, ahora, 3),
        "salto_presion_30min_hpa": salto_p,
        "cambio_temperatura_30min_c": cambio_t,
        "frente_de_rafagas": bool(
            salto_p is not None and cambio_t is not None
            and salto_p >= SALTO_PRESION_HPA and cambio_t <= -CAIDA_TEMPERATURA_C
        ),
    }


def _es_cwop(e: dict) -> bool:
    return "(CWOP)" in str(e.get("nombre") or "").upper()


def resumir(estaciones: list[dict]) -> dict[str, Any]:
    def maximo(campo, sin_cwop=False):
        vals = [e[campo] for e in estaciones
                if e.get(campo) is not None and not (sin_cwop and _es_cwop(e))]
        return max(vals) if vals else None

    return {
        "estaciones": len(estaciones),
        "lluvia_1h_mm": maximo("lluvia_1h_mm"),
        "lluvia_3h_mm": maximo("lluvia_3h_mm"),
        # 24/11/2026: el viento de las estaciones de aficionados (CWOP) no
        # entra al índice: Bahía Noroeste midió 3 km/h con 33 en el Centro
        # (reparada). Con lo medido mandando sobre el modelo, una estación
        # reparada bajaría el nivel de viento por error.
        "viento_max_kmh": maximo("viento_kmh", sin_cwop=True),
        "rafaga_max_kmh": maximo("rafaga_kmh", sin_cwop=True),
        "frente_de_rafagas": any(e["frente_de_rafagas"] for e in estaciones),
        "con_frente_de_rafagas": [e["nombre"] for e in estaciones if e["frente_de_rafagas"]],
    }


CALIDAD = Path(__file__).resolve().parents[2] / "data/ia/calibracion/calidad_estaciones.json"


def calidad_estaciones() -> dict:
    try:
        return (json.loads(CALIDAD.read_text(encoding="utf-8")).get("estaciones")) or {}
    except Exception:
        return {}


RADIO_LOCALIDAD_KM = 15.0
LOCALIDADES = Path(__file__).resolve().parents[2] / "config" / "localidades.json"


def por_localidad(estaciones: list[dict], path: Path = LOCALIDADES) -> dict[str, dict]:
    """Estaciones a <= 15 km de cada localidad (las demás no la representan)."""
    try:
        locs = json.loads(path.read_text(encoding="utf-8")).get("localidades") or []
    except Exception:
        return {}
    salida = {}
    for loc in locs:
        cerca = [
            e for e in estaciones
            if e.get("lat") is not None
            and distancia_km(e["lat"], e["lon"], loc["lat"], loc["lon"]) <= RADIO_LOCALIDAD_KM
        ]
        if cerca:
            salida[loc["id"]] = dict(resumir(cerca), nombres=[e["nombre"] for e in cerca])
    return salida


# Anemómetro trabado (24/11/2026): Punta Alta marcó 0,0 km/h en 51 mediciones
# seguidas con 5-45 km/h en Bahía Centro. Si una estación marca 0 en todas
# sus mediciones de las últimas 4 h (al menos 12) mientras otra estación de la
# región midió 15 km/h o más en ese tiempo, su viento no se usa.
TRABADO_HORAS = 4.0
TRABADO_MIN_MEDICIONES = 12
TRABADO_OTRAS_KMH = 15.0


def vientos_trabados(historial: dict[str, list[dict]], ahora: datetime) -> list[str]:
    desde = ahora - timedelta(hours=TRABADO_HORAS)

    def ventana(serie):
        return [r.get("viento") for r in serie
                if r.get("viento") is not None and (parse_utc(r.get("utc")) or desde) >= desde]

    vientos = {n: ventana(s) for n, s in historial.items()}
    trabados = []
    for n, vals in vientos.items():
        if len(vals) < TRABADO_MIN_MEDICIONES or any(v > 0 for v in vals):
            continue
        if any(max(o) >= TRABADO_OTRAS_KMH for m, o in vientos.items() if m != n and o):
            trabados.append(n)
    return sorted(trabados)


def main(ahora: datetime | None = None) -> int:
    ahora = ahora or datetime.now(timezone.utc)
    coords = coordenadas()
    obs = observaciones_frescas(ahora)
    historial = actualizar_historial(obs, ahora)
    frescas = {o["nombre"] for o in obs}
    estaciones = [
        resumen_estacion(n, s, ahora, coords) for n, s in historial.items() if n in frescas and s
    ]
    trabados = vientos_trabados(historial, ahora)
    for e in estaciones:
        if e["nombre"] in trabados:
            e.update(viento_kmh=None, rafaga_kmh=None, viento_trabado=True)
    # Control de calidad (calidad_y_calibracion.py, ciclo anterior): una
    # estación dudosa no entra en los resúmenes; con la lluvia dudosa se
    # usa todo menos su lluvia. El detalle queda en "estaciones".
    calidad = calidad_estaciones()
    aptas = []
    for e in estaciones:
        c = calidad.get(e["nombre"]) or {}
        e["calidad"] = c.get("estado", "sin_control")
        e["calidad_lluvia"] = c.get("estado_lluvia", "sin_control")
        if e["calidad"] == "dudosa":
            continue
        if e["calidad_lluvia"] == "dudosa":
            e = dict(e, lluvia_1h_mm=None, lluvia_3h_mm=None)
        aptas.append(e)
    bahia = [e for e in aptas if (e["distancia_bahia_km"] or 999) <= RADIO_BAHIA_KM]
    salida = {
        "version": "1.0.0",
        "generado_utc": ahora.isoformat(),
        "estaciones_frescas": len(estaciones),
        "bahia_blanca": resumir(bahia),
        "region": resumir(aptas),
        "localidades": por_localidad(aptas),
        "descartadas_por_calidad": sorted(e["nombre"] for e in estaciones if e["calidad"] == "dudosa"),
        # Lo lee también la página (app.py) para no mostrar ese viento.
        "viento_trabado": trabados,
        "estaciones": estaciones,
    }
    escribir(SALIDA, salida)
    print(json.dumps({"frescas": len(estaciones), "bahia": salida["bahia_blanca"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
