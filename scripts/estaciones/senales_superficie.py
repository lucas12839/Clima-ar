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


_RUMBOS = {
    "norte": 0, "nornoreste": 22.5, "nornordeste": 22.5, "noreste": 45, "nordeste": 45,
    "estenoreste": 67.5, "estenordeste": 67.5, "este": 90, "estesudeste": 112.5, "sudeste": 135,
    "sudsudeste": 157.5, "sur": 180, "sudsudoeste": 202.5, "sudoeste": 225, "oestesudoeste": 247.5,
    "oeste": 270, "oestenoroeste": 292.5, "noroeste": 315, "nornoroeste": 337.5,
}


def rumbo_grados(valor: Any) -> float | None:
    """Dirección del viento en grados (la BCP la da en palabras)."""
    n = numero(valor)
    if n is not None:
        return n
    return _RUMBOS.get(str(valor or "").strip().lower().replace(" ", "")) if valor else None


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
            "humedad": numero(v.get("humedad_relativa_pct")),
            "dir": rumbo_grados(v.get("direccion_viento")),
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
            "humedad": _smn_valor(v, "relative_humidity"),
            "dir": _smn_valor(v, "wind_from_direction"),
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
            "humedad": numero(v.get("humedad_relativa_pct")),
            "dir": numero(v.get("direccion_viento_grados")),
            "lluvia_1h_directa": numero(v.get("lluvia_1h_mm")),
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


# ----------------------------------------------------------------------------
# Control de calidad de cada dato de cada estación (24/11/2026)
# ----------------------------------------------------------------------------
# Antes solo se controlaba la lluvia. Pasaron sin control: el anemómetro de
# Punta Alta trabado en 0, la estación Noroeste (aficionado) reparada del
# viento y ráfagas imposibles. Un dato que falla no se usa en ningún lado
# (índices, página, Facebook) y queda anotado con el motivo.
RANGOS = {
    "temperatura_c": (-25.0, 50.0),
    "viento_kmh": (0.0, 200.0),
    "rafaga_kmh": (0.0, 250.0),
    "humedad_pct": (1.0, 100.0),
    "presion_hpa": (850.0, 1080.0),
}
VECINAS_KM = 40.0
VECINAS_MIN = 2
TEMP_VECINAS_C = 7.0          # más lejos que esto de la mediana de las vecinas
TEMP_SALTO_C = 6.0            # en 20 min o menos
TEMP_TRABADA_MIN_MEDICIONES = 12


def temperaturas_trabadas(historial: dict[str, list[dict]], ahora: datetime) -> list[str]:
    """Misma temperatura exacta durante 4 h mientras otra estación cambió 1 °C o más."""
    desde = ahora - timedelta(hours=TRABADO_HORAS)

    def ventana(serie):
        return [r.get("temp") for r in serie
                if r.get("temp") is not None and (parse_utc(r.get("utc")) or desde) >= desde]

    temps = {n: ventana(sr) for n, sr in historial.items()}
    otras_cambian = {n for n, v in temps.items() if len(v) >= 2 and max(v) - min(v) >= 1.0}
    return sorted(n for n, v in temps.items()
                  if len(v) >= TEMP_TRABADA_MIN_MEDICIONES and len(set(v)) == 1 and otras_cambian - {n})


def salto_temperatura(serie: list[dict]) -> bool:
    puntos = [(parse_utc(r.get("utc")), r.get("temp")) for r in serie if r.get("temp") is not None]
    puntos = [(t, v) for t, v in puntos if t is not None]
    if len(puntos) < 2:
        return False
    (t0, v0), (t1, v1) = puntos[-2], puntos[-1]
    return (t1 - t0) <= timedelta(minutes=20) and abs(v1 - v0) > TEMP_SALTO_C


def _mediana(vals: list[float]) -> float:
    v = sorted(vals)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


def controlar(estaciones: list[dict], historial: dict[str, list[dict]], ahora: datetime) -> list[dict]:
    """Revisa cada dato; lo que falla se pone en None y se anota en e["descartes"]."""
    trab_viento = set(vientos_trabados(historial, ahora))
    trab_temp = set(temperaturas_trabadas(historial, ahora))
    anotados = []

    def descartar(e, campo, motivo, aviso=True):
        if e.get(campo) is None:
            return
        e.setdefault("descartes", {})[campo] = motivo
        if aviso:
            anotados.append({"estacion": e["nombre"], "dato": campo, "valor": e[campo], "motivo": motivo})
        e[campo] = None

    for e in estaciones:
        e.setdefault("descartes", {})
        for campo, (lo, hi) in RANGOS.items():
            v = e.get(campo)
            if v is not None and not lo <= v <= hi:
                descartar(e, campo, f"fuera de rango ({lo:g} a {hi:g})")
        if e.get("red") == "cwop":
            # Política, no falla: no se sabe cómo están instaladas (Noroeste reparada).
            for campo in ("viento_kmh", "rafaga_kmh", "direccion_grados"):
                descartar(e, campo, "estación de aficionado: el viento no se usa", aviso=False)
        if e["nombre"] in trab_viento:
            for campo in ("viento_kmh", "rafaga_kmh", "direccion_grados"):
                descartar(e, campo, "anemómetro trabado: 0 km/h durante 4 h con viento en la región",
                          aviso=campo == "viento_kmh")
        if e["nombre"] in trab_temp:
            descartar(e, "temperatura_c", "temperatura trabada: el mismo valor durante 4 h")
        if salto_temperatura(historial.get(e["nombre"]) or []):
            descartar(e, "temperatura_c", f"salto de más de {TEMP_SALTO_C:g} °C en 20 min")
        if (e.get("rafaga_kmh") is not None and e.get("viento_kmh") is not None
                and e["rafaga_kmh"] < e["viento_kmh"]):
            descartar(e, "rafaga_kmh", "ráfaga menor que el viento medio")

    # Temperatura contra las vecinas (con los datos ya limpios).
    for e in estaciones:
        if e.get("temperatura_c") is None or e.get("lat") is None:
            continue
        vecinas = [o["temperatura_c"] for o in estaciones
                   if o is not e and o.get("temperatura_c") is not None and o.get("lat") is not None
                   and not o.get("descartes", {}).get("temperatura_c")
                   and distancia_km(e["lat"], e["lon"], o["lat"], o["lon"]) <= VECINAS_KM]
        if len(vecinas) >= VECINAS_MIN and abs(e["temperatura_c"] - _mediana(vecinas)) > TEMP_VECINAS_C:
            descartar(e, "temperatura_c",
                      f"{e['temperatura_c']:.1f} °C, a más de {TEMP_VECINAS_C:g} °C de las vecinas "
                      f"({_mediana(vecinas):.1f} °C)")
    return anotados


# ----------------------------------------------------------------------------
# Lo medido en cada localidad (una sola fuente para la página y Facebook)
# ----------------------------------------------------------------------------
PROPIA_KM = 8.0


def tiempo_ahora(estaciones: list[dict], path: Path = LOCALIDADES) -> dict[str, dict]:
    """Lo medido en cada localidad, ya controlado (24/11/2026).

    - Temperatura y humedad: la estación más cercana a 8 km o menos.
    - Lluvia de la última hora: la más cercana a 15 km o menos.
    - Viento y ráfaga: los mismos que usa la tarjeta de la localidad (el
      máximo de las estaciones a 15 km), de una misma estación, con su
      dirección. Así arriba y en la tarjeta nunca hay dos números distintos.
    Lo que no mide nadie lo completa la página con el modelo europeo del
    punto de la localidad (meteorologia_fusion.py)."""
    try:
        locs = json.loads(path.read_text(encoding="utf-8")).get("localidades") or []
    except Exception:
        return {}
    salida = {}
    for loc in locs:
        cerca = sorted(
            ((distancia_km(e["lat"], e["lon"], loc["lat"], loc["lon"]), e) for e in estaciones
             if e.get("lat") is not None),
            key=lambda x: x[0])
        cerca = [(d, e) for d, e in cerca if d <= RADIO_LOCALIDAD_KM]
        datos: dict[str, Any] = {}

        def dato(campo, clave, max_km):
            for d, e in cerca:
                if d <= max_km and e.get(campo) is not None:
                    datos[clave] = {"valor": e[campo], "estacion": e["nombre"], "km": round(d, 1),
                                    "utc": e["utc"], "red": e.get("red")}
                    return

        # Temperatura y humedad: solo una estación de la localidad (a 8 km).
        # Más lejos puede no representarla (Sierras Grandes está sobre las
        # sierras): ahí completa el modelo del punto de la localidad.
        dato("temperatura_c", "temperatura", PROPIA_KM)
        dato("humedad_pct", "humedad", PROPIA_KM)
        dato("lluvia_1h_mm", "lluvia_1h", RADIO_LOCALIDAD_KM)
        # Viento y ráfaga: los MISMOS máximos que usa la tarjeta (resumir), cada
        # uno con la estación que lo midió. La ráfaga solo si no es menor que
        # el viento que se muestra.
        con_viento = [(d, e) for d, e in cerca if e.get("viento_kmh") is not None]
        if con_viento:
            d, e = max(con_viento, key=lambda x: (x[1]["viento_kmh"], -x[0]))
            datos["viento"] = {"valor": e["viento_kmh"], "direccion": e.get("direccion_grados"),
                               "rafaga": None, "estacion": e["nombre"], "km": round(d, 1),
                               "utc": e["utc"], "red": e.get("red")}
            con_rafaga = [(dd, ee) for dd, ee in cerca if ee.get("rafaga_kmh") is not None]
            if con_rafaga:
                dr, er = max(con_rafaga, key=lambda x: (x[1]["rafaga_kmh"], -x[0]))
                if er["rafaga_kmh"] >= e["viento_kmh"]:
                    datos["viento"].update(rafaga=er["rafaga_kmh"], rafaga_estacion=er["nombre"],
                                           rafaga_km=round(dr, 1), rafaga_utc=er["utc"])
        if datos:
            salida[loc["id"]] = datos
    return salida


def main(ahora: datetime | None = None) -> int:
    ahora = ahora or datetime.now(timezone.utc)
    coords = coordenadas()
    obs = observaciones_frescas(ahora)
    historial = actualizar_historial(obs, ahora)
    frescas = {o["nombre"]: o for o in obs}
    estaciones = []
    for n, serie in historial.items():
        if n not in frescas or not serie:
            continue
        e = resumen_estacion(n, serie, ahora, coords)
        o = frescas[n]
        e.update(red=o["red"], humedad_pct=o.get("humedad"), direccion_grados=o.get("dir"))
        if e.get("lluvia_1h_mm") is None and o.get("lluvia_1h_directa") is not None:
            e["lluvia_1h_mm"] = o["lluvia_1h_directa"]
        estaciones.append(e)
    anotados = controlar(estaciones, historial, ahora)
    # Control de calidad de la lluvia (calidad_y_calibracion.py, ciclo
    # anterior): una estación dudosa no entra en los resúmenes; con la lluvia
    # dudosa se usa todo menos su lluvia. El detalle queda en "estaciones".
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
        "version": "1.1.0",
        "generado_utc": ahora.isoformat(),
        "estaciones_frescas": len(estaciones),
        "bahia_blanca": resumir(bahia),
        "region": resumir(aptas),
        "localidades": por_localidad(aptas),
        # Lo medido en cada localidad: lo muestran la página y Facebook.
        "tiempo_ahora": tiempo_ahora(aptas),
        "descartadas_por_calidad": sorted(e["nombre"] for e in estaciones if e["calidad"] == "dudosa"),
        # Datos que no pasaron el control (falla de la estación o dato imposible).
        "datos_descartados": anotados,
        "viento_trabado": sorted(e["nombre"] for e in estaciones
                                 if "anemómetro" in (e.get("descartes") or {}).get("viento_kmh", "")),
        "estaciones": estaciones,
    }
    escribir(SALIDA, salida)
    print(json.dumps({"frescas": len(estaciones), "descartados": len(anotados),
                      "bahia": salida["bahia_blanca"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
