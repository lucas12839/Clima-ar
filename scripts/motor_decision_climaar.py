import json
import math
from datetime import datetime, timezone
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clasificacion_tiempo import clasificar, mantener_amenazas, mantener_fenomenos  # noqa: E402
import indices_fenomenos as fen  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
RADAR = BASE / "data/radar/radar_nowcast.json"
PRED = BASE / "data/radar/seguimiento/prediccion_evolucion.json"
INTEL = BASE / "data/ia/inteligencia_tormenta.json"
METEO = BASE / "data/ia/estado_meteorologico.json"
OUT = BASE / "data/ia/resultado_final.json"
FLUJO = BASE / "data/radar/nowcast_flujo.json"
CAPAS = BASE / "data/radar/nowcast_capas.json"
CELDAS = BASE / "data/ia/celdas_localidad.json"
INDICIOS = BASE / "data/ia/indicios_severos.json"
# Eco en arco (etapa G): encima si su borde está a <= 10 km de la localidad; llegando si
# va hacia ella (su línea la cubre de costado) y llega en <= 60 min.
ARCO_ENCIMA_KM = 10.0
BAHIA_LAT_LON = (-38.7183, -62.2663)
ARCO_LLEGADA_MAX_MIN = 60.0
# Etapa E (23/11/2026): de "justo sobre la localidad" se muestra SOLO la tormenta
# eléctrica (celdas con rayos, probabilidad >= 0,7). Simulación final, 29 días y 29.239
# casos (docs/SIMULACION_FINAL.md): de cada 100 avisos, 2 a 9 sobraron (pasó una tormenta
# fuerte o con rayos encima). Detecta 1 de cada 4 o 5: lo demás lo cubre la llegada a 10 km.
# Sierra de la Ventana queda afuera (47 % de avisos de más: el radar la ve mal detrás de
# las sierras).
SOBRE_LOCALIDAD_RAYOS_MIN = 0.7
SOBRE_LOCALIDAD_EXCLUIDAS = ("sierra_de_la_ventana",)
FLUJO_DBZ_MIN = 40.0
GOES = BASE / "data/ia/goes19_intelligence.json"
SUPERFICIE = BASE / "data/ia/estaciones/senales_superficie.json"
FLUJO_CONF_MIN = 0.5

VERSION = "ClimaAR-final-3.0"
VALID_SEVERITIES = {"verde", "amarillo", "naranja", "rojo"}
VALID_CLASSES = {
    "sin_tormenta",
    "tormenta",
    "tormenta_fuerte",
    "tormenta_severa",
}


def load(path):
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def finite(v, default=None):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def clamp(v, lo=0.0, hi=1.0):
    x = finite(v, lo)
    return max(lo, min(hi, x))


def timestamp_key(value):
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc)


def first_number(*values):
    for v in values:
        x = finite(v)
        if x is not None:
            return x
    return None


def radar_intensity(nowcast):
    nuclei = nowcast.get("nucleos_georreferenciados")
    if not isinstance(nuclei, list):
        nuclei = nowcast.get("nucleos_analizados", [])
    if not isinstance(nuclei, list):
        nuclei = []

    intensities = []
    areas = []
    changes = []

    for n in nuclei:
        if not isinstance(n, dict):
            continue
        intensity = first_number(
            n.get("intensidad"),
            n.get("intensidad_max"),
            n.get("reflectividad_max"),
        )
        area = first_number(n.get("area_px"), n.get("area"), n.get("pixeles"))
        change = first_number(
            n.get("cambio_area_pct"),
            n.get("cambio_intensidad_pct"),
            n.get("movimiento", {}).get("cambio_area_pct")
            if isinstance(n.get("movimiento"), dict)
            else None,
        )
        if intensity is not None:
            intensities.append(intensity)
        if area is not None:
            areas.append(area)
        if change is not None:
            changes.append(change)

    top = max(intensities) if intensities else first_number(
        nowcast.get("intensidad_max"),
        nowcast.get("reflectividad_max"),
        nowcast.get("intensidad_nivel_max"),
    )
    area = sum(areas) if areas else first_number(nowcast.get("area_px"), nowcast.get("area"))
    change = max(changes) if changes else first_number(nowcast.get("cambio_area_pct"))

    # Esto es SOLO una señal de intensidad radar. Nunca se traduce directamente
    # en severidad meteorológica.
    score = 0.0
    reasons = []
    nivel_max = top

    if nivel_max is not None:
        score += min(nivel_max / 6.0, 1.0) * 0.55
        if nivel_max >= 6:
            reasons.append("RMA10 nivel MUY_FUERTE")
        elif nivel_max >= 5:
            reasons.append("RMA10 nivel FUERTE")
        elif nivel_max >= 4:
            reasons.append("RMA10 nivel MODERADO_FUERTE")
        elif nivel_max >= 3:
            reasons.append("RMA10 nivel MODERADO")

    if area is not None:
        if area >= 5000:
            score += 0.15
        elif area >= 2000:
            score += 0.10
        elif area >= 500:
            score += 0.05

    if change is not None:
        if change >= 30:
            score += 0.20
            reasons.append("crecimiento espacial rapido")
        elif change >= 15:
            score += 0.12
            reasons.append("crecimiento espacial")
        elif change <= -30:
            score -= 0.12
            reasons.append("contraccion rapida")
        elif change <= -15:
            score -= 0.06
            reasons.append("contraccion")

    score = clamp(score)
    trend = "estable"
    if change is not None:
        if change >= 15:
            trend = "fortaleciendose"
        elif change <= -15:
            trend = "debilitandose"

    return {
        "score": round(score, 3),
        "max_reflectividad": round(top, 1) if top is not None else None,
        "area_total_px": round(area, 1) if area is not None else None,
        "cambio_max_pct": round(change, 1) if change is not None else None,
        "tendencia": trend,
        "nucleos_considerados": len(nuclei),
        "razones": reasons,
        "clasificacion": "INTENSIDAD_RADAR_NO_SEVERIDAD",
    }


def prediction_summary(pred):
    out = {
        "schema_version": pred.get("schema_version"),
        "radar_timestamp": pred.get("radar_timestamp"),
        "intervalo_minutos": pred.get("intervalo_minutos"),
    }
    resumen = pred.get("resumen", {})
    if isinstance(resumen, dict):
        out["resumen"] = resumen
    cal = pred.get("calibracion", {})
    if isinstance(cal, dict):
        out["calibracion"] = cal
    return out


def intelligence_score(intel):
    indices = intel.get("indices_heuristicos", {})
    hazards = indices.get("peligros", {})
    if not isinstance(indices, dict):
        indices = {}
    if not isinstance(hazards, dict):
        hazards = {}
    candidates = [
        indices.get("tormenta"),
        indices.get("llegada_bahia"),
        indices.get("fortalecimiento"),
        indices.get("indice_hibrido_severidad"),
    ] + list(hazards.values())
    vals = [finite(x) for x in candidates if finite(x) is not None]
    return max(vals) if vals else 0.0


def calibrated_visual_evidence(intel):
    visual = intel.get("ia_visual_temporal", {}) if isinstance(intel, dict) else {}
    visual_probs = visual.get("probabilidades", {}) if isinstance(visual, dict) else {}
    calibration_flags = visual.get("calibracion_por_modelo", {}) if isinstance(visual, dict) else {}
    intel_radar = intel.get("radar", {}) if isinstance(intel, dict) else {}
    intel_radar_timestamp = (
        intel.get("radar_timestamp")
        or (intel_radar.get("frame_actual_utc") if isinstance(intel_radar, dict) else None)
    )
    visual_sync = (
        isinstance(visual, dict)
        and timestamp_key(visual.get("radar_timestamp")) == timestamp_key(intel_radar_timestamp)
        and visual.get("secuencia_compensada_por_movimiento") is True
    )
    scores = []
    if visual_sync:
        for horizon in ("5", "10", "20", "30"):
            row = visual_probs.get(horizon, {})
            if not isinstance(row, dict):
                continue
            for target in ("strengthened", "toward_bahia"):
                key = f"{horizon}m_{target}"
                flag = calibration_flags.get(key, {})
                if not isinstance(flag, dict) or flag.get("aplicada") is not True:
                    continue
                value = finite(row.get(target))
                if value is not None and 0.0 <= value <= 1.0:
                    scores.append(value)
    return (max(scores) if scores else 0.0), bool(scores)


def optical_flow_evidence(flujo, nowcast):
    """Evidencia del nowcast por flujo óptico (física, sin entrenamiento).

    Solo cuenta si está sincronizado con el radar del nowcast y pronostica
    un eco >=35 dBZ entrando en un radio de 10 km de Bahía Blanca dentro de
    60 min. Pesa más cuanto antes llega y si llega con >=45 dBZ.
    Devuelve (score 0..1, incorporada, resumen).
    """
    resumen = {"disponible": False}
    if not isinstance(flujo, dict) or flujo.get("disponible") is not True:
        if isinstance(flujo, dict):
            resumen["motivo"] = flujo.get("motivo")
        return 0.0, False, resumen
    frame = nowcast.get("frame_actual_utc") if isinstance(nowcast, dict) else None
    if timestamp_key(flujo.get("radar_timestamp")) != timestamp_key(frame):
        resumen["motivo"] = "DESINCRONIZADO"
        return 0.0, False, resumen

    bahia = flujo.get("bahia_blanca") or {}
    conf = clamp(flujo.get("confianza", 0.0))
    eta = finite(bahia.get("eta_tormenta_desde_ahora_min"))
    dbz = finite(bahia.get("dbz_al_llegar"), 0.0)
    resumen = {
        "disponible": True,
        "confianza": round(conf, 3),
        "eta_tormenta_min": eta,
        "dbz_al_llegar": bahia.get("dbz_al_llegar"),
        "velocidad_kmh": (flujo.get("movimiento") or {}).get("velocidad_media_kmh"),
        "direccion_hacia_grados": (flujo.get("movimiento") or {}).get("direccion_hacia_grados"),
    }
    # Validación retrospectiva (12/10-19/11/2026): las llegadas pronosticadas
    # de 35-39 dBZ fueron casi todas falsas alarmas (lluvia en el umbral);
    # las reales llegaron con >=40 dBZ. Solo cuenta como evidencia de
    # tormenta con >=40 dBZ y confianza >=0.5.
    resumen["umbral_evidencia_dbz"] = FLUJO_DBZ_MIN
    if eta is None or eta > 60 or dbz < FLUJO_DBZ_MIN or conf < FLUJO_CONF_MIN:
        return 0.0, False, resumen
    cercania = clamp(1.0 - eta / 90.0)
    fuerza = 1.0 if dbz >= 45 else 0.7
    score = clamp(conf * cercania * fuerza)
    return score, score > 0, resumen


# Con qué frecuencia llega de verdad una llegada "probable" del 5b a la zona de
# la localidad (tormenta a <= 10 km): simulación de 26 días con el radar
# atrasado 10-30 min, 24 % de falsas alarmas -> llegó 76 de cada 100
# (herramientas/simular_envolvente.py, docs/VALIDACION_NOWCAST_CAPAS.md).
CONFIABILIDAD_5B = {"casos": 100, "llego": 76, "se_desvio": 0, "se_deshizo": 0,
                    "fuente": "nowcast_capas, simulación de 26 días"}


def llegadas_5b(flujo, capas=None):
    """Llegadas que se ven: desde el 20/11/2026 salen del nowcast en capas (5b).

    Para cada localidad, la primera llegada de tormenta (>= 35 dBZ) a su zona
    con 10 km de margen, grado "probable" (2 de 25 caminos): es lo validado en
    vivo y en simulación. La llegada "justo sobre la localidad" de tormenta NO
    se usa (todavía falla demasiado). El dBZ al llegar es el máximo que el 5b
    pronostica sobre la localidad desde esa hora. Si el 5b no está o no es de
    la misma imagen de radar, queda lo del flujo óptico (como antes).
    Solo agrega campos: lo del flujo queda en *_flujo_optico."""
    import copy

    flujo = flujo if isinstance(flujo, dict) else {}
    if capas is None:
        capas = load(CAPAS) if CAPAS.exists() else {}
    capas = capas if isinstance(capas, dict) else {}
    lleg_cfg = capas.get("llegada_proxima_hora") or {}
    if (capas.get("disponible") is not True or lleg_cfg.get("usado") is not True
            or timestamp_key(capas.get("radar_timestamp")) != timestamp_key(flujo.get("radar_timestamp"))):
        out = dict(flujo)
        out["fuente_llegadas"] = "flujo_optico"
        return out
    out = copy.deepcopy(flujo)
    out["fuente_llegadas"] = "nowcast_capas"
    locs = out.setdefault("localidades", {})
    for loc_id, c in (capas.get("localidades") or {}).items():
        l5 = (((c.get("llegada_proxima_hora") or {}).get("tormenta") or {}).get("probable") or {})
        eta = finite(l5.get("hasta_10km"))
        pron = c.get("pronostico_desde_ahora") or {}
        dbz = None
        if eta is not None:
            vals = [finite((pron.get(f"{p}min") or {}).get("dbz")) for p in (10, 20, 30, 60) if p >= eta]
            vals = [v for v in vals if v is not None]
            dbz = round(max(vals), 1) if vals else None
        d = locs.setdefault(loc_id, {"nombre": c.get("nombre")})
        ll = d.setdefault("llegada", {})
        ll["eta_desde_ahora_min_flujo_optico"] = ll.get("eta_desde_ahora_min")
        ll["dbz_al_llegar_flujo_optico"] = ll.get("dbz_al_llegar")
        ll["eta_desde_ahora_min"] = eta
        ll["dbz_al_llegar"] = dbz
        ll["fuente"] = "nowcast_capas: tormenta a <= 10 km de la localidad, grado probable"
        if loc_id == "bahia_blanca":
            b = out.setdefault("bahia_blanca", {})
            b["eta_tormenta_desde_ahora_min_flujo_optico"] = b.get("eta_tormenta_desde_ahora_min")
            b["dbz_al_llegar_flujo_optico"] = b.get("dbz_al_llegar")
            b["eta_tormenta_desde_ahora_min"] = eta
            b["dbz_al_llegar"] = dbz
            b["fuente_llegada"] = ll["fuente"]
    return out


def _superficie_fresca():
    datos = load(SUPERFICIE) if SUPERFICIE.exists() else {}
    if not isinstance(datos, dict):
        return {}
    try:
        edad = (
            datetime.now(timezone.utc)
            - datetime.fromisoformat(str(datos.get("generado_utc")).replace("Z", "+00:00"))
        ).total_seconds() / 60.0
    except (TypeError, ValueError):
        return {}
    return datos if edad <= 20 else {}


def distancia_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def rayos_para(rayos, lat, lon):
    """Rayos vistos desde la localidad: la distancia se mide al rayo más cercano."""
    salida = {k: v for k, v in (rayos or {}).items() if k not in ("latitud", "longitud", "por_localidad")}
    flat, flon = finite((rayos or {}).get("latitud")), finite((rayos or {}).get("longitud"))
    if flat is not None and flon is not None and lat is not None and lon is not None:
        salida["distancia_km"] = round(distancia_km(lat, lon, flat, flon), 1)
    else:
        salida.pop("distancia_km", None)
    propio = ((rayos or {}).get("por_localidad") or {})
    return salida, propio


def radar_movido(loc_id):
    """Radar viejo movido hasta ahora por el nowcast (nowcast_capas.json), si tiene <= 90 min."""
    capas = load(CAPAS) if CAPAS.exists() else {}
    if not isinstance(capas, dict) or not capas.get("disponible"):
        return None
    try:
        gen = datetime.fromisoformat(str(capas.get("generado_utc")).replace("Z", "+00:00"))
    except ValueError:
        return None
    if gen.tzinfo is None:
        gen = gen.replace(tzinfo=timezone.utc)
    desde_gen = (datetime.now(timezone.utc) - gen).total_seconds() / 60.0
    edad = finite(capas.get("edad_radar_min"))
    # La lluvia se movió hasta la hora en que se generó: más de 15 min después ya no vale.
    if edad is None or not (-1.0 <= desde_gen <= 15.0):
        return None
    mov = (((capas.get("localidades") or {}).get(loc_id) or {}).get("radar_movido_a_ahora")) or {}
    if "mm_h_medio" not in mov:
        return None
    # Edad calculada al usarla (no la guardada).
    return {"dbz_p75": mov.get("dbz_p75"), "mm_h_medio": mov.get("mm_h_medio"),
            "edad_min": round(edad + desde_gen, 1)}


def topes_por_localidad():
    """Topes GOES-19 (canal 13) por localidad, si la imagen tiene <= 30 min."""
    goes = load(GOES) if GOES.exists() else {}
    c13 = ((goes.get("productos") or {}).get("abi_c13") or {}) if isinstance(goes, dict) else {}
    edad = finite((c13.get("source") or {}).get("age_minutes"))
    if edad is not None and edad > 30:
        return {}
    return c13.get("por_localidad") or {}


CALIBRACION_LLEGADAS = BASE / "config/calibracion_llegadas.json"
CASOS_MINIMOS = 20


def confiabilidad_llegada(loc_id, eta, dbz, tabla=None):
    """Cuántas veces llegó de verdad un pronóstico parecido (30 días de radar).
    Usa la localidad si tiene >= 20 casos; si no, todas juntas."""
    if eta is None or dbz is None:
        return None
    if tabla is None:
        tabla = load(CALIBRACION_LLEGADAS) if CALIBRACION_LLEGADAS.exists() else {}
    plazo = "0-20" if eta < 20 else ("20-40" if eta < 40 else "40-60")
    inten = "35-40" if dbz < 40 else ("40-45" if dbz < 45 else ">=45")
    for fuente, base in (("localidad", ((tabla.get("por_localidad") or {}).get(loc_id) or {})),
                         ("region", tabla.get("global") or {})):
        fila = (base.get(plazo) or {}).get(inten)
        if isinstance(fila, dict) and fila.get("casos", 0) >= CASOS_MINIMOS:
            return dict(fila, fraccion=round(fila["llego"] / fila["casos"], 2), base=fuente,
                        plazo_min=plazo, intensidad_dbz=inten)
    return None


def ambiente_localidad(ambiente, modelo_loc):
    """El ambiente regional con el viento del modelo en el punto de la localidad."""
    if not isinstance(ambiente, dict) or not modelo_loc:
        return ambiente
    cambios = {k: modelo_loc[v] for k, v in (("wind_gusts_10m", "rafagas"), ("wind_speed_10m", "viento_vel"))
               if modelo_loc.get(v) is not None}
    return dict(ambiente, **cambios) if cambios else ambiente


def tipo_por_localidades(flujo, rayos, ambiente, radar_ok, sincronizado):
    """Índices por fenómeno de cada localidad (Bahía Blanca, Punta Alta, ...)."""
    import localidades_radar

    locs = (flujo or {}).get("localidades") or {}
    if not isinstance(locs, dict):
        locs = {}
    # Todas las localidades, aunque el radar no haya llegado (satélite y estaciones).
    for loc in localidades_radar.cargar_localidades():
        locs.setdefault(loc["id"], {"nombre": loc["nombre"], "lat": loc["lat"], "lon": loc["lon"],
                                    "sin_radar": True})
    mov = (flujo or {}).get("movimiento") or {}
    sup_locs = (_superficie_fresca().get("localidades") or {})
    # Viento del modelo en el punto de cada localidad (24/11/2026): antes las
    # 7 usaban el de Bahía. El resto del ambiente (cizalladura, CAPE, nivel de
    # congelamiento) sigue siendo el regional.
    import modelo_localidades as ml
    meteo_loc = (load(METEO) if METEO.exists() else {}) or {}
    modelo_locs = meteo_loc.get("modelo_localidades") if ml.vigente(meteo_loc.get("modelo_localidades")) else None
    topes = topes_por_localidad()
    tabla = load(CALIBRACION_LLEGADAS) if CALIBRACION_LLEGADAS.exists() else {}
    celdas = celdas_de(flujo)
    arcos = arcos_de(flujo)
    previo = load(OUT) if OUT.exists() else {}
    previas = (((previo.get("tipo_tiempo") or {}).get("localidades")) or {}) if isinstance(previo, dict) else {}
    salida = {}
    for loc_id, d in locs.items():
        if not isinstance(d, dict):
            continue
        lleg = d.get("llegada") or {}
        r_loc, propio = rayos_para(rayos, d.get("lat"), d.get("lon"))
        if loc_id in propio and isinstance(propio[loc_id], dict):
            r_loc.update(propio[loc_id])
        radar_al_dia = bool(radar_ok and sincronizado and not d.get("sin_radar"))
        ev = fen.evaluar(
            radar={
                "disponible": radar_al_dia,
                "movido": None if radar_al_dia else radar_movido(loc_id),
                "dbz_ciudad": finite(d.get("dbz_p75")),
                "dbz_pico": finite(d.get("dbz_pico")),
                "mm_h_area": finite(d.get("mm_h_medio")),
                "dbz_30km": finite(lleg.get("dbz_entorno_20km") or d.get("dbz_pico")),
                "dbz_llegada": finite(lleg.get("dbz_al_llegar")),
                "eta_llegada_min": finite(lleg.get("eta_desde_ahora_min")),
                # Movimiento local (ecos a <= 20 km), no el promedio del radar.
                "velocidad_kmh": finite(lleg.get("velocidad_local_kmh")),
                # Viento por radar: solo el eco en arco (etapa G, docs/ETAPA_G.md).
                "arco": arco_para(d.get("lat"), d.get("lon"), arcos) if radar_al_dia else None,
                "velocidad_eco_kmh": finite(lleg.get("velocidad_eco_kmh")),
                "coherencia": finite(mov.get("coherencia")),
                "minutos_lluvia_intensa": finite(d.get("minutos_lluvia_intensa")),
                "lluvia_1h_mm": finite(d.get("lluvia_1h_mm")),
                "lluvia_3h_mm": finite(d.get("lluvia_3h_mm")),
                "lluvia_6h_mm": finite(d.get("lluvia_6h_mm")),
                "confiabilidad_llegada": (
                    CONFIABILIDAD_5B if (flujo or {}).get("fuente_llegadas") == "nowcast_capas"
                    else confiabilidad_llegada(
                        loc_id, finite(lleg.get("eta_desde_ahora_min")), finite(lleg.get("dbz_al_llegar")), tabla)),
            },
            superficie=sup_locs.get(loc_id) or {},
            rayos=r_loc,
            ambiente=ambiente_localidad(ambiente, ml.valor_actual(modelo_locs, loc_id)),
            satelite=topes.get(loc_id) or {},
        )
        ev["nombre"] = d.get("nombre", loc_id)
        ev["velocidad_local_kmh"] = finite(lleg.get("velocidad_local_kmh"))
        ev["dbz_entorno_20km"] = finite(lleg.get("dbz_entorno_20km"))
        ev["tendencia"] = tendencia(ev["dbz_entorno_20km"],
                                    finite((previas.get(loc_id) or {}).get("dbz_entorno_20km")))
        ev["velocidad_eco_kmh"] = finite(lleg.get("velocidad_eco_kmh"))
        ev["acumulado_radar_mm"] = {"1h": d.get("lluvia_1h_mm"), "3h": d.get("lluvia_3h_mm"),
                                    "6h": d.get("lluvia_6h_mm")}
        if sup_locs.get(loc_id):
            ev["estaciones"] = sup_locs[loc_id].get("nombres")
        ev["sobre_la_localidad"] = sobre_la_localidad(loc_id, celdas, radar_al_dia)
        salida[loc_id] = ev
    return salida


def arcos_de(flujo):
    """Ecos en arco de indicios_severos.json de la misma imagen de radar."""
    c = load(INDICIOS) if INDICIOS.exists() else {}
    if not isinstance(c, dict) or c.get("disponible") is not True:
        return []
    if timestamp_key(c.get("radar_timestamp")) != timestamp_key((flujo or {}).get("radar_timestamp")):
        return []
    return [t for t in c.get("tormentas") or [] if t.get("arco") and t.get("lat") is not None]


def arco_para(lat, lon, arcos):
    """El eco en arco que afecta a la localidad: encima (minutos 0) o llegando (<= 60 min)."""
    import math

    mejor = None
    if lat is None or lon is None:
        return None
    for a in arcos:
        cos0 = math.cos(math.radians(lat))
        dx = (lon - a["lon"]) * 111.32 * cos0          # de la tormenta a la localidad (km)
        dy = (lat - a["lat"]) * 110.57
        mitad = (finite(a.get("largo_km")) or 0.0) / 2.0
        radio = math.sqrt((finite(a.get("area_km2")) or 0.0) / math.pi)
        dist = max(0.0, math.hypot(dx, dy) - radio)
        if dist <= ARCO_ENCIMA_KM:
            cand = {"minutos": 0, "distancia_km": round(dist, 1), "velocidad_kmh": a.get("velocidad_kmh")}
        else:
            vel, rumbo = finite(a.get("velocidad_kmh")), finite(a.get("direccion_hacia_grados"))
            if not vel or rumbo is None:
                continue
            ux, uy = math.sin(math.radians(rumbo)), math.cos(math.radians(rumbo))
            adelante, costado = dx * ux + dy * uy, abs(-dx * uy + dy * ux)
            if adelante <= 0 or costado > mitad + ARCO_ENCIMA_KM:
                continue
            minutos = max(0.0, adelante - radio) / vel * 60.0
            if minutos > ARCO_LLEGADA_MAX_MIN:
                continue
            cand = {"minutos": round(minutos), "distancia_km": round(dist, 1), "velocidad_kmh": vel}
        if mejor is None or cand["minutos"] < mejor["minutos"]:
            mejor = cand
    return mejor


def celdas_de(flujo):
    """celdas_localidad.json de la misma imagen de radar (si no, nada)."""
    c = load(CELDAS) if CELDAS.exists() else {}
    if not isinstance(c, dict) or c.get("disponible") is not True:
        return {}
    if timestamp_key(c.get("radar_timestamp")) != timestamp_key((flujo or {}).get("radar_timestamp")):
        return {}
    return c.get("localidades") or {}


def sobre_la_localidad(loc_id, celdas, radar_al_dia):
    """Tormenta eléctrica que pasa SOBRE la localidad en los próximos 30 min (validada)."""
    c = (celdas or {}).get(loc_id) or {}
    p = finite(c.get("probabilidad_solo_rayos"))
    if (not radar_al_dia or loc_id in SOBRE_LOCALIDAD_EXCLUIDAS or p is None
            or p < SOBRE_LOCALIDAD_RAYOS_MIN):
        return None
    minutos = finite(c.get("minutos_solo_rayos"))
    return {"que": "tormenta eléctrica", "probabilidad": round(p, 2),
            "minutos": minutos, "texto": ("tormenta eléctrica encima" if minutos is not None and minutos <= 0
                                          else "tormenta eléctrica sobre la localidad en los próximos 30 min"),
            "validacion": "simulación final: de cada 100 avisos sobraron 2 a 9"}


TENDENCIA_DBZ = 5.0


def tendencia(actual, previo):
    """Cómo cambian las tormentas a <= 20 km entre un ciclo y el siguiente."""
    if actual is None or previo is None:
        return None
    if actual - previo >= TENDENCIA_DBZ:
        return "se intensifican"
    if previo - actual >= TENDENCIA_DBZ:
        return "se debilitan"
    return "estables"


LIMITES_FUENTES_MIN = {"radar": 30, "satelite": 30, "rayos": 15, "modelo": 90, "estaciones": 30}


def _edad_min(texto, ahora):
    try:
        t = datetime.fromisoformat(str(texto).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return round((ahora - t).total_seconds() / 60.0, 1)


def estado_fuentes(ahora=None):
    """Qué fuente está al día y cuál no: para no trabajar nunca a ciegas.

    Cada fuente: edad del último dato (min) y si está al día. "a_ciegas" es
    true solo si no queda ninguna fuente de observación (radar, satélite,
    rayos ni estaciones) al día."""
    ahora = ahora or datetime.now(timezone.utc)
    flujo = load(FLUJO) if FLUJO.exists() else {}
    goes = load(GOES) if GOES.exists() else {}
    meteo = load(METEO) if METEO.exists() else {}
    sup = load(SUPERFICIE) if SUPERFICIE.exists() else {}
    goes = goes if isinstance(goes, dict) else {}
    c13 = ((goes.get("productos") or {}).get("abi_c13") or {}).get("source") or {}
    ventana = ((goes.get("glm") or {}).get("ventana")) or {}
    edades = {
        "radar": _edad_min((flujo or {}).get("radar_timestamp") or (flujo or {}).get("frame_actual_utc"), ahora),
        "satelite": _edad_min(c13.get("end_time_utc"), ahora),
        "rayos": _edad_min(ventana.get("fin_ventana_utc"), ahora) if ventana.get("disponible") else None,
        "modelo": _edad_min(((meteo or {}).get("ecmwf_cache") or {}).get("actualizado_utc")
                            if ((meteo or {}).get("ecmwf_cache") or {}).get("usado")
                            else (meteo or {}).get("actualizado_utc"), ahora),
        "estaciones": _edad_min((sup or {}).get("generado_utc"), ahora)
                      if (sup or {}).get("estaciones_frescas") else None,
    }
    fuentes = {}
    for k, edad in edades.items():
        fuentes[k] = {"edad_min": edad,
                      "al_dia": edad is not None and edad <= LIMITES_FUENTES_MIN[k],
                      "limite_min": LIMITES_FUENTES_MIN[k]}
    fuentes["estaciones"]["frescas"] = (sup or {}).get("estaciones_frescas")
    fuentes["estaciones"]["descartadas_por_calidad"] = (sup or {}).get("descartadas_por_calidad") or []
    observacion = [k for k in ("radar", "satelite", "rayos", "estaciones") if fuentes[k]["al_dia"]]
    faltan = [k for k, v in fuentes.items() if not v["al_dia"]]
    return {
        "fuentes": fuentes,
        "al_dia": observacion,
        "atrasadas": faltan,
        "a_ciegas": not observacion,
        "resumen": ("Todas las fuentes al día" if not faltan
                    else "Sin datos al día de: " + ", ".join(faltan)),
    }


def resumen_regional(localidades):
    """Localidades con algún fenómeno, de mayor a menor."""
    filas = [
        {"id": k, "nombre": v["nombre"], "calificacion": v["calificacion"]["etiqueta"],
         "grado": v["calificacion"]["grado"], "resumen": v["resumen"],
         "minutos_hasta_llegada": v.get("minutos_hasta_llegada")}
        for k, v in localidades.items() if v["calificacion"]["grado"] > 0
    ]
    return sorted(filas, key=lambda f: -f["grado"])


def superficie_bahia():
    """Resumen de estaciones de Bahía Blanca si es reciente (<= 20 min)."""
    datos = load(SUPERFICIE) if SUPERFICIE.exists() else {}
    if not isinstance(datos, dict):
        return {}
    generado = datos.get("generado_utc")
    try:
        edad = (
            datetime.now(timezone.utc)
            - datetime.fromisoformat(str(generado).replace("Z", "+00:00"))
        ).total_seconds() / 60.0
    except (TypeError, ValueError):
        return {}
    if edad > 20:
        return {}
    return datos.get("bahia_blanca") or {}


def tipo_de_tiempo(flujo, intel, nowcast):
    """Escala descriptiva (llovizna ... tormenta potencialmente severa)."""
    flujo = flujo if isinstance(flujo, dict) else {}
    obs = flujo.get("observacion") or {}
    bahia = flujo.get("bahia_blanca") or {}
    mov = flujo.get("movimiento") or {}
    goes = load(GOES) if GOES.exists() else {}
    glm = (goes.get("glm") or {}) if isinstance(goes, dict) else {}
    rayos = {
        "flashes": (glm.get("counts") or {}).get("flashes"),
        "distancia_km": (glm.get("nearest_flash") or {}).get("distance_km"),
        "latitud": (glm.get("nearest_flash") or {}).get("latitude"),
        "longitud": (glm.get("nearest_flash") or {}).get("longitude"),
    }
    ventana = glm.get("ventana") or {}
    if isinstance(ventana, dict) and ventana.get("disponible"):
        rayos.update({
            "tasa_30km_por_min": ventana.get("tasa_30km_por_min"),
            "tasa_30km_previa_por_min": ventana.get("tasa_30km_previa_por_min"),
            "por_localidad": ventana.get("por_localidad") or {},
        })
    vigencia = (intel.get("vigencia_radar") or {}) if isinstance(intel, dict) else {}
    radar_ok = bool(obs) and vigencia.get("modo") != "radar_obsoleto"
    sincronizado = timestamp_key(flujo.get("radar_timestamp")) == timestamp_key(
        (nowcast or {}).get("frame_actual_utc")
    )
    ambiente = (intel.get("ambiente") or {}) if isinstance(intel, dict) else {}
    resultado = clasificar(
        dbz_ciudad=finite(obs.get("dbz_ciudad")),
        dbz_llegada=finite(bahia.get("dbz_al_llegar")),
        eta_llegada_min=finite(bahia.get("eta_tormenta_desde_ahora_min")),
        dbz_region=finite(obs.get("dbz_region")),
        dbz_30km=finite(obs.get("dbz_30km")),
        lluvia_3h_mm=finite(obs.get("lluvia_radar_3h_mm")),
        lluvia_6h_mm=finite(obs.get("lluvia_radar_6h_mm")),
        distancia_region_km=finite(obs.get("distancia_nucleo_region_km")),
        velocidad_kmh=finite(mov.get("velocidad_media_kmh")),
        coherencia=finite(mov.get("coherencia")),
        minutos_lluvia_intensa=finite(obs.get("minutos_lluvia_intensa")),
        rayos={k: rayos.get(k) for k in ("flashes", "distancia_km", "tasa_30km_por_min",
                                          "tasa_30km_previa_por_min") if rayos.get(k) is not None},
        ambiente=ambiente,
        radar_disponible=radar_ok and sincronizado,
        superficie=superficie_bahia(),
        arco=arco_para(BAHIA_LAT_LON[0], BAHIA_LAT_LON[1], arcos_de(flujo)) if (radar_ok and sincronizado) else None,
    )
    localidades = tipo_por_localidades(flujo, rayos, ambiente, radar_ok, sincronizado)
    # Bahía Blanca: los índices salen del análisis por localidad (lluvia media
    # del área, no el píxel más intenso). La escala 0-8 y las amenazas siguen igual.
    if "bahia_blanca" in localidades:
        bb = localidades["bahia_blanca"]
        for clave in ("ahora", "resumen_ahora", "al_llegar", "resumen_al_llegar",
                      "fenomenos", "resumen", "calificacion", "llegada", "ambiente"):
            resultado[clave] = bb[clave]
        if bb.get("minutos_hasta_llegada") is not None:
            resultado["minutos_hasta_llegada"] = bb["minutos_hasta_llegada"]
    resultado["localidades"] = localidades
    resultado["resumen_regional"] = resumen_regional(localidades)
    return resultado


def mantener_localidades(tipo, anterior, ahora_utc, anterior_utc):
    """Cada localidad mantiene su fenómeno severo 40 min, como Bahía Blanca."""
    from clasificacion_tiempo import mantener_fenomenos as _mf

    def parse(texto):
        try:
            return datetime.fromisoformat(str(texto).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None

    ahora = parse(ahora_utc)
    previas = ((anterior or {}).get("localidades") or {}) if isinstance(anterior, dict) else {}
    for loc_id, loc in (tipo.get("localidades") or {}).items():
        _mf(loc, previas.get(loc_id) if isinstance(previas, dict) else None, ahora, ahora_utc, anterior_utc, parse)
    tipo["resumen_regional"] = resumen_regional(tipo.get("localidades") or {})
    return tipo


def main():
    radar = load(RADAR)
    pred = load(PRED)
    intel = load(INTEL)
    meteo = load(METEO)

    nowcast = radar.get("nowcast", {})
    if not isinstance(nowcast, dict):
        nowcast = {}

    intensity = radar_intensity(nowcast)
    ia_score = intelligence_score(intel)

    # La evidencia visual solo influye si está calibrada y sincronizada.
    visual_score, visual_incorporated = calibrated_visual_evidence(intel)
    if visual_incorporated:
        ia_score = clamp(0.75 * ia_score + 0.25 * visual_score)

    # Flujo óptico: solo puede subir la señal (una llegada pronosticada).
    flujo_visible = llegadas_5b(load(FLUJO) if FLUJO.exists() else {})
    flujo_score, flujo_incorporado, flujo_resumen = optical_flow_evidence(
        flujo_visible,
        radar.get("nowcast", {}),
    )
    flujo_resumen["fuente_llegadas"] = flujo_visible.get("fuente_llegadas")
    if flujo_incorporado:
        ia_score = max(ia_score, clamp(0.75 * ia_score + 0.25 * flujo_score), flujo_score * 0.6)

    tipo_tiempo = tipo_de_tiempo(
        flujo_visible,
        intel,
        radar.get("nowcast", {}),
    )
    tipo_tiempo["fuente_llegadas"] = flujo_visible.get("fuente_llegadas")
    anterior = load(OUT) if OUT.exists() else {}
    tipo_tiempo = mantener_amenazas(
        tipo_tiempo,
        anterior.get("tipo_tiempo") if isinstance(anterior, dict) else None,
        str(nowcast.get("frame_actual_utc") or ""),
        ((anterior.get("radar") or {}).get("timestamp") if isinstance(anterior, dict) else None),
    )
    tipo_tiempo = mantener_localidades(
        tipo_tiempo,
        anterior.get("tipo_tiempo") if isinstance(anterior, dict) else None,
        str(nowcast.get("frame_actual_utc") or ""),
        ((anterior.get("radar") or {}).get("timestamp") if isinstance(anterior, dict) else None),
    )

    model_conf = clamp(
        intel.get("confianza_modelo", intel.get("confianza", 0.0))
    )

    radar_activity = bool(nowcast.get("actividad"))
    raw_movement = clamp(nowcast.get("confianza_movimiento", 0.0))
    speed = finite(nowcast.get("velocidad_kmh"))
    direction = finite(nowcast.get("direccion_grados"))
    moving_to_bahia = bool(nowcast.get("movimiento_hacia_bahia"))

    if speed is None or speed <= 5 or direction is None:
        movement = 0.0
    else:
        movement = raw_movement if moving_to_bahia else min(raw_movement, 0.5)

    indices = intel.get("indices_heuristicos", {})
    hazard_signals = indices.get("peligros", {})
    if not isinstance(indices, dict):
        indices = {}
    if not isinstance(hazard_signals, dict):
        hazard_signals = {}

    storm_signal = clamp(indices.get("tormenta", 0.0))
    arrival_signal = clamp(indices.get("llegada_bahia", 0.0))

    # Estado operativo: describe actividad/evolución. NO es severidad meteorológica.
    if radar_activity:
        operational_score = (
            0.30 * storm_signal
            + 0.25 * intensity["score"]
            + 0.20 * movement
            + 0.25 * ia_score
        )
    else:
        operational_score = min(
            0.15 * storm_signal + 0.10 * intensity["score"] + 0.75 * ia_score,
            0.35,
        )

    operational_score = clamp(operational_score)

    if operational_score >= 0.75:
        state = "actividad_muy_fuerte"
    elif operational_score >= 0.55:
        state = "fuerte"
    elif operational_score >= 0.35:
        state = "activa"
    elif operational_score >= 0.15:
        state = "incipiente"
    else:
        state = "sin_actividad_significativa"

    if intensity["tendencia"] == "fortaleciendose" and operational_score >= 0.35:
        state = "fortaleciendose"
    elif intensity["tendencia"] == "debilitandose" and operational_score < 0.60:
        state = "debilitandose"

    threat = intel.get("amenaza_bahia")
    if not isinstance(threat, dict):
        threat = {}
    threatens = threat.get("amenaza") is True
    # Una llegada pronosticada por flujo óptico en <=30 min con confianza
    # razonable también es amenaza para la ciudad.
    eta_flujo = finite(flujo_resumen.get("eta_tormenta_min"))
    if (
        flujo_incorporado
        and eta_flujo is not None
        and eta_flujo <= 30
        and flujo_resumen.get("confianza", 0) >= 0.5
    ):
        threatens = True

    # Un sistema que no amenaza a Bahia Blanca no puede presentarse como
    # estado fuerte/muy fuerte para la ciudad solo por su eco radar.
    if radar_activity and not threatens and state in (
        "actividad_muy_fuerte",
        "fuerte",
    ):
        state = "activa"

    severity_raw = intel.get("severidad")
    severity = (
        str(severity_raw).strip().lower()
        if severity_raw is not None
        else ""
    )
    severity = {"roja": "rojo"}.get(severity, severity)
    if severity not in VALID_SEVERITIES:
        severity = "no_determinada"

    intel_radar = intel.get("radar")
    if not isinstance(intel_radar, dict):
        intel_radar = {}

    weather_class = intel.get("clasificacion_meteorologica")
    if weather_class not in VALID_CLASSES:
        weather_class = "no_determinada"

    geo = nowcast.get("georreferenciacion", {})
    if not isinstance(geo, dict):
        geo = {}
    bahia = nowcast.get("bahia_blanca", {})
    if not isinstance(bahia, dict):
        bahia = {}

    principal_raw = nowcast.get("nucleo_principal_id")
    bahia_principal_raw = bahia.get("nucleo_principal_id")

    if (
        principal_raw is not None
        and bahia_principal_raw is not None
        and str(principal_raw) != str(bahia_principal_raw)
    ):
        raise RuntimeError(
            "ERROR: Decision Engine recibió principales inconsistentes: "
            f"nowcast={principal_raw} bahia={bahia_principal_raw}"
        )

    principal_id = first_number(principal_raw)

    if principal_id is None:
        # Sin principal solo es válido si el radar no tiene actividad.
        if radar_activity:
            raise RuntimeError(
                "ERROR: Decision Engine no recibió un principal operativo."
            )
    else:
        principal_id = int(principal_id)

    result = {
        "version": VERSION,
        "actualizado_utc": datetime.now(timezone.utc).isoformat(),
        "fuente_radar": "RMA10",
        "motor_decision": {
            "estado": state,
            "estado_operativo": state,
            "severidad_meteorologica": severity,
            "clasificacion_meteorologica": weather_class,
            "score_integrado": round(operational_score, 3),
            "actividad_radar": radar_activity,
            "componentes": {
                "radar_intensidad": intensity["score"],
                "radar_movimiento": movement,
                "indice_inteligencia": round(ia_score, 3),
                "ia": round(ia_score, 3),
                "ia_tipo_salida": "indice_heuristico",
                "ia_confianza": model_conf,
                "ia_visual_temporal_incorporada": visual_incorporated,
                "ia_visual_temporal_score": round(visual_score, 4),
                "flujo_optico_incorporado": flujo_incorporado,
                "flujo_optico_score": round(flujo_score, 4),
            },
            "flujo_optico": flujo_resumen,
            "tipo_tiempo": tipo_tiempo,
            "regla": (
                "La intensidad RMA10 es una señal; la severidad meteorológica "
                "proviene de Intelligence/IA/histórico y no del color radar."
            ),
        },
        "clasificacion_meteorologica": weather_class,
        "tipo_tiempo": tipo_tiempo,
        "estaciones_bahia": superficie_bahia(),
        "estado_fuentes": estado_fuentes(),
        "amenaza_bahia": threat,
        "clasificacion_historica": intel.get("clasificacion_historica"),
        "severidad_meteorologica": {
            "valor": severity,
            "fuente": "inteligencia_tormenta",
            "determinada": severity != "no_determinada",
        },
        "principal": {
            "track_id": principal_id,
            "fuente": "radar_nowcast",
        },
        "radar": {
            "timestamp": (
                radar.get("radar_timestamp")
                if isinstance(radar.get("radar_timestamp"), str)
                and radar.get("radar_timestamp").strip()
                else nowcast.get("frame_actual_utc")
            ),
            "nowcast": nowcast,
            "intensidad": intensity,
            "georreferenciacion": geo,
            "bahia_blanca": bahia,
        },
        "prediccion": prediction_summary(pred),
        "inteligencia": intel,
        "meteorologia": meteo,
        "senales_heuristicas": {
            "tormenta": storm_signal,
            "llegada_bahia": arrival_signal,
            "fortalecimiento": finite(indices.get("fortalecimiento")),
            "debilitamiento": finite(indices.get("debilitamiento")),
            "lluvia_fuerte": finite(hazard_signals.get("lluvia_fuerte")),
            "viento_fuerte": finite(hazard_signals.get("viento_fuerte")),
            "granizo": finite(hazard_signals.get("granizo")),
            "rayo": finite(hazard_signals.get("rayo")),
        },
        "resumen_operativo": {
            "probabilidad_tormenta": None,
            "probabilidad_llegada_bahia": None,
            "probabilidad_fortalecimiento": None,
            "probabilidad_debilitamiento": None,
            "lluvia_fuerte": None,
            "viento_fuerte": None,
            "granizo": None,
            "rayo": None,
            "eta_minutos": first_number(
                intel_radar.get("eta_minutos"),
                nowcast.get("eta_minutos"),
                bahia.get("eta_minutos"),
            ),
            "distancia_km": first_number(
                intel_radar.get("distancia_km"),
                nowcast.get("distancia_km"),
                bahia.get("distancia_principal_km"),
            ),
            "severidad_meteorologica": severity,
            "clasificacion_meteorologica": weather_class,
            "amenaza_bahia": threat,
            "vigencia_radar": intel_radar.get("vigencia_radar"),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False),  # compacto: 1 MB -> ~0,5 MB por ciclo (auditoría 20/11/2026)
        encoding="utf-8",
    )
    tmp.replace(OUT)

    print("CLIMAAR MOTOR DECISION OK")
    print("Estado operativo:", state)
    print("Severidad meteorologica:", severity)
    print("Score operativo:", round(operational_score, 3))
    print("Intensidad radar:", intensity["score"])
    print("Track principal:", principal_id)
    print("Salida:", OUT)


if __name__ == "__main__":
    main()
