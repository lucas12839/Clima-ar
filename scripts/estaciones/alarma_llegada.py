#!/usr/bin/env python3
"""Idea 3 (en paralelo, no cambia lo que se ve): estaciones como alarma de "ya llega".

Cuando una tormenta empuja aire frío hacia adelante (frente de ráfagas), las
estaciones que toca registran en minutos: salto del viento o ráfaga, caída de la
temperatura y subida de la presión. Las estaciones BCP miden cada minuto y
publican al instante: si la firma aparece en una estación que está "aguas arriba"
de una localidad (de donde viene la tormenta), la tormenta llega a la localidad en
distancia / velocidad minutos, antes de que lo muestre el radar (10-30 min tarde).

Reglas (física, sin entrenamiento):
- Solo datos de <= 15 min de edad, calculada AHORA con frescura.py (para todas las
  redes; el SMN de 90 min no sirve para "ya llega").
- Firma en la estación (últimos 30 min): pico de viento en los últimos 15 min (ráfaga
  >= 35 km/h, o el viento sube >= 10 km/h sobre lo que venía y llega a >= 15 km/h) Y
  además, desde el pico, caída de temperatura >= 1,5 °C o subida de presión >= 1,5 hPa
  (promedios de dos datos, para no confundir el ruido del sensor de presión).
- Tiene que haber tormenta cerca de la estación: eco de >= 35 dBZ a <= 40 km en la
  última imagen del radar o rayos a <= 30 km en los últimos 10 min. Sin eso, una
  racha de viento sola no es una tormenta.
- Localidad aguas abajo: a lo largo del movimiento a >= 5 km (si está al costado de
  la estación, no es "aguas abajo": 20/11, Arroyo Pantanoso -> Sierra de la Ventana
  falló 4 de 4) y de costado <= 30 km (un frente de ráfagas mide decenas de km de
  ancho; 20/11, Bahía Centro -> Punta Alta a 23 km de costado acertó 3 de 3). Llegada estimada =
  distancia a lo largo / velocidad, menos la edad del dato; solo hasta 60 min.
- Estación a <= 15 km de la localidad con firma ahora: "llegó".

Validación en vivo: cada alarma se verifica con lo que pasó después en la
localidad (eco de >= 35 dBZ en la zona de 10 km en las imágenes del radar, o firma
en una estación de la localidad) dentro de la hora estimada ± 20 min.

Lee (no escribe): historial_superficie.json, nowcast_flujo.json, goes19_state.json,
la imagen actual del radar. Escribe solo data/ia/estaciones/alarma_llegada.json.
Nunca corta el ciclo.
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

BASE = Path(__file__).resolve().parents[2]
for _p in (BASE, BASE / "scripts", BASE / "scripts" / "estaciones"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import frescura  # noqa: E402
import senales_superficie as ss  # noqa: E402

VERSION = "alarma-llegada-1.0"
SALIDA = BASE / "data/ia/estaciones/alarma_llegada.json"
FLUJO = BASE / "data/radar/nowcast_flujo.json"
GOES = BASE / "data/goes19/goes19_state.json"
LOCALIDADES = BASE / "config/localidades.json"

EDAD_MAX_MIN = 15.0
VENTANA_MIN = 30.0
RAFAGA_KMH = 35.0
SALTO_VIENTO_KMH = 10.0
VIENTO_MIN_KMH = 15.0
CAIDA_TEMP_C = 1.5
SUBIDA_PRESION_HPA = 1.5
RADIO_ECO_KM = 40.0
RADIO_RAYOS_KM = 30.0
UMBRAL_ECO_DBZ = 35.0
COSTADO_MAX_KM = 30.0
A_LO_LARGO_MIN_KM = 5.0
LLEGADA_MAX_MIN = 60.0
VELOCIDAD_MIN_KMH = 10.0
RADIO_LOCALIDAD_KM = 15.0
TOLERANCIA_MIN = 20.0
ZONA_VERIFICACION_KM = 10.0
PENDIENTES_MAX = 300


def parse_utc(texto: Any) -> datetime | None:
    return frescura.parse_utc(texto) if isinstance(texto, str) else None


def iso(v: datetime | None) -> str | None:
    return v.isoformat().replace("+00:00", "Z") if v else None


def leer(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def escribir(path: Path, datos: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as h:
            json.dump(datos, h, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _media(valores: list[float]) -> float:
    return sum(valores) / len(valores)


def firma(serie: list[dict], ahora: datetime) -> dict[str, Any] | None:
    """Firma de frente de ráfagas en los últimos 30 min de una estación (o None).

    El momento del frente es el pico de viento (o ráfaga) de los últimos 15 min.
    Se compara con lo que venía antes del pico: viento (mediana), temperatura
    (máxima antes contra mínima desde el pico) y presión (promedio de los 2 datos
    más bajos antes contra el de los 2 más altos desde el pico: el sensor de
    presión salta ±1 hPa solo)."""
    pts = []
    for r in serie:
        t = parse_utc(r.get("utc"))
        if t is not None and ahora - timedelta(minutes=VENTANA_MIN) <= t <= ahora:
            pts.append((t, r))
    if len(pts) < 3:
        return None
    pts.sort(key=lambda p: p[0])

    def num(r, campo):
        v = r.get(campo)
        return float(v) if v is not None else None

    recientes = [i for i, (t, _) in enumerate(pts) if t >= ahora - timedelta(minutes=VENTANA_MIN / 2) and i >= 1]
    mejor = None
    for i in recientes:
        r = pts[i][1]
        antes_v = [num(x, "viento") for _, x in pts[:i] if num(x, "viento") is not None]
        rafaga, viento = num(r, "rafaga"), num(r, "viento")
        señal = None
        if rafaga is not None and rafaga >= RAFAGA_KMH:
            señal = f"ráfaga {rafaga:.0f} km/h"
        elif viento is not None and antes_v:
            base = sorted(antes_v)[len(antes_v) // 2]
            if viento >= VIENTO_MIN_KMH and viento - base >= SALTO_VIENTO_KMH:
                señal = f"viento de {base:.0f} a {viento:.0f} km/h"
        if señal and (mejor is None or max(rafaga or 0, viento or 0) > mejor[2]):
            mejor = (i, señal, max(rafaga or 0, viento or 0))
    if mejor is None:
        return None
    i, señal_viento, _ = mejor
    antes, despues = pts[:i], pts[i:]
    señales = [señal_viento]
    t_antes = [num(x, "temp") for _, x in antes if num(x, "temp") is not None]
    t_desp = [num(x, "temp") for _, x in despues if num(x, "temp") is not None]
    caida = round(max(t_antes) - min(t_desp), 1) if t_antes and t_desp else None
    p_antes = sorted(num(x, "presion") for _, x in antes if num(x, "presion") is not None)
    p_desp = sorted((num(x, "presion") for _, x in despues if num(x, "presion") is not None), reverse=True)
    subida = None
    if len(p_antes) >= 2 and len(p_desp) >= 2:
        subida = round(_media(p_desp[:2]) - _media(p_antes[:2]), 1)
    if caida is not None and caida >= CAIDA_TEMP_C:
        señales.append(f"temperatura -{caida} °C")
    if subida is not None and subida >= SUBIDA_PRESION_HPA:
        señales.append(f"presión +{subida} hPa")
    if len(señales) < 2:
        return None
    return {"hora_utc": iso(pts[i][0]), "senales": señales,
            "caida_temperatura_c": caida, "subida_presion_hpa": subida}


def km_xy(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Este y norte (km) de un punto respecto de otro (plano local, alcanza para 240 km)."""
    return ((lon - lon0) * 111.32 * math.cos(math.radians(lat0)), (lat - lat0) * 110.57)


def movimiento(ahora: datetime) -> dict[str, Any] | None:
    """Velocidad y dirección (hacia dónde va) de las tormentas, del nowcast de este ciclo."""
    d = leer(FLUJO)
    m = d.get("movimiento") or {}
    vel, dire = m.get("velocidad_media_kmh"), m.get("direccion_hacia_grados")
    radar = parse_utc(d.get("radar_timestamp"))
    if vel is None or dire is None or radar is None or ahora - radar > timedelta(minutes=60):
        return None
    return {"velocidad_kmh": float(vel), "direccion_hacia_grados": float(dire), "radar_utc": iso(radar)}


def contexto_tormenta(ahora: datetime):
    """Funciones (lat, lon) -> hay eco fuerte cerca / hay rayos cerca. Sin datos: False."""
    eco = lambda lat, lon: False  # noqa: E731
    rayos = lambda lat, lon: False  # noqa: E731
    try:
        import nowcast_capas_nucleo as nc
        import nowcast_flujo_optico as nf
        import numpy as np

        d = nf.cargar_dbz(nf.ACTUAL) if nf.ACTUAL.exists() else None
        radar = parse_utc(nf.TIMESTAMP_FILE.read_text(encoding="utf-8").strip()) if nf.TIMESTAMP_FILE.exists() else None
        if d is not None and radar is not None and ahora - radar <= timedelta(minutes=60):
            campo = nc.campo_limpio(d, nc.cargar_clutter_total(d.shape))
            h, w = campo.shape
            km_px = nf.km_por_pixel(w)
            ys, xs = np.nonzero(np.nan_to_num(campo) >= UMBRAL_ECO_DBZ)

            def eco(lat, lon):  # noqa: F811
                if xs.size == 0:
                    return False
                x, y = nf.latlon_to_pixel(lat, lon, w, h)
                return bool((np.hypot(xs - x, ys - y) * km_px <= RADIO_ECO_KM).any())
    except Exception:
        pass
    ventana = ((leer(GOES).get("glm") or {}).get("ventana")) or {}
    fin = parse_utc(ventana.get("fin_ventana_utc"))
    if ventana.get("disponible") and fin is not None and ahora - fin <= timedelta(minutes=15):
        pos = [p for p in ventana.get("rayos_10min") or [] if len(p) >= 2]

        def rayos(lat, lon):  # noqa: F811
            return any(ss.distancia_km(lat, lon, p[0], p[1]) <= RADIO_RAYOS_KM for p in pos)
    return eco, rayos


def calcular(ahora: datetime | None = None, historial: dict | None = None,
             mov: dict | None = None, contexto=None, campos_verificacion=None) -> dict[str, Any]:
    ahora = ahora or datetime.now(timezone.utc)
    historial = ss.leer(ss.HISTORIAL) if historial is None else historial
    coords = ss.coordenadas()
    mov = movimiento(ahora) if mov is None else mov
    eco, rayos = contexto if contexto is not None else contexto_tormenta(ahora)
    try:
        locs = json.loads(LOCALIDADES.read_text(encoding="utf-8")).get("localidades") or []
    except Exception:
        locs = []
    previo = leer(SALIDA)

    estaciones = []
    for nombre, serie in (historial or {}).items():
        if not serie or nombre not in coords:
            continue
        ultima = serie[-1]
        edad = frescura.edad_min(ultima.get("utc"), ahora)
        if edad is None or not 0.0 <= edad <= EDAD_MAX_MIN:
            continue
        lat, lon = coords[nombre]
        f = firma(serie, ahora)
        registro = {"nombre": nombre, "lat": lat, "lon": lon, "edad_min": round(edad, 1), "firma": None}
        if f is not None:
            f["tormenta_cerca"] = bool(eco(lat, lon) or rayos(lat, lon))
            registro["firma"] = f
        estaciones.append(registro)
    con_firma = [e for e in estaciones if e["firma"] and e["firma"]["tormenta_cerca"]]

    localidades: dict[str, dict] = {}
    for loc in locs:
        cerca = [e for e in estaciones if ss.distancia_km(e["lat"], e["lon"], loc["lat"], loc["lon"]) <= RADIO_LOCALIDAD_KM]
        llego = [e["nombre"] for e in cerca if e["firma"] and e["firma"]["tormenta_cerca"]]
        avisos = []
        if mov and mov["velocidad_kmh"] >= VELOCIDAD_MIN_KMH:
            ang = math.radians(mov["direccion_hacia_grados"])
            ux, uy = math.sin(ang), math.cos(ang)       # hacia dónde va (este, norte)
            for e in con_firma:
                if e["nombre"] in llego:
                    continue
                dx, dy = km_xy(loc["lat"], loc["lon"], e["lat"], e["lon"])
                a_lo_largo = dx * ux + dy * uy
                costado = abs(-dx * uy + dy * ux)
                if a_lo_largo < A_LO_LARGO_MIN_KM or costado > COSTADO_MAX_KM:
                    continue
                desde = parse_utc(e["firma"]["hora_utc"]) or ahora
                minutos = a_lo_largo / mov["velocidad_kmh"] * 60.0 - (ahora - desde).total_seconds() / 60.0
                if minutos > LLEGADA_MAX_MIN:
                    continue
                avisos.append({"estacion": e["nombre"], "minutos": round(max(minutos, 0.0)),
                               "distancia_km": round(a_lo_largo, 1), "costado_km": round(costado, 1),
                               "senales": e["firma"]["senales"]})
        avisos.sort(key=lambda a: a["minutos"])
        estado = "llego" if llego else ("ya_llega" if avisos else None)
        localidades[loc["id"]] = {
            "estado": estado,
            "estaciones_con_firma": llego,
            "minutos": avisos[0]["minutos"] if avisos else (0 if llego else None),
            "hora_estimada_utc": iso(ahora + timedelta(minutes=avisos[0]["minutos"])) if avisos else None,
            "avisos": avisos[:3],
            "estaciones_cerca": len(cerca),
        }

    verif = verificar(previo, ahora, localidades, locs, campos_verificacion)
    for loc_id, r in localidades.items():
        if r["estado"] == "ya_llega":
            verif["pendientes"].append({"ahora_utc": iso(ahora), "localidad": loc_id, "minutos": r["minutos"],
                                        "estacion": r["avisos"][0]["estacion"]})
    # Una alarma por localidad cada 10 min (los refrescos no se duplican).
    vistos, unicos = set(), []
    for p in reversed(verif["pendientes"]):
        t = parse_utc(p.get("ahora_utc")) or ahora
        clave = (p.get("localidad"), t.strftime("%Y%m%d%H") + str(t.minute // 10))
        if clave not in vistos:
            vistos.add(clave)
            unicos.append(p)
    verif["pendientes"] = list(reversed(unicos))[-PENDIENTES_MAX:]
    return {
        "version": VERSION, "en_paralelo": True, "generado_utc": iso(ahora),
        "disponible": bool(estaciones), "motivo": None if estaciones else "SIN_ESTACIONES_FRESCAS",
        "movimiento": mov,
        "estaciones_frescas": len(estaciones),
        "estaciones_con_firma": [{"nombre": e["nombre"], **e["firma"]} for e in estaciones if e["firma"]],
        "localidades": localidades,
        **verif,
        "nota": ("Alarma de 'ya llega' por estaciones aguas arriba (frente de ráfagas). EN PARALELO: no "
                 "cambia lo que se ve hasta validarla en vivo (resumen)."),
    }


def _campos_radar(desde: datetime, hasta: datetime) -> dict[datetime, Any]:
    import nowcast_capas_nucleo as nc
    import nowcast_flujo_optico as nf

    salida = {}
    for p in sorted(nf.HISTORICO.glob("20*.png")):
        t = nf.timestamp_de_nombre(p)
        if t is None or not desde <= t <= hasta:
            continue
        d = nf.cargar_dbz(p)
        if d is not None:
            salida[t] = nc.campo_limpio(d, nc.cargar_clutter_total(d.shape))
    return salida


def verificar(previo: dict, ahora: datetime, localidades: dict, locs: list[dict], campos=None) -> dict:
    """Valida las alarmas cuya ventana (hora estimada ± 20 min) ya pasó."""
    ultimas = list(previo.get("ultimas_verificadas") or [])
    pend, acum = [], dict(previo.get("acumulado") or {"acierto": 0, "falsa_alarma": 0, "errores_min": []})
    candidatas = []
    for p in previo.get("pendientes") or []:
        t = parse_utc(p.get("ahora_utc"))
        if t is None:
            continue
        fin = t + timedelta(minutes=float(p.get("minutos") or 0) + TOLERANCIA_MIN)
        if fin <= ahora:
            candidatas.append((p, fin))
        else:
            pend.append(p)
    vencidas = []
    if candidatas:
        try:
            if campos is None:
                inicio = min(parse_utc(p["ahora_utc"]) for p, _ in candidatas) - timedelta(minutes=10)
                campos = _campos_radar(inicio, ahora)
        except Exception as exc:
            campos = {}
            previo["error_verificacion"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        ultima = max(campos) if campos else None
        for p, fin in candidatas:
            # Se valida recién cuando llegó la imagen del radar que cubre toda la ventana
            # (el radar llega 10-30 min tarde); si pasaron 3 h, se descarta sin contar.
            if ultima is not None and fin <= ultima:
                vencidas.append(p)
            elif ahora - fin < timedelta(hours=3):
                pend.append(p)
    if vencidas:
        try:
            import nowcast_capas_nucleo as nc
            import nowcast_flujo_optico as nf

            zonas, picos = {}, {}
            for t, c in campos.items():
                if not zonas:
                    h, w = c.shape
                    for loc in locs:
                        x, y = nf.latlon_to_pixel(loc["lat"], loc["lon"], w, h)
                        zonas[loc["id"]] = nf.disco(c.shape, x, y, ZONA_VERIFICACION_KM / nf.km_por_pixel(w))
                picos[t] = nc.picos_por_zona(c, zonas)
            llegadas_estacion = previo.get("llegadas_estacion") or {}
            for p in vencidas:
                t = parse_utc(p["ahora_utc"])
                loc = p.get("localidad")
                esperado = t + timedelta(minutes=float(p.get("minutos") or 0))
                # Desde la hora de la alarma (lo que ya estaba encima no cuenta como llegada).
                a = max(t, esperado - timedelta(minutes=TOLERANCIA_MIN))
                b = esperado + timedelta(minutes=TOLERANCIA_MIN)
                horas = [h for h in picos if a <= h <= b and loc in picos[h]]
                if not horas:
                    continue  # faltaron imágenes del radar: no se cuenta
                radar_ok = [h for h in horas if picos[h][loc] >= UMBRAL_ECO_DBZ]
                est_ok = [parse_utc(x) for x in llegadas_estacion.get(loc, []) if parse_utc(x) and a <= parse_utc(x) <= b]
                llegadas = sorted(radar_ok + est_ok)
                ultimas.append({"ahora_utc": p["ahora_utc"], "localidad": loc, "minutos": p.get("minutos"),
                                "estacion": p.get("estacion"), "llego": bool(llegadas),
                                "error_min": round((llegadas[0] - esperado).total_seconds() / 60.0) if llegadas else None})
                if llegadas:
                    acum["acierto"] += 1
                    acum["errores_min"] = (acum.get("errores_min") or [])[-200:] + [
                        round((llegadas[0] - esperado).total_seconds() / 60.0)]
                else:
                    acum["falsa_alarma"] += 1
        except Exception as exc:
            pend.extend(vencidas)
            previo["error_verificacion"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    # Llegadas medidas por estaciones (firma en la localidad), para verificar alarmas viejas.
    llegadas_estacion = previo.get("llegadas_estacion") or {}
    for loc_id, r in localidades.items():
        if r.get("estado") == "llego":
            lista = llegadas_estacion.setdefault(loc_id, [])
            if iso(ahora) not in lista:
                lista.append(iso(ahora))
    limite = ahora - timedelta(hours=3)
    llegadas_estacion = {k: [x for x in v if (parse_utc(x) or limite) > limite] for k, v in llegadas_estacion.items()}
    total = acum["acierto"] + acum["falsa_alarma"]
    errores = sorted(acum.get("errores_min") or [])
    resumen = {"alarmas_verificadas": total,
               "aciertos_pct": round(100 * acum["acierto"] / total, 1) if total else None,
               "falsas_alarmas_pct": round(100 * acum["falsa_alarma"] / total, 1) if total else None,
               "error_mediano_min": errores[len(errores) // 2] if errores else None}
    return {"pendientes": pend, "acumulado": acum, "resumen": resumen, "ultimas_verificadas": ultimas[-30:],
            "llegadas_estacion": {k: v for k, v in llegadas_estacion.items() if v}}


def main() -> int:
    try:
        salida = calcular()
    except Exception as exc:  # nunca corta el ciclo
        previo = leer(SALIDA)
        salida = {"version": VERSION, "en_paralelo": True, "disponible": False,
                  "motivo": f"{type(exc).__name__}: {exc}"[:200], "generado_utc": iso(datetime.now(timezone.utc)),
                  "pendientes": previo.get("pendientes") or [], "acumulado": previo.get("acumulado") or {},
                  "resumen": previo.get("resumen") or {}, "llegadas_estacion": previo.get("llegadas_estacion") or {}}
    escribir(SALIDA, salida)
    locs = salida.get("localidades") or {}
    print("Alarma de llegada por estaciones:", salida.get("disponible"),
          " ".join(f"{k}={v['estado']}" for k, v in locs.items() if v.get("estado")) or "sin alarmas",
          salida.get("motivo") or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
