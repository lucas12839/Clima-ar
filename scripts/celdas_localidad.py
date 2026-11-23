#!/usr/bin/env python3
"""Paso 2 (en paralelo, no cambia lo que se ve): ¿pasa una tormenta SOBRE cada
localidad en los próximos 30 min? Celda por celda, corregida con los rayos.

Cada núcleo de tormenta (>= 35 dBZ) de la última imagen del radar se mueve con
el movimiento del 5b hasta ahora; si hay rayos de los últimos minutos donde
debería estar, el lugar de ahora sale de los rayos (el radar llega con 20-30
min de atraso; los rayos con ~1 min). Las celdas de rayos sin núcleo en el
radar (nacieron después) entran como celdas nuevas. Después se prueban muchos
caminos posibles y el porcentaje es la fracción que pasa sobre la localidad.

Escribe data/ia/celdas_localidad.json y lo verifica solo con las imágenes de
radar que llegan después (validación en vivo, regla 6). Nunca corta el ciclo.
"""
from __future__ import annotations

import json
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

import celdas_rayos as cr  # noqa: E402
import corrector_tiempo_real as ct  # noqa: E402
import localidades_radar as lr  # noqa: E402
import nowcast_capas as capas  # noqa: E402
import nowcast_capas_nucleo as nc  # noqa: E402
import nowcast_flujo_optico as nf  # noqa: E402

# 1.2 (23/11/2026): la verdad de la validación en vivo sale de cada imagen limpia con el
# satélite, como el pronóstico. Antes se volvía a leer el PNG sin esa limpieza y los ecos
# de terreno de la noche (Tornquist, Médanos: eco sin ninguna nube encima) contaban como
# tormentas que pasaron: 136 "pasó sin aviso" falsos. Al cambiar de versión se reinicia
# lo acumulado.
VERSION = "celdas-localidad-1.2"
OBSERVADO_HORAS = 3
SALIDA = ROOT / "data" / "ia" / "celdas_localidad.json"
GOES = ROOT / "data" / "goes19" / "goes19_state.json"
HORIZONTE_MIN = 30.0
UMBRAL_TORMENTA = 35.0
# Grados (elegidos con herramientas/simular_celdas_rayos.py; ver docs/VALIDACION_CELDAS_RAYOS.md).
UMBRAL_PROBABLE = 0.5
UMBRAL_POSIBLE = 0.3
# Criterios que se validan en vivo (703 casos simulados, tormenta sobre la localidad en 30 min):
# - "combinado": el 5b la trae a la localidad en <= 30 min Y celdas >= 0,5 -> 62 % detectadas,
#   34 % falsas alarmas (el 5b solo: 73 % / 46 %).
# - "rayos": celda con rayos >= 0,7 -> pocas (14 %) pero muy seguras (8,5 % falsas alarmas).
CRITERIOS = {
    "combinado": lambda r: bool(r.get("b5")) and r.get("p", 0.0) >= UMBRAL_PROBABLE,
    "rayos": lambda r: r.get("pr", 0.0) >= 0.7,
    "celdas_probable": lambda r: r.get("p", 0.0) >= UMBRAL_PROBABLE,
    "celdas_posible": lambda r: r.get("p", 0.0) >= UMBRAL_POSIBLE,
    "5b_localidad": lambda r: bool(r.get("b5")),
}
PARAMETROS: dict[str, Any] = {}
# Idea 2 (20/11/2026): aviso de la localidad EXACTA solo para tormentas fuertes (>= 45 dBZ)
# o con rayos; lo demás lo cubre la zona de 10 km. Verdad propia: 45 dBZ o más dentro de
# la localidad o rayos dentro de ella (radio + 4 km, error de ubicación del GLM).
# Umbral elegido con herramientas/simular_aviso_fuerte.py (docs/VALIDACION_AVISO_FUERTE.md).
UMBRAL_FUERTE_DBZ = 45.0
UMBRAL_AVISO_FUERTE = 0.7
MARGEN_RAYO_KM = 4.0
PARAMETROS_FUERTE: dict[str, Any] = {"solo_fuertes_dbz": UMBRAL_FUERTE_DBZ}
PENDIENTES_MAX = 200
EDAD_RADAR_MAX_MIN = 90.0


def parse_utc(texto: Any) -> datetime | None:
    if not texto:
        return None
    try:
        v = datetime.fromisoformat(str(texto).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return (v if v.tzinfo else v.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def iso(v: datetime | None) -> str | None:
    return v.isoformat().replace("+00:00", "Z") if v else None


def leer(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def rayos_ahora(goes: dict, ahora: datetime, shape) -> tuple[list[dict], dict[str, Any]]:
    """Celdas de rayos de los últimos 10 min (con la edad de cada rayo respecto de ahora)."""
    ventana = ((goes.get("glm") or {}).get("ventana")) or {}
    fin = parse_utc(ventana.get("fin_ventana_utc"))
    info: dict[str, Any] = {"fin_ventana_utc": ventana.get("fin_ventana_utc")}
    if not ventana.get("disponible") or fin is None:
        info["estado"] = "sin_rayos"
        return [], info
    atraso = max(0.0, (ahora - fin).total_seconds() / 60.0)
    pos = [p for p in ventana.get("rayos_10min") or [] if len(p) >= 3]
    info.update({"estado": "ok", "rayos": len(pos), "edad_ventana_min": round(atraso, 1)})
    if not pos:
        return [], info
    h, w = shape
    arr = np.asarray(pos, dtype=float)
    x, y = ct.a_pixel(arr[:, 0], arr[:, 1], w, h)
    lista = cr.celdas(x, y, arr[:, 2] + atraso, shape, nf.km_por_pixel(w))
    info["celdas"] = len(lista)
    return lista, info


def grado(prob: float) -> str | None:
    return "probable" if prob >= UMBRAL_PROBABLE else ("posible" if prob >= UMBRAL_POSIBLE else None)


def verificar(previo: dict, campos_por_hora: dict[datetime, np.ndarray], zonas: dict, ultima: datetime) -> dict:
    """Valida los pronósticos guardados cuyo plazo (30 min) ya cubren las imágenes
    de radar llegadas: ¿hubo tormenta sobre la localidad entre ahora y +30?
    El aviso de la idea 2 se valida con su propia verdad: tormenta fuerte o rayos."""
    pend = []
    observado = previo.get("observado") or {}
    acum = previo.get("acumulado") or {}
    acum_fuerte = previo.get("acumulado_fuerte") or [0, 0, 0, 0]
    rayos_loc = previo.get("rayos_en_localidad") or {}
    for p in previo.get("pendientes") or []:
        desde = parse_utc(p.get("ahora_utc"))
        if desde is None:
            continue
        hasta = desde + timedelta(minutes=HORIZONTE_MIN)
        if hasta > ultima:
            pend.append(p)
            continue
        horas = [t for t in campos_por_hora if desde - timedelta(minutes=5) <= t <= hasta]
        if len(horas) < 2:
            continue  # faltaron imágenes: no se cuenta

        def pico(t, loc):
            # Lo observado en esa imagen cuando fue la actual (limpia con el satélite);
            # si no se guardó, la imagen guardada sin esa limpieza.
            guardado = (observado.get(iso(t)) or {}).get(loc)
            if guardado is not None:
                return guardado
            return nc.picos_por_zona(campos_por_hora[t], {loc: zonas[loc]})[loc]

        for loc, reg in (p.get("localidades") or {}).items():
            if loc not in zonas:
                continue
            hubo = any(pico(t, loc) >= UMBRAL_TORMENTA for t in horas)
            for nombre, criterio in CRITERIOS.items():
                a = acum.setdefault(nombre, [0, 0, 0, 0])
                dijo = criterio(reg)
                a[0 if dijo and hubo else 1 if dijo else 2 if hubo else 3] += 1
            if "pf" in reg:
                fuerte = any(pico(t, loc) >= UMBRAL_FUERTE_DBZ for t in horas)
                electrica = any(desde <= (parse_utc(x) or desde - timedelta(days=1)) <= hasta
                                for x in rayos_loc.get(loc, []))
                hubo_f = fuerte or electrica
                dijo = reg["pf"] >= UMBRAL_AVISO_FUERTE
                acum_fuerte[0 if dijo and hubo_f else 1 if dijo else 2 if hubo_f else 3] += 1
    previo["pendientes"] = pend[-PENDIENTES_MAX:]
    previo["acumulado"] = acum
    previo["acumulado_fuerte"] = acum_fuerte
    resumen = {}
    for nombre, (a, b, c, d) in acum.items():
        resumen[nombre] = {"paso_y_aviso": a, "aviso_y_no_paso": b, "paso_sin_aviso": c,
                           "deteccion_pct": round(100 * a / (a + c), 1) if a + c else None,
                           "falsas_alarmas_pct": round(100 * b / (a + b), 1) if a + b else None}
    a, b, c, _ = acum_fuerte
    resumen["aviso_fuerte"] = {"paso_y_aviso": a, "aviso_y_no_paso": b, "paso_sin_aviso": c,
                               "deteccion_pct": round(100 * a / (a + c), 1) if a + c else None,
                               "falsas_alarmas_pct": round(100 * b / (a + b), 1) if a + b else None,
                               "verdad": "45 dBZ o más dentro de la localidad, o rayos dentro de ella"}
    previo["resumen"] = resumen
    return previo


def registrar_rayos(previo: dict, goes: dict, locs: list[dict], ahora: datetime) -> dict[str, list[str]]:
    """Guarda cuándo cayeron rayos dentro de cada localidad (verdad de la idea 2).

    Cada ventana de rayos de 10 min se anota con su hora de fin; se guardan 3 h."""
    import math

    reg = {k: list(v) for k, v in (previo.get("rayos_en_localidad") or {}).items()}
    ventana = ((goes.get("glm") or {}).get("ventana")) or {}
    fin = parse_utc(ventana.get("fin_ventana_utc"))
    if ventana.get("disponible") and fin is not None and ahora - fin <= timedelta(minutes=15):
        pos = [p for p in ventana.get("rayos_10min") or [] if len(p) >= 2]
        for loc in locs:
            radio = float(loc.get("radio_km", 4.0)) + MARGEN_RAYO_KM
            cos0 = math.cos(math.radians(loc["lat"]))
            cerca = any(math.hypot((p[1] - loc["lon"]) * 111.32 * cos0, (p[0] - loc["lat"]) * 110.57) <= radio
                        for p in pos)
            if cerca:
                # La ventana cubre los 10 min anteriores: se anota el comienzo y el fin.
                for t in (fin - timedelta(minutes=10), fin):
                    if iso(t) not in reg.setdefault(loc["id"], []):
                        reg[loc["id"]].append(iso(t))
    limite = ahora - timedelta(hours=3)
    reg = {k: sorted(x for x in v if (parse_utc(x) or limite) > limite) for k, v in reg.items()}
    return {k: v for k, v in reg.items() if v}


def campos_recientes(ultima: datetime, horas: int = 2) -> dict[datetime, np.ndarray]:
    salida = {}
    for p in sorted(nf.HISTORICO.glob("20*.png")):
        t = nf.timestamp_de_nombre(p)
        if t is None or t < ultima - timedelta(hours=horas) or t > ultima:
            continue
        d = nf.cargar_dbz(p)
        if d is not None:
            salida[t] = nc.campo_limpio(d, nc.cargar_clutter_total(d.shape))
    return salida


def calcular(ahora: datetime | None = None) -> dict[str, Any]:
    ahora = ahora or datetime.now(timezone.utc)
    radar_ts = nf.TIMESTAMP_FILE.read_text(encoding="utf-8").strip() if nf.TIMESTAMP_FILE.exists() else None
    radar_dt = parse_utc(radar_ts)
    previo = leer(SALIDA)
    base = {"version": VERSION, "en_paralelo": True, "generado_utc": iso(ahora), "radar_timestamp": radar_ts,
            "horizonte_min": HORIZONTE_MIN, "umbrales": {"probable": UMBRAL_PROBABLE, "posible": UMBRAL_POSIBLE,
                                                         "aviso_fuerte": UMBRAL_AVISO_FUERTE,
                                                         "fuerte_dbz": UMBRAL_FUERTE_DBZ}}
    mismo = previo.get("version") == VERSION
    verif = {"pendientes": previo.get("pendientes") or [],
             "acumulado": (previo.get("acumulado") or {}) if mismo else {},
             "acumulado_fuerte": (previo.get("acumulado_fuerte") or [0, 0, 0, 0]) if mismo else [0, 0, 0, 0],
             "acumulado_desde_utc": previo.get("acumulado_desde_utc") if mismo else iso(ahora),
             "observado": previo.get("observado") or {},
             "rayos_en_localidad": previo.get("rayos_en_localidad") or {}}
    actual = nf.cargar_dbz(nf.ACTUAL) if (radar_dt and nf.ACTUAL.exists()) else None
    if actual is None:
        return {**base, "disponible": False, "motivo": "SIN_RADAR_ACTUAL", **verif}
    edad = (ahora - radar_dt).total_seconds() / 60.0
    shape = actual.shape
    h, w = shape
    km_px = nf.km_por_pixel(w)
    cfg = {**capas.CONFIG_DEFECTO, **capas.leer_json(capas.CONFIG)}
    clutter = nc.cargar_clutter_total(shape)
    goes = leer(GOES)
    datos = capas.datos_corrector(goes, ahora, shape)
    campo = nc.campo_limpio(actual, clutter)
    if datos.get("satelite") is not None:
        campo = ct.quitar_ecos_sin_nube(campo, datos["satelite"])
    zonas = {}
    locs_px = {}
    lista_locs = lr.cargar_localidades()
    try:
        verif["rayos_en_localidad"] = registrar_rayos(verif, goes, lista_locs, ahora)
    except Exception as exc:
        verif["error_rayos_localidad"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    for loc in lista_locs:
        x, y = nf.latlon_to_pixel(loc["lat"], loc["lon"], w, h)
        r = float(loc.get("radio_km", 4.0)) / km_px
        locs_px[loc["id"]] = (x, y, r)
        zonas[loc["id"]] = nf.disco(shape, x, y, r)
    # Lo observado en esta imagen (limpia con el satélite), para validar después.
    obs_ahora = nc.picos_por_zona(campo, zonas)
    verif["observado"][radar_ts and iso(radar_dt)] = {k: round(float(x), 1) for k, x in obs_ahora.items()}
    limite_obs = radar_dt - timedelta(hours=OBSERVADO_HORAS)
    verif["observado"] = {t: v for t, v in verif["observado"].items() if (parse_utc(t) or limite_obs) > limite_obs}
    # Validación en vivo con las imágenes que ya llegaron.
    try:
        verif = verificar(verif, campos_recientes(radar_dt), zonas, radar_dt)
    except Exception as exc:
        verif["error_verificacion"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    if edad > EDAD_RADAR_MAX_MIN:
        return {**base, "disponible": False, "motivo": "RADAR_DEMASIADO_VIEJO", "edad_radar_min": round(edad, 1),
                **verif}
    flujo = capas.movimiento(radar_dt, campo, actual, clutter, cfg)
    u, v = (flujo["u"], flujo["v"]) if flujo else (None, None)
    rayos, info_rayos = rayos_ahora(goes, ahora, shape)
    prob = cr.probabilidades_celdas(campo, u, v, km_px, edad, rayos, locs_px, PARAMETROS)
    prob_rayos = cr.probabilidades(rayos, u, v, km_px, locs_px, PARAMETROS)
    prob_fuerte = cr.probabilidades_celdas(campo, u, v, km_px, edad, rayos, locs_px,
                                           {**PARAMETROS, **PARAMETROS_FUERTE})
    capas_loc = (leer(capas.SALIDA).get("localidades") or {}) \
        if leer(capas.SALIDA).get("radar_timestamp") == radar_ts else {}
    localidades = {}
    for k, r in prob.items():
        b5 = ((((capas_loc.get(k) or {}).get("llegada_proxima_hora") or {}).get("tormenta") or {})
              .get("probable") or {}).get("localidad")
        reg = {"p": r["probabilidad"], "pr": prob_rayos[k]["probabilidad"],
               "b5": b5 is not None and b5 <= HORIZONTE_MIN, "pf": prob_fuerte[k]["probabilidad"]}
        pf = prob_fuerte[k]
        localidades[k] = {**r, "grado": grado(r["probabilidad"]),
                          "probabilidad_solo_rayos": prob_rayos[k]["probabilidad"],
                          "minutos_solo_rayos": prob_rayos[k].get("minutos"),
                          "5b_la_trae_en_30min": reg["b5"],
                          "aviso_combinado": CRITERIOS["combinado"](reg),
                          "aviso_rayos": CRITERIOS["rayos"](reg),
                          # Idea 2: tormenta fuerte o con rayos SOBRE la localidad (en paralelo).
                          "fuerte_o_rayos": {
                              "probabilidad": pf["probabilidad"],
                              "aviso": pf["probabilidad"] >= UMBRAL_AVISO_FUERTE,
                              "minutos": pf["minutos"] if pf["probabilidad"] >= UMBRAL_AVISO_FUERTE else None,
                              "hora_estimada_utc": iso(ahora + timedelta(minutes=pf["minutos"]))
                              if pf["minutos"] is not None and pf["probabilidad"] >= UMBRAL_AVISO_FUERTE else None,
                          },
                          "hora_estimada_utc": iso(ahora + timedelta(minutes=r["minutos"]))
                          if r["minutos"] is not None and grado(r["probabilidad"]) else None,
                          "_reg": reg}
    verif.setdefault("pendientes", []).append(
        {"ahora_utc": iso(ahora), "radar_utc": radar_ts,
         "localidades": {k: v.pop("_reg") for k, v in localidades.items()}})
    # Un pronóstico por imagen de radar y por cada 10 min (los refrescos no se duplican).
    vistos, unicos = set(), []
    for p in reversed(verif["pendientes"]):
        clave = (p.get("radar_utc"), (parse_utc(p.get("ahora_utc")) or ahora).strftime("%Y%m%d%H") +
                 str((parse_utc(p.get("ahora_utc")) or ahora).minute // 10))
        if clave in vistos:
            continue
        vistos.add(clave)
        unicos.append(p)
    verif["pendientes"] = list(reversed(unicos))[-PENDIENTES_MAX:]
    return {
        **base, "disponible": True, "motivo": None,
        "edad_radar_min": round(edad, 1),
        "movimiento": None if flujo is None else {k: flujo[k] for k in ("velocidad_kmh", "direccion_hacia_grados")},
        "rayos": info_rayos,
        "localidades": localidades,
        **verif,
        "nota": ("Probabilidad de que una tormenta (>= 35 dBZ) pase SOBRE la localidad en los próximos "
                 "30 min, celda por celda y corregida con los rayos. EN PARALELO: no cambia lo que se ve "
                 "hasta validarlo en vivo (resumen)."),
    }


def main() -> int:
    try:
        salida = calcular()
    except Exception as exc:  # nunca corta el ciclo
        previo = leer(SALIDA)
        salida = {"version": VERSION, "en_paralelo": True, "disponible": False,
                  "motivo": f"{type(exc).__name__}: {exc}"[:200], "generado_utc": iso(datetime.now(timezone.utc)),
                  "pendientes": previo.get("pendientes") or [], "acumulado": previo.get("acumulado") or {},
                  "acumulado_fuerte": previo.get("acumulado_fuerte") or [0, 0, 0, 0],
                  "rayos_en_localidad": previo.get("rayos_en_localidad") or {},
                  "resumen": previo.get("resumen") or {}}
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    tmp = SALIDA.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(salida, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, SALIDA)
    locs = salida.get("localidades") or {}
    print(f"Celdas por localidad: disponible={salida.get('disponible')} " +
          " ".join(f"{k}={v['probabilidad']}" for k, v in locs.items()) + f" {salida.get('motivo') or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
