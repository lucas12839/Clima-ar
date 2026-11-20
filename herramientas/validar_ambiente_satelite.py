#!/usr/bin/env python3
"""Punto 8: ¿el ambiente MEDIDO por el satélite anticipa los días con tormenta?
¿Mejor o peor que el modelo?

Para cada día con radar guardado: el satélite a las 15 y 18 UTC (12 y 15 hora
de Argentina, antes de la convección de la tarde) sobre los 240 km y sobre
Bahía Blanca; el ECMWF y el GFS de ese día (Open-Meteo, pronóstico histórico)
para Bahía a la misma hora; y lo que pasó después en el radar entre las 15 UTC
y las 06 UTC del día siguiente:
- tormenta en la región: alguna imagen con >= 300 píxeles (~260 km²) de 45 dBZ
  o más (con menos, los ecos falsos de cada día alcanzan: todos los días tienen
  algún píxel de 45 dBZ cerca del radar);
- tormenta en Bahía: alguna imagen con >= 15 píxeles de 45 dBZ a <= 30 km.

Compara los días con y sin tormenta y el acierto de reglas simples con cada
fuente. Uso (necesita internet: correr en GitHub):
  python herramientas/validar_ambiente_satelite.py [--desde 20261025] [--salida x.json]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import requests  # noqa: E402

import nowcast_flujo_optico as nf  # noqa: E402
from goes19 import ambiente as amb  # noqa: E402
from goes19 import geospatial as geo  # noqa: E402
from goes19 import goes19_engine as ge  # noqa: E402

BAHIA = {"id": "bahia_blanca", "lat": -38.7183, "lon": -62.2663}
UMBRAL_DBZ = 45.0
MIN_PIXELES_TORMENTA = 300
MIN_PIXELES_BAHIA = 15
RADIO_BAHIA_KM = 30.0
HORAS_SAT = (15, 18)


def radar_dia(dia: datetime, clutter_cache: dict) -> dict:
    """Lo que vio el radar entre las 15 UTC del día y las 06 UTC del siguiente."""
    ini, fin = dia.replace(hour=15), dia.replace(hour=15) + timedelta(hours=15)
    frames = [p for p in sorted(nf.HISTORICO.glob("20*.png"))
              if (t := nf.timestamp_de_nombre(p)) and ini <= t <= fin]
    max_px, max_bahia, n = 0, 0, 0
    primera = None
    for p in frames:
        d = nf.cargar_dbz(p)
        if d is None:
            continue
        if "m" not in clutter_cache:
            clutter_cache["m"] = nf.cargar_mascara_clutter(d.shape)
            km = nf.km_por_pixel(d.shape[1])
            cy = cx = (d.shape[0] - 1) / 2.0
            dx = (BAHIA["lon"] - geo.RADAR_LON) * 111.32 * math.cos(math.radians(BAHIA["lat"]))
            dy = (BAHIA["lat"] - geo.RADAR_LAT) * 111.195
            clutter_cache["disco"] = nf.disco(d.shape, cx + dx / km, cy - dy / km, RADIO_BAHIA_KM / km)
        c = nf.campo_limpio(d, clutter_cache["m"])
        n += 1
        px = int((c >= UMBRAL_DBZ).sum())
        if px >= MIN_PIXELES_TORMENTA and primera is None:
            primera = nf.timestamp_de_nombre(p)
        max_px = max(max_px, px)
        max_bahia = max(max_bahia, int((c[clutter_cache["disco"]] >= UMBRAL_DBZ).sum()))
    return {
        "imagenes": n,
        "max_pixeles_45dbz": max_px,
        "tormenta_region": max_px >= MIN_PIXELES_TORMENTA,
        "max_pixeles_45dbz_bahia": max_bahia,
        "tormenta_bahia": max_bahia >= MIN_PIXELES_BAHIA,
        "primera_tormenta_utc": nf.iso(primera),
    }


def satelite(hora: datetime) -> dict:
    """Ambiente medido en el barrido que empieza a esa hora."""
    ventanas, info = {}, {}
    for nombre, (prefijo, variables) in amb.PRODUCTOS.items():
        clave = f"val_{nombre}"
        ge.PRODUCT_SPECS[clave] = (prefijo, None)
        try:
            obj = ge.latest_product_object(clave, now=hora + timedelta(minutes=11), hours_back=1)
            if obj is None:
                info[nombre] = "sin archivo"
                continue
            ventanas[nombre] = amb.leer_ventana(obj["url"], variables)
            info[nombre] = obj["start_time_utc"]
        except Exception as exc:
            info[nombre] = f"error {type(exc).__name__}"
    if not ventanas:
        return {"archivos": info}
    reg = amb.region(ventanas)
    loc = amb.por_localidades(ventanas, [BAHIA], 25.0)["bahia_blanca"]
    return {"archivos": info, "region": reg, "bahia": loc}


def modelos(desde: datetime, hasta: datetime) -> dict:
    salida = {}
    for modelo in ("ecmwf_ifs025", "gfs_seamless"):
        try:
            r = requests.get(
                "https://historical-forecast-api.open-meteo.com/v1/forecast",
                params={"latitude": BAHIA["lat"], "longitude": BAHIA["lon"],
                        "start_date": f"{desde:%Y-%m-%d}", "end_date": f"{hasta:%Y-%m-%d}",
                        "hourly": "cape,total_column_integrated_water_vapour,lifted_index",
                        "models": modelo, "timezone": "UTC"},
                timeout=60)
            r.raise_for_status()
            h = r.json()["hourly"]
            salida[modelo] = {t: {"cape": c, "agua": w, "li": li} for t, c, w, li in
                              zip(h["time"], h.get("cape", []), h.get("total_column_integrated_water_vapour", []),
                                  h.get("lifted_index", []) or [None] * len(h["time"]))}
        except Exception as exc:
            salida[modelo] = {"error": str(exc)[:200]}
    return salida


def _med(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 1) if vals else None


def acierto(dias: list[dict], valor, umbral: float, mayor: bool, evento: str) -> dict | None:
    """Regla 'valor >= umbral' (o <=) para anticipar el evento."""
    a = b = c = d = 0
    for x in dias:
        v = valor(x)
        if v is None:
            continue
        pred = v >= umbral if mayor else v <= umbral
        obs = x["radar"][evento]
        a += pred and obs
        b += pred and not obs
        c += (not pred) and obs
        d += (not pred) and (not obs)
    n = a + b + c + d
    if n == 0:
        return None
    return {"umbral": umbral, "dias": n, "detectados": a, "falsas_alarmas": b, "omitidos": c,
            "correctos_sin": d, "deteccion_pct": round(100 * a / (a + c), 0) if a + c else None,
            "falsas_alarmas_pct": round(100 * b / (a + b), 0) if a + b else None,
            "csi": round(a / (a + b + c), 2) if a + b + c else None}


def mejor_umbral(dias, valor, umbrales, mayor, evento):
    res = [r for u in umbrales if (r := acierto(dias, valor, u, mayor, evento))]
    res = [r for r in res if r["csi"] is not None]
    return max(res, key=lambda r: (r["csi"], -r["falsas_alarmas"])) if res else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default="20261025")
    ap.add_argument("--hasta", default=None)
    ap.add_argument("--salida", default=None)
    a = ap.parse_args(argv)
    desde = datetime.strptime(a.desde, "%Y%m%d").replace(tzinfo=timezone.utc)
    ultimo = max(nf.timestamp_de_nombre(p) for p in nf.HISTORICO.glob("20*.png"))
    hasta = (datetime.strptime(a.hasta, "%Y%m%d").replace(tzinfo=timezone.utc) if a.hasta
             else (ultimo - timedelta(hours=15)).replace(hour=0, minute=0, second=0, microsecond=0))
    mods = modelos(desde, hasta + timedelta(days=1))
    cache: dict = {}
    dias = []
    d = desde
    while d <= hasta:
        t0 = time.time()
        x = {"dia": f"{d:%Y-%m-%d}", "radar": radar_dia(d, cache), "satelite": {}, "modelo": {}}
        if x["radar"]["imagenes"] < 40:
            print(f"{x['dia']}: pocas imágenes de radar ({x['radar']['imagenes']}), se saltea")
            d += timedelta(days=1)
            continue
        for h in HORAS_SAT:
            x["satelite"][str(h)] = satelite(d.replace(hour=h))
            for m, serie in mods.items():
                if "error" not in serie:
                    x["modelo"].setdefault(m, {})[str(h)] = serie.get(f"{d:%Y-%m-%d}T{h:02d}:00")
        dias.append(x)
        s15 = x["satelite"]["15"]
        print(f"{x['dia']}: tormenta región={x['radar']['tormenta_region']} bahía={x['radar']['tormenta_bahia']} "
              f"| sat 15 UTC CAPE p90={(s15.get('region') or {}).get('cape_p90_jkg')} "
              f"LI p10={(s15.get('region') or {}).get('indice_elevacion_p10_c')} "
              f"agua p90={(s15.get('region') or {}).get('agua_precipitable_p90_mm')} "
              f"despejado={(s15.get('region') or {}).get('fraccion_despejada')} "
              f"| ECMWF CAPE={((x['modelo'].get('ecmwf_ifs025') or {}).get('15') or {}).get('cape')} "
              f"({time.time() - t0:.0f} s)")
        d += timedelta(days=1)

    # Resumen ---------------------------------------------------------------
    def sat(h, campo, zona="region"):
        return lambda x: ((x["satelite"].get(h) or {}).get(zona) or {}).get(campo)

    def sat_perfil(h, campo):
        return lambda x: (((x["satelite"].get(h) or {}).get("region") or {}).get("perfil") or {}).get(campo)

    def mod(m, h, campo):
        return lambda x: ((x["modelo"].get(m) or {}).get(h) or {}).get(campo)

    variables = {
        "satélite CAPE p90 región 15 UTC": (sat("15", "cape_p90_jkg"), range(0, 2001, 100), True),
        "satélite CAPE p90 región 18 UTC": (sat("18", "cape_p90_jkg"), range(0, 2001, 100), True),
        "satélite índice de elevación p10 región 15 UTC": (sat("15", "indice_elevacion_p10_c"),
                                                            [x / 2 for x in range(-12, 9)], False),
        "satélite agua precipitable p90 región 15 UTC": (sat("15", "agua_precipitable_p90_mm"), range(10, 45, 2), True),
        "satélite gradiente 700-500 región 15 UTC": (sat_perfil("15", "gradiente_700_500_c_km"),
                                                      [x / 4 for x in range(20, 36)], True),
        "ECMWF CAPE Bahía 15 UTC": (mod("ecmwf_ifs025", "15", "cape"), range(0, 3001, 100), True),
        "ECMWF CAPE Bahía 18 UTC": (mod("ecmwf_ifs025", "18", "cape"), range(0, 3001, 100), True),
        "ECMWF agua Bahía 15 UTC": (mod("ecmwf_ifs025", "15", "agua"), range(10, 45, 2), True),
        "GFS CAPE Bahía 15 UTC": (mod("gfs_seamless", "15", "cape"), range(0, 3001, 100), True),
        "GFS índice de elevación Bahía 15 UTC": (mod("gfs_seamless", "15", "li"), [x / 2 for x in range(-12, 9)], False),
    }
    resumen = {"dias": len(dias),
               "dias_tormenta_region": sum(x["radar"]["tormenta_region"] for x in dias),
               "dias_tormenta_bahia": sum(x["radar"]["tormenta_bahia"] for x in dias),
               "despejado_15utc_mediana": _med([sat("15", "fraccion_despejada")(x) for x in dias]),
               "variables": {}}
    for nombre, (f, umbrales, mayor) in variables.items():
        con = [f(x) for x in dias if x["radar"]["tormenta_region"]]
        sin = [f(x) for x in dias if not x["radar"]["tormenta_region"]]
        resumen["variables"][nombre] = {
            "dias_con_dato": sum(v is not None for v in con + sin),
            "mediana_dias_con_tormenta": _med(con),
            "mediana_dias_sin_tormenta": _med(sin),
            "mejor_regla_region": mejor_umbral(dias, f, umbrales, mayor, "tormenta_region"),
        }
    print("\n===== RESUMEN")
    print(json.dumps(resumen, ensure_ascii=False, indent=1))
    if a.salida:
        Path(a.salida).write_text(json.dumps({"resumen": resumen, "dias": dias}, ensure_ascii=False,
                                             indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
