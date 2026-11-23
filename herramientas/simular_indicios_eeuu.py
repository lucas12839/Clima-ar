#!/usr/bin/env python3
"""Etapa G: los indicios de tormenta severa medidos con tormentas VERIFICADAS de EE. UU.

Por qué EE. UU.: en Argentina no hay una base pública oficial de tormentas severas
(dónde cayó granizo, de qué tamaño, qué ráfagas) y el radar de Córdoba de RELAMPAGO
tiene un contrato que prohíbe usarlo fuera de esa campaña. NOAA publica todo libre:
- Radar: MRMS "MergedReflectivityQCComposite" (máximo de la columna con control de
  calidad, como el ZH MAX de SINARAME), cada 2 min, 1 km (s3://noaa-mrms-pds).
- Verdad: NCEI Storm Events Database (registro OFICIAL verificado por el Servicio
  Meteorológico de EE. UU.): granizo con tamaño, ráfagas (medidas o estimadas) y
  tornados, con hora y lugar.
- Ambiente: análisis del modelo GFS archivados en Open-Meteo (mismas variables que usa
  producción: viento en 10 m, 850, 700 y 500 hPa, CAPE).

Para que se parezca a Bahía: cada caso se arma como un "radar virtual" de 240 km de
radio con la grilla de ClimaAR (512 x 512, 0,939 km por píxel), una imagen cada 10 min,
y el desvío de supercelda se mide hacia la DERECHA (hemisferio norte). Se usa la forma,
el movimiento y la duración; los números de intensidad no se trasladan a Bahía.

Uso (en GitHub, necesita red):
  python herramientas/simular_indicios_eeuu.py casos --anios 2023,2024 --cantidad 60 --salida casos.json
  python herramientas/simular_indicios_eeuu.py caso '<json del caso>' --salida registros.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import indicios_severos as ind  # noqa: E402

KM_PX = 240.0 / 255.5
N = 512
UA = {"User-Agent": "ClimaAR/1.0 (proyecto local Bahia Blanca; validacion)"}
NCEI = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
MRMS = "https://noaa-mrms-pds.s3.amazonaws.com/"
PRODUCTO = "CONUS/MergedReflectivityQCComposite_00.50/"
# Radares virtuales en las llanuras (donde las tormentas se parecen a las de la región
# pampeana: llanura, chorro de bajo nivel, granizo y líneas de viento).
CENTROS = {
    "oklahoma_city": (35.33, -97.28), "tulsa": (36.15, -95.99), "wichita": (37.65, -97.43),
    "dodge_city": (37.76, -99.97), "amarillo": (35.23, -101.71), "lubbock": (33.65, -101.81),
    "dallas": (32.57, -97.30), "midland": (31.94, -102.19), "omaha": (41.32, -96.37),
    "north_platte": (41.12, -100.68), "goodland": (39.37, -101.70), "denver": (39.79, -104.55),
    "sioux_falls": (43.59, -96.73), "des_moines": (41.73, -93.72), "kansas_city": (38.81, -94.26),
    "topeka": (39.07, -95.63), "aberdeen": (45.46, -98.41), "bismarck": (46.77, -100.76),
    "cheyenne": (41.15, -104.81), "san_angelo": (31.37, -100.49),
}
RADIO_KM = 240.0


def get(url: str, timeout: int = 120) -> bytes:
    import urllib.request
    for intento in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception:
            if intento == 2:
                raise
    return b""


def km_entre(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, a)))


# --------------------------------------------------------------------------
# Verdad oficial (NCEI Storm Events)
# --------------------------------------------------------------------------
def a_utc(texto: str, zona: str) -> datetime | None:
    try:
        local = datetime.strptime(texto, "%d-%b-%y %H:%M:%S")
        m = re.search(r"([+-]\d+)", zona or "")
        return (local - timedelta(hours=int(m.group(1)))).replace(tzinfo=timezone.utc) if m else None
    except (ValueError, AttributeError):
        return None


def reportes_anio(anio: int) -> list[dict]:
    """Granizo >= 1 pulgada (25 mm), ráfagas >= 50 nudos (93 km/h) y tornados, con hora UTC."""
    indice = get(NCEI).decode("latin-1")
    nombres = sorted(set(re.findall(rf"StormEvents_details-ftp_v1\.0_d{anio}_c\d+\.csv\.gz", indice)))
    if not nombres:
        return []
    texto = gzip.decompress(get(NCEI + nombres[-1], 300)).decode("latin-1")
    salida = []
    for f in csv.DictReader(io.StringIO(texto)):
        tipo = f["EVENT_TYPE"]
        if tipo not in ("Hail", "Thunderstorm Wind", "Tornado"):
            continue
        try:
            lat, lon = float(f["BEGIN_LAT"]), float(f["BEGIN_LON"])
        except ValueError:
            continue
        mag = float(f["MAGNITUDE"]) if f.get("MAGNITUDE") not in (None, "") else None
        if tipo == "Hail" and (mag is None or mag < 1.0):
            continue
        if tipo == "Thunderstorm Wind" and (mag is None or mag < 50.0):
            continue
        t = a_utc(f["BEGIN_DATE_TIME"], f["CZ_TIMEZONE"])
        if t is None:
            continue
        salida.append({"t": t.isoformat(), "tipo": {"Hail": "granizo", "Thunderstorm Wind": "viento",
                                                    "Tornado": "tornado"}[tipo],
                       "mag": mag, "medido": f.get("MAGNITUDE_TYPE") in ("MG", "MS"),
                       "lat": lat, "lon": lon, "id": f["EVENT_ID"]})
    return salida


def elegir_casos(anios: list[int], cantidad: int) -> list[dict]:
    """Las jornadas (UTC, de 12 a 12) con más reportes alrededor de algún radar virtual.
    Una sola jornada por fecha; se equilibran granizo, viento y tornado."""
    todos = []
    for anio in anios:
        todos += reportes_anio(anio)
    por_dia: dict[str, list[dict]] = {}
    for r in todos:
        t = datetime.fromisoformat(r["t"])
        if t.month not in (4, 5, 6, 7, 8, 9):
            continue
        dia = (t - timedelta(hours=12)).strftime("%Y-%m-%d")
        por_dia.setdefault(dia, []).append(r)
    candidatos = []
    for dia, rs in por_dia.items():
        mejor = None
        for nombre, (la, lo) in CENTROS.items():
            dentro = [r for r in rs if km_entre(la, lo, r["lat"], r["lon"]) <= RADIO_KM - 20]
            if len(dentro) >= 8 and (mejor is None or len(dentro) > len(mejor[1])):
                mejor = (nombre, dentro)
        if mejor:
            tipos = {k: sum(r["tipo"] == k for r in mejor[1]) for k in ("granizo", "viento", "tornado")}
            candidatos.append({"dia": dia, "centro": mejor[0], "lat": CENTROS[mejor[0]][0],
                               "lon": CENTROS[mejor[0]][1], "reportes": len(mejor[1]), "tipos": tipos,
                               "dominante": max(tipos, key=tipos.get)})
    elegidos = []
    for dom in ("granizo", "viento", "tornado"):
        grupo = sorted((c for c in candidatos if c["dominante"] == dom), key=lambda c: -c["reportes"])
        elegidos += grupo[: cantidad // 3]
    return sorted(elegidos, key=lambda c: c["dia"])


# --------------------------------------------------------------------------
# Radar virtual (MRMS -> grilla de ClimaAR)
# --------------------------------------------------------------------------
def grilla_latlon(lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    j = np.arange(N) - (N - 1) / 2.0
    xk, yk = np.meshgrid(j * KM_PX, -j * KM_PX)            # x este, y norte (fila 0 = norte)
    lat = lat0 + yk / 110.57
    lon = lon0 + xk / (111.32 * math.cos(math.radians(lat0)))
    dentro = np.hypot(xk, yk) <= RADIO_KM
    return lat, lon, dentro


def listar_mrms(dia: datetime) -> list[tuple[datetime, str]]:
    salida, token = [], None
    pref = PRODUCTO + dia.strftime("%Y%m%d") + "/"
    while True:
        url = f"{MRMS}?list-type=2&prefix={pref}" + (f"&continuation-token={_q(token)}" if token else "")
        x = get(url).decode()
        for k in re.findall(r"<Key>([^<]+)</Key>", x):
            m = re.search(r"_(\d{8})-(\d{6})\.grib2\.gz$", k)
            if m:
                salida.append((datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc), k))
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", x)
        if not m:
            return salida
        token = m.group(1)


def _q(s: str) -> str:
    import urllib.parse
    return urllib.parse.quote(s, safe="")


def leer_mrms(clave: str, lat: np.ndarray, lon: np.ndarray, dentro: np.ndarray) -> np.ndarray:
    import xarray as xr
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "m.grib2"
        p.write_bytes(gzip.decompress(get(MRMS + clave)))
        ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})
        v = ds[list(ds.data_vars)[0]]
        la, lo = v["latitude"].values, v["longitude"].values
        i = np.clip(np.rint((la[0] - lat) / (la[0] - la[1])).astype(int), 0, la.size - 1)
        jj = np.clip(np.rint(((lon % 360) - lo[0]) / (lo[1] - lo[0])).astype(int), 0, lo.size - 1)
        campo = np.asarray(v.values)[i, jj].astype(np.float32)
        ds.close()
    campo[~np.isfinite(campo) | (campo < 0)] = 0.0
    campo[~dentro] = 0.0
    return campo


def ambiente_gfs(lat0: float, lon0: float, dia: str) -> dict[str, dict]:
    variables = ("temperature_2m,dew_point_2m,surface_pressure,cape,wind_speed_10m,wind_direction_10m,"
                 "wind_speed_850hPa,wind_direction_850hPa,wind_speed_700hPa,wind_direction_700hPa,"
                 "wind_speed_500hPa,wind_direction_500hPa")
    d0 = datetime.fromisoformat(dia)
    url = ("https://historical-forecast-api.open-meteo.com/v1/forecast?"
           f"latitude={lat0}&longitude={lon0}&start_date={d0:%Y-%m-%d}&end_date={d0 + timedelta(days=1):%Y-%m-%d}"
           f"&models=gfs_seamless&hourly={variables}")
    h = json.loads(get(url)).get("hourly") or {}
    salida = {}
    for k, t in enumerate(h.get("time") or []):
        salida[t[:13]] = {n: (h[n][k] if h.get(n) else None) for n in h if n != "time"}
    return salida


# --------------------------------------------------------------------------
# Un caso
# --------------------------------------------------------------------------
def correr_caso(caso: dict, par: dict | None = None) -> dict:
    par = {**ind.PARAMETROS_DEFECTO, "hemisferio": "norte", **(par or {})}
    lat0, lon0 = caso["lat"], caso["lon"]
    dia = datetime.fromisoformat(caso["dia"]).replace(tzinfo=timezone.utc)
    reportes = [r for r in reportes_anio(dia.year)
                if dia + timedelta(hours=12) <= datetime.fromisoformat(r["t"]) < dia + timedelta(hours=36)
                and km_entre(lat0, lon0, r["lat"], r["lon"]) <= RADIO_KM]
    if not reportes:
        return {"caso": caso, "registros": [], "motivo": "sin reportes"}
    ts = sorted(datetime.fromisoformat(r["t"]) for r in reportes)
    ini = max(ts[0] - timedelta(minutes=120), dia + timedelta(hours=12))
    fin = min(ts[-1] + timedelta(minutes=60), ini + timedelta(hours=10))
    archivos = []
    for d in {ini.date(), fin.date()}:
        archivos += listar_mrms(datetime(d.year, d.month, d.day, tzinfo=timezone.utc))
    lat, lon, dentro = grilla_latlon(lat0, lon0)
    amb = ambiente_gfs(lat0, lon0, caso["dia"])
    # Reportes en la grilla (km desde el centro).
    for r in reportes:
        r["x_km"] = (r["lon"] - lon0) * 111.32 * math.cos(math.radians(lat0))
        r["y_km"] = (r["lat"] - lat0) * 110.57
    trs, nid, registros = [], 1, []
    t = ini.replace(minute=ini.minute - ini.minute % 10, second=0, microsecond=0)
    while t <= fin:
        cerca = [a for a in archivos if abs((a[0] - t).total_seconds()) <= 150]
        if not cerca:
            t += timedelta(minutes=10)
            continue
        clave = min(cerca, key=lambda a: abs((a[0] - t).total_seconds()))[1]
        try:
            campo = leer_mrms(clave, lat, lon, dentro)
        except Exception as exc:
            print("  error MRMS", clave[-30:], exc, file=sys.stderr)
            t += timedelta(minutes=10)
            continue
        minutos = (t - ini).total_seconds() / 60.0
        trs, nid = ind.seguir(trs, ind.celdas(campo, KM_PX, par), minutos, 10.0, KM_PX, nid, par)
        a = amb.get(t.strftime("%Y-%m-%dT%H")) or {}
        medio, ciz = ind.ambiente(a)
        for tr in trs:
            if tr.puntos[-1][0] != minutos:
                continue
            r = ind.indicios(tr, N, KM_PX, medio, ciz, par)
            x_km = (tr.ultima.x - (N - 1) / 2.0) * KM_PX
            y_km = ((N - 1) / 2.0 - tr.ultima.y) * KM_PX
            radio = math.sqrt(tr.ultima.area_km2 / math.pi)
            # Verdad: reportes oficiales que tocan la tormenta (su radio + 10 km) entre
            # 10 min antes y 60 min después de esta imagen, siguiendo su movimiento.
            mov = ind.movimiento_px_min(tr) or (0.0, 0.0)
            ve, vn = mov[0] * KM_PX, -mov[1] * KM_PX
            toca = []
            for rep in reportes:
                dt = (datetime.fromisoformat(rep["t"]) - t).total_seconds() / 60.0
                if not -10.0 <= dt <= 60.0:
                    continue
                d = math.hypot(rep["x_km"] - (x_km + ve * dt), rep["y_km"] - (y_km + vn * dt))
                if d <= radio + 10.0:
                    toca.append([rep["tipo"], rep["mag"], rep["medido"], round(dt, 1), round(d, 1)])
            registros.append({**r, "min": minutos, "x_km": round(x_km, 1), "y_km": round(y_km, 1),
                              "cape": a.get("cape"), "t2m": a.get("temperature_2m"), "td2m": a.get("dew_point_2m"),
                              "cizalladura_kmh": round(math.hypot(*ciz), 1) if ciz else None,
                              "reportes": toca})
        t += timedelta(minutes=10)
    return {"caso": caso, "registros": registros, "imagenes": int((fin - ini).total_seconds() // 600)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("casos")
    a.add_argument("--anios", default="2023,2024")
    a.add_argument("--cantidad", type=int, default=60)
    a.add_argument("--salida", required=True)
    b = sub.add_parser("caso")
    b.add_argument("caso")
    b.add_argument("--salida", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "casos":
        casos = elegir_casos([int(x) for x in args.anios.split(",")], args.cantidad)
        Path(args.salida).write_text(json.dumps(casos, ensure_ascii=False), encoding="utf-8")
        print(len(casos), "casos")
        for c in casos:
            print(c)
        return 0
    caso = json.loads(args.caso)
    res = correr_caso(caso)
    Path(args.salida).write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(caso["dia"], caso["centro"], len(res["registros"]), "registros")
    return 0


if __name__ == "__main__":
    sys.exit(main())
