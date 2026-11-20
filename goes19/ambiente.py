"""Punto 8: ambiente MEDIDO por GOES-19 antes de que se nuble.

Productos (disco completo cada 10 min, solo cielo despejado, píxel de 10 km),
comprobados el 20/11/2026 con herramientas/sondear_ambiente_goes.py:

- ABI-L2-DSIF  inestabilidad: CAPE (J/kg), LI, KI, TT, SI (K). ~3 min de atraso.
- ABI-L2-TPWF  agua precipitable de la columna (mm). ~3 min de atraso.
- ABI-L2-LVTPF perfil de temperatura (K) en 101 niveles de presión. ~13 min.
- ABI-L2-LVMPF perfil de humedad relativa (%) en 101 niveles. ~13 min.

Solo se usan píxeles con DQF_Overall == 0 (calidad buena: los nublados vienen
marcados y sin dato). Se lee solo la ventana de los 240 km (lectura parcial).

Del perfil se calcula lo que DSIF no trae: temperatura y humedad en 850, 700 y
500 hPa, punto de rocío en 850, gradiente térmico 700-500 hPa (granizo),
nivel de 0 °C (granizo que llega al suelo) y aire seco en capas medias
(ráfagas descendentes). Todo por física, sin entrenamiento.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from . import geospatial as geo

PRODUCTOS = {
    "inestabilidad": ("ABI-L2-DSIF", ("CAPE", "LI", "KI", "TT", "SI")),
    "agua_precipitable": ("ABI-L2-TPWF", ("TPW",)),
    "perfil_temperatura": ("ABI-L2-LVTPF", ("LVT",)),
    "perfil_humedad": ("ABI-L2-LVMPF", ("LVM",)),
}
CALIDAD_BUENA = 0
NIVELES_HPA = (850.0, 700.0, 500.0)
P_MAX_PERFIL = 1000.0   # debajo de 1000 hPa el perfil está bajo el suelo o extrapolado
P_MIN_PERFIL = 100.0
RD = 287.04             # J/(kg K)
G = 9.80665
MIN_PIXELES = 3         # menos que esto no representa a la localidad


# ------------------------------------------------------------------
# Lectura de la ventana
# ------------------------------------------------------------------

def leer_ventana(origen: str | Path, variables: tuple[str, ...], radio_km: float = geo.RADAR_RANGE_KM) -> dict:
    """Píxeles a <= radio_km del radar. Las variables 2D quedan (n,), las
    de perfil (n, niveles) con la presión en `presion_hpa`."""
    with geo.abrir_goes(origen) as ds:
        faltan = [v for v in variables if v not in ds]
        if faltan:
            raise ValueError(f"La variable '{faltan[0]}' no existe en el NetCDF.")
        lam, req, rpol, h = geo.get_projection(ds)
        xs, ys = geo.select_spatial_window(ds, radio_km=radio_km)
        xv = np.asarray(ds["x"].values[xs], dtype=np.float64)
        yv = np.asarray(ds["y"].values[ys], dtype=np.float64)
        crudos = {}
        for v in variables:
            var = ds[v]
            datos = np.asarray(var.isel(y=ys, x=xs).values, dtype=np.float64)
            if datos.ndim == 3:
                orden = [var.dims.index("y"), var.dims.index("x"), var.dims.index("pressure")]
                datos = np.transpose(datos, orden)
            crudos[v] = datos
        dqf = np.asarray(ds["DQF_Overall"].isel(y=ys, x=xs).values, dtype=np.float64)
        presion = np.asarray(ds["pressure"].values, dtype=np.float64) if "pressure" in ds else None
    xg, yg = np.meshgrid(xv, yv)
    lat, lon = geo.fixed_grid_to_latlon(xg, yg, lambda_0=lam, r_eq=req, r_pol=rpol, h=h)
    dist = geo.haversine_km(lat, lon, geo.RADAR_LAT, geo.RADAR_LON)
    mascara = np.isfinite(dist) & (dist <= radio_km)
    return {
        "datos": {v: d[mascara] for v, d in crudos.items()},
        "dqf": dqf[mascara],
        "latitude": lat[mascara],
        "longitude": lon[mascara],
        "distance_km": dist[mascara],
        "presion_hpa": presion,
    }


def buenos(ventana: dict) -> np.ndarray:
    """Píxeles con calidad buena (cielo despejado y medición válida)."""
    ok = ventana["dqf"] == CALIDAD_BUENA
    for d in ventana["datos"].values():
        ok &= np.isfinite(d) if d.ndim == 1 else np.isfinite(d).any(axis=1)
    return ok


# ------------------------------------------------------------------
# Perfil vertical
# ------------------------------------------------------------------

def _interp_logp(presion: np.ndarray, valores: np.ndarray, p: float) -> float | None:
    ok = np.isfinite(valores) & np.isfinite(presion) & (presion <= P_MAX_PERFIL) & (presion >= P_MIN_PERFIL)
    if ok.sum() < 2:
        return None
    lp, v = np.log(presion[ok]), valores[ok]
    orden = np.argsort(lp)
    lp, v = lp[orden], v[orden]
    x = math.log(p)
    if x < lp[0] or x > lp[-1]:
        return None
    return float(np.interp(x, lp, v))


def rocio_c(t_c: float | None, hr: float | None) -> float | None:
    """Punto de rocío (Magnus)."""
    if t_c is None or hr is None or hr <= 0:
        return None
    a, b = 17.62, 243.12
    g = math.log(min(hr, 100.0) / 100.0) + a * t_c / (b + t_c)
    return b * g / (a - g)


def espesor_km(p_abajo: float, p_arriba: float, t_media_k: float) -> float:
    """Distancia vertical entre dos niveles de presión (ecuación hipsométrica)."""
    return RD * t_media_k / G * math.log(p_abajo / p_arriba) / 1000.0


def nivel_cero(presion: np.ndarray, t_k: np.ndarray) -> dict | None:
    """Primer cruce de 0 °C subiendo desde 1000 hPa: presión y altura aproximada
    sobre el nivel de 1000 hPa (Bahía está casi al nivel del mar)."""
    ok = np.isfinite(t_k) & (presion <= P_MAX_PERFIL) & (presion >= P_MIN_PERFIL)
    p, t = presion[ok], t_k[ok]
    if p.size < 2:
        return None
    orden = np.argsort(-p)          # de abajo hacia arriba
    p, t = p[orden], t[orden]
    if t[0] <= 273.15:
        return {"presion_hpa": round(float(p[0]), 1), "altura_m": 0}
    altura = 0.0
    for i in range(1, p.size):
        tm = 0.5 * (t[i] + t[i - 1])
        dz = espesor_km(p[i - 1], p[i], tm) * 1000.0
        if t[i] <= 273.15 < t[i - 1]:
            f = (t[i - 1] - 273.15) / (t[i - 1] - t[i])
            pc = math.exp(math.log(p[i - 1]) + f * (math.log(p[i]) - math.log(p[i - 1])))
            return {"presion_hpa": round(pc, 1), "altura_m": int(round(altura + f * dz, -1))}
        altura += dz
    return None


def derivados_perfil(presion: np.ndarray, t_k: np.ndarray | None, hr: np.ndarray | None) -> dict[str, Any]:
    """Valores de una columna (o de la mediana de varias)."""
    out: dict[str, Any] = {}
    if t_k is not None:
        for p in NIVELES_HPA:
            v = _interp_logp(presion, t_k, p)
            out[f"temperatura_{int(p)}_c"] = round(v - 273.15, 1) if v is not None else None
        t7, t5 = _interp_logp(presion, t_k, 700.0), _interp_logp(presion, t_k, 500.0)
        if t7 is not None and t5 is not None:
            dz = espesor_km(700.0, 500.0, 0.5 * (t7 + t5))
            out["gradiente_700_500_c_km"] = round((t7 - t5) / dz, 2)
        out["nivel_0c"] = nivel_cero(presion, t_k)
    if hr is not None:
        for p in NIVELES_HPA:
            v = _interp_logp(presion, hr, p)
            out[f"humedad_{int(p)}_pct"] = round(min(max(v, 0.0), 100.0), 0) if v is not None else None
        capas = [_interp_logp(presion, hr, p) for p in (700.0, 600.0, 500.0)]
        capas = [c for c in capas if c is not None]
        out["humedad_media_700_500_pct"] = round(float(np.mean(capas)), 0) if capas else None
    if t_k is not None and hr is not None:
        t8 = out.get("temperatura_850_c")
        out["rocio_850_c"] = (round(rocio_c(t8, out.get("humedad_850_pct")), 1)
                              if t8 is not None and out.get("humedad_850_pct") else None)
    return out


# ------------------------------------------------------------------
# Resúmenes por zona
# ------------------------------------------------------------------

def _dist_km(lat, lon, lat0, lon0):
    return geo.haversine_km(np.asarray(lat), np.asarray(lon), lat0, lon0)


def resumen_zona(ventanas: dict[str, dict], seleccion: dict[str, np.ndarray]) -> dict[str, Any]:
    """Mediana de los píxeles buenos de la zona, producto por producto.
    `seleccion`: {producto: máscara booleana sobre su ventana}."""
    out: dict[str, Any] = {"pixeles_despejados": {}}
    inest = ventanas.get("inestabilidad")
    if inest is not None:
        m = seleccion["inestabilidad"] & buenos(inest)
        out["pixeles_despejados"]["inestabilidad"] = int(m.sum())
        if m.sum() >= MIN_PIXELES:
            d = inest["datos"]
            out["cape_jkg"] = round(float(np.median(d["CAPE"][m])), 0)
            out["cape_max_jkg"] = round(float(np.max(d["CAPE"][m])), 0)
            out["indice_elevacion_c"] = round(float(np.median(d["LI"][m])), 1)
            out["indice_k_c"] = round(float(np.median(d["KI"][m])), 1)
            out["total_totals_c"] = round(float(np.median(d["TT"][m])), 1)
            out["showalter_c"] = round(float(np.median(d["SI"][m])), 1)
    agua = ventanas.get("agua_precipitable")
    if agua is not None:
        m = seleccion["agua_precipitable"] & buenos(agua)
        out["pixeles_despejados"]["agua_precipitable"] = int(m.sum())
        if m.sum() >= MIN_PIXELES:
            out["agua_precipitable_mm"] = round(float(np.median(agua["datos"]["TPW"][m])), 1)
    tv, hv = ventanas.get("perfil_temperatura"), ventanas.get("perfil_humedad")
    if tv is not None or hv is not None:
        mt = seleccion["perfil_temperatura"] & buenos(tv) if tv is not None else None
        mh = seleccion["perfil_humedad"] & buenos(hv) if hv is not None else None
        n = int(max(mt.sum() if mt is not None else 0, mh.sum() if mh is not None else 0))
        out["pixeles_despejados"]["perfiles"] = n
        t_med = (np.nanmedian(tv["datos"]["LVT"][mt], axis=0)
                 if mt is not None and mt.sum() >= MIN_PIXELES else None)
        h_med = (np.nanmedian(hv["datos"]["LVM"][mh], axis=0)
                 if mh is not None and mh.sum() >= MIN_PIXELES else None)
        presion = (tv or hv)["presion_hpa"]
        if presion is not None and (t_med is not None or h_med is not None):
            out["perfil"] = derivados_perfil(presion, t_med, h_med)
    return out


def calificar(r: dict[str, Any]) -> dict[str, str]:
    """Lectura física del ambiente medido (reglas iniciales, EN EVALUACIÓN:
    se ajustan con la medición de aciertos del punto 9)."""
    out: dict[str, str] = {}
    cape, li = r.get("cape_jkg"), r.get("indice_elevacion_c")
    if cape is not None and li is not None:
        if cape < 100 and li > 2:
            out["inestabilidad"] = "estable"
        elif cape < 500:
            out["inestabilidad"] = "debil"
        elif cape < 1500:
            out["inestabilidad"] = "moderada"
        else:
            out["inestabilidad"] = "fuerte"
    tpw = r.get("agua_precipitable_mm")
    if tpw is not None:
        out["agua_en_la_columna"] = "muy_alta" if tpw >= 35 else ("alta" if tpw >= 25 else "normal")
    perfil = r.get("perfil") or {}
    grad = perfil.get("gradiente_700_500_c_km")
    if grad is not None:
        out["capas_medias"] = "muy_inestables" if grad >= 7.5 else ("inestables" if grad >= 6.5 else "normales")
    hr = perfil.get("humedad_media_700_500_pct")
    rocio = perfil.get("rocio_850_c")
    if hr is not None and rocio is not None:
        out["aire_seco_en_altura"] = "si" if (hr < 40 and rocio >= 8) else "no"
    cero = (perfil.get("nivel_0c") or {}).get("altura_m")
    if cero is not None:
        out["nivel_0c"] = "bajo" if cero < 3000 else ("medio" if cero < 4000 else "alto")
    return out


def por_localidades(ventanas: dict[str, dict], localidades: list[dict], radio_km: float) -> dict[str, dict]:
    salida = {}
    for loc in localidades:
        sel = {k: _dist_km(v["latitude"], v["longitude"], loc["lat"], loc["lon"]) <= radio_km
               for k, v in ventanas.items()}
        r = resumen_zona(ventanas, sel)
        r["lectura"] = calificar(r)
        salida[loc["id"]] = r
    return salida


def region(ventanas: dict[str, dict]) -> dict[str, Any]:
    """Toda la zona de 240 km: mediana, extremo y dónde está el aire más inestable."""
    sel = {k: np.ones(v["dqf"].shape, dtype=bool) for k, v in ventanas.items()}
    r = resumen_zona(ventanas, sel)
    inest = ventanas.get("inestabilidad")
    if inest is not None:
        m = buenos(inest)
        n = int(inest["dqf"].size)
        r["fraccion_despejada"] = round(float(m.sum()) / n, 2) if n else None
        if m.sum() >= MIN_PIXELES:
            cape = np.where(m, inest["datos"]["CAPE"], -np.inf)
            i = int(np.argmax(cape))
            r["mas_inestable"] = {
                "cape_jkg": round(float(cape[i]), 0),
                "latitud": round(float(inest["latitude"][i]), 3),
                "longitud": round(float(inest["longitude"][i]), 3),
                "distancia_radar_km": round(float(inest["distance_km"][i]), 0),
            }
            r["cape_p90_jkg"] = round(float(np.percentile(inest["datos"]["CAPE"][m], 90)), 0)
            r["indice_elevacion_p10_c"] = round(float(np.percentile(inest["datos"]["LI"][m], 10)), 1)
    agua = ventanas.get("agua_precipitable")
    if agua is not None:
        m = buenos(agua)
        if m.sum() >= MIN_PIXELES:
            r["agua_precipitable_p90_mm"] = round(float(np.percentile(agua["datos"]["TPW"][m], 90)), 1)
    r["lectura"] = calificar(r)
    return r
