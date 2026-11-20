#!/usr/bin/env python3
"""Nowcast por flujo óptico para el RMA10 (sin entrenamiento).

Método (física/extrapolación, mismo principio que pySTEPS):
1. Decodifica a dBZ el frame actual y uno anterior (5–30 min antes).
2. Quita ecos fijos (clutter) con una máscara de persistencia histórica.
3. Estima el campo de movimiento con flujo óptico Farnebäck (OpenCV).
4. Advecta el campo de reflectividad hacia adelante (semi-lagrangiano)
   a 10, 20, 30 y 60 min.
5. Mide qué llega a Bahía Blanca: dBZ máximo en un radio de 10 km y la
   primera hora en que un eco ≥35 dBZ entra en ese radio.

La confianza baja con el atraso del radar (15–40 min), con un hueco
largo entre frames, con movimiento desordenado y a más de 150 km del
radar (el haz pasa alto y subestima la reflectividad).

Uso:
  python scripts/nowcast_flujo_optico.py                 # ciclo operativo
  python scripts/nowcast_flujo_optico.py --generar-mascara  # desde el histórico
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.rma10_features import (  # noqa: E402
    BAHIA_LAT,
    BAHIA_LON,
    RADAR_RADIUS_KM,
    latlon_to_pixel,
    parse_utc,
)

RADAR_DIR = ROOT / "data" / "radar"
ACTUAL = RADAR_DIR / "actual.png"
TIMESTAMP_FILE = RADAR_DIR / "radar_timestamp.txt"
HISTORICO = RADAR_DIR / "historico"
SALIDA = RADAR_DIR / "nowcast_flujo.json"
# Campo de movimiento para el resto del ciclo (no se versiona: .gitignore).
CAMPO = RADAR_DIR / ".flujo_campo.npz"
HISTORIAL_CIUDAD = RADAR_DIR / "historial_ciudad.json"
HISTORIAL_HORAS = 8
LLUVIA_INTENSA_DBZ = 40.0
LLUVIA_PERSISTENTE_DBZ = 25.0
MASCARA_CLUTTER = ROOT / "config" / "clutter_rma10.png"

VERSION = "flujo-optico-1.0"
LEADS_MIN = (10, 20, 30, 60)
PASO_ETA_MIN = 5
UMBRAL_ECO_DBZ = 20.0
UMBRAL_TORMENTA_DBZ = 35.0
UMBRAL_FUERTE_DBZ = 45.0
RADIO_CIUDAD_KM = 10.0
RADIO_CERCANIA_KM = 30.0
MIN_PIXELES_CIUDAD = 6  # ~5 km²: evita que un píxel suelto cuente como tormenta
GAP_MIN_MIN = 5.0
GAP_MAX_MIN = 30.0
VELOCIDAD_MAX_KMH = 120.0
VELOCIDAD_SALTO_KMH = 15.0        # diferencia que dispara el control con varias imágenes
VELOCIDAD_SALTO_FRACCION = 0.5
ALCANCE_CONFIABLE_KM = 150.0
EDAD_NORMAL_MIN = 15.0
EDAD_MAX_MIN = 40.0
CLUTTER_FRACCION = 0.3


# ------------------------------------------------------------
# utilidades
# ------------------------------------------------------------

def iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if dt else None


def km_por_pixel(width: int) -> float:
    return RADAR_RADIUS_KM / max((width - 1) / 2.0, 1.0)


def timestamp_de_nombre(path: Path) -> datetime | None:
    stem = path.stem
    if len(stem) == 14 and stem.isdigit() and stem.startswith("20"):
        try:
            return datetime.strptime(stem, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def cargar_dbz(path: Path) -> np.ndarray | None:
    import procesador_radar

    with contextlib.redirect_stdout(io.StringIO()):
        return procesador_radar.cargar_dbz_rma10(path)


def escribir_json_atomico(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ------------------------------------------------------------
# máscara de clutter
# ------------------------------------------------------------

def cargar_mascara_clutter(shape: tuple[int, int]) -> np.ndarray:
    if not MASCARA_CLUTTER.exists():
        return np.zeros(shape, dtype=bool)
    from PIL import Image

    with Image.open(MASCARA_CLUTTER) as img:
        mask = np.asarray(img.convert("L")) > 127
    if mask.shape != shape:
        return np.zeros(shape, dtype=bool)
    return mask


CLUTTER_FUERTE_DBZ = 45.0
CLUTTER_FUERTE_FRACCION = 0.05


def generar_mascara_clutter(frames: list[Path], fraccion: float = CLUTTER_FRACCION) -> np.ndarray | None:
    """Clutter = píxeles con eco ≥20 dBZ en más de `fraccion` de los frames, o
    con ≥45 dBZ en más del 5 % (una tormenta real no repite tanto el mismo
    píxel; el clutter intermitente de las sierras sí)."""
    acumulado = None
    fuerte = None
    n = 0
    for path in frames:
        dbz = cargar_dbz(path)
        if dbz is None:
            continue
        valores = np.nan_to_num(dbz, nan=-99.0)
        eco = valores >= UMBRAL_ECO_DBZ
        alto = valores >= CLUTTER_FUERTE_DBZ
        acumulado = eco.astype(np.int32) if acumulado is None else acumulado + eco
        fuerte = alto.astype(np.int32) if fuerte is None else fuerte + alto
        n += 1
    if acumulado is None or n < 20:
        return None
    return ((acumulado / n) > fraccion) | ((fuerte / n) > CLUTTER_FUERTE_FRACCION)


# ------------------------------------------------------------
# núcleo físico
# ------------------------------------------------------------

def campo_limpio(dbz: np.ndarray, clutter: np.ndarray) -> np.ndarray:
    campo = np.nan_to_num(dbz.astype(np.float32), nan=0.0)
    campo[campo < UMBRAL_ECO_DBZ - 5.0] = 0.0
    campo[clutter] = 0.0
    return campo


def a_uint8(campo: np.ndarray) -> np.ndarray:
    return np.clip(campo / 70.0 * 255.0, 0, 255).astype(np.uint8)


def estimar_flujo(previo: np.ndarray, actual: np.ndarray, gap_min: float, km_px: float) -> dict[str, Any]:
    """Flujo óptico en px/min, suavizado y acotado a una velocidad física."""
    import cv2

    flow = cv2.calcOpticalFlowFarneback(
        a_uint8(previo), a_uint8(actual), None,
        pyr_scale=0.5, levels=4, winsize=31, iterations=5,
        poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    u = flow[..., 0] / gap_min
    v = flow[..., 1] / gap_min

    eco = (actual >= UMBRAL_ECO_DBZ) | (previo >= UMBRAL_ECO_DBZ)
    n_eco = int(eco.sum())
    if n_eco < 30:
        return {"valido": False, "motivo": "SIN_ECOS_SUFICIENTES", "pixeles_eco": n_eco}

    u_med = float(np.median(u[eco]))
    v_med = float(np.median(v[eco]))

    # Fuera de los ecos el flujo no está definido: se usa el vector medio.
    peso = cv2.GaussianBlur(eco.astype(np.float32), (0, 0), 8)
    u_s = cv2.GaussianBlur(np.where(eco, u, 0).astype(np.float32), (0, 0), 8)
    v_s = cv2.GaussianBlur(np.where(eco, v, 0).astype(np.float32), (0, 0), 8)
    w = np.clip(peso * 3.0, 0.0, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        u_f = np.where(peso > 1e-3, u_s / np.maximum(peso, 1e-3), u_med)
        v_f = np.where(peso > 1e-3, v_s / np.maximum(peso, 1e-3), v_med)
    u_f = w * u_f + (1 - w) * u_med
    v_f = w * v_f + (1 - w) * v_med

    vmax = VELOCIDAD_MAX_KMH / 60.0 / km_px
    rapidez = np.hypot(u_f, v_f)
    escala = np.where(rapidez > vmax, vmax / np.maximum(rapidez, 1e-6), 1.0)
    u_f *= escala
    v_f *= escala

    # Coherencia: fracción de ecos que se mueven a menos de 45° del medio.
    rap_eco = np.hypot(u[eco], v[eco])
    norma = math.hypot(u_med, v_med)
    if norma > 1e-6:
        coseno = (u[eco] * u_med + v[eco] * v_med) / np.maximum(rap_eco * norma, 1e-6)
        coherencia = float(np.mean(coseno > math.cos(math.radians(45))))
    else:
        coherencia = 0.0

    velocidad_kmh = norma * km_px * 60.0
    # dirección HACIA donde va (0 = norte, 90 = este); y crece hacia el sur.
    direccion = (math.degrees(math.atan2(u_med, -v_med)) + 360.0) % 360.0

    return {
        "valido": True,
        "u": u_f.astype(np.float32),
        "v": v_f.astype(np.float32),
        "pixeles_eco": n_eco,
        "velocidad_media_kmh": round(min(velocidad_kmh, VELOCIDAD_MAX_KMH), 1),
        "direccion_hacia_grados": round(direccion, 1),
        "coherencia": round(coherencia, 3),
    }


def advectar(campo: np.ndarray, u: np.ndarray, v: np.ndarray, minutos: float, pasos_por_10min: int = 4) -> np.ndarray:
    """Semi-lagrangiano hacia atrás: F(x, t) = F0(x - V·t), en sub-pasos."""
    import cv2

    h, w = campo.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    n = max(1, int(math.ceil(minutos / 10.0 * pasos_por_10min)))
    dt = minutos / n
    x = gx.copy()
    y = gy.copy()
    for _ in range(n):
        uu = cv2.remap(u, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        vv = cv2.remap(v, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        x = x - uu * dt
        y = y - vv * dt
    # Vecino más cercano: conserva los picos de reflectividad (bilineal los lava).
    return cv2.remap(campo, x, y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def disco(shape: tuple[int, int], cx: float, cy: float, radio_px: float) -> np.ndarray:
    h, w = shape
    gy, gx = np.ogrid[:h, :w]
    return (gx - cx) ** 2 + (gy - cy) ** 2 <= radio_px ** 2


def factor_alcance(distancia_km: float | None) -> float:
    """1.0 hasta 150 km; baja linealmente a 0.5 en 240 km."""
    if distancia_km is None or not math.isfinite(distancia_km):
        return 1.0
    if distancia_km <= ALCANCE_CONFIABLE_KM:
        return 1.0
    frac = (distancia_km - ALCANCE_CONFIABLE_KM) / (RADAR_RADIUS_KM - ALCANCE_CONFIABLE_KM)
    return round(max(0.5, 1.0 - 0.5 * min(frac, 1.0)), 3)


def factor_edad(edad_min: float | None) -> float:
    """1.0 hasta 15 min; baja a 0.4 en 40 min; 0 después."""
    if edad_min is None:
        return 1.0
    if edad_min <= EDAD_NORMAL_MIN:
        return 1.0
    if edad_min > EDAD_MAX_MIN:
        return 0.0
    return round(1.0 - 0.6 * (edad_min - EDAD_NORMAL_MIN) / (EDAD_MAX_MIN - EDAD_NORMAL_MIN), 3)


def controlar_velocidad(flujo: dict[str, Any], previas: list[float] | None) -> dict[str, Any]:
    """Control con varias imágenes: si la velocidad de este par de imágenes se
    aparta mucho de la de los pares anteriores, se usa la mediana (misma
    dirección). Una sola imagen con la hora corrida (SINARAME atrasado) no
    puede triplicar la velocidad de todas las tormentas (caso 19/11 23:10:
    58 km/h en un par, 21-24 km/h en los vecinos)."""
    previas = [v for v in (previas or []) if v is not None and v > 0]
    actual = flujo["velocidad_media_kmh"]
    control = {"velocidades_pares_previos_kmh": [round(v, 1) for v in previas], "corregida": False}
    if len(previas) >= 2 and actual > 0:
        robusta = float(np.median(previas + [actual]))
        if abs(actual - robusta) > max(VELOCIDAD_SALTO_KMH, VELOCIDAD_SALTO_FRACCION * robusta):
            factor = robusta / actual
            flujo["u"] = (flujo["u"] * factor).astype(np.float32)
            flujo["v"] = (flujo["v"] * factor).astype(np.float32)
            control.update({"corregida": True, "velocidad_de_este_par_kmh": actual,
                            "velocidad_usada_kmh": round(robusta, 1)})
            flujo["velocidad_media_kmh"] = round(robusta, 1)
    flujo["control_velocidad"] = control
    return flujo


def velocidad_par(previo_dbz: np.ndarray, actual_dbz: np.ndarray, gap_min: float,
                  clutter: np.ndarray) -> float | None:
    km_px = km_por_pixel(actual_dbz.shape[1])
    f = estimar_flujo(campo_limpio(previo_dbz, clutter), campo_limpio(actual_dbz, clutter), gap_min, km_px)
    return f["velocidad_media_kmh"] if f["valido"] else None


def velocidades_previas(desde_dt: datetime, desde_dbz: np.ndarray, clutter: np.ndarray,
                        pares: int = 2) -> list[float]:
    """Velocidad media de los pares de imágenes anteriores (para el control)."""
    salida = []
    actual_dt, actual_dbz = desde_dt, desde_dbz
    for _ in range(pares):
        previo = frame_previo(actual_dt)
        if previo is None:
            break
        dbz = cargar_dbz(previo[0])
        if dbz is None or dbz.shape != actual_dbz.shape:
            break
        gap = (actual_dt - previo[1]).total_seconds() / 60.0
        v = velocidad_par(dbz, actual_dbz, gap, clutter)
        if v is not None:
            salida.append(v)
        actual_dt, actual_dbz = previo[1], dbz
    return salida


def pronosticar(previo_dbz: np.ndarray, actual_dbz: np.ndarray, gap_min: float, clutter: np.ndarray,
                leads=LEADS_MIN, velocidades_previas_kmh: list[float] | None = None) -> dict[str, Any]:
    """Núcleo sin E/S: devuelve campos pronosticados y diagnóstico."""
    h, w = actual_dbz.shape
    km_px = km_por_pixel(w)
    previo = campo_limpio(previo_dbz, clutter)
    actual = campo_limpio(actual_dbz, clutter)
    flujo = estimar_flujo(previo, actual, gap_min, km_px)
    if not flujo["valido"]:
        return {"flujo": flujo, "campos": {}, "actual": actual}
    flujo = controlar_velocidad(flujo, velocidades_previas_kmh)
    campos = {int(t): advectar(actual, flujo["u"], flujo["v"], float(t)) for t in leads}
    return {"flujo": flujo, "campos": campos, "actual": actual}


def impacto_ciudad(actual: np.ndarray, flujo: dict[str, Any], max_min: int = 60) -> dict[str, Any]:
    h, w = actual.shape
    km_px = km_por_pixel(w)
    cx, cy = latlon_to_pixel(BAHIA_LAT, BAHIA_LON, w, h)
    zona = disco((h, w), cx, cy, RADIO_CIUDAD_KM / km_px)

    def max_en_zona(campo):
        """dBZ que cubre al menos MIN_PIXELES_CIUDAD píxeles de la zona."""
        vals = campo[zona]
        if vals.size < MIN_PIXELES_CIUDAD:
            return 0.0
        return float(np.partition(vals, -MIN_PIXELES_CIUDAD)[-MIN_PIXELES_CIUDAD])

    serie = []
    eta = None
    eta_fuerte = None
    dbz_eta = None
    for t in range(0, max_min + 1, PASO_ETA_MIN):
        campo = actual if t == 0 else advectar(actual, flujo["u"], flujo["v"], float(t))
        m = max_en_zona(campo)
        serie.append({"minutos": t, "dbz_max": round(m, 1)})
        if eta is None and m >= UMBRAL_TORMENTA_DBZ:
            eta = t
            dbz_eta = m
        if eta_fuerte is None and m >= UMBRAL_FUERTE_DBZ:
            eta_fuerte = t

    # Distancia al radar del eco que llegaría (punto de origen de la trayectoria).
    origen_km = None
    if eta is not None:
        x, y = cx, cy
        for _ in range(max(1, eta // 2)):
            ix = int(np.clip(round(x), 0, w - 1))
            iy = int(np.clip(round(y), 0, h - 1))
            x -= float(flujo["u"][iy, ix]) * 2.0
            y -= float(flujo["v"][iy, ix]) * 2.0
        origen_km = math.hypot(x - (w - 1) / 2.0, y - (h - 1) / 2.0) * km_px

    return {
        "radio_km": RADIO_CIUDAD_KM,
        "dbz_max_ahora": serie[0]["dbz_max"],
        "serie": serie,
        "eta_tormenta_min": eta,
        "dbz_al_llegar": round(dbz_eta, 1) if dbz_eta is not None else None,
        "eta_tormenta_fuerte_min": eta_fuerte,
        "distancia_radar_origen_km": round(origen_km, 1) if origen_km is not None else None,
    }


def leer_campo(radar_ts: str | None, path: Path | None = None):
    """(u, v, coherencia) en px/min si el campo es del mismo frame; si no, None."""
    path = path or CAMPO
    if not path.exists() or radar_ts is None:
        return None
    try:
        with np.load(path) as datos:
            if parse_utc(str(datos["radar_timestamp"])) != parse_utc(radar_ts):
                return None
            return datos["u"], datos["v"], float(datos["coherencia"])
    except Exception:
        return None


def velocidad_en(campo, x: float, y: float, radio_px: int = 6):
    """Vector medio (px/min) del flujo alrededor de (x, y)."""
    u, v, _ = campo
    h, w = u.shape
    x0, x1 = max(0, int(x) - radio_px), min(w, int(x) + radio_px + 1)
    y0, y1 = max(0, int(y) - radio_px), min(h, int(y) + radio_px + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    return float(u[y0:y1, x0:x1].mean()), float(v[y0:y1, x0:x1].mean())


def observar(campo: np.ndarray) -> dict[str, Any]:
    """Lo que ve el radar ahora: sobre la ciudad y en toda la región.

    Se exige un área mínima (MIN_PIXELES_CIUDAD) para que un píxel suelto no
    cuente. dBZ en la región = el núcleo más intenso de al menos esa área.
    """
    import cv2

    h, w = campo.shape
    km_px = km_por_pixel(w)
    cx, cy = latlon_to_pixel(BAHIA_LAT, BAHIA_LON, w, h)
    zona = disco((h, w), cx, cy, RADIO_CIUDAD_KM / km_px)

    def percentil_area(valores):
        if valores.size < MIN_PIXELES_CIUDAD:
            return 0.0
        return float(np.partition(valores, -MIN_PIXELES_CIUDAD)[-MIN_PIXELES_CIUDAD])

    dbz_ciudad = percentil_area(campo[zona])
    # Región: máximo que se mantiene en una ventana de 3x3 (descarta píxeles sueltos).
    erosion = cv2.erode(campo.astype(np.float32), np.ones((3, 3), np.uint8))
    dbz_region = float(erosion.max()) if erosion.size else 0.0
    distancia = None
    if dbz_region > 0:
        ys, xs = np.where(erosion >= dbz_region - 0.01)
        d = np.hypot(xs - cx, ys - cy) * km_px
        distancia = float(d.min())
    cerca = disco((h, w), cx, cy, RADIO_CERCANIA_KM / km_px)
    dbz_cerca = percentil_area(campo[cerca])
    return {
        "dbz_ciudad": round(dbz_ciudad, 1),
        "dbz_30km": round(dbz_cerca, 1),
        "dbz_region": round(dbz_region, 1),
        "distancia_nucleo_region_km": round(distancia, 1) if distancia is not None else None,
        "cobertura_eco_pct": round(float((campo >= UMBRAL_ECO_DBZ).mean() * 100.0), 2),
    }


def actualizar_historial_ciudad(dt: datetime, dbz_ciudad: float, path: Path | None = None) -> dict[str, float]:
    """Guarda dBZ sobre la ciudad por frame y devuelve las rachas actuales de
    lluvia intensa (>=40 dBZ) y persistente (>=25 dBZ), en minutos."""
    path = path or HISTORIAL_CIUDAD
    try:
        datos = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        datos = []
    registros = {r["utc"]: r["dbz"] for r in datos if isinstance(r, dict) and "utc" in r}
    registros[iso(dt)] = round(float(dbz_ciudad), 1)
    limite = dt - timedelta(hours=HISTORIAL_HORAS)
    ordenados = sorted(
        (parse_utc(k), v) for k, v in registros.items() if parse_utc(k) is not None
    )
    ordenados = [(t, v) for t, v in ordenados if limite <= t <= dt]
    try:
        escribir_json_atomico_lista(path, [{"utc": iso(t), "dbz": v} for t, v in ordenados])
    except Exception as exc:
        print(f"AVISO: no se pudo guardar el historial de la ciudad: {exc}")

    return {
        "minutos_lluvia_intensa": racha(ordenados, LLUVIA_INTENSA_DBZ),
        "minutos_lluvia_persistente": racha(ordenados, LLUVIA_PERSISTENTE_DBZ),
        "lluvia_radar_3h_mm": acumulado_mm(ordenados, dt, 3),
        "lluvia_radar_6h_mm": acumulado_mm(ordenados, dt, 6),
    }


def intensidad_mm_h(dbz: float) -> float:
    """Marshall-Palmer (Z = 200 R^1.6). Subestima la lluvia tropical eficiente."""
    if dbz < 15.0:
        return 0.0
    return (10 ** (dbz / 10.0) / 200.0) ** (1 / 1.6)


def acumulado_mm(ordenados, fin: datetime, horas: float) -> float:
    """Lluvia estimada por radar sobre la ciudad en las últimas `horas`.
    Cada frame representa hasta 20 min (los huecos más largos no suman)."""
    total = 0.0
    anterior = None
    for t, v in ordenados:
        if anterior is None:
            minutos = 10.0
        else:
            minutos = min((t - anterior).total_seconds() / 60.0, 20.0)
        anterior = t
        if fin - timedelta(hours=horas) < t <= fin:
            total += intensidad_mm_h(v) * minutos / 60.0
    return round(total, 1)


def racha(ordenados, umbral: float) -> float:
    """Minutos seguidos hasta el último frame con dBZ >= umbral (huecos <= 40 min, como los atrasos reales)."""
    minutos = 0.0
    fin = None
    previo = None
    for t, v in reversed(ordenados):
        if previo is not None and (previo - t).total_seconds() / 60.0 > 40.0:
            break
        if v < umbral:
            break
        fin = fin or t
        minutos = (fin - t).total_seconds() / 60.0 + 10.0
        previo = t
    return round(minutos, 1)


def escribir_json_atomico_lista(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ------------------------------------------------------------
# selección de frames
# ------------------------------------------------------------

def frame_previo(actual_dt: datetime, historico: Path = HISTORICO) -> tuple[Path, datetime] | None:
    """Frame anterior más cercano a 10 min, dentro de 5–30 min."""
    mejor = None
    for path in historico.glob("20*.png"):
        dt = timestamp_de_nombre(path)
        if dt is None:
            continue
        gap = (actual_dt - dt).total_seconds() / 60.0
        if GAP_MIN_MIN <= gap <= GAP_MAX_MIN:
            puntaje = abs(gap - 10.0)
            if mejor is None or puntaje < mejor[0]:
                mejor = (puntaje, path, dt)
    return (mejor[1], mejor[2]) if mejor else None


# ------------------------------------------------------------
# ciclo operativo
# ------------------------------------------------------------

def salida_no_disponible(radar_ts: str | None, motivo: str, **extra) -> dict[str, Any]:
    return {
        "version": VERSION,
        "disponible": False,
        "motivo": motivo,
        "radar_timestamp": radar_ts,
        "generado_utc": iso(datetime.now(timezone.utc)),
        **extra,
    }


def ejecutar(ahora: datetime | None = None) -> dict[str, Any]:
    ahora = ahora or datetime.now(timezone.utc)
    if not TIMESTAMP_FILE.exists() or not ACTUAL.exists():
        return salida_no_disponible(None, "SIN_RADAR_ACTUAL")

    radar_ts = TIMESTAMP_FILE.read_text(encoding="utf-8").strip()
    actual_dt = parse_utc(radar_ts)
    if actual_dt is None:
        return salida_no_disponible(radar_ts, "TIMESTAMP_INVALIDO")

    edad = max(0.0, (ahora - actual_dt).total_seconds() / 60.0)
    f_edad = factor_edad(edad)
    if f_edad <= 0.0:
        return salida_no_disponible(
            radar_ts, "RADAR_ATRASADO",
            edad_radar_min=round(edad, 1),
            nota="Más de 40 min sin radar: sin proyección de celdas; solo satélite y ambiente.",
        )

    actual_dbz = cargar_dbz(ACTUAL)
    if actual_dbz is None:
        return salida_no_disponible(radar_ts, "NO_SE_PUDO_DECODIFICAR_DBZ", edad_radar_min=round(edad, 1))
    clutter = cargar_mascara_clutter(actual_dbz.shape)

    # Observación del frame actual: siempre, haya o no movimiento calculable.
    limpio_actual = campo_limpio(actual_dbz, clutter)
    observacion = observar(limpio_actual)
    observacion.update(actualizar_historial_ciudad(actual_dt, observacion["dbz_ciudad"]))
    extra = {"edad_radar_min": round(edad, 1), "observacion": observacion}

    def por_localidad(flujo=None, campos=None):
        import localidades_radar
        try:
            return localidades_radar.analizar(actual_dt, limpio_actual, flujo, edad, campos)
        except Exception as exc:  # nunca cortar el ciclo por esto
            print(f"AVISO: análisis por localidad no disponible: {exc}")
            return {}

    def puntos_control():
        import localidades_radar
        try:
            return localidades_radar.analizar_puntos_control(actual_dt, limpio_actual)
        except Exception as exc:
            print(f"AVISO: puntos de control no disponibles: {exc}")
            return {}

    previo = frame_previo(actual_dt)
    if previo is None:
        return salida_no_disponible(radar_ts, "SIN_FRAME_PREVIO_5_30_MIN", localidades=por_localidad(), puntos_control=puntos_control(), **extra)
    previo_path, previo_dt = previo
    gap = (actual_dt - previo_dt).total_seconds() / 60.0

    previo_dbz = cargar_dbz(previo_path)
    if previo_dbz is None or actual_dbz.shape != previo_dbz.shape:
        return salida_no_disponible(radar_ts, "NO_SE_PUDO_DECODIFICAR_DBZ", localidades=por_localidad(), puntos_control=puntos_control(), **extra)

    try:
        previas = velocidades_previas(previo_dt, previo_dbz, clutter)
    except Exception as exc:  # el control nunca corta el ciclo
        print(f"AVISO: control de velocidad con varias imágenes no disponible: {exc}")
        previas = []
    resultado = pronosticar(previo_dbz, actual_dbz, gap, clutter, velocidades_previas_kmh=previas)
    flujo = resultado["flujo"]
    if not flujo["valido"]:
        return salida_no_disponible(
            radar_ts, flujo.get("motivo", "FLUJO_INVALIDO"),
            frame_previo=iso(previo_dt), localidades=por_localidad(), puntos_control=puntos_control(), **extra,
        )

    try:
        np.savez_compressed(
            CAMPO,
            u=flujo["u"].astype(np.float32),
            v=flujo["v"].astype(np.float32),
            radar_timestamp=np.array(radar_ts),
            coherencia=np.array(flujo["coherencia"]),
        )
    except Exception as exc:
        print(f"AVISO: no se pudo guardar el campo de flujo: {exc}")

    ciudad = impacto_ciudad(resultado["actual"], flujo)
    f_gap = 1.0 if gap <= 12 else round(max(0.5, 1.0 - (gap - 12) / 36.0), 3)
    f_alc = factor_alcance(ciudad.get("distancia_radar_origen_km"))
    confianza = round(max(0.0, min(1.0, (0.4 + 0.6 * flujo["coherencia"]) * f_edad * f_gap * f_alc)), 3)

    # Los minutos ya transcurridos desde el barrido se descuentan del ETA.
    eta = ciudad["eta_tormenta_min"]
    eta_desde_ahora = None if eta is None else round(max(eta - edad, 0.0), 1)

    pron = {}
    for t, campo in resultado["campos"].items():
        pron[str(t)] = {
            "pixeles_ge_35dbz": int((campo >= UMBRAL_TORMENTA_DBZ).sum()),
            "dbz_max_bahia": next((s["dbz_max"] for s in ciudad["serie"] if s["minutos"] == t), None),
        }

    return {
        "version": VERSION,
        "disponible": True,
        "metodo": "flujo_optico_farneback_advection_semilagrangiana",
        "radar_timestamp": radar_ts,
        "frame_previo": iso(previo_dt),
        "intervalo_frames_min": round(gap, 1),
        "edad_radar_min": round(edad, 1),
        "generado_utc": iso(ahora),
        "movimiento": {
            "velocidad_media_kmh": flujo["velocidad_media_kmh"],
            "direccion_hacia_grados": flujo["direccion_hacia_grados"],
            "coherencia": flujo["coherencia"],
            "pixeles_eco": flujo["pixeles_eco"],
            "control_varias_imagenes": flujo.get("control_velocidad"),
        },
        "bahia_blanca": {
            **ciudad,
            "eta_tormenta_desde_ahora_min": eta_desde_ahora,
            "tormenta_en_60_min": eta is not None,
        },
        "pronostico": pron,
        "observacion": observacion,
        "localidades": por_localidad(flujo, dict(resultado["campos"])),
        "puntos_control": puntos_control(),
        "confianza": confianza,
        "factores_confianza": {
            "coherencia_movimiento": flujo["coherencia"],
            "edad_radar": f_edad,
            "intervalo_frames": f_gap,
            "alcance_radar": f_alc,
        },
        "clutter_enmascarado": bool(clutter.any()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generar-mascara", action="store_true")
    parser.add_argument("--frames", type=int, default=400)
    args = parser.parse_args(argv)

    if args.generar_mascara:
        frames = sorted(HISTORICO.glob("20*.png"))[-args.frames:]
        mask = generar_mascara_clutter(frames)
        if mask is None:
            print("No hay suficientes frames para generar la máscara.")
            return 1
        from PIL import Image

        MASCARA_CLUTTER.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((mask * 255).astype(np.uint8)).save(MASCARA_CLUTTER)
        print(f"Máscara de clutter: {int(mask.sum())} píxeles -> {MASCARA_CLUTTER}")
        return 0

    # Solo para reproducir frames históricos (herramientas/reproducir_frames.py).
    ahora = parse_utc(os.environ.get("CLIMAAR_AHORA_UTC", "")) if os.environ.get("CLIMAAR_AHORA_UTC") else None
    try:
        salida = ejecutar(ahora)
    except Exception as exc:  # el nowcast nunca corta el ciclo
        salida = salida_no_disponible(None, f"ERROR: {str(exc)[:200]}")
    escribir_json_atomico(SALIDA, salida)
    resumen = {k: salida.get(k) for k in ("disponible", "motivo", "radar_timestamp", "confianza")}
    if salida.get("disponible"):
        resumen["eta_tormenta_min"] = salida["bahia_blanca"]["eta_tormenta_min"]
        resumen["velocidad_kmh"] = salida["movimiento"]["velocidad_media_kmh"]
    print(json.dumps(resumen, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
