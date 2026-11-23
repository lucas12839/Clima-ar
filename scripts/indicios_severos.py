#!/usr/bin/env python3
"""Etapa G: indicios de tormenta severa sin Doppler (forma, movimiento y duración).

El RMA10 de SINARAME no publica el Doppler, así que la rotación y el viento dentro de
la tormenta no se ven. Se usan tres indicios que salen solo de la reflectividad, del
seguimiento de cada tormenta y del viento del modelo:

1. Eco en arco: una tormenta en línea que se curva hacia adelante (el centro avanza
   más que los extremos). Firma clásica de viento destructivo (16/12/2023 en Bahía).
2. Desvío de supercelda: la tormenta se mueve distinto del viento medio (850-500 hPa),
   hacia un costado de la cizalladura. En el hemisferio SUR las superceldas que dan
   granizo grande y tornados se desvían a la IZQUIERDA; en el norte, a la derecha
   (RELAMPAGO, 10/11/2018: supercelda que se desvía a la izquierda en Córdoba).
3. Duración: tormenta con núcleo fuerte que dura mucho (las severas viven más).

El mismo código corre en producción (RMA10, data/radar/historico) y en la simulación
con tormentas verificadas de Estados Unidos (herramientas/simular_indicios_eeuu.py),
para no medir una cosa y usar otra. No usa números de intensidad propios del radar
más que para encontrar las tormentas; los cortes se eligen con la simulación (etapa G,
docs/ETAPA_G.md).

Funciones puras sobre un campo de dBZ en una grilla plana (km por píxel): x hacia el
este, y hacia el SUR (filas de la imagen).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Cortes por defecto (los define la simulación; ver docs/ETAPA_G.md).
PARAMETROS_DEFECTO: dict[str, Any] = {
    "umbral_envolvente_dbz": 35.0,   # dónde termina la tormenta
    "umbral_nucleo_dbz": 45.0,       # núcleo fuerte
    "pixeles_nucleo_min": 4,
    "area_min_km2": 20.0,
    "velocidad_max_kmh": 120.0,      # tope físico para unir una tormenta entre imágenes
    "ventana_movimiento_min": 30.0,  # movimiento con las últimas 30 min
    "hemisferio": "sur",
    # Arco
    "arco_largo_min_km": 40.0,
    "arco_curvatura_min": 0.15,      # avance del centro sobre los extremos / largo
    # Desvío
    "desvio_min_kmh": 20.0,
    "desvio_imagenes_min": 2,
    # Duración
    "duracion_min": 60.0,
}


@dataclass
class Celda:
    x: float                 # centro (píxeles)
    y: float
    area_km2: float
    pico_dbz: float
    pixeles: np.ndarray      # índices planos de la envolvente
    largo_km: float = 0.0


@dataclass
class Trayectoria:
    id: int
    puntos: list = field(default_factory=list)    # (minutos, x, y)
    picos: list = field(default_factory=list)
    ultima: Celda | None = None
    desvios: list = field(default_factory=list)   # desvío (km/h) en cada imagen
    arcos: list = field(default_factory=list)     # curvatura en cada imagen


# --------------------------------------------------------------------------
# Tormentas en una imagen
# --------------------------------------------------------------------------
def celdas(campo: np.ndarray, km_px: float, par: dict | None = None) -> list[Celda]:
    """Manchas de >= envolvente que contienen un núcleo fuerte."""
    import cv2

    par = {**PARAMETROS_DEFECTO, **(par or {})}
    campo = np.nan_to_num(np.asarray(campo, dtype=np.float32), nan=0.0)
    env = (campo >= par["umbral_envolvente_dbz"]).astype(np.uint8)
    n, etiquetas = cv2.connectedComponents(env, connectivity=8)
    nucleo = campo >= par["umbral_nucleo_dbz"]
    salida = []
    area_px = km_px * km_px
    planas = etiquetas.ravel()
    orden = np.argsort(planas, kind="stable")
    cortes = np.searchsorted(planas[orden], np.arange(1, n + 1))
    for k in range(1, n):
        idx = orden[cortes[k - 1]:cortes[k]]
        if idx.size * area_px < par["area_min_km2"]:
            continue
        if int(nucleo.ravel()[idx].sum()) < par["pixeles_nucleo_min"]:
            continue
        ys, xs = np.divmod(idx, campo.shape[1])
        salida.append(Celda(x=float(xs.mean()), y=float(ys.mean()), area_km2=float(idx.size * area_px),
                            pico_dbz=float(campo.ravel()[idx].max()), pixeles=idx))
    return salida


# --------------------------------------------------------------------------
# Seguimiento entre imágenes
# --------------------------------------------------------------------------
def seguir(trayectorias: list[Trayectoria], nuevas: list[Celda], minutos: float, dt_min: float,
           km_px: float, siguiente_id: int, par: dict | None = None) -> tuple[list[Trayectoria], int]:
    """Une cada tormenta nueva con la trayectoria cuya posición esperada (con su
    movimiento) quede más cerca, sin pasar el tope físico de velocidad."""
    par = {**PARAMETROS_DEFECTO, **(par or {})}
    max_px = par["velocidad_max_kmh"] / 60.0 * dt_min / km_px
    usadas: set[int] = set()
    vivas = []
    candidatos = []
    for t in trayectorias:
        ex, ey = esperado(t, minutos)
        for j, c in enumerate(nuevas):
            d = math.hypot(c.x - ex, c.y - ey)
            if d <= max_px + math.sqrt(c.area_km2) / km_px * 0.5:
                candidatos.append((d, t.id, j))
    asignadas_t: dict[int, int] = {}
    for d, tid, j in sorted(candidatos):
        if tid in asignadas_t or j in usadas:
            continue
        asignadas_t[tid] = j
        usadas.add(j)
    por_id = {t.id: t for t in trayectorias}
    for tid, j in asignadas_t.items():
        t = por_id[tid]
        c = nuevas[j]
        t.puntos.append((minutos, c.x, c.y))
        t.picos.append(c.pico_dbz)
        t.ultima = c
        vivas.append(t)
    for j, c in enumerate(nuevas):
        if j in usadas:
            continue
        vivas.append(Trayectoria(id=siguiente_id, puntos=[(minutos, c.x, c.y)], picos=[c.pico_dbz], ultima=c))
        siguiente_id += 1
    return vivas, siguiente_id


def esperado(t: Trayectoria, minutos: float) -> tuple[float, float]:
    mov = movimiento_px_min(t)
    _, x, y = t.puntos[-1]
    if mov is None:
        return x, y
    dt = minutos - t.puntos[-1][0]
    return x + mov[0] * dt, y + mov[1] * dt


def movimiento_px_min(t: Trayectoria, ventana_min: float = 30.0) -> tuple[float, float] | None:
    """Velocidad (píxeles/min) con un ajuste lineal de las posiciones de la ventana."""
    if len(t.puntos) < 2:
        return None
    fin = t.puntos[-1][0]
    p = [q for q in t.puntos if q[0] >= fin - ventana_min - 0.1]
    if len(p) < 2:
        p = t.puntos[-2:]
    ts = np.array([q[0] for q in p], dtype=float)
    if ts.max() - ts.min() <= 0:
        return None
    vx = float(np.polyfit(ts, [q[1] for q in p], 1)[0])
    vy = float(np.polyfit(ts, [q[2] for q in p], 1)[0])
    return vx, vy


# --------------------------------------------------------------------------
# Indicios
# --------------------------------------------------------------------------
def viento_uv(velocidad_kmh: float | None, direccion_desde: float | None) -> tuple[float, float] | None:
    """Viento meteorológico (de dónde viene) a componentes este/norte en km/h."""
    if velocidad_kmh is None or direccion_desde is None:
        return None
    r = math.radians(direccion_desde)
    return -velocidad_kmh * math.sin(r), -velocidad_kmh * math.cos(r)


def desvio_kmh(mov_este_norte_kmh: tuple[float, float], viento_medio: tuple[float, float],
               cizalladura: tuple[float, float], hemisferio: str = "sur") -> float | None:
    """Componente del movimiento de la tormenta respecto del viento medio, perpendicular a
    la cizalladura, hacia el lado de las superceldas (izquierda en el sur, derecha en el
    norte). Positivo = se desvía como una supercelda (km/h)."""
    sx, sy = cizalladura
    s = math.hypot(sx, sy)
    if s < 1e-6:
        return None
    # Perpendicular a la derecha de la cizalladura: (sy, -sx); a la izquierda: (-sy, sx).
    nx, ny = ((sy / s, -sx / s) if hemisferio == "norte" else (-sy / s, sx / s))
    dx = mov_este_norte_kmh[0] - viento_medio[0]
    dy = mov_este_norte_kmh[1] - viento_medio[1]
    return dx * nx + dy * ny


def curvatura_arco(celda: Celda, ancho: int, km_px: float, mov_px_min: tuple[float, float] | None
                   ) -> tuple[float, float] | None:
    """(largo_km, curvatura): la tormenta se proyecta sobre el eje del movimiento (adelante)
    y su perpendicular (a lo largo de la línea). Curvatura = cuánto más avanza el borde
    delantero del centro que el de los extremos, dividido por el largo."""
    if mov_px_min is None:
        return None
    vx, vy = mov_px_min
    v = math.hypot(vx, vy)
    if v < 1e-6:
        return None
    fx, fy = vx / v, vy / v                 # adelante
    px_, py_ = -fy, fx                      # a lo largo
    ys, xs = np.divmod(celda.pixeles, ancho)
    xs = xs.astype(float) - celda.x
    ys = ys.astype(float) - celda.y
    adelante = (xs * fx + ys * fy) * km_px
    largo_c = (xs * px_ + ys * py_) * km_px
    lo, hi = np.percentile(largo_c, 2), np.percentile(largo_c, 98)
    largo = float(hi - lo)
    if largo <= 0:
        return None
    bins = np.linspace(lo, hi, 11)
    frente = []
    for a, b in zip(bins[:-1], bins[1:]):
        sel = (largo_c >= a) & (largo_c <= b)
        frente.append(float(adelante[sel].max()) if sel.any() else np.nan)
    frente = np.array(frente)
    if np.isnan(frente).sum() > 3:
        return largo, 0.0
    centro = np.nanmedian(frente[4:7])
    extremos = np.nanmedian(np.concatenate([frente[:2], frente[-2:]]))
    return largo, float((centro - extremos) / largo)


def indicios(t: Trayectoria, ancho: int, km_px: float, viento_medio: tuple[float, float] | None,
             cizalladura: tuple[float, float] | None, par: dict | None = None) -> dict[str, Any]:
    """Indicios de la trayectoria en su última imagen (se acumulan en t)."""
    par = {**PARAMETROS_DEFECTO, **(par or {})}
    mov = movimiento_px_min(t, par["ventana_movimiento_min"])
    salida: dict[str, Any] = {"id": t.id, "duracion_min": round(t.puntos[-1][0] - t.puntos[0][0], 1),
                              "pico_dbz": round(t.picos[-1], 1), "area_km2": round(t.ultima.area_km2, 1)}
    if mov is not None:
        este, norte = mov[0] * km_px * 60.0, -mov[1] * km_px * 60.0
        salida["velocidad_kmh"] = round(math.hypot(este, norte), 1)
        salida["direccion_hacia_grados"] = round(math.degrees(math.atan2(este, norte)) % 360.0, 1)
        if viento_medio is not None and cizalladura is not None:
            d = desvio_kmh((este, norte), viento_medio, cizalladura, par["hemisferio"])
            if d is not None:
                t.desvios.append(d)
                salida["desvio_kmh"] = round(d, 1)
    arco = curvatura_arco(t.ultima, ancho, km_px, mov)
    if arco is not None:
        salida["largo_km"], salida["curvatura"] = round(arco[0], 1), round(arco[1], 3)
        t.arcos.append(arco[1] if arco[0] >= par["arco_largo_min_km"] else 0.0)
    recientes = t.desvios[-par["desvio_imagenes_min"]:]
    salida["supercelda"] = bool(len(recientes) >= par["desvio_imagenes_min"]
                                and all(d >= par["desvio_min_kmh"] for d in recientes))
    salida["arco"] = bool(t.arcos and t.arcos[-1] >= par["arco_curvatura_min"])
    salida["larga_duracion"] = bool(salida["duracion_min"] >= par["duracion_min"])
    return salida


def ambiente(viento: dict[str, Any]) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Viento medio 850-500 hPa y cizalladura (500 hPa - 10 m) con las claves de Open-Meteo."""
    capas = [viento_uv(viento.get(f"wind_speed_{n}hPa"), viento.get(f"wind_direction_{n}hPa"))
             for n in (850, 700, 500)]
    capas = [c for c in capas if c is not None]
    if len(capas) < 2:
        return None, None
    medio = (sum(c[0] for c in capas) / len(capas), sum(c[1] for c in capas) / len(capas))
    abajo = viento_uv(viento.get("wind_speed_10m"), viento.get("wind_direction_10m"))
    arriba = viento_uv(viento.get("wind_speed_500hPa"), viento.get("wind_direction_500hPa"))
    if abajo is None or arriba is None:
        return medio, None
    return medio, (arriba[0] - abajo[0], arriba[1] - abajo[1])


# --------------------------------------------------------------------------
# Producción: el eco en arco SE VE en el viento de cada localidad (decisión del 23/11/2026,
# motor_decision_climaar.arco_para); el resto queda en paralelo.
# --------------------------------------------------------------------------
# Decidido con la simulación de EE. UU. (docs/ETAPA_G.md): solo el ECO EN ARCO aporta de
# verdad. El desvío de supercelda y la duración quedan calculados en el módulo pero no se
# usan (el desvío casi no supera a la base; la duración llega cuando ya pasó lo severo).
ARCO_PRODUCCION: dict[str, Any] = {
    "umbral_envolvente_dbz": 40.0,
    "arco_largo_min_km": 60.0,
    "arco_curvatura_min": 0.05,
    "velocidad_min_kmh": 30.0,
}
VENTANA_MIN = 90.0


def _produccion(ahora=None) -> dict[str, Any]:
    import json
    import sys
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    base = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(base / "scripts"))
    import nowcast_capas_nucleo as nc
    import nowcast_flujo_optico as nf

    ahora = ahora or datetime.now(timezone.utc)
    ts = nf.TIMESTAMP_FILE.read_text(encoding="utf-8").strip() if nf.TIMESTAMP_FILE.exists() else None
    radar_dt = nf.parse_utc(ts) if ts else None
    salida: dict[str, Any] = {"version": "indicios-severos-1.0", "en_paralelo": False,
                              "modo": "se_muestra_eco_en_arco", "generado_utc": nf.iso(ahora),
                              "radar_timestamp": ts, "parametros": ARCO_PRODUCCION, "tormentas": []}
    if radar_dt is None:
        return {**salida, "disponible": False, "motivo": "SIN_RADAR_ACTUAL"}
    frames = sorted((t, p) for p in nf.HISTORICO.glob("20*.png")
                    if (t := nf.timestamp_de_nombre(p)) is not None
                    and radar_dt - timedelta(minutes=VENTANA_MIN) <= t <= radar_dt)
    if not frames or frames[-1][0] != radar_dt:
        return {**salida, "disponible": False, "motivo": "IMAGEN_ACTUAL_NO_GUARDADA"}
    par = {**PARAMETROS_DEFECTO, **ARCO_PRODUCCION}
    trs, nid, previo, clutter = [], 1, None, None
    ancho, km_px = None, None
    for t, p in frames:
        d = nf.cargar_dbz(p)
        if d is None:
            continue
        if clutter is None:
            clutter = nc.cargar_clutter_total(d.shape)
            ancho, km_px = d.shape[1], nf.km_por_pixel(d.shape[1])
        campo = nc.campo_limpio(d, clutter)
        if t == radar_dt:
            # La imagen actual con el mismo control de ecos sin nube que lo que se ve.
            campo, _ = nf.limpiar_para_localidades(d, clutter, radar_dt)
        dt = 10.0 if previo is None else (t - previo).total_seconds() / 60.0
        if dt > 25.0:
            trs = []   # hueco del radar: no se une lo de antes con lo de después
        minutos = (t - frames[0][0]).total_seconds() / 60.0
        trs, nid = seguir(trs, celdas(campo, km_px, par), minutos, dt, km_px, nid, par)
        previo = t
    actual = (radar_dt - frames[0][0]).total_seconds() / 60.0
    for tr in trs:
        if tr.puntos[-1][0] != actual:
            continue
        r = indicios(tr, ancho, km_px, None, None, par)
        vel = r.get("velocidad_kmh") or 0.0
        r["arco"] = bool(r["arco"] and par["velocidad_min_kmh"] <= vel <= par["velocidad_max_kmh"])
        from rma10_features import pixel_to_latlon
        lat, lon = pixel_to_latlon(tr.ultima.x, tr.ultima.y, ancho, ancho)
        salida["tormentas"].append({k: r.get(k) for k in ("arco", "largo_km", "curvatura", "velocidad_kmh",
                                                          "direccion_hacia_grados", "pico_dbz", "area_km2",
                                                          "duracion_min")}
                                   | {"lat": round(lat, 4), "lon": round(lon, 4)})
    salida["disponible"] = True
    salida["arcos"] = sum(1 for x in salida["tormentas"] if x["arco"])
    return salida


def main() -> int:
    import json
    from pathlib import Path

    base = Path(__file__).resolve().parents[1]
    destino = base / "data" / "ia" / "indicios_severos.json"
    try:
        salida = _produccion()
        destino.parent.mkdir(parents=True, exist_ok=True)
        tmp = destino.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(salida, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        tmp.replace(destino)
        if salida.get("disponible") and salida.get("tormentas"):
            dia = (salida.get("radar_timestamp") or "")[:10]
            historial = base / "data" / "ia" / "indicios" / f"{dia}.jsonl"
            historial.parent.mkdir(parents=True, exist_ok=True)
            previas = historial.read_text(encoding="utf-8") if historial.exists() else ""
            if f'"radar_timestamp":"{salida["radar_timestamp"]}"' not in previas:
                with historial.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"radar_timestamp": salida["radar_timestamp"],
                                         "tormentas": salida["tormentas"]}, ensure_ascii=False,
                                        separators=(",", ":")) + "\n")
        print(f"Indicios de tormenta severa: {salida.get('arcos', 0)} ecos en arco "
              f"({len(salida.get('tormentas') or [])} tormentas) - en paralelo.")
    except Exception as exc:  # nunca corta el ciclo
        print(f"::warning::Indicios de tormenta severa no disponibles: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
