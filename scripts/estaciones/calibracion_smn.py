#!/usr/bin/env python3
"""Calibración de la lluvia del radar con la lluvia OFICIAL del SMN (cada 6 h).

Las estaciones del SMN publican con horas de atraso: no sirven en vivo, pero
sí para calibrar. Una vez por día baja la lluvia de 6 h de las estaciones del
SMN dentro de 170 km del radar (Bahía Blanca Aero, Pigüé, Coronel Suárez, Tres
Arroyos, Río Colorado) y la compara con la que estima el radar crudo en el mismo
lugar y las mismas 6 h, sumando las imágenes guardadas (data/radar/historico).

Igual que en vivo: Marshall-Palmer sin corregir, tope de granizo, ecos fijos
fuera, círculo de 3 km alrededor de la estación. Una ventana cuenta solo si el
radar tiene al menos el 80 % de sus imágenes.

Escribe (solo este programa):
- data/ia/calibracion/registro_smn.json: cada ventana comparada (120 días).
- data/ia/calibracion/factor_lluvia_smn.json: factor = mediana de (SMN / radar)
  en las ventanas en que llovió de verdad; válido recién con >= 12 ventanas de
  >= 4 días distintos y >= 40 mm medidos. Mientras tanto no cambia nada.

Medido con el historial del 23/10 al 23/11/2026 (herramientas/calibrar_lluvia_smn.py):
16 ventanas con lluvia, 29 mm; el radar detectó todas, factor 0,20-0,29: pocos
casos para cambiar el factor inicial (0,4). Con este registro se junta solo.

Uso: python scripts/estaciones/calibracion_smn.py [--archivo historial.json] [--dias 4]
Nunca corta nada: ante un error lo informa y sale con 0.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[2]
for _p in (BASE, BASE / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HIST = BASE / "data" / "radar" / "historico"
CAL = BASE / "data" / "ia" / "calibracion"
REGISTRO = CAL / "registro_smn.json"
FACTOR = CAL / "factor_lluvia_smn.json"
ESTACIONES_SMN = BASE / "data" / "ia" / "estaciones" / "smn_stations.json"
URL = "https://w2b.smn.gov.ar/oapi/collections/urn%3Awmo%3Amd%3Aar-smn%3Aslt0ci/items"
RADAR = (-38.7342, -62.1654)
DIST_MAX_KM = 170.0
COBERTURA_MIN = 0.8
LLUVIA_MIN_MM = 1.0
RADAR_MIN_MM = 0.5
VENTANAS_MIN = 12
DIAS_MIN = 4
MM_MIN = 40.0
REGISTRO_DIAS = 120
IMAGENES_6H = 36


def parse(t: str) -> datetime:
    return datetime.fromisoformat(str(t).replace("Z", "+00:00")).astimezone(timezone.utc)


def iso(t: datetime) -> str:
    return t.isoformat().replace("+00:00", "Z")


def km(lat1, lon1, lat2, lon2) -> float:
    return math.hypot((lat1 - lat2) * 111.2, (lon1 - lon2) * 111.2 * math.cos(math.radians(lat1)))


def estaciones() -> dict[str, dict]:
    d = json.loads(ESTACIONES_SMN.read_text(encoding="utf-8"))
    salida = {}
    for f in d.get("features", []):
        lon, lat = f["geometry"]["coordinates"][:2]
        dist = km(lat, lon, *RADAR)
        if dist <= DIST_MAX_KM:
            salida[f["id"]] = {"nombre": f["properties"]["name"], "lat": lat, "lon": lon, "dist_km": round(dist)}
    return salida


def bajar(wigos: list[str], desde: datetime, hasta: datetime) -> dict[str, dict]:
    """Filas de lluvia del SMN (WIS2). Una estación que no responde queda vacía."""
    salida = {}
    for w in wigos:
        filas, offset = [], 0
        while True:
            q = {"f": "json", "wigos_station_identifier": w, "limit": 5000, "offset": offset,
                 "datetime": f"{iso(desde)}/{iso(hasta)}"}
            d = None
            for intento in range(3):
                try:
                    req = urllib.request.Request(URL + "?" + urllib.parse.urlencode(q),
                                                 headers={"User-Agent": "ClimaAR/1.0"})
                    d = json.loads(urllib.request.urlopen(req, timeout=90).read())
                    break
                except Exception as exc:  # noqa: PERF203
                    print(f"AVISO: SMN {w} intento {intento + 1}: {exc}")
                    time.sleep(5 * (intento + 1))
            if d is None:
                break
            f = d.get("features", [])
            for x in f:
                p = x.get("properties") or {}
                if "precip" in str(p.get("name")):
                    filas.append([p.get("phenomenonTime"), p.get("name"), p.get("value")])
            if len(f) < 5000:
                break
            offset += 5000
        salida[w] = {"filas": filas}
    return salida


def ventanas_6h(filas: list) -> dict[tuple[datetime, datetime], float]:
    """Lluvia de 6 h (traza = 0,1 mm). Si hay dos valores para la misma ventana, el mayor."""
    salida: dict[tuple[datetime, datetime], float] = {}
    for fila in filas:
        tiempo, nombre, valor = fila[0], fila[1], fila[2]
        if "precip" not in str(nombre) or not tiempo or "/" not in str(tiempo) or valor is None:
            continue
        a, b = (parse(x) for x in str(tiempo).split("/"))
        if (b - a) != timedelta(hours=6):
            continue
        mm = 0.1 if float(valor) < 0 else float(valor)
        salida[(a, b)] = max(salida.get((a, b), 0.0), mm)
    return salida


def frames(hist: Path = HIST) -> list[tuple[datetime, Path]]:
    salida = []
    for p in sorted(hist.glob("*.png")):
        try:
            salida.append((datetime.strptime(p.stem, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc), p))
        except ValueError:
            pass
    return salida


class Radar:
    """Lluvia del radar (sin corregir, mm/h) en el círculo de cada estación, por imagen."""

    def __init__(self, puntos: dict[str, dict]):
        import localidades_radar as lr
        import nowcast_capas_nucleo as nc
        import nowcast_flujo_optico as nf
        from scripts.rma10_features import latlon_to_pixel
        self.lr, self.nc, self.nf, self.a_px = lr, nc, nf, latlon_to_pixel
        self.puntos = puntos
        self.cache: dict[Path, dict[str, float] | None] = {}
        self.zonas = None
        self.clutter = None

    def en(self, path: Path) -> dict[str, float] | None:
        if path in self.cache:
            return self.cache[path]
        dbz = self.nf.cargar_dbz(path)
        if dbz is None:
            self.cache[path] = None
            return None
        if self.zonas is None:
            h, w = dbz.shape
            km_px = 240.0 / (w / 2.0 - 0.5)
            self.zonas = {}
            for k, p in self.puntos.items():
                cx, cy = self.a_px(p["lat"], p["lon"], w, h)
                self.zonas[k] = self.nf.disco((h, w), cx, cy, self.lr.RADIO_PUNTO_KM / km_px)
            self.clutter = self.nc.cargar_clutter_total(dbz.shape)
        campo = self.nc.campo_limpio(dbz, self.clutter)
        mmh = self.lr.mm_h(campo, corregir=False)
        valores = {k: float(mmh[z].mean()) if z.any() else 0.0 for k, z in self.zonas.items()}
        self.cache[path] = valores
        return valores


def comparar(smn: dict, puntos: dict[str, dict], lista: list[tuple[datetime, Path]]) -> list[dict]:
    if not lista:
        return []
    primero, ultimo = lista[0][0], lista[-1][0]
    radar = Radar(puntos)
    casos = []
    for wid, datos in smn.items():
        if wid not in puntos:
            continue
        for (a, b), mm in sorted(ventanas_6h(datos.get("filas") or []).items()):
            if a < primero or b > ultimo:
                continue
            dentro = [(t, p) for t, p in lista if a < t <= b]
            if len(dentro) < COBERTURA_MIN * IMAGENES_6H:
                continue
            total, ok = 0.0, 0
            anterior = a
            for t, p in dentro:
                v = radar.en(p)
                paso = min((t - anterior).total_seconds() / 3600.0, 0.25)  # nunca más de 15 min por imagen
                anterior = t
                if v is None:
                    continue
                total += v[wid] * paso
                ok += 1
            if ok < COBERTURA_MIN * IMAGENES_6H:
                continue
            if ok < IMAGENES_6H:
                total *= IMAGENES_6H / ok  # imágenes que SINARAME no publicó: promedio de la ventana
            casos.append({"estacion": puntos[wid]["nombre"], "wigos": wid, "dist_km": puntos[wid]["dist_km"],
                          "desde": iso(a), "hasta": iso(b), "smn_mm": round(mm, 1),
                          "radar_mm": round(total, 2), "imagenes": ok})
    return casos


def calcular_factor(registro: list[dict]) -> dict[str, Any]:
    llovio = [r for r in registro if r["smn_mm"] >= LLUVIA_MIN_MM]
    marco = [r for r in registro if r["radar_mm"] >= LLUVIA_MIN_MM]
    con_cociente = [r for r in llovio if r["radar_mm"] >= RADAR_MIN_MM]
    dias = sorted({r["desde"][:10] for r in con_cociente})
    total_smn = sum(r["smn_mm"] for r in con_cociente)
    total_rad = sum(r["radar_mm"] for r in con_cociente)
    salida = {
        "fuente": "SMN (WIS2), lluvia oficial de 6 h; radar crudo de SINARAME sin corregir",
        "ventanas_comparadas": len(registro),
        "ventanas_con_lluvia_smn": len(llovio),
        "radar_vio_lluvia_cuando_llovio_pct": round(100.0 * sum(1 for r in llovio if r["radar_mm"] >= 0.2)
                                                     / len(llovio), 1) if llovio else None,
        "radar_marco_lluvia_y_no_llovio": sum(1 for r in marco if r["smn_mm"] < 0.2),
        "ventanas_para_factor": len(con_cociente), "dias": dias,
        "smn_mm": round(total_smn, 1), "radar_mm": round(total_rad, 1),
        "factor_total": round(total_smn / total_rad, 3) if total_rad > 0 else None,
        "factor": None, "valido": False,
        "minimo_requerido": {"ventanas": VENTANAS_MIN, "dias": DIAS_MIN, "smn_mm": MM_MIN},
    }
    if con_cociente:
        salida["factor_mediano"] = round(statistics.median(r["smn_mm"] / r["radar_mm"] for r in con_cociente), 3)
    if len(con_cociente) >= VENTANAS_MIN and len(dias) >= DIAS_MIN and total_smn >= MM_MIN:
        salida["factor"] = salida["factor_mediano"]
        salida["valido"] = True
    por = {}
    for r in con_cociente:
        por.setdefault(r["estacion"], []).append(r)
    salida["por_estacion"] = {k: {"ventanas": len(v), "dist_km": v[0]["dist_km"],
                                  "smn_mm": round(sum(x["smn_mm"] for x in v), 1),
                                  "radar_mm": round(sum(x["radar_mm"] for x in v), 1)}
                              for k, v in sorted(por.items())}
    return salida


def actualizar(nuevos: list[dict], ahora: datetime, registro_path: Path = REGISTRO,
               factor_path: Path = FACTOR) -> dict[str, Any]:
    try:
        registro = json.loads(registro_path.read_text(encoding="utf-8")).get("ventanas") or []
    except (OSError, ValueError):
        registro = []
    por_clave = {(r["wigos"], r["desde"]): r for r in registro}
    for r in nuevos:
        por_clave[(r["wigos"], r["desde"])] = r  # la versión nueva corrige a la vieja
    limite = iso(ahora - timedelta(days=REGISTRO_DIAS))
    registro = sorted((r for r in por_clave.values() if r["hasta"] >= limite), key=lambda r: (r["desde"], r["wigos"]))
    factor = calcular_factor(registro)
    factor["actualizado_utc"] = iso(ahora)
    CAL.mkdir(parents=True, exist_ok=True)
    for path, datos in ((registro_path, {"actualizado_utc": iso(ahora), "ventanas": registro}), (factor_path, factor)):
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(datos, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)
    return factor


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archivo", default=None, help="historial del SMN ya bajado (en vez de bajarlo)")
    ap.add_argument("--dias", type=float, default=4.0)
    a = ap.parse_args(argv)
    try:
        ahora = datetime.now(timezone.utc)
        puntos = estaciones()
        if a.archivo:
            smn = json.loads(Path(a.archivo).read_text(encoding="utf-8"))
        else:
            smn = bajar(list(puntos), ahora - timedelta(days=a.dias), ahora)
        nuevos = comparar(smn, puntos, frames())
        factor = actualizar(nuevos, ahora)
        print(f"Calibración SMN: {len(nuevos)} ventanas nuevas; {factor['ventanas_para_factor']} con lluvia "
              f"({factor['smn_mm']} mm); factor {'%.3f' % factor['factor'] if factor['valido'] else 'todavía no'}"
              f" (mediana {factor.get('factor_mediano')}).")
    except Exception as exc:  # nunca corta nada
        print(f"::warning::Calibración con el SMN no disponible: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
