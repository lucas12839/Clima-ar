#!/usr/bin/env python3
"""Simulación del punto 5 con datos reales: vuelve a correr el aviso de
tormentas en formación imagen por imagen en un período pasado.

Usa las imágenes de radar guardadas (data/radar/historico) y baja de NOAA el
satélite de cada momento (canal 13 de ahora, 10 y 20 min antes, fase, altura
del tope y, con --rayos, los rayos). Como del pasado no quedan los núcleos
del radar, se arman acá desde la imagen (zonas de 40 dBZ o más, sin ecos
fijos) y se siguen por cercanía, solo para verificar qué tormentas nuevas
fueron avisadas antes.

Uso (horas UTC):
  python herramientas/simular_formacion.py 2026-11-19T18:00 2026-11-20T02:00 [--rayos]

Necesita internet (NOAA). Al final imprime las llegadas de eco con su ventaja,
las tormentas nuevas con y sin aviso y las probables falsas alarmas.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE), str(BASE / "scripts")]

import aviso_formacion as af  # noqa: E402
import georreferenciar_rma10 as geo  # noqa: E402
from goes19 import formacion, state as st  # noqa: E402
from goes19.goes19_engine import latest_product_object  # noqa: E402

HISTORICO = BASE / "data" / "radar" / "historico"
NUCLEO_DBZ = 40.0
NUCLEO_MIN_PX = 10


def frames(desde: datetime, hasta: datetime) -> list[tuple[datetime, Path]]:
    salida = []
    for p in sorted(HISTORICO.glob("*.png")):
        try:
            t = datetime.strptime(p.stem, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if desde <= t <= hasta:
            salida.append((t, p))
    return salida


def nucleos_desde_imagen(dbz: np.ndarray, mascara: np.ndarray | None) -> list[dict]:
    from scipy import ndimage

    fuerte = np.nan_to_num(dbz, nan=-99.0) >= NUCLEO_DBZ
    if mascara is not None:
        fuerte &= ~mascara
    etiquetas, n = ndimage.label(fuerte)
    salida = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(etiquetas == i)
        if xs.size < NUCLEO_MIN_PX:
            continue
        lat, lon = geo.pixel_to_latlon(float(xs.mean()), float(ys.mean()), dbz.shape[1], dbz.shape[0])
        salida.append({"pixeles": int(xs.size), "dbz": {"max": float(np.nanmax(dbz[ys, xs]))},
                       "georreferencia": {"latitud": lat, "longitud": lon}})
    return salida


def asignar_ids(nucleos: list[dict], previos: list[dict], siguiente: list[int]) -> None:
    usados = set()
    for n in nucleos:
        g = n["georreferencia"]
        mejor = None
        for p in previos:
            if p["track_id"] in usados:
                continue
            d = af._d(g["latitud"], g["longitud"], p["georreferencia"]["latitud"], p["georreferencia"]["longitud"])
            if d <= 15.0 and (mejor is None or d < mejor[0]):
                mejor = (d, p["track_id"])
        if mejor is None:
            siguiente[0] += 1
            n["track_id"] = siguiente[0]
        else:
            n["track_id"] = mejor[1]
        usados.add(n["track_id"])


def satelite(frame: datetime, temp: Path, con_rayos: bool) -> dict:
    campos: dict = {}
    meta = latest_product_object("abi_c13", now=frame, hours_back=1)
    if meta is None:
        return {"estado": "sin_satelite", "objetos": []}
    dominio, _ = st.leer_dominio(meta, ["CMI"], temp, "c13")
    campos["abi_c13"] = st._campo(np.asarray(dominio["datos"]["CMI"], dtype=float), dominio,
                                  st._hora_satelite(meta, dominio))
    inicio = meta.get("start_time_utc")
    for producto in st.SENALES_PRODUCTS:
        try:
            st.process_senal(producto, temp, frame, inicio, campos)
        except Exception:
            pass
    try:
        st.process_cloudtop("cloudtop_height", "HT", temp, frame, inicio, campos)
    except Exception:
        pass
    previos, _ = st.leer_c13_previos(temp, frame, inicio)
    rayos = None
    if con_rayos:
        try:
            from goes19.glm_ventana import ventana_y_rayos_radar

            _, rayos = ventana_y_rayos_radar(ahora=frame, radio_km=st.RADIO_TORMENTAS_KM, frame_radar=frame)
        except Exception:
            rayos = None
    return formacion.detectar(campos.get("abi_c13"), previos.get(10), previos.get(20), {
        "fase": campos.get("cloud_phase"), "altura_topes_2km": campos.get("cloudtop_height_2km"),
        "altura_topes": campos.get("cloudtop_height")}, rayos, frame)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("desde")
    ap.add_argument("hasta")
    ap.add_argument("--rayos", action="store_true")
    ap.add_argument("--salida", default="")
    a = ap.parse_args()
    desde = datetime.fromisoformat(a.desde).replace(tzinfo=timezone.utc)
    hasta = datetime.fromisoformat(a.hasta).replace(tzinfo=timezone.utc)
    lista = frames(desde, hasta)
    print(f"{len(lista)} imágenes de radar entre {desde:%d/%m %H:%M} y {hasta:%d/%m %H:%M} UTC")
    previo, nucleos_previos, siguiente = None, [], [0]
    eventos, lineas = [], []
    for frame, png in lista:
        dbz = af.leer_radar_dbz(png)
        mascara = af.leer_mascara(dbz.shape) if dbz is not None else None
        nucleos = nucleos_desde_imagen(dbz, mascara) if dbz is not None else []
        asignar_ids(nucleos, nucleos_previos, siguiente)
        nucleos_previos = nucleos
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fs = satelite(frame, Path(tmp), a.rayos)
        except Exception as exc:
            fs = {"estado": "error", "error": str(exc)[:200], "objetos": []}
        goes = {"reference_frame_utc": frame.isoformat(), "formacion_satelite": fs}
        radar = {"actualizado": af.iso(frame), "nowcast": {"nucleos_georreferenciados": nucleos}}
        s = af.calcular(goes, radar, frame, None, previo, dbz, mascara)
        previo = s
        hora_ar = (frame.hour - 3) % 24
        forma = [o for o in s["objetos"] if o["tipo"] == "formacion"]
        print(f"{hora_ar:02d}:{frame.minute:02d} AR | satélite {fs.get('estado')} | núcleos radar {len(nucleos)} | "
              f"nubes en formación {len(forma)} "
              + " ".join(f"[{o['seguimiento_id']} g{o['calificacion']['grado']} {o['distancia_radar_km']}km "
                         f"{o['tope_min_c']}°C -{o['enfriamiento_10min_c']}°C/10' {o['eco_radar'].get('clase')}]"
                         for o in forma))
        for e in s["eventos"]:
            print(f"     evento: {json.dumps(e, ensure_ascii=False)}")
        eventos += s["eventos"]
        lineas.append({"frame": af.iso(frame), "satelite": fs.get("estado"), "objetos": s["objetos"],
                       "eventos": s["eventos"]})

    llegadas = [e for e in eventos if e["tipo"] == "llego_eco"]
    con_aviso = [e for e in llegadas if (e.get("grado_max") or 0) >= 1]
    nuevas = [e for e in eventos if e["tipo"] == "tormenta_nueva_en_radar"]
    falsas = [e for e in eventos if e["tipo"] == "sin_eco" and e.get("probable_falsa_alarma")]
    print("\nRESUMEN")
    print(f"  nubes avisadas que después tuvieron eco de tormenta: {len(con_aviso)} "
          f"(ventajas en min: {[e['ventaja_min'] for e in con_aviso]})")
    print(f"  tormentas nuevas en el radar: {len(nuevas)}, con aviso previo "
          f"{sum(1 for e in nuevas if e['con_aviso_previo'])}, sin aviso "
          f"{sum(1 for e in nuevas if not e['con_aviso_previo'])}")
    print(f"  avisos sin eco en 60 min (probables falsas alarmas): {len(falsas)}")
    if a.salida:
        Path(a.salida).write_text(json.dumps(lineas, ensure_ascii=False, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
