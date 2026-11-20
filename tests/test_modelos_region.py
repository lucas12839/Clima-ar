"""Punto 7: modelos por localidad (WRF del SMN y ensambles), con la red simulada."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr

import modelos_region as mr

AHORA = datetime(2026, 11, 20, 5, 0, tzinfo=timezone.utc)
INIT = datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc)


def grilla():
    lat, lon = np.meshgrid(np.linspace(-40, -37, 80), np.linspace(-64, -60, 100), indexing="ij")
    return lat, lon


def nc_10m(path, h):
    lat, lon = grilla()
    # Como en los archivos reales: alguna hora viene con 59 s (14:09:59).
    tiempos = [np.datetime64((INIT + timedelta(hours=h, minutes=10 * i, seconds=-1 if i % 2 else 0)).replace(tzinfo=None))
               for i in range(6)]
    # 1 mm en cada intervalo de 10 min (PP de los archivos 10M = lluvia de esos 10 min).
    pp = np.stack([np.full(lat.shape, 1.0, dtype=np.float32) for i in range(6)])
    xr.Dataset({"PP": (("time", "y", "x"), pp)}, coords={"time": tiempos, "lat": (("y", "x"), lat),
                                                         "lon": (("y", "x"), lon)}).to_netcdf(path)


def nc_01h(path, h):
    lat, lon = grilla()
    t = [np.datetime64((INIT + timedelta(hours=h)).replace(tzinfo=None))]
    campos = {v: (("time", "y", "x"), np.full((1,) + lat.shape, val, dtype=np.float32))
              for v, val in (("T2", 20.0), ("HR2", 60.0), ("PSFC", 1005.0), ("magViento10", 10.0),
                             ("Freezing_level", 3500.0))}
    xr.Dataset(campos, coords={"time": t, "lat": (("y", "x"), lat), "lon": (("y", "x"), lon)}).to_netcdf(path)


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    for nombre in ("SALIDA", "CACHE_WRF", "CACHE_ENS"):
        monkeypatch.setattr(mr, nombre, tmp_path / f"{nombre.lower()}.json")
    claves = [f"DATA/WRF/DET/2026/11/20/00/WRFDETAR_{t}_20261120_00_{h:03d}.nc"
              for t in ("10M", "01H") for h in range(0, 73)]
    bajadas = []
    monkeypatch.setattr(mr, "listar", lambda prefijo: claves if prefijo.endswith("/2026/11/20/00/") else [])

    def bajar(url, destino):
        h = int(url[-6:-3])
        (nc_10m if "_10M_" in url else nc_01h)(destino, h)
        bajadas.append(url)
        return destino

    monkeypatch.setattr(mr, "bajar", bajar)

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            n = 4
            horas = [f"2026-11-20T{5 + i:02d}:00" for i in range(12)]
            h = {"time": horas}
            for m in range(n):
                h[f"cape_member{m:02d}_gfs025_ensemble"] = [1500.0 if m < 2 else 100.0] * 12
                h[f"precipitation_member{m:02d}_gfs025_ensemble"] = [0.0] * 11 + [6.0 if m == 0 else 0.0]
                h[f"wind_gusts_10m_member{m:02d}_gfs025_ensemble"] = [30.0] * 12
            return [{"hourly": h} for _ in range(7)]

    llamadas = []

    def get(url, params=None, timeout=None):
        llamadas.append(url)
        return Resp()

    import requests
    monkeypatch.setattr(requests, "get", get)
    return {"bajadas": bajadas, "llamadas": llamadas}


def test_wrf_por_localidad_y_cache(entorno):
    s = mr.ejecutar(AHORA)
    w = s["wrf_smn"]
    assert w["estado"] == "ok" and w["corrida_utc"] == "2026-11-20T00:00:00Z"
    bb = w["localidades"]["bahia_blanca"]
    # 1 mm cada 10 min durante 3 h = 18 mm.
    assert bb["lluvia_10min_max_mm"] == pytest.approx(1.0)
    assert bb["lluvia_3h_media_mm"] == pytest.approx(18.0, abs=1.0)
    assert bb["proxima_hora"]["viento_kmh"] == pytest.approx(36.0)
    serie = json.loads(mr.CACHE_WRF.read_text())["localidades"]["bahia_blanca"]["lluvia_10min"]
    assert all(r["utc"][14:19] in ("00:00", "10:00", "20:00", "30:00", "40:00", "50:00") for r in serie)
    assert all(r["mm_10min_max"] == pytest.approx(1.0) for r in serie)
    n = len(entorno["bajadas"])
    assert n > 0
    # Segundo ciclo: no vuelve a bajar la corrida (usa lo guardado).
    mr.ejecutar(AHORA + timedelta(minutes=5))
    assert len(entorno["bajadas"]) == n


def test_ensambles_probabilidades_y_renovacion_horaria(entorno):
    s = mr.ejecutar(AHORA)
    e = s["ensambles"]["localidades"]["punta_alta"]
    assert e["cape_jkg"]["prob_ge_1000_pct"] == 50.0
    assert e["lluvia_mm_h"]["prob_ge_5.0_pct"] == 25.0
    assert e["rafaga_kmh"]["prob_ge_60.0_pct"] == 0.0
    n = len([u for u in entorno["llamadas"] if "ensemble" in u])
    mr.ejecutar(AHORA + timedelta(minutes=20))
    assert len([u for u in entorno["llamadas"] if "ensemble" in u]) == n
    mr.ejecutar(AHORA + timedelta(minutes=70))
    assert len([u for u in entorno["llamadas"] if "ensemble" in u]) == n + 1


def test_falla_segura_sin_red(entorno, monkeypatch):
    def caer(*a, **k):
        raise ConnectionError("sin red")

    monkeypatch.setattr(mr, "listar", caer)
    import requests
    monkeypatch.setattr(requests, "get", caer)
    assert mr.main([]) == 0
    s = json.loads(mr.SALIDA.read_text(encoding="utf-8"))
    assert s["wrf_smn"]["estado"] == "sin_datos" and "sin red" in s["wrf_smn"]["error"]
    assert s["ensambles"]["estado"] == "sin_datos"
