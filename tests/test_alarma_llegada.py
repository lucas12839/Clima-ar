"""Idea 3: estaciones como alarma de "ya llega" (sin red)."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "scripts", ROOT / "scripts" / "estaciones"):
    sys.path.insert(0, str(p))

import alarma_llegada as al  # noqa: E402

AHORA = datetime(2026, 11, 20, 18, 0, tzinfo=timezone.utc)
SI = (lambda a, b: True, lambda a, b: False)
NO = (lambda a, b: False, lambda a, b: False)
# Viento hacia el este-sudeste (como el 20/11): Bahía Blanca Centro está aguas arriba de Punta Alta.
MOV = {"velocidad_kmh": 40.0, "direccion_hacia_grados": 110.0}


def serie(frente=True, ultimo_min=2, temp_caida=4.0, rafaga=60.0):
    s = []
    for m in range(40, ultimo_min - 1, -5):
        t = AHORA - timedelta(minutes=m)
        r = {"utc": t.isoformat(), "temp": 25.0, "presion": 1000.0, "viento": 8.0, "rafaga": None, "lluvia_dia": 0.0}
        if frente and m <= 10:
            r.update(temp=25.0 - temp_caida, presion=1002.0, viento=30.0, rafaga=rafaga)
        s.append(r)
    return s


def test_firma_de_frente_de_rafagas():
    f = al.firma(serie(), AHORA)
    assert f is not None and any("ráfaga" in x for x in f["senales"]) and len(f["senales"]) >= 2
    assert al.firma(serie(frente=False), AHORA) is None
    # Viento solo (sin bajar la temperatura ni subir la presión) no es frente de ráfagas.
    s = serie()
    for r in s:
        r["temp"], r["presion"] = 25.0, 1000.0
    assert al.firma(s, AHORA) is None


def test_alarma_aguas_abajo_y_llegada(tmp_path, monkeypatch):
    monkeypatch.setattr(al, "SALIDA", tmp_path / "a.json")
    r = al.calcular(AHORA, {"BAHIA BLANCA CENTRO": serie()}, MOV, SI, campos_verificacion={})
    locs = r["localidades"]
    assert locs["bahia_blanca"]["estado"] == "llego"
    assert locs["punta_alta"]["estado"] == "ya_llega"
    assert 0 <= locs["punta_alta"]["minutos"] <= 60
    assert locs["tornquist"]["estado"] is None          # al norte, no está en el camino
    assert r["pendientes"] and r["pendientes"][0]["localidad"] == "punta_alta"


def test_sin_tormenta_cerca_no_hay_alarma(tmp_path, monkeypatch):
    monkeypatch.setattr(al, "SALIDA", tmp_path / "a.json")
    r = al.calcular(AHORA, {"BAHIA BLANCA CENTRO": serie()}, MOV, NO, campos_verificacion={})
    assert all(v["estado"] is None for v in r["localidades"].values())
    assert r["estaciones_con_firma"] and r["estaciones_con_firma"][0]["tormenta_cerca"] is False


def test_dato_de_mas_de_15_min_no_se_usa(tmp_path, monkeypatch):
    monkeypatch.setattr(al, "SALIDA", tmp_path / "a.json")
    r = al.calcular(AHORA, {"BAHIA BLANCA CENTRO": serie(ultimo_min=20)}, MOV, SI, campos_verificacion={})
    assert r["estaciones_frescas"] == 0
    assert all(v["estado"] is None for v in r["localidades"].values())


def test_sin_movimiento_solo_llegada_local(tmp_path, monkeypatch):
    monkeypatch.setattr(al, "SALIDA", tmp_path / "a.json")
    r = al.calcular(AHORA, {"BAHIA BLANCA CENTRO": serie()}, None, SI, campos_verificacion={})
    assert r["localidades"]["bahia_blanca"]["estado"] == "llego"
    assert r["localidades"]["punta_alta"]["estado"] is None


def test_verificacion_en_vivo(tmp_path, monkeypatch):
    import nowcast_flujo_optico as nf

    monkeypatch.setattr(al, "SALIDA", tmp_path / "a.json")
    previo = {"pendientes": [
        {"ahora_utc": al.iso(AHORA - timedelta(minutes=60)), "localidad": "punta_alta", "minutos": 10, "estacion": "X"},
        {"ahora_utc": al.iso(AHORA - timedelta(minutes=60)), "localidad": "tornquist", "minutos": 10, "estacion": "X"},
    ]}
    (tmp_path / "a.json").write_text(json.dumps(previo), encoding="utf-8")
    shape = (480, 480)
    h, w = shape
    x, y = nf.latlon_to_pixel(-38.88, -62.075, w, h)       # tormenta sobre Punta Alta
    campos = {}
    for m in (55, 45, 35, 25):
        c = np.zeros(shape, dtype=np.float32)
        if m <= 45:
            c[int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3] = 50.0
        campos[AHORA - timedelta(minutes=m)] = c
    r = al.calcular(AHORA, {}, None, NO, campos_verificacion=campos)
    assert r["acumulado"]["acierto"] == 1 and r["acumulado"]["falsa_alarma"] == 1
    assert r["resumen"]["alarmas_verificadas"] == 2 and r["pendientes"] == []


def test_main_nunca_corta(tmp_path, monkeypatch):
    monkeypatch.setattr(al, "SALIDA", tmp_path / "a.json")
    monkeypatch.setattr(al, "calcular", lambda *a, **k: 1 / 0)
    assert al.main() == 0
    assert json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))["disponible"] is False
