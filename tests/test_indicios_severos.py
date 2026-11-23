import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import indicios_severos as ind  # noqa: E402

KM = 1.0


def test_desvio_segun_hemisferio():
    # Cizalladura hacia el este; viento medio hacia el este a 40 km/h.
    medio, ciz = (40.0, 0.0), (60.0, 0.0)
    # Tormenta que va hacia el este y además hacia el NORTE (izquierda de la cizalladura).
    izquierda = ind.desvio_kmh((40.0, 25.0), medio, ciz, "sur")
    assert izquierda == 25.0
    assert ind.desvio_kmh((40.0, 25.0), medio, ciz, "norte") == -25.0
    # Hacia el sur = derecha: supercelda del norte, no del sur.
    assert ind.desvio_kmh((40.0, -25.0), medio, ciz, "norte") == 25.0


def test_viento_meteorologico():
    u, v = ind.viento_uv(36.0, 270.0)       # viene del oeste: va hacia el este
    assert round(u, 6) == 36.0 and abs(v) < 1e-9


def _campo_arco(curvo: bool):
    c = np.zeros((200, 200), np.float32)
    for x in range(60, 141):                # línea norte-sur de 80 km que avanza hacia el este
        s = (x - 100) / 40.0
        adelanto = int(round(20 * (1 - s * s))) if curvo else 0
        c[x, 95 + adelanto:105 + adelanto] = 50.0
    return c


def test_arco_vs_linea_recta():
    for curvo, esperado in ((True, True), (False, False)):
        campo = _campo_arco(curvo)
        cs = ind.celdas(campo, KM)
        assert len(cs) == 1
        t = ind.Trayectoria(id=1, puntos=[(0.0, cs[0].x - 10, cs[0].y), (10.0, cs[0].x, cs[0].y)],
                            picos=[50.0, 50.0], ultima=cs[0])
        r = ind.indicios(t, 200, KM, None, None, {"arco_curvatura_min": 0.08})
        assert r["largo_km"] > 70
        assert r["arco"] is esperado, r


def test_seguimiento_respeta_el_tope_de_velocidad():
    a = np.zeros((300, 300), np.float32)
    a[100:110, 100:110] = 50.0
    t, nid = ind.seguir([], ind.celdas(a, KM), 0.0, 10.0, KM, 1)
    b = np.zeros_like(a)
    b[100:110, 108:118] = 50.0              # 8 km en 10 min: la misma
    t, nid = ind.seguir(t, ind.celdas(b, KM), 10.0, 10.0, KM, nid)
    assert [x.id for x in t] == [1] and len(t[0].puntos) == 2
    c = np.zeros_like(a)
    c[100:110, 250:260] = 50.0              # 140 km en 10 min: imposible, es otra
    t, nid = ind.seguir(t, ind.celdas(c, KM), 20.0, 10.0, KM, nid)
    assert sorted(x.id for x in t) == [2]


def test_supercelda_necesita_desvio_sostenido():
    a = np.zeros((300, 300), np.float32)
    trs, nid = [], 1
    # Tormenta que va hacia el noreste (izquierda de una cizalladura hacia el este).
    for k in range(4):
        a[:] = 0
        a[150 - 5 * k:160 - 5 * k, 50 + 8 * k:60 + 8 * k] = 55.0
        trs, nid = ind.seguir(trs, ind.celdas(a, KM), 10.0 * k, 10.0, KM, nid)
        r = ind.indicios(trs[0], 300, KM, (40.0, 0.0), (60.0, 0.0))
    assert r["desvio_kmh"] > 20 and r["supercelda"]
    assert not r["larga_duracion"]
    assert math.isclose(r["duracion_min"], 30.0)


def test_produccion_sin_radar(tmp_path, monkeypatch):
    import nowcast_flujo_optico as nf
    monkeypatch.setattr(nf, "TIMESTAMP_FILE", tmp_path / "no_existe.txt")
    s = ind._produccion()
    assert s["disponible"] is False and s["en_paralelo"] and s["modo"] == "en_paralelo_no_se_muestra"


def test_produccion_con_tormenta_del_13_11(tmp_path, monkeypatch):
    """13/11/2026 00:20 UTC (radar guardado): una línea en arco al suroeste, a 54 km/h."""
    import nowcast_flujo_optico as nf
    if not list(nf.HISTORICO.glob("20261113002000.png")):
        import pytest
        pytest.skip("sin el radar guardado de ese día")
    f = tmp_path / "ts.txt"
    f.write_text("2026-11-13T00:20:00+00:00")
    monkeypatch.setattr(nf, "TIMESTAMP_FILE", f)
    s = ind._produccion()
    assert s["disponible"] and s["arcos"] >= 1
    arco = next(t for t in s["tormentas"] if t["arco"])
    assert 30 <= arco["velocidad_kmh"] <= 120 and arco["largo_km"] >= 60
