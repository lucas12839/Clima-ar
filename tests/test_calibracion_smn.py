"""Calibración de la lluvia del radar con la lluvia oficial del SMN (23/11/2026)."""
from datetime import datetime, timedelta, timezone

import scripts.estaciones.calibracion_smn as cs


def _v(est, dia, smn, radar, wid="w1"):
    return {"estacion": est, "wigos": wid, "dist_km": 2, "desde": f"{dia}T00:00:00Z", "hasta": f"{dia}T06:00:00Z",
            "smn_mm": smn, "radar_mm": radar, "imagenes": 36}


def test_ventanas_de_6h_y_traza():
    filas = [["2026-11-13T00:00:00Z/2026-11-13T06:00:00Z", "total_precipitation", 0.2],
             ["2026-11-13T06:00:00Z/2026-11-13T12:00:00Z", "total_precipitation", -0.1],   # traza
             ["2026-11-13T12:00:00Z/2026-11-14T12:00:00Z", "total_precipitation", 2.0],    # 24 h: no
             ["2026-11-13T00:00:00Z/2026-11-13T06:00:00Z", "total_precipitation", 0.4]]    # repetida: la mayor
    v = cs.ventanas_6h(filas)
    assert len(v) == 2 and max(v.values()) == 0.4 and 0.1 in v.values()


def test_factor_solo_con_casos_suficientes():
    pocos = [_v("A", f"2026-11-{10 + i:02d}", 5.0, 20.0) for i in range(5)]
    f = cs.calcular_factor(pocos)
    assert f["valido"] is False and f["factor"] is None and f["factor_mediano"] == 0.25
    muchos = [_v("A", f"2026-11-{1 + i:02d}", 4.0, 10.0) for i in range(12)]
    f = cs.calcular_factor(muchos)
    assert f["valido"] is True and f["factor"] == 0.4 and len(f["dias"]) == 12


def test_radar_sin_lluvia_real_se_cuenta_aparte():
    f = cs.calcular_factor([_v("A", "2026-11-01", 0.0, 3.0), _v("A", "2026-11-02", 2.0, 0.0)])
    assert f["radar_marco_lluvia_y_no_llovio"] == 1 and f["radar_vio_lluvia_cuando_llovio_pct"] == 0.0
    assert f["ventanas_para_factor"] == 0


def test_registro_sin_duplicados_y_con_vencimiento(tmp_path):
    ahora = datetime(2026, 11, 23, 12, tzinfo=timezone.utc)
    reg, fac = tmp_path / "r.json", tmp_path / "f.json"
    cs.actualizar([_v("A", "2026-11-20", 1.0, 2.0)], ahora, reg, fac)
    cs.actualizar([_v("A", "2026-11-20", 3.0, 2.0), _v("A", "2026-06-01", 9.0, 9.0)], ahora, reg, fac)
    import json
    ventanas = json.loads(reg.read_text())["ventanas"]
    assert len(ventanas) == 1 and ventanas[0]["smn_mm"] == 3.0   # corrige y descarta lo de hace > 120 días


def test_el_radar_usa_el_factor_del_smn_si_las_estaciones_no_alcanzan(tmp_path, monkeypatch):
    import json
    import localidades_radar as lr
    auto, smn = tmp_path / "a.json", tmp_path / "s.json"
    auto.write_text(json.dumps({"valido": False, "factor": None}))
    smn.write_text(json.dumps({"valido": True, "factor": 0.3}))
    monkeypatch.setattr(lr, "CALIBRACION_AUTO", auto)
    monkeypatch.setattr(lr, "CALIBRACION_SMN", smn)
    assert lr.factor_lluvia() == {"factor": 0.3, "fuente": "smn"}
    auto.write_text(json.dumps({"valido": True, "factor": 0.5}))
    assert lr.factor_lluvia() == {"factor": 0.5, "fuente": "estaciones"}
    smn.write_text(json.dumps({"valido": False}))
    auto.write_text(json.dumps({"valido": False}))
    assert lr.factor_lluvia()["fuente"] == "inicial"
