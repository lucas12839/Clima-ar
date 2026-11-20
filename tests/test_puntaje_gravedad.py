"""Punto 4: puntaje de gravedad por tormenta (en paralelo)."""
from datetime import datetime, timezone

import puntaje_gravedad as pg

FRAME = datetime(2026, 11, 13, 19, 0, tzinfo=timezone.utc)
LOCS = [{"id": "punta_alta", "lat": -38.88, "lon": -62.075}]


def _nucleo(dbz, vel_px_min=0.5, track=1, lat=-38.88, lon=-62.075):
    return {"track_id": track, "pixeles": 300, "dbz": {"max": dbz + 1, "p99": dbz},
            "movimiento": {"velocidad_px_min": vel_px_min, "tendencia": "estable"},
            "georreferencia": {"latitud": lat, "longitud": lon}}


def _radar(nucleos):
    return {"actualizado": "2026-11-13T19:00:00Z", "nowcast": {"nucleos_georreferenciados": nucleos}}


def _goes(track=1, p10=-50.0, ot=0, ot_dif=None, sobre0=5.0, tasa=1.0, cantidad=10, salto=False,
          nivel=0.5, chicos=0.1, enfr=None):
    return {"reference_frame_utc": "2026-11-13T19:00:00+00:00", "por_tormenta": {"tormentas": [{
        "track_id": track,
        "c13": {"estado": "ok", "p10_c": p10, "enfriamiento_c_10min": enfr},
        "rayos": {"estado": "ok", "tasa_por_min": tasa, "cantidad": cantidad,
                  "salto_rayos": {"estado": "ok", "salto": salto, "nivel_sigma": nivel, "tasa_actual_por_min": tasa},
                  "tamano_rayos": {"cantidad": cantidad, "fraccion_chicos": chicos}},
        "senales_severas": {"topes_sobresalientes": {"estado": "ok", "cantidad": ot, "max_diferencia_yunque_c": ot_dif},
                            "congelamiento": {"estado": "ok", "tope_sobre_congelamiento_km": sobre0}}}]}}


def _meteo(cape=800.0, ciz_850=20.0, ciz_500=60.0, iso0=3500.0, hum=50.0):
    reg = {"tiempo_utc": "2026-11-13T19:00", "cape": cape, "freezing_level_height": iso0,
           "wind_speed_850hPa": ciz_850, "wind_direction_850hPa": 270.0,
           "wind_speed_500hPa": ciz_500, "wind_direction_500hPa": 270.0,
           "temperature_850hPa": 12.0, "temperature_500hPa": -15.0,
           "relative_humidity_700hPa": hum, "relative_humidity_500hPa": hum,
           "total_column_integrated_water_vapour": 30.0}
    return {"ambiente_futuro": {"puntos": [{"punto": "BB", "latitud": -38.72, "longitud": -62.27, "datos": [reg]}]}}


def _uno(nucleo, goes, meteo):
    r = pg.calcular(_radar([nucleo]), FRAME, goes, meteo, None, LOCS)
    return r, r["tormentas"][0]


def test_caso_punta_alta_radar_60_sin_confirmacion_no_es_severo():
    # 13/11: núcleo de ~60 dBZ crudo, granizo chico. Sin tope sobresaliente ni salto.
    r, t = _uno(_nucleo(60.0), _goes(p10=-48.0, tasa=3.0), _meteo(cape=600.0))
    assert t["granizo"]["grado"] < 4
    assert r["localidades"]["punta_alta"]["granizo_grado"] == t["granizo"]["grado"]


def test_supercelda_con_todas_las_senales_es_severa():
    _, t = _uno(_nucleo(64.0, vel_px_min=0.9),
                _goes(p10=-66.0, ot=2, ot_dif=12.0, sobre0=8.5, tasa=18.0, cantidad=180, salto=True, nivel=3.5,
                      chicos=0.6, enfr=7.0),
                _meteo(cape=2500.0, ciz_850=15.0, ciz_500=90.0, hum=30.0))
    assert t["granizo"]["grado"] == 4 and t["granizo"]["confirmado_por_otra_fuente"]
    assert t["rafagas"]["grado"] >= 3
    assert t["granizo"]["confianza"] == 1.0


def test_solo_radar_nunca_llega_a_severo_y_baja_la_confianza():
    _, t = _uno(_nucleo(66.0), None, None)
    assert t["granizo"]["grado"] == 3
    assert t["granizo"]["fuentes"] == ["radar"] and t["granizo"]["confianza"] == 0.4


def test_eco_fuerte_con_tope_tibio_y_sin_rayos_es_sospechoso():
    # Tornquist / sierras o eco cerca del radar: mucho dBZ, sin nube profunda ni rayos.
    _, t = _uno(_nucleo(62.0), _goes(p10=5.0, sobre0=None, tasa=0.0, cantidad=0, nivel=None, chicos=None),
                _meteo())
    assert t["sospecha_eco_no_meteorologico"] is True
    assert t["granizo"]["grado"] == 0 and t["rafagas"]["grado"] == 0


def test_cero_grados_muy_alto_baja_el_granizo():
    goes = _goes(p10=-60.0, ot=1, ot_dif=9.0, sobre0=6.0, tasa=8.0, cantidad=80)
    _, bajo = _uno(_nucleo(58.0), goes, _meteo(cape=1500.0, iso0=3000.0))
    _, alto = _uno(_nucleo(58.0), goes, _meteo(cape=1500.0, iso0=5000.0))
    assert alto["granizo"]["potencial"] < bajo["granizo"]["potencial"]


def test_lluvia_intensa_lenta_y_humeda():
    _, t = _uno(_nucleo(54.0, vel_px_min=0.1), _goes(tasa=4.0, cantidad=40), _meteo(cape=1200.0))
    assert t["lluvia_intensa"]["grado"] >= 3


def test_sin_tormentas_otro_frame_y_goes_de_otro_frame():
    assert pg.calcular(_radar([]), FRAME, None, None, None, LOCS)["estado"] == "sin_tormentas"
    otro = dict(_radar([_nucleo(50.0)]), actualizado="2026-11-13T18:50:00Z")
    assert pg.calcular(otro, FRAME, None, None, None, LOCS)["estado"] == "radar_de_otro_frame"
    g = _goes()
    g["reference_frame_utc"] = "2026-11-13T18:50:00+00:00"
    r, t = _uno(_nucleo(60.0), g, _meteo())
    assert r["goes_sincronizado"] is False and "satelite" not in t["granizo"]["fuentes"]


def test_rampa():
    assert pg.rampa(None, 0, 1) is None
    assert pg.rampa(5, 0, 10) == 0.5 and pg.rampa(-1, 0, 10) == 0.0 and pg.rampa(11, 0, 10) == 1.0


def test_integridad_del_puntaje(tmp_path, monkeypatch):
    import json

    import pytest
    import validar_integridad_climaar as vi

    monkeypatch.setattr(vi, "IA", tmp_path)
    nc = {"nucleos_georreferenciados": [{"track_id": 1}]}
    path = tmp_path / "puntaje_tormentas.json"
    ok = {"frame_radar_utc": "2026-11-13T19:00:00Z", "modo": "en_paralelo_no_se_muestra", "estado": "ok",
          "tormentas": [{"track_id": 1, "granizo": {"grado": 2}, "rafagas": {"grado": 0}, "lluvia_intensa": {"grado": 3}}]}
    path.write_text(json.dumps(ok), encoding="utf-8")
    vi.validar_puntaje(FRAME, nc)
    for cambio, texto in (({"modo": "publico"}, "paralelo"),
                          ({"tormentas": [{"track_id": 9}]}, "no tiene"),
                          ({"tormentas": [{"track_id": 1, "granizo": {"grado": 1}}]}, "inválido")):
        path.write_text(json.dumps({**ok, **cambio}), encoding="utf-8")
        with pytest.raises(SystemExit, match=texto):
            vi.validar_puntaje(FRAME, nc)
    path.write_text(json.dumps({**ok, "frame_radar_utc": "2026-11-13T18:50:00Z", "modo": "x"}), encoding="utf-8")
    vi.validar_puntaje(FRAME, nc)  # de otra imagen: permitido
