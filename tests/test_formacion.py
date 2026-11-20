"""Punto 5: nubes que se convierten en tormenta antes del radar (sin red)."""
import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from goes19 import formacion as fo
from goes19 import geospatial as gs

import aviso_formacion as af

FRAME = datetime(2026, 11, 20, 18, 0, tzinfo=timezone.utc)
HORA = FRAME - timedelta(minutes=2)
PROY = (math.radians(-75.2), 6378137.0, 6356752.31414, 35786023.0 + 6378137.0)

LAT0, LON0 = -38.10, -62.90          # nube nueva a ~95 km al noroeste del radar
LOCS = [{"id": "pueblo", "lat": -38.10, "lon": -62.65}]   # a ~22 km de la nube


def _grilla():
    lats = np.arange(-41.2, -36.2, 0.02)
    lons = np.arange(-65.4, -58.9, 0.025)
    la, lo = np.meshgrid(lats, lons)
    la, lo = la.ravel(), lo.ravel()
    m = gs.haversine_km(la, lo, gs.RADAR_LAT, gs.RADAR_LON) <= 265.0
    return la[m], lo[m]


LA, LO = _grilla()


def _campo(valores, minutos_antes=0.0):
    return {"valores": np.asarray(valores, dtype=float), "latitude": LA, "longitude": LO,
            "proyeccion": PROY, "hora_satelite": HORA - timedelta(minutes=minutos_antes)}


def _mancha(lat, lon, radio_km, temp_k, fondo=288.0, base=None):
    v = np.full(LA.size, fondo) if base is None else np.array(base, dtype=float)
    lat_v, lon_v = gs.posicion_aparente(lat, lon, 6.0, PROY)
    v[gs.haversine_km(LA, LO, lat_v, lon_v) <= radio_km] = temp_k
    return v


def test_nube_que_crece_se_detecta_con_cruce_de_0c():
    antes20 = _mancha(LAT0, LON0, 6, 280.0)
    antes10 = _mancha(LAT0, LON0, 7, 268.0)
    ahora = _mancha(LAT0, LON0, 8, 250.0)
    fase = np.where(ahora < 260, 4.0, 0.0)
    r = fo.detectar(_campo(ahora), _campo(antes10, 10), _campo(antes20, 20),
                    {"fase": _campo(fase)}, None, FRAME)
    assert r["estado"] == "ok"
    o = r["objetos"][0]
    assert abs(o["latitud"] - LAT0) < 0.05 and abs(o["longitud"] - LON0) < 0.05   # en el suelo
    assert o["tope_min_c"] == pytest.approx(250 - 273.15, abs=0.1)
    assert o["enfriamiento_10min_c"] >= 15 and o["enfriamiento_20min_c"] >= 25
    assert o["cruce_0c"] is True and o["fraccion_hielo"] == 1.0
    assert o["rayos"] == {"estado": "no_disponible"}
    json.dumps(r)


def test_nube_que_solo_se_mueve_no_cuenta():
    antes = _mancha(LAT0, LON0, 8, 240.0)
    ahora = _mancha(LAT0 - 0.07, LON0 + 0.05, 8, 240.0)   # ~9 km de movimiento, misma temperatura
    r = fo.detectar(_campo(ahora), _campo(antes, 10), None, {}, None, FRAME)
    assert r["estado"] == "sin_crecimiento" and r["objetos"] == []


def test_borde_de_yunque_que_avanza_no_cuenta():
    # Torre fría fija y yunque que se estira 25 km con el viento de altura.
    antes = _mancha(LAT0, LON0, 6, 210.0, base=_mancha(LAT0, LON0, 10, 225.0))
    ahora = _mancha(LAT0, LON0, 6, 210.0, base=_mancha(LAT0 - 0.1, LON0 + 0.2, 22, 228.0))
    r = fo.detectar(_campo(ahora), _campo(antes, 10), None, {}, None, FRAME)
    assert r["objetos"] == [] and r["descartados_borde_de_nube_fria"] >= 1


def test_sin_imagen_previa_y_sin_satelite():
    ahora = _mancha(LAT0, LON0, 8, 250.0)
    assert fo.detectar(_campo(ahora), None, None)["estado"] == "sin_imagen_previa"
    assert fo.detectar(None, None, None)["estado"] == "sin_satelite"
    # Imagen "previa" de otro momento (40 min): no se usa.
    assert fo.detectar(_campo(ahora), _campo(ahora, 40), None)["estado"] == "sin_imagen_previa"


def test_rayos_dentro_de_la_nube():
    antes = _mancha(LAT0, LON0, 6, 275.0)
    ahora = _mancha(LAT0, LON0, 8, 245.0)
    rayos = {"cobertura_min": 14.0, "minutos": np.array([1.0, 8.0, 3.0]),
             "latitude": np.array([LAT0, LAT0 + 0.01, -37.0]), "longitude": np.array([LON0, LON0, -60.0])}
    o = fo.detectar(_campo(ahora), _campo(antes, 10), None, {}, rayos, FRAME)["objetos"][0]
    assert o["rayos"]["cantidad"] == 2 and o["rayos"]["ultimos_5min"] == 1


# ----------------------------------------------------------------------
# Aviso: radar, grado, seguimiento y ventaja
# ----------------------------------------------------------------------

def _objeto(lat=LAT0, lon=LON0, enf=12.0, enf20=18.0, tope=-18.0, hielo=0.8, rayos=0, area=150.0):
    return {"latitud": lat, "longitud": lon, "distancia_radar_km": 95.0, "pixeles": 20,
            "area_km2": area, "radio_km": math.sqrt(area / math.pi), "tope_min_c": tope,
            "enfriamiento_10min_c": enf, "enfriamiento_20min_c": enf20, "cruce_0c": True,
            "fraccion_hielo": hielo, "altura_tope_km": 7.5, "vapor_menos_infrarrojo_c": -20.0,
            "rayos": {"estado": "ok", "cantidad": rayos, "cobertura_min": 14.0}}


def _goes(frame, objetos, hora=None):
    hora = hora or frame - timedelta(minutes=2)
    return {"reference_frame_utc": frame.isoformat(),
            "formacion_satelite": {"estado": "ok" if objetos else "sin_crecimiento",
                                   "hora_satelite_utc": af.iso(hora), "objetos": objetos}}


def _radar(frame, nucleos=()):
    return {"actualizado": af.iso(frame), "nowcast": {"nucleos_georreferenciados": [
        {"track_id": tid, "pixeles": 200, "dbz": {"max": dbz},
         "georreferencia": {"latitud": lat, "longitud": lon}} for tid, lat, lon, dbz in nucleos]}}


def _dbz_vacio():
    return np.full((512, 512), np.nan)


def _con_eco(lat, lon, valor, radio_px=4):
    d = _dbz_vacio()
    x, y = af.pixel_de(lat, lon, 512, 512)
    yy, xx = np.ogrid[:512, :512]
    d[(xx - x) ** 2 + (yy - y) ** 2 <= radio_px ** 2] = valor
    return d


METEO = {"ambiente_futuro": {"puntos": [{"punto": "X", "latitud": LAT0, "longitud": LON0, "datos": [
    {"tiempo_utc": "2026-11-20T18:00", "cape": 1800.0, "freezing_level_height": 3600.0}]}]}}


def test_pixel_de_es_la_inversa_de_la_georreferencia():
    import georreferenciar_rma10 as geo

    lat, lon = geo.pixel_to_latlon(300.0, 150.0, 512, 512)
    x, y = af.pixel_de(lat, lon, 512, 512)
    assert x == pytest.approx(300.0, abs=0.05) and y == pytest.approx(150.0, abs=0.05)


def test_ciclo_completo_aviso_seguimiento_y_ventaja():
    # Ciclo 1: nube que crece, sin eco en el radar.
    s1 = af.calcular(_goes(FRAME, [_objeto()]), _radar(FRAME), FRAME, METEO, None, _dbz_vacio(), None, LOCS)
    assert s1["estado"] == "ok" and s1["modo"] == "en_paralelo_no_se_muestra"
    o = s1["objetos"][0]
    assert o["tipo"] == "formacion" and o["eco_radar"]["clase"] == "sin_eco"
    c = o["calificacion"]
    assert c["grado"] == 1    # crece, sin eco débil ni rayos: vigilancia
    assert s1["localidades"]["pueblo"]["grado"] == 1
    assert s1["tracks_radar_conocidos"] == []
    assert s1["eventos"] == []

    # Ciclo 2: sigue creciendo, ahora con eco débil: inminente.
    f2 = FRAME + timedelta(minutes=10)
    s2 = af.calcular(_goes(f2, [_objeto(lat=LAT0 - 0.03, enf=14.0, tope=-23.0)]), _radar(f2), f2, METEO, s1,
                     _con_eco(LAT0 - 0.03, LON0, 28.0), None, LOCS)
    o2 = s2["objetos"][0]
    assert o2["seguimiento_id"] == o["seguimiento_id"]
    assert o2["calificacion"]["grado"] == 3 and "eco_debil_en_radar" in o2["calificacion"]["confirmaciones"]

    # Ciclo 3: la nube ya no se enfría (madura) y el radar ve 45 dBZ y un núcleo nuevo.
    f3 = f2 + timedelta(minutes=10)
    s3 = af.calcular(_goes(f3, []), _radar(f3, [(7, LAT0 - 0.05, LON0, 45.0)]), f3, METEO, s2,
                     _con_eco(LAT0 - 0.05, LON0, 45.0), None, LOCS)
    llegada = [e for e in s3["eventos"] if e["tipo"] == "llego_eco"][0]
    assert llegada["id"] == o["seguimiento_id"] and llegada["track_id_radar"] == 7
    assert llegada["ventaja_min"] == pytest.approx(22.0)   # primer aviso 17:58 -> radar 18:20
    nueva = [e for e in s3["eventos"] if e["tipo"] == "tormenta_nueva_en_radar"][0]
    assert nueva["con_aviso_previo"] is True and nueva["ventaja_min"] == pytest.approx(22.0)
    assert s3["seguimiento"] == []
    assert s3["resumen"]["tormentas_nuevas_con_aviso"] == 1

    # Repetir el ciclo 3 para la misma imagen da lo mismo (no se pierde la llegada).
    s3b = af.calcular(_goes(f3, []), _radar(f3, [(7, LAT0 - 0.05, LON0, 45.0)]), f3, METEO, s3,
                      _con_eco(LAT0 - 0.05, LON0, 45.0), None, LOCS)
    assert [e["tipo"] for e in s3b["eventos"]] == [e["tipo"] for e in s3["eventos"]]


def test_tormenta_nueva_sin_aviso_cuenta_como_no_anticipada():
    s1 = af.calcular(_goes(FRAME, []), _radar(FRAME), FRAME, METEO, None, _dbz_vacio(), None, LOCS)
    f2 = FRAME + timedelta(minutes=10)
    s2 = af.calcular(_goes(f2, []), _radar(f2, [(9, -38.4, -61.5, 50.0)]), f2, METEO, s1,
                     _con_eco(-38.4, -61.5, 50.0), None, LOCS)
    e = s2["eventos"][0]
    assert e["tipo"] == "tormenta_nueva_en_radar" and e["con_aviso_previo"] is False
    assert s2["resumen"]["tormentas_nuevas_sin_aviso"] == 1


def test_nube_sobre_tormenta_existente_no_es_formacion():
    s = af.calcular(_goes(FRAME, [_objeto()]), _radar(FRAME, [(3, LAT0, LON0, 55.0)]), FRAME, METEO, None,
                    _con_eco(LAT0, LON0, 55.0), None, LOCS)
    o = s["objetos"][0]
    assert o["tipo"] == "tormenta_existente_que_crece" and o["nucleo_radar"]["track_id"] == 3
    assert s["estado"] == "sin_formacion" and s["seguimiento"] == []


def test_ambiente_estable_y_nube_extensa_limitan_a_vigilancia():
    estable = {"ambiente_futuro": {"puntos": [{"punto": "X", "latitud": LAT0, "longitud": LON0, "datos": [
        {"tiempo_utc": "2026-11-20T18:00", "cape": 0.0}]}]}}
    s = af.calcular(_goes(FRAME, [_objeto(tope=-22.0, hielo=0.9)]), _radar(FRAME), FRAME, estable, None,
                    _dbz_vacio(), None)
    assert s["objetos"][0]["calificacion"]["grado"] == 1
    assert "ambiente estable" in s["objetos"][0]["calificacion"]["limites"]
    s = af.calcular(_goes(FRAME, [_objeto(enf=4.0, area=5000.0)]), _radar(FRAME), FRAME, METEO, None,
                    _dbz_vacio(), None)
    assert s["objetos"][0]["calificacion"]["grado"] <= 1


def test_grados_calibrados():
    def grado(**k):
        eco = k.pop("eco", None)
        dbz = _con_eco(LAT0, LON0, eco) if eco else _dbz_vacio()
        s = af.calcular(_goes(FRAME, [_objeto(**k)]), _radar(FRAME), FRAME, METEO, None, dbz, None)
        return s["objetos"][0]["calificacion"]["grado"]

    assert grado(tope=-5.0) == 0                      # cúmulo todavía chico
    assert grado(tope=-15.0) == 1                     # crece, sin confirmación
    assert grado(tope=-22.0, hielo=0.9) == 2          # cerca del radar, tope con hielo
    assert grado(tope=-15.0, eco=25.0) == 2           # eco débil
    assert grado(tope=-22.0, eco=25.0) == 3           # eco débil y tope bajo -20 °C
    assert grado(tope=-22.0, rayos=2) == 3            # primer rayo


def test_tope_muy_frio_sin_precipitacion_es_nube_alta():
    s = af.calcular(_goes(FRAME, [_objeto(tope=-36.0)]), _radar(FRAME), FRAME, METEO, None, _dbz_vacio(), None)
    c = s["objetos"][0]["calificacion"]
    assert c["grado"] == 0 and any("nube alta" in x for x in c["limites"])
    # Con eco débil o con rayos sí puede ser una tormenta naciendo.
    s = af.calcular(_goes(FRAME, [_objeto(tope=-36.0)]), _radar(FRAME), FRAME, METEO, None,
                    _con_eco(LAT0, LON0, 25.0), None)
    assert s["objetos"][0]["calificacion"]["grado"] >= 2


def test_tormenta_nueva_despues_de_la_llegada_cuenta_con_aviso():
    s1 = af.calcular(_goes(FRAME, [_objeto()]), _radar(FRAME), FRAME, METEO, None, _dbz_vacio(), None)
    f2 = FRAME + timedelta(minutes=10)
    # El radar ve 38 dBZ (llegó el eco) pero todavía no hay núcleo seguido.
    s2 = af.calcular(_goes(f2, []), _radar(f2), f2, METEO, s1, _con_eco(LAT0, LON0, 38.0), None)
    assert [e["tipo"] for e in s2["eventos"]] == ["llego_eco"]
    f3 = f2 + timedelta(minutes=10)
    s3 = af.calcular(_goes(f3, []), _radar(f3, [(5, LAT0 - 0.02, LON0 + 0.05, 48.0)]), f3, METEO, s2,
                     _con_eco(LAT0 - 0.02, LON0 + 0.05, 48.0), None)
    e = [x for x in s3["eventos"] if x["tipo"] == "tormenta_nueva_en_radar"][0]
    assert e["con_aviso_previo"] is True and e["ventaja_min"] == pytest.approx(12.0)


def test_sin_eco_en_60_min_es_probable_falsa_alarma():
    s = af.calcular(_goes(FRAME, [_objeto()]), _radar(FRAME), FRAME, METEO, None, _dbz_vacio(), None)
    for paso in range(1, 8):
        f = FRAME + timedelta(minutes=10 * paso)
        s = af.calcular(_goes(f, []), _radar(f), f, METEO, s, _dbz_vacio(), None)
        if any(e["tipo"] == "sin_eco" for e in s["eventos"]):
            break
    e = [e for e in s["eventos"] if e["tipo"] == "sin_eco"][0]
    assert e["probable_falsa_alarma"] is True and s["seguimiento"] == []


def test_sin_satelite_y_sin_actividad():
    s = af.calcular(None, _radar(FRAME), FRAME, None, None, _dbz_vacio(), None, LOCS)
    assert s["estado"] == "sin_satelite" and s["objetos"] == [] and s["seguimiento"] == []
    # Satélite de otro frame: no se usa.
    s = af.calcular(_goes(FRAME - timedelta(minutes=10), [_objeto()]), _radar(FRAME), FRAME, None, None,
                    _dbz_vacio(), None, LOCS)
    assert s["estado"] == "sin_satelite" and s["objetos"] == []
    # Sin radar: se avisa igual, pero como máximo "probable".
    s = af.calcular(_goes(FRAME, [_objeto(rayos=4)]), None, FRAME, METEO, None, None, None, LOCS)
    assert s["fuentes"]["radar"] == "no_disponible"
    assert s["objetos"][0]["calificacion"]["grado"] <= 2
    json.dumps(s)


def test_historial_una_linea_por_imagen(tmp_path):
    s = af.calcular(_goes(FRAME, [_objeto()]), _radar(FRAME), FRAME, METEO, None, _dbz_vacio(), None, LOCS)
    assert "guardado" in af._historial(s, tmp_path)
    assert "reemplazado" in af._historial(s, tmp_path)
    lineas = (tmp_path / "2026-11-20.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lineas) == 1 and json.loads(lineas[0])["objetos"][0]["grado"] == 1
    vacio = af.calcular(_goes(FRAME, []), _radar(FRAME), FRAME, None, None, _dbz_vacio(), None, LOCS)
    assert "no se guarda" in af._historial(vacio, tmp_path)
