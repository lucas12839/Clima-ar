"""Punto 5b: nowcast en capas (nunca a ciegas sin imagen de radar nueva)."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import localidades_radar as lr
import nowcast_capas as ncap
import nowcast_capas_nucleo as nc
import nowcast_flujo_optico as nf

SHAPE = (512, 512)
T0 = datetime(2026, 11, 19, 22, 0, tzinfo=timezone.utc)


def celda(cx, cy, radio=8, dbz=50.0):
    h, w = SHAPE
    gy, gx = np.ogrid[:h, :w]
    campo = np.full(SHAPE, np.nan, dtype=np.float32)
    campo[(gx - cx) ** 2 + (gy - cy) ** 2 <= radio ** 2] = dbz
    return campo


def bahia_px():
    loc = next(l for l in lr.cargar_localidades() if l["id"] == "bahia_blanca")
    return nf.latlon_to_pixel(loc["lat"], loc["lon"], 512, 512)


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Radar sintético: celda que avanza 1 px/min hacia el este, 40 px al oeste de Bahía."""
    hist = tmp_path / "historico"
    hist.mkdir()
    bx, by = bahia_px()
    campos = {}

    def poner(dt, campo):
        p = hist / f"{dt:%Y%m%d%H%M%S}.png"
        p.write_bytes(b"x")
        campos[p.name] = campo
        return p

    for i, n in enumerate((20, 10, 0)):
        poner(T0 - timedelta(minutes=n), celda(bx - 60 + 10 * i, by))
    actual = tmp_path / "actual.png"
    actual.write_bytes(b"x")
    campos["actual.png"] = celda(bx - 40, by)
    ts = tmp_path / "radar_timestamp.txt"
    ts.write_text(nf.iso(T0), encoding="utf-8")

    monkeypatch.setattr(nf, "HISTORICO", hist)
    monkeypatch.setattr(nf, "ACTUAL", actual)
    monkeypatch.setattr(nf, "TIMESTAMP_FILE", ts)
    monkeypatch.setattr(nf, "cargar_dbz", lambda p: campos.get(p.name))
    monkeypatch.setattr(nc, "cargar_clutter_total", lambda shape: np.zeros(shape, dtype=bool))
    monkeypatch.setattr(nc, "cargar_intermitente", lambda shape: None)
    monkeypatch.setattr(nf, "frame_previo", lambda dt, historico=None: (
        (hist / f"{dt - timedelta(minutes=10):%Y%m%d%H%M%S}.png", dt - timedelta(minutes=10))
        if (hist / f"{dt - timedelta(minutes=10):%Y%m%d%H%M%S}.png").exists() else None))
    for nombre in ("SALIDA", "VERIFICACION", "GOES", "SENALES", "METEO", "CONFIG", "C13_AHORA", "C13_RADAR"):
        monkeypatch.setattr(ncap, nombre, tmp_path / f"{nombre.lower()}.{'npz' if nombre.startswith('C13') else 'json'}")
    return {"tmp": tmp_path, "campos": campos, "poner": poner, "ts": ts, "bx": bx, "by": by}


def test_confianza_baja_de_a_poco_sin_cortar_a_los_40_min():
    assert nc.confianza_por_edad(5) == 1.0
    assert nc.confianza_por_edad(40) == pytest.approx(0.5, abs=0.01)
    assert 0 < nc.confianza_por_edad(120) < nc.confianza_por_edad(60) < nc.confianza_por_edad(40)


def test_serie_igual_a_la_adveccion_del_nowcast_existente():
    act = np.nan_to_num(celda(200, 250))
    u = np.full(SHAPE, 1.0, dtype=np.float32)
    v = np.zeros(SHAPE, dtype=np.float32)
    s = nc.serie(act, u, v, [30.0])
    assert np.array_equal(s[30.0], nf.advectar(act, u, v, 30.0))


def test_ahora_adelantado_con_radar_atrasado(entorno):
    # Radar de hace 30 min: la celda ya debería estar sobre Bahía Blanca.
    salida = ncap.ejecutar(T0 + timedelta(minutes=30))
    bb = salida["localidades"]["bahia_blanca"]
    assert salida["edad_radar_min"] == 30.0
    assert bb["radar_movido_a_ahora"]["movido_min"] == 30.0
    assert bb["radar_movido_a_ahora"]["categoria"] == "tormenta"
    assert bb["estado_ahora"]["categoria"] == "tormenta"
    # Lo que se mostraba hasta ahora (imagen vieja) no tenía nada sobre la ciudad.
    assert nc.picos_por_zona(np.nan_to_num(entorno["campos"]["actual.png"]),
                             ncap.zonas_localidades(SHAPE, lr.cargar_localidades()))["bahia_blanca"] < 20
    assert salida["movimiento"]["velocidad_kmh"] == pytest.approx(56, abs=15)


def test_sigue_pronosticando_con_mas_de_40_min_de_atraso(entorno):
    salida = ncap.ejecutar(T0 + timedelta(minutes=75))
    assert salida["edad_radar_min"] == 75.0
    bb = salida["localidades"]["bahia_blanca"]
    assert "pronostico_desde_ahora" in bb
    assert 0 < bb["confianza"] < 0.5


def test_llegada_con_porcentaje_de_escenarios(entorno):
    salida = ncap.ejecutar(T0 + timedelta(minutes=5))
    lleg = salida["localidades"]["bahia_blanca"]["tormenta_proxima_hora"]
    assert lleg["porcentaje_escenarios"] >= 60
    assert 10 <= lleg["minutos_desde_ahora"] <= 40


def test_lo_medido_manda_sobre_el_radar_viejo():
    # Rayos solos: tormenta eléctrica, sin inventar lluvia.
    rell = {"rayos": {"a_15km_10min": 6, "a_30km_10min": 9, "edad_min": 2}}
    e = ncap.estado_ahora(0.0, 50.0, rell)
    assert e["categoria"] == "sin_eco" and e["fenomenos"]["tormenta_electrica"] is True
    # Rayos + tope frío encima: tormenta con lluvia.
    rell["satelite"] = {"tope_p10_c_15km": -52.0, "edad_min": 5}
    e = ncap.estado_ahora(0.0, 50.0, rell)
    assert e["categoria"] == "tormenta" and e["manda"] == "medido"
    rell = {"estaciones": {"lluvia_1h_mm": 3.0, "edad_min": 5}}
    assert ncap.estado_ahora(5.0, 10.0, rell)["categoria"] == "lluvia"
    # Radar viejo con tormenta, pero ahora cielo sin nube alta ni rayos: se avisa que pudo pasar.
    rell = {"rayos": {"a_15km_10min": 0, "a_30km_10min": 0, "edad_min": 2},
            "satelite": {"tope_p10_c_15km": 5.0, "edad_min": 5}}
    assert ncap.estado_ahora(45.0, 50.0, rell)["categoria"] == "posible_ya_paso_o_se_deshizo"
    # Sin radar: no se dice "sin tormenta".
    assert ncap.estado_ahora(None, None, {})["categoria"] == "sin_datos_radar"


def test_capa_estable_y_firmas_de_estaciones():
    m = {"ambiente_actual": {"temperature_2m": 13.3, "temperature_925hPa": 14.1}}
    est = ncap.capa_estable(m)
    assert est["conveccion_elevada"] and est["factor_viento_superficie"] < 1
    assert not ncap.capa_estable({"ambiente_actual": {"temperature_2m": 25, "temperature_925hPa": 18}})["conveccion_elevada"]
    assert ncap.firma_estacion({"cambio_temperatura_30min_c": -7, "salto_presion_30min_hpa": 3, "rafaga_kmh": 90}) == "reventon"
    assert ncap.firma_estacion({"cambio_temperatura_30min_c": 0.5, "salto_presion_30min_hpa": 0.8, "rafaga_kmh": 25}) == "frente_de_rafagas_debil"
    assert ncap.firma_estacion({"cambio_temperatura_30min_c": 0.0, "salto_presion_30min_hpa": 0.0}) is None


def test_validacion_automatica_compara_con_la_imagen_que_llega(entorno):
    s1 = ncap.ejecutar(T0 + timedelta(minutes=2))
    verif = s1["_verificacion"]
    assert any(p["objetivo_utc"] == nf.iso(T0 + timedelta(minutes=10)) for p in verif["pendientes"])
    ncap.VERIFICACION.write_text(json.dumps(verif), encoding="utf-8")
    # Llega la imagen de T0+10 con la celda donde se pronosticó.
    bx, by = entorno["bx"], entorno["by"]
    nueva = T0 + timedelta(minutes=10)
    entorno["poner"](nueva, celda(bx - 30, by))
    entorno["campos"]["actual.png"] = celda(bx - 30, by)
    entorno["ts"].write_text(nf.iso(nueva), encoding="utf-8")
    s2 = ncap.ejecutar(nueva + timedelta(minutes=1))
    v2 = s2["_verificacion"]
    assert v2["ultima_imagen_verificada"] == nf.iso(nueva)
    assert v2["ultimas"] and v2["ultimas"][0]["objetivo_utc"] == nf.iso(nueva)
    assert v2["acumulado"]


def test_sin_actividad_y_sin_radar(entorno):
    vacio = np.full(SHAPE, np.nan, dtype=np.float32)
    for k in list(entorno["campos"]):
        entorno["campos"][k] = vacio
    s = ncap.ejecutar(T0 + timedelta(minutes=3))
    assert s["movimiento"] is None
    assert all(v["estado_ahora"]["categoria"] == "sin_eco" for v in s["localidades"].values())
    # Sin radar en absoluto: lo dice y sigue con lo medido.
    entorno["ts"].unlink()
    s = ncap.ejecutar(T0 + timedelta(minutes=3))
    assert s["motivo_sin_radar"] == "SIN_RADAR_ACTUAL"
    assert all(v["estado_ahora"]["categoria"] == "sin_datos_radar" for v in s["localidades"].values())


def test_main_nunca_corta_el_ciclo(entorno, monkeypatch):
    monkeypatch.setattr(ncap, "ejecutar", lambda ahora=None: (_ for _ in ()).throw(RuntimeError("x")))
    assert ncap.main([]) == 0
    s = json.loads(ncap.SALIDA.read_text(encoding="utf-8"))
    assert s["disponible"] is False and "ERROR" in s["motivo"]


def test_anvil_suma_crecimiento_y_falla_segura(entorno, monkeypatch):
    pytest.importorskip("pysteps")
    ncap.CONFIG.write_text(json.dumps({"anvil": True}), encoding="utf-8")
    s = ncap.ejecutar(T0 + timedelta(minutes=25))
    assert s["anvil"]["usado"] is True
    assert s["localidades"]["bahia_blanca"]["estado_ahora"]["categoria"] == "tormenta"
    # Sin pysteps (o con error) siguen los escenarios solos y lo dice.
    monkeypatch.setattr(nc, "anvil", lambda *a, **k: (_ for _ in ()).throw(ImportError("sin pysteps")))
    s = ncap.ejecutar(T0 + timedelta(minutes=25))
    assert s["anvil"]["usado"] is False and "ImportError" in s["anvil"]["motivo"]
    assert s["localidades"]["bahia_blanca"]["estado_ahora"]["categoria"] == "tormenta"


def test_combinar_anvil_solo_desde_30_min():
    a = {20.0: np.zeros(SHAPE, np.float32), 30.0: np.zeros(SHAPE, np.float32)}
    b = {20.0: np.full(SHAPE, 40.0, np.float32), 30.0: np.full(SHAPE, 40.0, np.float32)}
    c = nc.combinar_anvil(a, b)
    assert c[20.0].max() == 0.0 and c[30.0].min() == 40.0


def test_mascara_intermitente_no_borra_una_tormenta_que_pasa(monkeypatch):
    inter = np.zeros(SHAPE, dtype=bool)
    inter[100:103, 100:103] = True            # puntos de eco del terreno
    monkeypatch.setattr(nc, "cargar_intermitente", lambda shape: inter)
    sin_clutter = np.zeros(SHAPE, dtype=bool)
    # 1) Eco suelto justo sobre el clutter (terreno): se borra.
    suelto = np.zeros(SHAPE, np.float32)
    suelto[100:103, 100:103] = 55.0
    assert nc.campo_limpio(suelto, sin_clutter)[100:103, 100:103].max() == 0.0
    # 2) Tormenta que pasa por encima del clutter: queda entera, sin perder superficie.
    tormenta = np.nan_to_num(celda(101, 101, radio=10))
    limpio = nc.campo_limpio(tormenta, sin_clutter)
    assert np.array_equal(limpio >= 35, tormenta >= 35)


def test_corrector_con_rayos_y_satelite_medidos(entorno, monkeypatch):
    import corrector_tiempo_real as ct

    ncap.CONFIG.write_text(json.dumps({"corrector": True}), encoding="utf-8")
    ahora = T0 + timedelta(minutes=30)
    # Satélite de hace 5 min: cielo SIN nube alta en toda la zona -> la tormenta movida no puede estar.
    lat, lon = np.meshgrid(np.linspace(-41, -36.5, 260), np.linspace(-65.5, -59, 260), indexing="ij")
    npz = entorno["tmp"] / "c13.npz"
    np.savez(npz, lat=lat.ravel().astype(np.float32), lon=lon.ravel().astype(np.float32),
             temp_c=np.full(lat.size, 12.0, np.float32), hora=np.array(nf.iso(ahora - timedelta(minutes=5))))
    monkeypatch.setattr(ncap, "C13_AHORA", npz)
    s = ncap.ejecutar(ahora)
    assert s["corrector"]["usado"] is True
    assert s["localidades"]["bahia_blanca"]["radar_movido_a_ahora"]["categoria"] == "sin_eco"
    # Rayos de ahora sobre Punta Alta (sin eco en el radar): aparece la tormenta.
    loc = next(l for l in lr.cargar_localidades() if l["id"] == "punta_alta")
    goes = {"glm": {"ventana": {"disponible": True, "fin_ventana_utc": nf.iso(ahora - timedelta(minutes=1)),
                                "rayos_10min": [[loc["lat"], loc["lon"]]] * 4, "por_localidad": {}}}}
    ncap.GOES.write_text(json.dumps(goes), encoding="utf-8")
    np.savez(npz, lat=lat.ravel().astype(np.float32), lon=lon.ravel().astype(np.float32),
             temp_c=np.full(lat.size, -50.0, np.float32), hora=np.array(nf.iso(ahora - timedelta(minutes=5))))
    s = ncap.ejecutar(ahora)
    assert s["localidades"]["punta_alta"]["radar_movido_a_ahora"]["categoria"] == "tormenta"
    assert s["corrector"]["pixeles_cambiados"].get("agregado_con_rayos", 0) > 0


def test_control_de_calidad_borra_ecos_con_cielo_despejado(entorno, monkeypatch):
    # Caso 8/11: el radar marca tormenta y el satélite de la misma hora no ve ninguna nube.
    lat, lon = np.meshgrid(np.linspace(-41, -36.5, 260), np.linspace(-65.5, -59, 260), indexing="ij")
    np.savez(ncap.C13_RADAR, lat=lat.ravel().astype(np.float32), lon=lon.ravel().astype(np.float32),
             temp_c=np.full(lat.size, 4.0, np.float32), hora=np.array(nf.iso(T0)))
    s = ncap.ejecutar(T0 + timedelta(minutes=2))
    cc = s["control_calidad_satelite"]
    assert cc["usado"] and cc["fraccion_sin_nube"] == 1.0
    assert s["localidades"]["bahia_blanca"]["radar_movido_a_ahora"]["categoria"] == "sin_eco"
    assert "tormenta_proxima_hora" not in s["localidades"]["bahia_blanca"]
    # Con nube de tormenta encima, el eco se respeta.
    np.savez(ncap.C13_RADAR, lat=lat.ravel().astype(np.float32), lon=lon.ravel().astype(np.float32),
             temp_c=np.full(lat.size, -55.0, np.float32), hora=np.array(nf.iso(T0)))
    s = ncap.ejecutar(T0 + timedelta(minutes=2))
    assert s["control_calidad_satelite"]["pixeles_sin_nube"] == 0
