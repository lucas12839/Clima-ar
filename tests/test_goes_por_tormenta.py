"""GOES-19: ubicación de píxeles, recorte de 240 km, lectura parcial y datos
por tormenta sincronizados con el radar."""
import json
import math
import threading
from datetime import datetime, timedelta, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from goes19 import geospatial as gs
from goes19 import por_tormenta as pt

REQ, RPOL = 6378137.0, 6356752.31414
ALT = 35786023.0
LON0 = -75.2
PROY = (math.radians(LON0), REQ, RPOL, ALT + REQ)
KW = dict(lambda_0=PROY[0], r_eq=REQ, r_pol=RPOL, h=PROY[3])
FRAME = datetime(2026, 11, 19, 22, 40, tzinfo=timezone.utc)


# ------------------------------------------------------------------
# Ubicación de píxeles (fórmulas oficiales de NOAA)
# ------------------------------------------------------------------

def test_ejemplo_oficial_noaa():
    lat, lon = gs.fixed_grid_to_latlon(
        np.array([-0.024052]), np.array([0.095340]),
        lambda_0=math.radians(-75.0), r_eq=REQ, r_pol=RPOL, h=ALT + REQ)
    assert lat[0] == pytest.approx(33.846162, abs=1e-5)
    assert lon[0] == pytest.approx(-84.690932, abs=1e-5)


def test_bahia_blanca_en_la_grilla_y_vuelta():
    x, y = gs.geodetic_to_fixed_grid(gs.RADAR_LAT, gs.RADAR_LON, **KW)
    # Valores de la biblioteca de proyecciones PROJ (pyproj, proj=geos, sweep=x).
    assert x == pytest.approx(0.0299368, abs=2e-7)
    assert y == pytest.approx(-0.1059876, abs=2e-7)
    lat, lon = gs.fixed_grid_to_latlon(np.array([x]), np.array([y]), **KW)
    assert lat[0] == pytest.approx(gs.RADAR_LAT, abs=1e-6)
    assert lon[0] == pytest.approx(gs.RADAR_LON, abs=1e-6)


def test_paralaje_tope_10_km_corre_al_sureste():
    lat, lon = gs.posicion_aparente(gs.RADAR_LAT, gs.RADAR_LON, 10.0, PROY)
    d = gs.haversine_km(np.array([lat]), np.array([lon]), gs.RADAR_LAT, gs.RADAR_LON)[0]
    assert 9.0 < d < 12.5
    assert lat < gs.RADAR_LAT and lon > gs.RADAR_LON
    assert gs.posicion_aparente(gs.RADAR_LAT, gs.RADAR_LON, 0.0, PROY) == (gs.RADAR_LAT, gs.RADAR_LON)


# ------------------------------------------------------------------
# Archivo sintético con la grilla real de GOES-19 (zona de Bahía Blanca)
# ------------------------------------------------------------------

PASO = 56e-6  # 2 km en el punto bajo el satélite


def _grilla():
    x = np.arange(0.012, 0.048, PASO)
    y = np.arange(-0.124, -0.088, PASO)[::-1]  # de norte a sur, como en GOES
    return x, y


def _dataset():
    x, y = _grilla()
    xx, yy = np.meshgrid(x, y)
    lat, lon = gs.fixed_grid_to_latlon(xx, yy, **KW)
    d = gs.haversine_km(lat, lon, gs.RADAR_LAT, gs.RADAR_LON)
    cmi = (290.0 - 70.0 * np.exp(-((d / 30.0) ** 2))).astype("float32")
    ds = xr.Dataset(
        {"CMI": (("y", "x"), cmi),
         "DQF": (("y", "x"), np.zeros_like(cmi, dtype="int8")),
         "goes_imager_projection": ((), 0, {
             "grid_mapping_name": "geostationary",
             "perspective_point_height": ALT,
             "semi_major_axis": REQ,
             "semi_minor_axis": RPOL,
             "longitude_of_projection_origin": LON0,
             "sweep_angle_axis": "x"}),
         "y_image_bounds": (("n",), np.array([0.151872, -0.151872]))},
        coords={"x": x, "y": y},
    )
    return ds, lat, lon, d


@pytest.fixture(scope="module")
def archivo(tmp_path_factory):
    ds, *_ = _dataset()
    p = tmp_path_factory.mktemp("goes") / "OR_ABI-L2-CMIPF-M6C13_G19_prueba.nc"
    enc = {"CMI": {"chunksizes": (113, 113), "zlib": True}}
    ds.to_netcdf(p, engine="netcdf4", encoding=enc)
    return p


def test_ventana_minima_contiene_todo_el_circulo():
    ds, lat, lon, d = _dataset()
    xs, ys = gs.select_spatial_window(ds)
    dentro = np.isfinite(d) & (d <= gs.RADAR_RANGE_KM)
    filas, cols = np.where(dentro)
    assert filas.min() >= ys.start and filas.max() < ys.stop
    assert cols.min() >= xs.start and cols.max() < xs.stop
    # Mucho más chica que antes (~2000 km): solo un poco más que el círculo.
    assert (xs.stop - xs.start) < (cols.max() - cols.min() + 1) * 1.15 + 10
    assert (ys.stop - ys.start) < (filas.max() - filas.min() + 1) * 1.15 + 10


def test_recorte_identico_al_calculo_directo(archivo):
    ds, lat, lon, d = _dataset()
    r = gs.prepare_goes_domain(archivo, "CMI")
    dentro = np.isfinite(d) & (d <= gs.RADAR_RANGE_KM)
    assert r["data"].size == int(dentro.sum())
    assert np.allclose(np.sort(r["data"]), np.sort(ds["CMI"].values[dentro]))
    assert r["distance_km"].max() <= gs.RADAR_RANGE_KM
    # Bahía está en la parte sur del disco: el satélite pasa ~85 % del barrido.
    assert 0.8 < r["fraccion_escaneo_radar"] < 0.9


class _ConRangos(SimpleHTTPRequestHandler):
    """Servidor de prueba que entiende pedidos parciales (como el de NOAA)."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        ruta = Path(self.translate_path(self.path))
        datos = ruta.read_bytes()
        rango = self.headers.get("Range")
        if rango:
            a, b = rango.replace("bytes=", "").split("-")
            a = int(a)
            b = min(int(b) if b else len(datos) - 1, len(datos) - 1)
            parte = datos[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(datos)}")
            self.send_header("Content-Length", str(len(parte)))
            self.end_headers()
            self.wfile.write(parte)
            self.server.bytes_enviados += len(parte)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(datos)))
        self.end_headers()
        self.wfile.write(datos)
        self.server.bytes_enviados += len(datos)

    def do_HEAD(self):
        ruta = Path(self.translate_path(self.path))
        self.send_response(200)
        self.send_header("Content-Length", str(ruta.stat().st_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()


def test_lectura_parcial_por_internet_igual_a_la_completa(archivo):
    pytest.importorskip("fsspec")
    pytest.importorskip("h5netcdf")
    pytest.importorskip("aiohttp")
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), partial(_ConRangos, directory=str(archivo.parent)))
    servidor.bytes_enviados = 0
    hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
    hilo.start()
    try:
        url = f"http://127.0.0.1:{servidor.server_address[1]}/{archivo.name}"
        remoto = gs.prepare_goes_domain_multi(url, ["CMI", "DQF"])
        local = gs.prepare_goes_domain_multi(archivo, ["CMI", "DQF"])
    finally:
        servidor.shutdown()
    assert np.array_equal(remoto["datos"]["CMI"], local["datos"]["CMI"])
    assert np.array_equal(remoto["datos"]["DQF"], local["datos"]["DQF"])
    assert np.array_equal(remoto["latitude"], local["latitude"])
    assert servidor.bytes_enviados > 0


# ------------------------------------------------------------------
# Datos por tormenta
# ------------------------------------------------------------------

def _nowcast(nucleos, frame=FRAME):
    return {
        "actualizado": frame.isoformat().replace("+00:00", "Z"),
        "escala": {"km_por_pixel_estimado": 1.0},
        "nowcast": {"nucleos_georreferenciados": nucleos},
    }


def _nucleo(track, lat, lon, vel_px_min=0.0, rumbo=90.0, pixeles=100):
    return {"track_id": track, "nucleo_id": track, "pixeles": pixeles,
            "dbz": {"max": 55.0},
            "movimiento": {"velocidad_px_min": vel_px_min, "direccion_grados": rumbo},
            "georreferencia": {"latitud": lat, "longitud": lon}}


def _campo_frio(lat_c, lon_c, hora, frio_k=210.0):
    """Campo de 2 km con una mancha fría de 6 km alrededor de (lat_c, lon_c)."""
    lats = np.arange(lat_c - 0.6, lat_c + 0.6, 0.018)
    lons = np.arange(lon_c - 0.8, lon_c + 0.8, 0.023)
    la, lo = np.meshgrid(lats, lons)
    la, lo = la.ravel(), lo.ravel()
    d = gs.haversine_km(la, lo, lat_c, lon_c)
    v = np.where(d <= 6.0, frio_k, 285.0)
    return {"valores": v, "latitude": la, "longitude": lo, "proyeccion": PROY, "hora_satelite": hora}


def test_tormenta_se_encuentra_con_hora_y_perspectiva_corregidas():
    # Tormenta a 60 km al oeste de Bahía, va al este a 1 km/min. El satélite la
    # vio 8 min antes que el radar y con tope de 10 km.
    lat_r, lon_r = -38.73, -62.85
    t = pt.tormentas_desde_nowcast(_nowcast([_nucleo(7, lat_r, lon_r, vel_px_min=1.0, rumbo=90.0)]))[0]
    hora = FRAME - timedelta(minutes=8)
    lat_antes, lon_antes = pt.posicion_en(t, -8.0)
    lat_vista, lon_vista = gs.posicion_aparente(lat_antes, lon_antes, 10.0, PROY)
    campo = _campo_frio(lat_vista, lon_vista, hora)
    r = pt.calcular(_nowcast([_nucleo(7, lat_r, lon_r, vel_px_min=1.0, rumbo=90.0)]), FRAME,
                    {"c13": campo, "c08": None, "altura_topes": None, "presion_topes": None}, None)
    c13 = r["tormentas"][0]["c13"]
    assert r["estado"] == "ok"
    assert c13["estado"] == "ok"
    assert c13["sincronizacion"]["estado"] == "sincronizado"
    assert c13["sincronizacion"]["desfase_min"] == 8.0
    assert c13["min_c"] == pytest.approx(210.0 - 273.15, abs=0.1)
    # Sin correcciones el centro caería ~13 km al lado: la mancha fría sería
    # una parte chica de la tormenta.
    assert 10.0 < c13["corrimiento_km"] < 16.0


def test_altura_medida_se_usa_para_la_perspectiva():
    lat_r, lon_r = -38.9, -62.4
    nowcast = _nowcast([_nucleo(3, lat_r, lon_r)])
    hora = FRAME - timedelta(minutes=5)
    lat14, lon14 = gs.posicion_aparente(lat_r, lon_r, 14.0, PROY)
    altura = _campo_frio(lat14, lon14, hora)
    altura["valores"] = np.where(np.asarray(altura["valores"]) < 250, 14000.0, 3000.0)
    r = pt.calcular(nowcast, FRAME, {"c13": _campo_frio(lat14, lon14, hora), "c08": None,
                                     "altura_topes": altura, "presion_topes": None}, None)
    t = r["tormentas"][0]
    assert t["altura_paralaje_origen"] == "medida"
    assert t["altura_paralaje_km"] == pytest.approx(14.0, abs=0.1)
    assert t["altura_topes"]["max_km"] == pytest.approx(14.0, abs=0.01)


def test_satelite_muy_atrasado_no_se_usa():
    nowcast = _nowcast([_nucleo(1, -38.7, -62.3)])
    campo = _campo_frio(-38.7, -62.3, FRAME - timedelta(minutes=35))
    r = pt.calcular(nowcast, FRAME, {"c13": campo}, None)
    c13 = r["tormentas"][0]["c13"]
    assert c13["estado"] == "no_usado"
    assert c13["sincronizacion"]["estado"] == "desfase_excesivo"


def test_dato_posterior_al_radar_nunca_se_usa():
    s = pt.sincronizacion(FRAME + timedelta(minutes=5), FRAME)
    assert s["estado"] == "posterior_al_radar"


def test_sin_actividad_y_radar_de_otro_frame():
    vacio = pt.calcular(_nowcast([]), FRAME, {"c13": None}, None)
    assert vacio == {"estado": "sin_tormentas", "frame_radar_utc": "2026-11-19T22:40:00Z", "tormentas": []}
    otro = pt.calcular(_nowcast([_nucleo(1, -38.7, -62.3)], frame=FRAME - timedelta(minutes=10)),
                       FRAME, {"c13": None}, None)
    assert otro["estado"] == "radar_de_otro_frame" and otro["tormentas"] == []
    assert pt.calcular(None, FRAME, {}, None)["estado"] == "sin_datos_radar"


def test_rayos_se_atribuyen_con_la_posicion_de_cada_minuto():
    # Tormenta va al este a 1 km/min, radio 8 km.
    t = pt.tormentas_desde_nowcast(_nowcast([_nucleo(5, -38.7, -62.5, vel_px_min=1.0, rumbo=90.0, pixeles=1)]))[0]
    lat8, lon8 = pt.posicion_en(t, -8.0)           # dónde estaba hace 8 min
    rayos = {
        "cobertura_min": 10.0,
        "minutos": np.array([8.0, 8.0, 0.5, -1.0, 12.0]),
        "latitude": np.array([lat8, t["latitud"], t["latitud"], t["latitud"], t["latitud"]]),
        "longitude": np.array([lon8, t["longitud"] + 4.0 / (111.195 * math.cos(math.radians(-38.7))),
                               t["longitud"], t["longitud"], t["longitud"]]),
    }
    r = pt.rayos_por_tormenta(rayos, t, FRAME)
    # Cuentan: el de hace 8 min donde la tormenta ESTABA y el reciente.
    # No cuentan: el de hace 8 min a 12 km de donde estaba (aunque hoy quede
    # cerca), el posterior al radar ni el de hace 12 min (fuera de ventana).
    assert r["cantidad"] == 2
    assert r["tasa_por_min"] == 0.2
    assert pt.rayos_por_tormenta(None, t, FRAME)["estado"] == "no_disponible"


def test_enfriamiento_solo_entre_pasadas_de_la_misma_tormenta():
    nowcast = _nowcast([_nucleo(9, -38.7, -62.3)])
    h1 = FRAME - timedelta(minutes=12)
    previo = {"por_tormenta": {"tormentas": [{
        "track_id": 9,
        "c13": {"estado": "ok", "p10_c": -40.0,
                "sincronizacion": {"hora_satelite_utc": pt.iso(h1 - timedelta(minutes=10))}}}]}}
    r = pt.calcular(nowcast, FRAME, {"c13": _campo_frio(-38.7 - 0.09, -62.3 + 0.043, h1, frio_k=213.15)}, None, previo)
    c13 = r["tormentas"][0]["c13"]
    assert c13["estado"] == "ok"
    assert c13["enfriamiento_c_10min"] == pytest.approx((-40.0 - c13["p10_c"]), abs=0.05)
    # Otra tormenta (otro track) no se compara.
    previo["por_tormenta"]["tormentas"][0]["track_id"] = 99
    r2 = pt.calcular(nowcast, FRAME, {"c13": _campo_frio(-38.79, -62.257, h1, frio_k=213.15)}, None, previo)
    assert r2["tormentas"][0]["c13"]["enfriamiento_c_10min"] is None


def test_hora_local_del_barrido():
    ini = datetime(2026, 11, 19, 22, 30, tzinfo=timezone.utc)
    fin = ini + timedelta(minutes=9, seconds=40)
    assert pt.hora_local_escaneo(ini, fin, 0.85) == ini + (fin - ini) * 0.85
    assert pt.hora_local_escaneo(ini, fin, None) == fin


# ------------------------------------------------------------------
# Punto 2: señales de tormenta severa
# ------------------------------------------------------------------

from goes19 import senales_severas as ss  # noqa: E402


def _campo_ot(con_ot=True, yunque_k=218.0, tope_k=205.0):
    """Yunque frío de 25 km con (o sin) un tope sobresaliente en el centro."""
    lats = np.arange(-39.2, -38.2, 0.018)
    lons = np.arange(-62.9, -61.6, 0.023)
    la, lo = np.meshgrid(lats, lons)
    la, lo = la.ravel(), lo.ravel()
    d = gs.haversine_km(la, lo, -38.7, -62.25)
    bt = np.where(d <= 25, yunque_k, 285.0)
    if con_ot:
        bt = np.where(d <= 2.0, tope_k, bt)
    return bt, la, lo


def test_detecta_tope_sobresaliente():
    bt, la, lo = _campo_ot()
    ots = ss.detectar_topes_sobresalientes(bt, la, lo)
    assert len(ots) == 1
    assert ots[0]["bt_c"] == pytest.approx(205.0 - 273.15, abs=0.1)
    assert ots[0]["diferencia_c"] == pytest.approx(13.0, abs=0.1)


def test_no_detecta_sin_tope_o_con_yunque_tibio_o_tope_no_tan_frio():
    assert ss.detectar_topes_sobresalientes(*_campo_ot(con_ot=False)) == []
    # Yunque tibio (-30 °C): no hay yunque frío alrededor.
    assert ss.detectar_topes_sobresalientes(*_campo_ot(yunque_k=243.0, tope_k=212.0)) == []
    # Tope de -55 °C: no llega a -58 °C.
    assert ss.detectar_topes_sobresalientes(*_campo_ot(tope_k=218.15, yunque_k=226.0)) == []
    # Diferencia chica con el yunque (4 °C).
    assert ss.detectar_topes_sobresalientes(*_campo_ot(tope_k=210.0, yunque_k=214.0)) == []


def test_fase_y_congelamiento():
    r = ss.resumen_fase(np.array([4, 4, 4, 3, 0, np.nan]))
    assert r["pixeles"] == 5 and r["fraccion_hielo"] == 0.6 and r["fraccion_hielo_en_nubes"] == 0.75
    amb = {"ambiente_futuro": {"puntos": [
        {"punto": "lejos", "latitud": -36.0, "longitud": -60.0,
         "datos": [{"tiempo_utc": "2026-11-19T23:00", "freezing_level_height": 4000.0}]},
        {"punto": "cerca", "latitud": -38.7, "longitud": -62.3,
         "datos": [{"tiempo_utc": "2026-11-19T20:00", "freezing_level_height": 3500.0},
                   {"tiempo_utc": "2026-11-19T23:00", "freezing_level_height": 3100.0}]}]}}
    n = ss.nivel_congelamiento_para(amb, -38.72, -62.27, FRAME)
    assert n["estado"] == "ok" and n["metros"] == 3100 and n["punto"] == "cerca"
    assert ss.nivel_congelamiento_para(None, -38.7, -62.3, FRAME)["estado"] == "no_disponible"


def test_senales_por_tormenta_con_tope_sobresaliente_y_vapor():
    hora = FRAME - timedelta(minutes=3)
    lat_v, lon_v = gs.posicion_aparente(-38.7, -62.25, 10.0, PROY)
    bt, la, lo = _campo_ot()
    # Mover el campo para que el tope quede donde el satélite ve la tormenta.
    la = la + (lat_v - (-38.7))
    lo = lo + (lon_v - (-62.25))
    c13 = {"valores": bt, "latitude": la, "longitude": lo, "proyeccion": PROY, "hora_satelite": hora}
    c08 = dict(c13, valores=np.where(bt < 210, bt + 3.0, bt - 10.0))
    r = pt.calcular(_nowcast([_nucleo(2, -38.7, -62.25, pixeles=300)]), FRAME,
                    {"c13": c13, "c08": c08}, None)
    sev = r["tormentas"][0]["senales_severas"]
    assert r["topes_sobresalientes_zona"] == 1
    assert sev["topes_sobresalientes"]["cantidad"] == 1
    assert sev["topes_sobresalientes"]["max_diferencia_yunque_c"] == pytest.approx(13.0, abs=0.1)
    assert sev["vapor_menos_infrarrojo"]["max_c"] == pytest.approx(3.0, abs=0.01)
    assert sev["vapor_menos_infrarrojo"]["pixeles_positivos"] > 0
    assert sev["fase"] == {"estado": "no_disponible"}
    assert sev["congelamiento"]["estado"] == "no_disponible"


def test_vapor_infrarrojo_exige_mismo_barrido_y_grilla():
    hora = FRAME - timedelta(minutes=3)
    c = {"valores": np.array([1.0]), "latitude": np.array([1.0]), "longitude": np.array([1.0]), "hora_satelite": hora}
    assert ss.diferencia_vapor_infrarrojo(c, dict(c, hora_satelite=hora - timedelta(minutes=10))) is None
    assert ss.diferencia_vapor_infrarrojo(c, dict(c, latitude=np.array([2.0]))) is None
    assert ss.diferencia_vapor_infrarrojo(c, c) is not None
