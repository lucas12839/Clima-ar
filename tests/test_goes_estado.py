"""Estado GOES-19 completo (sin red): contrato del archivo, mismo barrido,
datos por tormenta con y sin actividad, y control de integridad."""
import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from goes19 import geospatial as gs
from goes19 import state as st
from goes19 import topes_localidad as tl

FRAME = datetime(2026, 11, 19, 22, 40, tzinfo=timezone.utc)
INICIO = "2026-11-19T22:30:20Z"
FIN = "2026-11-19T22:39:51Z"
PROY = (math.radians(-75.2), 6378137.0, 6356752.31414, 35786023.0 + 6378137.0)


def _dominio(variables):
    lats = np.arange(-41.2, -36.2, 0.02)
    lons = np.arange(-65.4, -58.9, 0.025)
    la, lo = np.meshgrid(lats, lons)
    la, lo = la.ravel(), lo.ravel()
    d = gs.haversine_km(la, lo, gs.RADAR_LAT, gs.RADAR_LON)
    m = d <= st.RADIO_TORMENTAS_KM
    la, lo, d = la[m], lo[m], d[m]
    # Tormenta fría a 50 km al oeste del radar (vista corrida por perspectiva).
    lat_v, lon_v = gs.posicion_aparente(-38.73, -62.74, 10.0, PROY)
    frio = gs.haversine_km(la, lo, lat_v, lon_v) <= 8
    datos = {}
    for v in variables:
        if v == "CMI":
            datos[v] = np.where(frio, 205.0, 288.0)
        elif v == "HT":
            datos[v] = np.where(frio, 12500.0, 2000.0)
        elif v == "PRES":
            datos[v] = np.where(frio, 180.0, 850.0)
        elif v == "Phase":
            datos[v] = np.where(frio, 4.0, 0.0)
        elif v == "DQF":
            datos[v] = np.zeros(la.size)
    return {"datos": datos, "latitude": la, "longitude": lo, "distance_km": d,
            "proyeccion": PROY, "fraccion_escaneo_radar": 0.85}


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    llamados = []

    def buscar(product, reference_time, inicio_escaneo):
        llamados.append((product, inicio_escaneo))
        return ({"key": f"{product}.nc", "url": f"https://x/{product}.nc", "start_time_utc": INICIO,
                 "end_time_utc": FIN, "size": 1}, True)

    def leer(metadata, variables, temp_dir, nombre):
        return _dominio(variables), {"modo": "parcial", "segundos": 0.5}

    def previo(product, start, now=None, tolerance_seconds=2.0):
        llamados.append((product + "_previo", start))
        fin = (datetime.fromisoformat(start.replace("Z", "+00:00")) + timedelta(seconds=571))
        return {"key": f"{product}_{start}.nc", "url": f"https://x/{start}.nc", "start_time_utc": start,
                "end_time_utc": fin.isoformat().replace("+00:00", "Z"), "size": 1}

    monkeypatch.setattr(st, "_buscar_objeto", buscar)
    monkeypatch.setattr(st, "product_object_for_scan", previo)
    monkeypatch.setattr(st, "latest_product_object", lambda product, now=None, hours_back=6: {
        "key": f"{product}.nc", "url": f"https://x/{product}.nc", "start_time_utc": INICIO,
        "end_time_utc": FIN, "size": 1})
    monkeypatch.setattr(st, "leer_dominio", leer)
    monkeypatch.setattr(st, "process_glm", lambda temp_dir, ref: {"status": "sin_actividad"})
    monkeypatch.setattr(st, "process_dmw", lambda temp_dir, ref: {"status": "unavailable"})

    import goes19.glm_ventana as gv

    def ventana(centro, radio_km, frame_radar):
        rayos = {"cobertura_min": 10.0, "minutos": np.array([2.0, 3.0]),
                 "latitude": np.array([-38.73, -38.73]), "longitude": np.array([-62.74, -62.75]),
                 "archivos": 30}
        return {"disponible": True, "por_localidad": {}}, rayos

    monkeypatch.setattr(gv, "ventana_y_rayos_radar", ventana)
    monkeypatch.setattr(st, "OUTPUT_FILE", tmp_path / "goes19_state.json")
    monkeypatch.setattr(st, "RADAR_NOWCAST_FILE", tmp_path / "radar_nowcast.json")
    # 24/11/2026: sin esto la prueba escribía data/goes19/rayos_mapa.json del repo.
    monkeypatch.setattr(st, "RAYOS_MAPA_FILE", tmp_path / "rayos_mapa.json")
    (tmp_path / "meteo.json").write_text(json.dumps({"ambiente_futuro": {"puntos": [
        {"punto": "BB_CENTRO", "latitud": -38.72, "longitud": -62.27,
         "datos": [{"tiempo_utc": "2026-11-19T23:00", "freezing_level_height": 2858.0}]}]}}), encoding="utf-8")
    monkeypatch.setattr(st, "METEO_FILE", tmp_path / "meteo.json")
    return tmp_path, llamados


def _escribir_nowcast(path, nucleos):
    path.write_text(json.dumps({
        "actualizado": "2026-11-19T22:40:00Z",
        "escala": {"km_por_pixel_estimado": 0.93933},
        "nowcast": {"nucleos_georreferenciados": nucleos},
    }), encoding="utf-8")


NUCLEO = {"track_id": 4, "nucleo_id": 1, "pixeles": 291, "dbz": {"max": 58.0},
          "movimiento": {"velocidad_px_min": 0.0, "direccion_grados": 90.0},
          "georreferencia": {"latitud": -38.73, "longitud": -62.74}}


def test_estado_con_tormenta_mantiene_contrato_y_agrega_por_tormenta(entorno):
    tmp, llamados = entorno
    _escribir_nowcast(tmp / "radar_nowcast.json", [NUCLEO])
    s = st.build_state(FRAME)
    json.dumps(s)  # serializable
    # Contrato existente intacto.
    assert list(s["products"]) == ["abi_c08", "abi_c13"]
    for nombre in ("abi_c08", "abi_c13"):
        p = s["products"][nombre]
        assert p["status"] == "ok"
        assert {"source", "domain", "statistics"} <= set(p)
        assert p["domain"]["radius_km"] == 240.0
    assert "por_localidad" in s["products"]["abi_c13"]
    assert set(s["cloudtop"]) == {"cloudtop_height", "cloudtop_pressure"}
    assert s["glm"]["ventana"]["disponible"] is True
    assert s["glm"]["rayos_frame_radar"]["fin_utc"] == "2026-11-19T22:40:00Z"
    # Estadísticas de 240 km: no se mezclan los píxeles extra del borde.
    n240 = s["products"]["abi_c13"]["domain"]["points"]
    assert n240 < len(_dominio(["CMI"])["datos"]["CMI"])
    # Mismo barrido: el canal 13 primero, los demás con su inicio.
    assert llamados[0] == ("abi_c13", None)
    assert all(inicio == INICIO for p, inicio in llamados[1:] if not p.endswith("_previo"))
    # Punto 5: canal 13 de 10 y 20 min antes del mismo barrido de referencia.
    previos = [inicio for p, inicio in llamados if p.endswith("_previo")]
    # (la primera es la previa de "satelite_ahora", punto 5b)
    assert previos == ["2026-11-19T22:20:20Z", "2026-11-19T22:20:20Z", "2026-11-19T22:10:20Z"]
    fs = s["formacion_satelite"]
    assert fs["frame_radar_utc"] == "2026-11-19T22:40:00Z"
    assert fs["lectura_previas"]["10"]["estado"] == "ok"
    # Punto 5b: la imagen más reciente del satélite por localidad.
    sa = s["satelite_ahora"]
    assert sa["estado"] == "ok" and "bahia_blanca" in sa["por_localidad"]
    assert sa["por_localidad"]["bahia_blanca"]["enfriamiento_c_10min"] == 0.0
    # La tormenta fija no se enfría (misma imagen antes y ahora): no es formación.
    assert fs["estado"] == "sin_crecimiento" and fs["objetos"] == []
    # Datos por tormenta.
    pt = s["por_tormenta"]
    assert pt["estado"] == "ok"
    t = pt["tormentas"][0]
    assert t["track_id"] == 4
    assert t["c13"]["estado"] == "ok" and t["c13"]["min_c"] < -60
    assert t["altura_topes"]["max_km"] == 12.5
    assert t["presion_topes"]["min_hpa"] == 180.0
    assert t["rayos"]["cantidad"] == 2
    assert t["altura_paralaje_origen"] == "medida"
    sinc = s["sincronizacion"]
    assert sinc["inicio_barrido_referencia_utc"] == INICIO
    assert sinc["productos"]["abi_c13"]["estado"] == "sincronizado"
    assert 0 < sinc["productos"]["abi_c13"]["desfase_min"] < 20
    assert s["version"] == "1.8"
    # Punto 2: señales severas por tormenta.
    sev = t["senales_severas"]
    assert sev["fase"]["estado"] == "ok" and sev["fase"]["fraccion_hielo_en_nubes"] == 1.0
    assert sev["altura_topes_2km"]["max_km"] == 12.5
    assert sev["congelamiento"]["estado"] == "ok"
    assert sev["congelamiento"]["tope_sobre_congelamiento_km"] == pytest.approx(12.5 - 2.858, abs=0.01)
    assert sev["topes_sobresalientes"]["estado"] == "ok"
    assert set(s["senales_severas"]["productos"]) == {"cloudtop_height_2km", "cloud_phase"}
    assert s["sincronizacion"]["productos"]["cloud_phase"]["estado"] == "sincronizado"


def test_estado_sin_actividad(entorno):
    tmp, _ = entorno
    _escribir_nowcast(tmp / "radar_nowcast.json", [])
    s = st.build_state(FRAME)
    assert s["por_tormenta"]["estado"] == "sin_tormentas"
    assert s["por_tormenta"]["tormentas"] == []
    assert s["products"]["abi_c13"]["status"] == "ok"
    json.dumps(s)


def test_si_falla_la_lectura_el_ciclo_sigue(entorno, monkeypatch):
    tmp, _ = entorno
    _escribir_nowcast(tmp / "radar_nowcast.json", [NUCLEO])

    def falla(*a, **k):
        raise RuntimeError("NOAA no responde")

    monkeypatch.setattr(st, "leer_dominio", falla)
    s = st.build_state(FRAME)
    assert s["products"]["abi_c13"]["status"] == "unavailable"
    t = s["por_tormenta"]["tormentas"][0]
    assert t["c13"] == {"estado": "no_disponible", "enfriamiento_c_10min": None}
    assert t["rayos"]["cantidad"] == 2  # los rayos siguen


def test_lectura_parcial_cae_a_descarga_completa(tmp_path, monkeypatch):
    llamadas = []

    def preparar(origen, variables, radio_km):
        llamadas.append(str(origen))
        if str(origen).startswith("https://"):
            raise OSError("sin pedidos parciales")
        return _dominio(variables)

    class R:
        content = b"x"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(st, "prepare_goes_domain_multi", preparar)
    monkeypatch.setattr(st.requests, "get", lambda url, timeout: R())
    dominio, info = st.leer_dominio({"url": "https://x/a.nc"}, ["CMI"], tmp_path, "abi_c13")
    assert info["modo"] == "completo" and "sin pedidos parciales" in info["error_parcial"]
    assert llamadas[0].startswith("https://") and not llamadas[1].startswith("https://")


def test_topes_por_localidad_ignora_estados_viejos_mal_ubicados(tmp_path):
    previo = tmp_path / "prev.json"
    previo.write_text(json.dumps({"version": "1.5", "products": {"abi_c13": {
        "source": {"end_time_utc": "2026-11-19T22:29:51Z"},
        "por_localidad": {"bahia_blanca": {"p10_c_30km": 10.0}}}}}), encoding="utf-8")
    locs = [{"id": "bahia_blanca", "lat": -38.7183, "lon": -62.2663}]
    r = tl.por_localidad([250.0] * 3, [-38.72] * 3, [-62.27] * 3, "2026-11-19T22:39:51Z", previo, locs)
    assert "enfriamiento_c_10min" not in r["bahia_blanca"]


# ------------------------------------------------------------------
# Control de integridad
# ------------------------------------------------------------------

def _validar(tmp_path, monkeypatch, goes, nucleos):
    import validar_integridad_climaar as vi

    (tmp_path / "data" / "goes19").mkdir(parents=True)
    (tmp_path / "data" / "goes19" / "goes19_state.json").write_text(json.dumps(goes), encoding="utf-8")
    monkeypatch.setattr(vi, "BASE", tmp_path)
    vi.validar_goes19(FRAME, {"nucleos_georreferenciados": nucleos})


def _goes(tormentas, estado="ok", hora="2026-11-19T22:38:00Z"):
    return {"reference_frame_utc": "2026-11-19T22:40:00+00:00",
            "sincronizacion": {"productos": {"abi_c13": {"hora_satelite_utc": hora}}},
            "por_tormenta": {"estado": estado, "frame_radar_utc": "2026-11-19T22:40:00Z", "tormentas": tormentas}}


def test_integridad_acepta_estado_coherente(tmp_path, monkeypatch):
    t = {"track_id": 4, "c13": {"estado": "ok", "sincronizacion": {"estado": "sincronizado"}}}
    _validar(tmp_path, monkeypatch, _goes([t]), [{"track_id": 4}])


def test_integridad_rechaza_tormenta_que_el_radar_no_tiene(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="no tiene"):
        _validar(tmp_path, monkeypatch, _goes([{"track_id": 99}]), [{"track_id": 4}])


def test_integridad_rechaza_dato_del_futuro(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="posterior"):
        _validar(tmp_path, monkeypatch, _goes([], estado="sin_tormentas", hora="2026-11-19T22:45:00Z"), [])


def test_integridad_rechaza_dato_sin_sincronizar_usado(tmp_path, monkeypatch):
    t = {"track_id": 4, "c13": {"estado": "ok", "sincronizacion": {"estado": "desfase_excesivo"}}}
    with pytest.raises(SystemExit, match="sin sincronizar"):
        _validar(tmp_path, monkeypatch, _goes([t]), [{"track_id": 4}])


def test_integridad_permite_goes_de_otro_frame_o_ausente(tmp_path, monkeypatch):
    goes = _goes([{"track_id": 99}])
    goes["reference_frame_utc"] = "2026-11-19T22:30:00+00:00"
    _validar(tmp_path, monkeypatch, goes, [])


def test_integridad_rechaza_senal_severa_sin_sincronizar(tmp_path, monkeypatch):
    t = {"track_id": 4, "senales_severas": {"fase": {"estado": "ok", "sincronizacion": {"estado": "desfase_excesivo"}}}}
    with pytest.raises(SystemExit, match="senales_severas.fase"):
        _validar(tmp_path, monkeypatch, _goes([t]), [{"track_id": 4}])


def test_integridad_rechaza_rayos_de_otra_ventana(tmp_path, monkeypatch):
    t = {"track_id": 4, "rayos": {"estado": "ok", "fin_utc": "2026-11-19T22:30:00Z"}}
    with pytest.raises(SystemExit):
        _validar(tmp_path, monkeypatch, _goes([t]), [{"track_id": 4}])
