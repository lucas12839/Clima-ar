import json
from datetime import datetime, timedelta, timezone

import meteorologia_fusion as mf


def preparar(tmp_path, monkeypatch, edad_min):
    amb = tmp_path / "ambiente_futuro.json"
    ahora = datetime.now(timezone.utc)
    amb.write_text(json.dumps({
        "actualizado_utc": (ahora - timedelta(minutes=edad_min)).isoformat(),
        "puntos": [{"punto": "BB_CENTRO", "datos": [{"tiempo_utc": ahora.isoformat(), "temperatura_c": 20}]}],
    }), encoding="utf-8")
    monkeypatch.setattr(mf, "AMBIENTE_FILE", amb)
    monkeypatch.setattr(mf, "FUSION_FILE", tmp_path / "fusion.json")
    monkeypatch.setattr(mf, "cargar_radar", lambda: {"disponible": True, "actividad": False})
    monkeypatch.setattr(mf, "cargar_sazb", lambda: {"disponible": False})
    monkeypatch.setattr(mf, "append_historico", lambda *a: None)
    monkeypatch.setattr(mf, "calcular_calidad", lambda *a, **k: {"score": 50, "nivel": "x", "puntos_ecmwf_validos": 1})
    monkeypatch.setattr(mf, "fusionar_actual_ecmwf_sazb", lambda a, s, now=None: {})
    llamadas = []
    monkeypatch.setattr(mf, "consultar_ecmwf", lambda: llamadas.append(1) or (_ for _ in ()).throw(TimeoutError("caído")))
    return amb, llamadas


def test_pronostico_reciente_no_consulta(tmp_path, monkeypatch):
    amb, llamadas = preparar(tmp_path, monkeypatch, 10)
    antes = amb.read_text()
    assert mf.main() == 0
    f = json.loads((tmp_path / "fusion.json").read_text())
    assert llamadas == [] and f["estado"] == "operativo" and f["ecmwf_cache"]["usado"]
    assert amb.read_text() == antes  # conserva su hora real


def test_pronostico_viejo_consulta_y_usa_respaldo(tmp_path, monkeypatch):
    _, llamadas = preparar(tmp_path, monkeypatch, 90)
    assert mf.main() == 0
    f = json.loads((tmp_path / "fusion.json").read_text())
    assert llamadas == [1] and f["estado"] == "operativo_con_ecmwf_anterior"
    assert f["ecmwf_cache"] == {"usado": False}


def test_nivel_congelamiento_del_perfil_ecmwf():
    import meteorologia_fusion as mf

    # Perfil real de Bahía Blanca, 19/11/2026 23 UTC (Open-Meteo /v1/ecmwf).
    r = {"temperature_2m": 17.5, "temperature_925hPa": 13.2, "geopotential_height_925hPa": 744,
         "temperature_850hPa": 8.8, "geopotential_height_850hPa": 1450,
         "temperature_700hPa": -1.1, "geopotential_height_700hPa": 3034,
         "temperature_600hPa": -8.3, "geopotential_height_600hPa": 4246,
         "temperature_500hPa": -17.7, "geopotential_height_500hPa": 5636}
    assert mf.nivel_congelamiento_m(r, 18) == 2858
    # Bajo cero en superficie: el nivel es el suelo.
    assert mf.nivel_congelamiento_m(dict(r, temperature_2m=-1.0), 18) == 20
    # Todo sobre cero hasta 500 hPa: no se inventa.
    assert mf.nivel_congelamiento_m(dict(r, temperature_500hPa=1.0, temperature_600hPa=3.0,
                                         temperature_700hPa=5.0), 18) is None
    # Sin perfil: None.
    assert mf.nivel_congelamiento_m({"temperature_2m": 10.0}, 18) is None
