"""Pronóstico por localidad y modelo de respaldo de la página (24/11/2026)."""
import os

os.environ.setdefault("CLIMAAR_REMOTE_DATA_ENABLED", "0")
import app  # noqa: E402


def test_openmeteo_un_solo_pedido_para_las_7(monkeypatch):
    pedidos = []

    class Resp:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    def get(url, timeout):
        pedidos.append(url)
        n = url.split("latitude=")[1].split("&")[0].count(",") + 1
        if "current=" in url:
            return Resp([{"current": {"temperature_2m": 10 + i}} for i in range(n)])
        return Resp([{"daily": {"time": ["2026-11-24", "2026-11-25"], "temperature_2m_max": [20 + i, 21 + i]}}
                     for i in range(n)])

    monkeypatch.setattr(app.requests, "get", get)
    for c in app._LOCALES_CACHE.values():
        c.update(ts=0.0, data=None, fallo=0.0)
    act = app.get_openmeteo_locales()
    assert act["bahia_blanca"]["temperatura"] == 10 and act["sierra_de_la_ventana"]["temperatura"] == 16
    app.get_openmeteo_locales()                            # guardado: no pide de nuevo
    pron = app.get_openmeteo_forecast_locales()
    assert pron["tornquist"][1]["maxima"] == 25
    assert len(pedidos) == 2
    for c in app._LOCALES_CACHE.values():
        c.update(ts=0.0, data=None, fallo=0.0)


def test_openmeteo_caido_no_rompe(monkeypatch):
    def get(url, timeout):
        raise app.requests.ConnectionError("sin red")

    monkeypatch.setattr(app.requests, "get", get)
    for c in app._LOCALES_CACHE.values():
        c.update(ts=0.0, data=None, fallo=0.0)
    assert app.get_openmeteo_locales() == {}
    assert app.get_openmeteo_forecast_locales() == {}


def test_modelo_caido_no_se_reintenta_en_cada_pedido(monkeypatch):
    intentos = []

    def get(url, timeout):
        intentos.append(url)
        raise app.requests.ConnectionError("sin red")

    monkeypatch.setattr(app.requests, "get", get)
    for c in app._LOCALES_CACHE.values():
        c.update(ts=0.0, data=None, fallo=0.0)
    monkeypatch.setitem(app._CURRENT_CACHE, "fallo", 0.0)
    monkeypatch.setitem(app._CURRENT_CACHE, "data", None)
    monkeypatch.setitem(app._CURRENT_CACHE, "ts", 0.0)
    for _ in range(5):
        app.get_openmeteo_locales()
        app.get_openmeteo_data()
    assert len(intentos) == 2          # uno de cada uno; los demás esperan el minuto
    for c in app._LOCALES_CACHE.values():
        c.update(ts=0.0, data=None, fallo=0.0)
    app._CURRENT_CACHE["fallo"] = 0.0

