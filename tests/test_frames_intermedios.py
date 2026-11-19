from datetime import datetime, timezone
import json

from scraper import capturar_radar_smn as sc


class Resp:
    def __init__(self, ok, body=b""):
        self.ok, self.status, self._b = ok, 200 if ok else 404, body
    def body(self):
        return self._b


class Pagina:
    def __init__(self, existentes):
        self.existentes, self.pedidas = existentes, []
        self.request = self
    def get(self, url, timeout=0):
        self.pedidas.append(url)
        return Resp(any(e in url for e in self.existentes), b"png")


def test_url_frame_usa_hora_local():
    url = sc.url_frame(datetime(2026, 11, 19, 8, 20, tzinfo=timezone.utc))
    assert url.endswith("cache/RMA10/20261119052000.png")


def test_recupera_solo_los_huecos(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "HIST", tmp_path)
    monkeypatch.setattr(sc, "PENDIENTES", tmp_path / "p.json")
    monkeypatch.setattr(sc, "validar_png", lambda d: (True, "ok"))
    ultimo = datetime(2026, 11, 19, 8, 0, tzinfo=timezone.utc)
    pagina = Pagina(existentes=["052000", "053000"])  # 08:20 y 08:30 UTC; 08:10 no existe
    r = sc.recuperar_intermedios(pagina, "20261119054000", ultimo=ultimo)
    assert [x["utc"][11:16] for x in r] == ["08:20", "08:30"]
    assert len(pagina.pedidas) == 3
    assert json.loads((tmp_path / "p.json").read_text()) == r


def test_sin_hueco_no_pide_nada(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "HIST", tmp_path)
    monkeypatch.setattr(sc, "PENDIENTES", tmp_path / "p.json")
    pagina = Pagina(existentes=[])
    ultimo = datetime(2026, 11, 19, 8, 30, tzinfo=timezone.utc)
    assert sc.recuperar_intermedios(pagina, "20261119054000", ultimo=ultimo) == []
    assert pagina.pedidas == []
