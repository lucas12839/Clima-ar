"""La ráfaga de la BCP (24/11/2026): la fila no tiene hora propia y antes se
descartaba siempre."""
from datetime import datetime, timedelta, timezone

from scripts.estaciones import colector_bcp as bcp


def pagina(ultimo):
    return f"""<table>
<tr><td>&Uacute;ltimo Dato Recibido</td><td>{ultimo}</td></tr>
<tr><td>&Uacute;ltima Temperatura</td><td>19.2 &ordm;C</td></tr>
<tr><td>Direcci&oacute;n del Viento</td><td>Este</td></tr>
<tr><td>Velocidad del Viento</td><td>21 km/h</td></tr>
<tr><td>M&aacute;xima R&aacute;faga &uacute;lt. 15 min</td><td>33 km/h</td></tr>
<tr><td>M&aacute;xima R&aacute;faga del d&iacute;a</td><td>73 km/h</td></tr>
<tr><td>Presi&oacute;n</td><td>1003 hPa</td></tr>
<tr><td>Instalada</td><td>22/12/2021 14:00:00</td></tr>
</table>"""


def test_rafaga_de_los_ultimos_15_min(monkeypatch):
    ar = timezone(timedelta(hours=-3))
    ultimo = (datetime.now(timezone.utc) - timedelta(minutes=4)).astimezone(ar).strftime("%d/%m/%Y %H:%M:%S")

    class R:
        text = pagina(ultimo)

        def raise_for_status(self):
            pass

    monkeypatch.setattr(bcp.requests, "get", lambda url, timeout: R())
    r = bcp.collect(104, "BAHIA BLANCA CENTRO")
    v = r["variables"]
    assert (v["viento_kmh"], v["rafaga_15min_kmh"]) == (21.0, 33.0)   # no la del día (73)
    assert r["estado"] == "valida" and r["timestamps"]["rafaga_15min"] == r["observation_time"]
