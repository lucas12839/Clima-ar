"""Control de calidad de cada dato de cada estación y lo medido por localidad (24/11/2026)."""
import json
from datetime import datetime, timedelta, timezone

from scripts.estaciones import senales_superficie as ss
from scripts import modelo_localidades as ml

AHORA = datetime(2026, 11, 24, 22, 0, tzinfo=timezone.utc)


def est(nombre, lat, lon, red="bcp", **v):
    base = {"nombre": nombre, "lat": lat, "lon": lon, "red": red, "utc": AHORA.isoformat(),
            "temperatura_c": 19.0, "viento_kmh": 20.0, "rafaga_kmh": 30.0, "humedad_pct": 40.0,
            "presion_hpa": 1010.0, "direccion_grados": 90.0, "lluvia_1h_mm": 0.0,
            "lluvia_3h_mm": 0.0, "frente_de_rafagas": False}
    base.update(v)
    return base


CENTRO = (-38.7128, -62.2658)
GRUMBEIN = (-38.7561, -62.1065)
PUNTA = (-38.8778, -62.0714)
NOROESTE = (-38.6893, -62.3342)


def test_datos_imposibles_y_politica_cwop():
    es = [est("BAHIA BLANCA CENTRO", *CENTRO, rafaga_kmh=12.0),            # ráfaga < viento
          est("GRUMBEIN", *GRUMBEIN, humedad_pct=0.0),                       # fuera de rango
          est("BAHIA BLANCA NOROESTE (CWOP)", *NOROESTE, red="cwop", viento_kmh=3.2, rafaga_kmh=3.2)]
    anot = ss.controlar(es, {}, AHORA)
    c, g, n = es
    assert c["rafaga_kmh"] is None and c["viento_kmh"] == 20.0
    assert g["humedad_pct"] is None
    assert n["viento_kmh"] is None and n["temperatura_c"] == 19.0
    # La política CWOP no es una falla: no se anota como aviso.
    assert {(a["estacion"], a["dato"]) for a in anot} == {("BAHIA BLANCA CENTRO", "rafaga_kmh"),
                                                          ("GRUMBEIN", "humedad_pct")}


def test_temperatura_lejos_de_las_vecinas():
    es = [est("A", *CENTRO, temperatura_c=19.0), est("B", *GRUMBEIN, temperatura_c=20.0),
          est("C", *NOROESTE, temperatura_c=19.5), est("D", *PUNTA, temperatura_c=35.0)]
    ss.controlar(es, {}, AHORA)
    assert es[3]["temperatura_c"] is None and "vecinas" in es[3]["descartes"]["temperatura_c"]
    assert all(e["temperatura_c"] is not None for e in es[:3])


def test_temperatura_trabada_y_salto():
    def serie(vals):
        return [{"utc": (AHORA - timedelta(minutes=10 * (len(vals) - 1 - i))).isoformat(), "temp": v}
                for i, v in enumerate(vals)]
    h = {"A": serie([18.0] * 20), "B": serie([18 + i * 0.2 for i in range(20)]), "C": serie([19, 19, 27.5])}
    assert ss.temperaturas_trabadas(h, AHORA) == ["A"]
    assert ss.salto_temperatura(h["C"]) and not ss.salto_temperatura(h["B"])


def test_tiempo_ahora_temperatura_propia_y_viento_de_la_zona(tmp_path):
    locs = tmp_path / "loc.json"
    locs.write_text(json.dumps({"localidades": [
        {"id": "punta_alta", "lat": -38.88, "lon": -62.075},
        {"id": "sierra_de_la_ventana", "lat": -38.1378, "lon": -61.7936}]}))
    es = [est("PUNTA ALTA", *PUNTA, viento_kmh=None, rafaga_kmh=None, temperatura_c=18.9),
          est("GRUMBEIN", *GRUMBEIN, viento_kmh=16.0, rafaga_kmh=25.0, temperatura_c=20.0),
          est("SIERRAS GRANDES", -38.1564, -61.9118, temperatura_c=12.0, viento_kmh=13.0)]
    t = ss.tiempo_ahora(es, locs)
    pa = t["punta_alta"]
    assert (pa["temperatura"]["valor"], pa["temperatura"]["estacion"]) == (18.9, "PUNTA ALTA")
    assert (pa["viento"]["valor"], pa["viento"]["rafaga"], pa["viento"]["estacion"]) == (16.0, 25.0, "GRUMBEIN")
    # Sierras Grandes (10,5 km, sobre las sierras) no da la temperatura del pueblo.
    sv = t.get("sierra_de_la_ventana", {})
    # Ni la temperatura ni el viento: está sobre las sierras (25/11/2026).
    assert "temperatura" not in sv and "viento" not in sv


def test_rumbo_en_palabras():
    assert ss.rumbo_grados("Estenordeste") == 67.5
    assert ss.rumbo_grados("Sudsudoeste") == 202.5
    assert ss.rumbo_grados(118.0) == 118.0
    assert ss.rumbo_grados(None) is None


def test_modelo_por_localidad_un_pedido_y_hora_actual():
    pedidos = []

    class R:
        status_code = 200

        def __init__(self, n):
            self.n = n

        def raise_for_status(self):
            pass

        def json(self):
            return [{"hourly": {"time": ["2026-11-24T21:00", "2026-11-24T22:00", "2026-11-24T23:00"],
                                "temperature_2m": [20 + i, 21 + i, 22 + i],
                                "wind_gusts_10m": [30, 31 + i, 32]}} for i in range(self.n)]

    def get(url, params, timeout):
        pedidos.append(params)
        return R(params["latitude"].count(",") + 1)

    m = ml.consultar(AHORA, get=get)
    assert len(pedidos) == 1 and len(m["localidades"]) == 7
    v = ml.valor_actual(m, "tornquist", AHORA)
    assert v["hora_utc"] == "2026-11-24T22:00" and v["rafagas"] == 31 + 4
    assert ml.valor_actual(m, "tornquist", AHORA + timedelta(hours=5)) is None      # fuera de la ventana
    assert ml.vigente(m, AHORA) and not ml.vigente(m, AHORA + timedelta(hours=6))


def test_modelo_caido_no_corta():
    def get(url, params, timeout):
        raise ConnectionError("sin red")
    assert ml.consultar(AHORA, get=get) is None


def test_viento_y_rafaga_maximos_como_la_tarjeta(tmp_path):
    locs = tmp_path / "loc.json"
    locs.write_text(json.dumps({"localidades": [{"id": "bahia_blanca", "lat": -38.7183, "lon": -62.2663}]}))
    es = [est("BAHIA BLANCA CENTRO", *CENTRO, viento_kmh=10.0, rafaga_kmh=30.0),
          est("GRUMBEIN", *GRUMBEIN, viento_kmh=15.0, rafaga_kmh=23.0)]
    v = ss.tiempo_ahora(es, locs)["bahia_blanca"]["viento"]
    r = ss.resumir(es)
    assert (v["valor"], v["rafaga"]) == (r["viento_max_kmh"], r["rafaga_max_kmh"]) == (15.0, 30.0)
    assert (v["estacion"], v["rafaga_estacion"]) == ("GRUMBEIN", "BAHIA BLANCA CENTRO")
    # Una ráfaga menor que el viento que se muestra no se muestra.
    es[0]["rafaga_kmh"], es[1]["rafaga_kmh"] = 12.0, None
    assert ss.tiempo_ahora(es, locs)["bahia_blanca"]["viento"]["rafaga"] is None
