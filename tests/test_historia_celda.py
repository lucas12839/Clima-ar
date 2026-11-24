"""Historia de cada tormenta (24/11/2026, sugerencia de Alex)."""
import os

os.environ.setdefault("CLIMAAR_REMOTE_DATA_ENABLED", "0")
import app  # noqa: E402

KMPX = app._km_por_pixel()


def punto(minuto, x, y, intensidad, pixeles=100):
    return {"timestamp": f"2026-11-24T15:{minuto:02d}:00Z", "x": x, "y": y,
            "intensidad": intensidad, "pixeles": pixeles}


def test_historia_completa():
    paso = 10 / KMPX                     # 10 km por imagen (60 km/h)
    h = [punto(m, 200 + i * paso, 300, n, 100 + i * 20)
         for i, (m, n) in enumerate([(0, 2), (10, 4), (20, 5), (30, 3)])]
    r = app._historia_celda(h)
    assert r["minutos"] == 30 and r["imagenes"] == 4
    assert r["recorrido_km"] == 30
    assert (r["intensidad_inicio"], r["intensidad_max"]) == (2, 5)
    assert r["intensidad_max_utc"].startswith("2026-11-24T15:20")
    assert r["desde_utc"].startswith("2026-11-24T15:00")


def test_salto_imposible_corta_la_historia():
    paso = 10 / KMPX
    h = [punto(0, 50, 50, 6),                       # otra tormenta lejos (heredada)
         punto(10, 300, 300, 2), punto(20, 300 + paso, 300, 3)]
    r = app._historia_celda(h)
    assert r["imagenes"] == 2 and r["intensidad_max"] == 3       # no se mezcla el 6 heredado


def test_una_sola_imagen_sin_historia():
    assert app._historia_celda([punto(0, 300, 300, 3)]) is None
    assert app._historia_celda([]) is None
