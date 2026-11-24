from datetime import datetime, timezone

import recoleccion_radares as rr

AHORA = datetime(2026, 11, 24, 2, 13, tzinfo=timezone.utc)  # 23:13 hora argentina
GB = rr.GB


def png(n=5000):
    return rr.PNG_FIRMA + b"x" * n + b"\x00\x00\x00\x00IEND\xaeB`\x82"


def test_turnos_cada_10_min_en_hora_argentina():
    t = list(rr.turnos(AHORA))
    assert t[0].strftime("%Y%m%d%H%M%S") == "20261123231000"
    assert t[1].strftime("%H%M") == "2300"
    assert len(t) == 18  # 3 horas (23:10 a 20:20)


def test_guarda_crudo_y_no_repite():
    pedidas = []

    def bajar(url):
        pedidas.append(url)
        return png() if "231000" in url or "230000" in url else None

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        e = rr.recolectar(d, {"RMA4": "Resistencia"}, AHORA, bajar, libre_fn=lambda: 10 * GB, pausa=0)
        assert e["radares"]["RMA4"]["nuevas_ultima_vuelta"] == 2
        assert e["radares"]["RMA4"]["ultima"] == "2026-11-23 23:10"
        assert (rr.ruta_imagen(d, "RMA4", list(rr.turnos(AHORA))[0])).read_bytes() == png()
        assert "cache/RMA4/20261123231000.png" in pedidas[0]
        pedidas.clear()
        e = rr.recolectar(d, {"RMA4": "Resistencia"}, AHORA, bajar, libre_fn=lambda: 10 * GB, pausa=0)
        assert e["radares"]["RMA4"]["nuevas_ultima_vuelta"] == 0
        assert e["radares"]["RMA4"]["imagenes"] == 2
        assert not any("231000" in u for u in pedidas)  # ya guardada: no se vuelve a pedir


def test_rechaza_imagen_cortada_y_sigue_con_errores_de_red(tmp_path):
    def bajar(url):
        if "231000" in url:
            return png()[:-6]  # cortada
        if "230000" in url:
            raise OSError("sin red")
        return None

    e = rr.recolectar(tmp_path, {"RMA1": "Córdoba"}, AHORA, bajar, libre_fn=lambda: 10 * GB, pausa=0)
    info = e["radares"]["RMA1"]
    assert info["nuevas_ultima_vuelta"] == 0
    assert len(info["errores_ultima_vuelta"]) == 2
    assert not list(tmp_path.rglob("*.png"))


def test_se_frena_con_poco_disco(tmp_path):
    e = rr.recolectar(tmp_path, {"RMA1": "Córdoba"}, AHORA, lambda u: png(),
                      libre_fn=lambda: 2 * GB, pausa=0)
    assert "libres" in e["detenido"]
    assert not list(tmp_path.rglob("*.png"))


def test_se_frena_al_llegar_al_tope(tmp_path):
    # tope chiquito: entra una sola imagen
    e = rr.recolectar(tmp_path, {"RMA1": "Córdoba", "RMA4": "Resistencia"}, AHORA,
                      lambda u: png(), tope_gb=5000 / GB, libre_fn=lambda: 10 * GB, pausa=0)
    assert "tope" in e["detenido"]
    assert len(list(tmp_path.rglob("*.png"))) == 1


def test_sin_imagenes_nuevas_no_falla(tmp_path):
    e = rr.recolectar(tmp_path, {"RMA1": "Córdoba"}, AHORA, lambda u: None,
                      libre_fn=lambda: 10 * GB, pausa=0)
    assert e["detenido"] is None
    assert e["radares"]["RMA1"]["imagenes"] == 0
    assert (tmp_path / "estado.json").exists()


def test_mascaras_recoleccion_eco_fijo(tmp_path):
    """Un eco que está siempre en el mismo lugar queda en la máscara; la lluvia que pasa, no."""
    import numpy as np
    from PIL import Image
    from datetime import datetime, timedelta, timezone

    import scripts.mascaras_recoleccion as mr

    carpeta = tmp_path / "RMA1" / "20261124"
    carpeta.mkdir(parents=True)
    t0 = datetime(2026, 11, 24, tzinfo=timezone.utc)
    for i in range(mr.MIN_IMAGENES):
        rgba = np.zeros((60, 60, 4), dtype=np.uint8)
        rgba[10:14, 10:14] = (200, 200, 0, 255)          # eco fijo
        x = i % 50
        rgba[40:44, x:x + 4] = (0, 200, 0, 255)          # lluvia que se mueve
        nombre = (t0 + timedelta(minutes=10 * i)).strftime("%Y%m%d%H%M%S")
        Image.fromarray(rgba, mode="RGBA").save(carpeta / f"{nombre}.png")
    resumen = mr.procesar(tmp_path, ahora=t0)
    assert "máscara nueva" in resumen["RMA1"]
    with Image.open(tmp_path / "RMA1" / "mascara_ecos_fijos.png") as img:
        m = np.asarray(img) > 0
    assert m[12, 12]
    assert not m[42, 20]
    # Al día: no se rehace antes de 24 h.
    assert mr.procesar(tmp_path, ahora=t0 + timedelta(hours=1))["RMA1"] == "al día"


def test_mascaras_recoleccion_faltan_imagenes(tmp_path):
    import scripts.mascaras_recoleccion as mr

    (tmp_path / "RMA4" / "20261124").mkdir(parents=True)
    assert mr.procesar(tmp_path)["RMA4"].startswith("faltan imágenes")
