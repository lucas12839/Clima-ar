"""Garantias de que los ciclos del Radar no pueden acumularse."""

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows"
sys.path.insert(0, str(ROOT / "scripts"))

import ciclo_radar as cr  # noqa: E402


def _wf(nombre):
    texto = (WF / nombre).read_text(encoding="utf-8")
    data = yaml.safe_load(texto)
    on = data.get("on", data.get(True))
    return data, on, texto


def _schedules(on):
    return (on or {}).get("schedule") or []


def test_radar_no_tiene_cron_ni_encadena():
    data, on, texto = _wf("probar-radar-smn.yml")
    assert not _schedules(on)
    assert "workflow_dispatch" in on
    assert "gh workflow run climaar-ciclo-radar" not in texto
    assert "gh workflow run probar-radar-smn" not in texto


def test_radar_mantiene_concurrencia_de_escritura():
    data, _, _ = _wf("probar-radar-smn.yml")
    assert data["concurrency"]["group"] == "climaar-production-writes"
    assert data["concurrency"]["cancel-in-progress"] is False


def test_solo_el_ciclo_lanza_el_radar():
    for f in WF.glob("*.yml"):
        texto = f.read_text(encoding="utf-8")
        lanza = re.search(
            r"gh workflow run\s+\\?\s*probar-radar-smn\.yml", texto
        )
        if f.name == "climaar-ciclo-radar.yml":
            assert lanza
        else:
            assert not lanza, f.name


def test_un_solo_cron_para_el_radar_y_solo_en_el_ciclo():
    con_cron = []
    for f in WF.glob("*.yml"):
        _, on, texto = _wf(f.name)
        if _schedules(on) and (
            "probar-radar-smn.yml" in texto
            or f.name == "climaar-ciclo-radar.yml"
        ):
            con_cron.append(f.name)
    assert con_cron == ["climaar-ciclo-radar.yml"]


def test_ciclo_sin_grupo_de_concurrencia_pendiente():
    data, on, texto = _wf("climaar-ciclo-radar.yml")
    # Un grupo dejaria una ejecucion pendiente = segundo ciclo.
    assert "concurrency" not in data
    pasos = data["jobs"]["ciclo"]["steps"]
    assert pasos[1]["id"] == "unico"
    for p in pasos[2:]:
        assert "steps.unico.outputs.continuar == 'true'" in p["if"]
    assert texto.count("gh workflow run climaar-ciclo-radar.yml") == 1
    assert texto.count("gh workflow run probar-radar-smn.yml") == 1
    assert data["jobs"]["ciclo"]["timeout-minutes"] <= 15


# ---- simulacion de la logica de decision ---------------------------------

def _r(i, status):
    return {"databaseId": i, "status": status}


def test_duplicado_mas_nuevo_se_descarta():
    runs = [_r(10, "in_progress"), _r(11, "in_progress")]
    assert cr.debe_continuar_ciclo(runs, 10)
    assert not cr.debe_continuar_ciclo(runs, 11)


def test_radar_no_se_duplica_en_cola_ni_ejecucion():
    assert not cr.debe_lanzar_radar([_r(1, "queued")])
    assert not cr.debe_lanzar_radar([_r(1, "in_progress")])
    assert cr.debe_lanzar_radar([_r(1, "completed")])


def test_no_encadena_si_hay_otro_ciclo():
    assert not cr.debe_encadenar([_r(5, "in_progress"), _r(6, "queued")], 5)
    assert cr.debe_encadenar([_r(5, "in_progress")], 5)


def test_simulacion_no_acumula_ciclos():
    """Ciclos por cron + cadena nunca dejan mas de 1 ciclo vivo."""
    import random

    rng = random.Random(7)
    vivos = {}  # id -> fase
    siguiente = 100
    radar_activos = 0
    for paso in range(2000):
        evento = rng.choice(["cron", "cron", "avanzar", "avanzar"])
        runs = [_r(i, "in_progress") for i in vivos]
        if evento == "cron":
            siguiente += 1
            vivos[siguiente] = "nuevo"
        for i in list(vivos):
            runs = [_r(j, "in_progress") for j in vivos]
            if vivos[i] == "nuevo":
                if cr.debe_continuar_ciclo(runs, i):
                    vivos[i] = "espera"
                    if cr.debe_lanzar_radar(
                        [_r(0, "in_progress")] * radar_activos
                    ):
                        radar_activos = 1
                else:
                    del vivos[i]
            elif vivos[i] == "espera" and evento == "avanzar":
                radar_activos = 0
                if cr.debe_encadenar(runs, i):
                    siguiente += 1
                    vivos[siguiente] = "nuevo"
                del vivos[i]
        esperando = [i for i, f in vivos.items() if f == "espera"]
        assert len(esperando) <= 1
        assert radar_activos <= 1
