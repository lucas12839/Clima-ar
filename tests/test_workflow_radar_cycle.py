"""Topologia de workflows: el radar es el reloj, el schedule solo vigila."""

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


def _cron(on):
    return (on or {}).get("schedule") or []


def test_unico_schedule_de_la_cadena_es_el_watchdog_del_detector():
    con_cron = {
        f.name
        for f in WF.glob("*.yml")
        if _cron(_wf(f.name)[1])
    }
    for nombre in (
        "probar-radar-smn.yml",
        "climaar-intelligence.yml",
        "climaar-ciclo-radar.yml",
        "climaar-watchdog.yml",
        "climaar-auto-radar.yml",
        "observaciones-sazb.yml",
    ):
        assert nombre not in con_cron, nombre
    assert "climaar-detector-sinarame.yml" in con_cron


def test_cron_del_detector_es_solo_watchdog():
    data, on, texto = _wf("climaar-detector-sinarame.yml")
    assert "WATCHDOG" in texto
    # el job del detector no despacha el radar por su cuenta: lo hace el script
    assert "gh workflow run" not in texto
    assert data["concurrency"]["group"] == "climaar-detector"
    assert data["concurrency"]["cancel-in-progress"] is False
    assert "--relevo" not in texto
    assert "--servicio-vivo" in texto
    assert "steps.latido.outputs.vivo != 'true'" in texto
    assert "--duracion-max-min 9" in texto
    assert data["jobs"]["watchdog"]["timeout-minutes"] <= 15


def test_radar_idempotente_y_protegido_por_concurrencia():
    data, on, texto = _wf("probar-radar-smn.yml")
    assert "workflow_dispatch" in on and not _cron(on)
    assert data["concurrency"]["group"] == "climaar-production-writes"
    assert data["concurrency"]["cancel-in-progress"] is False
    pasos = data["jobs"]["radar"]["steps"]
    nombres = [p.get("name", "") for p in pasos]
    i = next(i for i, n in enumerate(nombres) if n.startswith("Reclamar frame"))
    assert pasos[i]["id"] == "reclamar"
    # todo lo que procesa/publica depende del reclamo
    for p in pasos[i + 1:]:
        if p["name"].startswith(("Liberar", "Frame ya procesado")):
            continue
        assert "steps.reclamar.outputs.procesar == 'true'" in p["if"], p["name"]
    # orden de la cadena: claim -> radar -> tracking -> completar -> publicar
    orden = [
        "Reclamar frame", "Procesar radar", "Ejecutar seguimiento temporal",
        "Marcar frame como procesado", "Publicar artefacto",
        "Disparar Intelligence", "Persistir resultados",
    ]
    pos = [next(j for j, n in enumerate(nombres) if n.startswith(o)) for o in orden]
    assert pos == sorted(pos)


def test_radar_no_se_auto_dispara_ni_dispara_otros_radares():
    _, _, texto = _wf("probar-radar-smn.yml")
    assert "gh workflow run probar-radar-smn" not in texto
    assert "climaar-ciclo-radar" not in texto
    assert "climaar-intelligence.yml" in texto  # frame nuevo -> IA, mismo run id
    assert "radar_run_id" in texto


def test_solo_el_detector_despacha_el_radar():
    for f in WF.glob("*.yml"):
        texto = f.read_text(encoding="utf-8")
        assert not re.search(
            r"gh workflow run\s+\\?\s*probar-radar-smn\.yml", texto
        ), f.name
    src = (ROOT / "scripts" / "detector_sinarame.py").read_text()
    assert 'WORKFLOW_RADAR = "probar-radar-smn.yml"' in src


def test_ia_sin_cron_propio_y_con_modo_refresco():
    data, on, texto = _wf("climaar-intelligence.yml")
    assert not _cron(on)
    inputs = on["workflow_dispatch"]["inputs"]
    assert inputs["modo"]["default"] == "frame_nuevo"
    assert "refresco" in inputs["modo"]["options"]
    # frame_nuevo sigue exigiendo el run id exacto
    assert "requiere el ID exacto" in texto
    # refresco solo con radar activo completo y validado
    assert "verificar-activo" in texto
    assert data["concurrency"]["group"] == "climaar-production-writes"


def test_ciclo_radar_ya_no_lanza_radar_ni_encadena():
    data, on, texto = _wf("climaar-ciclo-radar.yml")
    assert list(on) == ["workflow_dispatch"]
    assert "gh workflow run probar-radar-smn" not in texto
    assert texto.count("gh workflow run climaar-detector-sinarame.yml") == 1
    assert "gh workflow run climaar-ciclo-radar" not in texto


def test_watchdog_y_auto_radar_no_lanzan_el_radar():
    for nombre in ("climaar-watchdog.yml", "climaar-auto-radar.yml"):
        _, _, texto = _wf(nombre)
        assert not re.search(
            r"gh workflow run\s+\\?\s*probar-radar-smn", texto
        )
        assert "gh workflow run" in texto
        assert "climaar-ciclo-radar.yml" in texto


def test_guardas_de_ejecucion_unica():
    r = lambda i, s: {"databaseId": i, "status": s}  # noqa: E731
    assert not cr.debe_lanzar_radar([r(1, "queued")])
    assert not cr.debe_lanzar_radar([r(1, "in_progress")])
    assert cr.debe_lanzar_radar([r(1, "completed")])
