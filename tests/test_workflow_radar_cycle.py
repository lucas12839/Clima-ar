"""Topologia de workflows: el radar es el reloj, el schedule solo vigila."""

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows"
sys.path.insert(0, str(ROOT / "scripts"))


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
        # 23/11/2026: la IA se dispara después de guardar el radar en main.
        "Persistir resultados", "Disparar Intelligence",
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


def test_workflows_viejos_borrados():
    # Auditoría 20/11/2026: eran manuales y no se usaban.
    for nombre in ("climaar-ciclo-radar.yml", "climaar-watchdog.yml", "climaar-auto-radar.yml"):
        assert not (WF / nombre).exists(), nombre


def test_reloj_encadenado_cwop_vigila_al_detector_y_sigue_solo():
    # GitHub corre los relojes de 10 min solo ~3 veces por día: la cadena se
    # relanza sola y en cada vuelta pide al vigilante que mire el latido.
    data, on, texto = _wf("climaar-cwop.yml")
    assert "workflow_dispatch" in on and _cron(on)
    assert data["concurrency"]["group"] == "climaar-reloj-cwop"
    assert data["concurrency"]["cancel-in-progress"] is False
    jobs = data["jobs"]
    assert "gh workflow run climaar-detector-sinarame.yml" in str(jobs["vigilar-detector"])
    sig = jobs["siguiente-vuelta"]
    assert sig["if"] == "always()"
    assert set(sig["needs"]) == {"vigilar-detector", "escuchar", "publicar"}
    assert "gh workflow run climaar-cwop.yml" in str(sig)
    assert "480" in str(sig)  # vuelta mínima de 8 min
    # El límite del trabajo tiene que dejar terminar esa espera (primera vuelta
    # real del 20/11: cortado a los 3 min y la cadena se detuvo).
    assert sig["timeout-minutes"] * 60 > 480 + 60
    assert data["permissions"]["actions"] == "write"
    # 23/11/2026: fuera del grupo de Radar/IA (el tercero que llega cancela al
    # que espera). Solo escribe su archivo y publica con rebase y reintentos.
    assert "concurrency" not in jobs["publicar"]
    assert "for intento in 1 2 3 4 5 6" in str(jobs["publicar"])
    # nunca lanza el radar
    assert not re.search(r"gh workflow run\s+\\?\s*probar-radar-smn", texto)


def test_checkout_corto_y_python_unico():
    # Auditoría 20/11/2026: bajar la historia completa (~540 MB) costaba 25-36 s.
    for f in WF.glob("*.yml"):
        texto = f.read_text(encoding="utf-8")
        assert "fetch-depth: 0" not in texto, f.name
        assert "lfs: true" not in texto, f.name
        for v in re.findall(r'python-version:\s*"([\d.]+)"', texto):
            assert v == "3.12", f.name


def test_solo_radar_ia_y_sazb_comparten_la_fila_de_escritura():
    # 23/11/2026: GitHub admite 1 en curso + 1 en espera por grupo; un tercero
    # cancela al que espera. Nada que corra solo cada pocos minutos puede entrar.
    permitidos = {"probar-radar-smn.yml", "climaar-intelligence.yml", "observaciones-sazb.yml"}
    for f in WF.glob("*.yml"):
        if "climaar-production-writes" in f.read_text(encoding="utf-8"):
            assert f.name in permitidos, f.name
