"""23/11/2026: la IA de 4 imágenes no se publicó porque la validación comparaba
la trayectoria del 5b contra el predictor viejo. Con una tormenta recién nacida
el viejo dice SIN_MOVIMIENTO_VALIDO y el 5b sí tiene movimiento: es válido."""

import json
import shutil
from pathlib import Path

import pytest

import scripts.validar_integridad_climaar as v

RAIZ = Path(__file__).resolve().parents[1]


def _copia(tmp_path):
    for sub in ("data/radar", "data/ia", "data/goes19", "data/meteorologia", "data/sazb", "config"):
        origen = RAIZ / sub
        if origen.exists():
            shutil.copytree(origen, tmp_path / sub, ignore=shutil.ignore_patterns("historico", "*.png"))
    return tmp_path


def _apuntar(monkeypatch, base):
    monkeypatch.setattr(v, "BASE", base)
    monkeypatch.setattr(v, "RADAR", base / "data" / "radar")
    monkeypatch.setattr(v, "TRACK", base / "data" / "radar" / "seguimiento")
    monkeypatch.setattr(v, "IA", base / "data" / "ia")


def _principal(base):
    nc = json.loads((base / "data/radar/radar_nowcast.json").read_text())["nowcast"]
    return nc, nc.get("nucleo_principal_id")


def _viejo_sin_movimiento(base, principal):
    ruta = base / "data/radar/seguimiento/prediccion_evolucion.json"
    ev = json.loads(ruta.read_text())
    for n in ev.get("nucleos", []):
        if str(n.get("track_id")) == str(principal):
            n["pronostico_disponible"] = False
            n["motivo"] = "SIN_MOVIMIENTO_VALIDO"
            n.pop("horizontes", None)
    ruta.write_text(json.dumps(ev))


def test_tormenta_nueva_con_trayectoria_del_5b_es_valida(tmp_path, monkeypatch):
    base = _copia(tmp_path)
    nc, principal = _principal(base)
    if principal is None or (nc.get("proyecciones_metadata") or {}).get("fuente") != "nowcast_capas":
        pytest.skip("los datos actuales no tienen tormenta con trayectoria del 5b")
    _viejo_sin_movimiento(base, principal)
    _apuntar(monkeypatch, base)
    v.main()  # antes: "pronostico_disponible no coincide con prediccion_evolucion"


def test_5b_sin_pronostico_sigue_fallando(tmp_path, monkeypatch):
    base = _copia(tmp_path)
    nc, principal = _principal(base)
    if principal is None or (nc.get("proyecciones_metadata") or {}).get("fuente") != "nowcast_capas":
        pytest.skip("los datos actuales no tienen tormenta con trayectoria del 5b")
    ruta = base / "data/radar/radar_nowcast.json"
    datos = json.loads(ruta.read_text())
    datos["nowcast"]["proyecciones_metadata"]["pronostico_disponible"] = False
    ruta.write_text(json.dumps(datos))
    _apuntar(monkeypatch, base)
    with pytest.raises(SystemExit):
        v.main()


def _envejecer(base, rel, ruta):
    p = base / rel
    if not p.exists():
        return False
    d = json.loads(p.read_text())
    obj = d
    for k in ruta[:-1]:
        obj = obj[k]
    obj[ruta[-1]] = "2026-01-01T00:00:00Z"
    p.write_text(json.dumps(d))
    return True


def test_pasos_en_paralelo_que_fallan_no_cortan_el_ciclo(tmp_path, monkeypatch):
    """Re-auditoría 23/11/2026: un archivo viejo de un paso que falló cortaba toda la IA."""
    base = _copia(tmp_path)
    for rel, ruta in (("data/ia/indicios_severos.json", ("radar_timestamp",)),
                      ("data/ia/celdas_localidad.json", ("radar_timestamp",)),
                      ("data/ia/ambiente_satelite.json", ("frame_radar_utc",))):
        _envejecer(base, rel, ruta)
    _apuntar(monkeypatch, base)
    v.main()


def test_5b_viejo_vale_solo_si_lo_que_se_ve_salio_del_respaldo(tmp_path, monkeypatch):
    base = _copia(tmp_path)
    if not _envejecer(base, "data/radar/nowcast_capas.json", ("radar_timestamp",)):
        pytest.skip("sin nowcast en capas en los datos actuales")
    final_p = base / "data/ia/resultado_final.json"
    final = json.loads(final_p.read_text())
    _apuntar(monkeypatch, base)
    if (final.get("tipo_tiempo") or {}).get("fuente_llegadas") == "nowcast_capas":
        with pytest.raises(SystemExit):
            v.main()   # dice que usó el 5b, pero el 5b es de otra imagen
    final.setdefault("tipo_tiempo", {})["fuente_llegadas"] = "flujo_optico"
    final_p.write_text(json.dumps(final))
    v.main()
