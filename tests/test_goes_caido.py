import json
from datetime import datetime, timezone

import tormenta_intelligence_timestamp as ti


def test_goes_no_disponible_no_corta_el_ciclo(tmp_path, monkeypatch):
    intel = tmp_path / "inteligencia.json"
    radar = tmp_path / "radar.json"
    intel.write_text(json.dumps({"indices_heuristicos": {"tormenta": 0.4}}), encoding="utf-8")
    radar.write_text(json.dumps({"nowcast": {"direccion_grados": 90}}), encoding="utf-8")
    monkeypatch.setattr(ti, "INTELLIGENCE_FILE", intel)
    monkeypatch.setattr(ti, "RADAR_FILE", radar)
    ts = datetime(2026, 11, 19, 9, 0, tzinfo=timezone.utc)
    goes = {  # lo que arma el ciclo cuando GOES-19 no responde
        "reference_frame": {"frame_actual_utc": "2026-11-19T09:00:00Z"},
        "acquisition": "unavailable",
        "senales": {"conveccion_satelital": 0.0, "calidad_goes": 0.0},
    }
    ti.apply_goes_to_intelligence(goes, ts)
    salida = json.loads(intel.read_text(encoding="utf-8"))
    assert salida["indices_heuristicos"]["tormenta"] == 0.4
    assert salida["goes19"]["senales"]["dmw_viento_guia"]["radar_direction_agreement"] is None
