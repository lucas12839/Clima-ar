#!/usr/bin/env python3
"""Validador integral de la cadena operativa ClimaAR.

No selecciona datos ni modifica archivos. Falla ante cualquier
desincronización, cambio de principal o inclusión de ecos estáticos.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
RADAR = BASE / "data" / "radar"
TRACK = RADAR / "seguimiento"
IA = BASE / "data" / "ia"


def fail(message: str) -> None:
    raise SystemExit(f"ERROR INTEGRIDAD CLIMAAR: {message}")


def load(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"archivo faltante o vacío: {path.relative_to(BASE)}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"JSON inválido {path.relative_to(BASE)}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON raíz no es objeto: {path.relative_to(BASE)}")
    return value


def parse_utc(value, name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        fail(f"{name} no contiene timestamp")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        fail(f"{name} tiene timestamp inválido: {value!r}")
    if dt.tzinfo is None:
        fail(f"{name} no tiene zona horaria: {value!r}")
    return dt.astimezone(timezone.utc)


def sid(value):
    if value is None:
        return None
    return str(value)


def require_same_timestamp(expected: datetime, value, name: str) -> None:
    actual = parse_utc(value, name)
    if actual != expected:
        fail(
            f"desincronización {name}: "
            f"{actual.isoformat()} != {expected.isoformat()}"
        )


def find_track(items, track_id, name):
    wanted = sid(track_id)
    for item in items:
        if isinstance(item, dict) and sid(item.get("track_id")) == wanted:
            return item
    fail(f"principal track {wanted} no existe en {name}")


def main() -> None:
    radar_ts = parse_utc(
        (RADAR / "radar_timestamp.txt").read_text(encoding="utf-8").strip(),
        "radar_timestamp.txt",
    )

    processing = load(RADAR / "resultado_procesamiento.json")
    tracking = load(TRACK / "seguimiento_temporal.json")
    state = load(TRACK / "estado.json")
    nowcast_file = load(RADAR / "radar_nowcast.json")
    ml = load(TRACK / "prediccion_ml_rma10.json")
    evolution = load(TRACK / "prediccion_evolucion.json")
    meteo = load(IA / "estado_meteorologico.json")
    intel = load(IA / "inteligencia_tormenta.json")
    final = load(IA / "resultado_final.json")

    nc = nowcast_file.get("nowcast")
    if not isinstance(nc, dict):
        fail("radar_nowcast.json no contiene nowcast")

    require_same_timestamp(radar_ts, nowcast_file.get("actualizado"), "nowcast.actualizado")
    require_same_timestamp(radar_ts, nc.get("actualizado"), "nowcast.nowcast.actualizado")
    require_same_timestamp(radar_ts, nc.get("frame_actual_utc"), "nowcast.frame_actual_utc")
    require_same_timestamp(radar_ts, tracking.get("radar_timestamp"), "tracking.radar_timestamp")
    require_same_timestamp(radar_ts, state.get("ultimo_radar_timestamp"), "estado.ultimo_radar_timestamp")
    require_same_timestamp(radar_ts, ml.get("frame_radar"), "ML.frame_radar")
    require_same_timestamp(radar_ts, evolution.get("radar_timestamp"), "evolution.radar_timestamp")

    radar_meteo = meteo.get("radar", {})
    if not isinstance(radar_meteo, dict):
        fail("estado_meteorologico.radar inválido")
    require_same_timestamp(radar_ts, radar_meteo.get("frame_actual_utc"), "meteorologia.radar.frame_actual_utc")

    radar_intel = intel.get("radar", {})
    if not isinstance(radar_intel, dict):
        fail("inteligencia.radar inválido")
    require_same_timestamp(radar_ts, radar_intel.get("frame_actual_utc"), "intelligence.radar.frame_actual_utc")

    final_radar = final.get("radar", {})
    if not isinstance(final_radar, dict):
        fail("resultado_final.radar inválido")
    final_nc = final_radar.get("nowcast", {})
    if not isinstance(final_nc, dict):
        fail("resultado_final.radar.nowcast inválido")
    require_same_timestamp(radar_ts, final_nc.get("frame_actual_utc"), "resultado_final.radar.nowcast.frame_actual_utc")

    candidate = tracking.get("track_principal_candidato_id")
    if candidate is None:
        fail("Tracking no definió track_principal_candidato_id")

    principal = nc.get("nucleo_principal_id")
    track_principal = nc.get("track_principal_id")
    candidate_nowcast = nc.get("track_principal_candidato_id")

    if principal is None:
        fail("Nowcast no tiene nucleo_principal_id")
    if sid(principal) != sid(candidate):
        fail(f"principal Tracking={candidate} != Nowcast={principal}")
    if sid(track_principal) != sid(candidate):
        fail(f"track_principal_id={track_principal} != Tracking={candidate}")
    if sid(candidate_nowcast) != sid(candidate):
        fail(f"Nowcast track_principal_candidato_id={candidate_nowcast} != Tracking={candidate}")

    bahia = nc.get("bahia_blanca", {})
    if not isinstance(bahia, dict):
        fail("nowcast.bahia_blanca inválido")
    if sid(bahia.get("nucleo_principal_id")) != sid(candidate):
        fail(
            "Bahía Blanca principal="
            f"{bahia.get('nucleo_principal_id')} != Tracking={candidate}"
        )

    state_track = find_track(state.get("tracks", []), candidate, "estado.json")
    if state_track.get("activo") is not True:
        fail(f"principal {candidate} no está activo en estado.json")
    if state_track.get("presente_en_ultimo_frame") is not True:
        fail(f"principal {candidate} no está presente en el último frame")

    nuclei = nc.get("nucleos")
    if not isinstance(nuclei, list) or not nuclei:
        fail("Nowcast no contiene núcleos operativos")
    principal_nucleus = find_track(nuclei, candidate, "nowcast.nucleos")

    geo_nuclei = nc.get("nucleos_georreferenciados")
    if not isinstance(geo_nuclei, list):
        fail("Nowcast no contiene nucleos_georreferenciados")
    find_track(geo_nuclei, candidate, "nowcast.nucleos_georreferenciados")

    for item in nuclei:
        if not isinstance(item, dict):
            fail("nowcast.nucleos contiene un elemento inválido")
        if str(item.get("estado", "")).upper() == "ECO_ESTATICO":
            fail("un ECO_ESTATICO llegó a nowcast.nucleos")

    dynamic_count = processing.get("nucleos_dinamicos")
    if isinstance(dynamic_count, int) and dynamic_count != len(nuclei):
        fail(
            f"nucleos_dinamicos={dynamic_count} != "
            f"nowcast.nucleos={len(nuclei)}"
        )

    ml_predictions = ml.get("predicciones")
    if not isinstance(ml_predictions, list):
        fail("ML.predicciones no es lista")
    ml_principal = find_track(ml_predictions, candidate, "ML.predicciones")
    if sid(ml_principal.get("track_id")) != sid(candidate):
        fail("ML principal inconsistente")

    evolution_nuclei = evolution.get("nucleos")
    if not isinstance(evolution_nuclei, list):
        fail("evolution.nucleos no es lista")
    evolution_track = find_track(evolution_nuclei, candidate, "prediccion_evolucion.nucleos")
    if evolution_track.get("pronostico_disponible") is not True:
        fail(f"evolution principal {candidate} no tiene pronóstico disponible")

    metadata = nc.get("proyecciones_metadata")
    if not isinstance(metadata, dict):
        fail("nowcast.proyecciones_metadata inválido")
    if sid(metadata.get("track_id")) != sid(candidate):
        fail(f"proyecciones_metadata.track_id={metadata.get('track_id')} != {candidate}")

    projections = nc.get("proyecciones")
    if not isinstance(projections, dict):
        fail("nowcast.proyecciones inválido")
    for horizon in ("5", "10", "20", "30"):
        projection = projections.get(horizon)
        if not isinstance(projection, dict):
            fail(f"falta proyección {horizon}")
        if sid(projection.get("track_id")) != sid(candidate):
            fail(
                f"proyección {horizon} track_id={projection.get('track_id')} "
                f"!= principal={candidate}"
            )
        projection_ts = parse_utc(
            projection.get("timestamp_utc"),
            f"proyección {horizon}.timestamp_utc",
        )
        expected_projection_ts = radar_ts + timedelta(minutes=int(horizon))
        if projection_ts != expected_projection_ts:
            fail(
                f"proyección {horizon} desincronizada: "
                f"{projection_ts.isoformat()} != "
                f"{expected_projection_ts.isoformat()}"
            )

    if sid(radar_intel.get("nucleo_principal_id")) != sid(candidate):
        fail(
            f"intelligence principal={radar_intel.get('nucleo_principal_id')} "
            f"!= {candidate}"
        )

    final_principal = final.get("principal", {})
    if not isinstance(final_principal, dict):
        fail("resultado_final.principal inválido")
    if sid(final_principal.get("track_id")) != sid(candidate):
        fail(
            f"resultado_final principal={final_principal.get('track_id')} "
            f"!= {candidate}"
        )

    if sid(radar_meteo.get("nucleo_principal_id")) != sid(candidate):
        fail(
            f"meteorologia principal={radar_meteo.get('nucleo_principal_id')} "
            f"!= {candidate}"
        )

    if not isinstance(final.get("severidad_meteorologica"), dict):
        fail("resultado_final.severidad_meteorologica inválido")

    print("CLIMAAR INTEGRIDAD: OK")
    print(f"Frame UTC: {radar_ts.isoformat()}")
    print(f"Principal track: {candidate}")
    print(f"Núcleos operativos: {len(nuclei)}")
    print("Radar / Tracking / Nowcast / Geo / ML / Evolution / ECMWF / Intelligence / Decision: sincronizados")


if __name__ == "__main__":
    main()
