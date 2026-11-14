#!/usr/bin/env python3
"""Colector SMN SYNOP con diagnóstico de consulta y frescura estricta.

Mantiene la política ClimaAR: una observación solo es fresca entre 0 y 15 min.
No convierte ausencia de datos en una observación válida.
"""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import requests

COLLECTION_URL = (
    "https://w2b.smn.gov.ar/oapi/collections/"
    "urn%3Awmo%3Amd%3Aar-smn%3Aslt0ci/items"
)
STATIONS = {
    "0-20000-0-87750": "BAHIA BLANCA AERO",
    "0-20000-0-87679": "PIGUE AERO",
    "0-20000-0-87637": "CORONEL SUAREZ AERO",
    "0-20000-0-87736": "RIO COLORADO",
    "0-20000-0-87688": "TRES ARROYOS",
}
MAX_AGE_MINUTES = 15.0
LOOKBACK_MINUTES = 120
OUTPUT = Path("data/ia/estaciones/observaciones/smn_synop_actual.json")
TIMEOUT_SECONDS = 20


def parse_time(value: Any):
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    if "/" in raw:
        raw = raw.split("/")[-1]
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def age_minutes(dt, now):
    if dt is None:
        return None
    return (now - dt).total_seconds() / 60.0


def fetch_features(session, wigos, now):
    """Prueba primero ventana temporal; si viene vacía, consulta el último lote."""
    start = now - timedelta(minutes=LOOKBACK_MINUTES)
    base = {
        "f": "json",
        "wigos_station_identifier": wigos,
        "limit": 100,
    }
    interval = (
        start.isoformat().replace("+00:00", "Z")
        + "/"
        + now.isoformat().replace("+00:00", "Z")
    )

    attempts = [
        ("wigos_datetime", {**base, "datetime": interval}),
        ("wigos_latest_sorted", {**base, "sortby": "-reportTime"}),
        ("wigos_latest", base),
    ]
    diagnostics = []

    for mode, params in attempts:
        try:
            response = session.get(
                COLLECTION_URL, params=params, timeout=TIMEOUT_SECONDS
            )
            response.raise_for_status()
            payload = response.json()
            features = payload.get("features", [])
            diagnostics.append({
                "mode": mode,
                "http_status": response.status_code,
                "features_count": len(features),
                "numberMatched": payload.get("numberMatched"),
                "numberReturned": payload.get("numberReturned"),
            })
            if features:
                return features, diagnostics, mode
        except Exception as exc:
            diagnostics.append({
                "mode": mode,
                "error": f"{type(exc).__name__}: {exc}",
            })

    return [], diagnostics, "sin_resultados"


def build_station_record(wigos, name, features, now, diagnostics, query_mode):
    reports = {}
    for feature in features:
        props = feature.get("properties") or {}
        report_id = props.get("reportId")
        if report_id:
            reports.setdefault(report_id, []).append(props)

    candidates = []
    for report_id, properties in reports.items():
        report_times = [
            parse_time(item.get("reportTime")) for item in properties
        ]
        report_times = [dt for dt in report_times if dt is not None]

        # Compatibilidad con respuestas antiguas que no incluyan reportTime.
        if not report_times:
            report_times = [
                parse_time(item.get("phenomenonTime")) for item in properties
            ]
            report_times = [dt for dt in report_times if dt is not None]

        if report_times:
            candidates.append((max(report_times), report_id, properties))

    if not candidates:
        return {
            "ok": False,
            "nombre": name,
            "wigos": wigos,
            "estado": "sin_datos" if not features else "sin_timestamp",
            "fresh": False,
            "query_mode": query_mode,
            "diagnostico_consulta": diagnostics,
        }

    report_time, report_id, properties = max(candidates, key=lambda x: x[0])
    age = age_minutes(report_time, now)

    if age is None:
        state = "sin_timestamp"
    elif age < 0:
        state = "timestamp_futuro"
    elif age > MAX_AGE_MINUTES:
        state = "descartada_por_antiguedad"
    else:
        state = "valida"

    variables = {}
    for prop in properties:
        variable_name = prop.get("name")
        if not variable_name:
            continue
        variable_time = parse_time(prop.get("phenomenonTime"))
        variable_age = age_minutes(variable_time, now)
        if (
            variable_age is None
            or variable_age < 0
            or variable_age > MAX_AGE_MINUTES
        ):
            continue
        variables.setdefault(variable_name, []).append({
            "value": prop.get("value"),
            "units": prop.get("units"),
            "phenomenonTime": prop.get("phenomenonTime"),
            "edad_minutos": round(variable_age, 2),
        })

    return {
        "ok": True,
        "nombre": name,
        "wigos": wigos,
        "reportId": report_id,
        "observation_time": report_time.isoformat(),
        "edad_minutos": round(age, 2) if age is not None else None,
        "fresh": state == "valida",
        "estado": state,
        "variables": variables,
        "query_mode": query_mode,
        "diagnostico_consulta": diagnostics,
    }


def collect(wigos, name, session=requests, now=None):
    now = now or datetime.now(timezone.utc)
    features, diagnostics, query_mode = fetch_features(session, wigos, now)
    return build_station_record(
        wigos, name, features, now, diagnostics, query_mode
    )


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    stations = []

    for wigos, name in STATIONS.items():
        try:
            stations.append(collect(wigos, name, now=now))
        except Exception as exc:
            stations.append({
                "ok": False,
                "nombre": name,
                "wigos": wigos,
                "estado": "error",
                "fresh": False,
                "error": f"{type(exc).__name__}: {exc}",
            })

    valid_count = sum(1 for item in stations if item.get("fresh") is True)
    result = {
        "ok": True,
        "collector": "smn_synop",
        "version": "1.1.0",
        "generated_utc": now.isoformat(),
        "source": COLLECTION_URL,
        "max_age_minutes": MAX_AGE_MINUTES,
        "lookback_minutes": LOOKBACK_MINUTES,
        "stations_total": len(stations),
        "stations_validas": valid_count,
        "stations": stations,
    }

    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(OUTPUT)

    print("=== COLLECTOR SMN SYNOP 1.1.0 ===")
    print(f"Total:   {len(stations)}")
    print(f"Validas: {valid_count}")
    for item in stations:
        age = item.get("edad_minutos")
        suffix = f" | {age} min" if age is not None else ""
        print(f"{item['nombre']}: {item['estado']}{suffix}")
        for diagnostic in item.get("diagnostico_consulta", []):
            if diagnostic.get("features_count", 0) or diagnostic.get("error"):
                print(f"  consulta {diagnostic['mode']}: {diagnostic}")

    print(f"Salida: {OUTPUT}")


if __name__ == "__main__":
    main()
