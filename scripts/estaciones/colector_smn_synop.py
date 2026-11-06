#!/usr/bin/env python3

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

URL = "https://w2b.smn.gov.ar/oapi/collections/urn%3Awmo%3Amd%3Aar-smn%3Aslt0ci/items"

STATIONS = {
    "0-20000-0-87750": "BAHIA BLANCA AERO",
    "0-20000-0-87679": "PIGUE AERO",
    "0-20000-0-87637": "CORONEL SUAREZ AERO",
    "0-20000-0-87736": "RIO COLORADO",
    "0-20000-0-87688": "TRES ARROYOS",
}

MAX_AGE = 15
LOOKBACK = 120
OUTPUT = Path("data/ia/estaciones/observaciones/smn_synop_actual.json")


def parse_time(value):
    if not value:
        return None
    value = value.replace("Z", "+00:00")
    if "/" in value:
        value = value.split("/")[-1]
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except ValueError:
        return None


def collect(wigos, name):
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=LOOKBACK)

    params = {
        "f": "json",
        "wigos_station_identifier": wigos,
        "datetime": (
            start.isoformat().replace("+00:00", "Z")
            + "/"
            + now.isoformat().replace("+00:00", "Z")
        ),
        "limit": 100,
    }

    r = requests.get(URL, params=params, timeout=20)
    r.raise_for_status()

    features = r.json().get("features", [])
    reports = {}

    for f in features:
        p = f.get("properties", {})
        rid = p.get("reportId")
        if rid:
            reports.setdefault(rid, []).append(p)

    if not reports:
        return {
            "ok": False,
            "nombre": name,
            "wigos": wigos,
            "estado": "sin_datos",
        }

    candidates = []

    for rid, items in reports.items():
        times = [parse_time(x.get("phenomenonTime")) for x in items]
        times = [t for t in times if t is not None]
        if times:
            candidates.append((max(times), rid, items))

    if not candidates:
        return {
            "ok": False,
            "nombre": name,
            "wigos": wigos,
            "estado": "sin_timestamp",
        }

    obs_time, report_id, items = max(candidates, key=lambda x: x[0])
    age = max(0, (now - obs_time).total_seconds() / 60)

    variables = {}

    for p in items:
        var = p.get("name")
        if not var:
            continue

        variables.setdefault(var, []).append({
            "value": p.get("value"),
            "units": p.get("units"),
            "phenomenonTime": p.get("phenomenonTime"),
        })

    return {
        "ok": True,
        "nombre": name,
        "wigos": wigos,
        "reportId": report_id,
        "observation_time": obs_time.isoformat(),
        "edad_minutos": round(age, 2),
        "fresh": age < MAX_AGE,
        "estado": "valida" if age < MAX_AGE else "descartada_por_antiguedad",
        "variables": variables,
    }


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    stations = []

    for wigos, name in STATIONS.items():
        try:
            stations.append(collect(wigos, name))
        except Exception as e:
            stations.append({
                "ok": False,
                "nombre": name,
                "wigos": wigos,
                "estado": "error",
                "error": str(e),
            })

    result = {
        "ok": True,
        "collector": "smn_synop",
        "version": "1.0.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "max_age_minutes": MAX_AGE,
        "stations_total": len(stations),
        "stations_validas": sum(
            1 for x in stations if x.get("fresh") is True
        ),
        "stations": stations,
    }

    tmp = OUTPUT.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(OUTPUT)

    print("=== COLLECTOR SMN SYNOP ===")
    print(f"Total:   {len(stations)}")
    print(
        "Validas: "
        f"{sum(1 for x in stations if x.get('fresh') is True)}"
    )

    for x in stations:
        age = x.get("edad_minutos")
        print(
            f"{x['nombre']}: {x['estado']}"
            + (f" | {age} min" if age is not None else "")
        )


if __name__ == "__main__":
    main()
