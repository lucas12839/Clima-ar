#!/usr/bin/env python3
"""Valida y resume las 14 estaciones operativas de ClimaAR.

No importa PyTorch ni utiliza el encoder neuronal experimental sin entrenar.
La frescura se calcula exclusivamente desde observation_time (0..15 min).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
CATALOG = BASE / "data/ia/estaciones/catalogo_estaciones.json"
BCP = BASE / "data/ia/estaciones/observaciones/bcp_actual.json"
SMN = BASE / "data/ia/estaciones/observaciones/smn_synop_actual.json"
OUTPUT = BASE / "data/ia/estaciones/red_observaciones_estado.json"
MAX_AGE = 15.0


def parse_time(value):
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc)


def load(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main():
    now = datetime.now(timezone.utc)
    catalog = load(CATALOG)
    bcp_data = load(BCP)
    smn_data = load(SMN)

    bcp = {str(x.get("station_id")): x for x in bcp_data.get("stations", [])}
    smn = {str(x.get("wigos")): x for x in smn_data.get("stations", [])}

    results = []
    for station in catalog.get("stations", []):
        network = station.get("network")
        record = None
        if network == "bcp":
            record = bcp.get(str(station.get("collector_id")))
        elif network == "smn":
            record = smn.get(str(station.get("wigos_id")))

        observed = parse_time(record.get("observation_time")) if record else None
        age = (now - observed).total_seconds() / 60 if observed else None
        valid = bool(
            station.get("active") is True
            and record
            and age is not None
            and 0 <= age <= MAX_AGE
        )
        if not station.get("active"):
            state = "inactiva"
        elif record is None:
            state = "sin_registro"
        elif observed is None:
            state = "timestamp_invalido_o_ausente"
        elif age < 0:
            state = "timestamp_futuro"
        elif age > MAX_AGE:
            state = "vencida"
        else:
            state = "valida"

        results.append({
            "name": station.get("name"),
            "network": network,
            "valid": valid,
            "state": state,
            "age_minutes": round(age, 2) if age is not None else None,
            "observation_time": observed.isoformat() if observed else None,
        })

    expected = int(catalog.get("station_count", len(results)))
    if len(results) != expected:
        raise RuntimeError(
            f"Catálogo inconsistente: {len(results)} estaciones, esperado {expected}"
        )

    valid_bcp = sum(1 for x in results if x["network"] == "bcp" and x["valid"])
    valid_smn = sum(1 for x in results if x["network"] == "smn" and x["valid"])
    total_valid = valid_bcp + valid_smn

    report = {
        "version": "1.0.0",
        "generated_utc": now.isoformat(),
        "max_age_minutes": MAX_AGE,
        "policy": "La edad se recalcula desde observation_time; futuro, ausente o >15 min se descarta.",
        "station_count": len(results),
        "valid_count": total_valid,
        "bcp": {"total": sum(x["network"] == "bcp" for x in results), "valid": valid_bcp},
        "smn": {"total": sum(x["network"] == "smn" for x in results), "valid": valid_smn},
        "status": "operativa" if total_valid == expected else "parcial",
        "stations": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(OUTPUT)

    print(f"Red de estaciones: {total_valid}/{expected} válidas")
    print(f"BCP: {valid_bcp}/{report['bcp']['total']} | SMN: {valid_smn}/{report['smn']['total']}")
    for item in results:
        age = f"{item['age_minutes']} min" if item["age_minutes"] is not None else "sin hora"
        print(f"{item['name']}: {item['state']} ({age})")
    print(f"Reporte: {OUTPUT}")
    # La falta de estaciones no debe detener el pronóstico radar/ECMWF.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
