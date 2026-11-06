#!/usr/bin/env python3
"""Historial inmutable de observaciones BCP y SMN para ClimaAR.

Solo biblioteca estándar. Por defecto escribe; --dry-run no modifica nada.
Los originales se conservan comprimidos y las revisiones no se sobrescriben.
"""
import argparse
import gzip
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
CATALOG = BASE / "data/ia/estaciones/catalogo_estaciones.json"
SOURCES = {
    "bcp": BASE / "data/ia/estaciones/observaciones/bcp_actual.json",
    "smn": BASE / "data/ia/estaciones/observaciones/smn_synop_actual.json",
}
HISTORY = BASE / "data/ia/estaciones/historico"
DB = HISTORY / "observaciones.sqlite3"
RAW = HISTORY / "originales"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def number(value):
    try:
        value = float(value)
        return value if value == value and abs(value) != float("inf") else None
    except (TypeError, ValueError):
        return None


def first_smn(variables, key):
    values = variables.get(key, [])
    if not isinstance(values, list):
        return None
    for item in values:
        if isinstance(item, dict):
            value = number(item.get("value"))
            if value is not None:
                return value
    return None


def normalize(network, variables):
    """Unidades comunes. Los valores ausentes permanecen como null."""
    if network == "bcp":
        wind = number(variables.get("viento_kmh"))
        gust = number(variables.get("rafaga_15min_kmh"))
        return {
            "temperature_c": number(variables.get("temperatura_c")),
            "dewpoint_c": None,
            "relative_humidity_pct": number(variables.get("humedad_relativa_pct")),
            "pressure_hpa": number(variables.get("presion_hpa")),
            "precipitation_mm": number(variables.get("precipitacion_dia_mm")),
            "wind_speed_ms": wind / 3.6 if wind is not None else None,
            "wind_direction": variables.get("direccion_viento"),
            "wind_gust_ms": gust / 3.6 if gust is not None else None,
            "radiation_wm2": number(variables.get("radiacion_wm2")),
            "soil_temperature_c": number(variables.get("temp_suelo_5cm_c")),
        }

    return {
        "temperature_c": first_smn(variables, "air_temperature"),
        "dewpoint_c": first_smn(variables, "dewpoint_temperature"),
        "relative_humidity_pct": None,
        "pressure_hpa": first_smn(variables, "pressure_reduced_to_mean_sea_level"),
        "pressure_change_3h_hpa": first_smn(variables, "3hour_pressure_change"),
        "precipitation_mm": first_smn(variables, "total_precipitation_or_total_water_equivalent"),
        "wind_speed_ms": first_smn(variables, "wind_speed"),
        "wind_direction": first_smn(variables, "wind_direction"),
        "visibility_m": first_smn(variables, "horizontal_visibility"),
        "cloud_cover_pct": first_smn(variables, "cloud_cover_total"),
        "cloud_base_m": first_smn(variables, "height_of_base_of_cloud"),
        "maximum_temperature_c": first_smn(variables, "maximum_temperature_at_height_and_over_period_specified"),
        "minimum_temperature_c": first_smn(variables, "minimum_temperature_at_height_and_over_period_specified"),
    }


def init_db(conn):
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS capture_runs (
        id INTEGER PRIMARY KEY,
        captured_at TEXT NOT NULL,
        bcp_sha256 TEXT NOT NULL,
        smn_sha256 TEXT NOT NULL,
        stations_total INTEGER NOT NULL,
        stations_fresh INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_capture_runs_time
        ON capture_runs(captured_at);

    CREATE TABLE IF NOT EXISTS raw_snapshots (
        id INTEGER PRIMARY KEY,
        network TEXT NOT NULL,
        source_generated_at TEXT,
        captured_at TEXT NOT NULL,
        content_sha256 TEXT NOT NULL UNIQUE,
        archive_path TEXT NOT NULL,
        stations_total INTEGER,
        stations_validas INTEGER
    );

    CREATE TABLE IF NOT EXISTS station_observations (
        id INTEGER PRIMARY KEY,
        network TEXT NOT NULL,
        station_key TEXT NOT NULL,
        station_name TEXT,
        observation_time TEXT,
        observation_key TEXT NOT NULL,
        captured_at TEXT NOT NULL,
        source_generated_at TEXT,
        age_minutes REAL,
        fresh INTEGER NOT NULL,
        status TEXT NOT NULL,
        quality_mask INTEGER NOT NULL,
        variables_json TEXT NOT NULL,
        normalized_json TEXT NOT NULL,
        payload_sha256 TEXT NOT NULL,
        raw_snapshot_sha256 TEXT NOT NULL,
        UNIQUE(network, station_key, observation_key, payload_sha256)
    );

    CREATE INDEX IF NOT EXISTS idx_obs_time
        ON station_observations(observation_time);
    CREATE INDEX IF NOT EXISTS idx_obs_station_time
        ON station_observations(network, station_key, observation_time);
    CREATE INDEX IF NOT EXISTS idx_obs_fresh
        ON station_observations(fresh, status);
    """)


def process(dry_run=False):
    catalog = read_json(CATALOG)
    catalog_map = {}
    for station in catalog.get("stations", []):
        if station.get("network") == "bcp":
            key = str(station.get("station_number"))
        else:
            key = str(station.get("wigos", ""))
        catalog_map[(station.get("network"), key)] = station.get("name")

    inputs = {}
    for network, path in SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Falta archivo de origen: {path}")
        raw = path.read_bytes()
        data = json.loads(raw)
        if not isinstance(data.get("stations"), list):
            raise ValueError(f"Formato inesperado en {path}")
        inputs[network] = (raw, data)

    captured = utc_now()
    pending = []
    for network, (raw, data) in inputs.items():
        for station in data["stations"]:
            key = (
                str(station.get("station_id"))
                if network == "bcp"
                else str(station.get("wigos", ""))
            )
            if not key:
                raise ValueError(f"Estación sin identificador en {network}")

            variables = station.get("variables") or {}
            if not isinstance(variables, dict):
                variables = {}

            obs_time = station.get("observation_time")
            age = None
            if obs_time:
                try:
                    observed = datetime.fromisoformat(
                        str(obs_time).replace("Z", "+00:00")
                    )
                    if observed.tzinfo is None:
                        observed = observed.replace(tzinfo=timezone.utc)
                    age = (datetime.now(timezone.utc) - observed.astimezone(timezone.utc)).total_seconds() / 60
                except (ValueError, TypeError):
                    age = None

            fresh = (
                age is not None
                and 0 <= age < 15
                and station.get("estado") == "valida"
            )
            status = str(station.get("estado", "sin_estado"))
            obs_time = station.get("observation_time")
            payload = json.dumps(
                station, sort_keys=True, ensure_ascii=False,
                separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            payload_hash = digest(payload)
            observation_key = str(obs_time) if obs_time else "sin_hora:" + payload_hash

            pending.append({
                "network": network,
                "station_key": key,
                "station_name": catalog_map.get((network, key), station.get("nombre")),
                "observation_time": obs_time,
                "observation_key": observation_key,
                "captured_at": captured,
                "source_generated_at": data.get("generated_utc"),
                "age_minutes": age,
                "fresh": int(fresh),
                "status": status,
                "quality_mask": int(fresh),
                "variables_json": json.dumps(
                    variables, ensure_ascii=False, sort_keys=True,
                    allow_nan=False
                ),
                "normalized_json": json.dumps(
                    normalize(network, variables), ensure_ascii=False,
                    sort_keys=True, allow_nan=False
                ),
                "payload_sha256": payload_hash,
                "raw_snapshot_sha256": digest(raw),
            })

    print("CAPTURA:", captured)
    print("ESTACIONES LEIDAS:", len(pending))
    print("ESTACIONES VIGENTES:", sum(x["fresh"] for x in pending))
    print("ESTACIONES NO VIGENTES:", sum(not x["fresh"] for x in pending))
    print("ORIGINALES:", {
        n: digest(raw) for n, (raw, _) in inputs.items()
    })

    if dry_run:
        print("MODO: DRY-RUN. No se escribió ningún archivo.")
        return

    HISTORY.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB, timeout=30)
    try:
        init_db(conn)
        with conn:
            conn.execute("""
                INSERT INTO capture_runs
                (captured_at, bcp_sha256, smn_sha256, stations_total, stations_fresh)
                VALUES (?, ?, ?, ?, ?)
            """, (
                captured,
                digest(inputs["bcp"][0]),
                digest(inputs["smn"][0]),
                len(pending),
                sum(x["fresh"] for x in pending),
            ))
            for network, (raw, data) in inputs.items():
                sha = digest(raw)
                archive = RAW / network / f"{sha}.json.gz"
                archive.parent.mkdir(parents=True, exist_ok=True)

                if not archive.exists():
                    temp = archive.with_suffix(".tmp")
                    temp.write_bytes(gzip.compress(raw, mtime=0))
                    temp.replace(archive)

                conn.execute("""
                    INSERT OR IGNORE INTO raw_snapshots
                    (network, source_generated_at, captured_at, content_sha256,
                     archive_path, stations_total, stations_validas)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    network, data.get("generated_utc"), captured, sha,
                    str(archive.relative_to(BASE)),
                    data.get("stations_total"),
                    data.get("stations_validas"),
                ))

            for x in pending:
                conn.execute("""
                    INSERT OR IGNORE INTO station_observations
                    (network, station_key, station_name, observation_time,
                     observation_key, captured_at, source_generated_at,
                     age_minutes, fresh, status, quality_mask, variables_json,
                     normalized_json, payload_sha256, raw_snapshot_sha256)
                    VALUES (
                     :network, :station_key, :station_name, :observation_time,
                     :observation_key, :captured_at, :source_generated_at,
                     :age_minutes, :fresh, :status, :quality_mask, :variables_json,
                     :normalized_json, :payload_sha256, :raw_snapshot_sha256
                    )
                """, x)

        print("BASE:", DB.relative_to(BASE))
        print("ORIGINALES COMPRIMIDOS:", RAW.relative_to(BASE))
        print("OBSERVACIONES TOTALES:", conn.execute(
            "SELECT COUNT(*) FROM station_observations"
        ).fetchone()[0])
        print("CAPTURAS ORIGINALES:", conn.execute(
            "SELECT COUNT(*) FROM raw_snapshots"
        ).fetchone()[0])
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    process(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
