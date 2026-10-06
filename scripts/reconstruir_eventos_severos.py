import csv
import json
import os
from datetime import datetime, timedelta

import requests

LAT = -38.71
LON = -62.26
TIMEZONE = "America/Argentina/Buenos_Aires"

EVENTS = {
    "2019-12-30": {
        "tipo": "tormenta_severa_regional",
        "descripcion": "Evento severo regional asociado a tornado reportado en zona Cabildo/Dique Paso de las Piedras.",
        "fuente": "SMN / fuentes historicas del evento",
    },
    "2023-12-16": {
        "tipo": "viento_danino",
        "descripcion": "Evento de viento severo con rafaga maxima oficial de 155 km/h en Bahia Blanca.",
        "fuente": "SMN Nota Tecnica 2024-191",
        "rafaga_max_oficial_kmh": 155,
    },
    "2025-03-07": {
        "tipo": "lluvia_extrema_inundacion",
        "descripcion": "Evento de precipitacion extrema e inundacion en Bahia Blanca.",
        "fuente": "SMN informe especial",
        "lluvia_6h_oficial_mm": 210,
        "lluvia_12h_oficial_mm": 290,
        "lluvia_evento_oficial_mm": 312,
    },
}

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "surface_pressure",
    "precipitation",
    "rain",
    "cloud_cover",
    "vapour_pressure_deficit",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "boundary_layer_height",
]

API_URL = "https://archive-api.open-meteo.com/v1/archive"
OUT_DIR = "data/eventos_severos"


def fetch_event(event_date):
    start = datetime.fromisoformat(event_date).date() - timedelta(days=3)
    end = datetime.fromisoformat(event_date).date() + timedelta(days=1)

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
    }

    response = requests.get(API_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    if "hourly" not in data or "time" not in data["hourly"]:
        raise RuntimeError(f"Respuesta sin datos horarios para {event_date}")

    return data


def value(values, index):
    if index >= len(values):
        return None
    return values[index]


def previous_value(rows, index, key, hours=1):
    j = index - hours
    if j < 0:
        return None
    return rows[j].get(key)


def rolling_sum(rows, index, key, hours):
    vals = []
    first = max(0, index - hours + 1)

    for j in range(first, index + 1):
        v = rows[j].get(key)
        if v is not None:
            vals.append(float(v))

    if len(vals) != hours:
        return None

    return sum(vals)


def build_rows(event_date, data):
    hourly = data["hourly"]
    times = hourly["time"]
    rows = []

    for i, timestamp in enumerate(times):
        row = {
            "event_date": event_date,
            "time": timestamp,
        }

        for key in HOURLY_VARS:
            row[key] = value(hourly.get(key, []), i)

        temp = row.get("temperature_2m")
        dew = row.get("dew_point_2m")
        gust = row.get("wind_gusts_10m")
        wind = row.get("wind_speed_10m")

        row["dewpoint_depression"] = (
            float(temp) - float(dew)
            if temp is not None and dew is not None
            else None
        )

        row["gust_excess"] = (
            float(gust) - float(wind)
            if gust is not None and wind is not None
            else None
        )

        row["temperature_change_1h"] = None
        row["dewpoint_change_1h"] = None
        row["pressure_change_1h"] = None

        prev_temp = previous_value(rows, i, "temperature_2m")
        prev_dew = previous_value(rows, i, "dew_point_2m")
        prev_pressure = previous_value(rows, i, "pressure_msl")

        if temp is not None and prev_temp is not None:
            row["temperature_change_1h"] = float(temp) - float(prev_temp)

        if dew is not None and prev_dew is not None:
            row["dewpoint_change_1h"] = float(dew) - float(prev_dew)

        pressure = row.get("pressure_msl")

        if pressure is not None and prev_pressure is not None:
            row["pressure_change_1h"] = float(pressure) - float(prev_pressure)

        rows.append(row)

    for i, row in enumerate(rows):
        row["precipitation_3h"] = rolling_sum(
            rows, i, "precipitation", 3
        )
        row["precipitation_6h"] = rolling_sum(
            rows, i, "precipitation", 6
        )
        row["precipitation_12h"] = rolling_sum(
            rows, i, "precipitation", 12
        )
        row["precipitation_24h"] = rolling_sum(
            rows, i, "precipitation", 24
        )

    return rows


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"No hay filas para escribir: {path}")

    fields = list(rows[0].keys())

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_rows = []

    resumen = {
        "version": "1.0.0",
        "generado_utc": datetime.utcnow().isoformat() + "Z",
        "ubicacion": {
            "nombre": "Bahia Blanca",
            "latitud": LAT,
            "longitud": LON,
            "timezone": TIMEZONE,
        },
        "eventos": [],
    }

    for event_date, metadata in EVENTS.items():
        print(f"Reconstruyendo {event_date}...")

        data = fetch_event(event_date)
        rows = build_rows(event_date, data)

        output = os.path.join(
            OUT_DIR,
            f"meteorologia_{event_date}.csv",
        )

        write_csv(output, rows)

        all_rows.extend(rows)

        item = dict(metadata)
        item["fecha"] = event_date
        item["filas"] = len(rows)
        item["archivo"] = output

        resumen["eventos"].append(item)

        print(f"  {len(rows)} filas -> {output}")

    combined = os.path.join(
        OUT_DIR,
        "meteorologia_tres_eventos.csv",
    )

    write_csv(combined, all_rows)

    resumen["total_filas"] = len(all_rows)
    resumen["archivo_conjunto"] = combined

    summary_path = os.path.join(
        OUT_DIR,
        "resumen_eventos.json",
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            resumen,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Total filas: {len(all_rows)}")
    print(f"Resumen: {summary_path}")
    print("Reconstruccion completada correctamente.")


if __name__ == "__main__":
    main()
