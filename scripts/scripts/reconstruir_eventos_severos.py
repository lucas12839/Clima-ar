import csv
import json
import time
from datetime import date, timedelta
from pathlib import Path

import requests


LAT = -38.71
LON = -62.26
TIMEZONE = "America/Argentina/Buenos_Aires"


EVENTS = {
    "2019-12-30": {
        "nombre": "Tormenta severa / tornado regional",
        "alcance": "regional, cerca de Cabildo y Dique Paso de las Piedras",
        "fuente_oficial": "SMN / reportes meteorologicos disponibles"
    },
    "2023-12-16": {
        "nombre": "Vientos destructivos",
        "alcance": "Bahia Blanca",
        "fuente_oficial": "SMN Nota Tecnica 2024-191",
        "racha_max_oficial_kmh": 155
    },
    "2025-03-07": {
        "nombre": "Lluvia extrema e inundacion",
        "alcance": "Bahia Blanca",
        "fuente_oficial": "SMN informe especial 4-8 marzo 2025",
        "lluvia_6h_oficial_mm": 210,
        "lluvia_12h_oficial_mm": 290,
        "lluvia_evento_oficial_mm": 312
    }
}


# Variables confirmadas para el Historical Weather API.
HOURLY = [
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


def fetch_event(event_date):
    """
    Descarga 72 horas antes y 24 horas despues
    del dia del evento.
    """

    event_day = date.fromisoformat(event_date)

    start = event_day - timedelta(days=3)
    end = event_day + timedelta(days=1)

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY),
        "timezone": TIMEZONE,
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }

    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params,
        timeout=90,
    )

    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise RuntimeError(
            f"Open-Meteo no devolvio datos horarios para {event_date}: {data}"
        )

    hourly = data["hourly"]

    rows = []

    for i, timestamp in enumerate(hourly["time"]):

        row = {
            "event_date": event_date,
            "time": timestamp
        }

        for variable in HOURLY:

            values = hourly.get(variable, [])

            if i < len(values):
                row[variable] = values[i]
            else:
                row[variable] = None

        rows.append(row)

    return rows


def add_derived_variables(rows):

    for i, row in enumerate(rows):

        def number(key):

            value = row.get(key)

            if value in (None, ""):
                return None

            return float(value)

        temperature = number("temperature_2m")
        dewpoint = number("dew_point_2m")
        pressure = number("pressure_msl")
        wind = number("wind_speed_10m")
        gust = number("wind_gusts_10m")

        # Diferencia temperatura - punto de rocio.
        if temperature is not None and dewpoint is not None:
            row["dewpoint_depression_c"] = (
                temperature - dewpoint
            )
        else:
            row["dewpoint_depression_c"] = None

        # Diferencia entre rafaga y viento sostenido.
        if gust is not None and wind is not None:
            row["wind_gust_excess_kmh"] = gust - wind
        else:
            row["wind_gust_excess_kmh"] = None

        # Cambios respecto de la hora anterior.
        if i == 0:

            row["pressure_change_1h_hpa"] = None
            row["temperature_change_1h_c"] = None
            row["dewpoint_change_1h_c"] = None

        else:

            previous = rows[i - 1]

            def previous_number(key):

                value = previous.get(key)

                if value in (None, ""):
                    return None

                return float(value)

            previous_pressure = previous_number("pressure_msl")
            previous_temperature = previous_number("temperature_2m")
            previous_dewpoint = previous_number("dew_point_2m")

            if pressure is not None and previous_pressure is not None:
                row["pressure_change_1h_hpa"] = (
                    pressure - previous_pressure
                )
            else:
                row["pressure_change_1h_hpa"] = None

            if temperature is not None and previous_temperature is not None:
                row["temperature_change_1h_c"] = (
                    temperature - previous_temperature
                )
            else:
                row["temperature_change_1h_c"] = None

            if dewpoint is not None and previous_dewpoint is not None:
                row["dewpoint_change_1h_c"] = (
                    dewpoint - previous_dewpoint
                )
            else:
                row["dewpoint_change_1h_c"] = None

    # Acumulados de precipitacion.
    for i, row in enumerate(rows):

        for hours in (3, 6, 12, 24):

            total = 0.0
            found = False

            start_index = max(0, i - hours + 1)

            for j in range(start_index, i + 1):

                value = rows[j].get("precipitation")

                if value not in (None, ""):

                    total += float(value)
                    found = True

            if found:
                row[f"precip_{hours}h_mm"] = round(total, 3)
            else:
                row[f"precip_{hours}h_mm"] = None

    return rows


def write_csv(path, rows):

    if not rows:
        raise RuntimeError(f"No hay datos para escribir en {path}")

    fields = list(rows[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)


def main():

    output_dir = Path("data/eventos_severos")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    all_rows = []

    summary = {
        "version": "1.0",
        "location": {
            "latitude": LAT,
            "longitude": LON,
            "timezone": TIMEZONE
        },
        "data_source": (
            "Open-Meteo Historical Weather API / "
            "reanalysis"
        ),
        "events": EVENTS
    }

    for event_date in EVENTS:

        print("")
        print("=" * 60)
        print(f"DESCARGANDO EVENTO {event_date}")
        print("=" * 60)

        rows = fetch_event(event_date)

        rows = add_derived_variables(rows)

        output_file = (
            output_dir /
            f"meteorologia_{event_date}.csv"
        )

        write_csv(
            output_file,
            rows
        )

        all_rows.extend(rows)

        print(
            f"OK: {len(rows)} registros"
        )

        print(
            f"Archivo: {output_file}"
        )

        time.sleep(1)

    # Dataset combinado.
    combined_file = (
        output_dir /
        "meteorologia_tres_eventos.csv"
    )

    write_csv(
        combined_file,
        all_rows
    )

    # Resumen.
    summary_file = (
        output_dir /
        "resumen_eventos.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("")
    print("=" * 60)
    print("RECONSTRUCCION COMPLETADA")
    print("=" * 60)

    print(
        f"Total de registros: {len(all_rows)}"
    )

    print(
        f"Dataset combinado: {combined_file}"
    )

    print(
        f"Resumen: {summary_file}"
    )


if __name__ == "__main__":
    main()
