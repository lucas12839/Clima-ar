import csv
import json
import os
from datetime import datetime, timedelta

import requests


LAT = -38.71
LON = -62.26
TIMEZONE = "America/Argentina/Buenos_Aires"

EVENTS = [
    "2023-12-16",
    "2025-03-07",
]

API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "cape",
    "lifted_index",
    "convective_inhibition",
    "freezing_level_height",
    "boundary_layer_height",
    "total_column_integrated_water_vapour",

    "temperature_850hPa",
    "relative_humidity_850hPa",
    "wind_speed_850hPa",
    "wind_direction_850hPa",

    "temperature_500hPa",
    "relative_humidity_500hPa",
    "wind_speed_500hPa",
    "wind_direction_500hPa",

    "temperature_300hPa",
    "wind_speed_300hPa",
    "wind_direction_300hPa",
]


OUT_DIR = "data/eventos_severos"


def descargar_evento(event_date):
    fecha = datetime.fromisoformat(event_date).date()

    inicio = fecha - timedelta(days=3)
    fin = fecha + timedelta(days=1)

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": inicio.isoformat(),
        "end_date": fin.isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
    }

    print(f"Descargando {event_date}...")
    print(f"Periodo: {inicio} -> {fin}")

    response = requests.get(
        API_URL,
        params=params,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise RuntimeError(
            f"La API no devolvio datos horarios para {event_date}"
        )

    return data


def construir_filas(event_date, data):
    hourly = data["hourly"]

    times = hourly.get("time", [])

    if not times:
        raise RuntimeError(
            f"No hay timestamps para {event_date}"
        )

    filas = []

    for i, timestamp in enumerate(times):

        fila = {
            "event_date": event_date,
            "time": timestamp,
        }

        for variable in HOURLY_VARS:

            valores = hourly.get(variable)

            if valores is None:
                fila[variable] = None
                continue

            if i >= len(valores):
                fila[variable] = None
                continue

            fila[variable] = valores[i]

        filas.append(fila)

    return filas


def guardar_csv(path, filas):
    if not filas:
        raise RuntimeError(
            f"No hay datos para guardar en {path}"
        )

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    campos = list(filas[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as archivo:

        writer = csv.DictWriter(
            archivo,
            fieldnames=campos,
        )

        writer.writeheader()
        writer.writerows(filas)


def validar_variables(filas):
    resultado = {}

    for variable in HOURLY_VARS:

        encontrados = 0

        for fila in filas:

            valor = fila.get(variable)

            if valor is not None:
                encontrados += 1

        resultado[variable] = {
            "registros_validos": encontrados,
            "total_registros": len(filas),
        }

    return resultado


def main():

    os.makedirs(
        OUT_DIR,
        exist_ok=True,
    )

    resumen = {
        "version": "1.0.0",
        "generado_utc": datetime.utcnow().isoformat() + "Z",
        "fuente": "Open-Meteo Historical Forecast API",
        "ubicacion": {
            "nombre": "Bahia Blanca",
            "latitud": LAT,
            "longitud": LON,
            "timezone": TIMEZONE,
        },
        "eventos": [],
    }

    todas_las_filas = []

    for event_date in EVENTS:

        data = descargar_evento(event_date)

        filas = construir_filas(
            event_date,
            data,
        )

        output = os.path.join(
            OUT_DIR,
            f"variables_convectivas_{event_date}.csv",
        )

        guardar_csv(
            output,
            filas,
        )

        validacion = validar_variables(
            filas
        )

        evento = {
            "fecha": event_date,
            "filas": len(filas),
            "archivo": output,
            "variables": validacion,
        }

        resumen["eventos"].append(evento)

        todas_las_filas.extend(filas)

        print(
            f"{event_date}: "
            f"{len(filas)} registros"
        )

    conjunto = os.path.join(
        OUT_DIR,
        "variables_convectivas_2023_2025.csv",
    )

    guardar_csv(
        conjunto,
        todas_las_filas,
    )

    resumen["total_filas"] = len(
        todas_las_filas
    )

    resumen["archivo_conjunto"] = conjunto

    resumen_path = os.path.join(
        OUT_DIR,
        "resumen_variables_convectivas.json",
    )

    with open(
        resumen_path,
        "w",
        encoding="utf-8",
    ) as archivo:

        json.dump(
            resumen,
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("================================")
    print("RECONSTRUCCION COMPLETADA")
    print("================================")
    print(
        f"Total registros: "
        f"{len(todas_las_filas)}"
    )
    print(
        f"Archivo: {conjunto}"
    )
    print(
        f"Resumen: {resumen_path}"
    )


if __name__ == "__main__":
    main()
