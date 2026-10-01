import argparse
import csv
import os
import time
from datetime import datetime, timedelta

import requests


BASE_URL = (
    "https://ssl.smn.gob.ar/dpd/descarga_opendata.php"
    "?file=observaciones/datohorario{fecha}.txt"
)

# Estaciones verificadas en la red del SMN para la zona/región de interés.
ESTACIONES = {
    "BAHIA BLANCA AERO",
    "PIGUE AERO",
    "CORONEL SUAREZ AERO",
    "TRES ARROYOS",
    "RIO COLORADO",
    "VIEDMA AERO",
    "SAN ANTONIO OESTE AERO",
}

CAMPOS = [
    "fecha",
    "hora",
    "temperatura",
    "humedad",
    "presion",
    "direccion_viento",
    "velocidad_viento",
    "estacion",
]


def descargar_dia(fecha):
    fecha_txt = fecha.strftime("%Y%m%d")
    url = BASE_URL.format(fecha=fecha_txt)

    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()

        if not r.content:
            return [], "archivo_vacio"

        texto = r.content.decode("latin-1", errors="replace")

        if not texto.strip():
            return [], "archivo_vacio"

        registros = []

        for linea in texto.splitlines():
            linea = linea.strip()

            if not linea:
                continue

            partes = linea.split()

            # La estructura puede variar; ignoramos líneas que no
            # tengan suficientes campos.
            if len(partes) < 8:
                continue

            try:
                estacion = " ".join(partes[7:]).strip()

                # Algunos archivos pueden traer columnas adicionales.
                # Buscamos solamente las estaciones que nos interesan.
                estacion_encontrada = None

                for nombre in ESTACIONES:
                    if estacion == nombre or nombre in estacion:
                        estacion_encontrada = nombre
                        break

                if estacion_encontrada is None:
                    continue

                registro = {
                    "fecha": partes[0],
                    "hora": partes[1],
                    "temperatura": partes[2],
                    "humedad": partes[3],
                    "presion": partes[4],
                    "direccion_viento": partes[5],
                    "velocidad_viento": partes[6],
                    "estacion": estacion_encontrada,
                }

                registros.append(registro)

            except Exception:
                continue

        return registros, None

    except Exception as e:
        return [], str(e)


def cargar_existentes(salida):
    existentes = set()

    if not os.path.exists(salida):
        return existentes

    try:
        with open(salida, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            for fila in reader:
                clave = (
                    fila.get("fecha", ""),
                    fila.get("hora", ""),
                    fila.get("estacion", ""),
                )

                existentes.add(clave)

    except Exception:
        pass

    return existentes


def preparar_archivo(salida):
    if not os.path.exists(salida) or os.path.getsize(salida) == 0:
        with open(salida, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CAMPOS)
            writer.writeheader()


def guardar_registros(salida, registros, existentes):
    nuevos = []

    for registro in registros:
        clave = (
            registro["fecha"],
            registro["hora"],
            registro["estacion"],
        )

        if clave not in existentes:
            existentes.add(clave)
            nuevos.append(registro)

    if not nuevos:
        return 0

    with open(salida, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CAMPOS)

        for registro in nuevos:
            writer.writerow(registro)

    return len(nuevos)


def main():
    parser = argparse.ArgumentParser(
        description="Descarga datos horarios históricos del SMN."
    )

    parser.add_argument(
        "--desde",
        required=True,
        help="Fecha inicial YYYY-MM-DD",
    )

    parser.add_argument(
        "--hasta",
        required=True,
        help="Fecha final YYYY-MM-DD",
    )

    parser.add_argument(
        "--salida",
        default="climaar_smn.csv",
        help="Archivo CSV de salida",
    )

    parser.add_argument(
        "--pausa",
        type=float,
        default=1.0,
        help="Pausa entre consultas en segundos",
    )

    args = parser.parse_args()

    desde = datetime.strptime(args.desde, "%Y-%m-%d").date()
    hasta = datetime.strptime(args.hasta, "%Y-%m-%d").date()

    if hasta < desde:
        raise ValueError("La fecha final no puede ser anterior a la inicial.")

    preparar_archivo(args.salida)

    existentes = cargar_existentes(args.salida)

    total_nuevos = 0
    dias_ok = 0
    dias_sin_datos = 0
    errores = 0

    fecha_actual = desde

    print("======================================")
    print("      
