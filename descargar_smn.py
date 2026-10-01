#!/usr/bin/env python3
"""
ClimaAR - Descargador de datos horarios del SMN

Fuente:
https://ssl.smn.gob.ar/dpd/descarga_opendata.php?file=observaciones/datohorarioAAAAMMDD.txt

Uso:
    python descargar_smn.py --desde 2024-01-01 --hasta 2024-12-31

Por defecto procesa Bahía Blanca y estaciones cercanas.
"""

import argparse
import csv
import io
import time
from datetime import date, timedelta
from pathlib import Path

import requests

BASE_URL = (
    "https://ssl.smn.gob.ar/dpd/descarga_opendata.php"
    "?file=observaciones/datohorario{date}.txt"
)

# Primera zona piloto de ClimaAR.
# Se puede ampliar sin tocar el resto del programa.
ESTACIONES = {
    "BAHIA BLANCA AERO",
    "BAHIA BLANCA",
    "TRES ARROYOS",
    "PIGUE AERO",
    "CORONEL SUAREZ AERO",
    "VIEDMA AERO",
    "SAN ANTONIO OESTE AERO",
    "NEUQUEN AERO",
}

COLUMNAS = [
    "fecha",
    "hora",
    "temperatura_c",
    "humedad_pct",
    "presion_hpa",
    "viento_direccion_grados",
    "viento_kmh",
    "estacion",
]

def normalizar(linea):
    """Convierte una línea de ancho fijo del SMN a campos."""
    partes = linea.split()
    if len(partes) < 8:
        return None

    # Los 7 primeros campos son fecha, hora, temp, humedad, presión,
    # dirección y velocidad. El resto corresponde al nombre de estación.
    fecha, hora, temp, hum, pres, direccion, velocidad = partes[:7]
    estacion = " ".join(partes[7:]).strip()

    if not fecha.isdigit() or len(fecha) != 8:
        return None
    if not hora.isdigit():
        return None

    return {
        "fecha": fecha,
        "hora": int(hora),
        "temperatura_c": temp,
        "humedad_pct": hum,
        "presion_hpa": pres,
        "viento_direccion_grados": direccion,
        "viento_kmh": velocidad,
        "estacion": estacion,
    }

def descargar_dia(session, dia):
    fecha = dia.strftime("%Y%m%d")
    url = BASE_URL.format(date=fecha)

    r = session.get(url, timeout=30)
    r.raise_for_status()

    texto = r.content.decode("latin-1", errors="replace")
    if not texto.strip():
        return []

    filas = []
    for linea in io.StringIO(texto):
        fila = normalizar(linea)
        if not fila:
            continue

        if fila["estacion"] in ESTACIONES:
            filas.append(fila)

    return filas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desde", required=True, help="YYYY-MM-DD")
    parser.add_argument("--hasta", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--salida",
        default="climaar_historico_region.csv",
        help="CSV de salida",
    )
    parser.add_argument(
        "--pausa",
        type=float,
        default=0.5,
        help="segundos entre descargas",
    )
    args = parser.parse_args()

    desde = date.fromisoformat(args.desde)
    hasta = date.fromisoformat(args.hasta)

    if hasta < desde:
        raise SystemExit("ERROR: --hasta no puede ser anterior a --desde")

    salida = Path(args.salida)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "ClimaAR/0.1 (proyecto meteorologico gratuito)"
    })

    # Reanuda: si el archivo ya existe, evita duplicar registros.
    existentes = set()
    if salida.exists():
        with salida.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                existentes.add(
                    (row["fecha"], row["hora"], row["estacion"])
                )

    escribir_cabecera = not salida.exists() or salida.stat().st_size == 0

    with salida.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS)
        if escribir_cabecera:
            writer.writeheader()

        dia = desde
        total = 0
        dias_ok = 0
        dias_vacios = 0
        errores = 0

        while dia <= hasta:
            try:
                filas = descargar_dia(session, dia)

                nuevas = []
                for fila in filas:
                    clave = (
                        fila["fecha"],
                        fila["hora"],
                        fila["estacion"],
                    )
                    if clave not in existentes:
                        nuevas.append(fila)
                        existentes.add(clave)

                for fila in nuevas:
                    writer.writerow(fila)

                f.flush()

                if filas:
                    dias_ok += 1
                    total += len(nuevas)
                    print(
                        f"[OK] {dia.isoformat()} | "
                        f"{len(filas)} registros encontrados | "
                        f"{len(nuevas)} nuevos"
                    )
                else:
                    dias_vacios += 1
                    print(f"[--] {dia.isoformat()} | sin datos de las estaciones")

            except Exception as exc:
                errores += 1
                print(f"[ERROR] {dia.isoformat()} | {exc}")

            dia += timedelta(days=1)
            time.sleep(args.pausa)

    print("\n--- RESUMEN ---")
    print(f"Días con datos: {dias_ok}")
    print(f"Días sin datos: {dias_vacios}")
    print(f"Errores: {errores}")
    print(f"Registros nuevos: {total}")
    print(f"Salida: {salida.resolve()}")

if __name__ == "__main__":
    main()
