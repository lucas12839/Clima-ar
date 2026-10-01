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
    fecha_url = fecha.strftime("%Y%m%d")
    url = BASE_URL.format(fecha=fecha_url)

    try:
        respuesta = requests.get(url, timeout=60)
        respuesta.raise_for_status()

        contenido = respuesta.content.decode(
            "latin-1",
            errors="replace"
        )

        if not contenido.strip():
            return [], "archivo_vacio"

        if "El archivo no existe." in contenido:
            return [], "archivo_inexistente"

        registros = []

        for linea in contenido.splitlines():

            linea = linea.strip()

            if not linea:
                continue

            partes = linea.split()

            if len(partes) < 8:
                continue

            fecha_dato = partes[0]
            hora = partes[1]
            temperatura = partes[2]
            humedad = partes[3]
            presion = partes[4]
            direccion = partes[5]
            velocidad = partes[6]

            estacion = " ".join(partes[7:]).strip()

            if estacion not in ESTACIONES:
                continue

            registros.append(
                {
                    "fecha": fecha_dato,
                    "hora": hora,
                    "temperatura": temperatura,
