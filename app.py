from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import csv
import json
import math
import zipfile

import requests
import joblib
import pandas as pd
import numpy as np


app = FastAPI(
    title="ClimaAR",
    description="Servicio meteorológico de ClimaAR",
    version="0.5.0"
)

RADAR_IMAGE_URL = (
    "https://estaticos.smn.gob.ar/vmsr/radar/"
    "RMA10_240_ZH_CMAX_20260928_091811Z.png"
)

RADAR_CSV = Path(
    "data/radar/radar_features.csv"
)

METRICAS = Path(
    "modelo/metricas_modelo.json"
)

MODELO = Path(
    "modelo/climaar_modelo_temperatura_1h.joblib"
)

SMN_URL = (
    "https://ssl.smn.gob.ar/dpd/descarga_opendata.php"
    "?file=observaciones/datohorario{}.txt"
)

ESTACION = "BAHIA BLANCA AERO"


def cargar_modelo():
    if not MODELO.exists():
        return None

    return joblib.load(MODELO)


def obtener_smn(fecha):
    try:
        url = SMN_URL.format(
            fecha.strftime("%Y%m%d")
        )

        r = requests.get(
            url,
            timeout=30
        )

        r.raise_for_status()

        texto = r.content.decode(
            "latin-1",
            errors="replace"
        )

        if (
            not texto.strip()
            or "El archivo no existe." in texto
        ):
            return []

        datos = []

        for linea in texto.splitlines():
            p = linea.strip().split()

            if len(p) < 8:
                continue

            estacion = " ".join(p[7:])

            if estacion != ESTACION:
                continue

            datos.append({
                "fecha": p[0],
                "hora": p[1],
                "temperatura": p[2],
                "humedad": p[3],
                "presion": p[4],
                "direccion_viento": p[5],
                "velocidad_viento": p[6],
                "estacion": estacion
            })

        return datos

    except Exception:
        return []


def obtener_datos_actuales():
    hoy = datetime.now(
        timezone.utc
    ).date()

    filas = []

    for dias in [1, 0]:
        fecha = hoy - timedelta(
            days=dias
        )

        filas.extend(
            obtener_smn(fecha)
        )

    if not filas:
        return None

    df = pd.DataFrame(filas)

    df["fecha_hora"] = pd.to_datetime(
        df["fecha"].astype(str)
        + " "
        + df["hora"].astype(str),
        errors="coerce"
    )

    columnas = [
        "temperatura",
        "humedad",
        "presion",
        "direccion_viento",
        "velocidad_viento"
    ]

    for columna in columnas:
        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )

    df = df.dropna(
        subset=["fecha_hora"]
    )

    df = df.sort_values(
        "fecha_hora"
    )

    df = df.drop_duplicates(
        "fecha_hora"
    )

    if df.empty:
        return None

    direccion = np.deg2rad(
        df["direccion_viento"]
    )

    df["viento_u"] = (
        -df["velocidad_viento"]
        * np.sin(direccion)
    )

    df["viento_v"] = (
        -df["velocidad_viento"]
        * np.cos(direccion)
    )

    rh = df["humedad"].clip(
        1,
        100
    )

    t = df["temperatura"]

    gamma = (
        np.log(rh / 100)
        + (17.625 * t)
        / (243.04 + t)
    )

    df["punto_rocio"] = (
        243.04 * gamma
        / (17.625 - gamma)
    )

    df["hora_sin"] = np.sin(
        2 * np.pi
        * df["fecha_hora"].dt.hour
        / 24
    )

    df["hora_cos"] = np.cos(
        2 * np.pi
        * df["fecha_hora"].dt.hour
        / 24
    )

    variables = [
        "temperatura",
        "humedad",
        "presion",
        "direccion_viento",
        "velocidad_viento",
        "viento_u",
        "viento_v",
        "punto_rocio",
        "hora_sin",
        "hora_cos"
    ]

    for horas in [1, 3, 6, 24]:
        for variable in variables:
            df[
                f"{variable}_lag{horas}"
            ] = df[variable].shift(
                horas
            )

    return df


@app.get("/")
def inicio():
    return {
        "app": "ClimaAR",
        "estado": "activo",
        "radar": "/radar",
        "prediccion": "/prediccion",
        "modelo": "/modelo",
        "estado_general": "/estado"
    }


@app.get("/health")
def health():
    return {
        "estado": "ok",
        "servicio": "ClimaAR",
        "hora_utc": datetime.now(
            timezone.utc
        ).isoformat()
    }


@app.get("/radar")
def radar():
    datos = []

    if RADAR_CSV.exists():
        with RADAR_CSV.open(
            "r",
            encoding="utf-8",
            newline=""
        ) as f:
            filas = list(
                csv.DictReader(f)
            )

            datos = filas[-20:]

    ultimo = (
        datos[-1]
        if datos
        else None
    )

    return {
        "estado": "ok",
        "radar": "Bahía Blanca",
        "frames_guardados": len(datos),
        "ultimo_frame": ultimo,
        "imagen_smn": RADAR_IMAGE_URL
    }


@app.get("/radar/imagen")
def radar_imagen():
    return RedirectResponse(
        url=RADAR_IMAGE_URL
    )


@app.get("/modelo")
def modelo():
    if not METRICAS.exists():
        return {
            "estado": "pendiente",
            "mensaje": "Métricas no disponibles."
        }

    with METRICAS.open(
        "r",
        encoding="utf-8"
    ) as f:
        metricas = json.load(f)

    return {
        "estado": "entrenado",
        "modelo": metricas
    }


@app.get("/prediccion")
def prediccion():
    modelo = cargar_modelo()

    if modelo is None:
        return {
            "estado": "pendiente",
            "mensaje": "Modelo no disponible."
        }

    df = obtener_datos_actuales()

    if df is None:
        return {
            "estado": "sin_datos",
            "mensaje": "No se pudieron obtener datos actuales del SMN."
        }

    fila = df.iloc[-1]

    try:
        columnas = list(
            modelo.feature_names_in_
        )

        X = pd.DataFrame(
            [[fila.get(c, np.nan)
              for c in columnas]],
            columns=columnas
        )

        if X.isna().any().any():
            faltantes = list(
                X.columns[
                    X.isna().iloc[0]
                ]
            )

            return {
                "estado": "sin_datos_suficientes",
                "faltantes": faltantes
            }

        pred = float(
            modelo.predict(X)[0]
        )

        return {
            "estado": "ok",
            "ubicacion": "Bahía Blanca",
            "hora_dato": str(
                fila["fecha_hora"]
            ),
            "temperatura_actual": float(
                fila["temperatura"]
            ),
            "temperatura_1h": round(
                pred,
                2
            ),
            "variacion_1h": round(
                pred
                - float(fila["temperatura"]),
                2
            )
        }

    except Exception as e:
        return {
            "estado": "error",
            "mensaje": str(e)
        }


@app.get("/estado")
def estado():
    frames = 0

    if RADAR_CSV.exists():
        with RADAR_CSV.open(
            "r",
            encoding="utf-8",
            newline=""
        ) as f:
            frames = max(
                sum(1 for _ in f) - 1,
                0
            )

    return {
        "climaar": "activo",
        "radar_frames": frames,
        "modelo_disponible": MODELO.exists(),
        "metricas_disponibles": METRICAS.exists(),
        "hora_utc": datetime.now(
            timezone.utc
        ).isoformat()
        }
