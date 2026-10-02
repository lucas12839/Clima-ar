from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import csv
import json
import math

import requests
import joblib
import pandas as pd
import numpy as np


app = FastAPI(
    title="ClimaAR",
    description="Sistema meteorológico y seguimiento de tormentas de ClimaAR",
    version="1.0.0"
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

RADAR_IMAGE_URL = (
    "https://estaticos.smn.gob.ar/vmsr/radar/"
    "RMA10_240_ZH_CMAX_20260928_091811Z.png"
)

RADAR_CSV = Path("data/radar/radar_features.csv")

METRICAS = Path("modelo/metricas_modelo.json")

MODELO = Path(
    "modelo/climaar_modelo_temperatura_1h.joblib"
)

SMN_URL = (
    "https://ssl.smn.gob.ar/dpd/descarga_opendata.php"
    "?file=observaciones/datohorario{}.txt"
)

ESTACION = "BAHIA BLANCA AERO"


# ============================================================
# FUNCIONES GENERALES
# ============================================================

def numero(valor, defecto=0.0):
    try:
        return float(valor)
    except (TypeError, ValueError):
        return defecto


def leer_radar():
    if not RADAR_CSV.exists():
        return []

    try:
        with RADAR_CSV.open(
            "r",
            encoding="utf-8",
            newline=""
        ) as f:
            return list(csv.DictReader(f))

    except Exception:
        return []


def cargar_modelo():
    if not MODELO.exists():
        return None

    try:
        return joblib.load(MODELO)
    except Exception:
        return None


# ============================================================
# DATOS ACTUALES DEL SMN
# ============================================================

def obtener_smn(fecha):

    try:

        url = SMN_URL.format(
            fecha.strftime("%Y%m%d")
        )

        respuesta = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": "ClimaAR/1.0"
            }
        )

        respuesta.raise_for_status()

        texto = respuesta.content.decode(
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

            partes = linea.strip().split()

            if len(partes) < 8:
                continue

            estacion = " ".join(
                partes[7:]
            ).strip()

            if "BAHIA BLANCA" not in estacion.upper():
                continue

            datos.append({
                "fecha": partes[0],
                "hora": partes[1],
                "temperatura": partes[2],
                "humedad": partes[3],
                "presion": partes[4],
                "direccion_viento": partes[5],
                "velocidad_viento": partes[6],
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

    # Intentar varios días para garantizar
    # suficientes datos para los lags del modelo.
    for dias in range(0, 4):

        fecha = hoy - timedelta(days=dias)

        datos = obtener_smn(fecha)

        if datos:
            filas.extend(datos)

    if not filas:
        return None

    df = pd.DataFrame(filas)

    df["fecha_hora"] = pd.to_datetime(
        df["fecha"].astype(str)
        + " "
        + df["hora"].astype(str),
        errors="coerce"
    )

    columnas_numericas = [
        "temperatura",
        "humedad",
        "presion",
        "direccion_viento",
        "velocidad_viento"
    ]

    for columna in columnas_numericas:

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

    # Componentes del viento
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

    # Punto de rocío
    rh = df["humedad"].clip(1, 100)

    temperatura = df["temperatura"]

    gamma = (
        np.log(rh / 100)
        + (
            17.625
            * temperatura
            / (243.04 + temperatura)
        )
    )

    df["punto_rocio"] = (
        243.04
        * gamma
        / (17.625 - gamma)
    )

    # Hora cíclica
    hora = df["fecha_hora"].dt.hour

    df["hora_sin"] = np.sin(
        2 * np.pi * hora / 24
    )

    df["hora_cos"] = np.cos(
        2 * np.pi * hora / 24
    )

    # Lags
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
            ] = df[variable].shift(horas)

    return df


# ============================================================
# ANÁLISIS DEL RADAR
# ============================================================

def analizar_tormenta():

    filas = leer_radar()

    if not filas:

        return {
            "estado": "sin_datos",
            "mensaje": (
                "Todavía no hay datos de radar."
            )
        }

    recientes = filas[-12:]

    ultimo = recientes[-1]

    max_dbz = numero(
        ultimo.get("max_dbz"),
        0
    )

    media_dbz = numero(
        ultimo.get("mean_dbz"),
        0
    )

    ge10 = numero(
        ultimo.get("pixels_ge_10dbz"),
        0
    )

    ge20 = numero(
        ultimo.get("pixels_ge_20dbz"),
        0
    )

    ge30 = numero(
        ultimo.get("pixels_ge_30dbz"),
        0
    )

    ge40 = numero(
        ultimo.get("pixels_ge_40dbz"),
        0
    )

    if max_dbz < 20:

        return {
            "estado": "sin_tormenta_detectada",
            "ubicacion": "Bahía Blanca",
            "max_dbz": max_dbz,
            "mean_dbz": media_dbz,
            "pixels_ge_10dbz": int(ge10),
            "pixels_ge_20dbz": int(ge20),
            "pixels_ge_30dbz": int(ge30),
            "pixels_ge_40dbz": int(ge40),
            "frames_analizados": len(recientes),
            "mensaje": (
                "No se detecta actividad "
                "convectiva significativa "
                "en el área analizada."
            )
        }

    if max_dbz < 30:
        intensidad = "moderada"
    elif max_dbz < 40:
        intensidad = "fuerte"
    elif max_dbz < 50:
        intensidad = "muy_fuerte"
    else:
        intensidad = "extrema"

    tendencia = "estable"

    referencia = None

    for frame in reversed(
        recientes[:-1]
    ):

        if numero(
            frame.get("max_dbz"),
            0
        ) >= 20:

            referencia = frame
            break

    if referencia is not None:

        max_anterior = numero(
            referencia.get("max_dbz"),
            0
        )

        ge30_anterior = numero(
            referencia.get("pixels_ge_30dbz"),
            0
        )

        aumento_dbz = (
            max_dbz
            - max_anterior
        )

        if (
            aumento_dbz >= 3
            or (
                ge30_anterior > 0
                and ge30 >= ge30_anterior * 1.20
            )
        ):

            tendencia = "fortaleciendose"

        elif (
            aumento_dbz <= -3
            or (
                ge30_anterior > 0
                and ge30 <= ge30_anterior * 0.80
            )
        ):

            tendencia = "debilitandose"

    activos = []

    for frame in recientes:

        x = frame.get(
            "centroid_x_ge_20dbz"
        )

        y = frame.get(
            "centroid_y_ge_20dbz"
        )

        if (
            x not in ("", None)
            and y not in ("", None)
        ):

            activos.append(frame)

    movimiento = "indeterminado"
    direccion = None
    velocidad_px_h = None

    if len(activos) >= 2:

        anterior = activos[-2]
        actual = activos[-1]

        ax = numero(
            anterior.get(
                "centroid_x_ge_20dbz"
            )
        )

        ay = numero(
            anterior.get(
                "centroid_y_ge_20dbz"
            )
        )

        bx = numero(
            actual.get(
                "centroid_x_ge_20dbz"
            )
        )

        by = numero(
            actual.get(
                "centroid_y_ge_20dbz"
            )
        )

        ta = numero(
            anterior.get(
                "frame_time"
            )
        )

        tb = numero(
            actual.get(
                "frame_time"
            )
        )

        minutos = (
            tb - ta
        ) / 60

        dx = bx - ax
        dy = by - ay

        distancia = math.hypot(
            dx,
            dy
        )

        if minutos > 0:

            velocidad_px_h = round(
                distancia
                / minutos
                * 60,
                2
            )

        if distancia >= 1:

            movimiento = "en_movimiento"

            if abs(dx) < 1:

                direccion = (
                    "sur"
                    if dy > 0
                    else "norte"
                )

            elif abs(dy) < 1:

                direccion = (
                    "este"
                    if dx > 0
                    else "oeste"
                )

            else:

                eje_x = (
                    "este"
                    if dx > 0
                    else "oeste"
                )

                eje_y = (
                    "sur"
                    if dy > 0
                    else "norte"
                )

                direccion = (
                    f"{eje_y}-{eje_x}"
                )

    aproximacion = "indeterminada"

    try:

        ancho = numero(
            ultimo.get("width"),
            512
        )

        alto = numero(
            ultimo.get("height"),
            512
        )

        actual = activos[-1]

        x = numero(
            actual.get(
                "centroid_x_ge_20dbz"
            ),
            ancho / 2
        )

        y = numero(
            actual.get(
                "centroid_y_ge_20dbz"
            ),
            alto / 2
        )

        distancia_actual = math.hypot(
            x - ancho / 2,
            y - alto / 2
        )

        if len(activos) >= 2:

            anterior = activos[-2]

            px = numero(
                anterior.get(
                    "centroid_x_ge_20dbz"
                ),
                ancho / 2
            )

            py = numero(
                anterior.get(
                    "centroid_y_ge_20dbz"
                ),
                alto / 2
            )

            distancia_anterior = math.hypot(
                px - ancho / 2,
                py - alto / 2
            )

            if (
                distancia_actual
                < distancia_anterior - 1
            ):

                aproximacion = "acercandose"

            elif (
                distancia_actual
                > distancia_anterior + 1
            ):

                aproximacion = "alejandose"

            else:

                aproximacion = "sin_cambio_claro"

    except Exception:

        aproximacion = "indeterminada"

    situacion = "normal"

    if max_dbz >= 30:
        situacion = "vigilancia"

    if max_dbz >= 40:
        situacion = "atencion"

    if (
        max_dbz >= 40
        and tendencia == "fortaleciendose"
    ):
        situacion = "atencion_alta"

    return {
        "estado": "tormenta_detectada",
        "ubicacion": "Bahía Blanca",
        "intensidad": intensidad,
        "max_dbz": max_dbz,
        "mean_dbz": media_dbz,
        "pixels_ge_10dbz": int(ge10),
        "pixels_ge_20dbz": int(ge20),
        "pixels_ge_30dbz": int(ge30),
        "pixels_ge_40dbz": int(ge40),
        "tendencia_intensidad": tendencia,
        "movimiento": movimiento,
        "direccion_movimiento": direccion,
        "velocidad_movimiento_px_h": velocidad_px_h,
        "aproximacion_a_bahia_blanca": aproximacion,
        "situacion": situacion,
        "frames_analizados": len(recientes),
        "frames_con_reflectividad_ge_20dbz": len(activos),
        "hora_frame": ultimo.get("frame_utc")
    }


# ============================================================
# ENDPOINT PRINCIPAL
# ============================================================

@app.get("/")
def inicio():

    return {
        "app": "ClimaAR",
        "version": "1.0.0",
        "estado": "activo",
        "radar": "/radar",
        "tormenta": "/tormenta",
        "prediccion": "/prediccion",
        "modelo": "/modelo",
        "estado_general": "/estado",
        "salud": "/health"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "estado": "ok",
        "servicio": "ClimaAR",
        "hora_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    }


# ============================================================
# RADAR
# ============================================================

@app.get("/radar")
def radar():

    datos = leer_radar()

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


# ============================================================
# ANÁLISIS DE TORMENTA
# ============================================================

@app.get("/tormenta")
def tormenta():

    return analizar_tormenta()


# ============================================================
# MÉTRICAS DEL MODELO
# ============================================================

@app.get("/modelo")
def modelo():

    if not METRICAS.exists():

        return {
            "estado": "pendiente",
            "mensaje": (
                "Métricas del modelo "
                "no disponibles."
            )
        }

    try:

        with METRICAS.open(
            "r",
            encoding="utf-8"
        ) as f:

            metricas = json.load(f)

        return {
            "estado": "entrenado",
            "modelo": metricas
        }

    except Exception as e:

        return {
            "estado": "error",
            "mensaje": str(e)
        }


# ============================================================
# PREDICCIÓN DE TEMPERATURA
# ============================================================

@app.get("/prediccion")
def prediccion():

    modelo = cargar_modelo()

    if modelo is None:

        return {
            "estado": "pendiente",
            "mensaje": (
                "Modelo no disponible."
            )
        }

    df = obtener_datos_actuales()

    if df is None:

        return {
            "estado": "sin_datos",
            "mensaje": (
                "No se pudieron obtener "
                "datos actuales del SMN."
            )
        }

    fila = df.iloc[-1]

    try:

        columnas = list(
            modelo.feature_names_in_
        )

        X = pd.DataFrame(
            [
                [
                    fila.get(
                        columna,
                        np.nan
                    )
                    for columna in columnas
                ]
            ],
            columns=columnas
        )

        if X.isna().any().any():

            faltantes = list(
                X.columns[
                    X.isna().iloc[0]
                ]
            )

            return {
                "estado": (
                    "sin_datos_suficientes"
                ),
                "faltantes": faltantes
            }

        prediccion = float(
            modelo.predict(X)[0]
        )

        actual = float(
            fila["temperatura"]
        )

        return {
            "estado": "ok",
            "ubicacion": "Bahía Blanca",
            "hora_dato": str(
                fila["fecha_hora"]
            ),
            "temperatura_actual": round(
                actual,
                2
            ),
            "temperatura_1h": round(
                prediccion,
                2
            ),
            "variacion_1h": round(
                prediccion - actual,
                2
            )
        }

    except Exception as e:

        return {
            "estado": "error",
            "mensaje": str(e)
        }


# ============================================================
# ESTADO GENERAL
# ============================================================

@app.get("/estado")
def estado():

    datos = leer_radar()

    tormenta_actual = analizar_tormenta()

    return {
        "climaar": "activo",
        "version": "1.0.0",
        "radar_frames": len(datos),
        "modelo_disponible": MODELO.exists(),
        "metricas_disponibles": METRICAS.exists(),
        "analisis_tormenta": tormenta_actual,
        "hora_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        )
            }
