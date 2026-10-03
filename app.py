from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
import csv
import json
import math

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse

app = FastAPI(
    title="ClimaAR",
    description="Seguimiento RMA10 y nowcasting experimental de tormentas para Bahía Blanca.",
    version="2.0.0",
)

RADAR_PAGE_URL = "https://ws2.smn.gob.ar/radar"
RADAR_CSV = Path("data/radar/radar_features.csv")
RADAR_LATEST = Path("data/radar/latest.png")
RADAR_RAW = Path("data/radar/raw")
METRICAS = Path("modelo/metricas_modelo.json")
MODELO = Path("modelo/climaar_modelo_temperatura_1h.joblib")

SMN_URL = (
    "https://ssl.smn.gob.ar/dpd/descarga_opendata.php"
    "?file=observaciones/datohorario{}.txt"
)


def numero(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def leer_radar() -> list[dict]:
    if not RADAR_CSV.exists():
        return []
    try:
        with RADAR_CSV.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.sort(key=lambda r: numero(r.get("frame_time"), 0))
        return rows
    except Exception:
        return []


def cargar_modelo():
    if not MODELO.exists():
        return None
    try:
        return joblib.load(MODELO)
    except Exception:
        return None


def obtener_smn(fecha):
    try:
        url = SMN_URL.format(fecha.strftime("%Y%m%d"))
        r = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "ClimaAR/2.0"},
        )
        r.raise_for_status()
        text = r.content.decode("latin-1", errors="replace")
        if not text.strip() or "El archivo no existe." in text:
            return []

        rows = []
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            station = " ".join(parts[7:]).strip()
            if "BAHIA BLANCA" not in station.upper():
                continue
            rows.append(
                {
                    "fecha": parts[0],
                    "hora": parts[1],
                    "temperatura": parts[2],
                    "humedad": parts[3],
                    "presion": parts[4],
                    "direccion_viento": parts[5],
                    "velocidad_viento": parts[6],
                    "estacion": station,
                }
            )
        return rows
    except Exception:
        return []


def obtener_datos_actuales():
    today = datetime.now(timezone.utc).date()
    rows = []
    for days in range(4):
        rows.extend(obtener_smn(today - timedelta(days=days)))

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["fecha_hora"] = pd.to_datetime(
        df["fecha"].astype(str) + " " + df["hora"].astype(str),
        errors="coerce",
    )
    numeric = [
        "temperatura",
        "humedad",
        "presion",
        "direccion_viento",
        "velocidad_viento",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["fecha_hora"])
        .sort_values("fecha_hora")
        .drop_duplicates("fecha_hora")
        .reset_index(drop=True)
    )
    if df.empty:
        return None

    direction = np.deg2rad(df["direccion_viento"])
    df["viento_u"] = -df["velocidad_viento"] * np.sin(direction)
    df["viento_v"] = -df["velocidad_viento"] * np.cos(direction)

    rh = df["humedad"].clip(1, 100)
    t = df["temperatura"]
    gamma = np.log(rh / 100) + (17.625 * t) / (243.04 + t)
    df["punto_rocio"] = 243.04 * gamma / (17.625 - gamma)

    hour = df["fecha_hora"].dt.hour
    df["hora_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hora_cos"] = np.cos(2 * np.pi * hour / 24)

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
        "hora_cos",
    ]
    for lag in [1, 3, 6, 24]:
        for var in variables:
            df[f"{var}_lag{lag}"] = df[var].shift(lag)
    return df


def ultimos_frames(n=4):
    rows = leer_radar()
    return rows[-n:]


def tendencia_lineal(rows, key):
    pairs = []
    for row in rows:
        value = row.get(key)
        time = numero(row.get("frame_time"), math.nan)
        value = numero(value, math.nan)
        if math.isfinite(time) and math.isfinite(value):
            pairs.append((time, value))
    if len(pairs) < 2:
        return None

    t0 = pairs[0][0]
    x = np.array([(t - t0) / 60.0 for t, _ in pairs], dtype=float)
    y = np.array([v for _, v in pairs], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), float(x[-1]), float(y[-1])


def analizar_tormenta():
    rows = leer_radar()
    if not rows:
        return {
            "estado": "sin_datos",
            "fuente": "SMN RMA10 ZH_MAX",
            "mensaje": "Todavía no hay frames RMA10 procesados.",
        }

    recent = rows[-12:]
    last = recent[-1]
    max_dbz = numero(last.get("max_dbz"), 0)
    mean_dbz = numero(last.get("mean_dbz"), 0)

    ge = {
        str(t): int(numero(last.get(f"pixels_ge_{t}dbz"), 0))
        for t in [20, 30, 40, 45, 50, 55, 60]
    }

    if max_dbz < 20:
        intensity = "sin_reflectividad_significativa"
    elif max_dbz < 30:
        intensity = "débil"
    elif max_dbz < 40:
        intensity = "fuerte"
    elif max_dbz < 50:
        intensity = "muy_fuerte"
    else:
        intensity = "muy_alta"

    slope = tendencia_lineal(recent[-4:], "max_dbz")
    if slope is None:
        trend = "indeterminada"
        slope_dbz_10m = None
    else:
        slope_dbz_min = slope[0]
        slope_dbz_10m = round(slope_dbz_min * 10, 2)
        if slope_dbz_10m >= 1.0:
            trend = "fortaleciendose"
        elif slope_dbz_10m <= -1.0:
            trend = "debilitandose"
        else:
            trend = "estable"

    movement = None
    speed_px_h = None
    direction = None
    active = []
    for row in recent:
        x = row.get("centroid_x_ge_40dbz")
        y = row.get("centroid_y_ge_40dbz")
        if x not in ("", None) and y not in ("", None):
            active.append(row)

    if len(active) >= 2:
        a, b = active[-2], active[-1]
        ax, ay = numero(a.get("centroid_x_ge_40dbz")), numero(a.get("centroid_y_ge_40dbz"))
        bx, by = numero(b.get("centroid_x_ge_40dbz")), numero(b.get("centroid_y_ge_40dbz"))
        dt = (numero(b.get("frame_time")) - numero(a.get("frame_time"))) / 60
        dx, dy = bx - ax, by - ay
        dist = math.hypot(dx, dy)
        if dt > 0:
            speed_px_h = round(dist / dt * 60, 2)
        if dist >= 1:
            movement = "en_movimiento"
            if abs(dx) >= abs(dy):
                direction = "este" if dx > 0 else "oeste"
            else:
                direction = "sur" if dy > 0 else "norte"
    else:
        movement = "sin_seguimiento_suficiente"

    return {
        "estado": "datos_radar_disponibles",
        "ubicacion": "Bahía Blanca",
        "fuente": "SMN RMA10 ZH_MAX",
        "intensidad_actual": intensity,
        "max_dbz": max_dbz,
        "mean_dbz": mean_dbz,
        "pixeles_por_umbral_dbz": ge,
        "tendencia_max_dbz": trend,
        "cambio_max_dbz_por_10min": slope_dbz_10m,
        "movimiento_celda_ge_40dbz": movement,
        "direccion_movimiento": direction,
        "velocidad_movimiento_px_h": speed_px_h,
        "frames_analizados": len(recent),
        "ultimo_frame_utc": last.get("frame_utc"),
    }


def cargar_nowcast():
    path = Path("modelo/climaar_nowcast_features.joblib")
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def prediccion_nowcast_ia(rows):
    artifact = cargar_nowcast()
    if artifact is None or len(rows) < 4:
        return None

    try:
        base = [
            "max_dbz",
            "mean_dbz",
            "pixels_ge_20dbz",
            "pixels_ge_30dbz",
            "pixels_ge_40dbz",
            "pixels_ge_45dbz",
            "pixels_ge_50dbz",
            "pixels_ge_55dbz",
            "pixels_ge_60dbz",
            "centroid_x_ge_40dbz",
            "centroid_y_ge_40dbz",
        ]
        values = []
        for row in rows[-4:]:
            for col in base:
                value = numero(row.get(col), 0)
                values.append(value)

        X = pd.DataFrame([values], columns=artifact["features"])
        max_model = artifact["models"]["target_max_dbz"]
        area_model = artifact["models"]["target_pixels_ge_40dbz"]

        max_pred = float(np.clip(max_model.predict(X)[0], -15, 75))
        area_pred = max(0.0, float(area_model.predict(X)[0]))

        return {
            "tipo": "modelo_ia_rma10",
            "horizonte_minutos": 10,
            "max_dbz_t_plus_10": round(max_pred, 1),
            "pixels_ge_40dbz_t_plus_10": round(area_pred, 1),
            "modelo_disponible": True,
            "nota": "Predicción del primer modelo entrenado con secuencias RMA10 propias.",
        }
    except Exception:
        return None


def nowcast_baseline():
    rows = ultimos_frames(4)
    if len(rows) < 2:
        return {
            "estado": "pendiente",
            "tipo": "baseline_experimental",
            "mensaje": "Se necesitan al menos 2 frames RMA10 para calcular tendencia.",
            "frames_disponibles": len(rows),
        }

    ia = prediccion_nowcast_ia(rows)
    if ia is not None:
        return {
            "estado": "ok",
            "tipo": "modelo_ia_rma10",
            "frames_utilizados": 4,
            "prediccion_10min": ia,
            "baseline": "disponible_en_codigo_si_se_necesita_comparacion",
        }

    slope = tendencia_lineal(rows, "max_dbz")
    if slope is None:
        return {"estado": "pendiente", "mensaje": "Sin suficientes datos numéricos."}

    slope_dbz_min, _, _, last_y = slope
    forecasts = []
    for minutes in [10, 20, 30]:
        predicted = float(np.clip(last_y + slope_dbz_min * minutes, -15, 75))
        forecasts.append(
            {
                "minutos": minutes,
                "max_dbz_estimado": round(predicted, 1),
                "tendencia": (
                    "fortalecimiento"
                    if slope_dbz_min > 0.1
                    else "debilitamiento"
                    if slope_dbz_min < -0.1
                    else "estable"
                ),
            }
        )

    confidence = round(0.45 + 0.45 * min(1.0, len(rows) / 4.0), 2)
    return {
        "estado": "ok",
        "tipo": "baseline_experimental",
        "nota": "Extrapolación temporal; será reemplazada/contrastada por el modelo de IA cuando haya suficientes secuencias.",
        "confianza_baseline": confidence,
        "frames_utilizados": len(rows),
        "ultimo_max_dbz": round(last_y, 1),
        "cambio_dbz_por_10min": round(slope_dbz_min * 10, 2),
        "pronostico": forecasts,
    }


@app.get("/")
def inicio():
    return {
        "app": "ClimaAR",
        "version": "2.0.0",
        "estado": "activo",
        "radar": "/radar",
        "radar_imagen": "/radar/imagen",
        "tormenta": "/tormenta",
        "nowcast": "/nowcast",
        "secuencia": "/secuencia",
        "modelo_ambiental": "/modelo",
        "modelo_nowcast": "/modelo/nowcast",
        "prediccion_temperatura": "/prediccion",
        "health": "/health",
    }


@app.get("/health")
def health():
    rows = leer_radar()
    return {
        "estado": "ok",
        "servicio": "ClimaAR",
        "fuente_radar": "SMN RMA10 ZH_MAX",
        "frames_procesados": len(rows),
        "ultimo_frame_utc": rows[-1].get("frame_utc") if rows else None,
        "hora_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/radar")
def radar():
    rows = leer_radar()
    return {
        "estado": "ok",
        "radar": "RMA10 Bahía Blanca",
        "producto": "ZH_MAX",
        "fuente_oficial": RADAR_PAGE_URL,
        "frames_guardados": len(rows),
        "ultimo_frame": rows[-1] if rows else None,
        "imagen_local_disponible": RADAR_LATEST.exists(),
    }


@app.get("/radar/imagen")
def radar_imagen():
    if RADAR_LATEST.exists():
        return FileResponse(
            RADAR_LATEST,
            media_type="image/png",
            filename="RMA10_ZH_MAX_latest.png",
        )
    return RedirectResponse(url=RADAR_PAGE_URL)


@app.get("/secuencia")
def secuencia():
    return {
        "estado": "ok",
        "fuente": "SMN RMA10 ZH_MAX",
        "frames_requeridos_para_modelo": 4,
        "frames": ultimos_frames(4),
    }


@app.get("/tormenta")
def tormenta():
    return analizar_tormenta()


@app.get("/nowcast")
def nowcast():
    return nowcast_baseline()


@app.get("/modelo")
def modelo():
    if not METRICAS.exists():
        return {
            "estado": "pendiente",
            "tipo": "baseline_ambiental",
            "mensaje": "Métricas del modelo no disponibles.",
        }
    try:
        return {
            "estado": "entrenado",
            "tipo": "baseline_ambiental",
            "modelo": json.loads(METRICAS.read_text(encoding="utf-8")),
        }
    except Exception as exc:
        return {"estado": "error", "mensaje": str(exc)}


@app.get("/modelo/nowcast")
def modelo_nowcast():
    path = Path("modelo/metricas_nowcast.json")
    if not path.exists():
        return {
            "estado": "pendiente",
            "mensaje": "El primer modelo RMA10 todavía no tiene suficientes secuencias para entrenarse.",
        }
    try:
        return {
            "estado": "entrenado",
            "modelo": json.loads(path.read_text(encoding="utf-8")),
        }
    except Exception as exc:
        return {"estado": "error", "mensaje": str(exc)}


@app.get("/prediccion")
def prediccion():
    model = cargar_modelo()
    if model is None:
        return {"estado": "pendiente", "mensaje": "Modelo ambiental no disponible."}

    df = obtener_datos_actuales()
    if df is None:
        return {"estado": "sin_datos", "mensaje": "No se pudieron obtener datos actuales del SMN."}

    row = df.iloc[-1]
    try:
        columns = list(model.feature_names_in_)
        X = pd.DataFrame(
            [[row.get(column, np.nan) for column in columns]],
            columns=columns,
        )
        if X.isna().any().any():
            return {
                "estado": "sin_datos_suficientes",
                "faltantes": list(X.columns[X.isna().iloc[0]]),
            }

        predicted = float(model.predict(X)[0])
        current = float(row["temperatura"])
        return {
            "estado": "ok",
            "tipo": "baseline_ambiental",
            "ubicacion": "Bahía Blanca",
            "hora_dato": str(row["fecha_hora"]),
            "temperatura_actual": round(current, 2),
            "temperatura_1h": round(predicted, 2),
            "variacion_1h": round(predicted - current, 2),
        }
    except Exception as exc:
        return {"estado": "error", "mensaje": str(exc)}


@app.get("/estado")
def estado():
    rows = leer_radar()
    return {
        "climaar": "activo",
        "version": "2.0.0",
        "fuente_radar": "SMN RMA10 ZH_MAX",
        "radar_frames": len(rows),
        "modelo_ambiental_disponible": MODELO.exists(),
        "nowcast_baseline": nowcast_baseline(),
        "analisis_tormenta": analizar_tormenta(),
        "hora_utc": datetime.now(timezone.utc).isoformat(),
    }
