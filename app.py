from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from datetime import datetime, timezone
from pathlib import Path
import csv
import json

app = FastAPI(
    title="ClimaAR",
    description="Servicio meteorológico de ClimaAR",
    version="0.4.0"
)

RADAR_IMAGE_URL = (
    "https://estaticos.smn.gob.ar/vmsr/radar/"
    "RMA10_240_ZH_CMAX_20260928_091811Z.png"
)

RADAR_CSV = Path("data/radar/radar_features.csv")
METRICAS = Path("modelo/metricas_modelo.json")


@app.get("/")
def inicio():
    return {
        "app": "ClimaAR",
        "estado": "activo",
        "radar": "/radar",
        "salud": "/health"
    }


@app.get("/health")
def health():
    return {
        "estado": "ok",
        "servicio": "ClimaAR",
        "hora_utc": datetime.now(timezone.utc).isoformat()
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
            filas = list(csv.DictReader(f))
            datos = filas[-20:]

    ultimo = datos[-1] if datos else None

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
            "mensaje": "Métricas del modelo todavía no disponibles."
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


@app.get("/estado")
def estado():
    frames = 0

    if RADAR_CSV.exists():
        with RADAR_CSV.open(
            "r",
            encoding="utf-8",
            newline=""
        ) as f:
            frames = sum(1 for _ in f) - 1

    return {
        "climaar": "activo",
        "radar_frames": max(frames, 0),
        "modelo_disponible": METRICAS.exists(),
        "hora_utc": datetime.now(timezone.utc).isoformat()
    }
