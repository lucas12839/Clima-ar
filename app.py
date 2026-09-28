
from fastapi import FastAPI
from datetime import datetime, timezone
from fastapi.responses import RedirectResponse

app = FastAPI(
    title="ClimaAR",
    description="Servicio meteorológico de ClimaAR",
    version="0.3.0"
)

RADAR_IMAGE_URL = (
    "https://estaticos.smn.gob.ar/vmsr/radar/"
    "RMA10_240_ZH_CMAX_20260928_091811Z.png"
)


@app.get("/")
def inicio():
    return {
        "app": "ClimaAR",
        "estado": "inicial",
        "mensaje": "Servicio iniciado correctamente.",
        "radar": "/radar"
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
    return {
        "estado": "prueba",
        "radar": "Bahía Blanca",
        "imagen_url": RADAR_IMAGE_URL,
        "actualizacion_automatica": False,
        "mensaje": "Imagen de radar de prueba del SMN."
    }


@app.get("/radar/image")
@app.get("/radar/imagen")
def radar_imagen():
    return RedirectResponse(url=RADAR_IMAGE_URL)
