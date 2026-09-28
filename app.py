from fastapi import FastAPI
from datetime import datetime, timezone

app = FastAPI(
    title="ClimaAR",
    description="Servicio inicial de ClimaAR",
    version="0.1.0"
)

@app.get("/")
def inicio():
    return {
        "app": "ClimaAR",
        "estado": "inicial",
        "mensaje": "Servicio iniciado. La conexión automática al radar todavía no está configurada."
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
        "imagen_url": "https://estaticos.smn.gob.ar/vmsr/radar/RMA10_240_ZH_CMAX_20260928_091811Z.png"
    }
