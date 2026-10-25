"""
ClimaAR - Ciclo GOES-19

Primera etapa:
- Descarga datos GOES-19
- Guarda archivos originales
- Procesa productos
- NO modifica RMA10
- NO modifica la IA
- NO modifica app.py
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import logging

from procesar_goes import procesar_goes


BASE_DIR = Path(__file__).resolve().parent.parent

GOES_DIR = BASE_DIR / "data" / "goes"
RAW_DIR = GOES_DIR / "raw"
LATEST_DIR = GOES_DIR / "latest"
FEATURES_DIR = GOES_DIR / "features"


PRODUCTOS = [
    "GLM",
    "CLOUD_TOP",
    "ABI_CMI",
    "DMW",
]


def preparar_directorios():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def crear_estado(producto, estado, archivo=None, detalle=None):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fuente": "GOES-19",
        "producto": producto,
        "archivo": archivo,
        "estado": estado,
        "detalle": detalle,
    }


def guardar_estado(estado):
    archivo = FEATURES_DIR / "estado_goes.json"

    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(
            estado,
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | GOES | %(levelname)s | %(message)s",
    )

    preparar_directorios()

    logging.info("Iniciando ciclo GOES-19")

    resultados = []

    for producto in PRODUCTOS:
        try:
            logging.info("Procesando producto: %s", producto)

            resultado = procesar_goes(
                producto=producto,
                raw_dir=RAW_DIR,
                latest_dir=LATEST_DIR,
                features_dir=FEATURES_DIR,
            )

            resultados.append(resultado)

        except Exception as e:
            logging.exception(
                "Error procesando %s",
                producto,
            )

            resultados.append(
                crear_estado(
                    producto=producto,
                    estado="ERROR",
                    detalle=str(e),
                )
            )

    guardar_estado(resultados)

    logging.info("Ciclo GOES-19 finalizado")


if __name__ == "__main__":
    main()
