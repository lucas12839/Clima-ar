"""
ClimaAR - Procesamiento inicial GOES-19

IMPORTANTE:
Esta versión NO incorpora todavía GOES a la inteligencia de ClimaAR.

Su función es:
- comprobar disponibilidad
- registrar productos
- mantener separación entre GOES y RMA10
- preparar la arquitectura para la siguiente etapa
"""

from pathlib import Path
from datetime import datetime, timezone
import json


PRODUCTOS_VALIDOS = {
    "GLM": {
        "nombre": "Geostationary Lightning Mapper",
        "descripcion": "Actividad eléctrica total",
    },
    "CLOUD_TOP": {
        "nombre": "Cloud Top Properties",
        "descripcion": "Altura, temperatura y presión de cima nubosa",
    },
    "ABI_CMI": {
        "nombre": "ABI Cloud and Moisture Imagery",
        "descripcion": "Estructura nubosa y vapor de agua",
    },
    "DMW": {
        "nombre": "Derived Motion Winds",
        "descripcion": "Vientos derivados por niveles",
    },
}


def procesar_goes(
    producto,
    raw_dir: Path,
    latest_dir: Path,
    features_dir: Path,
):
    if producto not in PRODUCTOS_VALIDOS:
        raise ValueError(
            f"Producto GOES no soportado: {producto}"
        )

    ahora = datetime.now(timezone.utc)

    informacion = PRODUCTOS_VALIDOS[producto]

    resultado = {
        "timestamp": ahora.isoformat(),
        "fuente": "GOES-19",
        "producto": producto,
        "nombre": informacion["nombre"],
        "descripcion": informacion["descripcion"],
        "estado": "PENDIENTE_DATOS",
        "archivo": None,
    }

    archivo_estado = (
        features_dir / f"{producto.lower()}_estado.json"
    )

    with open(
        archivo_estado,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            resultado,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return resultado
