"""
ClimaAR - Collector GOES-19

Descarga el último archivo disponible de:
- GLM-L2-LCFA
- ABI-L2-CTPF
- ABI-L2-CMIPF
- ABI-L2-DMWF

Fuente:
NOAA Open Data / AWS S3

IMPORTANTE:
- No modifica RMA10.
- No modifica app.py.
- No modifica la IA.
- No requiere credenciales AWS.
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
import xml.etree.ElementTree as ET
import requests


SATELLITE = "noaa-goes19"

BASE_URL = f"https://{SATELLITE}.s3.amazonaws.com"

BASE_DIR = Path(__file__).resolve().parent.parent.parent

GOES_DIR = BASE_DIR / "data" / "goes"
RAW_DIR = GOES_DIR / "raw"
LATEST_DIR = GOES_DIR / "latest"


PRODUCTS = {
    "GLM": "GLM-L2-LCFA",
    "CLOUD_TOP": "ABI-L2-CTPF",
    "ABI_CMI": "ABI-L2-CMIPF",
    "DMW": "ABI-L2-DMWF",
}


TIMEOUT = 60


def utc_now():
    return datetime.now(timezone.utc)


def julian_day(dt):
    return dt.timetuple().tm_yday


def listar_archivos(prefix):
    url = BASE_URL + "/"

    params = {
        "list-type": "2",
        "prefix": prefix,
        "max-keys": "1000",
    }

    response = requests.get(
        url,
        params=params,
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    root = ET.fromstring(response.text)

    namespace = {
        "s3": "http://s3.amazonaws.com/doc/2006-03-01/"
    }

    archivos = []

    for content in root.findall(
        "s3:Contents",
        namespace,
    ):
        key = content.find(
            "s3:Key",
            namespace,
        )

        last_modified = content.find(
            "s3:LastModified",
            namespace,
        )

        size = content.find(
            "s3:Size",
            namespace,
        )

        if key is None:
            continue

        archivos.append(
            {
                "key": key.text,
                "last_modified": (
                    last_modified.text
                    if last_modified is not None
                    else None
                ),
                "size": int(size.text)
                if size is not None
                else 0,
            }
        )

    return archivos


def buscar_ultimo(product_prefix):
    ahora = utc_now()

    # Buscamos hacia atrás hasta 6 horas.
    for offset in range(0, 7):
        momento = ahora - timedelta(hours=offset)

        year = momento.year
        day = julian_day(momento)
        hour = momento.hour

        prefix = (
            f"{product_prefix}/"
            f"{year}/"
            f"{day:03d}/"
            f"{hour:02d}/"
        )

        archivos = listar_archivos(prefix)

        if archivos:
            archivos.sort(
                key=lambda x: x["key"],
                reverse=True,
            )

            return archivos[0]

    return None


def descargar(archivo, producto):
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LATEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    key = archivo["key"]

    nombre = Path(key).name

    destino_raw = RAW_DIR / nombre
    destino_latest = LATEST_DIR / f"{producto}.nc"

    url = f"{BASE_URL}/{key}"

    print()
    print(f"GOES-19 | {producto}")
    print(f"Archivo: {nombre}")
    print(f"Tamaño: {archivo['size'] / 1024 / 1024:.1f} MB")
    print("Descargando...")

    with requests.get(
        url,
        stream=True,
        timeout=TIMEOUT,
    ) as response:

        response.raise_for_status()

        with open(
            destino_raw,
            "wb",
        ) as f:

            for bloque in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if bloque:
                    f.write(bloque)

    # Copia con nombre estable para el procesamiento.
    with open(
        destino_raw,
        "rb",
    ) as origen:

        with open(
            destino_latest,
            "wb",
        ) as destino:

            while True:
                bloque = origen.read(
                    1024 * 1024
                )

                if not bloque:
                    break

                destino.write(bloque)

    return {
        "producto": producto,
        "archivo": nombre,
        "url": url,
        "tamano_bytes": archivo["size"],
        "estado": "OK",
        "timestamp": utc_now().isoformat(),
    }


def main():
    print("=" * 60)
    print("ClimaAR - COLLECTOR GOES-19")
    print("=" * 60)

    resultados = []

    for producto, product_prefix in PRODUCTS.items():

        try:
            archivo = buscar_ultimo(
                product_prefix
            )

            if archivo is None:
                print(
                    f"{producto}: SIN DATOS"
                )

                resultados.append(
                    {
                        "producto": producto,
                        "estado": "SIN_DATOS",
                    }
                )

                continue

            resultado = descargar(
                archivo,
                producto,
            )

            resultados.append(
                resultado
            )

            print(
                f"{producto}: OK"
            )

        except Exception as error:

            print(
                f"{producto}: ERROR"
            )

            print(
                str(error)
            )

            resultados.append(
                {
                    "producto": producto,
                    "estado": "ERROR",
                    "detalle": str(error),
                }
            )

    print()
    print("=" * 60)
    print("RESULTADO")
    print("=" * 60)

    for resultado in resultados:
        print(
            resultado["producto"],
            "->",
            resultado["estado"],
        )


if __name__ == "__main__":
    main()
