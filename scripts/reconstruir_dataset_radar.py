#!/usr/bin/env python3

"""
ClimaAR - Reconstrucción del dataset histórico de radar.

Toma los PNG existentes en:

    data/radar/historico/

y vuelve a calcular:

    area_px
    distance_km
    dbz_max
    dbz_mean
    dbz_p90
    dbz_pixels
    nucleos

No descarga nada de Internet.

Utiliza exactamente el mismo decodificador
dBZ del radar RainViewer V7.1.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from PIL import Image

from scripts.radar_rainviewer import analyze_frame


# ============================================================
# ARCHIVOS
# ============================================================

HISTORY_DIR = Path(
    "data/radar/historico"
)

OUTPUT_CSV = Path(
    "data/radar/radar_features_rainviewer.csv"
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Los PNG históricos fueron generados como mosaicos
# de 3 x 3 tiles.
#
# El radar V7.1 utiliza 9 tiles cuando todos fueron
# descargados correctamente.

TILES_OK = 9


# ============================================================
# EXTRAER TIMESTAMP
# ============================================================

def extract_timestamp(path: Path):

    match = re.match(
        r"radar_(\d+)\.png$",
        path.name
    )

    if not match:
        return None

    return int(
        match.group(1)
    )


# ============================================================
# PROCESAR
# ============================================================

def main():

    print("")
    print("=" * 60)
    print("CLIMAAR - RECONSTRUCCIÓN DATASET RADAR")
    print("=" * 60)
    print("")

    if not HISTORY_DIR.exists():

        raise SystemExit(
            f"ERROR: no existe {HISTORY_DIR}"
        )

    files = []

    for path in HISTORY_DIR.glob(
        "radar_*.png"
    ):

        timestamp = extract_timestamp(
            path
        )

        if timestamp is not None:

            files.append(
                (
                    timestamp,
                    path
                )
            )

    files.sort(
        key=lambda item: item[0]
    )

    print(
        f"PNG encontrados: {len(files)}"
    )

    if len(files) < 20:

        raise SystemExit(
            "ERROR: hay menos de 20 PNG históricos. "
            "No se puede reconstruir un dataset "
            "temporal confiable."
        )

    print("")

    results = []

    # ========================================================
    # ANALIZAR CADA FRAME
    # ========================================================

    for index, (
        timestamp,
        path
    ) in enumerate(
        files,
        start=1
    ):

        print(
            f"[{index}/{len(files)}] "
            f"{path.name}"
        )

        try:

            image = Image.open(
                path
            ).convert(
                "RGBA"
            )

            result = analyze_frame(
                image=image,
                timestamp=timestamp,
                tiles_ok=TILES_OK,
                path=""
            )

            results.append(
                result
            )

            print(
                "    "
                f"area={result['area']} | "
                f"dBZmax={result['dbz_max']} | "
                f"p90={result['dbz_p90']} | "
                f"pix={result['dbz_pixels']} | "
                f"nucleos={len(result['components'])}"
            )

        except Exception as exc:

            print(
                "    ERROR:",
                exc
            )

    # ========================================================
    # VALIDACIÓN
    # ========================================================

    if not results:

        raise SystemExit(
            "ERROR: no se pudo analizar ningún PNG."
        )

    results.sort(
        key=lambda item:
        item["timestamp"]
    )

    print("")
    print(
        f"Frames reconstruidos: {len(results)}"
    )

    # ========================================================
    # GUARDAR CSV
    # ========================================================

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [

        "frame_time",
        "frame_utc",
        "source",
        "latitud",
        "longitud",
        "area_px",
        "distance_km",
        "dbz_max",
        "dbz_mean",
        "dbz_p90",
        "dbz_pixels",
        "tiles_ok",
        "nucleos"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for item in results:

            writer.writerow({

                "frame_time":
                    item["timestamp"],

                "frame_utc":
                    item["frame_utc"],

                "source":
                    "RainViewer",

                "latitud":
                    -38.71,

                "longitud":
                    -62.26,

                "area_px":
                    item["area"],

                "distance_km":
                    item["distance_km"],

                "dbz_max":
                    item["dbz_max"],

                "dbz_mean":
                    item["dbz_mean"],

                "dbz_p90":
                    item["dbz_p90"],

                "dbz_pixels":
                    item["dbz_pixels"],

                "tiles_ok":
                    item["tiles_ok"],

                "nucleos":
                    json.dumps(
                        item["components"],
                        ensure_ascii=False
                    )
            })

    # ========================================================
    # RESUMEN
    # ========================================================

    print("")
    print("=" * 60)
    print("RECONSTRUCCIÓN COMPLETADA")
    print("=" * 60)
    print("")
    print(
        f"Dataset: {OUTPUT_CSV}"
    )
    print(
        f"Frames: {len(results)}"
    )
    print("")

    # Estadísticas rápidas

    precipitation_frames = sum(
        1
        for item in results
        if item["area"] > 0
    )

    print(
        f"Frames con precipitación: "
        f"{precipitation_frames}"
    )

    print(
        f"Frames sin precipitación: "
        f"{len(results) - precipitation_frames}"
    )

    print("")


if __name__ == "__main__":

    main()
