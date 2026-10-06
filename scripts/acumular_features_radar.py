#!/usr/bin/env python3

import csv
import sys
from pathlib import Path


CURRENT = Path(
    "data/radar/radar_features_rainviewer.csv"
)

BACKUP = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path("/tmp/radar_features_previous.csv")
)

# 4320 frames = aproximadamente 30 días
# tomando un frame cada 10 minutos.
MAX_FEATURE_FRAMES = 4320


FIELDS = [
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
    "nucleos",
]


def read_rows(path):

    if (
        not path.exists()
        or
        path.stat().st_size == 0
    ):
        return {}

    rows = {}

    with path.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            try:
                timestamp = int(
                    float(
                        row.get(
                            "frame_time",
                            ""
                        )
                    )
                )

            except (
                TypeError,
                ValueError
            ):
                continue

            clean = {
                field:
                    row.get(
                        field,
                        ""
                    )
                for field in FIELDS
            }

            clean["frame_time"] = str(
                timestamp
            )

            rows[timestamp] = clean

    return rows


def main():

    previous = read_rows(
        BACKUP
    )

    current = read_rows(
        CURRENT
    )

    # El frame nuevo reemplaza al antiguo
    # si tienen el mismo timestamp.
    merged = previous

    merged.update(
        current
    )

    timestamps = sorted(
        merged
    )[
        -MAX_FEATURE_FRAMES:
    ]

    CURRENT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with CURRENT.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS
        )

        writer.writeheader()

        for timestamp in timestamps:

            writer.writerow(
                merged[timestamp]
            )

    print(
        f"Features anteriores: "
        f"{len(previous)}"
    )

    print(
        f"Frames nuevos: "
        f"{len(current)}"
    )

    print(
        f"Frames acumulados: "
        f"{len(timestamps)}"
    )

    print(
        "Límite histórico: "
        f"{MAX_FEATURE_FRAMES} "
        "(~30 días)"
    )

    if not timestamps:

        raise SystemExit(
            "ERROR: no quedaron frames "
            "en el dataset acumulado."
        )


if __name__ == "__main__":

    main()
