#!/usr/bin/env python3

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(".")
RADAR_DIR = ROOT / "data" / "radar"
RESULTADO = RADAR_DIR / "resultado_procesamiento.json"
NOWCAST = RADAR_DIR / "radar_nowcast.json"
TIMESTAMP_FILE = RADAR_DIR / "radar_timestamp.txt"
OUT = RADAR_DIR / "radar_features_rma10.csv"

MAX_ROWS = 4320


FIELDS = [
    "frame_time",
    "frame_utc",
    "generated_utc",
    "source",
    "frames_analizados",
    "actividad",
    "area_px",
    "intensidad_nivel_max",
    "distancia_km",
    "fortalecimiento",
    "cambio_area_pct",
    "velocidad_pixeles_min",
    "direccion_grados",
    "movimiento_hacia_bahia",
    "movimiento_hacia_centro_dominio",
    "eta_minutos",
    "nucleos",
    "nucleos_intensidad_3",
    "nucleos_intensidad_4",
    "nucleos_intensidad_5",
    "nucleos_intensidad_6",
]


def leer_json(path: Path) -> dict:
    if not path.exists():
        return {}

    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def numero(valor, default=""):
    if valor is None:
        return default

    try:
        return float(valor)
    except (TypeError, ValueError):
        return default


def entero(valor, default=0):
    try:
        return int(float(valor))
    except (TypeError, ValueError):
        return default


def booleano(valor):
    if isinstance(valor, bool):
        return valor

    if isinstance(valor, str):
        return valor.strip().lower() in {
            "true",
            "1",
            "si",
            "sí",
            "yes",
        }

    return bool(valor)


def obtener_timestamp():
    if not TIMESTAMP_FILE.exists():
        raise RuntimeError(
            "No existe data/radar/radar_timestamp.txt"
        )

    texto = TIMESTAMP_FILE.read_text(
        encoding="utf-8"
    ).strip()

    if not texto:
        raise RuntimeError(
            "radar_timestamp.txt está vacío"
        )

    try:
        dt = datetime.fromisoformat(
            texto.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Timestamp RMA10 inválido: {texto}"
        ) from exc

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return int(dt.timestamp()), dt


def obtener_nowcast():
    data = leer_json(NOWCAST)

    if not data:
        return {}

    nowcast = data.get("nowcast")

    if isinstance(nowcast, dict):
        return nowcast

    return data


def construir_fila():
    frame_time, frame_dt = obtener_timestamp()
    nowcast = obtener_nowcast()

    nucleos = nowcast.get("nucleos", [])

    if not isinstance(nucleos, list):
        nucleos = []

    intensidades = []

    for nucleo in nucleos:
        if not isinstance(nucleo, dict):
            continue

        intensidad = nucleo.get("intensidad")

        try:
            intensidad = int(float(intensidad))
        except (TypeError, ValueError):
            continue

        if 0 <= intensidad <= 6:
            intensidades.append(intensidad)

    row = {
        "frame_time": frame_time,
        "frame_utc": frame_dt.isoformat().replace(
            "+00:00", "Z"
        ),
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat().replace(
            "+00:00", "Z"
        ),
        "source": "SMN RMA10",
        "frames_analizados": entero(
            nowcast.get("frames_analizados")
        ),
        "actividad": booleano(
            nowcast.get("actividad")
        ),
        "area_px": numero(
            nowcast.get("area_px")
        ),
        "intensidad_nivel_max": entero(
            nowcast.get("intensidad_nivel_max")
        ),
        "distancia_km": numero(
            nowcast.get("distancia_km")
        ),
        "fortalecimiento": (
            nowcast.get("fortalecimiento")
            or ""
        ),
        "cambio_area_pct": numero(
            nowcast.get("cambio_area_pct")
        ),
        "velocidad_pixeles_min": numero(
            nowcast.get("velocidad_pixeles_min")
        ),
        "direccion_grados": numero(
            nowcast.get("direccion_grados")
        ),
        "movimiento_hacia_bahia": booleano(
            nowcast.get("movimiento_hacia_bahia")
        ),
        "movimiento_hacia_centro_dominio": booleano(
            nowcast.get(
                "movimiento_hacia_centro_dominio"
            )
        ),
        "eta_minutos": numero(
            nowcast.get("eta_minutos")
        ),
        "nucleos": len(nucleos),
        "nucleos_intensidad_3": sum(
            1 for x in intensidades if x >= 3
        ),
        "nucleos_intensidad_4": sum(
            1 for x in intensidades if x >= 4
        ),
        "nucleos_intensidad_5": sum(
            1 for x in intensidades if x >= 5
        ),
        "nucleos_intensidad_6": sum(
            1 for x in intensidades if x >= 6
        ),
    }

    return row


def cargar_existentes():
    if not OUT.exists():
        return []

    with OUT.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:
        return list(csv.DictReader(f))


def guardar(rows):
    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    rows.sort(
        key=lambda r: int(
            float(r["frame_time"])
        )
    )

    if len(rows) > MAX_ROWS:
        rows = rows[-MAX_ROWS:]

    with OUT.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                field: row.get(field, "")
                for field in FIELDS
            })


def main():
    if not RESULTADO.exists():
        raise RuntimeError(
            "No existe resultado_procesamiento.json"
        )

    row = construir_fila()

    existentes = cargar_existentes()

    frame_time = str(row["frame_time"])

    existentes = [
        r for r in existentes
        if str(r.get("frame_time", "")) != frame_time
    ]

    existentes.append(row)

    guardar(existentes)

    print(
        f"RMA10 dataset actualizado: "
        f"{len(existentes)} frames"
    )

    print(
        f"Frame: {row['frame_utc']}"
    )


if __name__ == "__main__":
    main()
