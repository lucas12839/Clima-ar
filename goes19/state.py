from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

from .goes19_engine import latest_product_object
from .geospatial import prepare_goes_domain


# ============================================================
# ClimaAR - GOES-19 State
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "goes19"
    / "goes19_state.json"
)

PRODUCTS = {
    "abi_c08": "CMI",
    "abi_c13": "CMI",
}


# ============================================================
# Descarga
# ============================================================

def download_product(
    product: str,
    destination: Path,
) -> dict:
    """Busca y descarga el producto GOES más reciente."""

    obj = latest_product_object(product)

    if obj is None:
        raise RuntimeError(
            f"No se encontró producto GOES: {product}"
        )

    response = requests.get(
        obj["url"],
        timeout=180,
    )

    response.raise_for_status()

    destination.write_bytes(
        response.content
    )

    return obj


# ============================================================
# Estadísticas espaciales
# ============================================================

def calculate_statistics(
    values: np.ndarray,
) -> dict:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    valid = np.isfinite(values)

    if not np.any(valid):
        raise RuntimeError(
            "El producto GOES no contiene valores válidos."
        )

    values = values[valid]

    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


# ============================================================
# Procesamiento de un producto
# ============================================================

def process_product(
    product: str,
    variable: str,
    temp_dir: Path,
) -> dict:

    file_path = (
        temp_dir
        / f"{product}.nc"
    )

    metadata = download_product(
        product,
        file_path,
    )

    result = prepare_goes_domain(
        file_path,
        variable=variable,
    )

    data = result["data"]

    statistics = calculate_statistics(
        data
    )

    return {
        "source": {
            "product": product,
            "key": metadata["key"],
            "url": metadata["url"],
            "last_modified_utc": metadata[
                "last_modified"
            ],
            "size_bytes": metadata["size"],
        },
        "domain": {
            "center_lat": -38.7342,
            "center_lon": -62.1654,
            "radius_km": 240.0,
            "points": int(len(data)),
        },
        "statistics": statistics,
    }


# ============================================================
# Estado GOES
# ============================================================

def build_state() -> dict:

    generated_at = datetime.now(
        timezone.utc
    ).isoformat()

    state = {
        "version": "1.0",
        "satellite": "GOES-19",
        "generated_at_utc": generated_at,
        "status": "ok",
        "domain": {
            "type": "RMA10_RADIUS",
            "center_lat": -38.7342,
            "center_lon": -62.1654,
            "radius_km": 240.0,
        },
        "products": {},
    }

    with tempfile.TemporaryDirectory(
        prefix="climaar_goes19_"
    ) as temp:

        temp_dir = Path(temp)

        for product, variable in PRODUCTS.items():

            try:

                state["products"][product] = (
                    process_product(
                        product,
                        variable,
                        temp_dir,
                    )
                )

            except Exception as exc:

                state["products"][product] = {
                    "status": "unavailable",
                    "error": str(exc),
                }

    available = [
        value
        for value in state["products"].values()
        if value.get("status", "ok") == "ok"
    ]

    if not available:
        state["status"] = "unavailable"

    elif len(available) < len(PRODUCTS):
        state["status"] = "partial"

    return state


# ============================================================
# Escritura
# ============================================================

def save_state(
    state: dict,
) -> None:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = (
        OUTPUT_FILE.with_suffix(
            ".tmp"
        )
    )

    temporary_file.write_text(
        json.dumps(
            state,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary_file.replace(
        OUTPUT_FILE
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "ClimaAR GOES-19 State Engine"
    )

    state = build_state()

    save_state(state)

    print(
        f"Estado: {state['status']}"
    )

    print(
        f"Salida: {OUTPUT_FILE}"
    )

    for product, data in state[
        "products"
    ].items():

        if data.get("status") == "unavailable":
            print(
                f"{product}: NO DISPONIBLE"
            )
            continue

        points = data[
            "domain"
        ]["points"]

        print(
            f"{product}: OK - "
            f"{points:,} puntos"
        )

    print(
        "OK: GOES-19 state generado"
    )


if __name__ == "__main__":
    main()
