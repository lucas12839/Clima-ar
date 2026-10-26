from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import xarray as xr

from .goes19_engine import latest_product_object
from .geospatial import (
    RADAR_LAT,
    RADAR_LON,
    RADAR_RANGE_KM,
    prepare_goes_domain,
)


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


CLOUDTOP_PRODUCTS = {
    "cloudtop_height": "HT",
    "cloudtop_pressure": "PRES",
}


CLOUDTOP_USABLE_DQF = {0, 1, 2}
CLOUDTOP_GOOD_DQF = {0}


def download_product(product: str, destination: Path) -> dict:
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

    destination.write_bytes(response.content)

    return obj


def calculate_statistics(values: np.ndarray) -> dict:
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


def haversine_distance_km(
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    """Calcula distancia desde el centro operativo de ClimaAR."""
    earth_radius_km = 6371.0

    lat1 = np.radians(RADAR_LAT)
    lon1 = np.radians(RADAR_LON)

    lat2 = np.radians(latitude)
    lon2 = np.radians(longitude)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return (
        2.0
        * earth_radius_km
        * np.arcsin(np.sqrt(a))
    )


def process_product(
    product: str,
    variable: str,
    temp_dir: Path,
) -> dict:
    """Procesa un producto ABI dentro del dominio ClimaAR."""
    file_path = temp_dir / f"{product}.nc"

    metadata = download_product(
        product,
        file_path,
    )

    result = prepare_goes_domain(
        file_path,
        variable=variable,
    )

    data = result["data"]

    statistics = calculate_statistics(data)

    return {
        "status": "ok",
        "source": {
            "product": product,
            "key": metadata["key"],
            "url": metadata["url"],
            "last_modified_utc": metadata["last_modified"],
            "size_bytes": metadata["size"],
        },
        "domain": {
            "center_lat": RADAR_LAT,
            "center_lon": RADAR_LON,
            "radius_km": RADAR_RANGE_KM,
            "points": int(len(data)),
        },
        "statistics": statistics,
    }


def process_cloudtop(
    product: str,
    variable: str,
    temp_dir: Path,
) -> dict:
    """Procesa Cloud Top Height/Pressure con control DQF."""
    file_path = temp_dir / f"{product}.nc"

    metadata = download_product(
        product,
        file_path,
    )

    value_result = prepare_goes_domain(
        file_path,
        variable=variable,
    )

    dqf_result = prepare_goes_domain(
        file_path,
        variable="DQF",
    )

    values = np.asarray(
        value_result["data"],
        dtype=np.float64,
    )

    dqf = np.asarray(
        dqf_result["data"],
        dtype=np.float64,
    )

    if values.size != dqf.size:
        raise RuntimeError(
            "Cloud Top: los datos y DQF no tienen "
            "la misma cantidad de puntos."
        )

    finite = (
        np.isfinite(values)
        & np.isfinite(dqf)
    )

    values = values[finite]
    dqf = dqf[finite].astype(np.int16)

    quality_counts = {
        "good": int(np.sum(dqf == 0)),
        "marginal": int(np.sum(dqf == 1)),
        "retrieval_attempted": int(np.sum(dqf == 2)),
        "bad": int(np.sum(dqf == 3)),
    }

    usable_mask = np.isin(
        dqf,
        list(CLOUDTOP_USABLE_DQF),
    )

    good_mask = np.isin(
        dqf,
        list(CLOUDTOP_GOOD_DQF),
    )

    usable_values = values[usable_mask]
    good_values = values[good_mask]

    if usable_values.size == 0:
        raise RuntimeError(
            f"{product}: no hay datos Cloud Top utilizables."
        )

    statistics = calculate_statistics(
        usable_values
    )

    good_statistics = None

    if good_values.size > 0:
        good_statistics = calculate_statistics(
            good_values
        )

    return {
        "status": "ok",
        "source": {
            "product": product,
            "key": metadata["key"],
            "url": metadata["url"],
            "last_modified_utc": metadata["last_modified"],
            "size_bytes": metadata["size"],
        },
        "domain": {
            "center_lat": RADAR_LAT,
            "center_lon": RADAR_LON,
            "radius_km": RADAR_RANGE_KM,
            "points": int(values.size),
        },
        "quality": {
            "dqf_usable": [0, 1, 2],
            "dqf_discarded": [3],
            "counts": quality_counts,
            "usable_points": int(
                usable_values.size
            ),
            "good_points": int(
                good_values.size
            ),
        },
        "statistics": statistics,
        "good_quality_statistics": good_statistics,
    }


def process_glm(temp_dir: Path) -> dict:
    """Procesa GLM dentro del dominio operativo de ClimaAR."""
    file_path = temp_dir / "glm.nc"

    metadata = download_product(
        "glm",
        file_path,
    )

    with xr.open_dataset(
        file_path,
        engine="netcdf4",
        mask_and_scale=True,
    ) as ds:

        required = [
            "event_lat",
            "event_lon",
            "group_lat",
            "group_lon",
            "flash_lat",
            "flash_lon",
            "flash_energy",
            "flash_area",
            "flash_quality_flag",
        ]

        missing = [
            name
            for name in required
            if name not in ds
        ]

        if missing:
            raise RuntimeError(
                "Faltan variables GLM: "
                + ", ".join(missing)
            )

        event_lat = ds["event_lat"].values
        event_lon = ds["event_lon"].values

        event_valid = (
            np.isfinite(event_lat)
            & np.isfinite(event_lon)
        )

        event_distance = haversine_distance_km(
            event_lat[event_valid],
            event_lon[event_valid],
        )

        event_inside = (
            event_distance <= RADAR_RANGE_KM
        )

        group_lat = ds["group_lat"].values
        group_lon = ds["group_lon"].values

        group_valid = (
            np.isfinite(group_lat)
            & np.isfinite(group_lon)
        )

        group_distance = haversine_distance_km(
            group_lat[group_valid],
            group_lon[group_valid],
        )

        group_inside = (
            group_distance <= RADAR_RANGE_KM
        )

        flash_lat = ds["flash_lat"].values
        flash_lon = ds["flash_lon"].values
        flash_energy = ds["flash_energy"].values
        flash_area = ds["flash_area"].values
        flash_quality = ds[
            "flash_quality_flag"
        ].values

        flash_valid = (
            np.isfinite(flash_lat)
            & np.isfinite(flash_lon)
            & np.isfinite(flash_energy)
            & np.isfinite(flash_area)
        )

        flash_lat = flash_lat[flash_valid]
        flash_lon = flash_lon[flash_valid]
        flash_energy = flash_energy[flash_valid]
        flash_area = flash_area[flash_valid]
        flash_quality = flash_quality[flash_valid]

        flash_distance = haversine_distance_km(
            flash_lat,
            flash_lon,
        )

        flash_inside = (
            flash_distance <= RADAR_RANGE_KM
        )

        domain_energy = flash_energy[
            flash_inside
        ]

        domain_area = flash_area[
            flash_inside
        ]

        domain_quality = flash_quality[
            flash_inside
        ]

        quality_0 = int(
            np.sum(domain_quality == 0)
        )

        quality_3 = int(
            np.sum(domain_quality == 3)
        )

        flashes_inside = int(
            np.sum(flash_inside)
        )

        events_inside = int(
            np.sum(event_inside)
        )

        groups_inside = int(
            np.sum(group_inside)
        )

        if flashes_inside == 0:
            activity_status = "sin_actividad"
        else:
            activity_status = "ok"

        if flashes_inside > 0:
            energy_stats = calculate_statistics(
                domain_energy
            )

            area_stats = calculate_statistics(
                domain_area
            )

            inside_distances = flash_distance[
                flash_inside
            ]

            inside_lat = flash_lat[
                flash_inside
            ]

            inside_lon = flash_lon[
                flash_inside
            ]

            nearest_position = int(
                np.argmin(inside_distances)
            )

            nearest_distance = float(
                inside_distances[
                    nearest_position
                ]
            )

            nearest_lat = float(
                inside_lat[
                    nearest_position
                ]
            )

            nearest_lon = float(
                inside_lon[
                    nearest_position
                ]
            )

        else:
            energy_stats = None
            area_stats = None
            nearest_distance = None
            nearest_lat = None
            nearest_lon = None

    return {
        "status": activity_status,
        "source": {
            "product": "glm",
            "key": metadata["key"],
            "url": metadata["url"],
            "last_modified_utc": metadata["last_modified"],
            "size_bytes": metadata["size"],
        },
        "domain": {
            "center_lat": RADAR_LAT,
            "center_lon": RADAR_LON,
            "radius_km": RADAR_RANGE_KM,
        },
        "counts": {
            "events": events_inside,
            "groups": groups_inside,
            "flashes": flashes_inside,
            "flashes_quality_0": quality_0,
            "flashes_quality_3": quality_3,
        },
        "flash_energy": energy_stats,
        "flash_area": area_stats,
        "nearest_flash": {
            "distance_km": nearest_distance,
            "latitude": nearest_lat,
            "longitude": nearest_lon,
        },
    }


def build_state() -> dict:
    generated_at = (
        datetime.now(timezone.utc).isoformat()
    )

    state = {
        "version": "1.2",
        "satellite": "GOES-19",
        "generated_at_utc": generated_at,
        "status": "ok",
        "domain": {
            "type": "RMA10_RADIUS",
            "center_lat": RADAR_LAT,
            "center_lon": RADAR_LON,
            "radius_km": RADAR_RANGE_KM,
        },
        "products": {},
        "cloudtop": {},
        "glm": None,
    }

    with tempfile.TemporaryDirectory(
        prefix="climaar_goes19_"
    ) as temp:

        temp_dir = Path(temp)

        # ABI C08 / C13
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

        # Cloud Top Height / Pressure
        for product, variable in (
            CLOUDTOP_PRODUCTS.items()
        ):
            try:
                state["cloudtop"][product] = (
                    process_cloudtop(
                        product,
                        variable,
                        temp_dir,
                    )
                )

            except Exception as exc:
                state["cloudtop"][product] = {
                    "status": "unavailable",
                    "error": str(exc),
                }

        # GLM
        try:
            state["glm"] = process_glm(
                temp_dir
            )

        except Exception as exc:
            state["glm"] = {
                "status": "unavailable",
                "error": str(exc),
            }

        # Estado general
        abi_available = [
            value
            for value in state["products"].values()
            if value.get("status") == "ok"
        ]

        cloudtop_available = [
            value
            for value in state["cloudtop"].values()
            if value.get("status") == "ok"
        ]

        glm_available = (
            state["glm"] is not None
            and state["glm"].get("status")
            in {
                "ok",
                "sin_actividad",
            }
        )

        total_components = (
            len(PRODUCTS)
            + len(CLOUDTOP_PRODUCTS)
            + 1
        )

        available_components = (
            len(abi_available)
            + len(cloudtop_available)
            + int(glm_available)
        )

        if available_components == 0:
            state["status"] = "unavailable"

        elif available_components < total_components:
            state["status"] = "partial"

        else:
            state["status"] = "ok"

    return state


def save_state(state: dict) -> None:
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = OUTPUT_FILE.with_suffix(
        ".tmp"
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


def main() -> None:
    print(
        "ClimaAR GOES-19 State Engine"
    )

    state = build_state()

    save_state(state)

    print(
        f"Estado: {state['status']}"
    )

    for product, data in (
        state["products"].items()
    ):
        if data.get("status") != "ok":
            print(
                f"{product}: "
                f"{data.get('status', 'ERROR')}"
            )
            continue

        points = data["domain"]["points"]

        print(
            f"{product}: OK - "
            f"{points:,} puntos"
        )

    for product, data in (
        state["cloudtop"].items()
    ):
        if data.get("status") != "ok":
            print(
                f"{product}: "
                f"{data.get('status', 'ERROR')}"
            )
            continue

        quality = data["quality"]

        print(
            f"{product}: OK - "
            f"{quality['usable_points']:,} utilizables - "
            f"{quality['good_points']:,} buena calidad"
        )

    glm = state.get("glm")

    if glm is None:
        print("GLM: NO DISPONIBLE")

    elif glm.get("status") == "unavailable":
        print("GLM: NO DISPONIBLE")

    else:
        counts = glm["counts"]

        print(
            "GLM: "
            f"{glm['status']} - "
            f"{counts['events']} eventos, "
            f"{counts['groups']} grupos, "
            f"{counts['flashes']} flashes"
        )

    print(
        "OK: GOES-19 state generado"
    )


if __name__ == "__main__":
    main()
