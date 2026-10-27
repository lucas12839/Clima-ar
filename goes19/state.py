from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone, timedelta
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

RADAR_TIMESTAMP_FILE = (
    BASE_DIR
    / "data"
    / "radar"
    / "radar_timestamp.txt"
)


def load_radar_reference_time() -> datetime:
    if not RADAR_TIMESTAMP_FILE.exists():
        raise RuntimeError(
            "No existe radar_timestamp.txt. "
            "No se puede sincronizar GOES-19 "
            "sin el frame RMA10."
        )

    value = (
        RADAR_TIMESTAMP_FILE
        .read_text(encoding="utf-8")
        .strip()
    )

    if not value:
        raise RuntimeError(
            "radar_timestamp.txt está vacío."
        )

    reference_time = datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )

    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(
            tzinfo=timezone.utc
        )

    return reference_time.astimezone(timezone.utc)


PRODUCTS = {
    "abi_c08": "CMI",
    "abi_c13": "CMI",
}


CLOUDTOP_PRODUCTS = {
    "cloudtop_height": "HT",
    "cloudtop_pressure": "PRES",
}


def build_source_metadata(
    product: str,
    metadata: dict,
) -> dict:
    """
    Construye la metadata temporal completa del producto GOES.

    Los timestamps s/e/c provienen del nombre del producto
    GOES y representan respectivamente:

        start_time_utc    -> inicio de observación
        end_time_utc      -> fin de observación
        creation_time_utc -> creación del producto

    last_modified_utc corresponde a S3 y se conserva
    únicamente como metadata de publicación/modificación.
    """

    return {
        "product": product,
        "key": metadata["key"],
        "url": metadata["url"],
        "start_time_utc": metadata.get(
            "start_time_utc"
        ),
        "end_time_utc": metadata.get(
            "end_time_utc"
        ),
        "creation_time_utc": metadata.get(
            "creation_time_utc"
        ),
        "last_modified_utc": metadata[
            "last_modified"
        ],
        "size_bytes": metadata["size"],
    }


def download_product(
    product: str,
    destination: Path,
    when: datetime | None = None,
) -> dict:
    """Busca y descarga el producto GOES más reciente."""

    obj = latest_product_object(
        product,
        now=when,
        hours_back=0,
    )

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
    reference_time: datetime,
) -> dict:

    file_path = temp_dir / f"{product}.nc"

    metadata = download_product(
        product,
        file_path,
        when=reference_time,
    )

    result = prepare_goes_domain(
        file_path,
        variable=variable,
    )

    data = result["data"]

    statistics = calculate_statistics(data)

    return {
        "status": "ok",
        "source": build_source_metadata(
            product,
            metadata,
        ),
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
    reference_time: datetime,
) -> dict:

    file_path = temp_dir / f"{product}.nc"

    metadata = download_product(
        product,
        file_path,
        when=reference_time,
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

    valid = (
        np.isfinite(values)
        & np.isfinite(dqf)
        & (dqf == 0)
    )

    usable = values[valid]

    if usable.size == 0:
        raise RuntimeError(
            f"{product}: no hay píxeles con DQF=0 "
            "dentro del dominio ClimaAR."
        )

    return {
        "status": "ok",
        "source": build_source_metadata(
            product,
            metadata,
        ),
        "domain": {
            "center_lat": RADAR_LAT,
            "center_lon": RADAR_LON,
            "radius_km": RADAR_RANGE_KM,
            "points": int(len(values)),
        },
        "quality": {
            "usable_points": int(usable.size),
            "good_quality_qf": int(np.sum(dqf == 0)),
            "non_good_quality_qf": int(
                np.sum(
                    np.isfinite(dqf)
                    & (dqf != 0)
                )
            ),
        },
        "statistics": calculate_statistics(
            usable
        ),
    }


def process_glm(
    temp_dir: Path,
    reference_time: datetime,
) -> dict:
    """Procesa GLM dentro del dominio operativo de ClimaAR."""

    file_path = temp_dir / "glm.nc"

    metadata = download_product(
        "glm",
        file_path,
        when=reference_time,
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

            inside_distances = (
                flash_distance[flash_inside]
            )

            inside_lat = flash_lat[
                flash_inside
            ]

            inside_lon = flash_lon[
                flash_inside
            ]

            nearest_position = int(
                np.argmin(
                    inside_distances
                )
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
        "source": build_source_metadata(
            "glm",
            metadata,
        ),
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


def process_dmw(
    temp_dir: Path,
    reference_time: datetime,
    search_hours: int = 12,
) -> dict:
    """
    Busca DMW hacia atrás hasta encontrar un archivo
    con vectores de viento de buena calidad dentro
    del dominio operativo de ClimaAR.
    """

    for offset in range(search_hours + 1):

        when = reference_time - timedelta(
            hours=offset
        )

        metadata = latest_product_object(
            "dmw",
            now=when,
            hours_back=0,
        )

        if metadata is None:
            continue

        file_path = (
            temp_dir
            / f"dmw_{offset}.nc"
        )

        response = requests.get(
            metadata["url"],
            timeout=180,
        )

        response.raise_for_status()

        file_path.write_bytes(
            response.content
        )

        with xr.open_dataset(
            file_path,
            engine="netcdf4",
            mask_and_scale=True,
        ) as ds:

            required = [
                "lat",
                "lon",
                "wind_speed",
                "wind_direction",
                "temperature",
                "pressure",
                "DQF",
            ]

            missing = [
                name
                for name in required
                if name not in ds
            ]

            if missing:
                continue

            lat = ds["lat"].values
            lon = ds["lon"].values
            wind_speed = ds[
                "wind_speed"
            ].values
            wind_direction = ds[
                "wind_direction"
            ].values
            temperature = ds[
                "temperature"
            ].values
            pressure = ds[
                "pressure"
            ].values
            dqf = ds["DQF"].values

            valid = (
                np.isfinite(lat)
                & np.isfinite(lon)
                & np.isfinite(wind_speed)
                & np.isfinite(wind_direction)
                & np.isfinite(temperature)
                & np.isfinite(pressure)
                & np.isfinite(dqf)
                & (dqf == 0)
            )

            if not np.any(valid):
                continue

            lat = lat[valid]
            lon = lon[valid]
            wind_speed = wind_speed[valid]
            wind_direction = wind_direction[valid]
            temperature = temperature[valid]
            pressure = pressure[valid]

            distance = haversine_distance_km(
                lat,
                lon,
            )

            inside = (
                distance <= RADAR_RANGE_KM
            )

            if not np.any(inside):
                continue

            lat_inside = lat[inside]
            lon_inside = lon[inside]
            speed_inside = wind_speed[
                inside
            ]
            direction_inside = (
                wind_direction[inside]
            )
            temperature_inside = (
                temperature[inside]
            )
            pressure_inside = (
                pressure[inside]
            )
            distance_inside = (
                distance[inside]
            )

            nearest_index = int(
                np.argmin(distance_inside)
            )

            return {
                "status": "ok",
                "source": build_source_metadata(
                    "dmw",
                    metadata,
                ),
                "domain": {
                    "center_lat": RADAR_LAT,
                    "center_lon": RADAR_LON,
                    "radius_km": RADAR_RANGE_KM,
                    "measurements": int(
                        len(speed_inside)
                    ),
                },
                "search": {
                    "hours_back": offset,
                },
                "wind_speed": (
                    calculate_statistics(
                        speed_inside
                    )
                ),
                "wind_direction": (
                    calculate_statistics(
                        direction_inside
                    )
                ),
                "temperature": (
                    calculate_statistics(
                        temperature_inside
                    )
                ),
                "pressure": (
                    calculate_statistics(
                        pressure_inside
                    )
                ),
                "nearest_measurement": {
                    "distance_km": float(
                        distance_inside[
                            nearest_index
                        ]
                    ),
                    "latitude": float(
                        lat_inside[
                            nearest_index
                        ]
                    ),
                    "longitude": float(
                        lon_inside[
                            nearest_index
                        ]
                    ),
                    "wind_speed": float(
                        speed_inside[
                            nearest_index
                        ]
                    ),
                    "wind_direction": float(
                        direction_inside[
                            nearest_index
                        ]
                    ),
                    "temperature": float(
                        temperature_inside[
                            nearest_index
                        ]
                    ),
                    "pressure": float(
                        pressure_inside[
                            nearest_index
                        ]
                    ),
                },
            }

    return {
        "status": "unavailable",
        "reason": (
            "No se encontró DMW con "
            "mediciones DQF=0 dentro de "
            f"{RADAR_RANGE_KM} km en las "
            f"últimas {search_hours} horas."
        ),
    }


def build_state(
    reference_time: datetime | None = None,
) -> dict:

    if reference_time is None:
        reference_time = load_radar_reference_time()

    generated_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    state = {
        "version": "1.4",
        "satellite": "GOES-19",
        "generated_at_utc": generated_at,
        "reference_frame_utc": reference_time.isoformat(),
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
        "dmw": None,
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
                        reference_time,
                    )
                )

            except Exception as exc:

                state["products"][product] = {
                    "status": "unavailable",
                    "error": str(exc),
                }

        # Cloud Top
        for product, variable in (
            CLOUDTOP_PRODUCTS.items()
        ):

            try:

                state["cloudtop"][product] = (
                    process_cloudtop(
                        product,
                        variable,
                        temp_dir,
                        reference_time,
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
                temp_dir,
                reference_time,
            )

        except Exception as exc:

            state["glm"] = {
                "status": "unavailable",
                "error": str(exc),
            }

        # DMW
        try:

            state["dmw"] = process_dmw(
                temp_dir,
                reference_time,
            )

        except Exception as exc:

            state["dmw"] = {
                "status": "unavailable",
                "error": str(exc),
            }

        # Estado general
        components = []

        components.extend(
            state["products"].values()
        )

        components.extend(
            state["cloudtop"].values()
        )

        components.append(
            state["glm"]
        )

        components.append(
            state["dmw"]
        )

        available_components = sum(
            1
            for value in components
            if value
            and value.get("status")
            in {
                "ok",
                "sin_actividad",
            }
        )

        total_components = len(
            components
        )

        if available_components == 0:

            state["status"] = (
                "unavailable"
            )

        elif available_components < total_components:

            state["status"] = "partial"

        else:

            state["status"] = "ok"

    return state


def save_state(
    state: dict,
) -> None:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = (
        OUTPUT_FILE.with_suffix(".tmp")
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
        f"Frame RMA10: "
        f"{state['reference_frame_utc']}"
    )

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

        points = data[
            "domain"
        ]["points"]

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

        quality = data[
            "quality"
        ]

        print(
            f"{product}: OK - "
            f"{quality['usable_points']:,} "
            "utilizables - "
            f"{quality['good_quality_qf']:,} "
            "buena calidad"
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

    dmw = state.get("dmw")

    if dmw is None:

        print("DMW: NO DISPONIBLE")

    elif dmw.get("status") != "ok":

        print(
            "DMW: "
            f"{dmw.get('status', 'ERROR')}"
        )

    else:

        measurements = dmw[
            "domain"
        ]["measurements"]

        hours_back = dmw[
            "search"
        ]["hours_back"]

        print(
            "DMW: OK - "
            f"{measurements} mediciones - "
            f"{hours_back} h atrás"
        )

    print(
        "OK: GOES-19 state generado"
    )


if __name__ == "__main__":
    main()
