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
OUTPUT_FILE = BASE_DIR / "data" / "goes19" / "goes19_state.json"
RADAR_TIMESTAMP_FILE = BASE_DIR / "data" / "radar" / "radar_timestamp.txt"

PRODUCTS = {
    "abi_c08": "CMI",
    "abi_c13": "CMI",
}

CLOUDTOP_PRODUCTS = {
    "cloudtop_height": "HT",
    "cloudtop_pressure": "PRES",
}

DMW_PRODUCTS = {
    "dmw_c02": "C02",
    "dmw_c07": "C07",
    "dmw_c08": "C08",
    "dmw_c09": "C09",
    "dmw_c10": "C10",
    "dmw_c14": "C14",
}

DMWV_PRODUCTS = {
    "dmwvf_c08": "C08",
}

GOOD_DMW_DQF = {-3, -2, -1, 0}
DMW_QUALITY_WEIGHT = {-3: 0.70, -2: 0.85, -1: 0.90, 0: 1.00}


def load_radar_reference_time() -> datetime:
    if not RADAR_TIMESTAMP_FILE.exists():
        raise RuntimeError("No existe radar_timestamp.txt.")
    value = RADAR_TIMESTAMP_FILE.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError("radar_timestamp.txt está vacío.")
    reference = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    return reference.astimezone(timezone.utc)


def build_source_metadata(product: str, metadata: dict) -> dict:
    return {
        "product": product,
        "key": metadata["key"],
        "url": metadata["url"],
        "start_time_utc": metadata.get("start_time_utc"),
        "end_time_utc": metadata.get("end_time_utc"),
        "creation_time_utc": metadata.get("creation_time_utc"),
        "last_modified_utc": metadata.get("last_modified"),
        "size_bytes": metadata.get("size"),
    }


def age_minutes(metadata: dict, reference_time: datetime) -> float | None:
    value = metadata.get("end_time_utc")
    if not value:
        return None
    end = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return max(0.0, (reference_time - end.astimezone(timezone.utc)).total_seconds() / 60.0)


def download_product(
    product: str,
    destination: Path,
    when: datetime,
    hours_back: int = 6,
) -> dict:
    metadata = latest_product_object(product, now=when, hours_back=hours_back)
    if metadata is None:
        raise RuntimeError(f"No se encontró producto GOES: {product}")

    response = requests.get(metadata["url"], timeout=180)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return metadata


def calculate_statistics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError("No hay valores válidos.")
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
    earth_radius_km = 6371.0088
    lat1 = np.radians(RADAR_LAT)
    lon1 = np.radians(RADAR_LON)
    lat2 = np.radians(latitude)
    lon2 = np.radians(longitude)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def process_product(
    product: str,
    variable: str,
    temp_dir: Path,
    reference_time: datetime,
) -> dict:
    path = temp_dir / f"{product}.nc"
    metadata = download_product(product, path, reference_time, hours_back=6)
    result = prepare_goes_domain(path, variable=variable)
    return {
        "status": "ok",
        "source": {
            **build_source_metadata(product, metadata),
            "age_minutes": age_minutes(metadata, reference_time),
        },
        "domain": {
            "center_lat": RADAR_LAT,
            "center_lon": RADAR_LON,
            "radius_km": RADAR_RANGE_KM,
            "points": int(len(result["data"])),
        },
        "statistics": calculate_statistics(result["data"]),
    }


def process_cloudtop(
    product: str,
    variable: str,
    temp_dir: Path,
    reference_time: datetime,
) -> dict:
    path = temp_dir / f"{product}.nc"
    metadata = download_product(product, path, reference_time, hours_back=6)
    values = np.asarray(
        prepare_goes_domain(path, variable=variable)["data"],
        dtype=np.float64,
    )
    dqf = np.asarray(
        prepare_goes_domain(path, variable="DQF")["data"],
        dtype=np.float64,
    )

    valid = np.isfinite(values) & np.isfinite(dqf) & (dqf == 0)
    usable = values[valid]
    if usable.size == 0:
        raise RuntimeError(f"{product}: no hay píxeles DQF=0.")

    return {
        "status": "ok",
        "source": {
            **build_source_metadata(product, metadata),
            "age_minutes": age_minutes(metadata, reference_time),
        },
        "domain": {
            "center_lat": RADAR_LAT,
            "center_lon": RADAR_LON,
            "radius_km": RADAR_RANGE_KM,
            "points": int(values.size),
        },
        "quality": {
            "usable_points": int(usable.size),
            "good_quality_qf": int(np.sum(dqf == 0)),
            "non_good_quality_qf": int(np.sum(np.isfinite(dqf) & (dqf != 0))),
        },
        "statistics": calculate_statistics(usable),
    }


def process_glm(temp_dir: Path, reference_time: datetime) -> dict:
    path = temp_dir / "glm.nc"
    metadata = download_product("glm", path, reference_time, hours_back=2)

    with xr.open_dataset(path, engine="netcdf4", mask_and_scale=True) as ds:
        required = [
            "event_lat", "event_lon", "group_lat", "group_lon",
            "flash_lat", "flash_lon", "flash_energy", "flash_area",
            "flash_quality_flag",
        ]
        missing = [name for name in required if name not in ds]
        if missing:
            raise RuntimeError("Faltan variables GLM: " + ", ".join(missing))

        event_lat = ds["event_lat"].values
        event_lon = ds["event_lon"].values
        group_lat = ds["group_lat"].values
        group_lon = ds["group_lon"].values
        flash_lat = ds["flash_lat"].values
        flash_lon = ds["flash_lon"].values
        flash_energy = ds["flash_energy"].values
        flash_area = ds["flash_area"].values
        flash_quality = ds["flash_quality_flag"].values

        event_valid = np.isfinite(event_lat) & np.isfinite(event_lon)
        group_valid = np.isfinite(group_lat) & np.isfinite(group_lon)
        flash_valid = (
            np.isfinite(flash_lat) & np.isfinite(flash_lon)
            & np.isfinite(flash_energy) & np.isfinite(flash_area)
        )

        event_distance = haversine_distance_km(
            event_lat[event_valid], event_lon[event_valid]
        )
        group_distance = haversine_distance_km(
            group_lat[group_valid], group_lon[group_valid]
        )

        flash_lat = flash_lat[flash_valid]
        flash_lon = flash_lon[flash_valid]
        flash_energy = flash_energy[flash_valid]
        flash_area = flash_area[flash_valid]
        flash_quality = flash_quality[flash_valid]

        flash_distance = haversine_distance_km(flash_lat, flash_lon)
        flash_inside = flash_distance <= RADAR_RANGE_KM

        domain_energy = flash_energy[flash_inside]
        domain_area = flash_area[flash_inside]
        domain_quality = flash_quality[flash_inside]

        flashes = int(domain_energy.size)
        events = int(np.sum(event_distance <= RADAR_RANGE_KM))
        groups = int(np.sum(group_distance <= RADAR_RANGE_KM))

        if flashes:
            nearest = int(np.argmin(flash_distance[flash_inside]))
            inside_distance = flash_distance[flash_inside]
            inside_lat = flash_lat[flash_inside]
            inside_lon = flash_lon[flash_inside]
            nearest_flash = {
                "distance_km": float(inside_distance[nearest]),
                "latitude": float(inside_lat[nearest]),
                "longitude": float(inside_lon[nearest]),
            }
            energy_stats = calculate_statistics(domain_energy)
            area_stats = calculate_statistics(domain_area)
        else:
            nearest_flash = {
                "distance_km": None,
                "latitude": None,
                "longitude": None,
            }
            energy_stats = None
            area_stats = None

    return {
        "status": "ok" if flashes else "sin_actividad",
        "source": {
            **build_source_metadata("glm", metadata),
            "age_minutes": age_minutes(metadata, reference_time),
        },
        "domain": {
            "center_lat": RADAR_LAT,
            "center_lon": RADAR_LON,
            "radius_km": RADAR_RANGE_KM,
        },
        "counts": {
            "events": events,
            "groups": groups,
            "flashes": flashes,
            "flashes_quality_0": int(np.sum(domain_quality == 0)),
            "flashes_quality_3": int(np.sum(domain_quality == 3)),
        },
        "flash_energy": energy_stats,
        "flash_area": area_stats,
        "nearest_flash": nearest_flash,
    }


def _open_wind_arrays(ds: xr.Dataset) -> dict[str, np.ndarray]:
    aliases = {
        "lat": ("lat",),
        "lon": ("lon",),
        "speed": ("wind_speed",),
        "direction": ("wind_direction",),
        "pressure": ("pressure",),
        "dqf": ("DQF",),
    }
    result = {}
    for target, names in aliases.items():
        found = next((name for name in names if name in ds), None)
        if found is None:
            raise RuntimeError(f"Falta variable DMW: {target}")
        result[target] = np.asarray(ds[found].values, dtype=np.float64)
    return result


def _dmw_quality_weight(dqf: np.ndarray) -> np.ndarray:
    weight = np.zeros(dqf.shape, dtype=np.float64)
    for code, value in DMW_QUALITY_WEIGHT.items():
        weight[dqf == code] = value
    return weight


def _vector_summary(speed: np.ndarray, direction_from: np.ndarray, weight: np.ndarray) -> dict:
    # Meteorological direction = FROM. Convert to vector components toward.
    radians = np.radians(direction_from)
    u = -speed * np.sin(radians)
    v = -speed * np.cos(radians)

    total_weight = float(np.sum(weight))
    if total_weight <= 0:
        raise RuntimeError("No hay pesos DMW válidos.")

    mean_u = float(np.sum(u * weight) / total_weight)
    mean_v = float(np.sum(v * weight) / total_weight)
    mean_speed = float(np.hypot(mean_u, mean_v))
    mean_direction = float((np.degrees(np.arctan2(-mean_u, -mean_v)) + 360.0) % 360.0)

    resultant = mean_speed / max(float(np.sum(speed * weight) / total_weight), 1e-9)

    return {
        "mean_speed": mean_speed,
        "mean_direction_from": mean_direction,
        "mean_u": mean_u,
        "mean_v": mean_v,
        "direction_resultant": resultant,
        "speed": calculate_statistics(speed),
    }


def process_dmw(
    temp_dir: Path,
    reference_time: datetime,
    search_hours: int = 3,
) -> dict:
    products = {**DMW_PRODUCTS, **DMWV_PRODUCTS}
    records = []
    downloaded_keys: set[str] = set()

    for product in products:
        metadata = latest_product_object(
            product,
            now=reference_time,
            hours_back=search_hours,
        )
        if metadata is None:
            continue
        if metadata["key"] in downloaded_keys:
            continue
        downloaded_keys.add(metadata["key"])

        path = temp_dir / f"{product}.nc"
        response = requests.get(metadata["url"], timeout=180)
        response.raise_for_status()
        path.write_bytes(response.content)

        with xr.open_dataset(path, engine="netcdf4", mask_and_scale=True) as ds:
            arrays = _open_wind_arrays(ds)
            lat = arrays["lat"]
            lon = arrays["lon"]
            speed = arrays["speed"]
            direction = arrays["direction"]
            pressure = arrays["pressure"]
            dqf = arrays["dqf"]

            valid = (
                np.isfinite(lat) & np.isfinite(lon)
                & np.isfinite(speed) & np.isfinite(direction)
                & np.isfinite(pressure) & np.isfinite(dqf)
                & np.isin(dqf.astype(np.int64), list(GOOD_DMW_DQF))
                & (speed >= 0)
            )

            lat = lat[valid]
            lon = lon[valid]
            speed = speed[valid]
            direction = direction[valid] % 360.0
            pressure = pressure[valid]
            dqf = dqf[valid]

            if speed.size == 0:
                continue

            distance = haversine_distance_km(lat, lon)
            inside = distance <= RADAR_RANGE_KM
            if not np.any(inside):
                continue

            lat = lat[inside]
            lon = lon[inside]
            speed = speed[inside]
            direction = direction[inside]
            pressure = pressure[inside]
            dqf = dqf[inside]
            distance = distance[inside]
            weight = _dmw_quality_weight(dqf)

            summary = _vector_summary(speed, direction, weight)
            records.append({
                "product": product,
                "channel": products[product],
                "source": {
                    **build_source_metadata(product, metadata),
                    "age_minutes": age_minutes(metadata, reference_time),
                },
                "vectors": int(speed.size),
                "quality": {
                    "dqf_-3": int(np.sum(dqf == -3)),
                    "dqf_-2": int(np.sum(dqf == -2)),
                    "dqf_-1": int(np.sum(dqf == -1)),
                    "dqf_0": int(np.sum(dqf == 0)),
                },
                "summary": summary,
                "pressure_mb": calculate_statistics(pressure),
                "nearest_vector": {
                    "distance_km": float(distance[np.argmin(distance)]),
                    "latitude": float(lat[np.argmin(distance)]),
                    "longitude": float(lon[np.argmin(distance)]),
                    "speed": float(speed[np.argmin(distance)]),
                    "direction_from": float(direction[np.argmin(distance)]),
                    "pressure_mb": float(pressure[np.argmin(distance)]),
                    "dqf": int(dqf[np.argmin(distance)]),
                },
            })

    if not records:
        return {
            "status": "unavailable",
            "reason": f"No se encontraron vectores DMW/DMWV válidos dentro de {RADAR_RANGE_KM} km.",
            "search_hours": search_hours,
            "products": [],
        }

    return {
        "status": "ok",
        "search_hours": search_hours,
        "products": records,
        "total_vectors": int(sum(x["vectors"] for x in records)),
    }


def build_state(reference_time: datetime | None = None) -> dict:
    if reference_time is None:
        reference_time = load_radar_reference_time()

    state = {
        "version": "1.5",
        "satellite": "GOES-19",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
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

    with tempfile.TemporaryDirectory(prefix="climaar_goes19_") as temp:
        temp_dir = Path(temp)

        for product, variable in PRODUCTS.items():
            try:
                state["products"][product] = process_product(
                    product, variable, temp_dir, reference_time
                )
            except Exception as exc:
                state["products"][product] = {
                    "status": "unavailable",
                    "error": str(exc),
                }

        for product, variable in CLOUDTOP_PRODUCTS.items():
            try:
                state["cloudtop"][product] = process_cloudtop(
                    product, variable, temp_dir, reference_time
                )
            except Exception as exc:
                state["cloudtop"][product] = {
                    "status": "unavailable",
                    "error": str(exc),
                }

        try:
            state["glm"] = process_glm(temp_dir, reference_time)
        except Exception as exc:
            state["glm"] = {"status": "unavailable", "error": str(exc)}

        try:
            state["dmw"] = process_dmw(temp_dir, reference_time)
        except Exception as exc:
            state["dmw"] = {"status": "unavailable", "error": str(exc)}

    components = list(state["products"].values()) + list(state["cloudtop"].values())
    components += [state["glm"], state["dmw"]]
    if any(isinstance(item, dict) and item.get("status") == "ok" for item in components):
        state["status"] = "ok"
    else:
        state["status"] = "degraded"

    return state


def main() -> None:
    reference_time = load_radar_reference_time()
    state = build_state(reference_time)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("ClimaAR GOES-19 State Engine")
    print(f"Frame RMA10: {reference_time.isoformat()}")
    print(f"Estado: {state['status']}")

    for name, value in state["products"].items():
        print(f"{name}: {value.get('status')}")

    for name, value in state["cloudtop"].items():
        print(f"{name}: {value.get('status')}")

    print(f"GLM: {state['glm'].get('status') if state['glm'] else 'unavailable'}")

    dmw = state["dmw"] or {}
    print(
        f"DMW: {dmw.get('status')} - "
        f"{dmw.get('total_vectors', 0)} vectores - "
        f"{len(dmw.get('products', []))} productos"
    )
    print(f"OK: GOES-19 state generado -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
