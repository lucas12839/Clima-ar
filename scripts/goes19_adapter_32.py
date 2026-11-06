from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr

from goes19.geospatial import (
    RADAR_LAT,
    RADAR_LON,
    RADAR_RANGE_KM,
    haversine_km,
    prepare_goes_domain,
)


GRID_SIZE = 64

# Orden fijo del tensor:
# 0 C08
# 1 C13
# 2 Cloud Top Height
# 3 Cloud Top Pressure
# 4 GLM
# 5 DMW speed
# 6 DMW direction sin
# 7 DMW direction cos
# 8 DMW u
# 9 DMW v
CHANNEL_NAMES = (
    "abi_c08",
    "abi_c13",
    "cloudtop_height",
    "cloudtop_pressure",
    "glm",
    "dmw_speed",
    "dmw_direction_sin",
    "dmw_direction_cos",
    "dmw_u",
    "dmw_v",
)


def _normalize(values: np.ndarray, low: float, high: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    result = (values - low) / (high - low)
    return np.clip(result, 0.0, 1.0)


def _bilinear_grid(
    latitude: np.ndarray,
    longitude: np.ndarray,
    values: np.ndarray,
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    """Rasteriza puntos GOES sobre una grilla regular local."""
    lat = np.asarray(latitude, dtype=np.float64).ravel()
    lon = np.asarray(longitude, dtype=np.float64).ravel()
    val = np.asarray(values, dtype=np.float64).ravel()

    valid = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & np.isfinite(val)
    )

    lat = lat[valid]
    lon = lon[valid]
    val = val[valid]

    if val.size == 0:
        return np.zeros((grid_size, grid_size), dtype=np.float32)

    lat_edges = np.linspace(
        RADAR_LAT - 2.25,
        RADAR_LAT + 2.25,
        grid_size + 1,
    )
    lon_edges = np.linspace(
        RADAR_LON - 3.0,
        RADAR_LON + 3.0,
        grid_size + 1,
    )

    iy = np.searchsorted(lat_edges, lat, side="right") - 1
    ix = np.searchsorted(lon_edges, lon, side="right") - 1

    inside = (
        (iy >= 0)
        & (iy < grid_size)
        & (ix >= 0)
        & (ix < grid_size)
    )

    iy = iy[inside]
    ix = ix[inside]
    val = val[inside]

    if val.size == 0:
        return np.zeros((grid_size, grid_size), dtype=np.float32)

    result = np.zeros((grid_size, grid_size), dtype=np.float64)
    count = np.zeros((grid_size, grid_size), dtype=np.float64)

    np.add.at(result, (iy, ix), val)
    np.add.at(count, (iy, ix), 1.0)

    valid_cells = count > 0
    result[valid_cells] /= count[valid_cells]

    return result.astype(np.float32)


def _read_goes_product(
    path: Path,
    variable: str,
) -> dict[str, np.ndarray]:
    result = prepare_goes_domain(path, variable=variable)

    return {
        "data": np.asarray(result["data"], dtype=np.float32),
        "latitude": np.asarray(result["latitude"], dtype=np.float32),
        "longitude": np.asarray(result["longitude"], dtype=np.float32),
    }


def _cloudtop_quality(
    path: Path,
    variable: str,
) -> dict[str, np.ndarray]:
    with xr.open_dataset(
        path,
        engine="netcdf4",
        mask_and_scale=True,
    ) as ds:
        values = np.asarray(ds[variable].values, dtype=np.float32)
        dqf = np.asarray(ds["DQF"].values, dtype=np.float32)

        while values.ndim > 2:
            values = values[0]

        while dqf.ndim > 2:
            dqf = dqf[0]

        x = np.asarray(ds["x"].values)
        y = np.asarray(ds["y"].values)

    xx, yy = np.meshgrid(x, y)

    # Reutilizamos la misma proyección oficial del módulo GOES.
    from goes19.geospatial import (
        get_projection,
        fixed_grid_to_latlon,
        haversine_km,
    )

    with xr.open_dataset(
        path,
        engine="netcdf4",
    ) as ds:
        lambda_0, r_eq, r_pol, h = get_projection(ds)

    lat, lon = fixed_grid_to_latlon(
        xx,
        yy,
        lambda_0=lambda_0,
        r_eq=r_eq,
        r_pol=r_pol,
        h=h,
    )

    distance = haversine_km(
        lat,
        lon,
        RADAR_LAT,
        RADAR_LON,
    )

    mask = (
        (distance <= RADAR_RANGE_KM)
        & np.isfinite(values)
        & np.isfinite(dqf)
        & (dqf == 0)
    )

    return {
        "data": values[mask],
        "latitude": lat[mask],
        "longitude": lon[mask],
    }


def build_goes_tensor(
    c08_path: Path,
    c13_path: Path,
    cth_path: Path,
    ctp_path: Path,
    glm: dict[str, Any] | None = None,
    dmw: dict[str, Any] | None = None,
    grid_size: int = GRID_SIZE,
) -> dict[str, Any]:

    c08 = _read_goes_product(c08_path, "CMI")
    c13 = _read_goes_product(c13_path, "CMI")

    cth = _cloudtop_quality(
        cth_path,
        "HT",
    )
    ctp = _cloudtop_quality(
        ctp_path,
        "PRES",
    )

    fields = []

    fields.append(
        _bilinear_grid(
            c08["latitude"],
            c08["longitude"],
            _normalize(c08["data"], 180.0, 330.0),
            grid_size,
        )
    )

    fields.append(
        _bilinear_grid(
            c13["latitude"],
            c13["longitude"],
            _normalize(c13["data"], 180.0, 330.0),
            grid_size,
        )
    )

    fields.append(
        _bilinear_grid(
            cth["latitude"],
            cth["longitude"],
            _normalize(cth["data"], 0.0, 18000.0),
            grid_size,
        )
    )

    fields.append(
        _bilinear_grid(
            ctp["latitude"],
            ctp["longitude"],
            _normalize(ctp["data"], 100.0, 1050.0),
            grid_size,
        )
    )

    # GLM
    if glm and glm.get("latitude") is not None:
        glm_field = _bilinear_grid(
            glm["latitude"],
            glm["longitude"],
            glm["value"],
            grid_size,
        )
    else:
        glm_field = np.zeros(
            (grid_size, grid_size),
            dtype=np.float32,
        )

    fields.append(glm_field)

    # DMW
    if dmw and dmw.get("latitude") is not None:
        lat = np.asarray(dmw["latitude"])
        lon = np.asarray(dmw["longitude"])
        speed = np.asarray(dmw["speed"])
        direction = np.radians(np.asarray(dmw["direction"]))

        u = -speed * np.sin(direction)
        v = -speed * np.cos(direction)

        fields.extend(
            [
                _bilinear_grid(
                    lat, lon,
                    _normalize(speed, 0.0, 100.0),
                    grid_size,
                ),
                _bilinear_grid(
                    lat, lon,
                    np.sin(direction),
                    grid_size,
                ),
                _bilinear_grid(
                    lat, lon,
                    np.cos(direction),
                    grid_size,
                ),
                _bilinear_grid(
                    lat, lon,
                    _normalize(u, -100.0, 100.0),
                    grid_size,
                ),
                _bilinear_grid(
                    lat, lon,
                    _normalize(v, -100.0, 100.0),
                    grid_size,
                ),
            ]
        )
    else:
        fields.extend(
            [
                np.zeros((grid_size, grid_size), dtype=np.float32)
                for _ in range(5)
            ]
        )

    tensor = np.stack(fields, axis=0).astype(np.float32)

    return {
        "tensor": torch.from_numpy(tensor).unsqueeze(0),
        "channels": CHANNEL_NAMES,
        "shape": tuple(tensor.shape),
        "center": (RADAR_LAT, RADAR_LON),
        "radius_km": RADAR_RANGE_KM,
    }


def main() -> None:
    print("GOES-19 ADAPTER 3.2")
    print(f"grid={GRID_SIZE}x{GRID_SIZE}")
    print(f"channels={len(CHANNEL_NAMES)}")
    print(f"domain={RADAR_RANGE_KM} km")
    print("OK")


if __name__ == "__main__":
    main()

# ============================================================
# 3.2 - INTEGRACION REAL GLM + DMW/DMWV
# ============================================================

import tempfile
from datetime import datetime, timezone
from urllib.request import urlopen, Request

from goes19.goes19_engine import latest_product_object
from goes19.state import (
    DMW_PRODUCTS,
    DMWV_PRODUCTS,
    GOOD_DMW_DQF,
    DMW_QUALITY_WEIGHT,
    _open_wind_arrays,
)


def _download_goes_object(metadata: dict[str, Any], destination: Path) -> Path:
    request = Request(
        metadata["url"],
        headers={"User-Agent": "ClimaAR/3.2"},
    )
    with urlopen(request, timeout=180) as response:
        destination.write_bytes(response.read())
    return destination


def _rasterize_weighted(
    latitude: np.ndarray,
    longitude: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray | None = None,
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    lat = np.asarray(latitude, dtype=np.float64).ravel()
    lon = np.asarray(longitude, dtype=np.float64).ravel()
    val = np.asarray(values, dtype=np.float64).ravel()

    if weights is None:
        weights = np.ones_like(val)
    else:
        weights = np.asarray(weights, dtype=np.float64).ravel()

    valid = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & np.isfinite(val)
        & np.isfinite(weights)
        & (weights > 0)
    )

    lat = lat[valid]
    lon = lon[valid]
    val = val[valid]
    weights = weights[valid]

    if val.size == 0:
        return np.zeros((grid_size, grid_size), dtype=np.float32)

    lat_edges = np.linspace(
        RADAR_LAT - 2.25,
        RADAR_LAT + 2.25,
        grid_size + 1,
    )
    lon_edges = np.linspace(
        RADAR_LON - 3.0,
        RADAR_LON + 3.0,
        grid_size + 1,
    )

    iy = np.searchsorted(lat_edges, lat, side="right") - 1
    ix = np.searchsorted(lon_edges, lon, side="right") - 1

    inside = (
        (iy >= 0)
        & (iy < grid_size)
        & (ix >= 0)
        & (ix < grid_size)
    )

    iy = iy[inside]
    ix = ix[inside]
    val = val[inside]
    weights = weights[inside]

    if val.size == 0:
        return np.zeros((grid_size, grid_size), dtype=np.float32)

    sums = np.zeros((grid_size, grid_size), dtype=np.float64)
    weight_sum = np.zeros((grid_size, grid_size), dtype=np.float64)

    np.add.at(sums, (iy, ix), val * weights)
    np.add.at(weight_sum, (iy, ix), weights)

    result = np.zeros_like(sums)
    valid_cells = weight_sum > 0
    result[valid_cells] = (
        sums[valid_cells] / weight_sum[valid_cells]
    )

    return result.astype(np.float32)


def _read_glm_netcdf(path: Path):
    with xr.open_dataset(
        path,
        engine="netcdf4",
        mask_and_scale=True,
    ) as ds:
        required = (
            "flash_lat",
            "flash_lon",
            "flash_energy",
            "flash_quality_flag",
        )
        missing = [name for name in required if name not in ds]
        if missing:
            raise RuntimeError(
                f"Faltan variables GLM: {missing}"
            )

        lat = np.asarray(
            ds["flash_lat"].values,
            dtype=np.float64,
        )
        lon = np.asarray(
            ds["flash_lon"].values,
            dtype=np.float64,
        )
        energy = np.asarray(
            ds["flash_energy"].values,
            dtype=np.float64,
        )
        quality = np.asarray(
            ds["flash_quality_flag"].values,
            dtype=np.int16,
        )

    valid = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & np.isfinite(energy)
        & (quality == 0)
        & (energy >= 0)
    )

    lat = lat[valid]
    lon = lon[valid]
    energy = energy[valid]

    if lat.size == 0:
        return lat, lon, energy

    distance = haversine_km(
        lat,
        lon,
        RADAR_LAT,
        RADAR_LON,
    )

    inside = distance <= RADAR_RANGE_KM

    return (
        lat[inside],
        lon[inside],
        energy[inside],
    )

def _build_glm_field(
    path: Path,
    grid_size: int = GRID_SIZE,
) -> np.ndarray:

    lat, lon, energy = _read_glm_netcdf(path)

    if energy.size == 0:
        return np.zeros(
            (grid_size, grid_size),
            dtype=np.float32,
        )

    # GLM: densidad/actividad espacial de energía.
    field = _rasterize_weighted(
        lat,
        lon,
        np.maximum(energy, 0.0),
        grid_size=grid_size,
    )

    maximum = float(np.nanmax(field))

    if maximum > 0:
        field = np.log1p(field)
        field /= max(float(np.nanmax(field)), 1e-9)

    return np.nan_to_num(
        field,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(np.float32)


def _build_dmw_fields(
    paths: list[tuple[Path, str]],
    grid_size: int = GRID_SIZE,
) -> list[np.ndarray]:

    speed_lat = []
    speed_lon = []
    speed_val = []
    speed_weight = []

    u_lat = []
    u_lon = []
    u_val = []
    u_weight = []

    v_lat = []
    v_lon = []
    v_val = []
    v_weight = []

    for path, _product in paths:
        with xr.open_dataset(
            path,
            engine="netcdf4",
            mask_and_scale=True,
        ) as ds:
            arrays = _open_wind_arrays(ds)

            lat = arrays["lat"]
            lon = arrays["lon"]
            speed = arrays["speed"]
            direction = arrays["direction"]
            dqf = arrays["dqf"]

        valid = (
            np.isfinite(lat)
            & np.isfinite(lon)
            & np.isfinite(speed)
            & np.isfinite(direction)
            & np.isfinite(dqf)
            & np.isin(
                dqf.astype(np.int64),
                list(GOOD_DMW_DQF),
            )
            & (speed >= 0)
        )

        lat = lat[valid]
        lon = lon[valid]
        speed = speed[valid]
        direction = direction[valid] % 360.0
        dqf = dqf[valid]

        if speed.size == 0:
            continue

        distance = haversine_km(
            lat,
            lon,
            RADAR_LAT,
            RADAR_LON,
        )

        inside = distance <= RADAR_RANGE_KM

        lat = lat[inside]
        lon = lon[inside]
        speed = speed[inside]
        direction = direction[inside]
        dqf = dqf[inside]

        if speed.size == 0:
            continue

        weight = np.zeros_like(speed, dtype=np.float64)

        for code, value in DMW_QUALITY_WEIGHT.items():
            weight[dqf == code] = value

        radians = np.radians(direction)

        u = -speed * np.sin(radians)
        v = -speed * np.cos(radians)

        speed_lat.append(lat)
        speed_lon.append(lon)
        speed_val.append(speed)
        speed_weight.append(weight)

        u_lat.append(lat)
        u_lon.append(lon)
        u_val.append(u)
        u_weight.append(weight)

        v_lat.append(lat)
        v_lon.append(lon)
        v_val.append(v)
        v_weight.append(weight)

    if not speed_val:
        return [
            np.zeros((grid_size, grid_size), dtype=np.float32)
            for _ in range(5)
        ]

    lat = np.concatenate(speed_lat)
    lon = np.concatenate(speed_lon)
    speed = np.concatenate(speed_val)
    weight = np.concatenate(speed_weight)

    u = np.concatenate(u_val)
    v = np.concatenate(v_val)

    direction = np.arctan2(-u, -v)

    speed_field = _rasterize_weighted(
        lat,
        lon,
        _normalize(speed, 0.0, 100.0),
        weight,
        grid_size,
    )

    direction_sin = _rasterize_weighted(
        lat,
        lon,
        np.sin(direction),
        weight,
        grid_size,
    )

    direction_cos = _rasterize_weighted(
        lat,
        lon,
        np.cos(direction),
        weight,
        grid_size,
    )

    u_field = _rasterize_weighted(
        lat,
        lon,
        _normalize(u, -100.0, 100.0),
        weight,
        grid_size,
    )

    v_field = _rasterize_weighted(
        lat,
        lon,
        _normalize(v, -100.0, 100.0),
        weight,
        grid_size,
    )

    return [
        np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        .astype(np.float32)
        for x in (
            speed_field,
            direction_sin,
            direction_cos,
            u_field,
            v_field,
        )
    ]


def build_goes_tensor_operational(
    reference_time: datetime,
    temp_dir: Path | None = None,
    grid_size: int = GRID_SIZE,
) -> dict[str, Any]:

    owns_temp = temp_dir is None

    if temp_dir is None:
        temp_dir = Path(
            tempfile.mkdtemp(prefix="climaar_goes32_")
        )

    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        product_paths: dict[str, Path] = {}

        required_products = {
            "abi_c08": "abi_c08",
            "abi_c13": "abi_c13",
            "cloudtop_height": "cloudtop_height",
            "cloudtop_pressure": "cloudtop_pressure",
        }

        for name, product in required_products.items():
            metadata = latest_product_object(
                product,
                now=reference_time,
                hours_back=6,
            )

            if metadata is None:
                raise RuntimeError(
                    f"No se encontró GOES: {product}"
                )

            path = temp_dir / f"{name}.nc"
            _download_goes_object(metadata, path)
            product_paths[name] = path

        glm_metadata = latest_product_object(
            "glm",
            now=reference_time,
            hours_back=3,
        )

        glm_field = np.zeros(
            (grid_size, grid_size),
            dtype=np.float32,
        )

        if glm_metadata is not None:
            glm_path = temp_dir / "glm.nc"
            _download_goes_object(
                glm_metadata,
                glm_path,
            )
            glm_field = _build_glm_field(
                glm_path,
                grid_size,
            )

        dmw_paths: list[tuple[Path, str]] = []

        for product in {
            **DMW_PRODUCTS,
            **DMWV_PRODUCTS,
        }:
            metadata = latest_product_object(
                product,
                now=reference_time,
                hours_back=3,
            )

            if metadata is None:
                continue

            path = temp_dir / f"{product}.nc"

            _download_goes_object(
                metadata,
                path,
            )

            dmw_paths.append(
                (path, product)
            )

        dmw_fields = _build_dmw_fields(
            dmw_paths,
            grid_size,
        )

        result = build_goes_tensor(
            product_paths["abi_c08"],
            product_paths["abi_c13"],
            product_paths["cloudtop_height"],
            product_paths["cloudtop_pressure"],
            glm={
                "field": glm_field,
            },
            grid_size=grid_size,
        )

        tensor = result["tensor"].clone()

        tensor[0, 4] = torch.from_numpy(
            glm_field
        )

        for index, field in enumerate(
            dmw_fields,
            start=5,
        ):
            tensor[0, index] = torch.from_numpy(
                field
            )

        result["tensor"] = tensor
        result["source"] = "GOES-19 real operational products"
        result["glm_nonzero"] = bool(
            np.any(glm_field > 0)
        )
        result["dmw_nonzero"] = [
            bool(np.any(x != 0))
            for x in dmw_fields
        ]

        return result

    finally:
        if owns_temp:
            import shutil
            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )
