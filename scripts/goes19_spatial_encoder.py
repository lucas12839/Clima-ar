from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


RADAR_LAT = -38.7342
RADAR_LON = -62.1654
RADAR_RADIUS_KM = 240.0


def haversine_km(lat, lon, lat0=RADAR_LAT, lon0=RADAR_LON):
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    r = 6371.0
    p1 = np.radians(lat)
    p2 = np.radians(lat0)
    dp = np.radians(lat - lat0)
    dl = np.radians(lon - lon0)

    a = (
        np.sin(dp / 2.0) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    )

    return 2.0 * r * np.arcsin(np.sqrt(a))


def extract_goes_field(path: str | Path, variable: str = "CMI"):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise KeyError(
                f"Variable {variable!r} no encontrada. "
                f"Disponibles: {list(ds.data_vars)}"
            )

        da = ds[variable]

        while da.ndim > 2:
            da = da.isel({da.dims[0]: 0})

        data = np.asarray(da.values, dtype=np.float32)

        if data.ndim != 2:
            raise ValueError(f"Campo GOES no 2D: shape={data.shape}")

        x = np.asarray(ds["x"].values)
        y = np.asarray(ds["y"].values)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("Coordenadas x/y GOES inválidas")

        return {
            "data": data,
            "x": x,
            "y": y,
            "variable": variable,
            "source": str(path),
        }


def build_spatial_field(path: str | Path, variable: str = "CMI"):
    field = extract_goes_field(path, variable)

    data = field["data"]
    x = field["x"]
    y = field["y"]

    xx, yy = np.meshgrid(x, y)
    valid = np.isfinite(data)

    if not np.any(valid):
        raise ValueError("El campo GOES no contiene datos válidos")

    return {
        "data": data,
        "x": xx,
        "y": yy,
        "valid_mask": valid,
        "variable": variable,
        "source": field["source"],
    }


def summarize_spatial_field(field):
    data = np.asarray(field["data"])
    valid = np.asarray(field["valid_mask"], dtype=bool)
    values = data[valid]

    if values.size == 0:
        raise ValueError("Campo GOES vacío")

    return {
        "variable": field["variable"],
        "shape": list(data.shape),
        "valid_pixels": int(values.size),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
    }


if __name__ == "__main__":
    print("GOES-19 SPATIAL MODULE OK")
    print(f"centro = ({RADAR_LAT}, {RADAR_LON})")
    print(f"radio = {RADAR_RADIUS_KM} km")
