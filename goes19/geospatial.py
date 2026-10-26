from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import xarray as xr


# ============================================================
# ClimaAR / GOES-19
# Conversión GOES Fixed Grid -> Lat/Lon
# y selección del dominio operativo de ClimaAR.
# ============================================================

RADAR_LAT = -38.7342
RADAR_LON = -62.1654
RADAR_RANGE_KM = 240.0


def _get_attr(ds: xr.Dataset, name: str) -> float:
    """Obtiene un atributo numérico del NetCDF."""
    if name not in ds.attrs:
        raise ValueError(f"Falta atributo GOES requerido: {name}")

    return float(ds.attrs[name])


def fixed_grid_to_latlon(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lambda_0: float,
    r_eq: float,
    r_pol: float,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convierte coordenadas GOES ABI Fixed Grid (radianes)
    a latitud/longitud.

    Implementación basada en la proyección geostacionaria
    utilizada por GOES-R.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_y = np.sin(y)
    cos_y = np.cos(y)

    ratio = r_eq / r_pol

    a = (
        sin_x**2
        + cos_x**2
        * (
            sin_y**2
            + ratio**2 * cos_y**2
        )
    )

    b = -2.0 * h * cos_x * cos_y

    c = h**2 - r_eq**2

    discriminant = b**2 - 4.0 * a * c

    # Los puntos fuera de la Tierra visible pueden producir
    # discriminantes negativos.
    valid = discriminant >= 0.0

    discriminant = np.where(
        valid,
        discriminant,
        np.nan,
    )

    rs = (
        -b - np.sqrt(discriminant)
    ) / (2.0 * a)

    sx = rs * cos_x * cos_y
    sy = -rs * sin_x
    sz = rs * cos_x * sin_y

    longitude = (
        np.arctan2(
            sy,
            h - sx,
        )
        + lambda_0
    )

    latitude = np.arctan(
        ratio**2
        * sz
        / np.sqrt(
            (h - sx) ** 2
            + sy**2
        )
    )

    return (
        np.degrees(latitude),
        np.degrees(longitude),
    )


def haversine_km(
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: float,
    lon0: float,
) -> np.ndarray:
    """Distancia geodésica aproximada en km."""

    earth_radius_km = 6371.0088

    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = math.radians(lat0)
    lon2 = math.radians(lon0)

    dlat = lat1 - lat2
    dlon = lon1 - lon2

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * math.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return (
        2.0
        * earth_radius_km
        * np.arcsin(
            np.sqrt(
                np.clip(a, 0.0, 1.0)
            )
        )
    )


def get_goes_latlon(
    ds: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Obtiene las coordenadas lat/lon de una imagen GOES ABI.
    """

    if "x" not in ds.coords:
        raise ValueError("El NetCDF GOES no contiene coordenada 'x'.")

    if "y" not in ds.coords:
        raise ValueError("El NetCDF GOES no contiene coordenada 'y'.")

    x = ds["x"].values
    y = ds["y"].values

    x_grid, y_grid = np.meshgrid(
        x,
        y,
    )

    # GOES normalmente almacena estos valores en:
    # proj_* / atributos de la variable de proyección.
    projection = None

    for name in ds.variables:
        variable = ds[name]

        if variable.attrs.get("grid_mapping_name") == "geostationary":
            projection = variable
            break

    if projection is None:
        raise ValueError(
            "No se encontró la variable de proyección "
            "geostationary en el NetCDF GOES."
        )

    attrs = projection.attrs

    required = (
        "longitude_of_projection_origin",
        "perspective_point_height",
        "semi_major_axis",
        "semi_minor_axis",
    )

    for name in required:
        if name not in attrs:
            raise ValueError(
                f"Falta atributo de proyección GOES: {name}"
            )

    lambda_0 = math.radians(
        float(attrs["longitude_of_projection_origin"])
    )

    h = float(attrs["perspective_point_height"])

    r_eq = float(attrs["semi_major_axis"])
    r_pol = float(attrs["semi_minor_axis"])

    return fixed_grid_to_latlon(
        x_grid,
        y_grid,
        lambda_0=lambda_0,
        r_eq=r_eq,
        r_pol=r_pol,
        h=h,
    )


def create_climaar_mask(
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    radar_lat: float = RADAR_LAT,
    radar_lon: float = RADAR_LON,
    radius_km: float = RADAR_RANGE_KM,
) -> np.ndarray:
    """
    Genera la máscara espacial operacional de ClimaAR.

    Incluye solamente puntos GOES ubicados dentro de
    los 240 km alrededor del radar RMA10.
    """

    distance = haversine_km(
        latitude,
        longitude,
        radar_lat,
        radar_lon,
    )

    return distance <= radius_km


def prepare_goes_domain(
    file_path: str | Path,
    variable: str = "CMI",
) -> dict[str, np.ndarray]:
    """
    Abre un NetCDF GOES, calcula lat/lon y devuelve
    solamente el dominio operacional de ClimaAR.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo GOES: {file_path}"
        )

    with xr.open_dataset(
        file_path,
        engine="netcdf4",
    ) as ds:

        if variable not in ds:
            raise ValueError(
                f"La variable '{variable}' no existe en el NetCDF."
            )

        latitude, longitude = get_goes_latlon(ds)

        mask = create_climaar_mask(
            latitude,
            longitude,
        )

        data = ds[variable].values

        return {
            "data": data[mask],
            "latitude": latitude[mask],
            "longitude": longitude[mask],
            "distance_km": haversine_km(
                latitude[mask],
                longitude[mask],
                RADAR_LAT,
                RADAR_LON,
            ),
        }


if __name__ == "__main__":
    print("ClimaAR GOES-19 geospatial module")
    print(f"Radar: {RADAR_LAT}, {RADAR_LON}")
    print(f"Radio operativo: {RADAR_RANGE_KM} km")
