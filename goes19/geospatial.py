from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import xarray as xr


# ============================================================
# ClimaAR - GOES-19 Geospatial Engine
# ============================================================

# Referencia espacial oficial utilizada por ClimaAR/RMA10
RADAR_LAT = -38.7342
RADAR_LON = -62.1654

# Dominio operacional de ClimaAR
RADAR_RANGE_KM = 240.0

# Margen angular para seleccionar una ventana preliminar.
# Después se aplica el radio exacto de 240 km.
WINDOW_MARGIN = 1.50


# ============================================================
# Utilidades
# ============================================================

def haversine_km(
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: float,
    lon0: float,
) -> np.ndarray:
    """Calcula distancia geodésica aproximada en kilómetros."""

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
    Convierte coordenadas GOES ABI Fixed Grid
    a latitud/longitud.

    x e y están expresados en radianes.
    h es la distancia desde el centro de la Tierra
    hasta el satélite.
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

    b = (
        -2.0
        * h
        * cos_x
        * cos_y
    )

    c = h**2 - r_eq**2

    discriminant = (
        b**2
        - 4.0 * a * c
    )

    valid = discriminant >= 0.0

    discriminant = np.where(
        valid,
        discriminant,
        np.nan,
    )

    rs = (
        -b
        - np.sqrt(discriminant)
    ) / (2.0 * a)

    sx = (
        rs
        * cos_x
        * cos_y
    )

    sy = (
        -rs
        * sin_x
    )

    sz = (
        rs
        * cos_x
        * sin_y
    )

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


def geodetic_to_fixed_grid(
    latitude: float,
    longitude: float,
    *,
    lambda_0: float,
    r_eq: float,
    r_pol: float,
    h: float,
) -> tuple[float, float]:
    """
    Convierte una coordenada lat/lon a GOES ABI Fixed Grid.
    """

    lat = math.radians(latitude)
    lon = math.radians(longitude)

    eccentricity_sq = (
        1.0
        - (r_pol / r_eq) ** 2
    )

    n = r_eq / math.sqrt(
        1.0
        - eccentricity_sq
        * math.sin(lat) ** 2
    )

    x_earth = (
        n
        * math.cos(lat)
        * math.cos(
            lon - lambda_0
        )
    )

    y_earth = (
        n
        * math.cos(lat)
        * math.sin(
            lon - lambda_0
        )
    )

    z_earth = (
        n
        * (1.0 - eccentricity_sq)
        * math.sin(lat)
    )

    x_scan = math.atan2(
        -y_earth,
        h - x_earth,
    )

    y_scan = math.atan2(
        z_earth,
        math.sqrt(
            (h - x_earth) ** 2
            + y_earth**2
        ),
    )

    return (
        x_scan,
        y_scan,
    )


# ============================================================
# Proyección GOES
# ============================================================

def get_projection(
    ds: xr.Dataset,
) -> tuple[float, float, float, float]:
    """Obtiene los parámetros oficiales de proyección GOES."""

    projection = None

    for name in ds.variables:
        variable = ds[name]

        if (
            variable.attrs.get(
                "grid_mapping_name"
            )
            == "geostationary"
        ):
            projection = variable
            break

    if projection is None:
        raise ValueError(
            "No se encontró una variable de "
            "proyección geostationary."
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
        float(
            attrs[
                "longitude_of_projection_origin"
            ]
        )
    )

    r_eq = float(
        attrs[
            "semi_major_axis"
        ]
    )

    r_pol = float(
        attrs[
            "semi_minor_axis"
        ]
    )

    # perspective_point_height es la altura sobre
    # la superficie terrestre.
    # Para las ecuaciones de la proyección necesitamos
    # la distancia desde el centro de la Tierra.
    h = (
        float(
            attrs[
                "perspective_point_height"
            ]
        )
        + r_eq
    )

    return (
        lambda_0,
        r_eq,
        r_pol,
        h,
    )


# ============================================================
# Ventana espacial
# ============================================================

def get_radar_fixed_grid_position(
    ds: xr.Dataset,
) -> tuple[float, float]:
    """Obtiene la posición del radar RMA10 en GOES Fixed Grid."""

    (
        lambda_0,
        r_eq,
        r_pol,
        h,
    ) = get_projection(ds)

    return geodetic_to_fixed_grid(
        RADAR_LAT,
        RADAR_LON,
        lambda_0=lambda_0,
        r_eq=r_eq,
        r_pol=r_pol,
        h=h,
    )


def select_spatial_window(
    ds: xr.Dataset,
) -> tuple[slice, slice]:
    """
    Selecciona una ventana alrededor del radar RMA10.

    La ventana es deliberadamente algo mayor que 240 km.
    Después se aplica el círculo exacto de 240 km.
    """

    x_center, y_center = (
        get_radar_fixed_grid_position(ds)
    )

    # Conversión aproximada de 240 km a ángulo terrestre.
    earth_radius_km = 6371.0088

    angular_radius = (
        RADAR_RANGE_KM
        / earth_radius_km
    )

    angular_radius *= WINDOW_MARGIN

    x_values = ds["x"].values
    y_values = ds["y"].values

    x_indices = np.where(
        np.abs(
            x_values - x_center
        )
        <= angular_radius
    )[0]

    y_indices = np.where(
        np.abs(
            y_values - y_center
        )
        <= angular_radius
    )[0]

    if (
        len(x_indices) == 0
        or len(y_indices) == 0
    ):
        raise ValueError(
            "No se encontró una ventana GOES "
            "alrededor del radar RMA10."
        )

    return (
        slice(
            int(x_indices[0]),
            int(x_indices[-1]) + 1,
        ),
        slice(
            int(y_indices[0]),
            int(y_indices[-1]) + 1,
        ),
    )


# ============================================================
# Dominio ClimaAR
# ============================================================

def prepare_goes_domain(
    file_path: str | Path,
    variable: str = "CMI",
) -> dict[str, np.ndarray]:
    """
    Procesa un NetCDF GOES y devuelve únicamente
    el dominio operacional de ClimaAR.

    Flujo:

        NetCDF completo
              ↓
        ventana alrededor RMA10
              ↓
        coordenadas lat/lon
              ↓
        círculo exacto de 240 km
              ↓
        datos finales
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
                f"La variable '{variable}' "
                "no existe en el NetCDF."
            )

        (
            lambda_0,
            r_eq,
            r_pol,
            h,
        ) = get_projection(ds)

        x_slice, y_slice = (
            select_spatial_window(ds)
        )

        subset = ds[
            variable
        ].isel(
            y=y_slice,
            x=x_slice,
        )

        data = subset.values

        x_values = subset["x"].values
        y_values = subset["y"].values

        x_grid, y_grid = np.meshgrid(
            x_values,
            y_values,
        )

        latitude, longitude = (
            fixed_grid_to_latlon(
                x_grid,
                y_grid,
                lambda_0=lambda_0,
                r_eq=r_eq,
                r_pol=r_pol,
                h=h,
            )
        )

        distance_km = haversine_km(
            latitude,
            longitude,
            RADAR_LAT,
            RADAR_LON,
        )

        mask = (
            distance_km
            <= RADAR_RANGE_KM
        )

        return {
            "data": data[mask],
            "latitude": latitude[mask],
            "longitude": longitude[mask],
            "distance_km": distance_km[mask],
        }


# ============================================================
# Prueba
# ============================================================

def main() -> None:

    print(
        "ClimaAR GOES-19 geospatial module"
    )

    print(
        f"Radar: "
        f"{RADAR_LAT}, {RADAR_LON}"
    )

    print(
        f"Radio operativo: "
        f"{RADAR_RANGE_KM} km"
    )

    if len(sys.argv) < 2:
        print(
            "Modo básico: módulo cargado correctamente."
        )
        print(
            "Para procesar un NetCDF:"
        )
        print(
            "python -m goes19.geospatial "
            "/ruta/al/archivo.nc"
        )
        return

    file_path = sys.argv[1]

    print(
        f"Procesando GOES: {file_path}"
    )

    result = prepare_goes_domain(
        file_path
    )

    data = result["data"]
    latitude = result["latitude"]
    longitude = result["longitude"]
    distance = result["distance_km"]

    print(
        f"Puntos dentro de 240 km: "
        f"{len(data):,}"
    )

    if len(data) == 0:
        raise RuntimeError(
            "El recorte de 240 km quedó vacío."
        )

    print(
        f"Latitud: "
        f"{latitude.min():.4f} "
        f"a "
        f"{latitude.max():.4f}"
    )

    print(
        f"Longitud: "
        f"{longitude.min():.4f} "
        f"a "
        f"{longitude.max():.4f}"
    )

    print(
        f"Distancia máxima: "
        f"{distance.max():.2f} km"
    )

    print(
        f"Valor mínimo: "
        f"{np.nanmin(data):.3f}"
    )

    print(
        f"Valor máximo: "
        f"{np.nanmax(data):.3f}"
    )

    print(
        "OK: GOES -> dominio ClimaAR 240 km"
    )


if __name__ == "__main__":
    main()
