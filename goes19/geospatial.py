from __future__ import annotations

import math
import sys
from contextlib import contextmanager
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

# Margen de la ventana preliminar: el borde del círculo se toma un 3 % más
# grande y se suman 2 píxeles por lado. Después se aplica el radio exacto.
WINDOW_MARGIN = 1.03
WINDOW_EXTRA_PIXELS = 2

# Lectura parcial por internet: tamaño de cada pedido al servidor de NOAA.
REMOTE_BLOCK_BYTES = 2 ** 18


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

    Fórmulas oficiales de NOAA (GOES-R Product User Guide, vol. 3, 4.2.8.1).
    Verificadas con el ejemplo del manual (x = -0.024052, y = 0.095340,
    lon0 = -75° -> 33.846162°, -84.690932°).

    x e y están expresados en radianes (x crece al este, y al norte).
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
            cos_y**2
            + ratio**2 * sin_y**2
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

    # Fuera del disco terrestre (espacio) no hay latitud/longitud.
    discriminant = np.where(
        discriminant >= 0.0,
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

    latitude = np.arctan(
        ratio**2
        * sz
        / np.sqrt(
            (h - sx) ** 2
            + sy**2
        )
    )

    longitude = (
        lambda_0
        - np.arctan(
            sy
            / (h - sx)
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
    altura_m: float = 0.0,
) -> tuple[float, float]:
    """
    Convierte una coordenada lat/lon a GOES ABI Fixed Grid (radianes).

    Fórmulas oficiales de NOAA (GOES-R Product User Guide, 4.2.8.2), con el
    radio de curvatura del elipsoide. altura_m: altura del punto sobre el
    elipsoide (0 = suelo). Con la altura del tope de una nube se obtiene
    dónde la "ve" el satélite.
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

    x_earth = (n + altura_m) * math.cos(lat) * math.cos(lon - lambda_0)
    y_earth = (n + altura_m) * math.cos(lat) * math.sin(lon - lambda_0)
    z_earth = (n * (1.0 - eccentricity_sq) + altura_m) * math.sin(lat)

    s_x = h - x_earth
    s_y = -y_earth
    s_z = z_earth

    x_scan = math.asin(
        -s_y
        / math.sqrt(s_x**2 + s_y**2 + s_z**2)
    )
    y_scan = math.atan(s_z / s_x)

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


def destino_km(
    lat0: float,
    lon0: float,
    rumbo_grados: float,
    distancia_km: float,
) -> tuple[float, float]:
    """Punto a una distancia y rumbo dados (esfera, suficiente para 300 km)."""

    earth_radius_km = 6371.0088
    d = distancia_km / earth_radius_km
    rumbo = math.radians(rumbo_grados)
    lat1 = math.radians(lat0)
    lon1 = math.radians(lon0)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(d)
        + math.cos(lat1) * math.sin(d) * math.cos(rumbo)
    )
    lon2 = lon1 + math.atan2(
        math.sin(rumbo) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def _indices_en_rango(valores: np.ndarray, minimo: float, maximo: float, extra: int) -> slice:
    paso = float(np.median(np.abs(np.diff(valores)))) if valores.size > 1 else 0.0
    margen = extra * paso
    idx = np.where((valores >= minimo - margen) & (valores <= maximo + margen))[0]
    if idx.size == 0:
        raise ValueError(
            "No se encontró una ventana GOES "
            "alrededor del radar RMA10."
        )
    return slice(int(idx[0]), int(idx[-1]) + 1)


def select_spatial_window(
    ds: xr.Dataset,
    radio_km: float = RADAR_RANGE_KM,
) -> tuple[slice, slice]:
    """
    Selecciona la ventana mínima de la grilla GOES que contiene el círculo
    de radio_km alrededor del radar RMA10.

    Se proyecta el borde del círculo (cada 5°) a la grilla del satélite y se
    toman sus extremos, con un margen chico. Así solo se leen los píxeles
    necesarios (importante para la lectura parcial por internet).
    """

    (
        lambda_0,
        r_eq,
        r_pol,
        h,
    ) = get_projection(ds)

    borde = [
        geodetic_to_fixed_grid(
            *destino_km(RADAR_LAT, RADAR_LON, rumbo, radio_km * WINDOW_MARGIN),
            lambda_0=lambda_0,
            r_eq=r_eq,
            r_pol=r_pol,
            h=h,
        )
        for rumbo in range(0, 360, 5)
    ]
    xs = [p[0] for p in borde]
    ys = [p[1] for p in borde]

    x_values = np.asarray(ds["x"].values, dtype=np.float64)
    y_values = np.asarray(ds["y"].values, dtype=np.float64)

    return (
        _indices_en_rango(x_values, min(xs), max(xs), WINDOW_EXTRA_PIXELS),
        _indices_en_rango(y_values, min(ys), max(ys), WINDOW_EXTRA_PIXELS),
    )


@contextmanager
def abrir_goes(origen: str | Path):
    """
    Abre un NetCDF GOES local o, si origen es una dirección https, lo lee por
    internet pidiendo solo los pedazos que se usan (lectura parcial).
    """

    texto = str(origen)
    if texto.startswith("http://") or texto.startswith("https://"):
        import fsspec

        fs = fsspec.filesystem("http")
        archivo = fs.open(
            texto,
            "rb",
            block_size=REMOTE_BLOCK_BYTES,
            cache_type="blockcache",
        )
        try:
            ds = xr.open_dataset(archivo, engine="h5netcdf")
            try:
                yield ds
            finally:
                ds.close()
        finally:
            archivo.close()
        return

    path = Path(origen)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo GOES: {path}"
        )
    with xr.open_dataset(path, engine="netcdf4") as ds:
        yield ds


def fraccion_escaneo(ds: xr.Dataset, y_punto: float) -> float | None:
    """
    Parte del barrido (0 = inicio, 1 = fin) en la que el satélite pasó por la
    fila y_punto. El disco completo se barre de norte a sur en 10 min.
    """

    try:
        attrs = ds["y_image_bounds"].values if "y_image_bounds" in ds else None
        if attrs is not None and np.size(attrs) == 2:
            norte, sur = float(np.max(attrs)), float(np.min(attrs))
        else:
            y = np.asarray(ds["y"].values, dtype=np.float64)
            norte, sur = float(np.nanmax(y)), float(np.nanmin(y))
    except Exception:
        return None
    if norte <= sur:
        return None
    return float(np.clip((norte - y_punto) / (norte - sur), 0.0, 1.0))


# ============================================================
# Dominio ClimaAR
# ============================================================

def prepare_goes_domain_multi(
    origen: str | Path,
    variables: list[str] | tuple[str, ...],
    radio_km: float = RADAR_RANGE_KM,
) -> dict:
    """
    Lee una o varias variables de un mismo NetCDF GOES y devuelve solo los
    píxeles a <= radio_km del radar RMA10.

    origen puede ser un archivo local o una dirección https (lectura
    parcial: solo se bajan los pedazos de la ventana).

    Devuelve:
        datos: {variable: valores}, latitude, longitude, distance_km,
        proyeccion (lambda_0, r_eq, r_pol, h), fraccion_escaneo_radar.
    """

    with abrir_goes(origen) as ds:

        faltan = [v for v in variables if v not in ds]
        if faltan:
            raise ValueError(
                f"La variable '{faltan[0]}' "
                "no existe en el NetCDF."
            )

        (
            lambda_0,
            r_eq,
            r_pol,
            h,
        ) = get_projection(ds)

        x_slice, y_slice = (
            select_spatial_window(ds, radio_km=radio_km)
        )

        x_values = np.asarray(ds["x"].values[x_slice], dtype=np.float64)
        y_values = np.asarray(ds["y"].values[y_slice], dtype=np.float64)

        x_radar, y_radar = geodetic_to_fixed_grid(
            RADAR_LAT,
            RADAR_LON,
            lambda_0=lambda_0,
            r_eq=r_eq,
            r_pol=r_pol,
            h=h,
        )
        fraccion = fraccion_escaneo(ds, y_radar)

        crudos = {
            variable: ds[variable].isel(y=y_slice, x=x_slice).values
            for variable in variables
        }

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

    mask = np.isfinite(distance_km) & (
        distance_km
        <= radio_km
    )

    return {
        "datos": {variable: valores[mask] for variable, valores in crudos.items()},
        "latitude": latitude[mask],
        "longitude": longitude[mask],
        "distance_km": distance_km[mask],
        "proyeccion": (lambda_0, r_eq, r_pol, h),
        "fraccion_escaneo_radar": fraccion,
    }


def prepare_goes_domain(
    file_path: str | Path,
    variable: str = "CMI",
    radio_km: float = RADAR_RANGE_KM,
) -> dict[str, np.ndarray]:
    """
    Procesa un NetCDF GOES y devuelve únicamente
    el dominio operacional de ClimaAR.

    Flujo:

        NetCDF (local o lectura parcial por internet)
              ↓
        ventana mínima alrededor RMA10
              ↓
        coordenadas lat/lon
              ↓
        círculo exacto de 240 km
              ↓
        datos finales
    """

    result = prepare_goes_domain_multi(file_path, [variable], radio_km=radio_km)
    return {
        "data": result["datos"][variable],
        "latitude": result["latitude"],
        "longitude": result["longitude"],
        "distance_km": result["distance_km"],
        "proyeccion": result["proyeccion"],
        "fraccion_escaneo_radar": result["fraccion_escaneo_radar"],
    }


def posicion_aparente(
    latitude: float,
    longitude: float,
    altura_km: float,
    proyeccion: tuple[float, float, float, float],
) -> tuple[float, float]:
    """
    Dónde ve el satélite un punto que está a altura_km sobre (latitude,
    longitude). GOES-19 mira Bahía Blanca de costado: el tope de una tormenta
    alta aparece corrido hacia el sur y el este (paralaje).
    """

    lambda_0, r_eq, r_pol, h = proyeccion
    if altura_km <= 0:
        return latitude, longitude
    x, y = geodetic_to_fixed_grid(
        latitude,
        longitude,
        lambda_0=lambda_0,
        r_eq=r_eq,
        r_pol=r_pol,
        h=h,
        altura_m=altura_km * 1000.0,
    )
    lat_ap, lon_ap = fixed_grid_to_latlon(
        np.array([x]),
        np.array([y]),
        lambda_0=lambda_0,
        r_eq=r_eq,
        r_pol=r_pol,
        h=h,
    )
    return float(lat_ap[0]), float(lon_ap[0])


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
