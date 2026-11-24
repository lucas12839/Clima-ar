from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
import xarray as xr

from . import por_tormenta
from .goes19_engine import latest_product_object, product_object_for_scan
from .geospatial import (
    RADAR_LAT,
    RADAR_LON,
    RADAR_RANGE_KM,
    prepare_goes_domain_multi,
)


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "data" / "goes19" / "goes19_state.json"
RADAR_TIMESTAMP_FILE = BASE_DIR / "data" / "radar" / "radar_timestamp.txt"
RADAR_NOWCAST_FILE = BASE_DIR / "data" / "radar" / "radar_nowcast.json"
METEO_FILE = BASE_DIR / "data" / "ia" / "estado_meteorologico.json"

# Punto 2: productos solo para señales por tormenta (no entran en las
# estadísticas generales). Variable principal, filtro de calidad.
SENALES_PRODUCTS = {
    "cloudtop_height_2km": ("HT", "DQF0"),     # altura del tope, 2 km
    "cloud_phase": ("Phase", "DQF0"),          # fase: agua/sobreenfriada/mixta/hielo
}

# Para los datos por tormenta se lee un poco más que 240 km: una tormenta en el
# borde puede verse corrida hacia afuera por la perspectiva del satélite.
RADIO_TORMENTAS_KM = RADAR_RANGE_KM + 25.0

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


def leer_dominio(
    metadata: dict,
    variables: list[str],
    temp_dir: Path,
    nombre: str,
) -> tuple[dict, dict]:
    """Lee solo la zona de Bahía Blanca. Primero por internet, pidiendo solo
    los pedazos del archivo que hacen falta; si falla, baja el archivo entero
    (como antes). Devuelve (dominio, info_de_lectura)."""
    inicio = datetime.now(timezone.utc)
    try:
        dominio = prepare_goes_domain_multi(metadata["url"], variables, radio_km=RADIO_TORMENTAS_KM)
        modo, error = "parcial", None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:300]
        path = temp_dir / f"{nombre}.nc"
        response = requests.get(metadata["url"], timeout=180)
        response.raise_for_status()
        path.write_bytes(response.content)
        dominio = prepare_goes_domain_multi(path, variables, radio_km=RADIO_TORMENTAS_KM)
        path.unlink(missing_ok=True)
        modo = "completo"
    info = {
        "modo": modo,
        "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 2),
    }
    if error:
        info["error_parcial"] = error
    return dominio, info


def _hora_satelite(metadata: dict, dominio: dict) -> datetime | None:
    return por_tormenta.hora_local_escaneo(
        por_tormenta.parse_utc(metadata.get("start_time_utc")),
        por_tormenta.parse_utc(metadata.get("end_time_utc")),
        dominio.get("fraccion_escaneo_radar"),
    )


def _campo(valores: np.ndarray, dominio: dict, hora: datetime | None) -> dict:
    return {
        "valores": valores,
        "latitude": dominio["latitude"],
        "longitude": dominio["longitude"],
        "proyeccion": dominio.get("proyeccion"),
        "hora_satelite": hora,
    }


def _buscar_objeto(product: str, reference_time: datetime, inicio_escaneo: str | None) -> tuple[dict, bool]:
    """Archivo del mismo barrido que el canal 13; si no está publicado, el más
    reciente anterior a la imagen de radar (marcado como otro barrido)."""
    if inicio_escaneo:
        obj = product_object_for_scan(product, inicio_escaneo, now=reference_time)
        if obj is not None:
            return obj, True
    obj = latest_product_object(product, now=reference_time, hours_back=6)
    if obj is None:
        raise RuntimeError(f"No se encontró producto GOES: {product}")
    mismo = bool(inicio_escaneo) and obj.get("start_time_utc") == inicio_escaneo
    return obj, mismo or not inicio_escaneo


def process_product(
    product: str,
    variable: str,
    temp_dir: Path,
    reference_time: datetime,
    inicio_escaneo: str | None = None,
    campos: dict | None = None,
) -> dict:
    metadata, mismo_escaneo = _buscar_objeto(product, reference_time, inicio_escaneo)
    dominio, lectura = leer_dominio(metadata, [variable], temp_dir, product)
    valores = np.asarray(dominio["datos"][variable], dtype=np.float64)
    en_240 = dominio["distance_km"] <= RADAR_RANGE_KM
    hora = _hora_satelite(metadata, dominio)
    if campos is not None:
        campos[product] = _campo(valores, dominio, hora)
    extra = {}
    if product == "abi_c13":
        guardar_c13_ahora(dominio, hora, C13_RADAR)  # el barrido de la hora del radar
        from .topes_localidad import por_localidad

        try:
            extra["por_localidad"] = por_localidad(
                valores[en_240], dominio["latitude"][en_240], dominio["longitude"][en_240],
                metadata.get("end_time_utc"), OUTPUT_FILE,
            )
        except Exception as exc:
            extra["por_localidad_error"] = str(exc)
    return {
        **extra,
        "status": "ok",
        "source": {
            **build_source_metadata(product, metadata),
            "age_minutes": age_minutes(metadata, reference_time),
        },
        "domain": {
            "center_lat": RADAR_LAT,
            "center_lon": RADAR_LON,
            "radius_km": RADAR_RANGE_KM,
            "points": int(np.sum(en_240)),
        },
        "statistics": calculate_statistics(valores[en_240]),
        "lectura": lectura,
        "sincronizacion": {
            **por_tormenta.sincronizacion(hora, reference_time),
            "mismo_escaneo_que_c13": mismo_escaneo,
        },
    }


def process_cloudtop(
    product: str,
    variable: str,
    temp_dir: Path,
    reference_time: datetime,
    inicio_escaneo: str | None = None,
    campos: dict | None = None,
) -> dict:
    metadata, mismo_escaneo = _buscar_objeto(product, reference_time, inicio_escaneo)
    dominio, lectura = leer_dominio(metadata, [variable, "DQF"], temp_dir, product)
    values_all = np.asarray(dominio["datos"][variable], dtype=np.float64)
    dqf_all = np.asarray(dominio["datos"]["DQF"], dtype=np.float64)
    en_240 = dominio["distance_km"] <= RADAR_RANGE_KM
    hora = _hora_satelite(metadata, dominio)
    buenos = np.isfinite(values_all) & np.isfinite(dqf_all) & (dqf_all == 0)
    if campos is not None:
        campos[product] = _campo(np.where(buenos, values_all, np.nan), dominio, hora)

    values = values_all[en_240]
    dqf = dqf_all[en_240]
    valid = buenos[en_240]
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
        "lectura": lectura,
        "sincronizacion": {
            **por_tormenta.sincronizacion(hora, reference_time),
            "mismo_escaneo_que_c13": mismo_escaneo,
        },
    }


def process_senal(
    product: str,
    temp_dir: Path,
    reference_time: datetime,
    inicio_escaneo: str | None,
    campos: dict,
) -> dict:
    """Producto para señales por tormenta: solo píxeles de buena calidad."""
    variable, _ = SENALES_PRODUCTS[product]
    metadata, mismo_escaneo = _buscar_objeto(product, reference_time, inicio_escaneo)
    dominio, lectura = leer_dominio(metadata, [variable, "DQF"], temp_dir, product)
    valores = np.asarray(dominio["datos"][variable], dtype=np.float64)
    dqf = np.asarray(dominio["datos"]["DQF"], dtype=np.float64)
    buenos = np.isfinite(valores) & np.isfinite(dqf) & (dqf == 0)
    hora = _hora_satelite(metadata, dominio)
    campos[product] = _campo(np.where(buenos, valores, np.nan), dominio, hora)
    en_240 = dominio["distance_km"] <= RADAR_RANGE_KM
    return {
        "status": "ok",
        "source": {
            **build_source_metadata(product, metadata),
            "age_minutes": age_minutes(metadata, reference_time),
        },
        "pixeles_buenos_240km": int(np.sum(buenos & en_240)),
        "lectura": lectura,
        "sincronizacion": {
            **por_tormenta.sincronizacion(hora, reference_time),
            "mismo_escaneo_que_c13": mismo_escaneo,
        },
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


def leer_c13_previos(temp_dir: Path, reference_time: datetime,
                     inicio_escaneo: str | None) -> tuple[dict[int, dict], dict]:
    """Canal 13 de los barridos de 10 y 20 min antes del de referencia (punto 5)."""
    from datetime import timedelta

    campos: dict[int, dict] = {}
    info: dict[str, Any] = {}
    inicio = por_tormenta.parse_utc(inicio_escaneo)
    if inicio is None:
        return campos, {"estado": "sin_barrido_de_referencia"}
    for minutos in (10, 20):
        objetivo = por_tormenta.iso(inicio - timedelta(minutes=minutos))
        try:
            metadata = product_object_for_scan("abi_c13", objetivo, now=reference_time,
                                               tolerance_seconds=60.0)
            if metadata is None:
                info[str(minutos)] = {"estado": "no_publicado"}
                continue
            dominio, lectura = leer_dominio(metadata, ["CMI"], temp_dir, f"abi_c13_{minutos}")
            campos[minutos] = _campo(np.asarray(dominio["datos"]["CMI"], dtype=np.float64), dominio,
                                     _hora_satelite(metadata, dominio))
            info[str(minutos)] = {"estado": "ok", "inicio_utc": metadata.get("start_time_utc"), **lectura}
        except Exception as exc:
            info[str(minutos)] = {"estado": "error", "error": f"{type(exc).__name__}: {exc}"[:200]}
    return campos, info


C13_AHORA = Path(tempfile.gettempdir()) / "climaar_c13_ahora.npz"
C13_RADAR = Path(tempfile.gettempdir()) / "climaar_c13_radar.npz"


MAPA_RAYOS_MAX = 600


def _rayos_para_mapa(rayos: dict) -> list[list[float]]:
    """[lat, lon, minutos antes de la imagen de radar], los más nuevos primero.
    Redondeado a ~100 m y con tope, para que el estado siga siendo liviano."""
    try:
        lat = np.asarray(rayos["latitude"], dtype=float)
        lon = np.asarray(rayos["longitude"], dtype=float)
        minutos = np.asarray(rayos["minutos"], dtype=float)
    except Exception:
        return []
    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(minutos)
    orden = np.argsort(minutos[ok], kind="stable")[:MAPA_RAYOS_MAX]
    return [
        [round(float(a), 3), round(float(b), 3), round(float(c), 1)]
        for a, b, c in zip(lat[ok][orden], lon[ok][orden], minutos[ok][orden])
    ]


def guardar_c13_ahora(dominio: dict, hora: datetime | None, destino: Path | None = None) -> None:
    """Deja la imagen más reciente del canal 13 (solo la zona) para el corrector del
    nowcast en capas, que corre más tarde en el MISMO ciclo. No se guarda en git."""
    try:
        np.savez_compressed(destino or C13_AHORA, lat=np.asarray(dominio["latitude"], np.float32),
                            lon=np.asarray(dominio["longitude"], np.float32),
                            temp_c=(np.asarray(dominio["datos"]["CMI"], np.float32) - 273.15),
                            hora=np.array(por_tormenta.iso(hora) or ""))
    except Exception:
        pass


def satelite_ahora(temp_dir: Path, ahora: datetime | None = None) -> dict:
    """Punto 5b: la imagen MÁS RECIENTE del canal 13 (no la de la hora del
    radar) y la anterior, por localidad. Con el radar atrasado o faltante, el
    satélite sigue diciendo qué pasa ahora sobre cada localidad."""
    from datetime import timedelta

    from .topes_localidad import con_enfriamiento, estadisticas

    ahora = ahora or datetime.now(timezone.utc)
    locs = []
    try:
        locs = json.loads((BASE_DIR / "config" / "localidades.json").read_text(encoding="utf-8"))["localidades"]
    except Exception:
        pass
    meta = latest_product_object("abi_c13", now=ahora, hours_back=1)
    if meta is None:
        return {"estado": "sin_imagen"}
    dominio, lectura = leer_dominio(meta, ["CMI"], temp_dir, "abi_c13_ahora")
    actual = estadisticas(dominio["datos"]["CMI"], dominio["latitude"], dominio["longitude"], locs)
    hora = _hora_satelite(meta, dominio)
    guardar_c13_ahora(dominio, hora)
    salida = {"estado": "ok", "hora_satelite_utc": por_tormenta.iso(hora),
              "inicio_utc": meta.get("start_time_utc"), "lectura": lectura,
              "edad_min": round((ahora - hora).total_seconds() / 60.0, 1) if hora else None}
    previo = {}
    minutos = None
    inicio = por_tormenta.parse_utc(meta.get("start_time_utc"))
    if inicio is not None:
        meta_p = product_object_for_scan("abi_c13", por_tormenta.iso(inicio - timedelta(minutes=10)),
                                         now=ahora, tolerance_seconds=60.0)
        if meta_p is not None:
            dom_p, _ = leer_dominio(meta_p, ["CMI"], temp_dir, "abi_c13_ahora_previa")
            previo = estadisticas(dom_p["datos"]["CMI"], dom_p["latitude"], dom_p["longitude"], locs)
            hora_p = _hora_satelite(meta_p, dom_p)
            minutos = (hora - hora_p).total_seconds() / 60.0 if hora and hora_p else None
    salida["por_localidad"] = con_enfriamiento(actual, previo, minutos)
    return salida


def _cargar_json(path: Path) -> dict | None:
    try:
        valor = json.loads(path.read_text(encoding="utf-8"))
        return valor if isinstance(valor, dict) else None
    except Exception:
        return None


def build_state(reference_time: datetime | None = None) -> dict:
    if reference_time is None:
        reference_time = load_radar_reference_time()

    estado_previo = _cargar_json(OUTPUT_FILE)

    state = {
        "version": "1.8",
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
    campos: dict[str, dict] = {}
    rayos_radar = None

    with tempfile.TemporaryDirectory(prefix="climaar_goes19_") as temp:
        temp_dir = Path(temp)

        # El canal 13 fija el barrido de referencia: los demás productos se
        # toman del MISMO barrido para no mezclar momentos distintos.
        inicio_escaneo = None
        orden = ["abi_c13"] + [p for p in PRODUCTS if p != "abi_c13"]
        for product in orden:
            try:
                state["products"][product] = process_product(
                    product, PRODUCTS[product], temp_dir, reference_time,
                    inicio_escaneo=inicio_escaneo, campos=campos,
                )
            except Exception as exc:
                state["products"][product] = {
                    "status": "unavailable",
                    "error": str(exc),
                }
            if product == "abi_c13":
                inicio_escaneo = (state["products"][product].get("source") or {}).get("start_time_utc")
        # Mismo orden de claves que antes en el archivo.
        state["products"] = {k: state["products"][k] for k in PRODUCTS}

        for product, variable in CLOUDTOP_PRODUCTS.items():
            try:
                state["cloudtop"][product] = process_cloudtop(
                    product, variable, temp_dir, reference_time,
                    inicio_escaneo=inicio_escaneo, campos=campos,
                )
            except Exception as exc:
                state["cloudtop"][product] = {
                    "status": "unavailable",
                    "error": str(exc),
                }

        senales_productos = {}
        for product in SENALES_PRODUCTS:
            try:
                senales_productos[product] = process_senal(
                    product, temp_dir, reference_time, inicio_escaneo, campos)
            except Exception as exc:
                senales_productos[product] = {"status": "unavailable", "error": str(exc)[:300]}

        try:
            state["glm"] = process_glm(temp_dir, reference_time)
        except Exception as exc:
            state["glm"] = {"status": "unavailable", "error": str(exc)}
        # Rayos: últimos 20 min por localidad (un archivo GLM = 20 s) y, en la
        # misma descarga, los 10 min que terminan en la imagen de radar.
        try:
            from .glm_ventana import ventana_y_rayos_radar

            state["glm"]["ventana"], rayos_radar = ventana_y_rayos_radar(
                centro=(RADAR_LAT, RADAR_LON), radio_km=RADIO_TORMENTAS_KM,
                frame_radar=reference_time,
            )
            if rayos_radar is not None:
                state["glm"]["rayos_frame_radar"] = {
                    "fin_utc": por_tormenta.iso(reference_time),
                    "ventana_min": por_tormenta.RAYOS_VENTANA_MIN,
                    "cobertura_min": round(float(rayos_radar["cobertura_min"]), 1),
                    "archivos": int(rayos_radar["archivos"]),
                    # Posiciones para el mapa de la app (vista Meteorólogos).
                    "posiciones": _rayos_para_mapa(rayos_radar),
                }
        except Exception as exc:
            state["glm"]["ventana"] = {"disponible": False, "motivo": f"ERROR: {exc}"}

        try:
            state["dmw"] = process_dmw(temp_dir, reference_time)
        except Exception as exc:
            state["dmw"] = {"status": "unavailable", "error": str(exc)}

        # Punto 5b: la imagen más reciente del satélite, para cubrir el radar
        # atrasado o faltante.
        try:
            state["satelite_ahora"] = satelite_ahora(temp_dir)
        except Exception as exc:
            state["satelite_ahora"] = {"estado": "error", "error": f"{type(exc).__name__}: {exc}"[:200]}

        # Punto 5: canal 13 de 10 y 20 min antes (mismo recorte), para ver
        # nubes que se están convirtiendo en tormenta.
        previos_c13, lectura_previos = leer_c13_previos(temp_dir, reference_time, inicio_escaneo)

    # Nubes en formación (estadística pura, sin red).
    try:
        from . import formacion

        state["formacion_satelite"] = formacion.detectar(
            campos.get("abi_c13"), previos_c13.get(10), previos_c13.get(20),
            {
                "fase": campos.get("cloud_phase"),
                "altura_topes_2km": campos.get("cloudtop_height_2km"),
                "altura_topes": campos.get("cloudtop_height"),
                "c08": campos.get("abi_c08"),
            },
            rayos_radar,
            reference_time,
        )
    except Exception as exc:
        state["formacion_satelite"] = {"estado": "error", "error": f"{type(exc).__name__}: {exc}"[:300],
                                       "objetos": []}
    state["formacion_satelite"]["lectura_previas"] = lectura_previos
    state["formacion_satelite"]["frame_radar_utc"] = por_tormenta.iso(reference_time)

    # Datos por tormenta (estadística pura, sin red).
    try:
        state["por_tormenta"] = por_tormenta.calcular(
            _cargar_json(RADAR_NOWCAST_FILE),
            reference_time,
            {
                "c13": campos.get("abi_c13"),
                "c08": campos.get("abi_c08"),
                "altura_topes": campos.get("cloudtop_height"),
                "presion_topes": campos.get("cloudtop_pressure"),
                "altura_topes_2km": campos.get("cloudtop_height_2km"),
                "fase": campos.get("cloud_phase"),
            },
            rayos_radar,
            estado_previo,
            _cargar_json(METEO_FILE),
        )
    except Exception as exc:
        state["por_tormenta"] = {
            "estado": "error",
            "error": f"{type(exc).__name__}: {exc}"[:300],
            "frame_radar_utc": por_tormenta.iso(reference_time),
            "tormentas": [],
        }

    state["senales_severas"] = {
        "productos": senales_productos,
        "topes_sobresalientes_zona": (state.get("por_tormenta") or {}).get("topes_sobresalientes_zona"),
        "metodo": {
            "topes_sobresalientes": "textura del infrarrojo (Bedka 2010): <= -58 °C, mínimo local a 4 km, "
                                    "yunque frío a 6-10 km, >= 6,5 °C más frío que el yunque",
            "vapor_menos_infrarrojo": "canal 8 - canal 13, mismo barrido, píxel a píxel",
            "congelamiento": "altura del tope (p90) menos nivel de 0 °C del modelo (punto y hora más cercanos)",
        },
    }

    state["sincronizacion"] = {
        "frame_radar_utc": por_tormenta.iso(reference_time),
        "inicio_barrido_referencia_utc": inicio_escaneo,
        "desfase_max_min": por_tormenta.DESFASE_MAX_MIN,
        "productos": {
            nombre: (item or {}).get("sincronizacion")
            for nombre, item in {**state["products"], **state["cloudtop"], **senales_productos}.items()
        },
    }

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

    # Regla 7 del plan: caso guardado (datos por tormenta de esta imagen).
    from .historial import guardar

    print(f"Caso guardado: {guardar(state)}")

    print("ClimaAR GOES-19 State Engine")
    print(f"Frame RMA10: {reference_time.isoformat()}")
    print(f"Estado: {state['status']}")

    for name, value in state["products"].items():
        print(f"{name}: {value.get('status')}")

    for name, value in state["cloudtop"].items():
        print(f"{name}: {value.get('status')}")

    for name, value in {**state["products"], **state["cloudtop"],
                        **(state.get("senales_severas") or {}).get("productos", {})}.items():
        lectura = value.get("lectura") or {}
        sinc = value.get("sincronizacion") or {}
        if lectura:
            print(
                f"  {name}: lectura {lectura.get('modo')} {lectura.get('segundos')} s, "
                f"{sinc.get('estado')} (desfase {sinc.get('desfase_min')} min, "
                f"mismo barrido: {sinc.get('mismo_escaneo_que_c13')})"
            )
    pt = state.get("por_tormenta") or {}
    print(f"Por tormenta: {pt.get('estado')} - {len(pt.get('tormentas') or [])} tormentas - "
          f"topes sobresalientes en la zona: {pt.get('topes_sobresalientes_zona')}")

    fs = state.get("formacion_satelite") or {}
    print(f"Nubes en formación (punto 5): {fs.get('estado')} - {len(fs.get('objetos') or [])} objetos - "
          f"imágenes previas: {fs.get('lectura_previas')}")

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
