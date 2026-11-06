
"""Encoder espacial independiente para la red de estaciones ClimaAR.

Estado: arquitectura inicial, pesos sin entrenar.
No produce predicciones meteorológicas.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


BASE = Path(__file__).resolve().parents[2]
CATALOG = BASE / "data/ia/estaciones/catalogo_estaciones.json"
BCP_FILE = BASE / "data/ia/estaciones/observaciones/bcp_actual.json"
SMN_FILE = BASE / "data/ia/estaciones/observaciones/smn_synop_actual.json"

BAHIA_LAT = -38.7196
BAHIA_LON = -62.2663
MAX_AGE_MINUTES = 15.0

# 17 variables meteorológicas, en orden fijo.
VARIABLES = (
    "temperature", "dewpoint", "humidity", "pressure",
    "pressure_change", "precipitation", "wind_speed",
    "wind_direction_sin", "wind_direction_cos", "wind_gust",
    "radiation", "soil_temperature", "visibility_km",
    "cloud_cover", "cloud_base_km", "maximum_temperature",
    "minimum_temperature",
)

# 17 valores + 17 máscaras + 5 datos espaciales/temporales.
INPUT_SIZE = 39


def _read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _first_smn(variables, key):
    entries = variables.get(key, [])
    if not isinstance(entries, list):
        return None
    for item in entries:
        if isinstance(item, dict):
            value = _number(item.get("value"))
            if value is not None:
                return value
    return None


def _direction(value):
    if value is None:
        return None
    degrees = _number(value)
    if degrees is None:
        compass = {
            "N": 0, "NORTE": 0,
            "NE": 45, "NORESTE": 45,
            "E": 90, "ESTE": 90,
            "SE": 135, "SUDESTE": 135,
            "S": 180, "SUR": 180,
            "SO": 225, "SUDOESTE": 225,
            "O": 270, "OESTE": 270,
            "NO": 315, "NOROESTE": 315,
        }
        degrees = compass.get(str(value).strip().upper())
    if degrees is None or not 0 <= degrees <= 360:
        return None
    radians = math.radians(degrees)
    return math.sin(radians), math.cos(radians)


def _distance_and_bearing(lat, lon):
    lat1, lat2 = math.radians(BAHIA_LAT), math.radians(lat)
    dlon = math.radians(lon - BAHIA_LON)
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    distance = 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.atan2(y, x)
    return distance, math.sin(bearing), math.cos(bearing)


def _normalize(values):
    scales = (50, 50, 100, 1100, 10, 100, 60, 1, 1, 60,
              1200, 50, 20, 100, 15, 50, 50)
    result = []
    masks = []
    for value, scale in zip(values, scales):
        valid = value is not None and math.isfinite(value)
        masks.append(1.0 if valid else 0.0)
        result.append(float(np.clip(value / scale, -5, 5)) if valid else 0.0)
    return result, masks


def _bcp_values(record):
    v = record.get("variables", {})
    direction = _direction(v.get("direccion_viento"))
    return [
        _number(v.get("temperatura_c")),
        None,  # BCP no aporta punto de rocío en este colector.
        _number(v.get("humedad_relativa_pct")),
        _number(v.get("presion_hpa")),
        None,
        _number(v.get("precipitacion_dia_mm")),
        (x / 3.6 if (x := _number(v.get("viento_kmh"))) is not None else None),
        direction[0] if direction else None,
        direction[1] if direction else None,
        (x / 3.6 if (x := _number(v.get("rafaga_15min_kmh"))) is not None else None),
        _number(v.get("radiacion_wm2")),
        _number(v.get("temp_suelo_5cm_c")),
        None, None, None, None, None,
    ]


def _smn_values(record):
    v = record.get("variables", {})
    direction = _direction(_first_smn(v, "wind_direction"))
    visibility = _first_smn(v, "horizontal_visibility")
    cloud_base = _first_smn(v, "height_of_base_of_cloud")
    return [
        _first_smn(v, "air_temperature"),
        _first_smn(v, "dewpoint_temperature"),
        None,
        _first_smn(v, "pressure_reduced_to_mean_sea_level"),
        _first_smn(v, "3hour_pressure_change"),
        _first_smn(v, "total_precipitation_or_total_water_equivalent"),
        _first_smn(v, "wind_speed"),
        direction[0] if direction else None,
        direction[1] if direction else None,
        None,
        None,
        None,
        visibility / 1000 if visibility is not None else None,
        _first_smn(v, "cloud_cover_total"),
        cloud_base / 1000 if cloud_base is not None else None,
        _first_smn(v, "maximum_temperature_at_height_and_over_period_specified"),
        _first_smn(v, "minimum_temperature_at_height_and_over_period_specified"),
    ]


def build_observation_batch(
    catalog_path=CATALOG,
    bcp_path=BCP_FILE,
    smn_path=SMN_FILE,
):
    """Construye tensores para las estaciones del catálogo.

    La frescura se determina con los campos calculados por los colectores:
    fresh, estado y edad_minutos. Una estación vencida nunca aporta valores.
    """
    catalog = _read_json(Path(catalog_path))
    bcp = _read_json(Path(bcp_path))
    smn = _read_json(Path(smn_path))

    bcp_by_id = {str(x.get("station_id")): x for x in bcp.get("stations", [])}
    smn_by_id = {str(x.get("wigos")): x for x in smn.get("stations", [])}

    rows, station_mask, ages = [], [], []

    for station in catalog.get("stations", []):
        record = None
        if station.get("network") == "bcp":
            record = bcp_by_id.get(str(station.get("station_number")))
        elif station.get("network") == "smn":
            record = smn_by_id.get(str(station.get("wigos")))

        age = _number(record.get("edad_minutos")) if record else None
        valid = bool(
            station.get("active")
            and record
            and record.get("fresh") is True
            and record.get("estado") == "valida"
            and age is not None
            and 0 <= age < MAX_AGE_MINUTES
        )

        values = (
            _bcp_values(record) if valid and station.get("network") == "bcp"
            else _smn_values(record) if valid
            else [None] * len(VARIABLES)
        )
        normalized, masks = _normalize(values)

        distance, bearing_sin, bearing_cos = _distance_and_bearing(
            float(station["latitude"]), float(station["longitude"])
        )
        altitude = _number(station.get("altitude_m"))

        metadata = [
            min(distance / 250.0, 2.0),
            bearing_sin,
            bearing_cos,
            min(max(altitude or 0.0, 0.0) / 3000.0, 2.0),
            min(age / MAX_AGE_MINUTES, 1.0) if valid else 1.0,
        ]

        rows.append(normalized + masks + metadata)
        station_mask.append(1.0 if valid else 0.0)
        ages.append(age if valid else None)

    if not rows:
        raise ValueError("El catálogo no contiene estaciones.")

    return {
        "features": torch.tensor([rows], dtype=torch.float32),
        "station_mask": torch.tensor([station_mask], dtype=torch.bool),
        "ages_minutes": ages,
        "station_names": [s["name"] for s in catalog["stations"]],
        "station_count": len(rows),
        "valid_count": int(sum(station_mask)),
    }


class ObservationNetworkEncoder(nn.Module):
    """Encoder espacial de estaciones. Pesos iniciales sin entrenamiento."""

    def __init__(self, input_size=INPUT_SIZE, embedding_size=128, heads=4):
        super().__init__()
        if embedding_size % heads:
            raise ValueError("embedding_size debe ser divisible por heads.")

        self.station_projection = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, embedding_size),
            nn.GELU(),
        )
        self.spatial_attention = nn.MultiheadAttention(
            embedding_size, heads, batch_first=True
        )
        self.norm = nn.LayerNorm(embedding_size)
        self.output = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.GELU(),
        )

    def forward(self, features, station_mask):
        if features.ndim != 3 or features.shape[-1] != INPUT_SIZE:
            raise ValueError(f"features debe tener forma [B,N,{INPUT_SIZE}].")
        if station_mask.shape != features.shape[:2]:
            raise ValueError("station_mask debe tener forma [B,N].")

        tokens = self.station_projection(features)
        outputs = torch.zeros_like(tokens)
        contexts = torch.zeros(
            features.shape[0], tokens.shape[-1],
            dtype=tokens.dtype, device=tokens.device
        )

        for batch_index in range(features.shape[0]):
            valid = station_mask[batch_index]
            if not bool(valid.any()):
                continue
            selected = tokens[batch_index:batch_index + 1, valid, :]
            attended, _ = self.spatial_attention(selected, selected, selected)
            encoded = self.norm(selected + attended)
            outputs[batch_index, valid] = encoded[0]
            contexts[batch_index] = self.output(encoded.mean(dim=1))[0]

        quality = station_mask.float().mean(dim=1, keepdim=True)
        return {
            "observation_context": contexts,
            "station_features": outputs,
            "station_mask": station_mask,
            "network_quality": quality,
            "training_status": "untrained",
        }


def load_and_encode(encoder, **paths):
    batch = build_observation_batch(**paths)
    result = encoder(batch["features"], batch["station_mask"])
    result["station_names"] = batch["station_names"]
    result["valid_count"] = batch["valid_count"]
    result["ages_minutes"] = batch["ages_minutes"]
    return result
