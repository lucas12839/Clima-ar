#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
BASE = Path("/opt/Clima-ar")
FUTURE_FILE = BASE / "data/meteorologia/ambiente_futuro.json"
OUTPUT_FILE = BASE / "data/ia/atmosphere_nwp_embedding.pt"

SCALAR_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "pressure_msl", "surface_pressure", "precipitation",
    "rain", "showers", "wind_gusts_10m", "cape",
    "vapour_pressure_deficit",
    "total_column_integrated_water_vapour",
]

LEVELS = (10, 100, 200)
def load_future():
    with FUTURE_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def wind_uv(speed, direction):
    rad = np.deg2rad(direction)
    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)
    return u, v


def get_value(point, name, default=0.0):
    value = point.get(name, default)
    if value is None:
        return default
    return float(value)


def build_features(data):
    point = data["puntos"][0]
    hourly = point["datos"]

    rows = []

    for item in hourly:
        row = []

        for name in SCALAR_VARS:
            row.append(get_value(item, name))

        for level in LEVELS:
            speed = get_value(item, f"wind_speed_{level}m")
            direction = get_value(item, f"wind_direction_{level}m")
            u, v = wind_uv(speed, direction)
            row.extend([u, v])

        rows.append(row)

    return np.asarray(rows, dtype=np.float32)

class AtmosphereNWPEncoder(nn.Module):
    def __init__(self, input_size=18, hidden_size=128):
        super().__init__()

        self.input = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.10,
        )

        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=8,
            batch_first=True,
        )

        self.trend = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.input(x)
        temporal, _ = self.gru(x)

        attended, _ = self.attention(
            temporal,
            temporal,
            temporal,
        )

        context = attended[:, -1, :]
        trend = self.trend(
            attended[:, -1, :] - attended[:, 0, :]
        )

        return {
            "atmospheric_context": context,
            "temporal_features": attended,
            "trend_features": trend,
        }

def main():
    data = load_future()
    features = build_features(data)

    x = torch.from_numpy(features).unsqueeze(0)

    model = AtmosphereNWPEncoder(
        input_size=features.shape[1],
        hidden_size=128,
    )

    model.eval()

    with torch.no_grad():
        result = model(x)

    output = {
        "model": "climaar-atmosphere-nwp-v1",
        "input_shape": tuple(x.shape),
        "atmospheric_context": result["atmospheric_context"],
        "temporal_features": result["temporal_features"],
        "trend_features": result["trend_features"],
    }

    torch.save(output, OUTPUT_FILE)

    print("ATMOSPHERE NWP ENCODER OK")
    print("input:", tuple(x.shape))
    print("context:", tuple(result["atmospheric_context"].shape))
    print("temporal:", tuple(result["temporal_features"].shape))
    print("trend:", tuple(result["trend_features"].shape))
    print("output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
