#!/usr/bin/env python3
"""Train the first ClimaAR radar nowcasting model from captured RMA10 features."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

CSV = Path("data/radar/radar_features.csv")
MODEL = Path("modelo/climaar_nowcast_features.joblib")
METRICS = Path("modelo/metricas_nowcast.json")

BASE = [
    "max_dbz",
    "mean_dbz",
    "pixels_ge_20dbz",
    "pixels_ge_30dbz",
    "pixels_ge_40dbz",
    "pixels_ge_45dbz",
    "pixels_ge_50dbz",
    "pixels_ge_55dbz",
    "pixels_ge_60dbz",
    "centroid_x_ge_40dbz",
    "centroid_y_ge_40dbz",
]

TARGETS = ["max_dbz", "pixels_ge_40dbz"]


def main():
    if not CSV.exists():
        raise SystemExit("No existe el histórico RMA10.")

    df = pd.read_csv(CSV)
    if len(df) < 50:
        raise SystemExit(
            f"Hay {len(df)} frames. Se necesitan al menos 50 para entrenar el primer modelo."
        )

    df["frame_time"] = pd.to_numeric(df["frame_time"], errors="coerce")
    df = df.sort_values("frame_time").drop_duplicates("frame_time").reset_index(drop=True)

    for col in BASE + TARGETS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing centroid values with the previous valid value and then zero.
    df[BASE] = df[BASE].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    rows = []
    for i in range(3, len(df) - 1):
        current = df.iloc[i]
        future = df.iloc[i + 1]

        dt = float(future["frame_time"] - current["frame_time"])
        if dt < 5 * 60 or dt > 20 * 60:
            continue

        values = []
        for j in range(i - 3, i + 1):
            values.extend(df.iloc[j][BASE].astype(float).tolist())

        target_max = float(future["max_dbz"])
        target_area = float(future["pixels_ge_40dbz"])
        rows.append(values + [target_max, target_area])

    if len(rows) < 30:
        raise SystemExit(
            f"Solo se pudieron formar {len(rows)} secuencias válidas; faltan datos temporales."
        )

    columns = [f"t{offset}_{name}" for offset in [-30, -20, -10, 0] for name in BASE]
    columns += ["target_max_dbz", "target_pixels_ge_40dbz"]
    data = pd.DataFrame(rows, columns=columns)

    feature_cols = columns[:-2]
    target_cols = columns[-2:]

    split = int(len(data) * 0.80)
    train, test = data.iloc[:split], data.iloc[split:]

    models = {}
    metrics = {
        "modelo": "HistGradientBoostingRegressor",
        "entrada": "4 frames RMA10 de 10 minutos",
        "salida": ["max_dbz_t_plus_10", "pixels_ge_40dbz_t_plus_10"],
        "secuencias": int(len(data)),
        "entrenamiento": int(len(train)),
        "prueba": int(len(test)),
    }

    for target in target_cols:
        model = HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )
        model.fit(train[feature_cols], train[target])
        pred = model.predict(test[feature_cols])

        mae = mean_absolute_error(test[target], pred)
        rmse = mean_squared_error(test[target], pred) ** 0.5
        metrics[target] = {"MAE": float(mae), "RMSE": float(rmse)}
        models[target] = model

    artifact = {
        "models": models,
        "features": feature_cols,
        "targets": target_cols,
        "interval_minutes": 10,
    }

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL)
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print("MODELO NOWCAST ENTRENADO")


if __name__ == "__main__":
    main()
