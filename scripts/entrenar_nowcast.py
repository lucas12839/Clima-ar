#!/usr/bin/env python3

"""
ClimaAR - Entrenamiento del modelo de nowcasting radar.

Usa el histórico real generado por RainViewer V7.1:
data/radar/radar_features_rainviewer.csv

Predice:
- dbz_max del próximo frame
- area_px del próximo frame

La validación es temporal:
80% entrenamiento
20% prueba

No mezcla datos futuros con datos de entrenamiento.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# ARCHIVOS
# ============================================================

CSV = Path(
    "data/radar/radar_features_rainviewer.csv"
)

MODEL = Path(
    "modelo/climaar_nowcast_features.joblib"
)

METRICS = Path(
    "modelo/metricas_nowcast.json"
)


# ============================================================
# VARIABLES DISPONIBLES EN RADAR V7.1
# ============================================================

BASE_FEATURES = [
    "area_px",
    "distance_km",
    "dbz_max",
    "dbz_mean",
    "dbz_p90",
    "dbz_pixels",
    "tiles_ok",
    "nucleos",
]


# ============================================================
# LIMPIEZA
# ============================================================

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    required = [
        "frame_time",
        *BASE_FEATURES,
    ]

    for column in required:

        if column not in df.columns:

            if column == "nucleos":
                df[column] = 0

            else:
                df[column] = np.nan

    df["frame_time"] = pd.to_numeric(
        df["frame_time"],
        errors="coerce"
    )

    for column in BASE_FEATURES:

        if column == "nucleos":

            # La columna actual puede contener una lista
            # como [] o una lista de núcleos.
            def count_nucleos(value):

                if pd.isna(value):
                    return 0

                text = str(value).strip()

                if text in ("", "[]", "nan"):
                    return 0

                try:

                    parsed = json.loads(text)

                    if isinstance(parsed, list):
                        return len(parsed)

                except Exception:
                    pass

                return 0

            df[column] = df[column].apply(
                count_nucleos
            )

        else:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.sort_values(
        "frame_time"
    )

    df = df.drop_duplicates(
        "frame_time"
    )

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# CONSTRUCCIÓN DE SECUENCIAS
# ============================================================

def build_sequences(
    df: pd.DataFrame
) -> pd.DataFrame:

    rows = []

    # Usamos los últimos 4 frames:
    #
    # t-30
    # t-20
    # t-10
    # t
    #
    # para predecir:
    #
    # t+10

    for i in range(
        3,
        len(df) - 1
    ):

        current = df.iloc[i]

        future = df.iloc[i + 1]

        current_time = float(
            current["frame_time"]
        )

        future_time = float(
            future["frame_time"]
        )

        delta_seconds = (
            future_time
            -
            current_time
        )

        # El radar debería actualizar aproximadamente
        # cada 10 minutos.
        #
        # Permitimos entre 5 y 20 minutos para tolerar
        # pequeños retrasos de RainViewer.

        if (
            delta_seconds < 5 * 60
            or
            delta_seconds > 20 * 60
        ):
            continue

        values = []

        valid_sequence = True

        for j in range(
            i - 3,
            i + 1
        ):

            frame = df.iloc[j]

            frame_values = []

            for feature in BASE_FEATURES:

                value = frame[feature]

                if pd.isna(value):

                    value = 0.0

                frame_values.append(
                    float(value)
                )

            values.extend(
                frame_values
            )

        target_dbz = future["dbz_max"]

        target_area = future["area_px"]

        if pd.isna(target_dbz):
            target_dbz = 0.0

        if pd.isna(target_area):
            target_area = 0.0

        if not valid_sequence:
            continue

        rows.append(
            values
            +
            [
                float(target_dbz),
                float(target_area),
            ]
        )

    if not rows:

        return pd.DataFrame()

    columns = []

    offsets = [
        -30,
        -20,
        -10,
        0,
    ]

    for offset in offsets:

        for feature in BASE_FEATURES:

            columns.append(
                f"t{offset}_{feature}"
            )

    columns.extend(
        [
            "target_dbz_max",
            "target_area_px",
        ]
    )

    return pd.DataFrame(
        rows,
        columns=columns
    )


# ============================================================
# MÉTRICAS
# ============================================================

def calculate_metrics(
    model,
    X_test,
    y_test
):

    prediction = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    rmse = mean_squared_error(
        y_test,
        prediction
    ) ** 0.5

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "============================================"
    )

    print(
        "CLIMAAR - ENTRENAMIENTO NOWCAST V7.1"
    )

    print(
        "============================================"
    )

    if not CSV.exists():

        raise SystemExit(
            f"No existe el archivo: {CSV}"
        )

    print(
        f"Archivo de entrada: {CSV}"
    )

    df = pd.read_csv(
        CSV
    )

    print(
        f"Filas originales: {len(df)}"
    )

    if len(df) < 50:

        raise SystemExit(
            f"
Solo hay {len(df)} frames.

Se necesitan al menos 50 frames
para intentar entrenar el modelo.
"
        )

    df = prepare_dataframe(
        df
    )

    print(
        f"Frames después de limpieza: {len(df)}"
    )

    sequences = build_sequences(
        df
    )

    if len(sequences) < 30:

        raise SystemExit(
            f"
Solo se pudieron construir "
            f"{len(sequences)} secuencias válidas.

Se necesitan al menos 30.
"
        )

    print(
        f"Secuencias válidas: {len(sequences)}"
    )

    # ========================================================
    # FEATURES / TARGETS
    # ========================================================

    feature_columns = [
        column
        for column in sequences.columns
        if column not in (
            "target_dbz_max",
            "target_area_px",
        )
    ]

    target_columns = [
        "target_dbz_max",
        "target_area_px",
    ]

    X = sequences[
        feature_columns
    ]

    # ========================================================
    # VALIDACIÓN TEMPORAL
    # ========================================================

    split = int(
        len(sequences)
        *
        0.80
    )

    if split <= 0 or split >= len(sequences):

        raise SystemExit(
            "No se pudo construir una división temporal válida."
        )

    X_train = X.iloc[
        :split
    ]

    X_test = X.iloc[
        split:
    ]

    print(
        f"Entrenamiento: {len(X_train)}"
    )

    print(
        f"Prueba temporal: {len(X_test)}"
    )

    models = {}

    metrics = {

        "version": "2.0",

        "modelo":
            "HistGradientBoostingRegressor",

        "fuente":
            "RainViewer V7.1",

        "archivo_entrada":
            str(CSV),

        "entrada":
            "4 frames consecutivos de radar",

        "intervalo_minutos":
            10,

        "horizonte_minutos":
            10,

        "secuencias":
            int(len(sequences)),

        "entrenamiento":
            int(len(X_train)),

        "prueba":
            int(len(X_test)),

        "validacion":
            {
                "tipo":
                    "temporal_holdout",

                "porcentaje_entrenamiento":
                    0.80,

                "porcentaje_prueba":
                    0.20,
            },

        "targets":
            target_columns,
    }

    # ========================================================
    # ENTRENAMIENTO
    # ========================================================

    for target in target_columns:

        print(
            f"Entrenando objetivo: {target}"
        )

        y = sequences[
            target
        ]

        y_train = y.iloc[
            :split
        ]

        y_test = y.iloc[
            split:
        ]

        model = HistGradientBoostingRegressor(

            max_iter=250,

            learning_rate=0.05,

            max_leaf_nodes=31,

            l2_regularization=1.0,

            random_state=42,
        )

        model.fit(
            X_train,
            y_train
        )

        result = calculate_metrics(
            model,
            X_test,
            y_test
        )

        metrics[target] = result

        models[target] = model

        print(
            f"MAE: {result['MAE']:.4f}"
        )

        print(
            f"RMSE: {result['RMSE']:.4f}"
        )

    # ========================================================
    # GUARDAR MODELO
    # ========================================================

    artifact = {

        "version":
            "2.0",

        "source":
            "RainViewer V7.1",

        "models":
            models,

        "features":
            feature_columns,

        "targets":
            target_columns,

        "interval_minutes":
            10,

        "horizon_minutes":
            10,

        "validation":
            {
                "type":
                    "temporal_holdout",

                "train_fraction":
                    0.80,

                "test_fraction":
                    0.20,
            },
    }

    MODEL.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        artifact,
        MODEL
    )

    METRICS.write_text(
        json.dumps(
            metrics,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "============================================"
    )

    print(
        "MODELO NOWCAST ENTRENADO"
    )

    print(
        f"Modelo: {MODEL}"
    )

    print(
        f"Métricas: {METRICS}"
    )

    print(
        "============================================"
    )

    print(
        json.dumps(
            metrics,
            indent=2
        )
    )


if __name__ == "__main__":

    main()
