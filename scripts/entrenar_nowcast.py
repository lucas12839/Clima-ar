#!/usr/bin/env python3

"""
ClimaAR - Entrenamiento del modelo de nowcasting radar.

Fuente:
    data/radar/radar_features_rainviewer.csv

Utiliza secuencias temporales de radar para predecir
la evolución del siguiente frame.

Targets:
    - dbz_max
    - area_px

Validación:
    Temporal holdout 80/20

IMPORTANTE:
    No se mezclan datos futuros con el entrenamiento.
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
# VARIABLES DEL RADAR V7.1
# ============================================================

FEATURES = [
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
# CONVERSIÓN DE NÚCLEOS
# ============================================================

def parse_nucleos(value):

    if pd.isna(value):
        return 0.0

    text = str(value).strip()

    if text in ("", "[]", "nan", "None"):
        return 0.0

    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            return float(len(parsed))

    except Exception:
        pass

    return 0.0


# ============================================================
# PREPARAR DATASET
# ============================================================

def prepare_dataframe(df):

    print("Columnas encontradas:")

    for column in df.columns:
        print(f"  - {column}")

    print("")

    # Compatibilidad con posibles nombres del radar.
    aliases = {
        "area_px": "area",
        "nucleos": "components",
    }

    for target, source in aliases.items():

        if target not in df.columns and source in df.columns:

            df[target] = df[source]

    # Crear columnas faltantes de forma segura.
    for column in FEATURES:

        if column not in df.columns:

            print(
                f"Aviso: falta {column}. "
                "Se rellenará con 0."
            )

            df[column] = 0.0

    # Timestamp.
    if "frame_time" in df.columns:

        df["frame_time"] = pd.to_numeric(
            df["frame_time"],
            errors="coerce"
        )

    elif "timestamp" in df.columns:

        df["frame_time"] = pd.to_numeric(
            df["timestamp"],
            errors="coerce"
        )

    else:

        raise SystemExit(
            "ERROR: el dataset no contiene "
            "frame_time ni timestamp."
        )

    # Conversión numérica.
    numeric_columns = [
        column
        for column in FEATURES
        if column != "nucleos"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Núcleos.
    if "nucleos" in df.columns:

        df["nucleos"] = df["nucleos"].apply(
            parse_nucleos
        )

    # Limpiar infinitos.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Orden temporal.
    df = df.sort_values(
        "frame_time"
    )

    # Eliminar frames duplicados.
    df = df.drop_duplicates(
        subset=["frame_time"]
    )

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# CONSTRUIR SECUENCIAS
# ============================================================

def build_sequences(df):

    rows = []

    # Utilizamos cuatro observaciones:
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

        delta_minutes = (
            future_time
            -
            current_time
        ) / 60.0

        # Aceptamos aproximadamente un frame
        # cada 10 minutos.
        #
        # Permitimos retrasos normales del radar.

        if (
            delta_minutes < 5
            or
            delta_minutes > 20
        ):
            continue

        values = []

        valid = True

        for j in range(
            i - 3,
            i + 1
        ):

            frame = df.iloc[j]

            for feature in FEATURES:

                value = frame[feature]

                if pd.isna(value):

                    value = 0.0

                try:

                    value = float(value)

                except Exception:

                    valid = False
                    value = 0.0

                values.append(
                    value
                )

        if not valid:
            continue

        target_dbz = future["dbz_max"]

        target_area = future["area_px"]

        if pd.isna(target_dbz):
            target_dbz = 0.0

        if pd.isna(target_area):
            target_area = 0.0

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

        for feature in FEATURES:

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
# ENTRENAR Y MEDIR
# ============================================================

def train_target(
    X_train,
    X_test,
    y_train,
    y_test
):

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

    prediction = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )

    return (
        model,
        {
            "MAE": float(mae),
            "RMSE": float(rmse),
        }
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("")
    print("==============================================")
    print("CLIMAAR - ENTRENAMIENTO NOWCAST")
    print("==============================================")
    print("")

    # --------------------------------------------------------
    # COMPROBAR CSV
    # --------------------------------------------------------

    if not CSV.exists():

        raise SystemExit(
            f"ERROR: no existe {CSV}"
        )

    print(
        f"Archivo: {CSV}"
    )

    df = pd.read_csv(
        CSV
    )

    print(
        f"Filas originales: {len(df)}"
    )

    if len(df) < 50:

        raise SystemExit(
            f"ERROR: solamente hay {len(df)} frames. "
            "Se necesitan al menos 50."
        )

    print("")

    # --------------------------------------------------------
    # PREPARAR
    # --------------------------------------------------------

    df = prepare_dataframe(
        df
    )

    print(
        f"Frames válidos: {len(df)}"
    )

    print("")

    # --------------------------------------------------------
    # SECUENCIAS
    # --------------------------------------------------------

    sequences = build_sequences(
        df
    )

    if sequences.empty:

        raise SystemExit(
            "ERROR: no se pudieron construir "
            "secuencias temporales válidas."
        )

    print(
        f"Secuencias construidas: {len(sequences)}"
    )

    if len(sequences) < 30:

        raise SystemExit(
            f"ERROR: solamente hay "
            f"{len(sequences)} secuencias. "
            "Se necesitan al menos 30."
        )

    print("")

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    target_columns = [
        "target_dbz_max",
        "target_area_px",
    ]

    feature_columns = [
        column
        for column in sequences.columns
        if column not in target_columns
    ]

    X = sequences[
        feature_columns
    ]

    # --------------------------------------------------------
    # VALIDACIÓN TEMPORAL
    # --------------------------------------------------------

    split = int(
        len(sequences) * 0.80
    )

    if split <= 0 or split >= len(sequences):

        raise SystemExit(
            "ERROR: división temporal inválida."
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

    print("")

    # --------------------------------------------------------
    # MODELOS
    # --------------------------------------------------------

    models = {}

    metrics = {

        "version": "2.0",

        "modelo":
            "HistGradientBoostingRegressor",

        "fuente":
            "RainViewer V7.1",

        "archivo_entrada":
            str(CSV),

        "horizonte_minutos":
            10,

        "frames_por_secuencia":
            4,

        "secuencias":
            int(len(sequences)),

        "entrenamiento":
            int(len(X_train)),

        "prueba":
            int(len(X_test)),

        "validacion": {

            "tipo":
                "temporal_holdout",

            "porcentaje_entrenamiento":
                0.80,

            "porcentaje_prueba":
                0.20,
        },

        "targets": target_columns,
    }

    # --------------------------------------------------------
    # DBZ
    # --------------------------------------------------------

    print(
        "Entrenando predicción de dbz_max..."
    )

    y_dbz = sequences[
        "target_dbz_max"
    ]

    y_train_dbz = y_dbz.iloc[
        :split
    ]

    y_test_dbz = y_dbz.iloc[
        split:
    ]

    (
        model_dbz,
        metrics_dbz
    ) = train_target(
        X_train,
        X_test,
        y_train_dbz,
        y_test_dbz
    )

    models[
        "target_dbz_max"
    ] = model_dbz

    metrics[
        "target_dbz_max"
    ] = metrics_dbz

    print(
        f"DBZ MAE: {metrics_dbz['MAE']:.4f}"
    )

    print(
        f"DBZ RMSE: {metrics_dbz['RMSE']:.4f}"
    )

    print("")

    # --------------------------------------------------------
    # AREA
    # --------------------------------------------------------

    print(
        "Entrenando predicción de area_px..."
    )

    y_area = sequences[
        "target_area_px"
    ]

    y_train_area = y_area.iloc[
        :split
    ]

    y_test_area = y_area.iloc[
        split:
    ]

    (
        model_area,
        metrics_area
    ) = train_target(
        X_train,
        X_test,
        y_train_area,
        y_test_area
    )

    models[
        "target_area_px"
    ] = model_area

    metrics[
        "target_area_px"
    ] = metrics_area

    print(
        f"AREA MAE: {metrics_area['MAE']:.4f}"
    )

    print(
        f"AREA RMSE: {metrics_area['RMSE']:.4f}"
    )

    print("")

    # --------------------------------------------------------
    # GUARDAR MODELO
    # --------------------------------------------------------

    MODEL.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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

        "horizon_minutes":
            10,

        "frames":
            4,

        "validation": {

            "type":
                "temporal_holdout",

            "train_fraction":
                0.80,

            "test_fraction":
                0.20,
        },
    }

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

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    print("")
    print("==============================================")
    print("ENTRENAMIENTO COMPLETADO")
    print("==============================================")
    print("")
    print(
        f"Modelo: {MODEL}"
    )
    print(
        f"Métricas: {METRICS}"
    )
    print("")
    print(
        json.dumps(
            metrics,
            indent=2
        )
    )
    print("")


if __name__ == "__main__":

    main()
