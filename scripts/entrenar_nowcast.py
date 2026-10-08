#!/usr/bin/env python3

"""
ClimaAR - Entrenamiento Nowcast Radar V3.

Mejoras:
- secuencias temporales realmente continuas;
- variables de estado + tendencias de los últimos 30 min;
- separación temporal con gap;
- rechazo de datasets completamente secos;
- métricas de error y dirección del cambio;
- no publica un modelo si no existe señal real.
"""

from __future__ import annotations

import json
import math
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

WEATHER_CSV = Path(
    "data/historico/clima_horario_1980_2026.csv"
)

MODEL = Path(
    "modelo/climaar_nowcast_features.joblib"
)

METRICS = Path(
    "modelo/metricas_nowcast.json"
)


# ============================================================
# VARIABLES BASE DEL RADAR
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


# Variables para calcular evolución
DELTA_FEATURES = [
    "area_px",
    "distance_km",
    "dbz_max",
    "dbz_mean",
    "dbz_p90",
    "dbz_pixels",
    "nucleos",
]

WEATHER_FEATURES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "pressure_msl", "surface_pressure", "precipitation", "rain",
    "cloud_cover", "wind_speed_10m", "wind_gusts_10m",
    "vapour_pressure_deficit", "boundary_layer_height",
]


# ============================================================
# REQUISITOS DEL DATASET
# ============================================================

MIN_FRAMES = 50

MIN_SEQUENCES = 30

MIN_PRECIP_FRAMES = 10

FRAME_MIN_MINUTES = 5

FRAME_MAX_MINUTES = 20

# Separación de seguridad entre entrenamiento y prueba
SEQUENCE_GAP = 4


# ============================================================
# CONVERTIR NÚCLEOS
# ============================================================

def parse_nucleos(value):

    if pd.isna(value):
        return 0.0

    text = str(value).strip()

    if text in (
        "",
        "[]",
        "nan",
        "None"
    ):
        return 0.0

    try:

        parsed = json.loads(text)

        if isinstance(parsed, list):

            return float(
                len(parsed)
            )

    except Exception:
        pass

    return 0.0


# ============================================================
# DATOS METEOROLÓGICOS HISTÓRICOS
# ============================================================

def load_weather_dataframe():
    if not WEATHER_CSV.exists():
        print(f"AVISO: no existe {WEATHER_CSV}. Se continuará solo con radar.")
        return None
    weather = pd.read_csv(WEATHER_CSV)
    time_column = next((c for c in ("time", "timestamp", "datetime", "date") if c in weather.columns), None)
    if time_column is None:
        print("AVISO: el histórico meteorológico no tiene columna temporal reconocible.")
        return None
    parsed = pd.to_datetime(weather[time_column], errors="coerce", utc=True)
    if parsed.isna().all():
        numeric = pd.to_numeric(weather[time_column], errors="coerce")
        parsed = pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)
    weather["_weather_time"] = parsed
    for column in WEATHER_FEATURES:
        if column not in weather.columns:
            weather[column] = np.nan
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
    return weather.dropna(subset=["_weather_time"]).sort_values("_weather_time")[
        ["_weather_time"] + WEATHER_FEATURES
    ].drop_duplicates("_weather_time")

def merge_weather_with_radar(df):
    weather = load_weather_dataframe()
    if weather is None or weather.empty:
        df["weather_match"] = 0.0
        for c in WEATHER_FEATURES:
            df[f"weather_{c}"] = np.nan
        return df
    radar = df.copy()
    radar["_radar_time"] = pd.to_datetime(
        pd.to_numeric(radar["frame_time"], errors="coerce"),
        unit="s", errors="coerce", utc=True
    )

    # Pandas puede cargar el CSV meteorológico con datetime64[us, UTC]
    # mientras que el radar queda en datetime64[ns, UTC]. Para evitar
    # incompatibilidades de merge_asof, ambos lados se normalizan a
    # epoch nanoseconds (int64).
    radar["_time_ns"] = radar["_radar_time"].astype("int64")
    weather["_time_ns"] = weather["_weather_time"].astype("int64")

    radar = radar.sort_values("_time_ns")
    weather = weather.sort_values("_time_ns")

    merged = pd.merge_asof(
        radar,
        weather,
        left_on="_time_ns",
        right_on="_time_ns",
        direction="nearest",
        tolerance=int(pd.Timedelta(minutes=90).value)
    )
    merged["weather_match"] = merged["_weather_time"].notna().astype(float)
    merged = merged.rename(columns={c: f"weather_{c}" for c in WEATHER_FEATURES})
    merged = merged.drop(columns=["_radar_time", "_weather_time", "_time_ns"], errors="ignore")
    print(f"Frames con datos meteorológicos asociados: {int(merged['weather_match'].sum())}/{len(merged)}")
    return merged.reset_index(drop=True)


# ============================================================
# PREPARAR DATASET
# ============================================================

def prepare_dataframe(df):

    aliases = {
        "area_px": "area",
        "nucleos": "components",
    }

    for target, source in aliases.items():

        if (
            target not in df.columns
            and
            source in df.columns
        ):

            df[target] = df[source]

    # Crear columnas faltantes
    # solamente como fallback.

    for column in BASE_FEATURES:

        if column not in df.columns:

            df[column] = 0.0

    # Timestamp

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
            "ERROR: falta frame_time/timestamp."
        )

    # Conversión numérica

    for column in BASE_FEATURES:

        if column == "nucleos":
            continue

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Núcleos

    df["nucleos"] = df[
        "nucleos"
    ].apply(
        parse_nucleos
    )

    # Limpiar infinitos

    df = df.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    # Los NaN de las variables radar
    # se interpretan como ausencia de señal.

    df[
        BASE_FEATURES
    ] = df[
        BASE_FEATURES
    ].fillna(0.0)

    # El timestamp sí debe ser válido.

    df = df.dropna(
        subset=[
            "frame_time"
        ]
    )

    # Orden temporal

    df = df.sort_values(
        "frame_time"
    )

    # Evitar frames duplicados

    df = df.drop_duplicates(
        subset=[
            "frame_time"
        ]
    )

    df = df.reset_index(drop=True)
    df = merge_weather_with_radar(df)
    return df


# ============================================================
# CONTROL DE CALIDAD
# ============================================================

def dataset_quality(df):

    precipitation = (
        df["area_px"] > 0
    )

    precipitation_frames = int(
        precipitation.sum()
    )

    dbz_frames = int(
        (
            df["dbz_pixels"] > 0
        ).sum()
    )

    unique_area = int(
        df["area_px"].nunique()
    )

    max_area = float(
        df["area_px"].max()
    )

    if len(df) < MIN_FRAMES:

        raise SystemExit(
            f"ERROR: hay {len(df)} frames. "
            f"Se necesitan al menos "
            f"{MIN_FRAMES}."
        )

    if (
        precipitation_frames
        <
        MIN_PRECIP_FRAMES
    ):

        raise SystemExit(
            f"ERROR: solo hay "
            f"{precipitation_frames} "
            "frames con precipitación. "
            f"Se necesitan al menos "
            f"{MIN_PRECIP_FRAMES}. "
            "No se entrenará un modelo "
            "artificialmente seco."
        )

    if (
        dbz_frames
        <
        MIN_PRECIP_FRAMES
    ):

        raise SystemExit(
            "ERROR: hay precipitación "
            "pero no suficiente señal "
            "dBZ válida."
        )

    if (
        unique_area < 3
        or
        max_area <= 0
    ):

        raise SystemExit(
            "ERROR: no existe variación "
            "espacial suficiente en el radar."
        )

    return {

        "frames":
            len(df),

        "frames_precipitacion":
            precipitation_frames,

        "frames_dbz":
            dbz_frames,

        "areas_distintas":
            unique_area,

        "area_max":
            max_area,

        "porcentaje_precipitacion":
            round(
                100.0
                *
                precipitation_frames
                /
                len(df),
                2
            ),
    }


# ============================================================
# COMPROBAR CONTINUIDAD TEMPORAL
# ============================================================

def continuous(
    a,
    b
):

    minutes = (
        float(b)
        -
        float(a)
    ) / 60.0

    return (
        FRAME_MIN_MINUTES
        <=
        minutes
        <=
        FRAME_MAX_MINUTES
    )


# ============================================================
# CONSTRUIR SECUENCIAS
# ============================================================

def build_sequences(df):

    rows = []

    feature_names = []

    # Los cuatro frames utilizados:
    #
    # t-30
    # t-20
    # t-10
    # t
    #
    # predicen:
    #
    # t+10

    for offset in (
        -30,
        -20,
        -10,
        0
    ):

        for feature in BASE_FEATURES:

            feature_names.append(
                f"t{offset}_{feature}"
            )

        feature_names.append(f"t{offset}_weather_match")
        for feature in WEATHER_FEATURES:
            feature_names.append(f"t{offset}_weather_{feature}")

    # Variables de tendencia

    for offset in (
        -20,
        -10,
        0
    ):

        for feature in DELTA_FEATURES:

            feature_names.append(
                f"delta{offset}_{feature}"
            )

    # Recorrer frames

    for i in range(
        3,
        len(df) - 1
    ):

        # Cinco frames:
        #
        # t-30
        # t-20
        # t-10
        # t
        # t+10

        times = [

            float(
                df.iloc[j][
                    "frame_time"
                ]
            )

            for j in range(
                i - 3,
                i + 2
            )
        ]

        # TODOS deben estar temporalmente
        # próximos.

        if not all(
            continuous(
                times[j],
                times[j + 1]
            )
            for j in range(4)
        ):

            continue

        values = []

        # Estado de los cuatro frames

        for j in range(
            i - 3,
            i + 1
        ):

            frame = df.iloc[j]

            for feature in BASE_FEATURES:

                values.append(float(frame[feature]))

            match = frame.get("weather_match", 0.0)
            values.append(float(match) if pd.notna(match) else 0.0)
            for feature in WEATHER_FEATURES:
                value = frame.get(f"weather_{feature}", np.nan)
                values.append(float(value) if pd.notna(value) else 0.0)

        # Tendencias entre frames

        for j in range(
            i - 2,
            i + 1
        ):

            current = df.iloc[j]

            previous = df.iloc[
                j - 1
            ]

            for feature in DELTA_FEATURES:

                values.append(

                    float(
                        current[
                            feature
                        ]
                    )
                    -
                    float(
                        previous[
                            feature
                        ]
                    )

                )

        # Target: siguiente frame

        future = df.iloc[
            i + 1
        ]

        values.extend([

            float(
                future[
                    "dbz_max"
                ]
            ),

            float(
                future[
                    "area_px"
                ]
            ),

        ])

        rows.append(
            values
        )

    columns = (
        feature_names
        +
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
# ENTRENAMIENTO
# ============================================================

def train_target(
    X_train,
    X_test,
    y_train,
    y_test,
    target
):

    model = HistGradientBoostingRegressor(

        max_iter=300,

        learning_rate=0.04,

        max_leaf_nodes=31,

        min_samples_leaf=5,

        l2_regularization=1.5,

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

    rmse = math.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )

    # --------------------------------------------------------
    # DIRECCIÓN DEL CAMBIO
    # --------------------------------------------------------

    if target == "target_dbz_max":

        reference = np.asarray(
            X_test[
                "t0_dbz_max"
            ]
        )

    else:

        reference = np.asarray(
            X_test[
                "t0_area_px"
            ]
        )

    actual_change = (
        np.asarray(y_test)
        -
        reference
    )

    predicted_change = (
        np.asarray(prediction)
        -
        reference
    )

    direction = float(
        np.mean(
            np.sign(
                actual_change
            )
            ==
            np.sign(
                predicted_change
            )
        )
    )

    return (

        model,

        {
            "MAE":
                float(mae),

            "RMSE":
                float(rmse),

            "direccion_cambio_correcta":
                round(
                    direction,
                    4
                ),
        }

    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("")
    print("=" * 50)
    print(
        "CLIMAAR - ENTRENAMIENTO NOWCAST V3"
    )
    print("=" * 50)
    print("")

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    if not CSV.exists():

        raise SystemExit(
            f"ERROR: no existe {CSV}"
        )

    df = pd.read_csv(
        CSV
    )

    print(
        f"Filas originales: {len(df)}"
    )

    # --------------------------------------------------------
    # PREPARAR
    # --------------------------------------------------------

    df = prepare_dataframe(
        df
    )

    print(
        f"Frames válidos: {len(df)}"
    )

    # --------------------------------------------------------
    # CALIDAD
    # --------------------------------------------------------

    quality = dataset_quality(
        df
    )

    print("")
    print(
        "CALIDAD DEL DATASET:"
    )

    print(
        json.dumps(
            quality,
            indent=2,
            ensure_ascii=False
        )
    )

    # --------------------------------------------------------
    # SECUENCIAS
    # --------------------------------------------------------

    sequences = build_sequences(
        df
    )

    print("")
    print(
        f"Secuencias continuas: "
        f"{len(sequences)}"
    )

    if len(sequences) < MIN_SEQUENCES:

        raise SystemExit(
            f"ERROR: solo hay "
            f"{len(sequences)} "
            "secuencias continuas. "
            f"Se necesitan al menos "
            f"{MIN_SEQUENCES}."
        )

    # --------------------------------------------------------
    # FEATURES / TARGETS
    # --------------------------------------------------------

    target_columns = [

        "target_dbz_max",

        "target_area_px",

    ]

    feature_columns = [

        column

        for column in sequences.columns

        if column
        not in target_columns

    ]

    X = sequences[
        feature_columns
    ].copy()

    # --------------------------------------------------------
    # HOLDOUT TEMPORAL
    # --------------------------------------------------------

    split = int(
        len(sequences)
        *
        0.80
    )

    train_end = (
        split
        -
        SEQUENCE_GAP
    )

    if (
        train_end < 15
        or
        split >= len(sequences)
    ):

        raise SystemExit(
            "ERROR: dataset demasiado "
            "pequeño para realizar "
            "un holdout temporal seguro."
        )

    X_train = X.iloc[
        :train_end
    ]

    X_test = X.iloc[
        split:
    ]

    print("")
    print(
        f"Entrenamiento: "
        f"{len(X_train)}"
    )

    print(
        f"Gap temporal: "
        f"{SEQUENCE_GAP} secuencias"
    )

    print(
        f"Prueba: "
        f"{len(X_test)}"
    )

    # --------------------------------------------------------
    # VERIFICAR QUE EL TEST TENGA LLUVIA
    # --------------------------------------------------------

    if not (
        sequences.iloc[
            split:
        ][
            "target_area_px"
        ]
        > 0
    ).any():

        raise SystemExit(

            "ERROR: el bloque temporal "
            "de prueba no contiene "
            "precipitación. "

            "Se necesita otro período "
            "con lluvia para validar "
            "el modelo correctamente."

        )

    # --------------------------------------------------------
    # MODELOS
    # --------------------------------------------------------

    models = {}

    metrics = {

        "version":
            "4.0",

        "fuente":
            "RainViewer V7.1",

        "horizonte_minutos":
            10,

        "frames_por_secuencia":
            4,

        "secuencias_totales":
            int(
                len(sequences)
            ),

        "entrenamiento":
            int(
                len(X_train)
            ),

        "prueba":
            int(
                len(X_test)
            ),

        "gap_secuencias":
            SEQUENCE_GAP,

        "calidad_dataset":
            quality,

        "validacion":
            "temporal_holdout_con_gap",

        "targets":
            target_columns,
    }

    # --------------------------------------------------------
    # DBZ
    # --------------------------------------------------------

    print("")
    print(
        "Entrenando dbz_max..."
    )

    y_dbz = sequences[
        "target_dbz_max"
    ]

    model_dbz, result_dbz = train_target(

        X_train,

        X_test,

        y_dbz.iloc[
            :train_end
        ],

        y_dbz.iloc[
            split:
        ],

        "target_dbz_max"

    )

    models[
        "target_dbz_max"
    ] = model_dbz

    metrics[
        "target_dbz_max"
    ] = result_dbz

    print(
        json.dumps(
            result_dbz,
            indent=2,
            ensure_ascii=False
        )
    )

    # --------------------------------------------------------
    # AREA
    # --------------------------------------------------------

    print("")
    print(
        "Entrenando area_px..."
    )

    y_area = sequences[
        "target_area_px"
    ]

    model_area, result_area = train_target(

        X_train,

        X_test,

        y_area.iloc[
            :train_end
        ],

        y_area.iloc[
            split:
        ],

        "target_area_px"

    )

    models[
        "target_area_px"
    ] = model_area

    metrics[
        "target_area_px"
    ] = result_area

    print(
        json.dumps(
            result_area,
            indent=2,
            ensure_ascii=False
        )
    )

    # --------------------------------------------------------
    # GUARDAR MODELO
    # --------------------------------------------------------

    artifact = {

        "version":
            "4.0",

        "source":
            "RainViewer + histórico meteorológico 1980-2026",

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

        "delta_features":
            DELTA_FEATURES,

        "weather_features":
            WEATHER_FEATURES,

        "weather_source":
            str(WEATHER_CSV),

        "validation": {

            "type":
                "temporal_holdout_with_gap",

            "train_fraction":
                0.80,

            "gap_sequences":
                SEQUENCE_GAP,

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
            indent=2,
            ensure_ascii=False
        ),

        encoding="utf-8"

    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("")
    print("=" * 50)
    print(
        "ENTRENAMIENTO NOWCAST V3 COMPLETADO"
    )
    print("=" * 50)
    print("")
    print(
        f"Modelo: {MODEL}"
    )
    print(
        f"Métricas: {METRICS}"
    )
    print("")


if __name__ == "__main__":

    main()
