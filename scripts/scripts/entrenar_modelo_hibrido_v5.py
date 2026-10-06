from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

LAT = -38.71
LON = -62.26

OUT = Path("modelo")
OUT.mkdir(exist_ok=True)

EVENT_DATES = [
    pd.Timestamp("2019-12-30"),
    pd.Timestamp("2023-12-16"),
    pd.Timestamp("2025-03-07")
]

print("=" * 70)
print("CLIMAAR - ENTRENAMIENTO HIBRIDO V5")
print("=" * 70)


def metricas(y, p):

    pred = (p >= 0.50).astype(int)

    if len(np.unique(y)) >= 2:
        auc = float(roc_auc_score(y, p))
        ap = float(average_precision_score(y, p))
    else:
        auc = None
        ap = None

    return {
        "roc_auc": auc,
        "average_precision": ap,
        "precision": float(
            precision_score(
                y,
                pred,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y,
                pred,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y,
                pred,
                zero_division=0
            )
        )
    }


# ============================================================
# MODELO DE SUPERFICIE
# ============================================================

print("")
print("=" * 70)
print("MODELO DE SUPERFICIE")
print("=" * 70)

path = Path(
    "data/historico/clima_horario_1980_2026.csv"
)

if not path.exists():
    raise RuntimeError(
        f"No existe {path}"
    )

df = pd.read_csv(path)

print(
    "Filas originales:",
    len(df)
)

df["time"] = pd.to_datetime(
    df["time"],
    errors="coerce"
)

for c in df.columns:

    if c != "time":

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

# ------------------------------------------------------------
# VARIABLES DERIVADAS
# ------------------------------------------------------------

if {
    "wind_speed_10m",
    "wind_direction_10m"
}.issubset(df.columns):

    rad = np.deg2rad(
        df["wind_direction_10m"]
    )

    df["wind_u"] = (
        df["wind_speed_10m"] *
        np.sin(rad)
    )

    df["wind_v"] = (
        df["wind_speed_10m"] *
        np.cos(rad)
    )

if {
    "temperature_2m",
    "dew_point_2m"
}.issubset(df.columns):

    df["dew_spread"] = (
        df["temperature_2m"] -
        df["dew_point_2m"]
    )

base = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "dew_spread",
    "wind_u",
    "wind_v"
]

for c in base:

    if c not in df.columns:
        continue

    for lag in [1, 3, 6, 12, 24]:

        df[f"{c}_chg{lag}h"] = (
            df[c] -
            df[c].shift(lag)
        )

    for win in [3, 6, 12, 24]:

        df[f"{c}_mean{win}h"] = (
            df[c]
            .rolling(
                win,
                min_periods=win
            )
            .mean()
        )

if "precipitation" in df.columns:

    for win in [3, 6, 12, 24]:

        df[f"rain{win}h"] = (
            df["precipitation"]
            .rolling(
                win,
                min_periods=win
            )
            .sum()
        )

if "pressure_msl" in df.columns:

    for h in [3, 6, 12, 24]:

        df[f"pressure_drop{h}h"] = (
            df["pressure_msl"] -
            df["pressure_msl"].shift(h)
        )

# ------------------------------------------------------------
# CICLOS TEMPORALES
# ------------------------------------------------------------

df["hour_sin"] = np.sin(
    2 * np.pi *
    df["time"].dt.hour / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi *
    df["time"].dt.hour / 24
)

df["month_sin"] = np.sin(
    2 * np.pi *
    df["time"].dt.month / 12
)

df["month_cos"] = np.cos(
    2 * np.pi *
    df["time"].dt.month / 12
)

# ------------------------------------------------------------
# ETIQUETA
# ------------------------------------------------------------

df["evento_proximas_24h"] = 0

for event_date in EVENT_DATES:

    inicio = (
        event_date -
        pd.Timedelta(hours=24)
    )

    fin = event_date

    mask = (
        (df["time"] >= inicio) &
        (df["time"] < fin)
    )

    df.loc[
        mask,
        "evento_proximas_24h"
    ] = 1

exclude = {
    "time",
    "evento_proximas_24h",
    "station",
    "station_id",
    "station_name",
    "estacion",
    "estacion_id",
    "location",
    "city",
    "wmo",
    "wmo_id",
    "icao",
    "icao_code",
    "_dist_bb"
}

features = [
    c for c in df.columns
    if c not in exclude
]

df = (
    df
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .dropna(
        subset=features +
        ["evento_proximas_24h"]
    )
    .sort_values("time")
    .reset_index(drop=True)
)

print(
    "Filas utilizables:",
    len(df)
)

print(
    "Variables:",
    len(features)
)

print(
    "Positivos:",
    int(
        df["evento_proximas_24h"].sum()
    )
)

# ------------------------------------------------------------
# VALIDACION LEAVE-ONE-EVENT-OUT
# ------------------------------------------------------------

validacion_superficie = {}

for event_date in EVENT_DATES:

    test_start = (
        event_date -
        pd.Timedelta(hours=48)
    )

    test_end = (
        event_date +
        pd.Timedelta(hours=24)
    )

    test_mask = (
        (df["time"] >= test_start) &
        (df["time"] <= test_end)
    )

    train = df[~test_mask]
    test = df[test_mask]

    train_pos = int(
        train["evento_proximas_24h"].sum()
    )

    test_pos = int(
        test["evento_proximas_24h"].sum()
    )

    print("")
    print(
        "Evento de prueba:",
        event_date.date()
    )

    print(
        "Train:",
        len(train)
    )

    print(
        "Test:",
        len(test)
    )

    print(
        "Positivos train:",
        train_pos
    )

    print(
        "Positivos test:",
        test_pos
    )

    if train_pos < 10:
        print("SKIP: pocos positivos en train")
        continue

    if test_pos < 5:
        print("SKIP: pocos positivos en test")
        continue

    model = RandomForestClassifier(
        n_estimators=600,
        max_features="sqrt",
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train[features],
        train["evento_proximas_24h"]
    )

    probability = (
        model
        .predict_proba(test[features])[:, 1]
    )

    resultado = metricas(
        test["evento_proximas_24h"].values,
        probability
    )

    resultado["train_rows"] = int(len(train))
    resultado["test_rows"] = int(len(test))
    resultado["train_positive"] = train_pos
    resultado["test_positive"] = test_pos

    validacion_superficie[
        str(event_date.date())
    ] = resultado

    print(
        json.dumps(
            resultado,
            indent=2
        )
    )

# ------------------------------------------------------------
# MODELO FINAL SUPERFICIE
# ------------------------------------------------------------

print("")
print(
    "Entrenando modelo final de superficie..."
)

modelo_superficie = RandomForestClassifier(
    n_estimators=1000,
    max_features="sqrt",
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

modelo_superficie.fit(
    df[features],
    df["evento_proximas_24h"]
)

joblib.dump(
    {
        "version":
            "climaar_predictor_hibrido_superficie_v5",

        "model":
            modelo_superficie,

        "features":
            features,

        "horizon_hours":
            24,

        "location": {
            "name":
                "Bahia Blanca",

            "lat":
                LAT,

            "lon":
                LON
        },

        "events":
            [
                str(x.date())
                for x in EVENT_DATES
            ],

        "validation":
            validacion_superficie
    },

    OUT /
    "climaar_predictor_hibrido_superficie_v5.joblib"
)

pd.DataFrame({
    "variable":
        features,

    "importancia":
        modelo_superficie.feature_importances_
}).sort_values(
    "importancia",
    ascending=False
).to_csv(
    OUT /
    "importancia_hibrido_superficie_v5.csv",
    index=False
)


# ============================================================
# MODELO CONVECTIVO
# ============================================================

print("")
print("=" * 70)
print("MODELO CONVECTIVO")
print("=" * 70)

conv_path = Path(
    "data/eventos_severos/"
    "variables_convectivas_2023_2025.csv"
)

if not conv_path.exists():

    raise RuntimeError(
        f"No existe {conv_path}"
    )

conv = pd.read_csv(conv_path)

conv["time"] = pd.to_datetime(
    conv["time"],
    errors="coerce"
)

for c in conv.columns:

    if c != "time":

        conv[c] = pd.to_numeric(
            conv[c],
            errors="coerce"
        )

conv["evento"] = 0

for event_date, hours in [
    (
        pd.Timestamp("2023-12-16"),
        24
    ),
    (
        pd.Timestamp("2025-03-07"),
        48
    )
]:

    inicio = (
        event_date -
        pd.Timedelta(hours=hours)
    )

    fin = event_date

    mask = (
        (conv["time"] >= inicio) &
        (conv["time"] < fin)
    )

    conv.loc[
        mask,
        "evento"
    ] = 1

if {
    "wind_speed_850hPa",
    "wind_speed_500hPa"
}.issubset(conv.columns):

    conv["shear_speed_850_500"] = (
        conv["wind_speed_850hPa"] -
        conv["wind_speed_500hPa"]
    ).abs()

if {
    "temperature_850hPa",
    "temperature_500hPa"
}.issubset(conv.columns):

    conv["lapse_850_500"] = (
        conv["temperature_850hPa"] -
        conv["temperature_500hPa"]
    )

for c in [
    "cape",
    "lifted_index",
    "convective_inhibition",
    "freezing_level_height",
    "boundary_layer_height",
    "total_column_integrated_water_vapour"
]:

    if c not in conv.columns:
        continue

    conv[f"{c}_chg3h"] = (
        conv[c] -
        conv[c].shift(3)
    )

conv_features = [
    c for c in conv.columns
    if c not in {
        "time",
        "evento"
    }
]

conv = (
    conv
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .dropna(
        subset=conv_features +
        ["evento"]
    )
    .sort_values("time")
    .reset_index(drop=True)
)

print(
    "Filas convectivas:",
    len(conv)
)

print(
    "Variables convectivas:",
    len(conv_features)
)

print(
    "Positivos:",
    int(conv["evento"].sum())
)

validacion_convectiva = {}

for event_date in [
    pd.Timestamp("2023-12-16"),
    pd.Timestamp("2025-03-07")
]:

    test_start = (
        event_date -
        pd.Timedelta(hours=72)
    )

    test_end = (
        event_date +
        pd.Timedelta(hours=24)
    )

    test_mask = (
        (conv["time"] >= test_start) &
        (conv["time"] <= test_end)
    )

    train = conv[~test_mask]
    test = conv[test_mask]

    train_pos = int(
        train["evento"].sum()
    )

    test_pos = int(
        test["evento"].sum()
    )

    print("")
    print(
        "Evento convectivo:",
        event_date.date()
    )

    print(
        "Train:",
        len(train)
    )

    print(
        "Test:",
        len(test)
    )

    print(
        "Positivos train:",
        train_pos
    )

    print(
        "Positivos test:",
        test_pos
    )

    if train_pos < 5 or test_pos < 5:
        print("SKIP")
        continue

    model = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train[conv_features],
        train["evento"]
    )

    probability = (
        model
        .predict_proba(test[conv_features])[:, 1]
    )

    resultado = metricas(
        test["evento"].values,
        probability
    )

    resultado["train_rows"] = int(len(train))
    resultado["test_rows"] = int(len(test))
    resultado["train_positive"] = train_pos
    resultado["test_positive"] = test_pos

    validacion_convectiva[
        str(event_date.date())
    ] = resultado

    print(
        json.dumps(
            resultado,
            indent=2
        )
    )

# ------------------------------------------------------------
# MODELO FINAL CONVECTIVO
# ------------------------------------------------------------

print("")
print(
    "Entrenando modelo convectivo final..."
)

modelo_convectivo = RandomForestClassifier(
    n_estimators=700,
    max_features="sqrt",
    min_samples_leaf=3,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

modelo_convectivo.fit(
    conv[conv_features],
    conv["evento"]
)

joblib.dump(
    {
        "version":
            "climaar_predictor_convectivo_v2",

        "model":
            modelo_convectivo,

        "features":
            conv_features,

        "location": {
            "name":
                "Bahia Blanca",

            "lat":
                LAT,

            "lon":
                LON
        },

        "events": [
            "2023-12-16",
            "2025-03-07"
        ],

        "validation":
            validacion_convectiva
    },

    OUT /
    "climaar_predictor_convectivo_v2.joblib"
)

pd.DataFrame({
    "variable":
        conv_features,

    "importancia":
        modelo_convectivo.feature_importances_
}).sort_values(
    "importancia",
    ascending=False
).to_csv(
    OUT /
    "importancia_convectiva_v2.csv",
    index=False
)


# ============================================================
# REPORTE
# ============================================================

reporte = {

    "version":
        "climaar_modelo_hibrido_v5",

    "ubicacion": {
        "nombre":
            "Bahia Blanca",

        "latitud":
            LAT,

        "longitud":
            LON
    },

    "eventos": [
        "2019-12-30",
        "2023-12-16",
        "2025-03-07"
    ],

    "superficie": {
        "filas":
            int(len(df)),

        "positivos":
            int(
                df[
                    "evento_proximas_24h"
                ].sum()
            ),

        "variables":
            len(features),

        "validacion":
            validacion_superficie
    },

    "convectivo": {
        "filas":
            int(len(conv)),

        "positivos":
            int(
                conv["evento"].sum()
            ),

        "variables":
            len(conv_features),

        "validacion":
            validacion_convectiva
    },

    "referencias_eventos": {

        "2023-12-16": {
            "rafaga_max_kmh":
                155
        },

        "2025-03-07": {
            "lluvia_6h_mm":
                210,

            "lluvia_12h_mm":
                290,

            "total_evento_mm":
                312
        }
    },

    "nota":
        "2019 se considera evento regional. "
        "No se inventan variables convectivas "
        "para 2019."
}

with open(
    OUT /
    "reporte_modelo_hibrido_v5.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        reporte,
        f,
        indent=2,
        ensure_ascii=False
    )

print("")
print("=" * 70)
print("ENTRENAMIENTO COMPLETADO CORRECTAMENTE")
print("=" * 70)

print(
    json.dumps(
        reporte,
        indent=2,
        ensure_ascii=False
    )
  )
