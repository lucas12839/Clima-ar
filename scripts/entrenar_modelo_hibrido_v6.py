from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

LAT = -38.71
LON = -62.26

OUT = Path("modelo")
OUT.mkdir(exist_ok=True)

SURFACE_DATA = Path(
    "data/historico/clima_horario_1980_2026.csv"
)

CONV_DATA = Path(
    "data/eventos_severos/variables_convectivas_2023_2025.csv"
)

EVENTS = pd.to_datetime([
    "2019-12-30",
    "2023-12-16",
    "2025-03-07",
])

CONV_EVENTS = pd.to_datetime([
    "2023-12-16",
    "2025-03-07",
])


def metricas(y_true, probability, threshold=0.50):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)

    result = {
        "roc_auc": None,
        "average_precision": None,
        "precision": float(
            precision_score(
                y_true, prediction, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, prediction, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, prediction, zero_division=0
            )
        ),
        "threshold": float(threshold),
    }

    if len(np.unique(y_true)) >= 2:
        result["roc_auc"] = float(
            roc_auc_score(y_true, probability)
        )
        result["average_precision"] = float(
            average_precision_score(y_true, probability)
        )

    return result


def mejor_umbral(y_true, probability):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)

    if len(np.unique(y_true)) < 2:
        return 0.50

    precision, recall, thresholds = precision_recall_curve(
        y_true, probability
    )

    if len(thresholds) == 0:
        return 0.50

    f1_values = (
        2.0 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )

    valid = np.isfinite(f1_values)

    if not valid.any():
        return 0.50

    best_index = int(np.nanargmax(
        np.where(valid, f1_values, -1.0)
    ))

    return float(
        np.clip(thresholds[best_index], 0.10, 0.90)
    )


def construir_variables_superficie(df):
    df = df.copy()

    df["time"] = pd.to_datetime(
        df["time"], errors="coerce"
    )

    df = (
        df.dropna(subset=["time"])
        .sort_values("time")
        .drop_duplicates(subset=["time"])
        .reset_index(drop=True)
    )

    for column in df.columns:
        if column != "time":
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

    if {
        "wind_speed_10m",
        "wind_direction_10m",
    }.issubset(df.columns):
        rad = np.deg2rad(df["wind_direction_10m"])
        df["wind_u"] = (
            df["wind_speed_10m"] * np.sin(rad)
        )
        df["wind_v"] = (
            df["wind_speed_10m"] * np.cos(rad)
        )

    if {
        "temperature_2m",
        "dew_point_2m",
    }.issubset(df.columns):
        df["dew_spread"] = (
            df["temperature_2m"]
            - df["dew_point_2m"]
        )

    base = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "pressure_msl",
        "surface_pressure",
        "precipitation",
        "rain",
        "cloud_cover",
        "wind_speed_10m",
        "wind_gusts_10m",
        "dew_spread",
        "wind_u",
        "wind_v",
        "vapour_pressure_deficit",
        "boundary_layer_height",
    ]

    base = [x for x in base if x in df.columns]

    for column in base:
        for hours in (1, 3, 6, 12, 24):
            df[f"{column}_chg{hours}h"] = (
                df[column] - df[column].shift(hours)
            )

    for column in base:
        for window in (3, 6, 12, 24):
            df[f"{column}_mean{window}h"] = (
                df[column]
                .rolling(window, min_periods=1)
                .mean()
            )

    if "precipitation" in df.columns:
        for window in (3, 6, 12, 24):
            df[f"rain{window}h"] = (
                df["precipitation"]
                .rolling(window, min_periods=1)
                .sum()
            )

    if "pressure_msl" in df.columns:
        for hours in (3, 6, 12, 24):
            df[f"pressure_drop{hours}h"] = (
                df["pressure_msl"]
                - df["pressure_msl"].shift(hours)
            )

    hour = df["time"].dt.hour
    month = df["time"].dt.month

    df["hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    df["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    return df


def etiquetar_eventos(df):
    df = df.copy()
    df["target"] = 0

    for event in EVENTS:
        start = event - pd.Timedelta(hours=24)
        end = event

        mask = (
            (df["time"] >= start)
            & (df["time"] < end)
        )

        df.loc[mask, "target"] = 1

    return df


def preparar_matriz(df):
    excluded = {
        "time",
        "target",
        "event_date",
    }

    variables = [
        column
        for column in df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    X = df[variables].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    medians = X.median(
        numeric_only=True
    )

    X = X.fillna(
        medians
    ).fillna(0.0)

    return X, variables, medians.to_dict()


def seleccionar_negativos(
    df,
    positive_mask,
    max_ratio=8,
):
    positives = df.loc[
        positive_mask
    ].copy()

    negatives = df.loc[
        ~positive_mask
    ].copy()

    if len(positives) == 0:
        raise RuntimeError(
            "No hay positivos para entrenar."
        )

    target_negatives = min(
        len(negatives),
        max(
            len(positives) * max_ratio,
            2000,
        ),
    )

    if len(negatives) <= target_negatives:
        return pd.concat(
            [
                positives,
                negatives,
            ],
            ignore_index=True,
        )

    rng = np.random.RandomState(42)

    near_masks = []

    for event in EVENTS:
        near_masks.append(
            (
                df["time"]
                >= event - pd.Timedelta(days=3)
            )
            & (
                df["time"]
                <= event + pd.Timedelta(days=3)
            )
        )

    near = df.loc[
        (~positive_mask)
        & np.logical_or.reduce(
            near_masks
        )
    ].copy()

    near_n = min(
        len(near),
        max(
            len(positives) * 3,
            500,
        ),
    )

    if len(near) > near_n:
        near = near.sample(
            n=near_n,
            random_state=rng,
        )

    remaining_needed = (
        target_negatives
        - len(near)
    )

    pool = negatives.drop(
        index=near.index,
        errors="ignore",
    )

    if (
        remaining_needed > 0
        and len(pool) > remaining_needed
    ):
        random_part = pool.sample(
            n=remaining_needed,
            random_state=rng,
        )
    else:
        random_part = pool

    result = pd.concat(
        [
            positives,
            near,
            random_part,
        ],
        ignore_index=True,
    )

    return (
        result
        .drop_duplicates(
            subset=["time"]
        )
        .sort_values("time")
        .reset_index(drop=True)
    )


def entrenar_superficie():
    if not SURFACE_DATA.exists():
        raise RuntimeError(
            f"No existe {SURFACE_DATA}"
        )

    print("=" * 70)
    print(
        "CLIMAAR - ENTRENAMIENTO "
        "SUPERFICIE V7"
    )
    print("=" * 70)

    df = pd.read_csv(
        SURFACE_DATA
    )

    print(
        "Filas originales:",
        len(df),
    )

    df = construir_variables_superficie(
        df
    )

    df = etiquetar_eventos(
        df
    )

    positive_mask = (
        df["target"]
        .astype(bool)
    )

    positives_total = int(
        positive_mask.sum()
    )

    if positives_total < 30:
        raise RuntimeError(
            "Cantidad de positivos "
            f"insuficiente: {positives_total}"
        )

    validation = {}

    for event in EVENTS:

        test_mask = (
            (
                df["time"]
                >= event
                - pd.Timedelta(hours=48)
            )
            &
            (
                df["time"]
                <= event
                + pd.Timedelta(hours=24)
            )
        )

        train_pool = df.loc[
            ~test_mask
        ].copy()

        test_df = df.loc[
            test_mask
        ].copy()

        train_positive = (
            train_pool["target"]
            .astype(bool)
        )

        train_df = seleccionar_negativos(
            train_pool,
            train_positive,
            max_ratio=8,
        )

        (
            X_train,
            variables,
            medians,
        ) = preparar_matriz(
            train_df
        )

        X_test = test_df[
            variables
        ].copy()

        X_test = X_test.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        X_test = X_test.fillna(
            pd.Series(medians)
        ).fillna(0.0)

        y_train = (
            train_df["target"]
            .astype(int)
            .to_numpy()
        )

        y_test = (
            test_df["target"]
            .astype(int)
            .to_numpy()
        )

        if (
            len(np.unique(y_train)) < 2
            or
            len(np.unique(y_test)) < 2
        ):
            validation[
                str(event.date())
            ] = {
                "skipped": True,
                "train_rows": int(
                    len(train_df)
                ),
                "test_rows": int(
                    len(test_df)
                ),
                "train_positive": int(
                    y_train.sum()
                ),
                "test_positive": int(
                    y_test.sum()
                ),
            }

            continue

        model = RandomForestClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight=(
                "balanced_subsample"
            ),
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        probability = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        threshold = mejor_umbral(
            y_test,
            probability,
        )

        result = metricas(
            y_test,
            probability,
            threshold,
        )

        result.update({
            "train_rows": int(
                len(train_df)
            ),
            "test_rows": int(
                len(test_df)
            ),
            "train_positive": int(
                y_train.sum()
            ),
            "test_positive": int(
                y_test.sum()
            ),
        })

        validation[
            str(event.date())
        ] = result

        print()
        print(
            "Evento:",
            event.date(),
        )

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    final_df = seleccionar_negativos(
        df,
        positive_mask,
        max_ratio=8,
    )

    (
        X_final,
        variables,
        medians,
    ) = preparar_matriz(
        final_df
    )

    y_final = (
        final_df["target"]
        .astype(int)
        .to_numpy()
    )

    print()
    print(
        "Entrenamiento final:"
    )

    print(
        "Filas:",
        len(final_df),
    )

    print(
        "Positivos:",
        int(y_final.sum()),
    )

    print(
        "Variables:",
        len(variables),
    )

    final_model = RandomForestClassifier(
        n_estimators=800,
        max_features="sqrt",
        min_samples_leaf=3,
        class_weight=(
            "balanced_subsample"
        ),
        random_state=42,
        n_jobs=-1,
    )

    final_model.fit(
        X_final,
        y_final,
    )

    model_path = (
        OUT
        / "climaar_predictor_hibrido_superficie_v6.joblib"
    )

    joblib.dump(
        {
            "version":
                "climaar_predictor_hibrido_superficie_v7",

            "model":
                final_model,

            "features":
                variables,

            "medians":
                medians,

            "horizon_hours":
                24,

            "location":
                {
                    "name":
                        "Bahia Blanca",

                    "lat":
                        LAT,

                    "lon":
                        LON,
                },

            "events":
                [
                    str(x.date())
                    for x in EVENTS
                ],

            "negative_sampling":
                {
                    "max_negative_positive_ratio":
                        8,

                    "near_event_weighted_sampling":
                        True,
                },

            "validation":
                validation,
        },
        model_path,
    )

    importance = pd.DataFrame({
        "variable":
            variables,

        "importancia":
            final_model
            .feature_importances_,
    }).sort_values(
        "importancia",
        ascending=False,
    )

    importance.to_csv(
        OUT
        / "importancia_hibrido_superficie_v6.csv",
        index=False,
    )

    return {
        "version":
            "v7",

        "rows_originales":
            int(len(df)),

        "rows_entrenamiento":
            int(len(final_df)),

        "positive":
            int(y_final.sum()),

        "variables":
            int(len(variables)),

        "validation":
            validation,
    }


def entrenar_convectivo():
    if not CONV_DATA.exists():
        raise RuntimeError(
            f"No existe {CONV_DATA}"
        )

    print()
    print("=" * 70)
    print(
        "CLIMAAR - MODELO "
        "CONVECTIVO V3"
    )
    print("=" * 70)

    df = pd.read_csv(
        CONV_DATA
    )

    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce",
    )

    df["event_date"] = pd.to_datetime(
        df["event_date"],
        errors="coerce",
    )

    df = (
        df
        .dropna(
            subset=[
                "time",
                "event_date",
            ]
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    for column in df.columns:
        if column not in {
            "time",
            "event_date",
        }:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df["target"] = 0

    for event in CONV_EVENTS:

        mask = (
            (
                df["event_date"]
                == event
            )
            &
            (
                df["time"]
                >= event
                - pd.Timedelta(hours=24)
            )
            &
            (
                df["time"]
                < event
            )
        )

        df.loc[
            mask,
            "target"
        ] = 1

    if {
        "wind_speed_850hPa",
        "wind_speed_500hPa",
    }.issubset(df.columns):

        df[
            "shear_speed_850_500"
        ] = (
            df["wind_speed_500hPa"]
            -
            df["wind_speed_850hPa"]
        ).abs()

    if {
        "temperature_850hPa",
        "temperature_500hPa",
    }.issubset(df.columns):

        df[
            "lapse_proxy_850_500"
        ] = (
            df["temperature_850hPa"]
            -
            df["temperature_500hPa"]
        )

    excluded = {
        "time",
        "event_date",
        "target",
    }

    variables = [
        c
        for c in df.columns
        if c not in excluded
        and pd.api.types.is_numeric_dtype(
            df[c]
        )
    ]

    X = df[
        variables
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    medians = X.median(
        numeric_only=True
    )

    X = X.fillna(
        medians
    ).fillna(0.0)

    y = (
        df["target"]
        .astype(int)
        .to_numpy()
    )

    validation = {}

    for event in CONV_EVENTS:

        test_mask = (
            df["event_date"]
            == event
        )

        train_mask = ~test_mask

        y_train = y[
            train_mask
        ]

        y_test = y[
            test_mask
        ]

        if (
            len(np.unique(y_train))
            < 2
            or
            len(np.unique(y_test))
            < 2
        ):
            validation[
                str(event.date())
            ] = {
                "skipped": True,
                "train_rows": int(
                    train_mask.sum()
                ),
                "test_rows": int(
                    test_mask.sum()
                ),
                "train_positive": int(
                    y_train.sum()
                ),
                "test_positive": int(
                    y_test.sum()
                ),
            }

            continue

        model = RandomForestClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight=(
                "balanced_subsample"
            ),
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X.loc[train_mask],
            y_train,
        )

        probability = (
            model
            .predict_proba(
                X.loc[test_mask]
            )[:, 1]
        )

        threshold = mejor_umbral(
            y_test,
            probability,
        )

        result = metricas(
            y_test,
            probability,
            threshold,
        )

        result.update({
            "train_rows": int(
                train_mask.sum()
            ),
            "test_rows": int(
                test_mask.sum()
            ),
            "train_positive": int(
                y_train.sum()
            ),
            "test_positive": int(
                y_test.sum()
            ),
        })

        validation[
            str(event.date())
        ] = result

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    final_model = RandomForestClassifier(
        n_estimators=800,
        max_features="sqrt",
        min_samples_leaf=3,
        class_weight=(
            "balanced_subsample"
        ),
        random_state=42,
        n_jobs=-1,
    )

    final_model.fit(
        X,
        y,
    )

    joblib.dump(
        {
            "version":
                "climaar_predictor_convectivo_v3",

            "model":
                final_model,

            "features":
                variables,

            "medians":
                medians.to_dict(),

            "location":
                {
                    "name":
                        "Bahia Blanca",

                    "lat":
                        LAT,

                    "lon":
                        LON,
                },

            "events":
                [
                    str(x.date())
                    for x in CONV_EVENTS
                ],

            "validation":
                validation,
        },

        OUT
        / "climaar_predictor_convectivo_v2.joblib",
    )

    importance = pd.DataFrame({
        "variable":
            variables,

        "importancia":
            final_model
            .feature_importances_,
    }).sort_values(
        "importancia",
        ascending=False,
    )

    importance.to_csv(
        OUT
        / "importancia_convectivo_v2.csv",
        index=False,
    )

    return {
        "version":
            "v3",

        "rows":
            int(len(df)),

        "positive":
            int(y.sum()),

        "variables":
            int(len(variables)),

        "validation":
            validation,
    }


def main():
    surface = entrenar_superficie()
    convective = entrenar_convectivo()

    report = {
        "version":
            "ClimaAR V7",

        "estado":
            "OK",

        "motor":
            (
                "RandomForest balanceado "
                "con validacion por evento"
            ),

        "superficie":
            surface,

        "convectivo":
            convective,
    }

    report_path = (
        OUT
        / "reporte_modelo_hibrido_v6.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print(
        "CLIMAAR - ENTRENAMIENTO "
        "COMPLETADO"
    )
    print("=" * 70)

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
