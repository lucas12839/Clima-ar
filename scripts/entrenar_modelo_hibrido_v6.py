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
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

OUT = Path("modelo")
OUT.mkdir(exist_ok=True)

SURFACE_DATA = Path(
    "data/historico/clima_horario_1980_2026.csv"
)

CONV_DATA = Path(
    "data/eventos_severos/variables_convectivas_2023_2025.csv"
)

EVENT_DATES = pd.to_datetime([
    "2019-12-30",
    "2023-12-16",
    "2025-03-07",
])


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return None

    return float(
        roc_auc_score(y, p)
    )


def metrics(y, p, threshold=0.50):
    pred = (
        p >= threshold
    ).astype(int)

    return {
        "roc_auc": safe_auc(y, p),
        "average_precision": (
            float(
                average_precision_score(y, p)
            )
            if len(np.unique(y)) >= 2
            else None
        ),
        "precision": float(
            precision_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "threshold": float(
            threshold
        ),
    }


def best_threshold(y, p):
    candidates = np.unique(
        np.clip(
            p,
            0.05,
            0.90,
        )
    )

    candidates = np.unique(
        np.r_[
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.40,
            0.50,
            candidates,
        ]
    )

    best = (
        0.20,
        -1.0,
    )

    for t in candidates:
        f = f1_score(
            y,
            (p >= t).astype(int),
            zero_division=0,
        )

        if f > best[1]:
            best = (
                float(t),
                float(f),
            )

    return best[0]


def numeric(df):
    for c in df.columns:
        if c != "time":
            df[c] = pd.to_numeric(
                df[c],
                errors="coerce",
            )

    return df


def build_features(raw):
    df = raw.copy()

    if "time" not in df.columns:
        raise RuntimeError(
            "El dataset no contiene la columna 'time'."
        )

    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce",
    )

    df = (
        df.dropna(
            subset=["time"]
        )
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    df = numeric(df)

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
        "vapour_pressure_deficit",
        "boundary_layer_height",
    ]

    base = [
        c for c in base
        if c in df.columns
    ]

    if {
        "temperature_2m",
        "dew_point_2m",
    } <= set(df.columns):

        df["dew_spread"] = (
            df["temperature_2m"]
            - df["dew_point_2m"]
        )

        base.append(
            "dew_spread"
        )

    if {
        "wind_speed_10m",
        "wind_direction_10m",
    } <= set(df.columns):

        r = np.deg2rad(
            df["wind_direction_10m"]
        )

        df["wind_u"] = (
            df["wind_speed_10m"]
            * np.sin(r)
        )

        df["wind_v"] = (
            df["wind_speed_10m"]
            * np.cos(r)
        )

        base += [
            "wind_u",
            "wind_v",
        ]

    hour = df["time"].dt.hour
    month = df["time"].dt.month

    df["hour_sin"] = (
        np.sin(
            2 * np.pi * hour / 24
        )
    )

    df["hour_cos"] = (
        np.cos(
            2 * np.pi * hour / 24
        )
    )

    df["month_sin"] = (
        np.sin(
            2 * np.pi * month / 12
        )
    )

    df["month_cos"] = (
        np.cos(
            2 * np.pi * month / 12
        )
    )

    return df


def add_targets(df):
    d = df.copy()

    precip_col = (
        "precipitation"
        if "precipitation" in d
        else (
            "rain"
            if "rain" in d
            else None
        )
    )

    gust_col = (
        "wind_gusts_10m"
        if "wind_gusts_10m" in d
        else None
    )

    if precip_col is None:
        raise RuntimeError(
            "No existe precipitation/rain en el dataset."
        )

    rain = (
        d[precip_col]
        .fillna(0)
        .clip(lower=0)
    )

    gust = (
        d[gust_col].fillna(0)
        if gust_col
        else pd.Series(
            0.0,
            index=d.index,
        )
    )

    future_rain_3 = sum(
        rain.shift(-h)
        for h in (1, 2, 3)
    )

    future_rain_6 = sum(
        rain.shift(-h)
        for h in range(1, 7)
    )

    past_rain_3 = sum(
        rain.shift(h)
        for h in (0, 1, 2)
    )

    future_gust_3 = pd.concat(
        [
            gust.shift(-h)
            for h in (1, 2, 3)
        ],
        axis=1,
    ).max(axis=1)

    current_rain = rain
    current_gust = gust

    d["target_formacion_1h3h"] = (
        (current_rain <= 0.10)
        &
        (future_rain_3 >= 1.0)
    ).astype(int)

    d["target_intensificacion_1h3h"] = (
        (
            (
                future_rain_3
                >= np.maximum(
                    2.0,
                    past_rain_3 * 1.50,
                )
            )
            &
            (future_rain_3 >= 2.0)
        )
        |
        (
            (
                future_gust_3
                >= current_gust + 10.0
            )
            &
            (future_gust_3 >= 45.0)
        )
    ).astype(int)

    d["target_impacto_3h6h"] = (
        (future_rain_6 >= 5.0)
        |
        (future_gust_3 >= 50.0)
    ).astype(int)

    d.loc[
        future_rain_6.isna(),
        [
            "target_formacion_1h3h",
            "target_intensificacion_1h3h",
            "target_impacto_3h6h",
        ],
    ] = np.nan

    return d


def make_matrix(df, target):
    excluded = {
        "time",
        target,
        "target_formacion_1h3h",
        "target_intensificacion_1h3h",
        "target_impacto_3h6h",
    }

    cols = [
        c for c in df.columns
        if (
            c not in excluded
            and pd.api.types.is_numeric_dtype(
                df[c]
            )
        )
    ]

    X = df[cols].replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    med = X.median(
        numeric_only=True
    ).to_dict()

    X = (
        X.fillna(
            pd.Series(med)
        )
        .fillna(0.0)
    )

    return (
        X,
        cols,
        med,
    )


def train_target(
    df,
    target,
    seed,
):
    work = df.dropna(
        subset=[target]
    ).copy()

    y = (
        work[target]
        .astype(int)
        .to_numpy()
    )

    X, features, medians = make_matrix(
        work,
        target,
    )

    cut = max(
        int(len(work) * 0.80),
        1,
    )

    X_train = X.iloc[:cut]
    X_test = X.iloc[cut:]

    y_train = y[:cut]
    y_test = y[cut:]

    if (
        len(np.unique(y_train)) < 2
        or len(np.unique(y_test)) < 2
    ):
        raise RuntimeError(
            f"Validacion invalida para {target}: "
            "falta una clase."
        )

    # Modelo de validacion compacto
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        max_features="sqrt",
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    p = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    threshold = best_threshold(
        y_test,
        p,
    )

    report = metrics(
        y_test,
        p,
        threshold,
    )

    report.update({
        "rows": int(
            len(work)
        ),
        "positive": int(
            y.sum()
        ),
        "train_rows": int(
            len(X_train)
        ),
        "test_rows": int(
            len(X_test)
        ),
        "train_positive": int(
            y_train.sum()
        ),
        "test_positive": int(
            y_test.sum()
        ),
    })

    # Modelo final compacto
    final = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        max_features="sqrt",
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )

    final.fit(
        X,
        y,
    )

    return (
        final,
        features,
        medians,
        report,
    )


def convective_reference():
    if not CONV_DATA.exists():
        return {
            "available": False
        }

    c = pd.read_csv(
        CONV_DATA
    )

    if "event_date" not in c.columns:
        return {
            "available": False
        }

    c["event_date"] = pd.to_datetime(
        c["event_date"],
        errors="coerce",
    )

    numeric_cols = [
        x for x in c.columns
        if x not in {
            "event_date",
            "time",
        }
    ]

    for x in numeric_cols:
        c[x] = pd.to_numeric(
            c[x],
            errors="coerce",
        )

    return {
        "available": True,
        "rows": int(
            len(c)
        ),
        "events": sorted(
            c["event_date"]
            .dropna()
            .dt.strftime(
                "%Y-%m-%d"
            )
            .unique()
            .tolist()
        ),
        "variables": int(
            len(numeric_cols)
        ),
        "role":
            "referencia_extremos_documentados",
    }


def main():
    if not SURFACE_DATA.exists():
        raise RuntimeError(
            f"No existe {SURFACE_DATA}"
        )

    print("=" * 72)

    print(
        "CLIMAAR V8 - IA CENTRAL DE "
        "FORMACION / INTENSIFICACION / IMPACTO"
    )

    print("=" * 72)

    raw = pd.read_csv(
        SURFACE_DATA
    )

    print(
        "Filas originales:",
        len(raw),
    )

    df = build_features(
        raw
    )

    df = add_targets(
        df
    )

    targets = [
        (
            "formacion",
            "target_formacion_1h3h",
            101,
        ),
        (
            "intensificacion",
            "target_intensificacion_1h3h",
            202,
        ),
        (
            "impacto",
            "target_impacto_3h6h",
            303,
        ),
    ]

    models = {}
    reports = {}

    for name, target, seed in targets:

        print(
            f"\n--- {name.upper()} ---"
        )

        (
            model,
            features,
            medians,
            report,
        ) = train_target(
            df,
            target,
            seed,
        )

        models[name] = {
            "version": "ClimaAR V8",
            "model": model,
            "features": features,
            "medians": medians,
            "target": target,
            "location": {
                "name": "Bahia Blanca",
                "lat": -38.71,
                "lon": -62.26,
            },
        }

        reports[name] = report

        print(
            json.dumps(
                report,
                indent=2,
            )
        )

    surface_path = (
        OUT
        / "climaar_predictor_hibrido_superficie_v6.joblib"
    )

    joblib.dump(
        {
            "version":
                "climaar_predictor_hibrido_superficie_v8",

            "model":
                models["impacto"]["model"],

            "features":
                models["impacto"]["features"],

            "medians":
                models["impacto"]["medians"],

            "target":
                models["impacto"]["target"],

            "horizon_hours": 6,

            "location": {
                "name": "Bahia Blanca",
                "lat": -38.71,
                "lon": -62.26,
            },

            "role": "impacto",
        },
        surface_path,
    )

    for name in (
        "formacion",
        "intensificacion",
        "impacto",
    ):
        joblib.dump(
            models[name],
            OUT
            / f"climaar_predictor_{name}_v8.joblib",
        )

    impact_model = (
        models["impacto"]["model"]
    )

    impact_features = (
        models["impacto"]["features"]
    )

    pd.DataFrame({
        "variable": impact_features,
        "importance":
            impact_model.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    ).to_csv(
        OUT
        / "importancia_hibrido_superficie_v6.csv",
        index=False,
    )

    conv_ref = convective_reference()

    report = {
        "version": "ClimaAR V8",

        "estado": "OK",

        "motor":
            "IA temporal multiobjetivo "
            "con validacion cronologica",

        "filas_originales": int(
            len(raw)
        ),

        "eventos_extremos_referencia": [
            str(x.date())
            for x in EVENT_DATES
        ],

        "objetivo": {
            "formacion":
                "precipitacion futura 1-3h "
                "partiendo de ausencia de lluvia",

            "intensificacion":
                "aumento futuro de lluvia "
                "o rafagas",

            "impacto":
                "lluvia futura 3-6h "
                "o rafaga >= 50 km/h",
        },

        "modelos": reports,

        "referencia_convectiva":
            conv_ref,

        "sin_fuga_temporal":
            True,

        "nota":
            "Los eventos extremos se usan "
            "como referencia de calibracion "
            "y no como unica definicion "
            "de tormenta.",
    }

    with open(
        OUT
        / "reporte_modelo_hibrido_v6.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        OUT
        / "reporte_modelo_hibrido_v8.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\nCLIMAAR V8 OK"
    )

    print(
        json.dumps(
            reports,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
