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
    f1_score,
)

LAT = -38.71
LON = -62.26

OUT = Path("modelo")
OUT.mkdir(exist_ok=True)

EVENTS = pd.to_datetime([
    "2019-12-30",
    "2023-12-16",
    "2025-03-07",
])

CONV_EVENTS = pd.to_datetime([
    "2023-12-16",
    "2025-03-07",
])


def calcular_metricas(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    pred = (p >= 0.50).astype(int)

    resultado = {
        "roc_auc": None,
        "average_precision": None,
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
        ),
        "threshold": 0.50,
    }

    if len(np.unique(y)) >= 2:
        resultado["roc_auc"] = float(
            roc_auc_score(y, p)
        )

        resultado["average_precision"] = float(
            average_precision_score(y, p)
        )

    return resultado


def construir_variables_superficie(df):

    df = df.copy()

    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce"
    )

    df = (
        df
        .dropna(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    for columna in df.columns:

        if columna != "time":

            df[columna] = pd.to_numeric(
                df[columna],
                errors="coerce"
            )

    # Viento U/V

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

    # Diferencia temperatura / punto de rocío

    if {
        "temperature_2m",
        "dew_point_2m"
    }.issubset(df.columns):

        df["dew_spread"] = (
            df["temperature_2m"] -
            df["dew_point_2m"]
        )

    variables_base = [

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

    variables_base = [
        x
        for x in variables_base
        if x in df.columns
    ]

    # Cambios temporales

    for columna in variables_base:

        for horas in (1, 3, 6, 12, 24):

            df[
                f"{columna}_chg{horas}h"
            ] = (
                df[columna] -
                df[columna].shift(horas)
            )

    # Promedios móviles

    for columna in variables_base:

        for ventana in (3, 6, 12, 24):

            df[
                f"{columna}_mean{ventana}h"
            ] = (
                df[columna]
                .rolling(
                    ventana,
                    min_periods=1
                )
                .mean()
            )

    # Lluvia acumulada

    if "precipitation" in df.columns:

        for ventana in (3, 6, 12, 24):

            df[
                f"rain{ventana}h"
            ] = (
                df["precipitation"]
                .rolling(
                    ventana,
                    min_periods=1
                )
                .sum()
            )

    # Caída de presión

    if "pressure_msl" in df.columns:

        for horas in (3, 6, 12, 24):

            df[
                f"pressure_drop{horas}h"
            ] = (
                df["pressure_msl"] -
                df["pressure_msl"].shift(horas)
            )

    # Ciclos temporales

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

    return df


def colocar_etiquetas(df, eventos):

    df = df.copy()

    df["target"] = 0

    for evento in eventos:

        inicio = (
            evento -
            pd.Timedelta(hours=24)
        )

        fin = evento

        mascara = (
            (df["time"] >= inicio) &
            (df["time"] < fin)
        )

        df.loc[
            mascara,
            "target"
        ] = 1

    return df


def preparar_matriz(df, excluir):

    variables = []

    for columna in df.columns:

        if columna in excluir:
            continue

        if pd.api.types.is_numeric_dtype(
            df[columna]
        ):
            variables.append(columna)

    X = df[variables].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # IMPORTANTE:
    # NO eliminamos las filas.
    # Rellenamos valores faltantes.

    medianas = X.median(
        numeric_only=True
    )

    X = X.fillna(medianas)
    X = X.fillna(0.0)

    return X, variables, medianas.to_dict()


def entrenar_superficie():

    print("=" * 70)
    print("CLIMAAR - MODELO DE SUPERFICIE V6")
    print("=" * 70)

    archivo = Path(
        "data/historico/"
        "clima_horario_1980_2026.csv"
    )

    if not archivo.exists():

        raise RuntimeError(
            f"No existe {archivo}"
        )

    print("Leyendo histórico...")

    df = pd.read_csv(
        archivo
    )

    print(
        "Filas originales:",
        len(df)
    )

    df = construir_variables_superficie(
        df
    )

    df = colocar_etiquetas(
        df,
        EVENTS
    )

    X, variables, medianas = preparar_matriz(
        df,
        {"target"}
    )

    y = (
        df["target"]
        .astype(int)
        .to_numpy()
    )

    positivos = int(
        y.sum()
    )

    print(
        "Filas utilizables:",
        len(df)
    )

    print(
        "Variables:",
        len(variables)
    )

    print(
        "Positivos:",
        positivos
    )

    if positivos < 30:

        raise RuntimeError(
            "Cantidad de positivos insuficiente: "
            f"{positivos}"
        )

    validacion = {}

    # --------------------------------------------------
    # VALIDACIÓN SEPARANDO CADA EVENTO
    # --------------------------------------------------

    for evento in EVENTS:

        test_mask = (
            (df["time"] >=
             evento -
             pd.Timedelta(hours=48))
            &
            (df["time"] <=
             evento +
             pd.Timedelta(hours=24))
        )

        train_mask = ~test_mask

        y_train = y[train_mask]
        y_test = y[test_mask]

        print()
        print(
            "Evento de prueba:",
            evento.date()
        )

        print(
            "Train:",
            int(train_mask.sum())
        )

        print(
            "Test:",
            int(test_mask.sum())
        )

        print(
            "Positivos train:",
            int(y_train.sum())
        )

        print(
            "Positivos test:",
            int(y_test.sum())
        )

        if (
            y_train.sum() < 10
            or
            y_test.sum() < 5
            or
            len(np.unique(y_train)) < 2
        ):

            validacion[
                str(evento.date())
            ] = {
                "skipped": True,
                "train_positive":
                    int(y_train.sum()),
                "test_positive":
                    int(y_test.sum()),
            }

            continue

        modelo = RandomForestClassifier(

            n_estimators=150,

            max_features="sqrt",

            min_samples_leaf=5,

            class_weight=
                "balanced_subsample",

            random_state=42,

            n_jobs=-1,
        )

        modelo.fit(
            X.loc[train_mask],
            y_train
        )

        probabilidad = (
            modelo
            .predict_proba(
                X.loc[test_mask]
            )[:, 1]
        )

        resultado = calcular_metricas(
            y_test,
            probabilidad
        )

        resultado[
            "train_rows"
        ] = int(train_mask.sum())

        resultado[
            "test_rows"
        ] = int(test_mask.sum())

        resultado[
            "train_positive"
        ] = int(y_train.sum())

        resultado[
            "test_positive"
        ] = int(y_test.sum())

        validacion[
            str(evento.date())
        ] = resultado

        print(
            json.dumps(
                resultado,
                indent=2
            )
        )

    # --------------------------------------------------
    # MODELO FINAL
    # --------------------------------------------------

    print()
    print(
        "Entrenando modelo final..."
    )

    modelo_final = RandomForestClassifier(

        n_estimators=350,

        max_features="sqrt",

        min_samples_leaf=5,

        class_weight=
            "balanced_subsample",

        random_state=42,

        n_jobs=-1,
    )

    modelo_final.fit(
        X,
        y
    )

    archivo_modelo = (
        OUT /
        "climaar_predictor_hibrido_superficie_v6.joblib"
    )

    joblib.dump(
        {
            "version":
                "climaar_predictor_hibrido_superficie_v6",

            "model":
                modelo_final,

            "features":
                variables,

            "medians":
                medianas,

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

            "validation":
                validacion,
        },
        archivo_modelo
    )

    importancia = pd.DataFrame({

        "variable":
            variables,

        "importancia":
            modelo_final
            .feature_importances_,

    })

    importancia = (
        importancia
        .sort_values(
            "importancia",
            ascending=False
        )
    )

    importancia.to_csv(

        OUT /
        "importancia_hibrido_superficie_v6.csv",

        index=False
    )

    return {

        "rows":
            len(df),

        "positive":
            positivos,

        "variables":
            len(variables),

        "validation":
            validacion,

    }


def entrenar_convectivo():

    print()
    print("=" * 70)
    print("CLIMAAR - MODELO CONVECTIVO V2")
    print("=" * 70)

    archivo = Path(
        "data/eventos_severos/"
        "variables_convectivas_2023_2025.csv"
    )

    if not archivo.exists():

        raise RuntimeError(
            f"No existe {archivo}"
        )

    df = pd.read_csv(
        archivo
    )

    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce"
    )

    df["event_date"] = pd.to_datetime(
        df["event_date"],
        errors="coerce"
    )

    df = (
        df
        .dropna(
            subset=[
                "time",
                "event_date"
            ]
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    for columna in df.columns:

        if columna not in {
            "time",
            "event_date"
        }:

            df[columna] = pd.to_numeric(
                df[columna],
                errors="coerce"
            )

    df["target"] = 0

    for evento in CONV_EVENTS:

        mascara = (

            (df["event_date"] == evento)

            &

            (
                df["time"] >=
                evento -
                pd.Timedelta(hours=24)
            )

            &

            (
                df["time"] <
                evento
            )
        )

        df.loc[
            mascara,
            "target"
        ] = 1

    # Cizalladura aproximada

    if {
        "wind_speed_850hPa",
        "wind_speed_500hPa"
    }.issubset(df.columns):

        df["shear_speed_850_500"] = (
            df["wind_speed_500hPa"] -
            df["wind_speed_850hPa"]
        ).abs()

    # Gradiente térmico aproximado

    if {
        "temperature_850hPa",
        "temperature_500hPa"
    }.issubset(df.columns):

        df["lapse_proxy_850_500"] = (
            df["temperature_850hPa"] -
            df["temperature_500hPa"]
        )

    X, variables, medianas = preparar_matriz(

        df,

        {
            "target",
            "event_date"
        }
    )

    y = (
        df["target"]
        .astype(int)
        .to_numpy()
    )

    print(
        "Filas:",
        len(df)
    )

    print(
        "Positivos:",
        int(y.sum())
    )

    validacion = {}

    for evento in CONV_EVENTS:

        test_mask = (
            df["event_date"] ==
            evento
        )

        train_mask = ~test_mask

        y_train = y[train_mask]
        y_test = y[test_mask]

        print()
        print(
            "Validando:",
            evento.date()
        )

        print(
            "Positivos train:",
            int(y_train.sum())
        )

        print(
            "Positivos test:",
            int(y_test.sum())
        )

        if (
            y_train.sum() < 10
            or
            y_test.sum() < 5
        ):

            validacion[
                str(evento.date())
            ] = {
                "skipped": True
            }

            continue

        modelo = RandomForestClassifier(

            n_estimators=250,

            max_features="sqrt",

            min_samples_leaf=2,

            class_weight=
                "balanced_subsample",

            random_state=42,

            n_jobs=-1,
        )

        modelo.fit(
            X.loc[train_mask],
            y_train
        )

        probabilidad = (
            modelo
            .predict_proba(
                X.loc[test_mask]
            )[:, 1]
        )

        validacion[
            str(evento.date())
        ] = calcular_metricas(
            y_test,
            probabilidad
        )

    print()
    print(
        "Entrenando modelo convectivo final..."
    )

    modelo_final = RandomForestClassifier(

        n_estimators=300,

        max_features="sqrt",

        min_samples_leaf=2,

        class_weight=
            "balanced_subsample",

        random_state=42,

        n_jobs=-1,
    )

    modelo_final.fit(
        X,
        y
    )

    joblib.dump(

        {
            "version":
                "climaar_predictor_convectivo_v2",

            "model":
                modelo_final,

            "features":
                variables,

            "medians":
                medianas,

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
                validacion,
        },

        OUT /
        "climaar_predictor_convectivo_v2.joblib"
    )

    pd.DataFrame({

        "variable":
            variables,

        "importancia":
            modelo_final
            .feature_importances_,

    }).sort_values(

        "importancia",

        ascending=False

    ).to_csv(

        OUT /
        "importancia_convectivo_v2.csv",

        index=False
    )

    return {

        "rows":
            len(df),

        "positive":
            int(y.sum()),

        "variables":
            len(variables),

        "validation":
            validacion,
    }


def main():

    print()
    print("=" * 70)
    print("CLIMAAR - ENTRENAMIENTO HIBRIDO V6")
    print("=" * 70)

    superficie = entrenar_superficie()

    convectivo = entrenar_convectivo()

    reporte = {

        "version":
            "6.0",

        "location":
            {
                "name":
                    "Bahia Blanca",

                "lat":
                    LAT,

                "lon":
                    LON,
            },

        "superficie":
            superficie,

        "convectivo":
            convectivo,

        "eventos":
            {
                "2019-12-30":
                    "tormenta severa regional",

                "2023-12-16":
                    "viento destructivo",

                "2025-03-07":
                    "lluvia extrema e inundacion",
            },

        "referencias_oficiales":
            {
                "2023":
                    "maximo oficial documentado 155 km/h",

                "2025":
                    "210 mm/6h, 290 mm/12h y 312 mm durante el evento",
            },

        "nota":
            "Modelo experimental. "
            "No se inventan variables convectivas para 2019.",
    }

    archivo_reporte = (
        OUT /
        "reporte_modelo_hibrido_v6.json"
    )

    archivo_reporte.write_text(

        json.dumps(
            reporte,
            indent=2,
            ensure_ascii=False
        ),

        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 70)

    print(
        json.dumps(
            reporte,
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":

    main()
