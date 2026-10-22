#!/usr/bin/env python3
"""
Entrena el clasificador historico sin_tormenta / tormenta / tormenta_fuerte /
tormenta_severa con TODO el historico horario disponible y el catalogo de
eventos severos documentados, y audita cada fuente de datos del proyecto.

Etiquetas: resultado observado en las proximas 3 horas (t .. t+3 h) en el
historico horario de Bahia Blanca. Variables: solo informacion disponible en
el instante t. El historico es una reconstruccion horaria (reanalisis) que
subestima picos convectivos, por eso los umbrales son proxies y los eventos
severos documentados fuerzan la clase severa alrededor de su pico.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clasificador_tormenta import (  # noqa: E402
    CLASSES,
    FEATURES,
    MODEL_FILE,
    CLASS_WEIGHTS,
)

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
HOURLY = DATA / "historico" / "clima_horario_1980_2026.csv"
EVENTS = DATA / "historico" / "eventos_severos.csv"
METRICS = BASE / "modelo" / "clasificador_tormenta_metricas.json"
AUDIT = BASE / "modelo" / "auditoria_datos_historicos.json"

WINDOW_HOURS = 3
TEST_FROM_YEAR = 2019

# (gust km/h, 1 h mm, ventana mm) por clase; las rafagas solo cuentan si
# hay lluvia acompanante (un dia ventoso seco no es tormenta).
THRESHOLDS = {
    3: (90.0, 15.0, 30.0),
    2: (70.0, 8.0, 15.0),
    1: (60.0, 2.0, 4.0),
}


def forward_max(series, hours):
    reversed_roll = series[::-1].rolling(hours + 1, min_periods=1).max()
    return reversed_roll[::-1]


def forward_sum(series, hours):
    reversed_roll = series[::-1].rolling(hours + 1, min_periods=1).sum()
    return reversed_roll[::-1]


def build_dataset():
    df = pd.read_csv(HOURLY, parse_dates=["time"]).sort_values("time")
    df = df.reset_index(drop=True)

    gust_f = forward_max(df["wind_gusts_10m"], WINDOW_HOURS)
    p1_f = forward_max(df["precipitation"], WINDOW_HOURS)
    pw_f = forward_sum(df["precipitation"], WINDOW_HOURS)

    label = np.zeros(len(df), dtype=int)
    rain = pw_f >= 0.5

    for cls in (1, 2, 3):
        gust_t, p1_t, pw_t = THRESHOLDS[cls]
        hit = (p1_f >= p1_t) | (pw_f >= pw_t) | ((gust_f >= gust_t) & rain)
        label[hit.to_numpy()] = cls

    # Eventos severos documentados: pico del dia del evento -> clase severa
    # en las horas cuyo horizonte contiene ese pico.
    events = pd.read_csv(EVENTS)
    forced = []
    for _, event in events.iterrows():
        if int(event.get("tormenta_severa", 0)) != 1:
            continue
        day = pd.Timestamp(event["fecha"])
        mask = (df["time"] >= day) & (df["time"] < day + pd.Timedelta(days=1))
        if not mask.any():
            continue
        score = (
            df.loc[mask, "wind_gusts_10m"] / 90.0
            + df.loc[mask, "precipitation"] / 15.0
        )
        peak = int(score.idxmax())
        lo = max(0, peak - WINDOW_HOURS)
        label[lo:peak + 1] = 3
        forced.append(
            {
                "fecha": str(event["fecha"]),
                "evento": str(event["evento"]),
                "hora_pico": str(df.loc[peak, "time"]),
                "horas_forzadas": int(peak + 1 - lo),
                "verificacion": str(event.get("estado_verificacion")),
            }
        )

    t = df["time"]
    hour = t.dt.hour
    month = t.dt.month
    wind_rad = np.radians(df["wind_direction_10m"])

    features = pd.DataFrame(
        {
            "temperature_2m": df["temperature_2m"],
            "relative_humidity_2m": df["relative_humidity_2m"],
            "dew_spread": df["temperature_2m"] - df["dew_point_2m"],
            "pressure_msl": df["pressure_msl"],
            "wind_speed_10m": df["wind_speed_10m"],
            "wind_u": df["wind_speed_10m"] * np.sin(wind_rad),
            "wind_v": df["wind_speed_10m"] * np.cos(wind_rad),
            "wind_gusts_10m": df["wind_gusts_10m"],
            "gust_excess": df["wind_gusts_10m"] - df["wind_speed_10m"],
            "precipitation": df["precipitation"],
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        }
    )[list(FEATURES)]

    # Las ultimas horas no tienen horizonte completo.
    valid = np.arange(len(df)) < len(df) - WINDOW_HOURS

    return features[valid], pd.Series(label[valid]), t[valid], forced


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.06,
        max_depth=6,
        min_samples_leaf=40,
        class_weight="balanced",
        random_state=42,
    )


def evaluate(model, x_test, y_test):
    proba = model.predict_proba(x_test)
    classes = list(model.classes_)
    pred = np.array(classes)[proba.argmax(axis=1)]
    full = np.zeros((len(x_test), len(CLASSES)))
    for i, c in enumerate(classes):
        full[:, c] = proba[:, i]
    index = full @ np.array(CLASS_WEIGHTS)

    result = {
        "muestras": int(len(y_test)),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(y_test, pred)), 4
        ),
        "matriz_confusion": confusion_matrix(
            y_test, pred, labels=list(range(len(CLASSES)))
        ).tolist(),
        "distribucion_real": {
            CLASSES[i]: int((y_test == i).sum()) for i in range(len(CLASSES))
        },
    }

    for cls in (1, 2, 3):
        positives = int((y_test >= cls).sum())
        key = f"auc_{CLASSES[cls]}_o_superior"
        if 0 < positives < len(y_test):
            score = full[:, cls:].sum(axis=1)
            result[key] = round(
                float(roc_auc_score((y_test >= cls).astype(int), score)), 4
            )
        else:
            result[key] = None

    result["indice_medio_por_clase_real"] = {
        CLASSES[i]: (
            round(float(index[y_test.to_numpy() == i].mean()), 4)
            if (y_test == i).any()
            else None
        )
        for i in range(len(CLASSES))
    }
    return result


def audit_sources():
    def info(path, usage, note, date_col=None):
        p = BASE / path
        out = {"archivo": path, "usado_para": usage, "nota": note}
        if not p.exists():
            out["existe"] = False
            return out
        out["existe"] = True
        try:
            frame = pd.read_csv(p)
            out["filas"] = int(len(frame))
            if date_col and date_col in frame.columns:
                dates = pd.to_datetime(
                    frame[date_col], errors="coerce", utc=True
                ).dropna()
                if len(dates):
                    out["desde"] = str(dates.min())
                    out["hasta"] = str(dates.max())
        except Exception as exc:
            out["error"] = str(exc)
        return out

    return [
        info(
            "data/historico/clima_horario_1980_2026.csv",
            "ENTRENAMIENTO clasificador de 4 clases (variables y etiquetas)",
            "46 anios horarios sin huecos; reconstruccion que subestima "
            "picos convectivos.",
            "time",
        ),
        info(
            "data/historico/eventos_severos.csv",
            "ENTRENAMIENTO: fuerza clase severa en el pico de cada evento",
            "5 eventos documentados; 2 siguen 'pendiente_verificacion'.",
            "fecha",
        ),
        info(
            "data/historico/ventanas_eventos_24h.csv",
            "AUDITORIA: ventanas de 24 h previas a eventos",
            "Ya contenido en el historico horario; no se duplica.",
            "time",
        ),
        info(
            "data/eventos_severos/variables_convectivas_2023_2025.csv",
            "AUDITORIA: CAPE/LI/niveles de presion en 2 eventos",
            "Solo 2 eventos; insuficiente para entrenar. Se usa como "
            "referencia en reglas ambientales del Storm Engine.",
            "time",
        ),
        info(
            "data/eventos_severos/meteorologia_tres_eventos.csv",
            "AUDITORIA: 3 eventos con variables extendidas",
            "Muestra demasiado chica para entrenar por separado.",
            "time",
        ),
        info(
            "data/radar/rma10_dataset_historico.csv",
            "MODELO RMA10 (fortalecimiento/debilitamiento/hacia Bahia)",
            "Entrenado en data/models/rma10_model.joblib; sin verdad "
            "terreno de clase meteorologica, por eso no entrena el "
            "clasificador de 4 clases.",
            "frame_utc",
        ),
        info(
            "data/radar/radar_features_rma10.csv",
            "AUDITORIA: serie de actividad radar RMA10",
            "Sin solape temporal con el historico horario (termina el "
            "2026-10-03; RMA10 empieza el 2026-10-12).",
            "frame_utc",
        ),
        info(
            "data/radar/radar_features.csv",
            "NO USADO: decodificador RainViewer legado",
            "Fuente distinta de RMA10 (ver reconstruir_dataset_radar.py).",
            "frame_utc",
        ),
        info(
            "data/ia/historial_inteligencia_tormenta.csv",
            "NO USADO PARA ENTRENAR: salida del propio motor",
            "Entrenar con salidas propias seria circular.",
            "timestamp_utc",
        ),
        info(
            "data/ia/historial_meteorologico.csv",
            "NO USADO PARA ENTRENAR: instantaneas del propio pipeline",
            "Sin etiqueta de resultado observado.",
            "timestamp_utc",
        ),
    ]


def main():
    x, y, times, forced = build_dataset()
    years = times.dt.year

    train = (years < TEST_FROM_YEAR).to_numpy()
    test = ~train

    holdout = make_model().fit(x[train], y[train])
    metrics_test = evaluate(holdout, x[test], y[test])

    event_checks = []
    for item in forced:
        peak = pd.Timestamp(item["hora_pico"])
        window = (times >= peak - pd.Timedelta(hours=6)) & (times <= peak)
        window = window.to_numpy()
        if not window.any():
            continue
        proba = holdout.predict_proba(x[window])
        full = np.zeros((window.sum(), len(CLASSES)))
        for i, c in enumerate(holdout.classes_):
            full[:, c] = proba[:, i]
        event_checks.append(
            {
                **item,
                "en_entrenamiento": bool(pd.Timestamp(item["fecha"]).year < TEST_FROM_YEAR),
                "indice_maximo_6h_previas": round(
                    float((full @ np.array(CLASS_WEIGHTS)).max()), 4
                ),
            }
        )

    baseline_index = (
        holdout.predict_proba(x[test])
        @ np.array([CLASS_WEIGHTS[c] for c in holdout.classes_])
    )

    final = make_model().fit(x, y)

    medians = x.median().to_dict()
    bundle = {
        "version": "clasificador-tormenta-v1",
        "model": final,
        "features": FEATURES,
        "classes": CLASSES,
        "medians": {k: float(v) for k, v in medians.items()},
        "entrenado_utc": datetime.now(timezone.utc).isoformat(),
    }
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE, compress=3)

    report = {
        "version": bundle["version"],
        "generado_utc": bundle["entrenado_utc"],
        "ventana_etiqueta_horas": WINDOW_HOURS,
        "umbrales_proxy": {
            CLASSES[k]: {
                "rafaga_kmh_con_lluvia": v[0],
                "lluvia_1h_mm": v[1],
                "lluvia_ventana_mm": v[2],
            }
            for k, v in THRESHOLDS.items()
        },
        "filas_totales": int(len(y)),
        "distribucion_total": {
            CLASSES[i]: int((y == i).sum()) for i in range(len(CLASSES))
        },
        "corte_validacion": f"entrenamiento < {TEST_FROM_YEAR}; prueba >= {TEST_FROM_YEAR}",
        "validacion": metrics_test,
        "indice_medio_prueba": round(float(baseline_index.mean()), 4),
        "eventos_documentados": event_checks,
        "limitaciones": [
            "Probabilidades NO calibradas: se publican como indices.",
            "La ventana de la etiqueta incluye la hora actual, por lo que "
            "parte del acierto es persistencia (lluvia/rafaga presentes).",
            "Pocos eventos severos documentados; la clase severa es la "
            "menos confiable.",
            "El historico horario es una reconstruccion y subestima picos "
            "convectivos.",
            "No hay etiquetas de clase asociadas a radar RMA10 todavia: el "
            "radar decide si el sistema amenaza a Bahia Blanca, el "
            "historico decide cuan fuerte es el ambiente.",
        ],
        "reentrenar": "python scripts/entrenar_clasificador_tormenta.py",
    }
    METRICS.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    AUDIT.write_text(
        json.dumps(
            {
                "generado_utc": bundle["entrenado_utc"],
                "fuentes": audit_sources(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(report["distribucion_total"], indent=1))
    print(json.dumps(metrics_test, indent=1)[:1500])
    print(json.dumps(event_checks, indent=1))


if __name__ == "__main__":
    main()
