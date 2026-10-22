"""
Clasificador historico de tormentas para Bahia Blanca.

Clases: sin_tormenta / tormenta / tormenta_fuerte / tormenta_severa.

Se entrena con scripts/entrenar_clasificador_tormenta.py a partir del
historico horario 1980-2026 y del catalogo de eventos severos. Este modulo
es compartido por el entrenamiento y por el Storm Engine para que ambos
construyan exactamente las mismas variables.
"""

import math
from pathlib import Path

CLASSES = (
    "sin_tormenta",
    "tormenta",
    "tormenta_fuerte",
    "tormenta_severa",
)

FEATURES = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_spread",
    "pressure_msl",
    "wind_speed_10m",
    "wind_u",
    "wind_v",
    "wind_gusts_10m",
    "gust_excess",
    "precipitation",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
)

MODEL_FILE = (
    Path(__file__).resolve().parents[1]
    / "modelo"
    / "climaar_clasificador_tormenta_v1.joblib"
)

# Peso de cada clase para el indice 0-1 de severidad esperada.
CLASS_WEIGHTS = (0.0, 0.33, 0.66, 1.0)


def _num(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def feature_values(env, when):
    """Variables del clasificador a partir de un ambiente y un datetime UTC."""

    env = env or {}
    temp = _num(env.get("temperature_2m"))
    dew = _num(env.get("dew_point_2m"))
    speed = _num(env.get("wind_speed_10m"))
    direction = _num(env.get("wind_direction_10m"))
    gust = _num(env.get("wind_gusts_10m"))

    values = {
        "temperature_2m": temp,
        "relative_humidity_2m": _num(env.get("relative_humidity_2m")),
        "pressure_msl": _num(env.get("pressure_msl")),
        "wind_speed_10m": speed,
        "wind_gusts_10m": gust,
        "precipitation": _num(env.get("precipitation")),
        "dew_spread": (
            temp - dew if temp is not None and dew is not None else None
        ),
        "gust_excess": (
            gust - speed if gust is not None and speed is not None else None
        ),
    }

    if speed is not None and direction is not None:
        rad = math.radians(direction)
        values["wind_u"] = speed * math.sin(rad)
        values["wind_v"] = speed * math.cos(rad)
    else:
        values["wind_u"] = values["wind_v"] = None

    hour = when.hour + when.minute / 60.0
    values["hour_sin"] = math.sin(2 * math.pi * hour / 24)
    values["hour_cos"] = math.cos(2 * math.pi * hour / 24)
    values["month_sin"] = math.sin(2 * math.pi * (when.month - 1) / 12)
    values["month_cos"] = math.cos(2 * math.pi * (when.month - 1) / 12)
    return values


_BUNDLE = None


def load_bundle():
    global _BUNDLE

    if _BUNDLE is not None:
        return _BUNDLE

    try:
        import joblib

        if MODEL_FILE.is_file():
            bundle = joblib.load(MODEL_FILE)
            if (
                isinstance(bundle, dict)
                and bundle.get("model") is not None
                and tuple(bundle.get("features", ())) == FEATURES
            ):
                _BUNDLE = bundle
    except Exception:
        _BUNDLE = None

    return _BUNDLE


def predict(env, when):
    """
    Devuelve None si no hay modelo o ambiente. Si no, un diccionario con
    la clase mas probable, los indices por clase (no calibrados) y un
    indice 0-1 de severidad esperada segun el historico.
    """

    bundle = load_bundle()

    if bundle is None or not env:
        return None

    values = feature_values(env, when)
    medians = bundle.get("medians", {})
    row = []
    missing = 0

    for name in FEATURES:
        value = values.get(name)
        if value is None:
            value = medians.get(name, 0.0)
            missing += 1
        row.append(float(value))

    if missing > 4:
        return None

    try:
        import pandas as pd

        proba = bundle["model"].predict_proba(
            pd.DataFrame([row], columns=list(FEATURES))
        )[0]
        classes = list(bundle["model"].classes_)
    except Exception:
        return None

    by_class = {name: 0.0 for name in CLASSES}
    for index, label in enumerate(classes):
        by_class[CLASSES[int(label)]] = float(proba[index])

    index_value = sum(
        by_class[name] * weight
        for name, weight in zip(CLASSES, CLASS_WEIGHTS)
    )
    best = max(by_class, key=by_class.get)

    return {
        "clase": best,
        "indices": {k: round(v, 4) for k, v in by_class.items()},
        "indice_severidad_historica": round(index_value, 4),
        "variables_faltantes": missing,
        "modelo": bundle.get("version"),
        "calibrado": False,
    }
