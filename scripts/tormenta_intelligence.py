import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# CLIMAAR INTELLIGENCE
# MOTOR DE INTELIGENCIA DE TORMENTAS
#
# Version 1.0.0
#
# Consume:
#   - Radar V7 RainViewer
#   - Estado meteorologico fusionado
#   - Ambiente ECMWF futuro
#
# Produce:
#   - Probabilidad de tormenta
#   - Probabilidad de llegada a Bahia Blanca
#   - Fortalecimiento / debilitamiento
#   - ETA
#   - Duracion estimada
#   - Peligros
#   - Clasificacion de tormenta
#   - Severidad ClimaAR
#   - Confianza
#   - Historial para futura calibracion / ML
#
# IMPORTANTE:
# Esta version es un motor fisico-estadistico inicial.
# No finge ser un modelo ML entrenado cuando todavia no
# existe suficiente historico etiquetado de tormentas.
# ============================================================


VERSION = "1.0.0"

BASE_DIR = Path(__file__).resolve().parents[1]

IA_DIR = BASE_DIR / "data" / "ia"

RADAR_FILE = (
    BASE_DIR
    / "data"
    / "radar"
    / "radar_nowcast.json"
)

FUSION_FILE = (
    IA_DIR
    / "estado_meteorologico.json"
)

OUTPUT_FILE = (
    IA_DIR
    / "inteligencia_tormenta.json"
)

HISTORY_FILE = (
    IA_DIR
    / "historial_inteligencia_tormenta.csv"
)


LAT = -38.71
LON = -62.26


# ============================================================
# HISTORICO
# ============================================================

HISTORY_FIELDS = [

    "timestamp_utc",

    "estado",

    "severidad",

    "prob_tormenta",

    "prob_llegada_bahia",

    "prob_fortalecimiento",

    "prob_debilitamiento",

    "prob_lluvia_fuerte",

    "prob_viento_fuerte",

    "prob_granizo",

    "prob_rayo",

    "eta_minutos",

    "duracion_estimada_minutos",

    "velocidad_kmh",

    "direccion_grados",

    "cape",

    "humedad",

    "rafaga_kmh",

    "calidad_datos",

    "confianza_modelo",
]


# ============================================================
# UTILIDADES
# ============================================================

def now_utc():

    return datetime.now(
        timezone.utc
    ).isoformat()


def finite(
    value,
    default=None
):

    try:

        if value is None:
            return default

        value = float(value)

        if math.isfinite(value):
            return value

        return default

    except Exception:

        return default


def clamp(
    value,
    low=0.0,
    high=1.0
):

    value = finite(
        value,
        0.0
    )

    return max(
        low,
        min(
            high,
            value
        )
    )


def load_json(
    path
):

    try:

        with path.open(
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return None


def save_json(
    path,
    data
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        )

    temp.replace(
        path
    )


# ============================================================
# FECHAS
# ============================================================

def parse_dt(
    value
):

    if not value:
        return None

    try:

        text = str(
            value
        ).strip()

        if text.endswith(
            "Z"
        ):

            text = (
                text[:-1]
                + "+00:00"
            )

        dt = datetime.fromisoformat(
            text
        )

        if dt.tzinfo is None:

            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt.astimezone(
            timezone.utc
        )

    except Exception:

        return None


def nearest_record(
    records,
    target=None
):

    if not records:
        return None

    if target is None:

        target = datetime.now(
            timezone.utc
        )

    best = None

    best_diff = None

    for row in records:

        dt = parse_dt(
            row.get(
                "tiempo_utc"
            )
        )

        if dt is None:
            continue

        diff = abs(
            (
                dt
                - target
            ).total_seconds()
        )

        if (
            best_diff is None
            or
            diff < best_diff
        ):

            best = row

            best_diff = diff

    return best


# ============================================================
# ENTRADAS
# ============================================================

def load_inputs():

    fusion = load_json(
        FUSION_FILE
    )

    radar = load_json(
        RADAR_FILE
    )

    if not isinstance(
        fusion,
        dict
    ):

        raise RuntimeError(
            "No existe estado_meteorologico.json valido."
        )

    if not isinstance(
        radar,
        dict
    ):

        radar = {}

    return (
        fusion,
        radar
    )


# ============================================================
# AMBIENTE ACTUAL
# ============================================================

def current_environment(
    fusion
):

    actual = fusion.get(
        "ambiente_actual"
    )

    if isinstance(
        actual,
        dict
    ):

        return actual

    future = fusion.get(
        "ambiente_futuro",
        {}
    )

    if not isinstance(
        future,
        dict
    ):

        return None

    points = future.get(
        "puntos",
        []
    )

    if not isinstance(
        points,
        list
    ):

        return None

    for point in points:

        if (
            point.get(
                "punto"
            )
            ==
            "BB_CENTRO"
        ):

            return nearest_record(
                point.get(
                    "datos",
                    []
                )
            )

    return None


# ============================================================
# AMBIENTE FUTURO
# ============================================================

def future_points(
    fusion
):

    future = fusion.get(
        "ambiente_futuro",
        {}
    )

    if not isinstance(
        future,
        dict
    ):

        return []

    points = future.get(
        "puntos",
        []
    )

    if not isinstance(
        points,
        list
    ):

        return []

    return points


def environment_along_projection(
    points,
    projection
):

    if not projection:
        return None

    projected_lat = finite(
        projection.get(
            "latitud"
        ),
        LAT
    )

    projected_lon = finite(
        projection.get(
            "longitud"
        ),
        LON
    )

    minutes = finite(
        projection.get(
            "minutos"
        ),
        0.0
    )

    target = datetime.now(
        timezone.utc
    )

    if minutes:

        target = datetime.fromtimestamp(
            target.timestamp()
            +
            minutes * 60.0,
            timezone.utc
        )

    best = None

    best_score = None

    for point in points:

        if not isinstance(
            point,
            dict
        ):

            continue

        point_lat = finite(
            point.get(
                "latitud"
            ),
            LAT
        )

        point_lon = finite(
            point.get(
                "longitud"
            ),
            LON
        )

        spatial = math.hypot(
            point_lat
            -
            projected_lat,

            point_lon
            -
            projected_lon
        )

        row = nearest_record(
            point.get(
                "datos",
                []
            ),
            target
        )

        if row is None:
            continue

        dt = parse_dt(
            row.get(
                "tiempo_utc"
            )
        )

        if dt is None:
            continue

        temporal = (
            abs(
                (
                    dt
                    -
                    target
                ).total_seconds()
            )
            / 3600.0
        )

        score = (
            spatial * 3.0
            +
            temporal
        )

        if (
            best_score is None
            or
            score < best_score
        ):

            best = row

            best_score = score

    return best


# ============================================================
# RADAR
# ============================================================

def radar_summary(
    radar
):

    nowcast = radar.get(
        "nowcast",
        {}
    )

    if not isinstance(
        nowcast,
        dict
    ):

        nowcast = {}

    return nowcast


def projection_map(
    nowcast
):

    projections = nowcast.get(
        "proyecciones",
        {}
    )

    if not isinstance(
        projections,
        dict
    ):

        return {}

    return projections


def principal_core(
    nowcast
):

    cores = nowcast.get(
        "nucleos",
        []
    )

    if not isinstance(
        cores,
        list
    ):

        return None

    principal_id = nowcast.get(
        "nucleo_principal_id"
    )

    if principal_id is not None:

        for core in cores:

            if (
                str(
                    core.get(
                        "id"
                    )
                )
                ==
                str(
                    principal_id
                )
            ):

                return core

    if cores:

        return cores[0]

    return None


# ============================================================
# POTENCIAL CONVECTIVO DEL AMBIENTE
# ============================================================

def probability_from_environment(
    env
):

    if not env:

        return (
            0.05,
            {
                "datos_ambientales":
                    "ausentes"
            }
        )

    cape = finite(
        env.get(
            "cape"
        ),
        0.0
    )

    humidity = finite(
        env.get(
            "relative_humidity_2m"
        ),
        0.0
    )

    dew = finite(
        env.get(
            "dew_point_2m"
        )
    )

    temperature = finite(
        env.get(
            "temperature_2m"
        )
    )

    showers = finite(
        env.get(
            "showers"
        ),
        0.0
    )

    rain = finite(
        env.get(
            "rain"
        ),
        0.0
    )

    precipitation = finite(
        env.get(
            "precipitation"
        ),
        0.0
    )

    cloud = finite(
        env.get(
            "cloud_cover"
        ),
        0.0
    )

    score = 0.0

    reasons = []

    if cape >= 1500:

        score += 0.35

        reasons.append(
            "CAPE alto"
        )

    elif cape >= 800:

        score += 0.25

        reasons.append(
            "CAPE moderado-alto"
        )

    elif cape >= 300:

        score += 0.12

        reasons.append(
            "CAPE presente"
        )

    if humidity >= 75:

        score += 0.15

        reasons.append(
            "humedad alta"
        )

    elif humidity >= 60:

        score += 0.08

    if (
        dew is not None
        and
        temperature is not None
    ):

        spread = (
            temperature
            -
            dew
        )

        if spread <= 8:

            score += 0.15

            reasons.append(
                "capa baja humeda"
            )

        elif spread <= 12:

            score += 0.07

    if (
        showers > 0
        or
        rain > 0
        or
        precipitation > 0
    ):

        score += 0.18

        reasons.append(
            "precipitacion prevista"
        )

    if cloud >= 80:

        score += 0.05

    return (
        clamp(score),
        {
            "razones":
                reasons
        }
    )


# ============================================================
# PROBABILIDAD DE TORMENTA
# ============================================================

def active_storm_probability(
    nowcast,
    env
):

    activity = bool(
        nowcast.get(
            "actividad"
        )
    )

    area = finite(
        nowcast.get(
            "area_px"
        ),
        0.0
    )

    trend = nowcast.get(
        "fortalecimiento"
    )

    environment_probability, _ = (
        probability_from_environment(
            env
        )
    )

    if (
        not activity
        or
        area <= 0
    ):

        return clamp(
            environment_probability
            * 0.8
        )

    probability = 0.45

    if area >= 500:

        probability += 0.15

    if area >= 2000:

        probability += 0.10

    if (
        trend
        ==
        "fortaleciendose"
    ):

        probability += 0.12

    elif (
        trend
        ==
        "debilitandose"
    ):

        probability -= 0.08

    probability += (
        environment_probability
        * 0.20
    )

    return clamp(
        probability,
        0.20,
        0.98
    )


# ============================================================
# PROBABILIDAD DE IMPACTO
# ============================================================

def impact_probability(
    nowcast
):

    if not nowcast.get(
        "actividad"
    ):

        return 0.0

    toward = bool(
        nowcast.get(
            "movimiento_hacia_bahia"
        )
    )

    distance = finite(
        nowcast.get(
            "distancia_km"
        )
    )

    eta = finite(
        nowcast.get(
            "eta_minutos"
        )
    )

    confidence = clamp(
        nowcast.get(
            "confianza_movimiento"
        )
    )

    if not toward:

        return (
            0.10
            *
            confidence
        )

    proximity = 0.25

    if distance is not None:

        proximity = clamp(
            1.0
            -
            distance / 180.0,
            0.15,
            1.0
        )

    if eta is not None:

        if eta <= 120:

            proximity += 0.15

        if eta <= 60:

            proximity += 0.10

    return clamp(
        0.35
        +
        0.45 * proximity
        +
        0.20 * confidence
    )


# ============================================================
# FORTALECIMIENTO
# ============================================================

def strengthening_probability(
    nowcast,
    env
):

    if not nowcast.get(
        "actividad"
    ):

        return None

    trend = nowcast.get(
        "fortalecimiento"
    )

    change = finite(
        nowcast.get(
            "cambio_area_pct"
        )
    )

    environment_probability, _ = (
        probability_from_environment(
            env
        )
    )

    probability = 0.35

    if (
        trend
        ==
        "fortaleciendose"
    ):

        probability += 0.30

    elif (
        trend
        ==
        "debilitandose"
    ):

        probability -= 0.25

    if change is not None:

        if change >= 30:

            probability += 0.15

        elif change >= 15:

            probability += 0.08

        elif change <= -30:

            probability -= 0.12

    probability += (
        environment_probability
        -
        0.25
    ) * 0.30

    return clamp(
        probability
    )


# ============================================================
# PELIGROS
# ============================================================

def hazard_probabilities(
    nowcast,
    env
):

    activity = bool(
        nowcast.get(
            "actividad"
        )
    )

    if not activity:

        return {

            "lluvia_fuerte":
                0.02,

            "viento_fuerte":
                0.02,

            "granizo":
                0.01,

            "rayo":
                0.01,
        }

    area = finite(
        nowcast.get(
            "area_px"
        ),
        0.0
    )

    trend = nowcast.get(
        "fortalecimiento"
    )

    cape = finite(
        env.get(
            "cape"
        )
        if env
        else None,
        0.0
    )

    gust = finite(
        env.get(
            "wind_gusts_10m"
        )
        if env
        else None,
        0.0
    )

    rain = finite(
        env.get(
            "rain"
        )
        if env
        else None,
        0.0
    )

    showers = finite(
        env.get(
            "showers"
        )
        if env
        else None,
        0.0
    )

    lightning = finite(
        env.get(
            "lightning_density"
        )
        if env
        else None,
        0.0
    )

    area_factor = clamp(
        area / 4000.0
    )

    # ----------------------------------------
    # LLUVIA FUERTE
    # ----------------------------------------

    heavy_rain = (
        0.12
        +
        0.25 * area_factor
    )

    if (
        rain >= 5
        or
        showers >= 5
    ):

        heavy_rain += 0.25

    if (
        trend
        ==
        "fortaleciendose"
    ):

        heavy_rain += 0.10

    # ----------------------------------------
    # VIENTO
    # ----------------------------------------

    strong_wind = (
        0.08
        +
        0.20 * area_factor
    )

    if gust >= 60:

        strong_wind += 0.30

    elif gust >= 45:

        strong_wind += 0.18

    if (
        trend
        ==
        "fortaleciendose"
    ):

        strong_wind += 0.08

    # ----------------------------------------
    # GRANIZO
    # ----------------------------------------

    hail = (
        0.02
        +
        0.18 * area_factor
    )

    if cape >= 1500:

        hail += 0.15

    elif cape >= 800:

        hail += 0.08

    # ----------------------------------------
    # RAYOS
    # ----------------------------------------

    lightning_probability = (
        0.03
        +
        0.18 * area_factor
    )

    if cape >= 800:

        lightning_probability += 0.18

    if lightning > 0:

        lightning_probability += min(
            0.45,
            lightning / 5.0
        )

    return {

        "lluvia_fuerte":
            clamp(
                heavy_rain
            ),

        "viento_fuerte":
            clamp(
                strong_wind
            ),

        "granizo":
            clamp(
                hail
            ),

        "rayo":
            clamp(
                lightning_probability
            ),
    }


# ============================================================
# CLASIFICACION
# ============================================================

def classify_storm(
    nowcast,
    env,
    hazards
):

    if not nowcast.get(
        "actividad"
    ):

        return (
            "sin_tormenta_activa"
        )

    trend = nowcast.get(
        "fortalecimiento"
    )

    cape = finite(
        env.get(
            "cape"
        )
        if env
        else None,
        0.0
    )

    rain = finite(
        env.get(
            "rain"
        )
        if env
        else None,
        0.0
    )

    gust = finite(
        env.get(
            "wind_gusts_10m"
        )
        if env
        else None,
        0.0
    )

    severe_score = max(
        hazards[
            "viento_fuerte"
        ],

        hazards[
            "granizo"
        ],

        hazards[
            "rayo"
        ]
    )

    if (
        severe_score >= 0.70
        and
        (
            cape >= 800
            or
            gust >= 60
        )
    ):

        return (
            "tormenta_severa_potencial"
        )

    if (
        trend
        ==
        "fortaleciendose"
        and
        (
            cape >= 500
            or
            rain > 0
        )
    ):

        return (
            "tormenta_convectiva_en_desarrollo"
        )

    if (
        rain > 0
        or
        hazards[
            "lluvia_fuerte"
        ] >= 0.30
    ):

        return (
            "tormenta_con_lluvia"
        )

    return (
        "tormenta_activa"
    )


# ============================================================
# SEVERIDAD CLIMAAR
# ============================================================

def severity(
    nowcast,
    hazards,
    confidence
):

    if not nowcast.get(
        "actividad"
    ):

        return "verde"

    max_hazard = max(
        hazards.values()
    )

    impact = impact_probability(
        nowcast
    )

    raw = (
        0.55
        * max_hazard
        +
        0.30
        * impact
        +
        0.15
        * confidence
    )

    if raw >= 0.78:

        level = "rojo"

    elif raw >= 0.55:

        level = "naranja"

    elif raw >= 0.28:

        level = "amarillo"

    else:

        level = "verde"

    # --------------------------------------------------------
    # LIMITACION CIENTIFICA ACTUAL
    #
    # El V7 todavía no extrae dBZ cuantitativo ni tiene
    # observación directa de rayos.
    #
    # Por eso NO permitimos rojo en esta primera versión.
    # Cuando integremos reflectividad + rayos + mejores
    # variables convectivas, se habilita la clasificación roja.
    # --------------------------------------------------------

    if level == "rojo":

        level = "naranja"

    return level


# ============================================================
# DURACION
# ============================================================

def duration_estimate(
    nowcast,
    env,
    impact
):

    if not nowcast.get(
        "actividad"
    ):

        return None

    speed = finite(
        nowcast.get(
            "velocidad_kmh"
        )
    )

    distance = finite(
        nowcast.get(
            "distancia_km"
        )
    )

    if (
        speed is not None
        and
        speed > 5
        and
        distance is not None
    ):

        passage = max(
            20.0,
            min(
                180.0,
                60.0
                *
                80.0
                /
                speed
            )
        )

    else:

        passage = 60.0

    rain = finite(
        env.get(
            "rain"
        )
        if env
        else None,
        0.0
    )

    showers = finite(
        env.get(
            "showers"
        )
        if env
        else None,
        0.0
    )

    if (
        rain > 0
        or
        showers > 0
    ):

        passage += 20.0

    if impact < 0.35:

        passage *= 0.7

    return int(
        round(
            max(
                20.0,
                min(
                    240.0,
                    passage
                )
            )
        )
    )


# ============================================================
# CONFIANZA
# ============================================================

def confidence_score(
    fusion,
    nowcast,
    env
):

    quality = finite(
        fusion.get(
            "calidad_datos",
            {}
        ).get(
            "score"
        ),
        0.0
    ) / 100.0

    radar_confidence = clamp(
        nowcast.get(
            "confianza_movimiento"
        )
    )

    has_environment = (
        1.0
        if env
        else 0.0
    )

    # No superar 70% en esta etapa.
    #
    # Falta todavía:
    #   - dBZ cuantitativo
    #   - rayos observados
    #   - dataset histórico etiquetado
    #   - calibración estadística
    #   - modelo ML entrenado
    #

    return round(
        min(
            0.70,
            (
                0.45
                * quality
                +
                0.35
                * radar_confidence
                +
                0.20
                * has_environment
            )
        ),
        2
    )


# ============================================================
# RAZONES
# ============================================================

def build_reasons(
    nowcast,
    env,
    hazards,
    class_name,
    impact
):

    reasons = []

    if not nowcast.get(
        "actividad"
    ):

        reasons.append(
            "No se detecta precipitacion activa en el radar V7."
        )

    else:

        reasons.append(
            "Existe actividad de precipitacion en el radar V7."
        )

        if nowcast.get(
            "movimiento_hacia_bahia"
        ):

            reasons.append(
                "El movimiento estimado apunta hacia Bahia Blanca."
            )

        if (
            nowcast.get(
                "fortalecimiento"
            )
            ==
            "fortaleciendose"
        ):

            reasons.append(
                "El area radar esta aumentando."
            )

        elif (
            nowcast.get(
                "fortalecimiento"
            )
            ==
            "debilitandose"
        ):

            reasons.append(
                "El area radar esta disminuyendo."
            )

    cape = finite(
        env.get(
            "cape"
        )
        if env
        else None
    )

    gust = finite(
        env.get(
            "wind_gusts_10m"
        )
        if env
        else None
    )

    rain = finite(
        env.get(
            "rain"
        )
        if env
        else None
    )

    if cape is not None:

        reasons.append(
            f"CAPE ECMWF: {cape:.0f} J/kg."
        )

    if gust is not None:

        reasons.append(
            f"Rafaga prevista ECMWF: {gust:.1f} km/h."
        )

    if (
        rain is not None
        and
        rain > 0
    ):

        reasons.append(
            f"Lluvia prevista: {rain:.1f} mm."
        )

    reasons.append(
        "Probabilidad estimada de impacto en Bahia Blanca: "
        f"{impact:.0%}."
    )

    reasons.append(
        "La severidad es una estimacion independiente y no reemplaza avisos oficiales."
    )

    if (
        class_name
        ==
        "sin_tormenta_activa"
    ):

        reasons.append(
            "El motor no inventa una tormenta cuando el radar no la observa."
        )

    return reasons


# ============================================================
# HISTORICO
# ============================================================

def append_history(
    result
):

    HISTORY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    exists = (
        HISTORY_FILE.exists()
    )

    row = {

        "timestamp_utc":
            result[
                "actualizado_utc"
            ],

        "estado":
            result[
                "estado"
            ],

        "severidad":
            result[
                "severidad"
            ],

        "prob_tormenta":
            result[
                "probabilidades"
            ][
                "tormenta"
            ],

        "prob_llegada_bahia":
            result[
                "probabilidades"
            ][
                "llegada_bahia"
            ],

        "prob_fortalecimiento":
            result[
                "probabilidades"
            ][
                "fortalecimiento"
            ],

        "prob_debilitamiento":
            result[
                "probabilidades"
            ][
                "debilitamiento"
            ],

        "prob_lluvia_fuerte":
            result[
                "peligros"
            ][
                "lluvia_fuerte"
            ],

        "prob_viento_fuerte":
            result[
                "peligros"
            ][
                "viento_fuerte"
            ],

        "prob_granizo":
            result[
                "peligros"
            ][
                "granizo"
            ],

        "prob_rayo":
            result[
                "peligros"
            ][
                "rayo"
            ],

        "eta_minutos":
            result[
                "eta_minutos"
            ],

        "duracion_estimada_minutos":
            result[
                "duracion_estimada_minutos"
            ],

        "velocidad_kmh":
            result[
                "radar"
            ].get(
                "velocidad_kmh"
            ),

        "direccion_grados":
            result[
                "radar"
            ].get(
                "direccion_grados"
            ),

        "cape":
            (
                result[
                    "ambiente"
                ].get(
                    "cape"
                )
                if result[
                    "ambiente"
                ]
                else None
            ),

        "humedad":
            (
                result[
                    "ambiente"
                ].get(
                    "relative_humidity_2m"
                )
                if result[
                    "ambiente"
                ]
                else None
            ),

        "rafaga_kmh":
            (
                result[
                    "ambiente"
                ].get(
                    "wind_gusts_10m"
                )
                if result[
                    "ambiente"
                ]
                else None
            ),

        "calidad_datos":
            result[
                "calidad_datos"
            ],

        "confianza_modelo":
            result[
                "confianza_modelo"
            ],
    }

    with HISTORY_FILE.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=HISTORY_FIELDS
        )

        if not exists:

            writer.writeheader()

        writer.writerow(
            row
        )


# ============================================================
# MAIN
# ============================================================

def main():

    fusion, radar_file = (
        load_inputs()
    )

    nowcast = radar_summary(
        radar_file
    )

    environment = (
        current_environment(
            fusion
        )
    )

    future = future_points(
        fusion
    )

    # --------------------------------------------------------
    # PROBABILIDADES
    # --------------------------------------------------------

    storm_probability = (
        active_storm_probability(
            nowcast,
            environment
        )
    )

    impact_probability_value = (
        impact_probability(
            nowcast
        )
    )

    strengthening = (
        strengthening_probability(
            nowcast,
            environment
        )
    )

    if strengthening is None:

        weakening = None

    else:

        weakening = clamp(
            1.0
            -
            strengthening
        )

    # --------------------------------------------------------
    # PELIGROS
    # --------------------------------------------------------

    hazards = (
        hazard_probabilities(
            nowcast,
            environment
        )
    )

    # --------------------------------------------------------
    # CLASIFICACION
    # --------------------------------------------------------

    storm_class = (
        classify_storm(
            nowcast,
            environment,
            hazards
        )
    )

    # --------------------------------------------------------
    # CONFIANZA
    # --------------------------------------------------------

    confidence = (
        confidence_score(
            fusion,
            nowcast,
            environment
        )
    )

    # --------------------------------------------------------
    # SEVERIDAD
    # --------------------------------------------------------

    severity_level = (
        severity(
            nowcast,
            hazards,
            confidence
        )
    )

    # --------------------------------------------------------
    # DURACION
    # --------------------------------------------------------

    duration = (
        duration_estimate(
            nowcast,
            environment,
            impact_probability_value
        )
    )

    # --------------------------------------------------------
    # AMBIENTE SOBRE TRAYECTORIA
    # --------------------------------------------------------

    trajectory_environment = {}

    for (
        key,
        projection
    ) in projection_map(
        nowcast
    ).items():

        if not isinstance(
            projection,
            dict
        ):

            continue

        projected_environment = (
            environment_along_projection(
                future,
                projection
            )
        )

        if projected_environment:

            trajectory_environment[
                str(key)
            ] = {

                "minutos":
                    projection.get(
                        "minutos"
                    ),

                "latitud":
                    projection.get(
                        "latitud"
                    ),

                "longitud":
                    projection.get(
                        "longitud"
                    ),

                "ambiente":
                    projected_environment,
            }

    # --------------------------------------------------------
    # RAZONES
    # --------------------------------------------------------

    reasons = build_reasons(
        nowcast,
        environment,
        hazards,
        storm_class,
        impact_probability_value
    )

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    result = {

        "version":
            VERSION,

        "motor":
            "ClimaAR Intelligence - Storm Engine",

        "actualizado_utc":
            now_utc(),

        "ubicacion": {

            "ciudad":
                "Bahia Blanca",

            "latitud":
                LAT,

            "longitud":
                LON,
        },

        "estado":
            storm_class,

        "severidad":
            severity_level,

        "probabilidades": {

            "tormenta":
                round(
                    storm_probability,
                    3
                ),

            "llegada_bahia":
                round(
                    impact_probability_value,
                    3
                ),

            "fortalecimiento":
                (
                    round(
                        strengthening,
                        3
                    )
                    if strengthening is not None
                    else None
                ),

            "debilitamiento":
                (
                    round(
                        weakening,
                        3
                    )
                    if weakening is not None
                    else None
                ),
        },

        "peligros": {

            key:
                round(
                    value,
                    3
                )

            for key, value
            in hazards.items()
        },

        "eta_minutos":
            nowcast.get(
                "eta_minutos"
            ),

        "duracion_estimada_minutos":
            duration,

        "radar":
            nowcast,

        "ambiente":
            environment,

        "ambiente_en_trayectoria":
            trajectory_environment,

        "calidad_datos":
            finite(
                fusion.get(
                    "calidad_datos",
                    {}
                ).get(
                    "score"
                ),
                0
            ),

        "confianza_modelo":
            confidence,

        "limitaciones": [

            "El radar V7 actual no extrae dBZ cuantitativo.",

            "No hay observacion directa de rayos integrada en esta etapa.",

            "No se debe interpretar como alerta oficial.",

            "Las probabilidades son estimaciones calibrables y no certezas.",

        ],

        "razones":
            reasons,
    }

    # --------------------------------------------------------
    # GUARDAR
    # --------------------------------------------------------

    save_json(
        OUTPUT_FILE,
        result
    )

    append_history(
        result
    )

    # --------------------------------------------------------
    # CONSOLA
    # --------------------------------------------------------

    print(
        "============================================"
    )

    print(
        "ClimaAR Storm Engine"
    )

    print(
        f"Version {VERSION}"
    )

    print(
        "============================================"
    )

    print(
        f"Estado: {storm_class}"
    )

    print(
        f"Severidad: {severity_level}"
    )

    print(
        "Probabilidad tormenta: "
        f"{storm_probability:.0%}"
    )

    print(
        "Impacto Bahia Blanca: "
        f"{impact_probability_value:.0%}"
    )

    if strengthening is not None:

        print(
            "Fortalecimiento: "
            f"{strengthening:.0%}"
        )

    print(
        "Confianza modelo: "
        f"{confidence:.0%}"
    )

    print(
        "STORM ENGINE OK"
    )

    return 0


if __name__ == "__main__":

    sys.exit(
        main()
  )
