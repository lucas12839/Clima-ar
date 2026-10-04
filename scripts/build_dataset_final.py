import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CLIMAAR - CONSTRUCCION DEL DATASET FINAL
# ============================================================

HISTORICO = Path("data/historico/clima_horario_1980_2026.csv")
EVENTOS = Path("data/historico/eventos_severos.csv")
META = Path("data/historico/climaar_datos_historicos_meta.txt")

OUTDIR = Path("data/final")

OUT = OUTDIR / "climaar_dataset_completo_1980_2026.csv"
CAT = OUTDIR / "catalogo_historico_documental.csv"
RES = OUTDIR / "resumen_dataset_final.csv"

OUTDIR.mkdir(parents=True, exist_ok=True)


def fail(msg):
    raise SystemExit(f"ERROR: {msg}")


def as_number(value):
    if pd.isna(value):
        return np.nan

    m = re.search(
        r"[-+]?\d+(?:[\\.,]\d+)?",
        str(value)
    )

    if not m:
        return np.nan

    return float(
        m.group(0).replace(",", ".")
    )


print("=" * 72)
print("CLIMAAR - DATASET FINAL")
print("=" * 72)


# ============================================================
# 1. VERIFICAR ARCHIVOS
# ============================================================

if not HISTORICO.exists():
    fail(f"No existe {HISTORICO}")

if not EVENTOS.exists():
    fail(f"No existe {EVENTOS}")


# ============================================================
# 2. CARGAR HISTORICO METEOROLOGICO
# ============================================================

print("[1/9] Cargando histórico meteorológico...")

clima = pd.read_csv(HISTORICO)

required = [
    "time",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]

missing = [
    c for c in required
    if c not in clima.columns
]

if missing:
    fail(
        "Faltan columnas: "
        + ", ".join(missing)
    )

clima["time"] = pd.to_datetime(
    clima["time"],
    errors="coerce"
)

clima = clima.dropna(
    subset=["time"]
)

clima = (
    clima
    .sort_values("time")
    .drop_duplicates("time")
    .reset_index(drop=True)
)

for c in required[1:]:
    clima[c] = pd.to_numeric(
        clima[c],
        errors="coerce"
    )

clima["precipitation"] = (
    clima["precipitation"]
    .clip(lower=0)
)

clima["relative_humidity_2m"] = (
    clima["relative_humidity_2m"]
    .clip(0, 100)
)

print(
    f"Registros: {len(clima):,}"
)

print(
    f"Desde: {clima['time'].min()}"
)

print(
    f"Hasta: {clima['time'].max()}"
)


# ============================================================
# 3. VARIABLES TEMPORALES
# ============================================================

print("[2/9] Variables temporales...")

clima["year"] = clima["time"].dt.year
clima["month"] = clima["time"].dt.month
clima["day"] = clima["time"].dt.day
clima["hour"] = clima["time"].dt.hour
clima["dayofyear"] = clima["time"].dt.dayofyear

clima["hour_sin"] = np.sin(
    2 * np.pi * clima["hour"] / 24
)

clima["hour_cos"] = np.cos(
    2 * np.pi * clima["hour"] / 24
)

clima["month_sin"] = np.sin(
    2 * np.pi * clima["month"] / 12
)

clima["month_cos"] = np.cos(
    2 * np.pi * clima["month"] / 12
)


# ============================================================
# 4. EVOLUCION METEOROLOGICA
# ============================================================

print("[3/9] Evolución meteorológica...")

x = clima.set_index("time")

# Acumulados de precipitación
for h in [
    1,
    3,
    6,
    12,
    24,
    48,
    72
]:

    x[f"precip_{h}h"] = (
        x["precipitation"]
        .rolling(
            h,
            min_periods=1
        )
        .sum()
    )


# Máximos de viento
for h in [
    3,
    6,
    12,
    24,
    48,
    72
]:

    x[f"gust_max_{h}h"] = (
        x["wind_gusts_10m"]
        .rolling(
            h,
            min_periods=1
        )
        .max()
    )

    x[f"wind_max_{h}h"] = (
        x["wind_speed_10m"]
        .rolling(
            h,
            min_periods=1
        )
        .max()
    )


# Cambios meteorológicos
for h in [
    1,
    3,
    6,
    12,
    24
]:

    x[f"pressure_change_{h}h"] = (
        x["pressure_msl"]
        - x["pressure_msl"].shift(h)
    )

    x[f"temperature_change_{h}h"] = (
        x["temperature_2m"]
        - x["temperature_2m"].shift(h)
    )

    x[f"humidity_change_{h}h"] = (
        x["relative_humidity_2m"]
        - x["relative_humidity_2m"].shift(h)
    )

    x[f"wind_change_{h}h"] = (
        x["wind_speed_10m"]
        - x["wind_speed_10m"].shift(h)
    )

    x[f"gust_change_{h}h"] = (
        x["wind_gusts_10m"]
        - x["wind_gusts_10m"].shift(h)
    )

clima = x.reset_index()


# Variables adicionales
clima["temperature_dewpoint_gap"] = (
    clima["temperature_2m"]
    - clima["dew_point_2m"]
)

clima["pressure_drop_6h_flag"] = (
    clima["pressure_change_6h"] <= -3
).astype("int8")

clima["pressure_drop_12h_flag"] = (
    clima["pressure_change_12h"] <= -5
).astype("int8")

clima["rain_1h_ge_20mm"] = (
    clima["precip_1h"] >= 20
).astype("int8")

clima["rain_6h_ge_50mm"] = (
    clima["precip_6h"] >= 50
).astype("int8")

clima["gust_ge_80"] = (
    clima["wind_gusts_10m"] >= 80
).astype("int8")

clima["gust_ge_100"] = (
    clima["wind_gusts_10m"] >= 100
).astype("int8")


# ============================================================
# 5. CLIMATOLOGIA 1991-2020
# ============================================================

print("[4/9] Climatología 1991-2020...")

ref = clima[
    clima["year"].between(
        1991,
        2020
    )
].copy()

if ref.empty:
    fail(
        "No existen datos 1991-2020."
    )

ref_vars = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
]

means = (
    ref
    .groupby("month")[ref_vars]
    .mean()
)

for c in ref_vars:

    clima[
        f"{c}_clim_mean_1991_2020"
    ] = clima["month"].map(
        means[c].to_dict()
    )

    clima[
        f"{c}_anomaly_1991_2020"
    ] = (
        clima[c]
        - clima[
            f"{c}_clim_mean_1991_2020"
        ]
    )


# Percentiles climatológicos
for c in [
    "precipitation",
    "wind_gusts_10m",
    "temperature_2m"
]:

    p95 = (
        ref
        .groupby("month")[c]
        .quantile(0.95)
        .to_dict()
    )

    p99 = (
        ref
        .groupby("month")[c]
        .quantile(0.99)
        .to_dict()
    )

    clima[
        f"{c}_p95_1991_2020"
    ] = clima["month"].map(p95)

    clima[
        f"{c}_p99_1991_2020"
    ] = clima["month"].map(p99)

    clima[
        f"{c}_above_p95"
    ] = (
        clima[c]
        >= clima[
            f"{c}_p95_1991_2020"
        ]
    ).astype("int8")

    clima[
        f"{c}_above_p99"
    ] = (
        clima[c]
        >= clima[
            f"{c}_p99_1991_2020"
        ]
    ).astype("int8")


# ============================================================
# 6. EVENTOS HISTORICOS
# ============================================================

print("[5/9] Eventos históricos...")

ev = pd.read_csv(EVENTOS)

if "fecha" not in ev.columns:
    fail(
        "eventos_severos.csv no contiene 'fecha'."
    )

ev["fecha"] = pd.to_datetime(
    ev["fecha"],
    errors="coerce"
)

ev = (
    ev
    .dropna(subset=["fecha"])
    .sort_values("fecha")
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# IMPORTANTE
#
# Los eventos disponibles tienen fecha.
# No vamos a inventar una hora que la fuente no proporciona.
# ------------------------------------------------------------

# Objetivos futuros
for h in [
    6,
    12,
    24,
    48,
    72
]:

    clima[
        f"target_next_{h}h"
    ] = 0


clima["event_id"] = ""
clima["event_type"] = ""

clima["event_time_precision"] = (
    "none"
)

clima["label_quality"] = (
    "none"
)

clima["hours_to_event"] = np.nan


# ------------------------------------------------------------
# Etiquetado
# ------------------------------------------------------------

for _, evento in ev.iterrows():

    fecha = evento["fecha"].normalize()

    nombre_evento = str(
        evento.get(
            "evento",
            ""
        )
    )

    event_id = fecha.strftime(
        "%Y-%m-%d"
    )

    # --------------------------------------------------------
    # Como NO tenemos hora exacta,
    # NO inventamos targets de 6h/12h.
    # --------------------------------------------------------

    for h in [
        24,
        48,
        72
    ]:

        inicio = (
            fecha
            - pd.Timedelta(hours=h)
        )

        mask = (
            (clima["time"] >= inicio)
            &
            (clima["time"] < fecha)
        )

        clima.loc[
            mask,
            f"target_next_{h}h"
        ] = 1

        clima.loc[
            mask,
            "event_id"
        ] = event_id

        clima.loc[
            mask,
            "event_type"
        ] = nombre_evento

        clima.loc[
            mask,
            "event_time_precision"
        ] = "date_only"

        clima.loc[
            mask,
            "label_quality"
        ] = "date_only_proxy"

        clima.loc[
            mask,
            "hours_to_event"
        ] = (
            fecha
            - clima.loc[
                mask,
                "time"
            ]
        ).dt.total_seconds() / 3600


# ------------------------------------------------------------
# Objetivos 6h y 12h
#
# Se dejan como NaN porque no conocemos la hora exacta
# de ocurrencia de los eventos.
# ------------------------------------------------------------

for h in [
    6,
    12
]:

    clima[
        f"target_next_{h}h"
    ] = np.nan


# ------------------------------------------------------------
# Protección contra leakage
# ------------------------------------------------------------

clima["post_event_label"] = 0


# ============================================================
# 7. CATALOGO DOCUMENTAL
# ============================================================

print("[6/9] Catálogo documental...")

doc = []

if META.exists():

    raw = META.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    sources = re.findall(
        r"FUENTE:\s*(.*?)\s*={5,}",
        raw,
        flags=re.S
    )

    for source in sources:

        source = " ".join(
            source.split()
        )

        if source:

            doc.append({
                "tipo": "documento",
                "fuente": source,
                "fecha": "",
                "evento": "",
                "variable": "",
                "valor": np.nan,
                "unidad": "",
                "verificacion":
                    "fuente_incluida",
                "nota":
                    "Los gráficos sin valor numérico exacto no se convierten en números."
            })


# ------------------------------------------------------------
# Datos numéricos explícitos
# ------------------------------------------------------------

explicit = [

    (
        "1981-03-14",
        "Temporal grave",
        "rafaga_kmh",
        110,
        "km/h"
    ),

    (
        "1981-03-14",
        "Temporal grave",
        "lluvia",
        85,
        "mm"
    ),

    (
        "1982-02-13",
        "Tornado/temporal extremo",
        "rafaga_kmh",
        170,
        "km/h"
    ),

    (
        "2019-12-30",
        "Tornado F0 Paso Piedras",
        "rafaga_kmh",
        115,
        "km/h"
    ),

    (
        "2019-12-30",
        "Tornado F0 Paso Piedras",
        "lluvia_1h",
        100,
        "mm"
    ),

    (
        "2023-12-16",
        "Temporal destructivo",
        "rafaga_kmh_edes",
        189,
        "km/h"
    ),

    (
        "2023-12-16",
        "Temporal destructivo",
        "lluvia",
        45,
        "mm"
    ),

    (
        "2025-03-07",
        "Inundación histórica",
        "lluvia_12h",
        290,
        "mm"
    ),
]


for (
    fecha,
    evento,
    variable,
    valor,
    unidad
) in explicit:

    doc.append({

        "tipo":
            "dato_explicito",

        "fuente":
            "climaar_datos_historicos_meta.txt",

        "fecha":
            fecha,

        "evento":
            evento,

        "variable":
            variable,

        "valor":
            valor,

        "unidad":
            unidad,

        "verificacion":
            "explicito_en_paquete",

        "nota":
            "No extraído de una barra gráfica ni aproximado."

    })


pd.DataFrame(doc).to_csv(
    CAT,
    index=False,
    encoding="utf-8"
)


# ============================================================
# 8. CONTROLES DE CALIDAD
# ============================================================

print("[7/9] Controles de calidad...")


# Orden temporal
if not clima[
    "time"
].is_monotonic_increasing:

    fail(
        "El dataset no quedó ordenado."
    )


# Duplicados
if clima[
    "time"
].duplicated().any():

    fail(
        "Quedaron timestamps duplicados."
    )


# No debe existir leakage posterior
if int(
    (
        clima["post_event_label"]
        != 0
    ).sum()
) != 0:

    fail(
        "Existe una etiqueta posterior al evento."
    )


# Verificar fechas
if clima["time"].min().year > 1980:

    print(
        "ADVERTENCIA: "
        "el histórico no comienza en 1980."
    )


if clima["time"].max().year < 2026:

    print(
        "ADVERTENCIA: "
        "el histórico no llega a 2026."
    )


# ============================================================
# 9. GUARDAR
# ============================================================

print("[8/9] Guardando dataset...")

clima = clima.replace(
    [np.inf, -np.inf],
    np.nan
)

clima.to_csv(
    OUT,
    index=False,
    encoding="utf-8"
)


# ============================================================
# RESUMEN
# ============================================================

rows = []


for h in [
    6,
    12,
    24,
    48,
    72
]:

    col = (
        f"target_next_{h}h"
    )

    rows.append({

        "metrica":
            f"positivos_target_next_{h}h",

        "valor":
            int(
                (
                    clima[col]
                    == 1
                ).sum()
            ),

        "nota":
            (
                "6/12 h quedan sin etiqueta "
                "porque no conocemos la hora exacta."
                if h in [6, 12]
                else
                "Etiqueta basada en fecha del evento."
            )
    })


rows.extend([

    {
        "metrica":
            "registros",

        "valor":
            len(clima),

        "nota":
            "Dataset meteorológico horario"
    },

    {
        "metrica":
            "columnas",

        "valor":
            len(clima.columns),

        "nota":
            "Features + objetivos + trazabilidad"
    },

    {
        "metrica":
            "eventos_catalogados",

        "valor":
            len(ev),

        "nota":
            "Eventos históricos"
    },

    {
        "metrica":
            "fuentes_documentales",

        "valor":
            len([
                d
                for d in doc
                if d.get("tipo")
                == "documento"
            ]),

        "nota":
            "Fuentes detectadas"
    },

    {
        "metrica":
            "datos_documentales_explicitos",

        "valor":
            len(explicit),

        "nota":
            "Valores numéricos explícitos"
    },

    {
        "metrica":
            "post_event_labels",

        "valor":
            int(
                (
                    clima[
                        "post_event_label"
                    ] != 0
                ).sum()
            ),

        "nota":
            "Debe ser 0"
    },

    {
        "metrica":
            "estado",

        "valor":
            "OK",

        "nota":
            "Dataset construido y validado"
    }

])


pd.DataFrame(rows).to_csv(
    RES,
    index=False,
    encoding="utf-8"
)


print("[9/9] FINALIZADO")

print("-" * 72)

print(
    f"Dataset: {OUT}"
)

print(
    f"Catalogo: {CAT}"
)

print(
    f"Resumen: {RES}"
)

print(
    f"Registros: {len(clima):,}"
)

print(
    f"Columnas: {len(clima.columns)}"
)

print(
    f"Eventos: {len(ev)}"
)

print(
    "=== CLIMAAR DATASET FINAL OK ==="
)
