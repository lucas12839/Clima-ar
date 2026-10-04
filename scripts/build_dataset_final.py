import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# CLIMAAR - CONSTRUCCION FINAL DEL DATASET
# ============================================================

BASE = Path(".")
HIST = BASE / "data" / "historico"
FINAL = BASE / "data" / "final"

FINAL.mkdir(parents=True, exist_ok=True)

CLIMA_FILE = HIST / "clima_horario_1980_2026.csv"
EVENTOS_FILE = HIST / "eventos_severos.csv"
META_FILE = HIST / "climaar_datos_historicos_meta.txt"

OUTPUT_DATASET = FINAL / "climaar_dataset_completo_1980_2026.csv"
OUTPUT_CATALOGO = FINAL / "catalogo_historico_documental.csv"
OUTPUT_RESUMEN = FINAL / "resumen_dataset_final.csv"


print("=" * 70)
print("CLIMAAR - DATASET FINAL")
print("=" * 70)


# ============================================================
# 1. VERIFICAR ARCHIVOS
# ============================================================

if not CLIMA_FILE.exists():
    raise FileNotFoundError(
        f"No existe: {CLIMA_FILE}"
    )

if not EVENTOS_FILE.exists():
    raise FileNotFoundError(
        f"No existe: {EVENTOS_FILE}"
    )


# ============================================================
# 2. CARGAR HISTORICO METEOROLOGICO
# ============================================================

print("\n[1/8] Cargando histórico meteorológico...")

df = pd.read_csv(CLIMA_FILE)

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
    if c not in df.columns
]

if missing:
    raise ValueError(
        "Faltan columnas en el histórico: "
        + ", ".join(missing)
    )

df["time"] = pd.to_datetime(
    df["time"],
    errors="coerce"
)

df = df.dropna(subset=["time"]).copy()
df = df.sort_values("time")
df = df.drop_duplicates(
    subset=["time"]
).reset_index(drop=True)

print(f"Registros: {len(df):,}")
print(
    f"Desde: {df['time'].min()}"
)
print(
    f"Hasta: {df['time'].max()}"
)


# ============================================================
# 3. LIMPIEZA NUMERICA
# ============================================================

print("\n[2/8] Limpiando variables...")

numeric_cols = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df["precipitation"] = (
    df["precipitation"]
    .clip(lower=0)
)

df["relative_humidity_2m"] = (
    df["relative_humidity_2m"]
    .clip(0, 100)
)


# ============================================================
# 4. VARIABLES TEMPORALES
# ============================================================

print("\n[3/8] Generando variables temporales...")

df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month
df["day"] = df["time"].dt.day
df["hour"] = df["time"].dt.hour
df["dayofyear"] = df["time"].dt.dayofyear

df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)

df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)


# ============================================================
# 5. VARIABLES METEOROLOGICAS MOVILES
# ============================================================

print("\n[4/8] Calculando evolución meteorológica...")

df["precip_1h"] = (
    df["precipitation"]
)

for h in [3, 6, 12, 24, 48, 72]:
    df[f"precip_{h}h"] = (
        df["precipitation"]
        .rolling(h, min_periods=1)
        .sum()
    )

for h in [3, 6, 12, 24, 48, 72]:
    df[f"gust_max_{h}h"] = (
        df["wind_gusts_10m"]
        .rolling(h, min_periods=1)
        .max()
    )

for h in [3, 6, 12, 24, 48, 72]:
    df[f"wind_max_{h}h"] = (
        df["wind_speed_10m"]
        .rolling(h, min_periods=1)
        .max()
    )


# ============================================================
# 6. CAMBIOS METEOROLOGICOS
# ============================================================

print("\n[5/8] Calculando cambios...")

for h in [1, 3, 6, 12, 24]:
    df[f"pressure_change_{h}h"] = (
        df["pressure_msl"]
        - df["pressure_msl"].shift(h)
    )

    df[f"temperature_change_{h}h"] = (
        df["temperature_2m"]
        - df["temperature_2m"].shift(h)
    )

    df[f"humidity_change_{h}h"] = (
        df["relative_humidity_2m"]
        - df["relative_humidity_2m"].shift(h)
    )

    df[f"wind_change_{h}h"] = (
        df["wind_speed_10m"]
        - df["wind_speed_10m"].shift(h)
    )

    df[f"gust_change_{h}h"] = (
        df["wind_gusts_10m"]
        - df["wind_gusts_10m"].shift(h)
    )


# ============================================================
# 7. CLIMATOLOGIA 1991-2020
# ============================================================

print("\n[6/8] Calculando climatología 1991-2020...")

clim = df[
    (df["year"] >= 1991)
    & (df["year"] <= 2020)
].copy()

if len(clim) == 0:
    raise ValueError(
        "No hay datos suficientes para climatología 1991-2020."
    )

clim_month = (
    clim
    .groupby("month")
    .agg(
        clima_temp_media=(
            "temperature_2m",
            "mean"
        ),
        clima_humedad_media=(
            "relative_humidity_2m",
            "mean"
        ),
        clima_presion_media=(
            "pressure_msl",
            "mean"
        ),
        clima_precip_media=(
            "precipitation",
            "mean"
        ),
        clima_viento_media=(
            "wind_speed_10m",
            "mean"
        ),
        clima_rafaga_media=(
            "wind_gusts_10m",
            "mean"
        ),
        clima_precip_p95=(
            "precipitation",
            lambda x: x.quantile(0.95)
        ),
        clima_precip_p99=(
            "precipitation",
            lambda x: x.quantile(0.99)
        ),
        clima_rafaga_p95=(
            "wind_gusts_10m",
            lambda x: x.quantile(0.95)
        ),
        clima_rafaga_p99=(
            "wind_gusts_10m",
            lambda x: x.quantile(0.99)
        ),
        clima_temp_p95=(
            "temperature_2m",
            lambda x: x.quantile(0.95)
        ),
        clima_temp_p05=(
            "temperature_2m",
            lambda x: x.quantile(0.05)
        ),
    )
    .reset_index()
)

df = df.merge(
    clim_month,
    on="month",
    how="left"
)

df["anomalia_temperatura"] = (
    df["temperature_2m"]
    - df["clima_temp_media"]
)

df["anomalia_humedad"] = (
    df["relative_humidity_2m"]
    - df["clima_humedad_media"]
)

df["anomalia_presion"] = (
    df["pressure_msl"]
    - df["clima_presion_media"]
)

df["anomalia_precipitacion"] = (
    df["precipitation"]
    - df["clima_precip_media"]
)

df["anomalia_viento"] = (
    df["wind_speed_10m"]
    - df["clima_viento_media"]
)

df["anomalia_rafaga"] = (
    df["wind_gusts_10m"]
    - df["clima_rafaga_media"]
)


# ============================================================
# 8. EVENTOS SEVEROS
# ============================================================

print("\n[7/8] Incorporando eventos históricos...")

eventos = pd.read_csv(
    EVENTOS_FILE
)

if "fecha" not in eventos.columns:
    raise ValueError(
        "eventos_severos.csv no tiene columna fecha."
    )

eventos["fecha"] = pd.to_datetime(
    eventos["fecha"],
    errors="coerce"
)

eventos = eventos.dropna(
    subset=["fecha"]
).copy()

eventos["fecha_dia"] = (
    eventos["fecha"].dt.normalize()
)

# Inicializar etiquetas
df["evento_severo"] = 0
df["evento_6h"] = 0
df["evento_12h"] = 0
df["evento_24h"] = 0
df["evento_48h"] = 0
df["evento_72h"] = 0

df["evento_tipo"] = ""
df["evento_rafaga_kmh"] = np.nan
df["evento_lluvia_mm"] = np.nan
df["evento_tornado"] = 0
df["evento_inundacion"] = 0
df["evento_granizo"] = 0

for _, ev in eventos.iterrows():

    fecha = ev["fecha"]

    # Momento del evento.
    # Solo usamos información previa para los objetivos.
    inicio = fecha - pd.Timedelta(hours=72)
    fin = fecha

    mask72 = (
        (df["time"] >= inicio)
        & (df["time"] < fin)
    )

    df.loc[mask72, "evento_72h"] = 1

    for h, col in [
        (6, "evento_6h"),
        (12, "evento_12h"),
        (24, "evento_24h"),
        (48, "evento_48h"),
        (72, "evento_72h"),
    ]:

        inicio_h = (
            fecha
            - pd.Timedelta(hours=h)
        )

        mask = (
            (df["time"] >= inicio_h)
            & (df["time"] < fecha)
        )

        df.loc[mask, col] = 1

    # El evento queda asociado al momento exacto
    # cuando existe un registro horario correspondiente.
    exact = (
        df["time"]
        == fecha
    )

    if exact.any():

        df.loc[
            exact,
            "evento_severo"
        ] = 1

        if "evento" in eventos.columns:
            df.loc[
                exact,
                "evento_tipo"
            ] = str(
                ev.get("evento", "")
            )

        if "rafaga_kmh" in eventos.columns:
            df.loc[
                exact,
                "evento_rafaga_kmh"
            ] = pd.to_numeric(
                ev.get("rafaga_kmh"),
                errors="coerce"
            )

        if "lluvia_mm" in eventos.columns:
            df.loc[
                exact,
                "evento_lluvia_mm"
            ] = pd.to_numeric(
                ev.get("lluvia_mm"),
                errors="coerce"
            )

        for source_col, target_col in [
            ("tornado", "evento_tornado"),
            ("inundacion", "evento_inundacion"),
            ("granizo", "evento_granizo"),
        ]:
            if source_col in eventos.columns:
                df.loc[
                    exact,
                    target_col
                ] = pd.to_numeric(
                    ev.get(source_col),
                    errors="coerce"
                )


# ============================================================
# 9. INFORMACION DOCUMENTAL
# ============================================================

print("\n[8/8] Registrando fuentes documentales...")

documentos = []

if META_FILE.exists():

    texto = META_FILE.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    documentos.append({
        "fuente": "climaar_datos_historicos_meta.txt",
        "disponible": 1,
        "tamano_caracteres": len(texto),
        "observacion":
            "Paquete textual de fuentes meteorológicas. "
            "Los gráficos sin valor numérico exacto "
            "no se convierten automáticamente en números."
    })

else:

    documentos.append({
        "fuente": "climaar_datos_historicos_meta.txt",
        "disponible": 0,
        "tamano_caracteres": 0,
        "observacion":
            "Archivo no encontrado en el repositorio."
    })


catalogo = pd.DataFrame(documentos)

catalogo.to_csv(
    OUTPUT_CATALOGO,
    index=False,
    encoding="utf-8"
)


# ============================================================
# 10. METADATOS DE FUENTES
# ============================================================

df["documentacion_historica_disponible"] = (
    1 if META_FILE.exists() else 0
)

df["climatologia_fuente"] = (
    "Open-Meteo histórico 1991-2020"
)

df["eventos_fuente"] = (
    "data/historico/eventos_severos.csv"
)

df["dataset_version"] = (
    "ClimaAR-FINAL-1.0"
)


# ============================================================
# 11. LIMPIEZA FINAL
# ============================================================

print("\nPreparando dataset final...")

# Reemplazar infinitos
df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

# No eliminamos filas por NaN:
# los primeros registros naturalmente
# no tienen historial suficiente.

df = df.sort_values(
    "time"
).reset_index(drop=True)


# ============================================================
# 12. GUARDAR DATASET
# ============================================================

df.to_csv(
    OUTPUT_DATASET,
    index=False,
    encoding="utf-8"
)


# ============================================================
# 13. RESUMEN
# ============================================================

positivos_6 = int(
    df["evento_6h"].sum()
)

positivos_12 = int(
    df["evento_12h"].sum()
)

positivos_24 = int(
    df["evento_24h"].sum()
)

positivos_48 = int(
    df["evento_48h"].sum()
)

positivos_72 = int(
    df["evento_72h"].sum()
)

resumen = pd.DataFrame([{
    "dataset_version":
        "ClimaAR-FINAL-1.0",

    "registros":
        len(df),

    "columnas":
        len(df.columns),

    "fecha_inicio":
        str(df["time"].min()),

    "fecha_fin":
        str(df["time"].max()),

    "eventos_catalogados":
        len(eventos),

    "muestras_evento_6h":
        positivos_6,

    "muestras_evento_12h":
        positivos_12,

    "muestras_evento_24h":
        positivos_24,

    "muestras_evento_48h":
        positivos_48,

    "muestras_evento_72h":
        positivos_72,

    "documentacion_historica":
        int(META_FILE.exists()),

    "fuente_climatologia":
        "1991-2020",

    "estado":
        "dataset construido correctamente"
}])

resumen.to_csv(
    OUTPUT_RESUMEN,
    index=False,
    encoding="utf-8"
)


# ============================================================
# 14. RESULTADO
# ============================================================

print("\n" + "=" * 70)
print("DATASET FINAL GENERADO")
print("=" * 70)

print(
    f"Registros: {len(df):,}"
)

print(
    f"Columnas: {len(df.columns)}"
)

print(
    f"Eventos: {len(eventos)}"
)

print(
    f"Positivos 6h: {positivos_6:,}"
)

print(
    f"Positivos 12h: {positivos_12:,}"
)

print(
    f"Positivos 24h: {positivos_24:,}"
)

print(
    f"Positivos 48h: {positivos_48:,}"
)

print(
    f"Positivos 72h: {positivos_72:,}"
)

print("\nArchivos creados:")

print(
    OUTPUT_DATASET
)

print(
    OUTPUT_CATALOGO
)

print(
    OUTPUT_RESUMEN
)

print("\n=== CLIMAAR DATASET FINAL OK ===")
