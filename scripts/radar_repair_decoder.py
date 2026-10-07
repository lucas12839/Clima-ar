import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from scripts import radar_rainviewer as rv
except ImportError:
    import radar_rainviewer as rv


# ============================================================
# CLIMAAR - RADAR REPAIR DECODER V7.4
#
# Decoder robusto para RainViewer Universal Blue.
#
# IMPORTANTE:
# RainViewer utiliza una tabla RGBA oficial.
# Algunos valores bajos tienen alpha menor a 255.
# Por eso NO usamos alpha como condición obligatoria
# para detectar precipitación.
# ============================================================

BASE = Path(__file__).resolve().parents[1]

RADAR_DIR = BASE / "data" / "radar"
HISTORY_DIR = RADAR_DIR / "historico"

NOWCAST_FILE = RADAR_DIR / "radar_nowcast.json"

MAX_FRAMES = 13

# Distancia máxima entre el RGB recibido y la tabla oficial.
PALETTE_MATCH_DISTANCE = 36.0

# Umbral meteorológico mínimo.
MIN_DBZ = 10.0


# ============================================================
# TABLA OFICIAL UNIVERSAL BLUE
#
# RainViewer color scheme 2.
#
# Desde 10 dBZ hasta 95 dBZ.
# ============================================================

DBZ_HEX = [
    "cec08796",
    "d2c48ba0",
    "d6c88faa",
    "dacc93b4",
    "ded097be",

    "88ddeeff",
    "6cd1ebff",
    "51c5e8ff",
    "36bae5ff",
    "1baee2ff",

    "00a3e0ff",
    "009ad5ff",
    "0091caff",
    "0088bfff",
    "007fb4ff",

    "0077aaff",
    "0070a3ff",
    "00699cff",
    "006295ff",
    "005b8eff",

    "005588ff",
    "005180ff",
    "004e78ff",
    "004a70ff",
    "004768ff",

    "ffee00ff",
    "ffe000ff",
    "ffd200ff",
    "ffc500ff",
    "ffb700ff",

    "ffaa00ff",
    "ff9f00ff",
    "ff9500ff",
    "ff8b00ff",
    "ff8100ff",

    "ff4400ff",
    "f23600ff",
    "e62800ff",
    "d91b00ff",
    "cd0d00ff",

    "c10000ff",
    "a80000ff",
    "8f0000ff",
    "760000ff",
    "5d0000ff",

    "ffaaffff",
    "ff9fffff",
    "ff95ffff",
    "ff8bffff",
    "ff81ffff",

    "ff77ffff",
    "ff6cffff",
    "ff62ffff",
    "ff58ffff",
    "ff4effff",

    "ffffffff",
    "ffffffff",
    "ffffffff",
    "ffffffff",
    "ffffffff",

    "ffffffff",
    "ffffffff",
    "ffffffff",
    "ffffffff",
    "ffffffff",

    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
    "00ff00ff",

    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
    "00ff00ff",

    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
    "00ff00ff",

    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
    "00ff00ff",

    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
    "00ff00ff",
]


DBZ_VALUES = np.arange(
    10,
    96,
    dtype=np.float32
)


DBZ_RGBA = np.array(
    [
        [
            int(value[i:i + 2], 16)
            for i in (0, 2, 4, 6)
        ]
        for value in DBZ_HEX
    ],
    dtype=np.float32
)


DBZ_RGB = DBZ_RGBA[:, :3]


# ============================================================
# DECODIFICADOR ROBUSTO
# ============================================================

def rgba_to_dbz_robusto(image):

    array = np.asarray(
        image
    ).astype(
        np.float32
    )

    rgb = array[:, :, :3]
    alpha = array[:, :, 3]

    height, width = rgb.shape[:2]

    flat = rgb.reshape(
        -1,
        3
    )

    # --------------------------------------------------------
    # Distancia RGB contra toda la tabla oficial.
    # --------------------------------------------------------

    distances = (
        (
            flat[:, None, :]
            -
            DBZ_RGB[None, :, :]
        )
        ** 2
    ).sum(
        axis=2
    )

    nearest_index = np.argmin(
        distances,
        axis=1
    )

    nearest_distance = np.sqrt(
        np.min(
            distances,
            axis=1
        )
    )

    dbz = DBZ_VALUES[
        nearest_index
    ].reshape(
        height,
        width
    ).astype(
        np.float32
    )

    nearest_distance = nearest_distance.reshape(
        height,
        width
    )

    # --------------------------------------------------------
    # Detección robusta.
    #
    # NO exigimos alpha > 0.
    #
    # El color RGB es la señal principal.
    # Esto permite recuperar píxeles radar cuyo alpha
    # haya sido alterado durante el procesamiento PNG.
    # --------------------------------------------------------

    valid_color = (
        nearest_distance
        <=
        PALETTE_MATCH_DISTANCE
    )

    # --------------------------------------------------------
    # Evitar confundir fondo negro/transparente con radar.
    #
    # El fondo habitual es aproximadamente 0,0,0.
    # --------------------------------------------------------

    background_black = (
        (rgb[:, :, 0] <= 3)
        &
        (rgb[:, :, 1] <= 3)
        &
        (rgb[:, :, 2] <= 3)
    )

    valid = (
        valid_color
        &
        ~background_black
    )

    dbz[
        ~valid
    ] = np.nan

    return dbz


# ============================================================
# MÉTRICAS
# ============================================================

def calculate_metrics(dbz):

    valid = dbz[
        np.isfinite(dbz)
    ]

    precip = valid[
        valid >= MIN_DBZ
    ]

    if precip.size == 0:

        return {
            "dbz_max": None,
            "dbz_mean": None,
            "dbz_p90": None,
            "dbz_pixels": 0,
        }

    return {
        "dbz_max": float(
            np.max(
                precip
            )
        ),

        "dbz_mean": round(
            float(
                np.mean(
                    precip
                )
            ),
            1
        ),

        "dbz_p90": round(
            float(
                np.percentile(
                    precip,
                    90
                )
            ),
            1
        ),

        "dbz_pixels": int(
            precip.size
        ),
    }


# ============================================================
# ANALIZAR FRAME
# ============================================================

def analyze_frame(
    image,
    timestamp
):

    array = np.asarray(
        image
    )

    rgb = array[:, :, :3]
    alpha = array[:, :, 3]

    dbz = rgba_to_dbz_robusto(
        image
    )

    metrics = calculate_metrics(
        dbz
    )

    precipitation_mask = (
        np.isfinite(dbz)
        &
        (
            dbz >= MIN_DBZ
        )
    )

    valid_palette = np.isfinite(
        dbz
    )

    diagnostics = {

        "modo_imagen":
            image.mode,

        "ancho":
            int(image.width),

        "alto":
            int(image.height),

        "pixeles_totales":
            int(
                image.width
                *
                image.height
            ),

        "pixeles_alpha":
            int(
                np.count_nonzero(
                    alpha > 0
                )
            ),

        "pixeles_rgb_no_negros":
            int(
                np.count_nonzero(
                    np.any(
                        rgb > 3,
                        axis=2
                    )
                )
            ),

        "pixeles_paleta_detectados":
            int(
                np.count_nonzero(
                    valid_palette
                )
            ),

        "pixeles_precipitacion":
            int(
                np.count_nonzero(
                    precipitation_mask
                )
            ),

        "alpha_max":
            int(
                np.max(alpha)
            ),

        "tolerancia_color":
            PALETTE_MATCH_DISTANCE,
    }

    return {

        "timestamp":
            int(timestamp),

        "frame_utc":
            datetime.fromtimestamp(
                int(timestamp),
                timezone.utc
            ).isoformat(),

        "area":
            int(
                np.count_nonzero(
                    precipitation_mask
                )
            ),

        "components":
            [],

        "distance_km":
            None,

        **metrics,

        "diagnostico_paleta":
            diagnostics,
    }


# ============================================================
# BUSCAR HISTORIAL
# ============================================================

def load_history():

    files = []

    if not HISTORY_DIR.exists():
        return files

    for path in HISTORY_DIR.glob(
        "radar_*.png"
    ):

        try:

            timestamp = int(
                path.stem.split(
                    "_"
                )[-1]
            )

        except (
            ValueError,
            IndexError
        ):

            continue

        if (
            path.is_file()
            and
            path.stat().st_size > 100
        ):

            files.append(
                (
                    timestamp,
                    path
                )
            )

    files.sort(
        key=lambda item: item[0]
    )

    return files[
        -MAX_FRAMES:
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    files = load_history()

    if not files:

        raise RuntimeError(
            "No hay frames históricos para reparar."
        )

    print(
        "===================================="
    )

    print(
        "CLIMAAR RADAR DECODER V7.4"
    )

    print(
        "Frames encontrados:",
        len(files)
    )

    print(
        "===================================="
    )

    results = []

    for timestamp, path in files:

        print()
        print(
            "Analizando:",
            path.name
        )

        image = Image.open(
            path
        ).convert(
            "RGBA"
        )

        result = analyze_frame(
            image,
            timestamp
        )

        results.append(
            result
        )

        d = result[
            "diagnostico_paleta"
        ]

        print(
            "Modo:",
            d["modo_imagen"]
        )

        print(
            "Tamaño:",
            d["ancho"],
            "x",
            d["alto"]
        )

        print(
            "Alpha > 0:",
            d["pixeles_alpha"]
        )

        print(
            "RGB no negro:",
            d["pixeles_rgb_no_negros"]
        )

        print(
            "Paleta detectada:",
            d["pixeles_paleta_detectados"]
        )

        print(
            "Precipitación:",
            d["pixeles_precipitacion"]
        )

        print(
            "dBZ máximo:",
            result["dbz_max"]
        )

    # --------------------------------------------------------
    # Orden temporal
    # --------------------------------------------------------

    results.sort(
        key=lambda item:
        item["timestamp"]
    )

    # --------------------------------------------------------
    # Utilizamos el motor existente de movimiento.
    # --------------------------------------------------------

    nowcast = rv.analyze_sequence(
        results
    )

    latest = results[-1]

    output = {

        "version":
            "7.4",

        "app":
            "ClimaAR",

        "ubicacion": {

            "latitud":
                rv.LAT,

            "longitud":
                rv.LON,

            "ciudad":
                "Bahia Blanca",
        },

        "fuente":
            "RainViewer",

        "rainviewer_color_scheme":
            "Universal Blue (2)",

        "dbz_decode":
            "tabla oficial Universal Blue RGBA + deteccion RGB robusta",

        "nowcast":
            nowcast,

        "historial": {

            "frames_procesados":
                len(results),

            "frames_disponibles":
                len(files),

            "max_history_frames":
                rv.MAX_HISTORY_FRAMES,
        },

        "diagnostico": {

            "metodo":
                "decoder_rgb_robusto",

            "tolerancia_color":
                PALETTE_MATCH_DISTANCE,

            "ultimo_frame":
                latest[
                    "diagnostico_paleta"
                ],
        },
    }

    NOWCAST_FILE.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print(
        "===================================="
    )

    print(
        "NOWCAST REPARADO V7.4"
    )

    print(
        "===================================="
    )

    print(
        json.dumps(
            nowcast,
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
