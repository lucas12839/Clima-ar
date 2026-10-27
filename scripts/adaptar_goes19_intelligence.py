from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]

GOES_STATE_FILE = (
    BASE_DIR
    / "data"
    / "goes19"
    / "goes19_state.json"
)

RADAR_NOWCAST_FILE = (
    BASE_DIR
    / "data"
    / "radar"
    / "radar_nowcast.json"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "ia"
    / "goes19_intelligence.json"
)

VERSION = "1.1.0"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"{path.name} no contiene un objeto JSON."
        )

    return data


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def parse_utc(value: Any) -> datetime | None:
    if not value:
        return None

    text = str(value).strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def load_reference_frame_utc() -> datetime:
    radar = load_json(RADAR_NOWCAST_FILE)

    nowcast = radar.get("nowcast")

    if not isinstance(nowcast, dict):
        raise ValueError(
            "radar_nowcast.json no contiene el bloque 'nowcast'."
        )

    frame_value = nowcast.get("frame_actual_utc")

    frame = parse_utc(frame_value)

    if frame is None:
        raise ValueError(
            "No se pudo obtener nowcast.frame_actual_utc "
            "desde radar_nowcast.json."
        )

    return frame


def temporal_info(
    value: Any,
    reference_time: datetime,
) -> dict[str, Any]:
    timestamp = parse_utc(value)

    if timestamp is None:
        return {
            "age_minutes": None,
            "offset_minutes": None,
            "temporal_validity": "unknown",
        }

    offset_minutes = (
        reference_time - timestamp
    ).total_seconds() / 60.0

    if offset_minutes < 0:
        return {
            "age_minutes": None,
            "offset_minutes": round(offset_minutes, 3),
            "temporal_validity": "future_relative_to_frame",
        }

    return {
        "age_minutes": round(offset_minutes, 3),
        "offset_minutes": round(offset_minutes, 3),
        "temporal_validity": "valid",
    }


def product_summary(
    name: str,
    data: Any,
    reference_time: datetime,
) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {
            "status": "unavailable",
            "reason": "Producto sin datos.",
        }

    source = data.get("source")
    domain = data.get("domain")
    quality = data.get("quality")
    statistics = data.get("statistics")

    if not isinstance(source, dict):
        source = {}

    if not isinstance(domain, dict):
        domain = {}

    if not isinstance(quality, dict):
        quality = {}

    if not isinstance(statistics, dict):
        statistics = {}

    last_modified = source.get(
        "last_modified_utc"
    )

    temporal = temporal_info(
        last_modified,
        reference_time,
    )

    status = data.get(
        "status",
        "unavailable",
    )

    if (
        temporal["temporal_validity"]
        == "future_relative_to_frame"
    ):
        status = "future_relative_to_frame"

    return {
        "status": status,
        "source": {
            "product": source.get(
                "product",
                name,
            ),
            "key": source.get("key"),
            "last_modified_utc": last_modified,
            "age_minutes": temporal[
                "age_minutes"
            ],
            "offset_minutes": temporal[
                "offset_minutes"
            ],
            "temporal_validity": temporal[
                "temporal_validity"
            ],
            "size_bytes": source.get(
                "size_bytes"
            ),
        },
        "domain": {
            "center_lat": domain.get(
                "center_lat"
            ),
            "center_lon": domain.get(
                "center_lon"
            ),
            "radius_km": domain.get(
                "radius_km"
            ),
            "points": domain.get("points"),
        },
        "quality": quality,
        "statistics": statistics,
    }


def cloudtop_summary(
    name: str,
    data: Any,
    reference_time: datetime,
) -> dict[str, Any]:
    return product_summary(
        name,
        data,
        reference_time,
    )


def glm_summary(
    data: Any,
    reference_time: datetime,
) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {
            "status": "unavailable",
            "counts": {},
        }

    source = data.get("source")
    domain = data.get("domain")
    counts = data.get("counts")
    energy = data.get("flash_energy")
    area = data.get("flash_area")
    nearest = data.get("nearest_flash")

    if not isinstance(source, dict):
        source = {}

    if not isinstance(domain, dict):
        domain = {}

    if not isinstance(counts, dict):
        counts = {}

    if not isinstance(nearest, dict):
        nearest = {}

    last_modified = source.get(
        "last_modified_utc"
    )

    temporal = temporal_info(
        last_modified,
        reference_time,
    )

    status = data.get(
        "status",
        "unavailable",
    )

    if (
        temporal["temporal_validity"]
        == "future_relative_to_frame"
    ):
        status = "future_relative_to_frame"

    return {
        "status": status,
        "source": {
            "product": source.get(
                "product",
                "glm",
            ),
            "key": source.get("key"),
            "last_modified_utc": last_modified,
            "age_minutes": temporal[
                "age_minutes"
            ],
            "offset_minutes": temporal[
                "offset_minutes"
            ],
            "temporal_validity": temporal[
                "temporal_validity"
            ],
            "size_bytes": source.get(
                "size_bytes"
            ),
        },
        "domain": {
            "center_lat": domain.get(
                "center_lat"
            ),
            "center_lon": domain.get(
                "center_lon"
            ),
            "radius_km": domain.get(
                "radius_km"
            ),
        },
        "counts": counts,
        "flash_energy": energy,
        "flash_area": area,
        "nearest_flash": nearest,
    }


def dmw_summary(
    data: Any,
    reference_time: datetime,
) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {
            "status": "unavailable",
        }

    source = data.get("source")
    domain = data.get("domain")
    search = data.get("search")

    if not isinstance(source, dict):
        source = {}

    if not isinstance(domain, dict):
        domain = {}

    if not isinstance(search, dict):
        search = {}

    last_modified = source.get(
        "last_modified_utc"
    )

    temporal = temporal_info(
        last_modified,
        reference_time,
    )

    status = data.get(
        "status",
        "unavailable",
    )

    if (
        temporal["temporal_validity"]
        == "future_relative_to_frame"
    ):
        status = "future_relative_to_frame"

    return {
        "status": status,
        "source": {
            "product": source.get(
                "product",
                "dmw",
            ),
            "key": source.get("key"),
            "last_modified_utc": last_modified,
            "age_minutes": temporal[
                "age_minutes"
            ],
            "offset_minutes": temporal[
                "offset_minutes"
            ],
            "temporal_validity": temporal[
                "temporal_validity"
            ],
            "size_bytes": source.get(
                "size_bytes"
            ),
        },
        "domain": {
            "center_lat": domain.get(
                "center_lat"
            ),
            "center_lon": domain.get(
                "center_lon"
            ),
            "radius_km": domain.get(
                "radius_km"
            ),
            "measurements": domain.get(
                "measurements"
            ),
        },
        "search": search,
        "wind_speed": data.get(
            "wind_speed"
        ),
        "wind_direction": data.get(
            "wind_direction"
        ),
        "temperature": data.get(
            "temperature"
        ),
        "pressure": data.get(
            "pressure"
        ),
        "nearest_measurement": data.get(
            "nearest_measurement"
        ),
    }


def build_intelligence_state(
    state: dict[str, Any],
    reference_time: datetime,
) -> dict[str, Any]:
    domain = state.get("domain")

    if not isinstance(domain, dict):
        domain = {}

    products = state.get("products")

    if not isinstance(products, dict):
        products = {}

    cloudtop = state.get("cloudtop")

    if not isinstance(cloudtop, dict):
        cloudtop = {}

    normalized_products = {
        "abi_c08": product_summary(
            "abi_c08",
            products.get("abi_c08"),
            reference_time,
        ),
        "abi_c13": product_summary(
            "abi_c13",
            products.get("abi_c13"),
            reference_time,
        ),
    }

    normalized_cloudtop = {
        "cloudtop_height": cloudtop_summary(
            "cloudtop_height",
            cloudtop.get(
                "cloudtop_height"
            ),
            reference_time,
        ),
        "cloudtop_pressure": cloudtop_summary(
            "cloudtop_pressure",
            cloudtop.get(
                "cloudtop_pressure"
            ),
            reference_time,
        ),
    }

    glm = glm_summary(
        state.get("glm"),
        reference_time,
    )

    dmw = dmw_summary(
        state.get("dmw"),
        reference_time,
    )

    components = [
        *normalized_products.values(),
        *normalized_cloudtop.values(),
        glm,
        dmw,
    ]

    available = sum(
        1
        for component in components
        if component.get("status")
        in {
            "ok",
            "sin_actividad",
        }
    )

    generated_at = state.get(
        "generated_at_utc"
    )

    state_temporal = temporal_info(
        generated_at,
        reference_time,
    )

    return {
        "version": VERSION,
        "motor": (
            "ClimaAR GOES-19 "
            "Intelligence Adapter"
        ),
        "source_state_version": state.get(
            "version"
        ),
        "satellite": state.get(
            "satellite",
            "GOES-19",
        ),

        "reference_frame": {
            "source": (
                "RMA10 radar_nowcast"
            ),
            "frame_actual_utc": (
                reference_time.isoformat()
                .replace("+00:00", "Z")
            ),
        },

        "generated_at_utc": generated_at,

        "state_temporal": {
            "age_minutes": state_temporal[
                "age_minutes"
            ],
            "offset_minutes": state_temporal[
                "offset_minutes"
            ],
            "temporal_validity": state_temporal[
                "temporal_validity"
            ],
        },

        "status": state.get(
            "status",
            "unavailable",
        ),

        "domain": {
            "type": domain.get("type"),
            "center_lat": domain.get(
                "center_lat"
            ),
            "center_lon": domain.get(
                "center_lon"
            ),
            "radius_km": domain.get(
                "radius_km"
            ),
        },

        "productos": normalized_products,
        "cloudtop": normalized_cloudtop,
        "glm": glm,
        "dmw": dmw,

        "calidad": {
            "componentes_disponibles": available,
            "componentes_totales": len(
                components
            ),
            "estado": (
                "ok"
                if available == len(
                    components
                )
                else (
                    "partial"
                    if available > 0
                    else "unavailable"
                )
            ),
        },
    }


def save_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    temporary.replace(path)


def main() -> None:
    reference_frame = (
        load_reference_frame_utc()
    )

    state = load_json(
        GOES_STATE_FILE
    )

    adapted = build_intelligence_state(
        state,
        reference_frame,
    )

    save_json(
        OUTPUT_FILE,
        adapted,
    )

    print(
        "OK: GOES-19 adaptado para "
        "ClimaAR Intelligence"
    )
    print(
        f"Frame RMA10: "
        f"{adapted['reference_frame']['frame_actual_utc']}"
    )
    print(
        f"Estado: {adapted['status']}"
    )
    print(
        "Componentes: "
        f"{adapted['calidad']['componentes_disponibles']}/"
        f"{adapted['calidad']['componentes_totales']}"
    )
    print(
        f"Salida: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
