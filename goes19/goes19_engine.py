from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .s3_client import latest_object


PRODUCT_PREFIXES = {
    "abi": "ABI-L2-MCMIPF",
    "glm": "GLM-L2-LCFA",
    "cloudtop_height": "ABI-L2-ACHAF",
    "cloudtop_temperature": "ABI-L2-ACHTF",
    "cloudtop_pressure": "ABI-L2-CTPF",
    "dmw": "ABI-L2-DMWF",
}


def current_hour_prefix(
    product: str,
    now: datetime | None = None,
) -> str:
    if product not in PRODUCT_PREFIXES:
        raise ValueError(f"Producto GOES no soportado: {product}")

    now = now or datetime.now(timezone.utc)

    return f"{PRODUCT_PREFIXES[product]}/{now:%Y/%j/%H}/"


def discover_latest_products(
    now: datetime | None = None,
) -> dict[str, dict[str, Any] | None]:
    return {
        product: latest_object(
            current_hour_prefix(product, now)
        )
        for product in PRODUCT_PREFIXES
    }
