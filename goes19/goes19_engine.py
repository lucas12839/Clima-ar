from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

from .s3_client import latest_object, latest_matching_object


PRODUCT_SPECS = {
    "abi_c02": ("ABI-L2-CMIPF", "M6C02"),
    "abi_c08": ("ABI-L2-CMIPF", "M6C08"),
    "abi_c13": ("ABI-L2-CMIPF", "M6C13"),
    "glm": ("GLM-L2-LCFA", None),
    "cloudtop_height": ("ABI-L2-ACHAF", None),
    "cloudtop_temperature": ("ABI-L2-ACHTF", None),
    "cloudtop_pressure": ("ABI-L2-CTPF", None),
    "dmw": ("ABI-L2-DMWF", None),
}


def current_hour_prefix(product_prefix: str, when: datetime) -> str:
    return f"{product_prefix}/{when:%Y/%j/%H}/"


def latest_product_object(
    product: str,
    now: datetime | None = None,
    hours_back: int = 6,
) -> dict[str, Any] | None:
    if product not in PRODUCT_SPECS:
        raise ValueError(f"Producto GOES no soportado: {product}")

    if hours_back < 0:
        raise ValueError("hours_back no puede ser negativo")

    now = now or datetime.now(timezone.utc)

    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    product_prefix, key_contains = PRODUCT_SPECS[product]

    for offset in range(hours_back + 1):
        when = now - timedelta(hours=offset)
        prefix = current_hour_prefix(product_prefix, when)

        if key_contains:
            found = latest_matching_object(
                prefix=prefix,
                key_contains=key_contains,
            )
        else:
            found = latest_object(prefix)

        if found is not None:
            return found

    return None


def discover_latest_products(
    now: datetime | None = None,
    hours_back: int = 6,
) -> dict[str, dict[str, Any] | None]:
    return {
        product: latest_product_object(
            product,
            now=now,
            hours_back=hours_back,
        )
        for product in PRODUCT_SPECS
    }
