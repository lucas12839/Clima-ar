from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

from .s3_client import list_objects
from .s3_client import S3_ENDPOINT


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


def current_hour_prefix(
    product_prefix: str,
    when: datetime,
) -> str:
    return f"{product_prefix}/{when:%Y/%j/%H}/"


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    ).astimezone(timezone.utc)


def latest_product_object(
    product: str,
    now: datetime | None = None,
    hours_back: int = 6,
) -> dict[str, Any] | None:

    if product not in PRODUCT_SPECS:
        raise ValueError(
            f"Producto GOES no soportado: {product}"
        )

    if hours_back < 0:
        raise ValueError(
            "hours_back no puede ser negativo"
        )

    reference_time = (
        now or datetime.now(timezone.utc)
    )

    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(
            tzinfo=timezone.utc
        )

    reference_time = reference_time.astimezone(
        timezone.utc
    )

    product_prefix, key_contains = (
        PRODUCT_SPECS[product]
    )

    best = None
    best_time = None

    for offset in range(hours_back + 1):

        when = (
            reference_time
            - timedelta(hours=offset)
        )

        prefix = current_hour_prefix(
            product_prefix,
            when,
        )

        objects = list_objects(
            prefix=prefix,
            max_keys=1000,
        )

        for obj in objects:

            key = obj.get("key", "")

            if (
                key_contains
                and key_contains not in key
            ):
                continue

            last_modified = obj.get(
                "last_modified"
            )

            if not last_modified:
                continue

            try:
                object_time = _parse_utc(
                    last_modified
                )
            except (TypeError, ValueError):
                continue

            # Nunca permitir datos posteriores
            # al frame RMA10 que estamos analizando.
            if object_time > reference_time:
                continue

            if (
                best_time is None
                or object_time > best_time
            ):
                best = obj
                best_time = object_time

    return best


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
