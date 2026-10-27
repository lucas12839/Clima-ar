from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

from .s3_client import list_objects


PRODUCT_SPECS = {
    "abi_c02": ("ABI-L2-CMIPF", "M6C02"),
    "abi_c08": ("ABI-L2-CMIPF", "M6C08"),
    "abi_c13": ("ABI-L2-CMIPF", "M6C13"),
    "glm": ("GLM-L2-LCFA", None),
    "cloudtop_height": ("ABI-L2-ACHAF", None),
    "cloudtop_temperature": ("ABI-L2-ACHTF", None),
    "cloudtop_pressure": ("ABI-L2-CTPF", None),

    # DMW Full Disk principal.
    # DMW Vapor Full Disk como respaldo.
    "dmw": (
        ("ABI-L2-DMWF", None),
        ("ABI-L2-DMWVF", None),
    ),
}


def current_hour_prefix(
    product_prefix: str,
    when: datetime,
) -> str:
    return f"{product_prefix}/{when:%Y/%j/%H}/"


def current_day_prefix(
    product_prefix: str,
    when: datetime,
) -> str:
    return f"{product_prefix}/{when:%Y/%j}/"


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    ).astimezone(timezone.utc)


def _product_candidates(
    product: str,
) -> tuple[tuple[str, str | None], ...]:

    spec = PRODUCT_SPECS[product]

    # Productos normales:
    # ("ABI-L2-CMIPF", "M6C08")
    if (
        isinstance(spec, tuple)
        and len(spec) == 2
        and isinstance(spec[0], str)
    ):
        return (spec,)

    # Productos con múltiples fuentes:
    # (("ABI-L2-DMWF", None), ("ABI-L2-DMWVF", None))
    return tuple(spec)


def _list_candidate_objects(
    product_prefix: str,
    when: datetime,
    timeout: int = 20,
) -> list[dict[str, Any]]:
    """
    Primero busca en la hora exacta.

    Si esa hora no contiene objetos, busca dentro del día.
    Esto evita depender de una suposición rígida sobre el
    instante exacto de publicación del producto.
    """

    objects = list_objects(
        prefix=current_hour_prefix(
            product_prefix,
            when,
        ),
        max_keys=1000,
        timeout=timeout,
    )

    if objects:
        return objects

    return list_objects(
        prefix=current_day_prefix(
            product_prefix,
            when,
        ),
        max_keys=1000,
        timeout=timeout,
    )


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

    best = None
    best_time = None
    best_priority = None

    candidates = _product_candidates(product)

    for offset in range(hours_back + 1):

        when = (
            reference_time
            - timedelta(hours=offset)
        )

        for priority, (
            product_prefix,
            key_contains,
        ) in enumerate(candidates):

            objects = _list_candidate_objects(
                product_prefix,
                when,
            )

            for obj in objects:

                key = obj.get(
                    "key",
                    "",
                )

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
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                # CRÍTICO:
                # jamás usar GOES posterior al frame RMA10.
                if object_time > reference_time:
                    continue

                # El timestamp manda.
                # La prioridad solo resuelve empates.
                if (
                    best_time is None
                    or object_time > best_time
                    or (
                        object_time == best_time
                        and (
                            best_priority is None
                            or priority < best_priority
                        )
                    )
                ):

                    best = dict(obj)

                    best[
                        "product_prefix"
                    ] = product_prefix

                    best[
                        "product_priority"
                    ] = priority

                    best_time = object_time
                    best_priority = priority

    return best


def discover_latest_products(
    now: datetime | None = None,
    hours_back: int = 6,
) -> dict[
    str,
    dict[str, Any] | None,
]:

    return {
        product: latest_product_object(
            product,
            now=now,
            hours_back=hours_back,
        )
        for product in PRODUCT_SPECS
    }
