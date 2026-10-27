from __future__ import annotations

from datetime import datetime, timezone, timedelta
import re
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

    # DMW: ABI bands 2, 7, 8, 9, 10 and 14.
    "dmw_c02": ("ABI-L2-DMWF", "M6C02"),
    "dmw_c07": ("ABI-L2-DMWF", "M6C07"),
    "dmw_c08": ("ABI-L2-DMWF", "M6C08"),
    "dmw_c09": ("ABI-L2-DMWF", "M6C09"),
    "dmw_c10": ("ABI-L2-DMWF", "M6C10"),
    "dmw_c14": ("ABI-L2-DMWF", "M6C14"),

    # DMWV: clear-sky water-vapor winds from ABI band 8 only.
    "dmwvf_c08": ("ABI-L2-DMWVF", "M6C08"),
}


# Importante:
# Se usa lookahead (?=...) para NO consumir el "_" o "." que separa
# los timestamps s/e/c. De esta forma finditer encuentra los tres.
GOES_TIME_RE = re.compile(
    r"_(?P<kind>[sec])(?P<timestamp>\d{13,16})(?=_|\.|$)"
)


def current_hour_prefix(product_prefix: str, when: datetime) -> str:
    return f"{product_prefix}/{when:%Y/%j/%H}/"


def current_day_prefix(product_prefix: str, when: datetime) -> str:
    return f"{product_prefix}/{when:%Y/%j}/"


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    ).astimezone(timezone.utc)


def _parse_goes_timestamp(value: str) -> datetime:
    if not value.isdigit() or len(value) < 13:
        raise ValueError(f"Timestamp GOES inválido: {value}")

    year = int(value[0:4])
    day_of_year = int(value[4:7])
    hour = int(value[7:9])
    minute = int(value[9:11])
    second = int(value[11:13])

    fractional = value[13:]
    microsecond = 0

    if fractional:
        microsecond = int(
            round(float(f"0.{fractional}") * 1_000_000)
        )

        if microsecond >= 1_000_000:
            second += 1
            microsecond -= 1_000_000

    base = datetime(
        year=year,
        month=1,
        day=1,
        hour=hour,
        minute=minute,
        second=second,
        microsecond=microsecond,
        tzinfo=timezone.utc,
    )

    return base + timedelta(days=day_of_year - 1)


def _parse_goes_filename_times(key: str) -> dict[str, datetime]:
    parsed: dict[str, datetime] = {}

    for match in GOES_TIME_RE.finditer(key):
        parsed[match.group("kind")] = _parse_goes_timestamp(
            match.group("timestamp")
        )

    missing = [x for x in ("s", "e", "c") if x not in parsed]

    if missing:
        raise ValueError(
            f"No se pudieron extraer todos los timestamps GOES de {key}. "
            f"Faltan: {', '.join(missing)}"
        )

    return {
        "start_time": parsed["s"],
        "end_time": parsed["e"],
        "creation_time": parsed["c"],
    }


def _enrich_object_metadata(
    obj: dict[str, Any],
) -> dict[str, Any] | None:
    key = obj.get("key", "")

    if not key:
        return None

    try:
        times = _parse_goes_filename_times(key)
    except (TypeError, ValueError):
        return None

    enriched = dict(obj)

    for name, value in times.items():
        enriched[f"{name}_utc"] = (
            value.isoformat().replace("+00:00", "Z")
        )

    return enriched


def _list_candidate_objects(
    product_prefix: str,
    when: datetime,
    timeout: int = 20,
) -> list[dict[str, Any]]:
    objects = list_objects(
        prefix=current_hour_prefix(product_prefix, when),
        max_keys=1000,
        timeout=timeout,
    )

    if objects:
        return objects

    return list_objects(
        prefix=current_day_prefix(product_prefix, when),
        max_keys=1000,
        timeout=timeout,
    )


def _normalise_reference_time(
    now: datetime | None,
) -> datetime:
    value = now or datetime.now(timezone.utc)

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


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

    reference_time = _normalise_reference_time(now)

    product_prefix, key_contains = PRODUCT_SPECS[product]

    candidates: list[dict[str, Any]] = []

    for offset in range(hours_back + 1):
        when = reference_time - timedelta(hours=offset)

        objects = _list_candidate_objects(
            product_prefix,
            when,
        )

        for raw_obj in objects:
            key = raw_obj.get("key", "")

            if key_contains and key_contains not in key:
                continue

            obj = _enrich_object_metadata(raw_obj)

            if obj is None:
                continue

            try:
                start_time = _parse_utc(
                    obj["start_time_utc"]
                )

                end_time = _parse_utc(
                    obj["end_time_utc"]
                )

            except (
                TypeError,
                ValueError,
                KeyError,
            ):
                continue

            if end_time < start_time:
                continue

            if end_time > reference_time:
                continue

            obj["product"] = product
            obj["product_prefix"] = product_prefix
            obj["hours_back"] = offset

            candidates.append(obj)

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda obj: _parse_utc(
            obj["end_time_utc"]
        ),
    )


def latest_product_objects(
    products: list[str],
    now: datetime | None = None,
    hours_back: int = 6,
) -> dict[str, dict[str, Any] | None]:

    reference_time = _normalise_reference_time(now)

    return {
        product: latest_product_object(
            product,
            now=reference_time,
            hours_back=hours_back,
        )
        for product in products
    }


def discover_latest_products(
    now: datetime | None = None,
    hours_back: int = 6,
) -> dict[str, dict[str, Any] | None]:

    return latest_product_objects(
        list(PRODUCT_SPECS),
        now=now,
        hours_back=hours_back,
    )
