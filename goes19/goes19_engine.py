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

    # DMW Full Disk principal.
    # DMW Vapor Full Disk como respaldo.
    "dmw": (
        ("ABI-L2-DMWF", None),
        ("ABI-L2-DMWVF", None),
    ),
}


# ============================================================
# GOES filename timestamps
# ============================================================
#
# Ejemplo:
#
# OR_ABI-L2-CMIPF-M6C08_G19_
# s20263000230203_
# e20263000239511_
# c20263000239583.nc
#
# Formato:
#
# YYYY + DDD + HHMMSS + décima de segundo
#
# Ejemplo:
# 2026 + 300 + 02:30:20.3
#
# Admitimos también variantes con más precisión decimal.
#
GOES_TIME_RE = re.compile(
    r"_(?P<kind>[sec])"
    r"(?P<timestamp>\d{13,16})"
    r"(?:_|\.|$)"
)


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


def _parse_goes_timestamp(
    value: str,
) -> datetime:
    """
    Convierte un timestamp GOES del tipo:

        YYYYDDDHHMMSSf

    o variantes con mayor precisión decimal.

    Ejemplo:

        20263000230203
        -> 2026-10-27 02:30:20.3 UTC
    """

    if not value.isdigit():
        raise ValueError(
            f"Timestamp GOES inválido: {value}"
        )

    if len(value) < 13:
        raise ValueError(
            f"Timestamp GOES demasiado corto: {value}"
        )

    year = int(value[0:4])
    day_of_year = int(value[4:7])
    hour = int(value[7:9])
    minute = int(value[9:11])
    second = int(value[11:13])

    fractional_digits = value[13:]

    if fractional_digits:
        fraction = float(
            f"0.{fractional_digits}"
        )
        microsecond = int(
            round(fraction * 1_000_000)
        )
    else:
        microsecond = 0

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

    return base + timedelta(
        days=day_of_year - 1
    )


def _parse_goes_filename_times(
    key: str,
) -> dict[str, datetime]:
    """
    Extrae del nombre del archivo GOES:

        s = inicio del producto
        e = fin del producto
        c = creación del producto

    Los timestamps del nombre son la referencia temporal
    meteorológica del producto.

    LastModified de S3 se conserva por separado y NO se
    utiliza como reloj meteorológico.
    """

    matches = GOES_TIME_RE.finditer(key)

    parsed: dict[str, datetime] = {}

    for match in matches:

        kind = match.group("kind")
        timestamp = match.group("timestamp")

        parsed[kind] = _parse_goes_timestamp(
            timestamp
        )

    required = ("s", "e", "c")

    missing = [
        name
        for name in required
        if name not in parsed
    ]

    if missing:
        raise ValueError(
            "No se pudieron extraer todos los "
            "timestamps GOES del archivo "
            f"{key}. Faltan: {', '.join(missing)}"
        )

    return {
        "start_time": parsed["s"],
        "end_time": parsed["e"],
        "creation_time": parsed["c"],
    }


def _enrich_object_metadata(
    obj: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Agrega al objeto S3 los timestamps meteorológicos
    extraídos del nombre del archivo.

    Si el nombre no permite extraerlos de forma segura,
    el objeto se descarta del selector temporal.
    """

    key = obj.get("key", "")

    if not key:
        return None

    try:
        times = _parse_goes_filename_times(
            key
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    enriched = dict(obj)

    enriched["start_time_utc"] = (
        times["start_time"].isoformat()
        .replace("+00:00", "Z")
    )

    enriched["end_time_utc"] = (
        times["end_time"].isoformat()
        .replace("+00:00", "Z")
    )

    enriched["creation_time_utc"] = (
        times["creation_time"].isoformat()
        .replace("+00:00", "Z")
    )

    return enriched


def _product_candidates(
    product: str,
) -> tuple[tuple[str, str | None], ...]:

    spec = PRODUCT_SPECS[product]

    # Productos normales:
    #
    # ("ABI-L2-CMIPF", "M6C08")
    #
    if (
        isinstance(spec, tuple)
        and len(spec) == 2
        and isinstance(spec[0], str)
    ):
        return (spec,)

    # Productos con múltiples fuentes:
    #
    # (
    #   ("ABI-L2-DMWF", None),
    #   ("ABI-L2-DMWVF", None),
    # )
    #
    return tuple(spec)


def _list_candidate_objects(
    product_prefix: str,
    when: datetime,
    timeout: int = 20,
) -> list[dict[str, Any]]:
    """
    Busca objetos del producto dentro de la hora indicada.

    Si no existen objetos en esa hora, amplía la búsqueda
    al día.

    La validación temporal se realiza posteriormente,
    utilizando los timestamps meteorológicos del propio
    nombre del producto.
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
    """
    Devuelve el producto GOES meteorológicamente más
    reciente que ya haya terminado al momento del frame
    de referencia.

    Regla temporal fundamental de ClimaAR:

        product_end_time <= reference_time

    Esto evita contaminar un frame histórico RMA10 con
    información GOES que todavía no estaba disponible
    dentro de ese intervalo temporal.

    La selección se basa en el timestamp del propio
    producto GOES, no en S3 LastModified.
    """

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

    best: dict[str, Any] | None = None
    best_end_time: datetime | None = None
    best_priority: int | None = None

    candidates = _product_candidates(product)

    for offset in range(hours_back + 1):

        hour_reference = (
            reference_time
            - timedelta(hours=offset)
        )

        for priority, (
            product_prefix,
            key_contains,
        ) in enumerate(candidates):

            objects = _list_candidate_objects(
                product_prefix,
                hour_reference,
            )

            for raw_obj in objects:

                key = raw_obj.get(
                    "key",
                    "",
                )

                if (
                    key_contains
                    and key_contains not in key
                ):
                    continue

                obj = _enrich_object_metadata(
                    raw_obj
                )

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

                # =====================================================
                # VALIDACIÓN ESTRUCTURAL
                # =====================================================
                #
                # Un producto no puede terminar antes de comenzar.
                #
                if end_time < start_time:
                    continue

                # =====================================================
                # REGLA TEMPORAL CRÍTICA
                # =====================================================
                #
                # El producto completo debe haber terminado antes
                # o exactamente en el frame RMA10.
                #
                # Esto es deliberadamente más estricto que usar:
                #
                #     start_time <= reference_time
                #
                # porque un producto que todavía está siendo generado
                # al momento del frame contiene información posterior
                # al frame que estamos intentando reconstruir.
                #
                if end_time > reference_time:
                    continue

                # =====================================================
                # SELECCIÓN
                # =====================================================
                #
                # Elegimos el producto cuyo FIN de observación sea
                # más reciente sin superar el frame.
                #
                # La prioridad solo desempata productos con exactamente
                # el mismo end_time.
                #
                if (
                    best_end_time is None
                    or end_time > best_end_time
                    or (
                        end_time == best_end_time
                        and (
                            best_priority is None
                            or priority < best_priority
                        )
                    )
                ):

                    best = obj

                    best[
                        "product_prefix"
                    ] = product_prefix

                    best[
                        "product_priority"
                    ] = priority

                    best_end_time = end_time
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
