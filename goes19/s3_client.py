from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import xml.etree.ElementTree as ET

import requests

from .config import AWS_REGION, BUCKET

S3_ENDPOINT = f"https://{BUCKET}.s3.{AWS_REGION}.amazonaws.com"


def list_objects(
    prefix: str,
    max_keys: int = 1000,
    timeout: int = 20,
) -> list[dict[str, Any]]:
    """Lista objetos públicos del bucket NOAA GOES-19."""
    if not prefix:
        raise ValueError("prefix no puede estar vacío")

    if max_keys < 1 or max_keys > 1000:
        raise ValueError("max_keys debe estar entre 1 y 1000")

    response = requests.get(
        S3_ENDPOINT,
        params={
            "list-type": "2",
            "prefix": prefix,
            "max-keys": max_keys,
        },
        timeout=timeout,
    )
    response.raise_for_status()

    root = ET.fromstring(response.text)

    namespace = {
        "s3": "http://s3.amazonaws.com/doc/2006-03-01/"
    }

    objects = []

    for item in root.findall("s3:Contents", namespace):
        key = item.findtext(
            "s3:Key",
            default="",
            namespaces=namespace,
        )

        last_modified = item.findtext(
            "s3:LastModified",
            default="",
            namespaces=namespace,
        )

        size = item.findtext(
            "s3:Size",
            default="0",
            namespaces=namespace,
        )

        objects.append(
            {
                "key": key,
                "last_modified": last_modified,
                "size": int(size),
                "url": f"{S3_ENDPOINT}/{key}",
            }
        )

    return objects


def latest_object(
    prefix: str,
    timeout: int = 20,
) -> dict[str, Any] | None:
    """Devuelve el objeto más reciente encontrado."""
    objects = list_objects(
        prefix=prefix,
        max_keys=1000,
        timeout=timeout,
    )

    if not objects:
        return None

    return max(
        objects,
        key=lambda obj: datetime.fromisoformat(
            obj["last_modified"].replace("Z", "+00:00")
        )
        if obj["last_modified"]
        else datetime.min.replace(tzinfo=timezone.utc),
    )


def latest_matching_object(
    prefix: str,
    key_contains: str,
    timeout: int = 20,
) -> dict[str, Any] | None:
    """Devuelve el objeto más reciente cuyo nombre contiene el texto indicado."""
    objects = list_objects(
        prefix=prefix,
        max_keys=1000,
        timeout=timeout,
    )

    matches = [
        obj
        for obj in objects
        if key_contains in obj["key"]
    ]

    if not matches:
        return None

    return max(
        matches,
        key=lambda obj: datetime.fromisoformat(
            obj["last_modified"].replace("Z", "+00:00")
        )
    )
