#!/usr/bin/env python3
"""Capture the official SMN RMA10 Bahía Blanca ZH_MAX image."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import requests

PAGE_URL = os.getenv("SMN_RADAR_PAGE_URL", "https://ws2.smn.gob.ar/radar")
OVERRIDE_URL = os.getenv("SMN_RADAR_IMAGE_URL", "").strip()
RAW_DIR = Path(os.getenv("RMA10_RAW_DIR", "data/radar/raw"))
LATEST = Path(os.getenv("RMA10_LATEST_PATH", "data/radar/latest.png"))

RMA_RE = re.compile(
    r"https?://[^\"'<>\\s]+RMA10[^\"'<>\\s]+\\.png"
    r"|/[^\"'<>\\s]*RMA10[^\"'<>\\s]*\\.png",
    re.IGNORECASE,
)


class RadarParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_dict = dict(attrs)
        for key in ("src", "data-src", "href"):
            value = attrs_dict.get(key)
            if value:
                self._add(value)

    def handle_data(self, data: str) -> None:
        for match in RMA_RE.finditer(data):
            self._add(match.group(0))

    def _add(self, value: str) -> None:
        if "RMA10" in value.upper() and value.lower().endswith(".png"):
            self.urls.append(urljoin(PAGE_URL, value))


def discover_url(session: requests.Session) -> str:
    if OVERRIDE_URL:
        return OVERRIDE_URL

    response = session.get(
        PAGE_URL,
        timeout=30,
        headers={"User-Agent": "ClimaAR/2.0 (+official-SMN-radar-monitoring)"},
    )
    response.raise_for_status()

    parser = RadarParser()
    parser.feed(response.text)

    candidates = []
    for url in parser.urls:
        if url not in candidates:
            candidates.append(url)

    # Prefer the actual RMA10 ZH/COLMAX product when several RMA10 images exist.
    preferred = [
        u for u in candidates
        if any(token in u.upper() for token in ("ZH_MAX", "ZHMAX", "CMAX"))
    ]
    candidates = preferred or candidates

    if not candidates:
        # The SMN page can expose the image URL inside JavaScript rather than an img tag.
        for match in RMA_RE.finditer(response.text):
            url = urljoin(PAGE_URL, match.group(0))
            if url not in candidates:
                candidates.append(url)

    if not candidates:
        raise RuntimeError(
            "No se encontró una URL RMA10 en la página oficial del SMN. "
            "Revisar SMN_RADAR_IMAGE_URL si el sitio cambia su HTML."
        )

    return candidates[0]


def download(session: requests.Session, url: str) -> bytes:
    response = session.get(
        url,
        timeout=45,
        headers={
            "User-Agent": "ClimaAR/2.0",
            "Referer": PAGE_URL,
        },
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "image" not in content_type and not response.content.startswith(b"\x89PNG"):
        raise RuntimeError(
            f"La URL no devolvió una imagen PNG válida: "
            f"content-type={content_type!r}"
        )
    if len(response.content) < 10_000:
        raise RuntimeError("La imagen recibida es demasiado pequeña.")
    return response.content


def main() -> None:
    session = requests.Session()
    url = discover_url(session)
    content = download(session, url)

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LATEST.parent.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / f"RMA10_ZH_MAX_{timestamp}.png"
    raw_path.write_bytes(content)
    LATEST.write_bytes(content)

    metadata = RAW_DIR / f"RMA10_ZH_MAX_{timestamp}.txt"
    metadata.write_text(
        f"captured_utc={now.isoformat()}\n"
        f"source_url={url}\n"
        f"source_page={PAGE_URL}\n",
        encoding="utf-8",
    )

    print(f"RMA10 capturado: {raw_path}")
    print(f"Fuente: {url}")
    print(f"Bytes: {len(content)}")


if __name__ == "__main__":
    main()
