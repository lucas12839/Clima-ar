#!/usr/bin/env python3
"""ClimaAR - RainViewer ingestion FIXED."""
import json, math, shutil, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

LAT = -38.0055
LON = -62.0
ZOOM = 7
SIZE = 512
COLOR_SCHEME = 2
SMOOTH = 0
SNOW = 0

BASE = Path(__file__).resolve().parents[1]
RADAR = BASE / "data" / "radar"
HISTORY = RADAR / "historico"
ACTUAL = RADAR / "actual.png"
NOWCAST = RADAR / "radar_nowcast.json"
API = "https://api.rainviewer.com/public/weather-maps.json"
FRAMES = 6
MAX_HISTORY = 144


def api_data():
    last = None

    for n in range(3):
        try:
            r = requests.get(
                API,
                timeout=20,
                headers={
                    "Cache-Control": "no-cache",
                    "User-Agent": "ClimaAR/1.0"
                }
            )

            r.raise_for_status()
            d = r.json()

            past = d.get("radar", {}).get("past", [])

            if not past:
                raise Runtime
