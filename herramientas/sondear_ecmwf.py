#!/usr/bin/env python3
"""Diagnóstico: qué variables en altura acepta el endpoint ECMWF de Open-Meteo."""
import json
import sys

import requests

BASE = {"latitude": -38.71, "longitude": -62.26, "forecast_hours": 3, "timezone": "UTC",
        "wind_speed_unit": "kmh"}
VARS = ["wind_speed_850hPa", "wind_direction_850hPa", "wind_speed_500hPa", "wind_direction_500hPa",
        "temperature_850hPa", "temperature_700hPa", "temperature_500hPa",
        "relative_humidity_700hPa", "relative_humidity_500hPa", "lifted_index",
        "wind_speed_700hPa", "wind_direction_700hPa", "geopotential_height_500hPa"]
for url in ("https://api.open-meteo.com/v1/ecmwf", "https://api.open-meteo.com/v1/forecast"):
    for v in VARS:
        r = requests.get(url, params=dict(BASE, hourly=v), timeout=60)
        try:
            val = r.json().get("hourly", {}).get(v)
        except Exception:
            val = r.text[:80]
        print(url.rsplit("/", 1)[1], v, r.status_code, str(val)[:80])
    r = requests.get(url, params=dict(BASE, hourly=",".join(VARS)), timeout=60)
    print(url, "TODAS", r.status_code, r.text[:200])
sys.exit(0)
