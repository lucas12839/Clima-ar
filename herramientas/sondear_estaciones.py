#!/usr/bin/env python3
"""Diagnóstico 2: ¿la página pública de Weather Underground trae los datos en el HTML?"""
import re
import sys

import requests

UA = {"User-Agent": "ClimaAR/1.0 (deteccion de tormentas, Bahia Blanca; GitHub lucas12839/clima-ar)"}
for est in ("IBAHIA7", "IBABAHIA6", "IPUNTA96"):
    txt = requests.get(f"https://www.wunderground.com/dashboard/pws/{est}", headers=UA, timeout=30).text
    print(f"\n=== {est} tamaño {len(txt)}")
    for clave in ("obsTimeUtc", "precipTotal", "precipRate", "windGust", "stationID", "observations",
                  "Offline", "offline", "apiKey", "api.weather.com", "lastUpdate", "neighborhood"):
        i = txt.find(clave)
        print(f"  {clave}: {'SÍ' if i >= 0 else 'no'}", re.sub(r"\s+", " ", txt[max(0, i - 80):i + 220]) if i >= 0 else "")
# Página de lluvia acumulada del barrio y la ciudad (sin cuenta)
for url in ("https://www.wunderground.com/weather/ar/bahia-blanca/IBAHIA7",
            "https://www.wunderground.com/history/daily/ar/bahia-blanca/SAZB/date/2026-11-13"):
    txt = requests.get(url, headers=UA, timeout=30).text
    print(f"\n=== {url} tamaño {len(txt)}")
    for clave in ("precipTotal", "obsTimeUtc", "observations", "Precipitation"):
        i = txt.find(clave)
        print(f"  {clave}: {'SÍ' if i >= 0 else 'no'}", re.sub(r"\s+", " ", txt[max(0, i - 60):i + 200]) if i >= 0 else "")
sys.exit(0)
