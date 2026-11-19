#!/usr/bin/env python3
"""Diagnóstico 3: red CWOP (findu) e INTA SIGA."""
import re
import sys

import requests

UA = {"User-Agent": "ClimaAR/1.0 (deteccion de tormentas, Bahia Blanca; GitHub lucas12839/clima-ar)"}


def ver(url, patron=None, n=700):
    try:
        r = requests.get(url, headers=UA, timeout=30)
        t = r.text
        print(f"\n=== {url}\n  estado {r.status_code} tamaño {len(t)}")
        if patron:
            print("  encontrados:", sorted(set(re.findall(patron, t)))[:40])
        print("  ", re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t))[:n])
        return t
    except Exception as exc:
        print(f"\n=== {url}\n  ERROR {exc}")
        return ""


for u in ("http://www.findu.com/cgi-bin/wxnear.cgi?lat=-38.72&lon=-62.27&radius=200",
          "http://www.findu.com/cgi-bin/near.cgi?lat=-38.72&lon=-62.27&radius=200",
          "http://www.findu.com/robots.txt"):
    ver(u, r"call=([A-Z0-9-]+)")
for u in ("https://siga.inta.gob.ar/document/series/listado_estaciones.json",
          "https://siga.inta.gob.ar/#/data",
          "https://siga.inta.gob.ar/api/estaciones"):
    ver(u, n=500)
sys.exit(0)
