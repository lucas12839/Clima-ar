#!/usr/bin/env python3
"""Diagnóstico: qué redes de estaciones de la zona se pueden leer sin cuenta.

Solo prueba páginas públicas, sin iniciar sesión y sin claves. Revisa también
robots.txt (las indicaciones de cada sitio sobre lectura automática).
"""
import json
import re
import sys
import time

import requests

UA = {"User-Agent": "ClimaAR/1.0 (deteccion de tormentas, Bahia Blanca; contacto via GitHub lucas12839/clima-ar)"}
LAT, LON = -38.72, -62.27


def get(url, **kw):
    t = time.time()
    try:
        r = requests.get(url, headers=UA, timeout=30, **kw)
        return r.status_code, r.text, time.time() - t, r.headers.get("content-type", "")
    except Exception as exc:
        return None, str(exc), time.time() - t, ""


def mostrar(nombre, url, patron=None, n=600):
    code, txt, dt, ct = get(url)
    print(f"\n=== {nombre}\n{url}\n  estado={code} tipo={ct[:40]} tamaño={len(txt)} ({dt:.1f}s)")
    if patron:
        encontrados = re.findall(patron, txt)[:30]
        print("  encontrados:", encontrados)
    else:
        print("  ", re.sub(r"\s+", " ", txt[:n]))
    return code, txt


def robots(dominio):
    code, txt, _, _ = get(f"https://{dominio}/robots.txt")
    print(f"\n--- robots.txt de {dominio} (estado {code})")
    print("  ", re.sub(r"\n+", " | ", txt[:900]))


# 1. CWOP / APRS (red abierta pensada para compartir datos)
for d in ("www.findu.com",):
    robots(d)
mostrar("findu: estaciones meteorológicas cerca de Bahía Blanca",
        f"http://www.findu.com/cgi-bin/near.cgi?lat={LAT}&lon={LON}&radius=150&type=wx", r"call=([A-Z0-9-]+)")

# 2. Met Office WOW (red abierta de observaciones)
robots("wow.metoffice.gov.uk")
mostrar("WOW: sitios cerca", f"https://wow.metoffice.gov.uk/sites/sitesInArea?minLat=-39.2&maxLat=-38.0&minLon=-63.0&maxLon=-61.5")

# 3. Weathercloud (páginas públicas de estaciones)
robots("app.weathercloud.net")
mostrar("Weathercloud: búsqueda Bahía Blanca", "https://app.weathercloud.net/page/search?q=bahia+blanca", r"/d(\d{6,12})")

# 4. INTA SIGA (red oficial)
robots("siga.inta.gob.ar")
mostrar("INTA SIGA", "https://siga.inta.gob.ar/", n=400)

# 5. Weather Underground: ¿la página pública trae los datos en el HTML?
robots("www.wunderground.com")
for est in ("IBAHIA7", "IBABAHIA6", "IPUNTA96"):
    code, txt = mostrar(f"WU {est}", f"https://www.wunderground.com/dashboard/pws/{est}",
                        r'"(?:temp|precipTotal|windGust|obsTimeUtc)":\s*"?([-\d.:TZ]+)', 0)
    print("  menciona 'Offline':", "Offline" in txt)

# 6. Windy: página pública de estaciones
robots("www.windy.com")
robots("node.windy.com")
sys.exit(0)
