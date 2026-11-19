#!/usr/bin/env python3
"""Diagnóstico 4: ¿las estaciones CWOP cercanas están al día y miden bien?

1) Escucha la red APRS-IS en modo solo lectura (permitido) durante 12 min,
   filtrando las 3 estaciones, y muestra cada transmisión con su hora.
2) Mira una vez la página de cada estación en findu (diagnóstico manual).
"""
import re
import socket
import sys
import time
from datetime import datetime, timezone

import requests

ESTACIONES = ["LU4DRH-11", "FW6154", "EW9010"]
UA = {"User-Agent": "ClimaAR/1.0 diagnostico (GitHub lucas12839/clima-ar)"}

for c in []:
    try:
        t = requests.get(f"http://wx.findu.com/cgi-bin/wxpage.cgi?call={c}&last=48", headers=UA, timeout=30).text
        plano = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t))
        print(f"\n=== findu {c} ({len(t)} bytes)\n  {plano[:1500]}")
    except Exception as exc:
        print(f"\n=== findu {c} ERROR {exc}")
    try:
        t = requests.get(f"http://map.findu.com/{c}", headers=UA, timeout=30).text
        pos = re.findall(r"(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)", t)[:3]
        print(f"  posición (map.findu): {pos}")
    except Exception as exc:
        print("  posición ERROR", exc)

print("\n=== APRS-IS (solo lectura) 12 min")
try:
    s = socket.create_connection(("rotate.aprs2.net", 14580), timeout=30)
    s.sendall(("user CLIMAAR pass -1 vers ClimaAR 1.0 filter b/" + "/".join(ESTACIONES) + "\r\n").encode())
    s.settimeout(30)
    fin = time.time() + 1080
    buf = b""
    while time.time() < fin:
        try:
            d = s.recv(4096)
        except socket.timeout:
            continue
        if not d:
            break
        buf += d
        while b"\n" in buf:
            linea, buf = buf.split(b"\n", 1)
            txt = linea.decode("latin-1").strip()
            if not txt.startswith("#"):
                print(datetime.now(timezone.utc).strftime("%H:%M:%S"), txt[:220])
    s.close()
except Exception as exc:
    print("ERROR APRS-IS", exc)
sys.exit(0)
