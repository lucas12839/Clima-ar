"""Prueba única: qué estaciones abiertas hay alrededor de Bahía Blanca (240 km).

1. BCP (Bolsa de Cereales y Productos de Bahía Blanca): lee el mapa público de
   estaciones (MapaG_NvoBCP.asp), toma cada estación con su número y ubicación y
   mira la hora de su último dato y qué variables publica. Un pedido por segundo.
2. CWOP (red abierta de estaciones ciudadanas): escucha APRS-IS en modo solo
   lectura 12 min con filtro de 250 km y anota las estaciones meteorológicas.
No guarda nada en el repo.
"""
import json
import math
import re
import socket
import time
from datetime import datetime, timedelta, timezone

import requests

BAHIA = (-38.7196, -62.2724)
UA = {"User-Agent": "ClimaAR (lectura de datos publicos)"}


def dist(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    x = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(x))


def rumbo(a, b):
    dx = (b[1] - a[1]) * 111.32 * math.cos(math.radians(a[0]))
    dy = (b[0] - a[0]) * 110.57
    return (math.degrees(math.atan2(dx, dy)) + 360) % 360


def bcp():
    print("=== BCP coordenadas ===")
    html = requests.get("https://info.bcp.org.ar/Estaciones/MapaG_NvoBCP.asp", timeout=30, headers=UA).text
    vals = {}
    for m in re.finditer(r"jsDatos\[(\d+)\]\[(\d+)\]\[(\w+)\]\s*=\s*([^;\n]+)", html):
        if m.group(2) in ("0", "1"):
            vals.setdefault(int(m.group(1)), {})[(m.group(2), m.group(3))] = m.group(4).strip()[:30]
    print("con datos:", len(vals))
    for i in sorted(vals):
        print("COORD", i, json.dumps({f"{k[0]}_{k[1]}": v for k, v in vals[i].items()}))
    if not vals:
        i = html.find("jsDatos[0][0]")
        print("MUESTRA:", re.sub(r"\s+", " ", html[i: i + 800]))


def cwop(minutos=12):
    print("\n=== CWOP (APRS-IS, solo escucha) ===")
    s = socket.create_connection(("rotate.aprs2.net", 14580), timeout=30)
    s.sendall(b"user CLIMAAR pass -1 vers ClimaAR 1.0 filter r/-38.72/-62.27/250 t/w\r\n")
    s.settimeout(30)
    fin = time.time() + minutos * 60
    buf, vistas = b"", {}
    while time.time() < fin:
        try:
            datos = s.recv(4096)
        except socket.timeout:
            continue
        if not datos:
            break
        buf += datos
        *lineas, buf = buf.split(b"\n")
        for l in lineas:
            l = l.decode("latin-1", "replace").strip()
            if l.startswith("#"):
                continue
            m = re.match(r"^([A-Z0-9-]+)>.*?[!=/@](?:\d{6}[zh/])?(\d{4}\.\d{2})([NS]).(\d{5}\.\d{2})([EW])_", l)
            if not m:
                continue
            lat = int(m[2][:2]) + float(m[2][2:]) / 60 * (1) ; lat = -lat if m[3] == "S" else lat
            lon = int(m[4][:3]) + float(m[4][3:]) / 60 ; lon = -lon if m[5] == "W" else lon
            v = vistas.setdefault(m[1], {"n": 0, "lat": round(lat, 4), "lon": round(lon, 4), "ejemplo": l[:160]})
            v["n"] += 1
    s.close()
    for k, v in sorted(vistas.items(), key=lambda kv: dist(BAHIA, (kv[1]["lat"], kv[1]["lon"]))):
        print(k, v["n"], "transmisiones", round(dist(BAHIA, (v["lat"], v["lon"]))), "km", round(rumbo(BAHIA, (v["lat"], v["lon"]))), "°", v["ejemplo"])
    print("estaciones CWOP:", len(vistas))


if __name__ == "__main__":
    try:
        bcp()
    except Exception as e:
        print("BCP ERROR", e)
