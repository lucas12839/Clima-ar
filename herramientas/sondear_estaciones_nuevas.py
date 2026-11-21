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
    print("=== BCP ===")
    url = "https://info.bcp.org.ar/Estaciones/MapaG_NvoBCP.asp"
    html = requests.get(url, timeout=30, headers=UA).text
    tabla = {}
    for m in re.finditer(r"(\w*Estac\w*)\[(\d+)\]\s*=\s*([^;\n]+)", html):
        tabla.setdefault(int(m.group(2)), {})[m.group(1)] = m.group(3).strip().strip("'\"")[:60]
    campos = sorted({k for v in tabla.values() for k in v})
    print("campos:", campos)
    print("estaciones en el mapa:", len(tabla))
    filas = []
    for i, f in sorted(tabla.items()):
        lat = lon = None
        for k, v in f.items():
            if re.fullmatch(r"-3\d\.\d+", v) and lat is None and "at" in k.lower():
                lat = float(v)
            if re.fullmatch(r"-6\d\.\d+", v) and lon is None and ("on" in k.lower() or "ng" in k.lower()):
                lon = float(v)
        if lat is None:
            nums = [float(v) for v in f.values() if re.fullmatch(r"-3\d\.\d+", v)]
            lat = nums[0] if nums else None
        if lon is None:
            nums = [float(v) for v in f.values() if re.fullmatch(r"-6\d\.\d+", v)]
            lon = nums[0] if nums else None
        filas.append((i, f, lat, lon))
        print("EST", i, json.dumps(f, ensure_ascii=False))
    ahora = datetime.now(timezone(timedelta(hours=-3)))
    print("\n=== estado de cada estación habilitada ===")
    for i, f, lat, lon in filas:
        sitio = f.get("sitioEstacion")
        if not sitio or not sitio.isdigit():
            continue
        time.sleep(1.0)
        try:
            t = requests.get(f"https://info.bcp.org.ar/Estaciones/Mapa_NvoBCP.asp?Estacion={sitio}", timeout=20, headers=UA).text
        except Exception as e:
            print(i, sitio, "error", e)
            continue
        ult = re.search(r"(?:&Uacute;ltimo|Último)\s+Dato\s+Recibido.*?(\d{2}/\d{2}/\d{4}\s+\d{1,2}:\d{2}:\d{2})", t, re.S | re.I)
        edad = None
        if ult:
            dt = datetime.strptime(ult.group(1), "%d/%m/%Y %H:%M:%S").replace(tzinfo=timezone(timedelta(hours=-3)))
            edad = round((ahora - dt).total_seconds() / 60, 1)

        def val(p):
            m = re.search(p + r".*?</td>\s*<td[^>]*>(.*?)</td>", t, re.I | re.S)
            return re.sub(r"<[^>]+>|&nbsp;|\s+", " ", m.group(1)).strip()[:20] if m else None
        c = (lat, lon) if lat and lon else None
        print(json.dumps({"i": i, "sitio": sitio, "nombre": f.get("nombreEstacion") or f.get("descEstacion"),
                          "habil": f.get("habilEstacion"), "baja": f.get("bajadefEstacion"), "edad_min": edad,
                          "dist_km": round(dist(BAHIA, c)) if c else None, "rumbo": round(rumbo(BAHIA, c)) if c else None,
                          "lat": lat, "lon": lon, "temp": val(r"ltima Temperatura"),
                          "presion": val(r"Presi(?:ó|&oacute;)n"), "viento": val(r"Velocidad del Viento"),
                          "rafaga": val(r"M(?:á|&aacute;)xima R(?:á|&aacute;)faga")}, ensure_ascii=False))


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
