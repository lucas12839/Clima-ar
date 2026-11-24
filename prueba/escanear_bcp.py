"""Escanea toda la red BCP (ids 1-130): nombre, coordenadas, último dato y variables."""
import json, re, sys, time
from datetime import datetime, timezone
sys.path.insert(0, "scripts/estaciones"); sys.path.insert(0, ".")
import requests
import colector_bcp as cb
salida = []
for i in range(1, 131):
    url = cb.BASE_URL.format(i)
    try:
        r = requests.get(url, timeout=20); t = r.text
    except Exception as e:
        salida.append({"id": i, "error": str(e)[:80]}); continue
    if i == 1:
        open("muestra_bcp_1.html", "w").write(t)
    nombre = re.search(r"<title>(.*?)</title>", t, re.S | re.I)
    coords = re.findall(r"(-3[6-9]\.\d{3,}|-4[01]\.\d{3,})[^\d-]{1,40}(-6[0-6]\.\d{3,})", t)
    ult = cb.find_latest(t); dt = cb.parse_dt(ult) if ult else None
    edad = (datetime.now(timezone.utc) - dt).total_seconds() / 60 if dt else None
    salida.append({"id": i, "nombre": cb.clean(nombre.group(1)) if nombre else None,
                   "coords": coords[:2], "ultimo": ult, "edad_min": round(edad, 1) if edad is not None else None,
                   "viento": cb.find_value(t, r"Velocidad del Viento"),
                   "rafaga": cb.find_value(t, r"M[aá]xima R[aá]faga.*?15 min|M&aacute;xima R&aacute;faga.*?15 min"),
                   "lluvia": cb.find_value(t, r"Precipitaciones del d"),
                   "presion": cb.find_value(t, r"Presi[oó]n|Presi&oacute;n"),
                   "temp": cb.find_value(t, r"Última Temperatura|&Uacute;ltima Temperatura"),
                   "largo": len(t)})
    time.sleep(0.5)
json.dump(salida, open("escaneo_bcp.json", "w"), ensure_ascii=False, indent=1)
print(len(salida))
