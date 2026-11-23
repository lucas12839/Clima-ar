#!/usr/bin/env python3
import json, re, sys, socket
from pathlib import Path
import requests
sys.path.insert(0, "scripts/estaciones")
import colector_bcp as cb

out = {"bcp": {}, "lp": {}, "simparh": {}}
for sid in range(1, 121):
    try:
        r = cb.collect(sid, f"ID{sid}")
        t = requests.get(cb.BASE_URL.format(sid), timeout=15).text
        plano = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"(?s)<(script|style).*?</\1>", " ", t)))
        i = plano.find("Estaci")
        r["texto"] = plano[:400]
        out["bcp"][sid] = r
    except Exception as e:
        out["bcp"][sid] = {"error": str(e)[:100]}

base = "https://lapampa.redesclimaticas.com"
try:
    out["lp"]["estaciones"] = requests.get(base + "/api/web/v1/site-map/stations", timeout=30).json()
except Exception as e:
    out["lp"]["error"] = str(e)
# rutas de la API en los programas de la página
rutas = set()
try:
    html = requests.get(base + "/", timeout=30).text
    for js in set(re.findall(r'/assets/[A-Za-z0-9_.-]+\.js', html)):
        t = requests.get(base + js, timeout=30).text
        rutas.update(re.findall(r'["`\'](/api/[A-Za-z0-9_/${}.:-]+)', t))
        for m in re.findall(r'import\(["`\']\./([A-Za-z0-9_.-]+\.js)', t):
            try:
                t2 = requests.get(base + "/assets/" + m, timeout=30).text
                rutas.update(re.findall(r'["`\'](/api/[A-Za-z0-9_/${}.:-]+)', t2))
            except Exception:
                pass
except Exception as e:
    out["lp"]["error_js"] = str(e)
out["lp"]["rutas"] = sorted(rutas)[:200]
codigo = None
try:
    codigo = out["lp"]["estaciones"]["stations"][0]["code"]
except Exception:
    pass
pruebas = {}
for ruta in sorted(rutas):
    if "station" in ruta.lower() and codigo:
        u = base + re.sub(r"\$\{[^}]+\}|:[a-zA-Z_]+", str(codigo), ruta)
        try:
            rr = requests.get(u, timeout=20)
            pruebas[u] = [rr.status_code, rr.text[:400]]
        except Exception as e:
            pruebas[u] = [0, str(e)[:100]]
out["lp"]["pruebas"] = pruebas

for host in ["public-map-simparh.ada.gba.gov.ar", "simparh.ada.gba.gov.ar", "ada.gba.gov.ar", "www.ada.gba.gov.ar"]:
    try:
        ip = socket.gethostbyname(host)
        rr = requests.get("https://" + host + "/", timeout=20)
        out["simparh"][host] = [ip, rr.status_code, rr.text[:500]]
    except Exception as e:
        out["simparh"][host] = ["error", str(e)[:150]]
print("SONDEO_JSON " + json.dumps(out, ensure_ascii=False, default=str))
