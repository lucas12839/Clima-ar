#!/usr/bin/env python3
"""Sondeo: qué datos piden las páginas de SIMPARH (PBA) y Redes Climáticas La Pampa, y estado de estaciones BCP."""
import json, re, sys, time
from playwright.sync_api import sync_playwright
import requests

URLS = ["https://public-map-simparh.ada.gba.gov.ar/", "https://lapampa.redesclimaticas.com/"]

def sondear(url, page):
    vistos = []
    def resp(r):
        try:
            ct = r.headers.get("content-type", "")
            if any(x in ct for x in ("json", "xml", "csv", "text/plain")) or re.search(r"api|json|estacion|station|dato|geo", r.url, re.I):
                body = ""
                try:
                    body = r.text()[:600]
                except Exception:
                    pass
                vistos.append({"url": r.url, "status": r.status, "tipo": ct, "muestra": body})
        except Exception as e:
            pass
    page.on("response", resp)
    try:
        page.goto(url, wait_until="networkidle", timeout=90000)
    except Exception as e:
        print("GOTO", url, e)
    time.sleep(8)
    texto = page.inner_text("body")[:3000] if page.query_selector("body") else ""
    links = page.eval_on_selector_all("a", "els => els.map(e => e.href)")[:200]
    page.remove_listener("response", resp)
    return {"url": url, "peticiones": vistos[:80], "texto": texto, "links": links}

out = {}
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    for u in URLS:
        out[u] = sondear(u, pg)
    b.close()

# BCP: las 11 que faltan (y su estado ahora)
bcp = {}
for n in range(1, 121):
    try:
        t = requests.get(f"https://info.bcp.org.ar/Estaciones/Mapa_NvoBCP.asp?Estacion={n}", timeout=20).text
    except Exception as e:
        continue
    nombre = re.search(r"Estaci[oó]n[^<]*?:?\s*</?[^>]*>\s*([^<]{3,40})<", t)
    fecha = re.search(r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})", t)
    lluvia = re.search(r"(Precipitaci[oó]n|Lluvia)[^<]*</td>\s*<td[^>]*>([^<]*)<", t, re.I | re.S)
    titulo = re.search(r"<title>([^<]*)</title>", t, re.I)
    bcp[n] = {"titulo": titulo.group(1).strip() if titulo else None,
              "nombre": nombre.group(1).strip() if nombre else None,
              "fecha": fecha.group(1) if fecha else None,
              "lluvia": lluvia.group(2).strip() if lluvia else None,
              "largo": len(t), "muestra": re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t))[:300]}
out["bcp"] = bcp
print("SONDEO_JSON " + json.dumps(out, ensure_ascii=False))
