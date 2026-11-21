"""Prueba única: qué productos carga de verdad la página pública de SINARAME.

Abre la página como un navegador común, anota cada pedido y lo que llega por la
conexión en vivo, y lista las opciones del menú. No guarda nada en el repo, no
fuerza claves ni bloqueos y hace muy pocos pedidos.
"""
import json
import re
import sys
from collections import Counter

from playwright.sync_api import sync_playwright

URL = "https://radares.hidricosargentina.gob.ar/"
PALABRAS = re.compile(r"VRAD|VEL|DOPPLER|ZDR|KDP|RHOHV|WRAD|COLMAX|ZH|TH|CM|PROD|product", re.I)
pedidos, mensajes = [], []


def main():
    canal = sys.argv[1] if len(sys.argv) > 1 else None
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, channel=canal) if canal else p.chromium.launch(headless=True)
        page = b.new_page()
        page.on("response", lambda r: pedidos.append(
            (r.status, r.request.resource_type, r.headers.get("content-type", ""), r.url)))

        def ws(w):
            mensajes.append(("ABRE", w.url))
            w.on("framereceived", lambda d: mensajes.append(("<-", str(d)[:3000])))
            w.on("framesent", lambda d: mensajes.append(("->", str(d)[:1500])))
        page.on("websocket", ws)

        page.goto(URL, wait_until="networkidle", timeout=90000)
        page.wait_for_timeout(15000)

        print("=== TEXTOS DEL MENÚ (botones, opciones, etiquetas) ===")
        textos = page.eval_on_selector_all(
            "button, option, label, a, li, [role=option], [role=menuitem], select",
            "els => els.map(e => (e.innerText || e.value || '').trim()).filter(t => t)")
        for t in sorted(set(textos)):
            print(" ", t.replace("\n", " | ")[:150])

        # Elegir el RMA10 (Bahía Blanca) como haría una persona, si aparece.
        for i, sel in enumerate(page.query_selector_all("select")):
            ops = sel.eval_on_selector_all("option", "o => o.map(x => [x.value, x.innerText])") if False else \
                page.evaluate("s => [...s.options].map(o => [o.value, o.text])", sel)
            print(f"select {i}:", ops[:30])
            for v, t in ops:
                if "RMA10" in t:
                    sel.select_option(value=v)
                    page.wait_for_timeout(20000)
                    print("elegido", t)
        js = page.evaluate("fetch('/js/radar.bwas0cb2os.js').then(r => r.text())")
        print("radar.js palabras:", sorted(set(PALABRAS.findall(js))))
        print("radar.js rutas:", sorted(set(re.findall(r"['\"`]([^'\"`]*(?:cache|png|api|/)[^'\"`]*)['\"`]", js)))[:60])
        textos2 = page.eval_on_selector_all(
            "button, option, label, a, li, [role=option], select",
            "els => els.map(e => (e.innerText || e.value || '').trim()).filter(t => t)")
        nuevos = sorted(set(textos2) - set(textos))
        print("=== OPCIONES NUEVAS TRAS ELEGIR EL RADAR ===")
        for t in nuevos:
            print(" ", t.replace("\n", " | ")[:150])
        b.close()

    print("\n=== PEDIDOS (sin repetir) ===")
    vistos = Counter()
    for st, tipo, ct, url in pedidos:
        clave = re.sub(r"\d{14}", "<FECHA>", url)
        vistos[(st, tipo, ct.split(";")[0], clave)] += 1
    for (st, tipo, ct, url), n in sorted(vistos.items(), key=lambda x: x[0][3]):
        print(f"  {st} {tipo:10s} {ct:28s} x{n} {url[:220]}")

    print("\n=== CONEXIÓN EN VIVO ===")
    print("mensajes:", len(mensajes))
    for d, m in mensajes[:40]:
        print(f"  {d} {m[:600]}")
    print("\n=== PALABRAS DE PRODUCTOS EN LA CONEXIÓN EN VIVO ===")
    c = Counter(w.upper() for _, m in mensajes for w in PALABRAS.findall(m))
    print(json.dumps(c, ensure_ascii=False))
    carpetas = sorted(set(re.findall(r"cache/[\w\-]+(?:/[\w\-]+)?", " ".join(m for _, m in mensajes) + " ".join(u for *_, u in pedidos))))
    print("carpetas cache:", carpetas)


if __name__ == "__main__":
    main()
