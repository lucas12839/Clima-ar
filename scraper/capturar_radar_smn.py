#!/usr/bin/env python3
from pathlib import Path
import io, re, time
from PIL import Image
from playwright.sync_api import sync_playwright

URL='https://radares.hidricosargentina.gob.ar/'
OUT=Path('data/radar'); HIST=OUT/'historico'; ACTUAL=OUT/'actual.png'; TSFILE=OUT/'radar_timestamp.txt'
MAX_RETRIES=5

def validar_png(data):
    if not data or len(data)<1000: return False, 'archivo vacio/pequeno'
    try:
        im=Image.open(io.BytesIO(data)).convert('RGBA')
        if im.width<100 or im.height<100: return False, f'resolucion {im.width}x{im.height}'
        alpha=im.getchannel('A')
        if alpha.getextrema()[1] < 3: return False, 'imagen completamente transparente'
        visible=sum(1 for a in alpha.getdata() if a>2)
        if visible/(im.width*im.height)<0.01: return False, 'menos de 1% de pixeles visibles'
        return True, f'{im.width}x{im.height}, {len(data)} bytes'
    except Exception as e: return False, f'PNG invalido: {e}'

def extraer_url_rma10(page):
    urls=page.locator('img').evaluate_all("els => els.map(e => e.src || e.getAttribute('src') || '')")
    for u in urls:
        if '/cache/RMA10/' in u and u.lower().endswith('.png'): return u
    html=page.content()
    m=re.search(r'https?://[^\"\']*/cache/RMA10/[^\"\']+\.png|/cache/RMA10/[^\"\']+\.png',html,re.I)
    return m.group(0) if m else None

def main():
    OUT.mkdir(parents=True,exist_ok=True); HIST.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':1920,'height':1080})
        try:
            for intento in range(1,MAX_RETRIES+1):
                print(f'Intento {intento}/{MAX_RETRIES}')
                try:
                    page.goto(URL,wait_until='domcontentloaded',timeout=60000)
                    page.wait_for_timeout(10000)
                    selects=page.locator('select'); selector=None
                    for i in range(selects.count()):
                        s=selects.nth(i)
                        try: txt=s.inner_text()
                        except Exception: continue
                        if 'Bahía Blanca' in txt or 'Bahia Blanca' in txt or 'RMA10' in txt: selector=s; break
                    if selector is None: raise RuntimeError('No se encontro selector RMA10')
                    opts=selector.locator('option'); value=None
                    for i in range(opts.count()):
                        o=opts.nth(i); txt=o.inner_text().strip(); val=o.get_attribute('value')
                        if 'Bahía Blanca' in txt or 'Bahia Blanca' in txt or 'RMA10' in txt: value=val; break
                    if value is None: raise RuntimeError('No se encontro Bahía Blanca RMA10')
                    with page.expect_response(lambda r:'/cache/RMA10/' in r.url and '.png' in r.url.lower(),timeout=90000) as expected:
                        selector.select_option(value=value)
                    response=expected.value; data=response.body()
                    ok,detail=validar_png(data)
                    print(response.url); print(f'HTTP {response.status} | {detail}')
                    if not ok: raise RuntimeError(detail)
                    name=response.url.rstrip('/').split('/')[-1]
                    m=re.search(r'(\d{14})\.png$',name)
                    if not m: raise RuntimeError(f'URL sin timestamp: {name}')
                    ts=m.group(1)
                    ACTUAL.write_bytes(data); (HIST/name).write_bytes(data)
                    iso=f'{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:{ts[12:14]}Z'
                    TSFILE.write_text(iso,encoding='utf-8')
                    print(f'OK RMA10 {name} -> {iso}')
                    return
                except Exception as e:
                    print(f'Fallo: {e}')
                    if intento<MAX_RETRIES: time.sleep(4)
                    else:
                        page.screenshot(path=str(OUT/'error_rma10.png'),full_page=True)
                        raise
        finally: browser.close()

if __name__=='__main__': main()
