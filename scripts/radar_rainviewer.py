#!/usr/bin/env python3
"""ClimaAR - RainViewer ingestion FIXED."""
import json, math, shutil, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

LAT = -38.0055
LON = -62.0
ZOOM = 7
SIZE = 512
COLOR_SCHEME = 2
SMOOTH = 0
SNOW = 0

BASE = Path(__file__).resolve().parents[1]
RADAR = BASE / "data" / "radar"
HISTORY = RADAR / "historico"
ACTUAL = RADAR / "actual.png"
NOWCAST = RADAR / "radar_nowcast.json"
API = "https://api.rainviewer.com/public/weather-maps.json"
FRAMES = 6
MAX_HISTORY = 144

def api_data():
    last = None
    for n in range(3):
        try:
            r = requests.get(API, timeout=20,
                             headers={"Cache-Control":"no-cache","User-Agent":"ClimaAR/1.0"})
            r.raise_for_status()
            d = r.json()
            past = d.get("radar", {}).get("past", [])
            if not past:
                raise RuntimeError("RainViewer sin frames past")
            return d
        except Exception as e:
            last = e
            time.sleep(1 + n)
    raise RuntimeError(f"RainViewer no disponible: {last}")

def url_for(data, frame):
    host = str(data.get("host", "https://tilecache.rainviewer.com")).rstrip("/")
    path = str(frame["path"])
    if not path.startswith("/"):
        path = "/" + path
    return f"{host}{path}/{SIZE}/{ZOOM}/{LAT}/{LON}/{COLOR_SCHEME}/{SMOOTH}_{SNOW}.png"

def download(data, frame):
    u = url_for(data, frame)
    r = requests.get(u, timeout=30, headers={"User-Agent":"ClimaAR/1.0"})
    r.raise_for_status()
    if not r.content.startswith(b"\x89PNG"):
        raise RuntimeError("RainViewer no devolvió PNG")
    return r.content

def components(dbz):
    mask = np.isfinite(dbz) & (dbz >= 25)
    if not np.any(mask):
        return []
    if cv2 is None:
        y,x = np.where(mask)
        return [{"id":1,"x":int(x.mean()),"y":int(y.mean()),
                 "pixels":int(len(x)),"dbz_max":float(np.nanmax(dbz[mask]))}] if len(x)>=20 else []
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out=[]
    for i in range(1,n):
        px=int(stats[i,cv2.CC_STAT_AREA])
        if px<20: continue
        vals=dbz[lab==i]
        out.append({"id":i,"x":int(round(cents[i][0])),"y":int(round(cents[i][1])),
                    "pixels":px,"dbz_max":float(np.nanmax(vals))})
    return sorted(out,key=lambda x:(x["dbz_max"],x["pixels"]),reverse=True)[:10]

# Universal Blue palette used by RainViewer. Approximate dBZ mapping.
PALETTE = np.array([
[206,192,135],[210,196,139],[214,200,143],[218,204,147],[222,208,151],
[136,221,238],[108,209,235],[81,197,232],[54,186,229],[27,174,226],
[0,163,224],[0,154,213],[0,145,202],[0,136,191],[0,127,180],
[0,119,170],[0,112,163],[0,105,156],[0,98,149],[0,91,142],
[0,85,136],[0,81,128],[0,78,120],[0,74,112],[0,71,104],
[255,238,0],[255,224,0],[255,210,0],[255,197,0],[255,183,0],
[255,170,0],[255,159,0],[255,149,0],[255,139,0],[255,129,0],
[255,68,0],[242,54,0],[230,40,0],[217,27,0],[205,13,0],
[193,0,0],[168,0,0],[143,0,0],[118,0,0],[93,0,0],
[255,170,255],[255,159,255],[255,149,255],[255,139,255],[255,129,255],
[255,119,255],[255,108,255],[255,98,255],[255,88,255],[255,78,255]
], dtype=np.float32)
DBZ=np.linspace(10,65,len(PALETTE),dtype=np.float32)

def analyze(img, ts):
    a=np.asarray(img.convert("RGBA"),dtype=np.float32)
    rgb=a[:,:,:3]
    alpha=a[:,:,3]
    flat=rgb.reshape(-1,3)
    dist=((flat[:,None,:]-PALETTE[None,:,:])**2).sum(2)
    idx=np.argmin(dist,1)
    near=np.sqrt(np.min(dist,1)).reshape(rgb.shape[:2])
    dbz=DBZ[idx].reshape(rgb.shape[:2])
    valid=(alpha>0)&(near<=60)
    dbz[~valid]=np.nan
    precip=np.isfinite(dbz)&(dbz>=10)
    vals=dbz[precip]
    return {
        "timestamp":int(ts),
        "frame_utc":datetime.fromtimestamp(int(ts),timezone.utc).isoformat(),
        "area":int(precip.sum()),
        "cobertura_pct":round(float(precip.sum()/max(1,valid.sum())*100),2),
        "dbz_max":round(float(np.max(vals)),1) if vals.size else 0,
        "dbz_mean":round(float(np.mean(vals)),1) if vals.size else 0,
        "dbz_p90":round(float(np.percentile(vals,90)),1) if vals.size else 0,
        "dbz_pixels":int(vals.size),
        "sin_precipitacion":not bool(vals.size),
        "components":components(dbz)
    }

def analyze_sequence(results):
    if not results:
        return {"estado":"sin_datos","confianza":0,"nowcast_disponible":False}
    cur=results[-1]
    if cur["area"]==0:
        return {"estado":"sin_precipitacion","confianza":100,"nowcast_disponible":False,
                "dbz_max":0,"area":0,"movimiento":{"direccion":None,"velocidad_kmh":0,"confianza":0}}
    if len(results)<2 or not cur["components"] or not results[-2]["components"]:
        return {"estado":"precipitacion_detectada","confianza":70,"nowcast_disponible":False,
                "dbz_max":cur["dbz_max"],"area":cur["area"],
                "movimiento":{"direccion":None,"velocidad_kmh":0,"confianza":0}}
    p=results[-2]["components"][0]; c=cur["components"][0]
    dx=c["x"]-p["x"]; dy=c["y"]-p["y"]
    dt=max(60,cur["timestamp"]-results[-2]["timestamp"])
    # At zoom 7 around Bahia Blanca, this is an approximate motion scale.
    km_per_px=0.9
    dist=math.hypot(dx,dy)*km_per_px
    speed=dist/(dt/3600)
    direction=(math.degrees(math.atan2(dx,-dy))+360)%360
    names=["N","NE","E","SE","S","SO","O","NO"]
    text=names[int((direction+22.5)//45)%8]
    return {"estado":"tormenta_activa" if cur["dbz_max"]>=40 else "precipitacion_activa",
            "confianza":85,"nowcast_disponible":True,"dbz_max":cur["dbz_max"],
            "dbz_mean":cur["dbz_mean"],"dbz_p90":cur["dbz_p90"],"area":cur["area"],
            "movimiento":{"direccion_grados":round(direction,1),"direccion_texto":text,
                          "velocidad_kmh":round(speed,1),"distancia_km":round(dist,1),
                          "confianza":75,"dx_pixels":dx,"dy_pixels":dy}}

def main():
    RADAR.mkdir(parents=True,exist_ok=True); HISTORY.mkdir(parents=True,exist_ok=True)
    try:
        data=api_data()
        frames=data["radar"]["past"][-FRAMES:]
        got=[]
        for f in frames:
            ts=int(f["time"]); out=HISTORY/f"radar_{ts}.png"
            try:
                if not out.exists() or out.stat().st_size<500:
                    out.write_bytes(download(data,f))
                got.append((ts,out,"rainviewer"))
            except Exception as e:
                print("Frame falló:",ts,e)
        if not got:
            raise RuntimeError("No se pudo descargar ningún frame")
    except Exception as e:
        print("RAINVIEWER FALLBACK:",e)
        cached=sorted(HISTORY.glob("radar_*.png"),key=lambda p:int(p.stem.split("_")[-1]))[-FRAMES:]
        if not cached:
            raise
        got=[(int(p.stem.split("_")[-1]),p,"fallback-local") for p in cached]
        data=None

    results=[]
    for ts,p,source in got:
        with Image.open(p) as im:
            x=analyze(im,ts)
        x["source"]=source; results.append(x)
    results.sort(key=lambda x:x["timestamp"])
    latest=results[-1]
    shutil.copyfile(HISTORY/f"radar_{latest['timestamp']}.png",ACTUAL)
    now=analyze_sequence(results)
    age=(datetime.now(timezone.utc).timestamp()-latest["timestamp"])/60
    output={"version":"8.0-rainviewer-fixed","app":"ClimaAR",
            "ubicacion":{"latitud":LAT,"longitud":LON,"ciudad":"Bahia Blanca"},
            "fuente":"RainViewer","frame_actual":latest["timestamp"],
            "frame_actual_utc":latest["frame_utc"],"frame_edad_minutos":round(age,1),
            "nowcast":now,"radar":latest,
            "diagnostico":{"frames_procesados":len(results),
                           "fallback_activo":latest["source"]=="fallback-local",
                           "frame_es_viejo":age>35}}
    NOWCAST.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    print("CLIMAAR RAINVIEWER OK")
    print("Frame:",latest["frame_utc"],"dBZ:",latest["dbz_max"],
          "fallback:",latest["source"]=="fallback-local")

if __name__=="__main__":
    main()
