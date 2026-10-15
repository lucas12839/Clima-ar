#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path('.')
HIST = ROOT / 'data' / 'radar' / 'historico'
OUT = ROOT / 'data' / 'radar' / 'rma10_dataset_historico.csv'
SUMMARY = ROOT / 'data' / 'radar' / 'rma10_dataset_resumen.json'

TIMESTAMP_RE = re.compile(r'^(\d{14})\.png$')
RADAR_LAT = -38.7342
RADAR_LON = -62.1654
BAHIA_LAT = -38.7196
BAHIA_LON = -62.2724
RANGE_KM = 240.0
MAX_MATCH_KM = 55.0
MAX_MISSED = 2
HORIZONS = (5, 10, 20, 30)

NIVELES = {0:'FONDO',1:'MUY_DEBIL',2:'DEBIL',3:'MODERADA',4:'FUERTE',5:'MUY_FUERTE',6:'SEVERA'}

def clasificar_color(r,g,b):
    if max(r,g,b) < 25: return 0
    if b > g and g > r*1.5: return 1
    if g > r*1.35 and g >= b*0.95: return 2
    if r > 100 and g > 100 and b < 100 and abs(r-g) < 100: return 3
    if r > 130 and 60 < g < 180 and b < 90 and r > g*1.15: return 4
    if r > 130 and r > g*1.35 and r > b*1.35: return 5
    if r > 100 and b > 100 and r > g*1.3: return 6
    return 0

def detect(path):
    im=Image.open(path).convert('RGB')
    w,h=im.size
    pix=im.load(); cls=[[0]*w for _ in range(h)]
    for y in range(h):
        row=cls[y]
        for x in range(w):
            r,g,b=pix[x,y]; row[x]=clasificar_color(r,g,b)
    seen=bytearray(w*h); out=[]; neigh=((-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1))
    for y0 in range(h):
        for x0 in range(w):
            idx=y0*w+x0
            if seen[idx] or cls[y0][x0] <= 0: seen[idx]=1; continue
            q=deque([(x0,y0)]); seen[idx]=1; pts=[]
            while q:
                x,y=q.popleft(); pts.append((x,y))
                for dx,dy in neigh:
                    nx,ny=x+dx,y+dy
                    if 0<=nx<w and 0<=ny<h:
                        ni=ny*w+nx
                        if not seen[ni] and cls[ny][nx]>0:
                            seen[ni]=1; q.append((nx,ny))
            if len(pts)<10: continue
            xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
            inten=max(cls[y][x] for x,y in pts)
            out.append({'x':sum(xs)/len(xs),'y':sum(ys)/len(ys),'pixeles':len(pts),'ancho':max(xs)-min(xs)+1,'alto':max(ys)-min(ys)+1,'intensidad':inten})
    return w,h,out

def ts_from(path):
    m=TIMESTAMP_RE.match(path.name)
    if not m:return None
    return datetime.strptime(m.group(1),'%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)

def xy_to_geo(x,y,w,h):
    # Same nominal 240 km radius geometry used by ClimaAR georeferencing.
    cx,cy=w/2.0,h/2.0
    dx=(x-cx)/max(w/2.0,1)*RANGE_KM
    dy=(cy-y)/max(h/2.0,1)*RANGE_KM
    km_lat=110.574; km_lon=111.320*math.cos(math.radians(RADAR_LAT))
    return RADAR_LAT+dy/km_lat, RADAR_LON+dx/km_lon

def distance_km(lat1,lon1,lat2,lon2):
    r=6371.0; p1=math.radians(lat1); p2=math.radians(lat2); dp=math.radians(lat2-lat1); dl=math.radians(lon2-lon1)
    a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(min(1,math.sqrt(a)))

def bearing(lat1,lon1,lat2,lon2):
    p1=math.radians(lat1); p2=math.radians(lat2); dl=math.radians(lon2-lon1)
    y=math.sin(dl)*math.cos(p2); x=math.cos(p1)*math.sin(p2)-math.sin(p1)*math.cos(p2)*math.cos(dl)
    return (math.degrees(math.atan2(y,x))+360)%360

def angle_diff(a,b): return abs((a-b+180)%360-180)

def build_tracks(frames):
    tracks=[]; next_id=1
    for fi,fr in enumerate(frames):
        nuclei=fr['nuclei']; dt=fr['dt']; w=fr['w']; h=fr['h']
        for n in nuclei: n['track_id']=None
        candidates=[]
        for tr in tracks:
            if tr['active'] and tr['missed']<=MAX_MISSED:
                last=tr['obs'][-1]; elapsed=(dt-last['dt']).total_seconds()/60
                if elapsed<=0 or elapsed>60: continue
                if len(tr['obs'])>=2:
                    prev=tr['obs'][-2]; eprev=(last['dt']-prev['dt']).total_seconds()/60
                    vx=(last['x']-prev['x'])/eprev if eprev>0 else 0
                    vy=(last['y']-prev['y'])/eprev if eprev>0 else 0
                    px=last['x']+vx*elapsed; py=last['y']+vy*elapsed
                else: px,py=last['x'],last['y']
                maxpix=max(20.0, min(180.0, 20+elapsed*4))
                for j,n in enumerate(nuclei):
                    d=math.hypot(n['x']-px,n['y']-py)
                    if d<=maxpix:
                        candidates.append((d,j,tr))
        used_n=set(); used_t=set()
        for d,j,tr in sorted(candidates,key=lambda z:z[0]):
            if j in used_n or tr['id'] in used_t: continue
            n=nuclei[j]; n['track_id']=tr['id']; tr['obs'].append({'dt':dt,**n}); tr['missed']=0; used_n.add(j); used_t.add(tr['id'])
        for tr in tracks:
            if tr['id'] not in used_t and tr['active']:
                tr['missed']+=1
                if tr['missed']>MAX_MISSED: tr['active']=False
        for n in nuclei:
            if n['track_id'] is None:
                tr={'id':next_id,'active':True,'missed':0,'obs':[{'dt':dt,**n}]}; tracks.append(tr); n['track_id']=next_id; next_id+=1
        fr['tracks']={n['track_id']:n for n in nuclei}
    return tracks

def nearest_future(obs, target, max_error):
    best=None; besterr=None
    for o in obs:
        if o['dt']<=target: continue
        err=abs((o['dt']-target).total_seconds()/60)
        if err<=max_error and (besterr is None or err<besterr): best=o; besterr=err
    return best

def main():
    files=[]
    for p in HIST.glob('*.png'):
        dt=ts_from(p)
        if dt is not None: files.append((dt,p))
    files.sort()
    valid=[]; skipped=0
    for dt,p in files:
        try:
            im=Image.open(p).convert('RGB')
            if len(im.getcolors(maxcolors=10000000) or []) < 5: skipped += 1; continue
            w,h,n=detect(p)
            valid.append({'dt':dt,'path':p.name,'w':w,'h':h,'nuclei':n})
        except Exception: skipped+=1
    if len(valid)<3: raise SystemExit('No hay suficientes frames RMA10 validos.')
    build_tracks(valid)
    rows=[]
    for tr in build_tracks(valid):
        obs=tr['obs']
        for i,o in enumerate(obs):
            lat,lon=xy_to_geo(o['x'],o['y'],valid[i if False else 0]['w'],valid[i if False else 0]['h'])
            # Use the dimensions stored on the observation's frame.
            fr=next((f for f in valid if f['dt']==o['dt']),valid[0]); lat,lon=xy_to_geo(o['x'],o['y'],fr['w'],fr['h'])
            dist=distance_km(lat,lon,BAHIA_LAT,BAHIA_LON); rumbo=bearing(lat,lon,BAHIA_LAT,BAHIA_LON)
            for horizon in HORIZONS:
                target=o['dt'].__class__.fromtimestamp(o['dt'].timestamp()+horizon*60,tz=timezone.utc)
                fut=nearest_future(obs,target,max(5.0,min(15.0,horizon)))
                if not fut: continue
                delta=fut['intensidad']-o['intensidad']; dtm=(fut['dt']-o['dt']).total_seconds()/60
                dx=fut['x']-o['x']; dy=fut['y']-o['y']; speed=math.hypot(dx,dy)/dtm if dtm>0 else 0
                flat,bearing_motion=xy_to_geo(o['x']+dx,o['y']+dy,fr['w'],fr['h']),None
                blat,blon=xy_to_geo(fut['x'],fut['y'],fr['w'],fr['h'])
                motion_bearing=(math.degrees(math.atan2(dx,-dy))+360)%360
                rows.append({'frame_utc':o['dt'].isoformat().replace('+00:00','Z'),'track_id':tr['id'],'horizonte_min':horizon,'x':round(o['x'],3),'y':round(o['y'],3),'area_px':o['pixeles'],'width_px':o['ancho'],'height_px':o['alto'],'intensity':o['intensidad'],'lat':round(lat,6),'lon':round(lon,6),'distance_km':round(dist,3),'bearing_to_bahia_deg':round(rumbo,2),'future_intensity':fut['intensidad'],'intensity_delta':delta,'strengthened':int(delta>=1),'weakened':int(delta<=-1),'future_x':round(fut['x'],3),'future_y':round(fut['y'],3),'displacement_px':round(math.hypot(dx,dy),3),'speed_px_min':round(speed,4),'motion_bearing_deg':round(motion_bearing,2),'toward_bahia':int(angle_diff(motion_bearing,rumbo)<=45 and speed>0.05),'future_distance_km':round(distance_km(blat,blon,BAHIA_LAT,BAHIA_LON),3),'label_valid':1,'source':'SMN RMA10'})
    OUT.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0].keys()) if rows else []
    with OUT.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary={'frames_encontrados':len(files),'frames_validos':len(valid),'frames_descartados':skipped,'registros_entrenamiento':len(rows),'horizontes':HORIZONS,'tracks':len({r['track_id'] for r in rows}),'strengthened':sum(r['strengthened'] for r in rows),'weakened':sum(r['weakened'] for r in rows),'toward_bahia':sum(r['toward_bahia'] for r in rows),'generated_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}
    SUMMARY.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
