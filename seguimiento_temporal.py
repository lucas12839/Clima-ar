#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime, timezone
import json, math, re

RADAR_DIR = Path('data/radar')
RESULTADO_FILE = RADAR_DIR / 'resultado_procesamiento.json'
SEGUIMIENTO_DIR = RADAR_DIR / 'seguimiento'
ESTADO_FILE = SEGUIMIENTO_DIR / 'estado.json'
RESULTADO_SEGUIMIENTO = SEGUIMIENTO_DIR / 'seguimiento_temporal.json'
HISTORICO_DIR = RADAR_DIR / 'historico'
TIMESTAMP_FILE = RADAR_DIR / 'radar_timestamp.txt'

STATE_SCHEMA_VERSION = 2
DISTANCIA_BASE = 25.0
DISTANCIA_MAXIMA = 90.0
CAMBIO_AREA_CRECE = 15.0
CAMBIO_AREA_DEBILITA = -15.0
MOVIMIENTO_MINIMO = 5.0
MAX_HISTORIAL = 12
MAX_MISSED_FRAMES = 2
MIN_INTERVAL_MIN = 2.0
MAX_INTERVAL_MIN = 30.0


def ahora_utc():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def parse_dt(value):
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z') if dt else None


def cargar_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f'No se pudo leer {path}: {e}')
        return None


def guardar_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def obtener_timestamp_radar():
    if TIMESTAMP_FILE.exists():
        dt = parse_dt(TIMESTAMP_FILE.read_text(encoding='utf-8').strip())
        if dt:
            return dt
    candidatos = []
    for p in HISTORICO_DIR.glob('*.png'):
        m = re.search(r'(\d{14})', p.stem)
        if m:
            try:
                candidatos.append(datetime.strptime(m.group(1), '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc))
            except ValueError:
                pass
    return max(candidatos) if candidatos else None


def obtener_nucleos(resultado):
    if not isinstance(resultado, dict):
        return []
    for key in ('nucleos', 'núcleos', 'cells', 'ecos', 'detected_cells'):
        v = resultado.get(key)
        if isinstance(v, list):
            return v
    return []


def normalizar_nucleo(n):
    if not isinstance(n, dict): return None
    try:
        return {
            'x': float(n.get('centro_x', n.get('x', 0))),
            'y': float(n.get('centro_y', n.get('y', 0))),
            'pixeles': max(1, int(n.get('pixeles', n.get('area', 0)))),
            'ancho': max(1.0, float(n.get('ancho', n.get('width', 1)))),
            'alto': max(1.0, float(n.get('alto', n.get('height', 1)))),
            'intensidad': n.get('intensidad_maxima', n.get('intensity', 'DESCONOCIDA')),
            'nivel_intensidad': n.get('nivel_intensidad', 'DESCONOCIDO'),
            'estado': n.get('estado', 'DESCONOCIDO'),
            'confianza': float(n.get('confianza', 0)),
        }
    except Exception:
        return None


def distancia(a,b): return math.hypot(a['x']-b['x'], a['y']-b['y'])


def iou(a,b):
    al, ar = a['x']-a['ancho']/2, a['x']+a['ancho']/2
    at, ab = a['y']-a['alto']/2, a['y']+a['alto']/2
    bl, br = b['x']-b['ancho']/2, b['x']+b['ancho']/2
    bt, bb = b['y']-b['alto']/2, b['y']+b['alto']/2
    inter = max(0.0, min(ar,br)-max(al,bl))*max(0.0, min(ab,bb)-max(at,bt))
    union = a['ancho']*a['alto'] + b['ancho']*b['alto'] - inter
    return inter/union if union > 0 else 0.0


def umbral_distancia(minutos):
    return min(DISTANCIA_MAXIMA, max(DISTANCIA_BASE, DISTANCIA_BASE + minutos*5.0))


def compatibilidad(a,b,minutos):
    dist = distancia(a,b); limite = umbral_distancia(minutos)
    if dist > limite: return None
    aa, ab = max(1,a['pixeles']), max(1,b['pixeles'])
    ratio_area = max(aa,ab)/min(aa,ab)
    ov = iou(a,b)
    rw = max(a['ancho'],b['ancho'])/max(1,min(a['ancho'],b['ancho']))
    rh = max(a['alto'],b['alto'])/max(1,min(a['alto'],b['alto']))
    if ratio_area > 2.5 and ov < .15: return None
    if rw > 2.8 and rh > 2.0: return None
    if rw > 3.5 or rh > 3.5: return None
    if dist > max(35, limite*.65) and (ov < .08 or ratio_area > 2.0): return None
    if ov <= 0:
        if dist > 20 or ratio_area > 1.5 or rw > 1.6 or rh > 1.6: return None
    close = max(0, 1-dist/limite)
    sim_area = max(0, 1-abs(math.log(aa/ab))/math.log(2.5))
    sim_shape = max(0, 1-(abs(math.log(rw))+abs(math.log(rh)))/math.log(3.0))
    score = 35*close + 35*ov + 20*sim_area + 10*sim_shape
    if score < 45: return None
    return {'puntaje':score,'distancia':dist,'iou':ov,'cambio_area':(aa-ab)/ab*100,'ratio_area':ratio_area,'ratio_ancho':rw,'ratio_alto':rh}


def rumbo(dx,dy):
    if abs(dx)<1e-9 and abs(dy)<1e-9: return None, None
    deg = (math.degrees(math.atan2(dx, -dy)) + 360) % 360
    sectores = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSO','SO','OSO','O','ONO','NO','NNO']
    return deg, sectores[int((deg+11.25)//22.5)%16]


def crear_track(tid,a,ts):
    return {'id':tid,'primer_frame':ts,'ultimo_frame':ts,'frames':1,'missed_frames':0,'activo':True,
            'x':a['x'],'y':a['y'],'pixeles':a['pixeles'],'intensidad':a['intensidad'],
            'nivel_intensidad':a['nivel_intensidad'],'estado':'NUEVO','historial':[{'timestamp':ts,'x':a['x'],'y':a['y'],'pixeles':a['pixeles'],'intensidad':a['intensidad'],'estado':'NUEVO'}]}


def historial_limpio(track):
    raw = track.get('historial', []) if isinstance(track.get('historial', []), list) else []
    by_ts = {}
    for h in raw:
        if not isinstance(h,dict): continue
        dt = parse_dt(h.get('timestamp'))
        if not dt: continue
        h = dict(h); h['timestamp'] = iso(dt); by_ts[h['timestamp']] = h
    hist = sorted(by_ts.values(), key=lambda h: parse_dt(h['timestamp']))[-MAX_HISTORIAL:]
    track['historial'] = hist
    if hist:
        track['primer_frame'] = hist[0]['timestamp']
        track['ultimo_frame'] = hist[-1]['timestamp']
    return hist


def sanitizar_estado(state):
    if not isinstance(state,dict) or state.get('schema_version') != STATE_SCHEMA_VERSION:
        return None, 'SCHEMA_ANTERIOR'
    prev = parse_dt(state.get('radar_timestamp'))
    if not prev: return None, 'RADAR_TIMESTAMP_INVALIDO'
    tracks = []
    for t in state.get('tracks',[]):
        if not isinstance(t,dict): continue
        h = historial_limpio(t)
        if not h: continue
        # Si había campos legacy inconsistentes, se reparan desde el historial.
        t['primer_frame'] = h[0]['timestamp']; t['ultimo_frame'] = h[-1]['timestamp']
        t['activo'] = bool(t.get('activo',True))
        tracks.append(t)
    # El estado de núcleos debe representar exactamente el último radar procesado.
    nucleos=[]
    for n in state.get('nucleos',[]):
        nn=normalizar_nucleo(n)
        if nn:
            nn['track_id']=n.get('track_id')
            nucleos.append(nn)
    state = dict(state); state['tracks']=tracks; state['nucleos']=nucleos; state['radar_timestamp']=iso(prev); state['schema_version']=STATE_SCHEMA_VERSION
    return state, None


def siguiente_id(tracks):
    return max([int(t.get('id',0)) for t in tracks if str(t.get('id','')).isdigit()] or [0])+1


def actualizar_track(track,a,prev,match,minutos,ts):
    dx=a['x']-prev['x']; dy=a['y']-prev['y']; disp=math.hypot(dx,dy)
    speed=disp/minutos
    prev_speed=float(track.get('velocidad_pixeles_min',0) or 0)
    accel=(speed-prev_speed)/minutos if track.get('frames',1)>1 else None
    deg,sector=rumbo(dx,dy)
    old_deg=track.get('rumbo_grados')
    turn=None if deg is None or old_deg is None else abs((deg-old_deg+180)%360-180)
    change=match['cambio_area']
    evo='CRECIENDO' if change>=CAMBIO_AREA_CRECE else 'DEBILITANDOSE' if change<=CAMBIO_AREA_DEBILITA else 'ESTABLE'
    confidence=max(0,min(99,match['puntaje']*.75 + match['iou']*20 + (5 if disp>=MOVIMIENTO_MINIMO else 0)))
    trend='ACELERANDO' if accel is not None and accel>0.05 else 'DESACELERANDO' if accel is not None and accel<-0.05 else 'VELOCIDAD_ESTABLE'
    h=track.setdefault('historial',[])
    h.append({'timestamp':ts,'x':a['x'],'y':a['y'],'pixeles':a['pixeles'],'intensidad':a['intensidad'],'estado':evo,
              'desplazamiento_pixeles':round(disp,2),'velocidad_pixeles_min':round(speed,3),
              'rumbo_grados':round(deg,1) if deg is not None else None,'direccion':sector,
              'cambio_rumbo_grados':round(turn,1) if turn is not None else None,
              'aceleracion_pixeles_min2':round(accel,4) if accel is not None else None,
              'tendencia_velocidad':trend,'confianza_movimiento':round(confidence,1),
              'cambio_area_porcentaje':round(change,1)})
    track['historial']=h[-MAX_HISTORIAL:]
    track.update({'ultimo_frame':ts,'frames':int(track.get('frames',0))+1,'missed_frames':0,'activo':True,
                  'x':a['x'],'y':a['y'],'pixeles':a['pixeles'],'intensidad':a['intensidad'],
                  'nivel_intensidad':a['nivel_intensidad'],'estado':evo,
                  'desplazamiento_pixeles':round(disp,2),'velocidad_pixeles_min':round(speed,3),
                  'rumbo_grados':round(deg,1) if deg is not None else None,'direccion':sector,
                  'cambio_rumbo_grados':round(turn,1) if turn is not None else None,
                  'aceleracion_pixeles_min2':round(accel,4) if accel is not None else None,
                  'tendencia_velocidad':trend,'confianza_movimiento':round(confidence,1),
                  'cambio_area_porcentaje':round(change,1),'iou':round(match['iou'],3),
                  'ratio_area':round(match['ratio_area'],3)})
    return evo,disp,deg,sector,confidence,accel,turn,trend


def guardar_resultado(data): guardar_json(RESULTADO_SEGUIMIENTO,data)


def analizar():
    print('======================================\nClimaAR - SEGUIMIENTO TEMPORAL v2\n======================================')
    resultado=cargar_json(RESULTADO_FILE)
    if resultado is None: raise RuntimeError('No existe resultado_procesamiento.json')
    actuales=[n for raw in obtener_nucleos(resultado) if (n:=normalizar_nucleo(raw))]
    radar_dt=obtener_timestamp_radar(); radar_ts=iso(radar_dt)
    print(f'Núcleos actuales: {len(actuales)}'); print(f'Frame recibido: {radar_ts or "NO DISPONIBLE"}')
    state_raw=cargar_json(ESTADO_FILE)
    state,reason=sanitizar_estado(state_raw) if state_raw else (None,None)
    if state_raw and state is None: print(f'Estado anterior descartado: {reason}. Se inicia historial limpio.')
    if radar_dt is None: raise RuntimeError('No se pudo determinar el timestamp real del radar.')
    if state is None:
        tracks=[]; nucleos=[]; tid=1
        for a in actuales:
            tracks.append(crear_track(tid,a,radar_ts)); nucleos.append({**a,'track_id':tid}); tid+=1
        clean={'schema_version':STATE_SCHEMA_VERSION,'timestamp':ahora_utc(),'radar_timestamp':radar_ts,'nucleos':nucleos,'tracks':tracks}
        guardar_json(ESTADO_FILE,clean)
        guardar_resultado({'timestamp':ahora_utc(),'radar_timestamp':radar_ts,'intervalo_minutos':None,'calidad_intervalo':'PRIMER_FRAME','frame_anterior_disponible':False,
                           'nucleos_actuales':len(actuales),'nucleos_nuevos':len(actuales),'nucleos_seguimiento':0,'nucleos_creciendo':0,'nucleos_debilitandose':0,'nucleos_movimiento':0,'nucleos_estables':0,'nucleos_desaparecidos':0,'asociaciones':[],'tracks':tracks})
        print('Primer frame limpio guardado.'); return
    prev_dt=parse_dt(state['radar_timestamp']); delta=(radar_dt-prev_dt).total_seconds()/60
    print(f'Último radar procesado: {iso(prev_dt)}'); print(f'Intervalo: {delta:.1f} min')
    if delta <= 0:
        out={'timestamp':ahora_utc(),'radar_timestamp':iso(prev_dt),'frame_recibido':radar_ts,'intervalo_minutos':round(delta,2),
             'calidad_intervalo':'FRAME_REPETIDO_O_ATRASADO','motivo':'El frame recibido no es posterior al último estado procesado.',
             'frame_anterior_disponible':True,'nucleos_actuales':len(actuales),'nucleos_nuevos':0,'nucleos_seguimiento':0,'nucleos_creciendo':0,'nucleos_debilitandose':0,'nucleos_movimiento':0,'nucleos_estables':0,'nucleos_desaparecidos':0,'asociaciones':[],'tracks':state['tracks']}
        guardar_json(ESTADO_FILE,state); guardar_resultado(out); print('NO SE ACTUALIZA EL ESTADO.'); return
    if delta > MAX_INTERVAL_MIN:
        # Brecha demasiado grande: no inventar movimiento. El frame recibido pasa a ser nueva base.
        tracks=[]; nucleos=[]; tid=1
        for a in actuales:
            tracks.append(crear_track(tid,a,radar_ts)); nucleos.append({**a,'track_id':tid}); tid+=1
        clean={'schema_version':STATE_SCHEMA_VERSION,'timestamp':ahora_utc(),'radar_timestamp':radar_ts,'nucleos':nucleos,'tracks':tracks}
        guardar_json(ESTADO_FILE,clean)
        guardar_resultado({'timestamp':ahora_utc(),'radar_timestamp':radar_ts,'intervalo_minutos':round(delta,2),'calidad_intervalo':'BRECHA_DEMASIADO_GRANDE','frame_anterior_disponible':True,
                           'nucleos_actuales':len(actuales),'nucleos_nuevos':len(actuales),'nucleos_seguimiento':0,'nucleos_creciendo':0,'nucleos_debilitandose':0,'nucleos_movimiento':0,'nucleos_estables':0,'nucleos_desaparecidos':0,'asociaciones':[],'tracks':tracks})
        print('Brecha >30 min: nueva base, sin movimiento inventado.'); return
    minutos=delta
    anteriores=[]
    for raw in state.get('nucleos',[]):
        n=normalizar_nucleo(raw)
        if n: n['track_id']=raw.get('track_id'); anteriores.append(n)
    tracks=[dict(t) for t in state.get('tracks',[]) if isinstance(t,dict)]
    candidatos=[]
    for ia,a in enumerate(actuales):
        for ip,b in enumerate(anteriores):
            m=compatibilidad(a,b,minutos)
            if m is not None: candidatos.append((m['puntaje'],ia,ip,m))
    candidatos.sort(reverse=True,key=lambda z:z[0]); asign={}; usados=set()
    for _,ia,ip,m in candidatos:
        if ia not in asign and ip not in usados: asign[ia]=(ip,m); usados.add(ip)
    byid={int(t['id']):t for t in tracks if str(t.get('id','')).isdigit()}; nextid=siguiente_id(tracks)
    estado_n=[]; asociaciones=[]; nuevos=creciendo=debilitando=movimiento=estables=0
    for ia,a in enumerate(actuales):
        if ia not in asign:
            t=crear_track(nextid,a,radar_ts); tracks.append(t); byid[nextid]=t
            estado_n.append({**a,'track_id':nextid}); asociaciones.append({'track_id':nextid,'tipo':'NUEVO'}); nuevos+=1; nextid+=1; continue
        ip,m=asign[ia]; prev=anteriores[ip]; tid=prev.get('track_id')
        try: tid=int(tid)
        except Exception: tid=nextid; tracks.append(crear_track(tid,a,radar_ts)); byid[tid]=tracks[-1]; nextid+=1
        t=byid.get(tid)
        evo,disp,deg,sector,conf,accel,turn,trend=actualizar_track(t,a,prev,m,minutos,radar_ts)
        if evo=='CRECIENDO': creciendo+=1
        elif evo=='DEBILITANDOSE': debilitando+=1
        else: estables+=1
        if disp>=MOVIMIENTO_MINIMO: movimiento+=1
        estado_n.append({**a,'track_id':tid})
        asociaciones.append({'track_id':tid,'tipo':'SEGUIMIENTO','distancia_pixeles':round(m['distancia'],2),'iou':round(m['iou'],3),'puntaje':round(m['puntaje'],2),
                             'cambio_area_porcentaje':round(m['cambio_area'],1),'desplazamiento_pixeles':round(disp,2),'velocidad_pixeles_min':round(t['velocidad_pixeles_min'],3),
                             'rumbo_grados':t['rumbo_grados'],'direccion':t['direccion'],'cambio_rumbo_grados':t['cambio_rumbo_grados'],
                             'aceleracion_pixeles_min2':t['aceleracion_pixeles_min2'],'tendencia_velocidad':t['tendencia_velocidad'],'confianza_movimiento':t['confianza_movimiento']})
    desaparecidos=0
    for ip,p in enumerate(anteriores):
        if ip in usados: continue
        try: tid=int(p.get('track_id'))
        except Exception: continue
        t=byid.get(tid)
        if t:
            t['missed_frames']=int(t.get('missed_frames',0))+1; t['estado']='DESAPARECIDO'; desaparecidos+=1
            if t['missed_frames']>MAX_MISSED_FRAMES: t['activo']=False
    tracks=[t for t in tracks if t.get('activo',True)]
    clean={'schema_version':STATE_SCHEMA_VERSION,'timestamp':ahora_utc(),'radar_timestamp':radar_ts,'nucleos':estado_n,'tracks':tracks}
    guardar_json(ESTADO_FILE,clean)
    guardar_resultado({'timestamp':ahora_utc(),'radar_timestamp':radar_ts,'intervalo_minutos':round(minutos,2),'calidad_intervalo':'VALIDO',
                       'frame_anterior_disponible':True,'nucleos_actuales':len(actuales),'nucleos_nuevos':nuevos,'nucleos_seguimiento':len(asign),'nucleos_creciendo':creciendo,'nucleos_debilitandose':debilitando,'nucleos_movimiento':movimiento,'nucleos_estables':estables,'nucleos_desaparecidos':desaparecidos,'asociaciones':asociaciones,'tracks':tracks})
    print(f'Seguimiento: {len(asign)} | Nuevos: {nuevos} | Moviéndose: {movimiento} | Creciendo: {creciendo} | Debilitándose: {debilitando}')


if __name__=='__main__': analizar()
