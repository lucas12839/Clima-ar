#!/usr/bin/env python3
"""Publicaciones automáticas de ClimaAR en su página de Facebook (24/11/2026).

Corre en el VPS cada 5 minutos (deploy/climaar-facebook.timer). Lee los datos
del servidor de la app que corre en el mismo VPS (http://127.0.0.1:8000) y:

1. Todas las mañanas (desde las 7:30, hora de Argentina) publica cómo viene
   el día, y todas las noches (desde las 20:30) cómo viene mañana, con imagen.
2. Con algo CONFIRMADO en las localidades (reglas en la sección "Tormentas", más abajo:
   tormenta con radar y rayos, lluvia fuerte con radar y estación, o viento
   fuerte medido 15 min seguidos), publica una "ADVERTENCIA"
   con la imagen del radar (nunca la palabra "alerta") y va informando con
   cada actualización del radar, sin llenar la página (24/11/2026):
   - si algo cambió (localidad nueva, nivel más alto o tormenta que ahora va
     hacia una localidad): una nueva, con 30 min como mínimo entre una y otra;
   - si sigue igual: una actualización cada 60 min (fuerte o severa) o cada
     90 min (moderada);
   - como máximo 12 por día (las de nivel 4, posiblemente severa, no tienen tope);
   - cuando se van, una última: "se alejaron las tormentas".
   Nunca con un radar de más de 20 min (no se informa con datos viejos).

Cada texto lleva el enlace a la página oficial de ClimaAR (sin enlace al SMN).
A Make se le mandan tres campos: mensaje, imagen_url y tipo ("manana" o
"advertencia").

Publica a través de Make.com (servicio que ya tiene permiso de Meta para
publicar en páginas): ClimaAR le manda el texto y la dirección de la imagen,
y Make lo publica en la página. La imagen la sirve la página de ClimaAR en
/publicaciones/ (deploy/Caddyfile).

Configuración (/etc/climaar-facebook.env, solo la lee root):
  MAKE_WEBHOOK_URL=...  dirección privada del escenario de Make (no va al repositorio)
  FB_ACTIVO=1           0 = no publica nada (prueba)
Estado: /var/lib/climaar-facebook/estado.json (qué y cuándo se publicó).

Uso a mano:
  python scripts/facebook_climaar.py --probar --salida /tmp/fb   # arma texto e imagen, no publica
  python scripts/facebook_climaar.py --forzar manana             # publica la de la mañana ya
  python scripts/facebook_climaar.py --forzar advertencia        # publica la advertencia ya (si hay tormenta)
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
TZ_AR = timezone(timedelta(hours=-3))
API = os.environ.get("CLIMAAR_API", "http://127.0.0.1:8000").rstrip("/")
URL_PUBLICA = os.environ.get("CLIMAAR_URL_PUBLICA", "https://climaar-bahiablanca.duckdns.org")
ESTADO = Path(os.environ.get("CLIMAAR_FB_ESTADO", "/var/lib/climaar-facebook/estado.json"))

HORA_MANANA = (7, 30)
HORA_MANANA_LIMITE = 11            # pasadas las 11 ya no se publica la de la mañana
HORA_NOCHE = (20, 30)
HORA_NOCHE_LIMITE = 23
RADAR_MAX_EDAD_MIN = 20
NIVEL_MINIMO = 2
MAX_DIA = 12                       # advertencias por día (nivel 4 sin tope)
ESPERA_MIN = 30                    # mínimo entre dos advertencias
REPETIR_FUERTE_MIN = 60            # sigue igual: actualización (fuerte o severa)
REPETIR_MODERADO_MIN = 90          # sigue igual: actualización (moderada)
FIN_SIN_NADA_MIN = 20              # "terminó" recién tras 20 min seguidos sin nada
FIN_DENTRO_DE_H = 6                # "se alejaron" solo si hubo advertencia en las últimas 6 h
# Sin enlace a las alertas del SMN (decisión de Lucas, 24/11/2026: el SMN
# declara alerta para una zona entera y ClimaAR analiza cada localidad).
# Cada publicación lleva el enlace a la página de ClimaAR.
PIE_ENLACE = "Página oficial de ClimaAR: " + URL_PUBLICA
# 25/11/2026 (Lucas): sin "no reemplaza los avisos oficiales"; en los informes,
# el aviso de que ClimaAR no pide dinero; hashtags sin #PuntaAlta.
PIE = ""
PIE_INFORME = ("ClimaAR no pide dinero ni donaciones. Somos autónomos. "
               "Nos ayudaría mucho compartiendo y siguiéndonos. ¡Gracias! ✅")
HASHTAGS = "#BahíaBlanca #Clima #Tiempo"
# Con tormentas (advertencias, y los informes si hay actividad): breve y con esto al final.
SEGUIR = ("Si querés seguir informado, no te olvides de seguir nuestra página de Facebook o nuestra "
          "página oficial. Vigilamos el tiempo las 24 horas: " + URL_PUBLICA)
TIPO_ADV = "advertencia"
TIPO_FIN = "fin"
TIPOS_TORMENTA = (TIPO_ADV, "tormenta", TIPO_FIN)


def sin_alerta(texto: str) -> str:
    """Decisión de Lucas: nunca la palabra "alerta" (se usa "advertencia")."""
    def cambiar(m):
        w = m.group(0)
        nuevo = "advertencia" + ("s" if w.lower().endswith("s") else "")
        if w.isupper():
            return nuevo.upper()
        return nuevo.capitalize() if w[0].isupper() else nuevo
    return re.sub(r"\balertas?\b", cambiar, texto, flags=re.IGNORECASE)

NIVEL_TXT = {1: "débil", 2: "moderado", 3: "fuerte", 4: "severo"}
PUNTOS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
NOMBRE_PUNTO = {"N": "norte", "NNE": "norte", "NE": "noreste", "ENE": "este", "E": "este", "ESE": "este",
                "SE": "sureste", "SSE": "sur", "S": "sur", "SSO": "sur", "SO": "suroeste", "OSO": "oeste",
                "O": "oeste", "ONO": "oeste", "NO": "noroeste", "NNO": "norte"}
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
         "octubre", "noviembre", "diciembre"]


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
ESPERAS_SERVIDOR = (15, 30)        # el servidor se reinicia al actualizarse: se espera y reintenta


def _pedir(ruta: str, timeout: float, **kw):
    import time

    import requests

    for espera in (*ESPERAS_SERVIDOR, None):
        try:
            r = requests.get(API + ruta, timeout=timeout, **kw)
            if r.status_code < 500:
                r.raise_for_status()
                return r
            if espera is None:
                r.raise_for_status()
        except requests.ConnectionError:
            if espera is None:
                raise
        time.sleep(espera)


def pedir_json(ruta: str, timeout: float = 30.0):
    return _pedir(ruta, timeout, headers={"Accept": "application/json"}).json()


def pedir_bytes(ruta: str, timeout: float = 30.0) -> bytes:
    return _pedir(ruta, timeout).content


def num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def fmt(v, d=0):
    x = num(v)
    if x is None:
        return "—"
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def punto(grados):
    g = num(grados)
    return None if g is None else PUNTOS[round((g % 360) / 22.5) % 16]


def ahora_ar(ahora: datetime | None = None) -> datetime:
    return (ahora or datetime.now(timezone.utc)).astimezone(TZ_AR)


def hora_txt(dt: datetime) -> str:
    return dt.astimezone(TZ_AR).strftime("%H:%M")


def fecha_txt(dt: datetime) -> str:
    dt = dt.astimezone(TZ_AR)
    return f"{DIAS[dt.weekday()]} {dt.day} de {MESES[dt.month - 1]}"


def cielo(codigo) -> str:
    c = num(codigo)
    if c is None:
        return ""
    c = int(c)
    if c >= 95:
        return "tormentas"
    if c >= 80:
        return "chaparrones"
    if 71 <= c <= 77:
        return "nieve"
    if c >= 51:
        return "lluvia"
    if c >= 45:
        return "niebla"
    return {0: "despejado", 1: "mayormente despejado", 2: "parcialmente nublado", 3: "nublado"}.get(c, "")


# ----------------------------------------------------------------------------
# Estado (qué se publicó)
# ----------------------------------------------------------------------------
def leer_estado() -> dict:
    try:
        e = json.loads(ESTADO.read_text(encoding="utf-8"))
        return e if isinstance(e, dict) else {}
    except (OSError, ValueError):
        return {}


def guardar_estado(e: dict, ahora: datetime | None = None) -> None:
    ESTADO.parent.mkdir(parents=True, exist_ok=True)
    tmp = ESTADO.with_suffix(".tmp")
    # Solo las publicaciones de los últimos 3 días.
    limite = ((ahora or datetime.now(timezone.utc)) - timedelta(days=3)).isoformat()
    e["publicaciones"] = [p for p in e.get("publicaciones", []) if p.get("utc", "") >= limite]
    tmp.write_text(json.dumps(e, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, ESTADO)


# ----------------------------------------------------------------------------
# Mañana
# ----------------------------------------------------------------------------
def _toca_franja(estado: dict, ahora: datetime, desde: tuple, hasta: int, clave: str) -> bool:
    local = ahora_ar(ahora)
    if (local.hour, local.minute) < desde or local.hour >= hasta:
        return False
    return estado.get(clave) != local.date().isoformat()


def toca_manana(estado: dict, ahora: datetime) -> bool:
    return _toca_franja(estado, ahora, HORA_MANANA, HORA_MANANA_LIMITE, "manana_fecha")


def toca_noche(estado: dict, ahora: datetime) -> bool:
    return _toca_franja(estado, ahora, HORA_NOCHE, HORA_NOCHE_LIMITE, "noche_fecha")


def manana_de(pron: dict, ahora: datetime) -> dict | None:
    """El pronóstico de mañana (el día siguiente al de hoy en Argentina)."""
    fecha = (ahora_ar(ahora).date() + timedelta(days=1)).isoformat()
    return next((d for d in (pron or {}).get("dias") or [] if d.get("fecha") == fecha), None)


def _lineas_dia(dia: dict) -> list[str]:
    lineas = []
    c = cielo(dia.get("codigo_tiempo"))
    lineas.append(f"🌡️ Máxima {fmt(dia.get('maxima'))}° y mínima {fmt(dia.get('minima'))}°.")
    if c:
        lineas.append(f"☁️ Cielo: {c}.")
    prob = num(dia.get("prob_lluvia"))
    mm = num(dia.get("lluvia_mm"))
    if prob is not None:
        lineas.append(f"☔ Probabilidad de lluvia: {fmt(prob)} %"
                      + (f" (unos {fmt(mm, 1)} mm)" if mm and mm >= 0.5 else "") + ".")
    dirv = punto(dia.get("viento_direccion"))
    lineas.append(f"💨 Viento del {NOMBRE_PUNTO.get(dirv, '—')} hasta {fmt(dia.get('viento'))} km/h,"
                  f" ráfagas de {fmt(dia.get('rafagas'))} km/h.")
    return lineas


CALCULAR = object()   # "sit" no recibido: se calcula con situacion()
QUE_HAY = {"tormenta": "Tormenta", "lluvia": "Lluvia fuerte", "viento": "Viento fuerte"}


def _activos(resumen: dict, sit=CALCULAR) -> list[str]:
    """Lo confirmado ahora (mismo criterio que las advertencias), con su nombre.
    25/11/2026: decía "Tormentas ahora en..." con cualquier nivel 2, aunque
    fuera solo viento."""
    edad = num((resumen or {}).get("frame_edad_minutos"))
    if sit is CALCULAR:
        sit = situacion(resumen)
    if edad is None or edad > RADAR_MAX_EDAD_MIN:
        sit = None
    if not sit:
        return []
    lineas = []
    for clase in CLASES:
        locs = [l for l in sit["localidades"] if l["clase"] == clase]
        if locs:
            lineas.append(f"⚠️ {QUE_HAY[clase]} ahora en: " + ", ".join(
                f"{l['nombre']} (nivel {l['nivel']})" for l in locs) + ".")
    return lineas + [""]


def texto_manana(resumen: dict, pron: dict, ahora: datetime, sit=CALCULAR) -> str:
    """Misma forma que el informe de la noche (25/11/2026, Lucas)."""
    hoy = (pron or {}).get("hoy") or {}
    t = (resumen or {}).get("tiempo_actual") or {}
    lineas = [f"Buen día, Bahía Blanca. Así viene el {fecha_txt(ahora)}:", ""]
    if hoy:
        dia = _lineas_dia(hoy)
        if num(t.get("temperatura")) is not None and t.get("disponible"):
            dia[0] = dia[0].replace("🌡️ ", f"🌡️ Ahora {fmt(t.get('temperatura'))}°. ", 1)
        lineas += dia
    elif num(t.get("temperatura")) is not None and t.get("disponible"):
        lineas.append(f"🌡️ Ahora {fmt(t.get('temperatura'))}°.")
    # Con tormentas, lo primero que se lee es dónde están (breve e informativo).
    activos = _activos(resumen, sit)
    lineas[2:2] = activos
    lineas += ["", SEGUIR if activos else PIE_ENLACE, "", PIE_INFORME, HASHTAGS]
    return "\n".join(lineas)


# Informe de la noche (25/11/2026): cómo sigue el tiempo ESTA noche y la
# madrugada (de 21 a 7), así complementa al de la mañana (día y tarde) en vez
# de repetirlo.
NOCHE_DESDE_H = 21
NOCHE_HASTA_H = 7


def horas_noche(pron: dict, ahora: datetime) -> list[dict]:
    """Las horas de 21 a 7 (hora argentina) de esta noche, con su hora local."""
    hoy = ahora_ar(ahora).date()
    desde = datetime(hoy.year, hoy.month, hoy.day, NOCHE_DESDE_H)
    hasta = desde + timedelta(hours=24 - NOCHE_DESDE_H + NOCHE_HASTA_H)
    salida = []
    for h in (pron or {}).get("horas") or []:
        try:
            t = datetime.fromisoformat(str(h.get("hora")))
        except ValueError:
            continue
        if desde <= t <= hasta:
            salida.append(dict(h, _t=t))
    return salida


def resumen_noche(horas: list[dict]) -> dict | None:
    if len(horas) < 6:
        return None

    def vals(k):
        return [num(h.get(k)) for h in horas if num(h.get(k)) is not None]

    temps = [(num(h.get("temperatura")), h["_t"]) for h in horas if num(h.get("temperatura")) is not None]
    if not temps:
        return None
    tmin, hmin = min(temps, key=lambda x: x[0])

    def a_la(hora):
        return next((num(h.get("temperatura")) for h in horas if h["_t"].hour == hora), None)

    mitad = len(horas) // 2
    cod1 = [num(h.get("codigo_tiempo")) for h in horas[:mitad] if num(h.get("codigo_tiempo")) is not None]
    cod2 = [num(h.get("codigo_tiempo")) for h in horas[mitad:] if num(h.get("codigo_tiempo")) is not None]
    moda = lambda xs: max(set(xs), key=xs.count) if xs else None  # noqa: E731
    viento = vals("viento")
    raf = [(num(h.get("rafagas")), h["_t"]) for h in horas if num(h.get("rafagas")) is not None]
    rmax, hraf = max(raf, key=lambda x: x[0]) if raf else (None, None)
    v1 = vals("viento")[:mitad]
    v2 = vals("viento")[mitad:]
    tendencia = None
    if v1 and v2:
        m1, m2 = sum(v1) / len(v1), sum(v2) / len(v2)
        tendencia = "aumenta de madrugada" if m2 - m1 >= 8 else "se calma de madrugada" if m1 - m2 >= 8 else None
    dirs = vals("viento_direccion")
    return {
        "minima": tmin, "hora_minima": hmin.hour,
        "medianoche": a_la(0), "amanecer": a_la(6),
        "cielo_1": cielo(moda(cod1)), "cielo_2": cielo(moda(cod2)),
        "prob_lluvia": max(vals("prob_lluvia"), default=None),
        "lluvia_mm": sum(vals("lluvia_mm")) if vals("lluvia_mm") else None,
        "viento_min": min(viento) if viento else None, "viento_max": max(viento) if viento else None,
        "direccion": dirs[len(dirs) // 2] if dirs else None,
        "rafaga_max": rmax, "hora_rafaga": hraf.hour if hraf else None,
        "tendencia_viento": tendencia,
        "puntos": [(h["_t"].hour, num(h.get("temperatura")), num(h.get("viento")), num(h.get("codigo_tiempo")))
                   for h in horas if h["_t"].hour in (21, 0, 3, 6)],
    }


def texto_noche(resumen: dict, pron: dict, ahora: datetime, sit=CALCULAR) -> str:
    n = resumen_noche(horas_noche(pron, ahora)) or {}
    t = (resumen or {}).get("tiempo_actual") or {}
    lineas = ["Buenas noches, Bahía Blanca. Así sigue el tiempo esta noche y en la madrugada:", ""]
    if n:
        temp = []
        if num(t.get("temperatura")) is not None and t.get("disponible"):
            temp.append(f"Ahora {fmt(t.get('temperatura'))}°")
        if n.get("medianoche") is not None:
            temp.append(f"a la medianoche {fmt(n['medianoche'])}°")
        if n.get("amanecer") is not None:
            temp.append(f"al amanecer {fmt(n['amanecer'])}°")
        minima = f"Mínima de {fmt(n['minima'])}° cerca de las {n['hora_minima']}."
        lineas.append("🌡️ " + (", ".join(temp) + ". " if temp else "") + minima)
        c1, c2 = n.get("cielo_1"), n.get("cielo_2")
        if c1 and c2 and c1 != c2:
            lineas.append(f"☁️ Cielo: {c1} a la noche y {c2} de madrugada.")
        elif c1 or c2:
            lineas.append(f"☁️ Cielo: {c1 or c2}.")
        if n.get("prob_lluvia") is not None:
            mm = n.get("lluvia_mm") or 0
            lineas.append(f"☔ Probabilidad de lluvia: hasta {fmt(n['prob_lluvia'])} %"
                          + (f" (unos {fmt(mm, 1)} mm)" if mm >= 0.5 else "") + ".")
        if n.get("viento_max") is not None:
            dirv = NOMBRE_PUNTO.get(punto(n.get("direccion")), "")
            rango = (f"de {fmt(n['viento_min'])} a {fmt(n['viento_max'])} km/h"
                     if n["viento_max"] - n["viento_min"] >= 5 else f"de unos {fmt(n['viento_max'])} km/h")
            raf = f", ráfagas de hasta {fmt(n['rafaga_max'])} km/h" if n.get("rafaga_max") else ""
            tend = f" ({n['tendencia_viento']})" if n.get("tendencia_viento") else ""
            lineas.append(f"💨 Viento{(' del ' + dirv) if dirv else ''} {rango}{raf}{tend}.")
    elif num(t.get("temperatura")) is not None and t.get("disponible"):
        lineas.append(f"🌡️ Ahora {fmt(t.get('temperatura'))}°.")
    # Con tormentas, lo primero que se lee es dónde están (breve e informativo).
    activos = _activos(resumen, sit)
    lineas[2:2] = activos
    lineas += ["", SEGUIR if activos else PIE_ENLACE, "", PIE_INFORME, HASHTAGS]
    return "\n".join(lineas)


def imagen_noche(resumen: dict, pron: dict, ahora: datetime) -> bytes:
    from PIL import Image, ImageDraw

    n = resumen_noche(horas_noche(pron, ahora)) or {}
    im = Image.new("RGB", (1080, 1080), FONDO)
    d = ImageDraw.Draw(im)
    cabecera(d, "Esta noche y la madrugada", f"Bahía Blanca · {fecha_txt(ahora)}, de 21 a 7 h")
    d.rounded_rectangle((60, 250, 1020, 640), 28, fill=TARJETA)
    d.text((100, 270), "mínima", font=fuente(36), fill=SUAVE)
    d.text((100, 305), f"{fmt(n.get('minima'))}°", font=fuente(150, True), fill=TEXTO)
    if n.get("hora_minima") is not None:
        d.text((100, 480), f"cerca de las {n['hora_minima']} h", font=fuente(36), fill=SUAVE)
    c = n.get("cielo_2") or n.get("cielo_1")
    if c:
        tam = 44
        while tam > 24 and d.textlength(c.capitalize(), font=fuente(tam, True)) > 430:
            tam -= 2
        d.text((100, 545), c.capitalize(), font=fuente(tam, True), fill=ACENTO)
    filas = [
        ("Lluvia", f"hasta {fmt(n.get('prob_lluvia'))} %"),
        ("Viento", f"{fmt(n.get('viento_max'))} km/h {punto(n.get('direccion')) or ''}"),
        ("Ráfagas", f"{fmt(n.get('rafaga_max'))} km/h"),
    ]
    for i, (k, v) in enumerate(filas):
        d.text((560, 290 + i * 110), k, font=fuente(30), fill=SUAVE)
        d.text((560, 328 + i * 110), v, font=fuente(42, True), fill=TEXTO)
    # Cuatro momentos de la noche
    d.text((60, 680), "Hora por hora", font=fuente(32, True), fill=TEXTO)
    puntos = n.get("puntos") or []
    ancho = 225
    for i, (hora, temp, viento, cod) in enumerate(puntos[:4]):
        x0 = 60 + i * (ancho + 20)
        d.rounded_rectangle((x0, 735, x0 + ancho, 960), 22, fill=TARJETA)
        d.text((x0 + ancho / 2, 775), f"{hora:02d} h", font=fuente(32, True), fill=SUAVE, anchor="mm")
        d.text((x0 + ancho / 2, 845), f"{fmt(temp)}°", font=fuente(64, True), fill=TEXTO, anchor="mm")
        d.text((x0 + ancho / 2, 920), f"{fmt(viento)} km/h", font=fuente(28), fill=ACENTO, anchor="mm")
    pie_imagen(d)
    salida = io.BytesIO()
    im.save(salida, "PNG", optimize=True)
    return salida.getvalue()


# ----------------------------------------------------------------------------
# Tormentas
# ----------------------------------------------------------------------------
# 25/11/2026 (Lucas): se publicaron advertencias de "tormentas" con viento solo
# (ráfagas de 51 km/h medidas, sin lluvia ni rayos) y después un "se alejaron
# las tormentas" de algo que nunca existió. Regla desde ahora: no se publica
# nada que no esté confirmado por mediciones reales, y cada cosa con su nombre.
#
# - TORMENTA: el radar al día ve lluvia sobre la localidad (o una que llega con
#   llegada probable) Y el satélite ve rayos (dos fuentes distintas), o la celda
#   con rayos validada que pasa sobre la localidad. Un rayo sin eco de radar
#   (el detector del satélite a veces marca destellos falsos) o un eco sin
#   rayos (puede ser un eco falso del radar) no alcanzan solos.
# - LLUVIA sin rayos: solo lluvia fuerte (nivel 3 o más) del radar al día y
#   confirmada por una estación que mide lluvia en la localidad.
# - VIENTO sin tormenta: solo medido por estaciones (no el pronóstico del
#   modelo) y desde el nivel 3 (ráfagas de 70 km/h o más).
# Lo estimado (radar viejo movido, nube fría del satélite) y lo que se mantiene
# de minutos atrás nunca sirve para confirmar.
VIENTO_SOLO_NIVEL_MINIMO = 3
LLUVIA_SOLA_NIVEL_MINIMO = 3
CLASES = ("tormenta", "lluvia", "viento")     # de más a menos importante
FUENTES_RADAR = ("radar",)                    # lluvia medida por el radar al día


def _g(f: dict | None) -> int:
    return int(num((f or {}).get("grado")) or 0)


def _fuente(f: dict | None) -> str:
    return str((f or {}).get("fuente") or "")


def _medido(f: dict | None) -> bool:
    """Detectado ahora (no mantenido de minutos atrás)."""
    return bool(f) and not f.get("mantenido") and f.get("minutos_desde_deteccion") is None


def _llega_probable(l: dict) -> bool:
    if num(l.get("minutos_hasta_llegada")) is None:
        return False
    lleg = l.get("llegada")
    return not isinstance(lleg, dict) or lleg.get("probable") is not False


def _encima(l: dict) -> dict | None:
    """Celda con rayos validada sobre la localidad (ahora o en 30 min)."""
    s = l.get("sobre_la_localidad")
    if isinstance(s, dict):
        return s
    return {"minutos": 0} if s is True else None


def evaluar_localidad(l: dict) -> dict | None:
    """Qué hay de verdad en la localidad, solo con lo confirmado. None = nada para publicar."""
    f = l.get("fenomenos") or {}
    lluvia, viento, granizo, rayos = (f.get(k) or {} for k in ("lluvia", "viento", "granizo", "electrica"))
    llega = (l.get("al_llegar") or {}) if _llega_probable(l) else {}

    eco_ahora = _medido(lluvia) and _fuente(lluvia) in FUENTES_RADAR and _g(lluvia) >= 1
    eco_llega = _fuente(llega.get("lluvia")) in FUENTES_RADAR and _g(llega.get("lluvia")) >= 1
    radar_ve_lluvia = eco_ahora or (num((l.get("acumulado_radar_mm") or {}).get("1h")) or 0) > 0
    estacion_mide_lluvia = _estacion_mide_lluvia(l)
    hay_rayos = _medido(rayos) and _g(rayos) >= 1
    encima = _encima(l)

    usados = {}
    if (hay_rayos and (eco_ahora or eco_llega or estacion_mide_lluvia)) or encima:
        clase, minimo = "tormenta", NIVEL_MINIMO
        if _g(lluvia) >= 1 and _fuente(lluvia) in ("radar", "estaciones"):
            usados["lluvia"] = lluvia
        if _g(llega.get("lluvia")) > _g(usados.get("lluvia")):
            usados["lluvia"] = llega["lluvia"]
        if hay_rayos:
            usados["electrica"] = rayos
        # El granizo y el viento de tormenta salen del mismo radar: valen
        # porque la tormenta ya está confirmada por dos fuentes.
        for ev in (granizo, llega.get("granizo")):
            if _g(ev) > _g(usados.get("granizo")) and _fuente(ev).startswith("radar"):
                usados["granizo"] = ev
        for ev in (viento, llega.get("viento")):
            if _g(ev) > _g(usados.get("viento")) and _fuente(ev) not in ("", "modelo"):
                usados["viento"] = ev
    elif (_medido(lluvia) and _fuente(lluvia) in ("radar", "estaciones")
          and _g(lluvia) >= LLUVIA_SOLA_NIVEL_MINIMO and radar_ve_lluvia and estacion_mide_lluvia):
        clase, minimo = "lluvia", LLUVIA_SOLA_NIVEL_MINIMO
        usados["lluvia"] = lluvia
    elif _medido(viento) and _fuente(viento) == "estaciones" and _g(viento) >= VIENTO_SOLO_NIVEL_MINIMO:
        clase, minimo = "viento", VIENTO_SOLO_NIVEL_MINIMO
        usados["viento"] = viento
    else:
        return None
    # Una ráfaga fuerte medida también se informa junto a la lluvia.
    if clase == "lluvia" and _medido(viento) and _fuente(viento) == "estaciones" and _g(viento) >= 2:
        usados["viento"] = viento

    nivel = max((_g(v) for v in usados.values()), default=0)
    if clase == "tormenta" and encima and nivel < NIVEL_MINIMO:
        nivel = NIVEL_MINIMO                  # tormenta eléctrica encima: moderada como mínimo
    if nivel < minimo:
        return None
    arriba = [v for v in usados.values() if _g(v) == nivel]
    return {"clase": clase, "nivel": nivel, "usados": usados,
            "posible": nivel == 4 and not any(v.get("confirmado") for v in arriba),
            "encima": encima}


def _estacion_mide_lluvia(l: dict) -> bool:
    """Una estación de la localidad (dato fresco) mide lluvia en la última hora."""
    f = (l.get("fenomenos") or {}).get("lluvia") or {}
    if _fuente(f) == "estaciones" and _g(f) >= 1 and _medido(f):
        return True
    t = l.get("tiempo_actual") or {}
    medida = num(t.get("precipitacion"))
    return "precipitacion" in (t.get("fuentes") or {}) and medida is not None and medida > 0


def _frase(clave: str, ev: dict) -> str:
    cat = str(ev.get("categoria") or NIVEL_TXT.get(_g(ev), "")).strip()
    if clave == "lluvia":
        return f"lluvia {cat}".strip()
    if clave == "viento":
        # El número que da el nivel: la ráfaga, o el viento sostenido si es
        # el que lo sube (45 km/h sostenidos ya es "fuerte").
        raf, medio = num(ev.get("rafaga_kmh")), num(ev.get("viento_kmh"))
        nums = []
        if medio is not None and medio >= 45:
            nums.append(f"{fmt(medio)} km/h sostenidos")
        if raf is not None and raf >= 30:
            nums.append(f"ráfagas de {fmt(raf)} km/h")
        return f"viento {cat}".strip() + (f" ({', '.join(nums)})" if nums else "")
    if clave == "granizo":
        return f"granizo {cat}".strip()
    return {"aislada": "rayos aislados", "moderada": "rayos seguidos", "frecuente": "rayos frecuentes",
            "severa": "actividad eléctrica severa"}.get(cat, "rayos")


def descripcion(usados: dict) -> str:
    """Solo lo confirmado, de lo más fuerte a lo más débil."""
    orden = ("lluvia", "electrica", "granizo", "viento")
    partes = [_frase(k, usados[k]) for k in sorted(usados, key=lambda k: (-_g(usados[k]), orden.index(k)))]
    if not partes:
        return ""
    texto = partes[0][0].upper() + partes[0][1:]
    if len(partes) > 1:
        texto += " con " + (", ".join(partes[1:-1]) + " y " if len(partes) > 2 else "") + partes[-1]
    return texto


def situacion(resumen: dict) -> dict | None:
    """Localidades con algo confirmado para publicar (tormenta, lluvia fuerte o viento fuerte)."""
    locs = []
    for l in (resumen or {}).get("localidades") or []:
        ev = evaluar_localidad(l)
        if not ev:
            continue
        encima = ev["encima"]
        min_encima = num((encima or {}).get("minutos"))
        locs.append({
            "id": l.get("id"),
            "nombre": l.get("nombre"),
            "clase": ev["clase"],
            "tormenta": ev["clase"] == "tormenta",
            "nivel": ev["nivel"],
            "posible": ev["posible"],
            "resumen": descripcion(ev["usados"]),
            "minutos": num(l.get("minutos_hasta_llegada")) if _llega_probable(l) else None,
            # "encima" solo si la celda con rayos ya está sobre la localidad.
            "encima": bool(encima) and (min_encima is None or min_encima <= 0),
            "pronto": bool(encima) and min_encima is not None and min_encima > 0,
        })
    if not locs:
        return None
    locs.sort(key=lambda x: (CLASES.index(x["clase"]), -x["nivel"],
                             x["minutos"] if x["minutos"] is not None else 999))
    clase = min((x["clase"] for x in locs), key=CLASES.index)
    return {
        "nivel": max(x["nivel"] for x in locs),
        "clase": clase,
        "solo_viento": clase == "viento",
        "localidades": locs,
        "firma": {x["id"]: [x["nivel"], x["minutos"] is not None or x["encima"], x["clase"]] for x in locs},
    }


# 25/11/2026: el 24/11 a las 16:43 la estación Bahía Blanca Centro marcó una
# sola vez 45 km/h (antes 18, después 29) mientras Grünbein y la Noroeste
# medían 11. Un dato suelto así no se publica: el viento solo (sin tormenta) se
# informa recién cuando se sigue midiendo al menos 15 min (dos lecturas).
VIENTO_SOSTENIDO_MIN = 15


def sostenido(sit: dict | None, estado: dict, ahora: datetime) -> dict | None:
    """Saca el viento solo que todavía no se midió durante 15 min seguidos.
    Guarda en el estado desde cuándo se viene midiendo en cada localidad."""
    vistos = estado.get("viento_desde") if isinstance(estado.get("viento_desde"), dict) else {}
    ahora_txt = ahora.isoformat()
    actuales = {l["id"] for l in (sit or {}).get("localidades", []) if l["clase"] == "viento"}
    vistos = {k: v for k, v in vistos.items() if k in actuales}
    for k in actuales:
        vistos.setdefault(k, ahora_txt)
    estado["viento_desde"] = vistos
    if not sit:
        return None

    def listo(l):
        if l["clase"] != "viento":
            return True
        desde = datetime.fromisoformat(vistos[l["id"]])
        return (ahora - desde).total_seconds() / 60.0 >= VIENTO_SOSTENIDO_MIN

    locs = [l for l in sit["localidades"] if listo(l)]
    if not locs:
        return None
    clase = min((x["clase"] for x in locs), key=CLASES.index)
    return dict(sit, localidades=locs, nivel=max(x["nivel"] for x in locs), clase=clase,
                solo_viento=clase == "viento", firma={k: v for k, v in sit["firma"].items()
                                                      if k in {x["id"] for x in locs}})


def cambio(firma_nueva: dict, firma_vieja: dict | None) -> bool:
    """Localidad nueva, nivel más alto, algo que ahora llega o que cambió de
    tipo (por ejemplo, de viento a tormenta)."""
    if not firma_vieja:
        return True
    for loc, nueva in firma_nueva.items():
        nivel, llega = nueva[0], nueva[1]
        clase = nueva[2] if len(nueva) > 2 else None
        previo = firma_vieja.get(loc)
        if previo is None or nivel > previo[0] or (llega and not previo[1]):
            return True
        if clase is not None and len(previo) > 2 and previo[2] != clase:
            return True
    return False


def _pubs_tormenta(estado: dict) -> list[dict]:
    return [p for p in estado.get("publicaciones", []) if p.get("tipo") in TIPOS_TORMENTA]


def _minutos_desde(p: dict | None, ahora: datetime) -> float | None:
    return (ahora - datetime.fromisoformat(p["utc"])).total_seconds() / 60.0 if p else None


def toca_tormenta(estado: dict, sit: dict | None, resumen: dict, ahora: datetime) -> tuple[bool, str]:
    if not sit:
        return False, "nada confirmado para publicar"
    edad = num((resumen or {}).get("frame_edad_minutos"))
    if edad is None or edad > RADAR_MAX_EDAD_MIN:
        return False, f"radar viejo ({fmt(edad)} min): no se informa con datos viejos"
    pubs = _pubs_tormenta(estado)
    ultima = pubs[-1] if pubs else None
    minutos = _minutos_desde(ultima, ahora)
    if minutos is not None and minutos < ESPERA_MIN:
        return False, f"última hace {fmt(minutos)} min (mínimo {ESPERA_MIN})"
    hoy = ahora_ar(ahora).date().isoformat()
    del_dia = [p for p in pubs if p.get("tipo") != TIPO_FIN
               and ahora_ar(datetime.fromisoformat(p["utc"])).date().isoformat() == hoy]
    if len(del_dia) >= MAX_DIA and sit["nivel"] < 4:
        return False, f"ya hubo {len(del_dia)} advertencias hoy (máximo {MAX_DIA})"
    previa = ultima.get("firma") if ultima and ultima.get("tipo") != TIPO_FIN else None
    if cambio(sit["firma"], previa):
        return True, "cambió la situación"
    repetir = REPETIR_FUERTE_MIN if sit["nivel"] >= 3 else REPETIR_MODERADO_MIN
    if minutos is None or minutos >= repetir:
        return True, "actualización: la tormenta sigue"
    return False, "sin cambios"


def toca_fin(estado: dict, sit: dict | None, resumen: dict, ahora: datetime) -> bool:
    """Una última publicación cuando ya no hay nada (con radar al día)."""
    if sit:
        return False
    edad = num((resumen or {}).get("frame_edad_minutos"))
    if edad is None or edad > RADAR_MAX_EDAD_MIN:
        return False
    pubs = _pubs_tormenta(estado)
    if not pubs or pubs[-1].get("tipo") != TIPO_ADV:
        return False
    # Sin nada durante 20 min seguidos (no por un bajón de una sola lectura).
    nada = estado.get("nada_desde")
    if not nada or (ahora - datetime.fromisoformat(nada)).total_seconds() / 60.0 < FIN_SIN_NADA_MIN:
        return False
    minutos = _minutos_desde(pubs[-1], ahora)
    return ESPERA_MIN <= minutos <= FIN_DENTRO_DE_H * 60


def clase_publicada(p: dict | None) -> str:
    """Qué anunciaba una advertencia guardada (las viejas no lo guardaban)."""
    p = p or {}
    if p.get("clase") in CLASES:
        return p["clase"]
    clases = [v[2] for v in (p.get("firma") or {}).values() if isinstance(v, list) and len(v) > 2]
    return min(clases, key=CLASES.index) if clases else "tormenta"


FIN_TITULO = {"tormenta": "Se alejaron las tormentas", "lluvia": "Paró la lluvia fuerte",
              "viento": "Bajó el viento"}
FIN_TEXTO = {"tormenta": "Por ahora no hay tormentas sobre las localidades ni acercándose.",
             "lluvia": "Por ahora no hay lluvia fuerte en las localidades.",
             "viento": "Por ahora no se mide viento fuerte en las localidades."}


def texto_fin(resumen: dict, ahora: datetime, clase: str = "tormenta") -> str:
    frame = resumen.get("frame_actual_utc")
    hora_radar = hora_txt(datetime.fromisoformat(frame.replace("Z", "+00:00"))) if frame else None
    lineas = ["✅ " + FIN_TITULO[clase], "", FIN_TEXTO[clase]]
    if hora_radar and clase != "viento":
        lineas.append(f"Radar de las {hora_radar}.")
    lineas += ["", SEGUIR, "", HASHTAGS]
    return "\n".join(lineas)


def imagen_con_tormentas(resumen: dict, ahora: datetime, sit=CALCULAR) -> bytes | None:
    """25/11/2026 (Lucas): si hay tormentas, los informes llevan la imagen del
    radar (la misma de las advertencias) en vez de la tarjeta del pronóstico."""
    if sit is CALCULAR:
        sit = situacion(resumen)
    edad = num((resumen or {}).get("frame_edad_minutos"))
    if not sit or edad is None or edad > RADAR_MAX_EDAD_MIN:
        return None
    try:
        radar = pedir_bytes("/radar/mapa")
    except Exception:
        radar = None
    return imagen_tormenta(sit, resumen, radar, ahora)


def _que(n: int, clase: str = "tormenta") -> str:
    if clase is True:                 # compatibilidad: _que(n, solo_viento)
        clase = "viento"
    elif clase is False:
        clase = "tormenta"
    if clase == "viento":
        return "Viento fuerte en la zona" if n <= 3 else "Viento posiblemente severo en la zona"
    if clase == "lluvia":
        return "Lluvia fuerte en la zona" if n <= 3 else "Lluvia posiblemente severa en la zona"
    return {2: "Tormentas moderadas en la zona", 3: "Tormenta fuerte en la zona"}.get(
        n, "Tormenta posiblemente severa en la zona")


def texto_tormenta(sit: dict, resumen: dict, ahora: datetime) -> str:
    n = max(l["nivel"] for l in sit["localidades"] if l["clase"] == sit["clase"])
    frame = resumen.get("frame_actual_utc")
    hora_radar = hora_txt(datetime.fromisoformat(frame.replace("Z", "+00:00"))) if frame else None
    titulo = f"⚠️ ADVERTENCIA · {_que(n, sit['clase'])}"
    lineas = [titulo, ""]
    for l in sit["localidades"][:5]:
        nivel = f"nivel {l['nivel']} ({'posible ' if l['posible'] and l['nivel'] == 4 else ''}{NIVEL_TXT[l['nivel']]})"
        linea = f"📍 {l['nombre']}: {nivel}."
        if l["resumen"]:
            linea += f" {l['resumen']}."
        if l["encima"]:
            linea += " La tormenta está encima."
        elif l["minutos"] is not None:
            llega = ahora + timedelta(minutes=l["minutos"])
            linea += f" Llega en ~{fmt(l['minutos'])} min (cerca de las {hora_txt(llega)})."
        elif l.get("pronto"):
            linea += " Puede pasar sobre la localidad en los próximos 30 min."
        lineas.append(linea)
    lineas += [""]
    if hora_radar and sit["clase"] != "viento":
        lineas.append(f"Radar de las {hora_radar}.")
    etiqueta = {"tormenta": " #Tormenta", "lluvia": " #Lluvia", "viento": " #Viento"}[sit["clase"]]
    lineas += ["", SEGUIR, "", HASHTAGS + etiqueta]
    return "\n".join(lineas)


# ----------------------------------------------------------------------------
# Imágenes (1080 x 1080)
# ----------------------------------------------------------------------------
FONDO = (15, 27, 45)
TARJETA = (22, 38, 61)
TEXTO = (238, 243, 251)
SUAVE = (159, 177, 204)
ACENTO = (95, 208, 224)
NIVEL_COLOR = {0: (51, 70, 95), 1: (159, 216, 240), 2: (111, 164, 242), 3: (106, 99, 230), 4: (176, 59, 212)}


def fuente(tam: int, negrita: bool = False):
    from PIL import ImageFont

    nombres = (["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"] if negrita
               else ["DejaVuSans.ttf", "LiberationSans-Regular.ttf"])
    for carpeta in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
                    "/usr/share/fonts/dejavu", "/usr/share/fonts/TTF"):
        for n in nombres:
            p = Path(carpeta) / n
            if p.exists():
                return ImageFont.truetype(str(p), tam)
    return ImageFont.load_default(size=tam)


def cabecera(d, titulo: str, sub: str):
    d.text((60, 50), "Clima", font=fuente(54, True), fill=TEXTO)
    w = d.textlength("Clima", font=fuente(54, True))
    d.text((60 + w, 50), "AR", font=fuente(54, True), fill=ACENTO)
    d.text((60, 130), titulo, font=fuente(40, True), fill=TEXTO)
    d.text((60, 182), sub, font=fuente(28), fill=SUAVE)


def pie_imagen(d, alto=1080):
    d.text((60, alto - 70), URL_PUBLICA.replace("https://", ""), font=fuente(26, True), fill=ACENTO)


def imagen_manana(resumen: dict, pron: dict, ahora: datetime) -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (1080, 1080), FONDO)
    d = ImageDraw.Draw(im)
    cabecera(d, "Cómo viene el día", f"Bahía Blanca · {fecha_txt(ahora)}")
    hoy = (pron or {}).get("hoy") or {}
    d.rounded_rectangle((60, 250, 1020, 640), 28, fill=TARJETA)
    d.text((100, 280), f"{fmt(hoy.get('maxima'))}°", font=fuente(150, True), fill=TEXTO)
    d.text((100, 460), f"mínima {fmt(hoy.get('minima'))}°", font=fuente(40), fill=SUAVE)
    c = cielo(hoy.get("codigo_tiempo"))
    # El cielo no puede pisar la columna de la derecha (lluvia, viento, ráfagas):
    # se achica la letra hasta que entre en 430 px.
    if c:
        tam = 44
        while tam > 24 and d.textlength(c.capitalize(), font=fuente(tam, True)) > 430:
            tam -= 2
        d.text((100, 530), c.capitalize(), font=fuente(tam, True), fill=ACENTO)
    x = 560
    filas = [
        ("Lluvia", f"{fmt(hoy.get('prob_lluvia'))} %" + (f" · {fmt(hoy.get('lluvia_mm'), 1)} mm"
                                                          if (num(hoy.get('lluvia_mm')) or 0) >= 0.5 else "")),
        ("Viento", f"{fmt(hoy.get('viento'))} km/h {punto(hoy.get('viento_direccion')) or ''}"),
        ("Ráfagas", f"{fmt(hoy.get('rafagas'))} km/h"),
    ]
    for i, (k, v) in enumerate(filas):
        d.text((x, 290 + i * 110), k, font=fuente(30), fill=SUAVE)
        d.text((x, 328 + i * 110), v, font=fuente(42, True), fill=TEXTO)
    # Localidades con su nivel
    d.text((60, 680), "Las 7 localidades ahora", font=fuente(32, True), fill=TEXTO)
    locs = (resumen or {}).get("localidades") or []
    for i, l in enumerate(locs[:8]):
        col, fil = i % 2, i // 2
        x0, y0 = 60 + col * 485, 735 + fil * 68
        n = int(num(l.get("nivel")) or 0)
        d.rounded_rectangle((x0, y0, x0 + 46, y0 + 46), 10, fill=NIVEL_COLOR.get(n, NIVEL_COLOR[0]))
        d.text((x0 + 23, y0 + 23), str(n or "–"), font=fuente(28, True),
               fill=(11, 36, 51) if n in (1, 2) else TEXTO, anchor="mm")
        d.text((x0 + 62, y0 + 6), str(l.get("nombre") or ""), font=fuente(30), fill=TEXTO)
    pie_imagen(d)
    salida = io.BytesIO()
    im.save(salida, "PNG", optimize=True)
    return salida.getvalue()


VISTA_KM = 110.0      # medio ancho del mapa de la publicación (zona de las localidades)


def _vista(sit: dict, resumen: dict):
    """Cuadro de ~220 km centrado en las localidades con actividad (o Bahía)."""
    pos = {l.get("id"): (num(l.get("lat")), num(l.get("lon"))) for l in resumen.get("localidades") or []}
    pts = [pos[l["id"]] for l in sit["localidades"] if pos.get(l["id"]) and None not in pos[l["id"]]]
    lat = sum(p[0] for p in pts) / len(pts) if pts else -38.72
    lon = sum(p[1] for p in pts) / len(pts) if pts else -62.27
    dlat = VISTA_KM / 110.574
    dlon = VISTA_KM / (111.32 * math.cos(math.radians(lat)))
    return [[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]]


def _proyector(bounds, lado):
    (la0, lo0), (la1, lo1) = bounds

    def p(lat, lon):
        return ((lon - lo0) / (lo1 - lo0) * lado, (la1 - lat) / (la1 - la0) * lado)
    return p


def imagen_tormenta(sit: dict, resumen: dict, radar_png: bytes | None, ahora: datetime,
                    fin: bool = False) -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (1080, 1350), FONDO)   # vertical 4:5, la que mejor se ve en Facebook
    d = ImageDraw.Draw(im)
    clase = sit.get("clase") or ("viento" if sit.get("solo_viento") else "tormenta")
    n = max([l["nivel"] for l in sit["localidades"] if l.get("clase", clase) == clase] or [sit["nivel"]])
    titulo = FIN_TITULO[clase].upper() if fin else "ADVERTENCIA"
    que = ("Nada fuerte en las localidades" if fin
           else _que(n, clase).replace(" en la zona", ""))
    frame = resumen.get("frame_actual_utc")
    hora = hora_txt(datetime.fromisoformat(frame.replace("Z", "+00:00"))) if frame else None
    local = ahora_ar(ahora)
    sub = que + (f" · radar de las {hora}" if hora and clase != "viento" else "") + f" ({local.day}/{local.month})"
    cabecera(d, titulo, sub)

    # El mapa se dibuja aparte (así nada se sale del cuadro) y se pega.
    lado, x0, y0 = 880, 100, 250   # deja lugar para 2 localidades y la dirección
    vista = _vista(sit, resumen)
    p = _proyector(vista, lado)
    mapa = Image.new("RGB", (lado, lado), (11, 21, 36))
    dm = ImageDraw.Draw(mapa)
    try:
        geo = json.loads((RAIZ / "frontend" / "cartografia.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        geo = {"features": []}

    def lineas(geom):
        t, c = geom.get("type"), geom.get("coordinates") or []
        if t == "LineString":
            yield c
        elif t in ("MultiLineString", "Polygon"):
            yield from c
        elif t == "MultiPolygon":
            for pol in c:
                yield from pol

    estilos = {"provincias": ((150, 165, 190), 2), "partidos": ((80, 100, 130), 1),
               "rutas": ((150, 125, 80), 2), "rios": ((63, 127, 191), 2)}
    for f in geo.get("features", []):
        if f["properties"].get("capa") == "tierra" and f["geometry"]["type"] in ("Polygon", "MultiPolygon"):
            for anillo in lineas(f["geometry"]):
                if len(anillo) > 2:
                    dm.polygon([p(la, lo) for lo, la in anillo], fill=(22, 35, 58), outline=(52, 80, 122))
    for f in geo.get("features", []):
        capa = f["properties"].get("capa")
        if capa in estilos:
            color, ancho = estilos[capa]
            for linea in lineas(f["geometry"]):
                if len(linea) > 1:
                    dm.line([p(la, lo) for lo, la in linea], fill=color, width=ancho)

    # Radar: se recorta la parte de la vista y se agranda al cuadro.
    bounds = ((resumen.get("radar") or {}).get("bounds")) or [[-40.8926, -64.9315], [-36.5758, -59.3993]]
    if radar_png:
        try:
            rad = Image.open(io.BytesIO(radar_png)).convert("RGBA")
            (b0, c0), (b1, c1) = bounds
            (v0, w0), (v1, w1) = vista
            W, H = rad.size
            caja = ((w0 - c0) / (c1 - c0) * W, (b1 - v1) / (b1 - b0) * H,
                    (w1 - c0) / (c1 - c0) * W, (b1 - v0) / (b1 - b0) * H)
            rad = rad.transform((lado, lado), Image.Transform.EXTENT, caja, Image.Resampling.BILINEAR)
            mapa.paste(rad, (0, 0), rad)
        except Exception:
            pass
    # Anillo del alcance del radar (240 km) y localidades
    cx, cy = p(-38.7342, -62.1654)
    radio = 240.0 / (2 * VISTA_KM) * lado
    dm.ellipse((cx - radio, cy - radio, cx + radio, cy + radio), outline=(95, 208, 224), width=2)
    activos = {l["id"] for l in sit["localidades"]}
    for l in resumen.get("localidades") or []:
        if num(l.get("lat")) is None:
            continue
        x, y = p(l["lat"], l["lon"])
        act = l.get("id") in activos
        dm.ellipse((x - 7, y - 7, x + 7, y + 7), fill=ACENTO if act else TEXTO, outline=(11, 21, 36), width=2)
        if act or l.get("id") not in ("ingeniero_white", "general_cerri"):
            dm.text((x + 12, y - 18), l["nombre"], font=fuente(28 if act else 24, True),
                    fill=ACENTO if act else TEXTO, stroke_width=4, stroke_fill=(11, 21, 36))
    dm.text((lado - 16, lado - 14), "Radar RMA10 · SINARAME", font=fuente(20), fill=SUAVE, anchor="rd")
    im.paste(mapa, (x0, y0))
    d.rectangle((x0, y0, x0 + lado, y0 + lado), outline=(36, 56, 90), width=2)

    # Debajo del mapa: hasta 2 localidades con su nivel y llegada
    y = y0 + lado + 24
    for l in sit["localidades"][:2]:
        col = NIVEL_COLOR[l["nivel"]]
        d.rounded_rectangle((60, y, 112, y + 52), 12, fill=col)
        d.text((86, y + 26), str(l["nivel"]), font=fuente(32, True),
               fill=(11, 36, 51) if l["nivel"] in (1, 2) else TEXTO, anchor="mm")
        txt = l["nombre"]
        if l["encima"]:
            txt += " · tormenta encima"
        elif l["minutos"] is not None:
            txt += f" · llega en ~{fmt(l['minutos'])} min"
        d.text((130, y + 8), txt, font=fuente(32, True), fill=TEXTO)
        y += 64
    pie_imagen(d, 1350)
    salida = io.BytesIO()
    im.save(salida, "PNG", optimize=True)
    return salida.getvalue()


# ----------------------------------------------------------------------------
# Facebook
# ----------------------------------------------------------------------------
PUBLICO = Path(os.environ.get("CLIMAAR_FB_PUBLICO", "/var/lib/climaar-facebook/publico"))
DIAS_IMAGENES = 3


def guardar_imagen_publica(png: bytes, tipo: str, ahora: datetime) -> str:
    """Deja la imagen donde la sirve la página (/publicaciones/...) y devuelve su dirección.

    Make la descarga de ahí para publicarla. Se borran las de más de 3 días."""
    PUBLICO.mkdir(parents=True, exist_ok=True)
    nombre = f"{ahora.astimezone(TZ_AR):%Y%m%d-%H%M%S}-{tipo}.png"
    tmp = PUBLICO / f".{nombre}.tmp"
    tmp.write_bytes(png)
    os.replace(tmp, PUBLICO / nombre)
    limite = time_now() - DIAS_IMAGENES * 86400
    for viejo in PUBLICO.glob("*.png"):
        try:
            if viejo.stat().st_mtime < limite:
                viejo.unlink()
        except OSError:
            pass
    return f"{URL_PUBLICA}/publicaciones/{nombre}"


def time_now() -> float:
    import time

    return time.time()


def publicar(texto: str, png: bytes, tipo: str, ahora: datetime) -> str:
    """Publica en la página de Facebook a través de Make (24/11/2026).

    Make ya tiene permiso de Meta para publicar en páginas: ClimaAR le manda
    al "webhook" (dirección privada que da Make) los tres campos que usa el
    escenario: mensaje, imagen_url y tipo. Make baja la imagen de imagen_url."""
    import requests

    webhook = os.environ.get("MAKE_WEBHOOK_URL", "").strip()
    if not webhook.startswith("https://"):
        raise RuntimeError("falta MAKE_WEBHOOK_URL en /etc/climaar-facebook.env")
    imagen = guardar_imagen_publica(png, tipo, ahora)
    # Antes de avisarle a Make se controla que la imagen se vea desde afuera
    # (si no, Make fallaría y la publicación se perdería sin aviso).
    try:
        v = requests.get(imagen, timeout=20)
        ok = v.status_code == 200 and v.content[:8] == b"\x89PNG\r\n\x1a\n"
    except requests.RequestException as exc:
        raise RuntimeError(f"la imagen no se ve desde afuera ({imagen}): {type(exc).__name__}") from None
    if not ok:
        raise RuntimeError(f"la imagen no se ve desde afuera ({imagen}): respuesta {v.status_code}")
    # Mismo formato que la prueba que Make publicó bien (formulario con los
    # campos mensaje, imagen_url y tipo, más la imagen como archivo "imagen").
    r = requests.post(
        webhook,
        data={"mensaje": texto, "imagen_url": imagen, "tipo": tipo},
        files={"imagen": (imagen.rsplit("/", 1)[-1], png, "image/png")},
        timeout=60,
    )
    if r.status_code >= 300:
        # Nunca se escribe la dirección del webhook en el registro.
        raise RuntimeError(f"Make respondió {r.status_code}: {r.text[:200]}")
    return imagen


# ----------------------------------------------------------------------------
# Ciclo
# ----------------------------------------------------------------------------
def ciclo(probar: bool = False, salida: Path | None = None, forzar: str | None = None,
          ahora: datetime | None = None) -> list[str]:
    ahora = ahora or datetime.now(timezone.utc)
    activo = os.environ.get("FB_ACTIVO", "1").strip() != "0"
    estado = leer_estado()
    hechos = []
    resumen = pedir_json("/resumen")
    # Lo confirmado, una sola vez por ciclo: lo usan los informes y las advertencias.
    sit = sostenido(situacion(resumen), estado, ahora)
    if sit:
        estado.pop("nada_desde", None)
    else:
        estado.setdefault("nada_desde", ahora.isoformat())

    def emitir(tipo: str, texto: str, png: bytes, extra: dict) -> None:
        texto = sin_alerta(texto)
        if probar or not activo:
            if salida:
                salida.mkdir(parents=True, exist_ok=True)
                (salida / f"{tipo}.txt").write_text(texto, encoding="utf-8")
                (salida / f"{tipo}.png").write_bytes(png)
            hechos.append(f"{tipo}: armado sin publicar")
            if probar:
                return
            post = None
        else:
            post = publicar(texto, png, tipo, ahora)
            hechos.append(f"{tipo}: publicado {post}")
        estado.setdefault("publicaciones", []).append(
            dict(extra, tipo=tipo, utc=ahora.isoformat(), id=post))

    # 1) Mañana
    if forzar == "manana" or (forzar is None and toca_manana(estado, ahora)):
        pron = pedir_json("/pronostico")
        if (pron or {}).get("hoy") or forzar:
            emitir("manana", texto_manana(resumen, pron, ahora, sit),
                   imagen_con_tormentas(resumen, ahora, sit) or imagen_manana(resumen, pron, ahora), {})
            if not probar:
                estado["manana_fecha"] = ahora_ar(ahora).date().isoformat()
        else:
            hechos.append("mañana: sin pronóstico todavía, se reintenta")

    # 1b) Noche: cómo viene mañana
    if forzar == "noche" or (forzar is None and toca_noche(estado, ahora)):
        pron = pedir_json("/pronostico")
        if resumen_noche(horas_noche(pron, ahora)) or forzar:
            emitir("noche", texto_noche(resumen, pron, ahora, sit),
                   imagen_con_tormentas(resumen, ahora, sit) or imagen_noche(resumen, pron, ahora), {})
            if not probar:
                estado["noche_fecha"] = ahora_ar(ahora).date().isoformat()
        else:
            hechos.append("noche: sin pronóstico todavía, se reintenta")

    # 2) Tormentas, lluvia fuerte o viento fuerte (lo mismo que dicen los informes)
    forzar_adv = forzar in (TIPO_ADV, "tormenta")
    toca, motivo = (True, "forzada") if (forzar_adv and sit) else toca_tormenta(estado, sit, resumen, ahora)
    if toca:
        try:
            radar = pedir_bytes("/radar/mapa")
        except Exception:
            radar = None
        emitir(TIPO_ADV, texto_tormenta(sit, resumen, ahora), imagen_tormenta(sit, resumen, radar, ahora),
               {"firma": sit["firma"], "nivel": sit["nivel"], "clase": sit["clase"], "motivo": motivo})
    elif forzar_adv:
        hechos.append(f"{TIPO_ADV}: no se puede forzar, no hay nada confirmado para publicar")
    elif forzar is None and toca_fin(estado, sit, resumen, ahora):
        try:
            radar = pedir_bytes("/radar/mapa")
        except Exception:
            radar = None
        clase = clase_publicada(_pubs_tormenta(estado)[-1])
        vacia = {"nivel": 0, "clase": clase, "localidades": [], "firma": {}}
        emitir(TIPO_FIN, texto_fin(resumen, ahora, clase),
               imagen_tormenta(vacia, resumen, radar, ahora, fin=True),
               {"firma": {}, "nivel": 0, "clase": clase, "motivo": "terminó"})
    else:
        hechos.append(f"{TIPO_ADV}: no ({motivo})")

    if not probar:
        estado["ultimo_ciclo_utc"] = ahora.isoformat()
        estado.pop("ultimo_error", None)
        guardar_estado(estado, ahora)
    return hechos


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Publicaciones de ClimaAR en Facebook")
    ap.add_argument("--probar", action="store_true", help="arma texto e imagen sin publicar ni guardar estado")
    ap.add_argument("--salida", type=Path, help="carpeta donde dejar lo armado (con --probar)")
    ap.add_argument("--forzar", choices=["manana", "noche", TIPO_ADV, "tormenta"], help="publicar ya, sin esperar la hora")
    a = ap.parse_args(argv)
    try:
        for h in ciclo(probar=a.probar, salida=a.salida, forzar=a.forzar):
            print("facebook:", h)
    except Exception as exc:
        print(f"facebook: ERROR {type(exc).__name__}: {exc}")
        if not a.probar:
            e = leer_estado()
            e["ultimo_error"] = {"utc": datetime.now(timezone.utc).isoformat(), "error": str(exc)[:300]}
            try:
                guardar_estado(e)
            except OSError:
                pass
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
