#!/usr/bin/env python3
import json
import time
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
import html
import requests

BASE_URL = "https://info.bcp.org.ar/Estaciones/Mapa_NvoBCP.asp?Estacion={}"
MAX_AGE = 15


def is_fresh_age(age):
    return age is not None and 0 <= age <= MAX_AGE
BASE = Path(__file__).resolve().parents[2]
OUTPUT = BASE / "data/ia/estaciones/observaciones/bcp_actual.json"

STATIONS = {
    1: "GRUMBEIN",
    # 2: "CABILDO" y 17: "SAN ELOY" desactivadas el 23/11/2026: llegan con horas de atraso.
    81: "PUAN",
    # 11: "PIGUE" desactivada el 23/11/2026: llega 20-70 min tarde en casi la mitad de los ciclos.
    15: "CURA MALAL",
    60: "ARROYO PANTANOSO",
    # 83: "PASMAN" desactivada el 23/11/2026: sin datos desde fines de octubre.
    104: "BAHIA BLANCA CENTRO",
    # Estaciones nuevas en puntos estratégicos (búsqueda del 20/11/2026,
    # docs/ESTACIONES_NUEVAS.md): todas publican cada pocos minutos.
    3: "LA VITICOLA",        # 28 km al norte de Bahía, camino a Tornquist
    76: "PUNTA ALTA",        # dentro de Punta Alta (la localidad no tenía estación)
    59: "VILLA VENTANA",     # entre Tornquist y Sierra de la Ventana
    97: "SIERRAS GRANDES",   # sobre las sierras, junto a Sierra de la Ventana
    95: "MACACHIN",          # 205 km al noroeste (La Pampa): de ahí vienen muchas tormentas
}

def clean(text):
    return html.unescape(text).replace("\xa0", " ").strip()

def parse_dt(value):
    try:
        return datetime.strptime(
            value.strip(), "%d/%m/%Y %H:%M:%S"
        ).replace(tzinfo=timezone(timedelta(hours=-3))).astimezone(timezone.utc)
    except Exception:
        return None

def find_value(text, label):
    pattern = (
        rf"{label}.*?</td>\s*"
        rf"<td[^>]*>(.*?)</td>"
    )
    m = re.search(pattern, text, re.I | re.S)
    if not m or m.group(1) is None:
        return None

    value = re.sub(r"<[^>]+>", " ", m.group(1))
    value = clean(value)

    if not value or value.upper() == "N/D":
        return None

    return value

def find_timestamp(text, label):
    pattern = (
        rf"{label}.*?</td>\s*"
        rf"<td[^>]*[^>]*>.*?"
        rf"(\d{{2}}/\d{{2}}/\d{{4}}\s+\d{{1,2}}:\d{{2}}:\d{{2}})"
    )
    m = re.search(pattern, text, re.I | re.S)
    return m.group(1) if m else None

def find_latest(text):
    m = re.search(
        r"(?:&Uacute;ltimo|Último)\s+D(?:ato|ato)\s+Recibido.*?"
        r"(\d{2}/\d{2}/\d{4}\s+\d{1,2}:\d{2}:\d{2})",
        text,
        re.I | re.S,
    )
    return m.group(1) if m else None

def number(value):
    if value is None:
        return None
    m = re.search(r"-?\d+(?:[.,]\d+)?", value)
    if not m:
        return None
    return float(m.group(0).replace(",", "."))

def collect(sid, name):
    url = BASE_URL.format(sid)
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    text = r.text

    latest_raw = find_latest(text)

    if not latest_raw:
        return {
            "ok": False,
            "station_id": sid,
            "nombre": name,
            "estado": "sin_dato",
        }

    obs_time = parse_dt(latest_raw)
    if obs_time is None:
        return {
            "ok": False,
            "station_id": sid,
            "nombre": name,
            "estado": "timestamp_invalido",
        }

    now = datetime.now(timezone.utc)
    age = (now - obs_time).total_seconds() / 60

    # "Máxima Ráfaga últ. 15 min" es de los 15 min anteriores al último dato
    # recibido: esa fila no tiene hora propia. 24/11/2026: antes se exigía una
    # hora y la búsqueda encontraba otra fecha de la página (o ninguna), así
    # que la ráfaga de TODAS las estaciones BCP se descartaba siempre (Bahía
    # Centro marcaba 33 km/h y ClimaAR no tenía ninguna ráfaga medida).
    rafaga_val = number(find_value(text, r"M[aá]xima R[aá]faga.*?15 min|M&aacute;xima R&aacute;faga.*?15 min"))
    if rafaga_val is not None and not 0 <= rafaga_val <= 250:
        rafaga_val = None
    variables = {
        "temperatura_c": number(find_value(text, r"Última Temperatura|&Uacute;ltima Temperatura")),
        "precipitacion_dia_mm": number(find_value(text, r"Precipitaciones del d")),
        "viento_kmh": number(find_value(text, r"Velocidad del Viento")),
        "direccion_viento": find_value(text, r"Direcci[oó]n del Viento|Direcci&oacute;n del Viento"),
        "humedad_relativa_pct": number(find_value(text, r"Humedad Relativa|Humedad Relativa")),
        "presion_hpa": number(find_value(text, r"Presi[oó]n|Presi&oacute;n")),
        "radiacion_wm2": number(find_value(text, r"Radiaci[oó]n|Radiaci&oacute;n")),
        "temp_suelo_5cm_c": number(find_value(text, r"Temperatura a 5 cm del Suelo")),
        "rafaga_15min_kmh": rafaga_val,
    }
    return {
        "ok": True,
        "station_id": sid,
        "nombre": name,
        "observation_time": obs_time.isoformat(),
        "edad_minutos": round(age, 2),
        "fresh": is_fresh_age(age),
        "estado": "valida" if is_fresh_age(age) else (
            "timestamp_futuro" if age < 0 else "descartada_por_antiguedad"
        ),
        "variables": variables,
        "timestamps": {
            # Sin hora propia: los 15 min anteriores al último dato.
            "rafaga_15min": obs_time.isoformat() if rafaga_val is not None else None,
        },
    }

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Re-auditoría 23/11/2026: de a una tardaba ~26 s por ciclo y un solo corte
    # de red dejaba esa estación afuera. Ahora todas a la vez y con un reintento.
    from concurrent.futures import ThreadPoolExecutor

    def una(item):
        sid, name = item
        ultimo = None
        for intento in range(2):
            try:
                return collect(sid, name)
            except Exception as e:  # noqa: PERF203
                ultimo = e
                time.sleep(2)
        return {
            "ok": False,
            "station_id": sid,
            "nombre": name,
            "estado": "error",
            "error": str(ultimo),
        }

    with ThreadPoolExecutor(max_workers=6) as pool:
        stations = list(pool.map(una, STATIONS.items()))  # mismo orden que STATIONS

    result = {
        "ok": True,
        "collector": "bcp",
        "version": "1.0.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "max_age_minutes": MAX_AGE,
        "stations_total": len(stations),
        "stations_validas": sum(
            1 for x in stations if x.get("fresh") is True
        ),
        "stations": stations,
    }

    tmp = OUTPUT.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(OUTPUT)

    print("=== COLLECTOR BCP ===")
    print(f"Total:   {len(stations)}")
    print(f"Validas: {result['stations_validas']}")

    for x in stations:
        age = x.get("edad_minutos")
        print(
            f"{x['nombre']}: {x['estado']}"
            + (f" | {age} min" if age is not None else "")
        )

if __name__ == "__main__":
    main()
