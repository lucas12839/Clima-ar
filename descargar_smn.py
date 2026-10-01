import argparse
import csv
import time
from datetime import datetime, timedelta
import requests

URL = "https://ssl.smn.gob.ar/dpd/descarga_opendata.php?file=observaciones/datohorario{}.txt"

ESTACIONES = {
    "BAHIA BLANCA AERO", "PIGUE AERO", "CORONEL SUAREZ AERO",
    "TRES ARROYOS", "RIO COLORADO", "VIEDMA AERO",
    "SAN ANTONIO OESTE AERO"
}

CAMPOS = ["fecha","hora","temperatura","humedad","presion",
          "direccion_viento","velocidad_viento","estacion"]

def obtener(fecha):
    try:
        u = URL.format(fecha.strftime("%Y%m%d"))
        r = requests.get(u, timeout=60)
        r.raise_for_status()
        texto = r.content.decode("latin-1", errors="replace")
        if not texto.strip() or "El archivo no existe." in texto:
            return []
        datos = []
        for linea in texto.splitlines():
            p = linea.strip().split()
            if len(p) < 8:
                continue
            estacion = " ".join(p[7:])
            if estacion in ESTACIONES:
                datos.append(dict(zip(CAMPOS, p[:7] + [estacion])))
        return datos
    except Exception as e:
        print("[ERROR]", fecha, e)
        return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", required=True)
    ap.add_argument("--hasta", required=True)
    ap.add_argument("--salida", default="climaar_smn.csv")
    ap.add_argument("--pausa", type=float, default=1)
    a = ap.parse_args()

    desde = datetime.strptime(a.desde, "%Y-%m-%d").date()
    hasta = datetime.strptime(a.hasta, "%Y-%m-%d").date()

    with open(a.salida, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()

        fecha = desde
        total = 0

        while fecha <= hasta:
            datos = obtener(fecha)
            for fila in datos:
                w.writerow(fila)
            total += len(datos)
            print(f"[OK] {fecha} | {len(datos)} registros")
            fecha += timedelta(days=1)
            if fecha <= hasta:
                time.sleep(a.pausa)

    print("================================")
    print("REGISTROS TOTALES:", total)
    print("ARCHIVO:", a.salida)
    print("================================")

if __name__ == "__main__":
    main()
