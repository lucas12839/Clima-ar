"""Recolector de radares de otras zonas (calibración, 23/11/2026).

Guarda las imágenes crudas de radares SINARAME con tormentas frecuentes
(por defecto Córdoba RMA1, Resistencia RMA4 y Tostado RMA21) para calibrar el seguimiento,
las trayectorias, las falsas alarmas y las señales de tormenta severa con
muchas más tormentas reales que las que llegan a Bahía Blanca.

- No toca nada del ciclo de Bahía: no lee ni escribe archivos del proyecto,
  no usa el navegador y solo usa la biblioteca estándar de Python.
- Lo corre cada 5 min el reloj climaar-recoleccion.timer
  (instalar con deploy/instalar_recoleccion.sh).
- Guarda las imágenes tal cual llegan (crudas). La máscara de ecos fijos y
  cartografía de cada radar se arma después con lo recolectado.
- Topes de seguridad: deja de guardar si el disco queda con menos de
  --libre-min-gb libres o si la recolección llega a --tope-gb.
- Único programa que escribe en la carpeta de destino (incluido estado.json).
"""

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

URL_BASE = "https://radares.hidricosargentina.gob.ar/cache/{radar}/{nombre}.png"
RADARES = {"RMA1": "Córdoba", "RMA4": "Resistencia", "RMA21": "Tostado"}
# Argentina no tiene horario de verano: UTC-3 fijo (SINARAME nombra en hora local).
ARGENTINA = timezone(timedelta(hours=-3))
PASO_MIN = 10
VENTANA_H = 3            # SINARAME a veces publica con 30 min o más de atraso
LIBRE_MIN_GB = 3.0
TOPE_GB = 3.0
MAX_BYTES_IMAGEN = 3 * 1024 * 1024
PNG_FIRMA = b"\x89PNG\r\n\x1a\n"
GB = 1024 ** 3


def png_valido(datos):
    """Imagen PNG completa (firma al principio y bloque IEND al final)."""
    return (
        len(datos) > 1000
        and datos.startswith(PNG_FIRMA)
        and datos[-12:-4] == b"\x00\x00\x00\x00IEND"
    )


def turnos(ahora_utc, ventana_h=VENTANA_H):
    """Horarios de publicación (cada 10 min, hora argentina) de la ventana, del más nuevo al más viejo."""
    local = ahora_utc.astimezone(ARGENTINA)
    t = local.replace(minute=local.minute - local.minute % PASO_MIN, second=0, microsecond=0)
    fin = local - timedelta(hours=ventana_h)
    while t >= fin:
        yield t
        t -= timedelta(minutes=PASO_MIN)


def ruta_imagen(destino, radar, t_local):
    return Path(destino) / radar / t_local.strftime("%Y%m%d") / (t_local.strftime("%Y%m%d%H%M%S") + ".png")


def bajar(url, timeout=20):
    """Devuelve los bytes de la imagen, None si todavía no existe (404) o lanza error."""
    pedido = urllib.request.Request(url, headers={"User-Agent": "ClimaAR-recoleccion/1.0"})
    try:
        with urllib.request.urlopen(pedido, timeout=timeout) as resp:
            return resp.read(MAX_BYTES_IMAGEN + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def tamano_carpeta(carpeta):
    total = 0
    for raiz, _, archivos in os.walk(carpeta):
        for a in archivos:
            try:
                total += os.path.getsize(os.path.join(raiz, a))
            except OSError:
                pass
    return total


def motivo_freno(destino, usados, libre_min_gb, tope_gb, libre=None):
    libre = shutil.disk_usage(destino).free if libre is None else libre
    if libre < libre_min_gb * GB:
        return f"disco con menos de {libre_min_gb:g} GB libres"
    if usados >= tope_gb * GB:
        return f"la recolección llegó al tope de {tope_gb:g} GB"
    return None


def leer_estado(destino):
    try:
        return json.loads((Path(destino) / "estado.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def guardar_estado(destino, estado):
    ruta = Path(destino) / "estado.json"
    tmp = ruta.with_suffix(".tmp")
    tmp.write_text(json.dumps(estado, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, ruta)


def recolectar(destino, radares=RADARES, ahora_utc=None, bajar_fn=bajar,
               libre_min_gb=LIBRE_MIN_GB, tope_gb=TOPE_GB, libre_fn=None, pausa=0.2):
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    ahora_utc = ahora_utc or datetime.now(timezone.utc)
    estado = leer_estado(destino)
    por_radar = estado.get("radares", {})
    usados = tamano_carpeta(destino)
    freno = None

    for radar, nombre in radares.items():
        info = por_radar.get(radar, {"nombre": nombre, "imagenes": 0, "ultima": None})
        info["nombre"] = nombre
        nuevas, errores = 0, []
        for t in turnos(ahora_utc):
            ruta = ruta_imagen(destino, radar, t)
            if ruta.exists():
                continue
            freno = motivo_freno(destino, usados, libre_min_gb, tope_gb,
                                 None if libre_fn is None else libre_fn())
            if freno:
                break
            url = URL_BASE.format(radar=radar, nombre=t.strftime("%Y%m%d%H%M%S"))
            try:
                datos = bajar_fn(url)
            except Exception as exc:  # red caída o SINARAME con error: se reintenta la próxima vuelta
                errores.append(f"{t:%H:%M} {type(exc).__name__}: {str(exc)[:80]}")
                continue
            finally:
                if pausa:
                    time.sleep(pausa)
            if datos is None:
                continue
            if len(datos) > MAX_BYTES_IMAGEN or not png_valido(datos):
                errores.append(f"{t:%H:%M} imagen inválida ({len(datos)} bytes)")
                continue
            ruta.parent.mkdir(parents=True, exist_ok=True)
            tmp = ruta.with_suffix(".tmp")
            tmp.write_bytes(datos)
            os.replace(tmp, ruta)
            usados += len(datos)
            nuevas += 1
            info["imagenes"] = info.get("imagenes", 0) + 1
            hora = t.strftime("%Y-%m-%d %H:%M")
            if not info.get("ultima") or hora > info["ultima"]:
                info["ultima"] = hora
        info["nuevas_ultima_vuelta"] = nuevas
        info["errores_ultima_vuelta"] = errores[:5]
        por_radar[radar] = info
        if freno:
            break

    estado = {
        "actualizado_hora_argentina": ahora_utc.astimezone(ARGENTINA).strftime("%Y-%m-%d %H:%M"),
        "ocupado_mb": round(usados / 1024 ** 2, 1),
        "tope_gb": tope_gb,
        "libre_min_gb": libre_min_gb,
        "detenido": freno,
        "radares": por_radar,
    }
    guardar_estado(destino, estado)
    return estado


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--destino", default="/var/lib/climaar-recoleccion")
    p.add_argument("--radares", default=",".join(RADARES),
                   help="códigos separados por coma (por defecto RMA1,RMA4,RMA21)")
    p.add_argument("--libre-min-gb", type=float, default=LIBRE_MIN_GB)
    p.add_argument("--tope-gb", type=float, default=TOPE_GB)
    a = p.parse_args(argv)
    radares = {r: RADARES.get(r, r) for r in a.radares.split(",") if r}
    estado = recolectar(a.destino, radares, libre_min_gb=a.libre_min_gb, tope_gb=a.tope_gb)
    resumen = ", ".join(
        f"{r}: +{i['nuevas_ultima_vuelta']} (total {i['imagenes']}, última {i['ultima']})"
        for r, i in estado["radares"].items()
    )
    print(f"recoleccion: {resumen}; ocupado {estado['ocupado_mb']} MB"
          + (f"; DETENIDA: {estado['detenido']}" if estado["detenido"] else ""))
    armar_mascaras(a.destino)
    return 0


def armar_mascaras(destino, timeout=180):
    """Máscara de ecos fijos de cada radar, una vez por día (scripts/mascaras_recoleccion.py).

    Necesita numpy y Pillow: usa el Python del proyecto. Si no está o falla,
    la recolección sigue igual."""
    raiz = Path(__file__).resolve().parents[1]
    for py in (raiz / ".venv-ia/bin/python", raiz / ".venv/bin/python"):
        if py.exists():
            break
    else:
        return
    try:
        import subprocess
        salida = subprocess.run([str(py), str(raiz / "scripts/mascaras_recoleccion.py"), "--destino", str(destino)],
                                capture_output=True, text=True, timeout=timeout)
        texto = (salida.stdout or salida.stderr or "").strip()
        if texto:
            print(texto[-600:])
    except Exception as exc:  # nunca frena la recolección
        print(f"mascaras: no se pudieron armar ({type(exc).__name__})")


if __name__ == "__main__":
    sys.exit(main())
