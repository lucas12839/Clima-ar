#!/usr/bin/env python3
"""
ClimaAR - Estado del reloj radar (idempotencia y cola).

El reloj meteorologico es el timestamp REAL del frame RMA10 publicado por
SINARAME, nunca el reloj de GitHub. Este modulo mantiene, en
data/radar/estado_radar.json, el estado que comparten el collector, el
procesamiento y la Inteligencia:

  last_seen_radar_timestamp       ultimo frame visto en SINARAME
  last_processed_radar_timestamp  ultimo frame procesado por completo
  radar_timestamp                 frame del radar activo
  radar_age_minutos / vigencia_radar   (calculados en cada lectura)
  estado_collector / estado_procesamiento

Reglas:
  * un mismo radar_timestamp se procesa UNA sola vez (reclamar);
  * si llega un frame nuevo mientras otro se procesa, se encola como
    pendiente (solo el mas nuevo) y no se lanza un segundo procesamiento;
  * todas las escrituras son atomicas (tmp + fsync + replace) y bajo
    un lock de archivo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

ESTADO_PATH = Path("data/radar/estado_radar.json")
SCHEMA = 1

# Un reclamo de procesamiento sin completar mas viejo que esto se
# considera muerto (job cancelado) y puede tomarse de nuevo.
CLAIM_TTL_MIN = 30.0

# Mismos limites que Intelligence (apply_radar_age).
RADAR_FRESCO_MAX_MIN = 20.0
RADAR_ESTIMACION_MAX_MIN = 180.0


# ------------------------------------------------------------
# utilidades
# ------------------------------------------------------------

def parse_utc(valor):
    if not isinstance(valor, str) or not valor.strip():
        return None
    texto = valor.strip()
    if texto.endswith("Z"):
        texto = texto[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(texto)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc)


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat()


def ahora_utc():
    return datetime.now(timezone.utc)


def _vacio():
    return {
        "schema": SCHEMA,
        "last_seen_radar_timestamp": None,
        "last_processed_radar_timestamp": None,
        "radar_timestamp": None,
        "estado_collector": "idle",
        "estado_procesamiento": "idle",
        "procesando_radar_timestamp": None,
        "procesando_desde_utc": None,
        "pendiente_radar_timestamp": None,
        "ultimo_error": None,
        "actualizado_utc": None,
    }


def edad_radar_minutos(radar_timestamp, ahora=None):
    dt = parse_utc(radar_timestamp)
    if dt is None:
        return None
    ahora = ahora or ahora_utc()
    return max(0.0, (ahora - dt).total_seconds() / 60.0)


def vigencia(edad):
    """Clasificacion de la edad del radar (limites de Intelligence)."""
    if edad is None:
        return "sin_radar"
    if edad <= RADAR_FRESCO_MAX_MIN:
        return "radar_vigente"
    if edad <= RADAR_ESTIMACION_MAX_MIN:
        return "estimado_desde_ultimo_radar_valido"
    return "radar_obsoleto"


def escribir_json_atomico(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def publicar_frame_atomico(data, destino, validador=None):
    """
    Descarga a temporal -> valida -> reemplaza el radar activo.
    La IA nunca ve un archivo parcial: os.replace es atomico y el
    temporal vive en el mismo directorio.
    """
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if validador is not None:
            ok, detalle = validador(tmp.read_bytes())
            if not ok:
                raise ValueError(f"frame invalido: {detalle}")
        os.replace(tmp, destino)
    finally:
        if tmp.exists():
            tmp.unlink()


@contextmanager
def _lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with open(lock_path, "w") as lf:
        if fcntl is not None:
            fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lf, fcntl.LOCK_UN)


def leer(path=ESTADO_PATH):
    estado = _vacio()
    try:
        crudo = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(crudo, dict):
            estado.update(crudo)
    except (OSError, ValueError):
        pass
    return estado


def _guardar(estado, path, ahora):
    estado["actualizado_utc"] = iso(ahora)
    edad = edad_radar_minutos(estado.get("radar_timestamp"), ahora)
    estado["radar_age_minutos"] = round(edad, 1) if edad is not None else None
    estado["vigencia_radar"] = vigencia(edad)
    escribir_json_atomico(path, estado)


def snapshot(path=ESTADO_PATH, ahora=None):
    """Estado con la edad del radar recalculada AHORA (para la IA)."""
    ahora = ahora or ahora_utc()
    estado = leer(path)
    edad = edad_radar_minutos(estado.get("radar_timestamp"), ahora)
    estado["radar_age_minutos"] = round(edad, 1) if edad is not None else None
    estado["vigencia_radar"] = vigencia(edad)
    return estado


# ------------------------------------------------------------
# transiciones
# ------------------------------------------------------------

def registrar_visto(ts, path=ESTADO_PATH, ahora=None, estado_collector=None):
    ahora = ahora or ahora_utc()
    with _lock(path):
        estado = leer(path)
        visto = parse_utc(estado.get("last_seen_radar_timestamp"))
        nuevo = parse_utc(ts)
        if nuevo is not None and (visto is None or nuevo > visto):
            estado["last_seen_radar_timestamp"] = iso(nuevo)
        if estado_collector:
            estado["estado_collector"] = estado_collector
        _guardar(estado, path, ahora)
        return estado


def reclamar(ts, path=ESTADO_PATH, ahora=None):
    """
    Devuelve (procesar, motivo). Solo UNA llamada por radar_timestamp
    devuelve True.
    """
    ahora = ahora or ahora_utc()
    nuevo = parse_utc(ts)
    if nuevo is None:
        return False, "timestamp_invalido"

    with _lock(path):
        estado = leer(path)

        procesado = parse_utc(estado.get("last_processed_radar_timestamp"))
        if procesado is not None and nuevo <= procesado:
            return False, "ya_procesado"

        en_curso = parse_utc(estado.get("procesando_radar_timestamp"))
        desde = parse_utc(estado.get("procesando_desde_utc"))
        vivo = (
            en_curso is not None
            and desde is not None
            and (ahora - desde) < timedelta(minutes=CLAIM_TTL_MIN)
        )

        if vivo:
            if nuevo == en_curso:
                return False, "en_proceso"
            if nuevo > en_curso:
                pend = parse_utc(estado.get("pendiente_radar_timestamp"))
                if pend is None or nuevo > pend:
                    estado["pendiente_radar_timestamp"] = iso(nuevo)
                    _guardar(estado, path, ahora)
                return False, "encolado"
            return False, "obsoleto"

        visto = parse_utc(estado.get("last_seen_radar_timestamp"))
        if visto is None or nuevo > visto:
            estado["last_seen_radar_timestamp"] = iso(nuevo)
        estado["procesando_radar_timestamp"] = iso(nuevo)
        estado["procesando_desde_utc"] = iso(ahora)
        estado["estado_procesamiento"] = "procesando"
        estado["estado_collector"] = "procesando"
        estado["ultimo_error"] = None
        _guardar(estado, path, ahora)
        return True, "reclamado"


def completar(ts, path=ESTADO_PATH, ahora=None):
    ahora = ahora or ahora_utc()
    nuevo = parse_utc(ts)
    with _lock(path):
        estado = leer(path)
        en_curso = parse_utc(estado.get("procesando_radar_timestamp"))
        if nuevo is None or en_curso != nuevo:
            raise RuntimeError(
                "completar sin reclamo vigente para "
                f"{ts!r} (reclamo={estado.get('procesando_radar_timestamp')!r})"
            )
        estado["last_processed_radar_timestamp"] = iso(nuevo)
        estado["radar_timestamp"] = iso(nuevo)
        estado["procesando_radar_timestamp"] = None
        estado["procesando_desde_utc"] = None
        pend = parse_utc(estado.get("pendiente_radar_timestamp"))
        if pend is not None and pend <= nuevo:
            estado["pendiente_radar_timestamp"] = None
        estado["estado_procesamiento"] = "completado"
        estado["estado_collector"] = "idle"
        estado["ultimo_error"] = None
        _guardar(estado, path, ahora)
        return estado


def fallar(ts, motivo, path=ESTADO_PATH, ahora=None):
    """Libera el reclamo: el mismo frame podra reintentarse."""
    ahora = ahora or ahora_utc()
    nuevo = parse_utc(ts)
    with _lock(path):
        estado = leer(path)
        if parse_utc(estado.get("procesando_radar_timestamp")) == nuevo:
            estado["procesando_radar_timestamp"] = None
            estado["procesando_desde_utc"] = None
        estado["estado_procesamiento"] = "error"
        estado["estado_collector"] = "error"
        estado["ultimo_error"] = str(motivo)[:300]
        _guardar(estado, path, ahora)
        return estado


# ------------------------------------------------------------
# CLI (usada por los workflows)
# ------------------------------------------------------------

def _salida_github(clave, valor):
    ruta = os.environ.get("GITHUB_OUTPUT")
    if ruta:
        with open(ruta, "a", encoding="utf-8") as f:
            f.write(f"{clave}={valor}\n")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "accion",
        choices=(
            "reclamar", "completar", "fallar", "visto", "verificar-activo",
            "mostrar",
        ),
    )
    p.add_argument("--timestamp")
    p.add_argument("--motivo", default="")
    p.add_argument("--estado", default=str(ESTADO_PATH))
    a = p.parse_args(argv)

    ts = a.timestamp
    if ts is None and a.accion != "mostrar":
        archivo = Path("data/radar/radar_timestamp.txt")
        ts = archivo.read_text(encoding="utf-8").strip() if archivo.exists() else None

    if a.accion == "reclamar":
        ok, motivo = reclamar(ts, a.estado)
        print(f"reclamar {ts}: procesar={ok} motivo={motivo}")
        _salida_github("procesar", "true" if ok else "false")
        _salida_github("motivo", motivo)
        _salida_github("radar_timestamp", ts or "")
    elif a.accion == "completar":
        completar(ts, a.estado)
        print(f"procesamiento completado: {ts}")
    elif a.accion == "fallar":
        fallar(ts, a.motivo, a.estado)
        print(f"procesamiento fallido liberado: {ts}")
    elif a.accion == "visto":
        registrar_visto(ts, a.estado)
    elif a.accion == "verificar-activo":
        # Para refrescos de Intelligence: el radar activo debe ser un
        # frame completo ya procesado.
        est = leer(a.estado)
        activo = parse_utc(
            Path("data/radar/radar_timestamp.txt").read_text(
                encoding="utf-8"
            ).strip()
        )
        procesado = parse_utc(est.get("last_processed_radar_timestamp"))
        ok = activo is not None and activo == procesado
        print(f"radar activo={activo} procesado={procesado} ok={ok}")
        if not ok:
            return 1
    else:
        print(json.dumps(snapshot(a.estado), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
