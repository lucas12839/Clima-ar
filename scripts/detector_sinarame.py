#!/usr/bin/env python3
"""
ClimaAR - Detector de frames nuevos de SINARAME (reloj meteorologico).

Consulta SINARAME con frecuencia corta y configurable. La consulta es
liviana: solo lista los timestamps RMA10 publicados
(scraper/capturar_radar_smn.py --solo-detectar). NO descarga, NO procesa.

  * timestamp nuevo (mayor al ultimo procesado)  -> despacha UNA vez el
    pipeline Radar -> Tracking -> Nowcast -> IA con ese radar_timestamp;
  * sin cambios -> termina el tick enseguida y, entre frames, pide un
    refresco de Inteligencia con el ultimo radar valido (sin reprocesar
    radar) cada CLIMAAR_REFRESCO_MIN minutos;
  * un fallo de SINARAME nunca detiene la Inteligencia.

El reloj es el timestamp real del frame, nunca el de GitHub.

Debe correr como servicio persistente (ver docs/ARQUITECTURA_RELOJ_RADAR.md):
    python scripts/detector_sinarame.py --loop
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import radar_estado as re_  # noqa: E402

ESTADO_LOCAL = Path(".climaar/detector_estado.json")

WORKFLOW_RADAR = "probar-radar-smn.yml"
WORKFLOW_INTELLIGENCE = "climaar-intelligence.yml"
WORKFLOW_DETECTOR = "climaar-detector-sinarame.yml"


def _env_float(nombre, defecto):
    try:
        return float(os.environ.get(nombre, defecto))
    except ValueError:
        return float(defecto)


INTERVALO_SEG = _env_float("CLIMAAR_DETECTOR_INTERVALO_SEG", 60)
REFRESCO_MIN = _env_float("CLIMAAR_REFRESCO_MIN", 10)
REDESPACHO_MIN = _env_float("CLIMAAR_REDESPACHO_MIN", 10)


# ------------------------------------------------------------
# decisiones puras
# ------------------------------------------------------------

def decidir(
    nuevo_ts,
    ultimo_procesado,
    despachado_ts,
    despachado_utc,
    ahora,
    redespacho_min=REDESPACHO_MIN,
):
    """sin_datos | sin_cambio | en_proceso | despachar"""
    nuevo = re_.parse_utc(nuevo_ts)
    if nuevo is None:
        return "sin_datos"

    procesado = re_.parse_utc(ultimo_procesado)
    if procesado is not None and nuevo <= procesado:
        return "sin_cambio"

    if re_.parse_utc(despachado_ts) == nuevo:
        enviado = re_.parse_utc(despachado_utc)
        if enviado is not None and (
            ahora - enviado
        ) < timedelta(minutes=redespacho_min):
            return "en_proceso"

    return "despachar"


def refresco_debido(ahora, ultimo_refresco_utc, refresco_min=REFRESCO_MIN):
    if refresco_min <= 0:
        return False
    ultimo = re_.parse_utc(ultimo_refresco_utc)
    return ultimo is None or (
        ahora - ultimo
    ) >= timedelta(minutes=refresco_min)


# ------------------------------------------------------------
# GitHub API (solo stdlib)
# ------------------------------------------------------------

class GitHub:

    def __init__(self, repo, token, ref="main"):
        self.repo = repo
        self.token = token
        self.ref = ref

    def _req(self, metodo, ruta, cuerpo=None, accept="application/vnd.github+json"):
        data = json.dumps(cuerpo).encode() if cuerpo is not None else None
        req = urllib.request.Request(
            f"https://api.github.com/repos/{self.repo}/{ruta}",
            data=data,
            method=metodo,
            headers={
                "Authorization": "Bearer " + str(self.token),
                "Accept": accept,
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()

    def despachar(self, workflow, inputs=None):
        self._req(
            "POST",
            f"actions/workflows/{workflow}/dispatches",
            {"ref": self.ref, "inputs": inputs or {}},
        )

    def activos(self, workflow):
        total = 0
        for estado in ("queued", "in_progress"):
            raw = self._req(
                "GET",
                f"actions/workflows/{workflow}/runs?status={estado}&per_page=1",
            )
            total += int(json.loads(raw).get("total_count", 0))
        return total

    def ultimo_procesado(self):
        raw = self._req(
            "GET",
            f"contents/data/radar/estado_radar.json?ref={self.ref}",
            accept="application/vnd.github.raw",
        )
        return json.loads(raw).get("last_processed_radar_timestamp")


# ------------------------------------------------------------
# consulta liviana a SINARAME
# ------------------------------------------------------------

def consultar_sinarame(timeout=240):
    """Devuelve el timestamp UTC (ISO) mas reciente publicado."""
    raiz = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        salida = Path(tmp) / "candidatos.json"
        subprocess.run(
            [
                sys.executable,
                str(raiz / "scraper" / "capturar_radar_smn.py"),
                "--solo-detectar",
                "--salida",
                str(salida),
            ],
            cwd=raiz,
            check=True,
            timeout=timeout,
            stdout=subprocess.DEVNULL,
        )
        datos = json.loads(salida.read_text(encoding="utf-8"))
    fechas = [
        re_.parse_utc(c.get("utc")) for c in datos.get("candidatos", [])
    ]
    fechas = [f for f in fechas if f is not None]
    return re_.iso(max(fechas)) if fechas else None


# ------------------------------------------------------------
# detector
# ------------------------------------------------------------

class Detector:

    def __init__(
        self,
        consultar,
        leer_procesado,
        despachar_radar,
        despachar_refresco,
        hay_actividad,
        reloj=re_.ahora_utc,
        estado_path=ESTADO_LOCAL,
        refresco_min=REFRESCO_MIN,
        redespacho_min=REDESPACHO_MIN,
    ):
        self.consultar = consultar
        self.leer_procesado = leer_procesado
        self.despachar_radar = despachar_radar
        self.despachar_refresco = despachar_refresco
        self.hay_actividad = hay_actividad
        self.reloj = reloj
        self.estado_path = Path(estado_path)
        self.refresco_min = refresco_min
        self.redespacho_min = redespacho_min

    def _estado(self):
        try:
            return json.loads(self.estado_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _guardar(self, estado):
        re_.escribir_json_atomico(self.estado_path, estado)

    def tick(self):
        ahora = self.reloj()
        est = self._estado()
        est["last_check_utc"] = re_.iso(ahora)
        accion = "sin_cambio"

        try:
            nuevo = self.consultar()
            est["collector_state"] = "ok"
            est["errores_consecutivos"] = 0
        except Exception as exc:  # SINARAME caido: la IA sigue
            nuevo = None
            accion = "error_sinarame"
            est["collector_state"] = "error"
            est["ultimo_error"] = str(exc)[:200]
            est["errores_consecutivos"] = int(
                est.get("errores_consecutivos", 0)
            ) + 1

        if nuevo:
            visto = re_.parse_utc(est.get("last_seen_radar_timestamp"))
            if visto is None or re_.parse_utc(nuevo) > visto:
                est["last_seen_radar_timestamp"] = nuevo

        procesado = est.get("last_processed_radar_timestamp")
        try:
            remoto = self.leer_procesado()
            if remoto:
                procesado = remoto
                est["last_processed_radar_timestamp"] = remoto
        except Exception:
            pass  # se usa el ultimo conocido

        if nuevo:
            decision = decidir(
                nuevo,
                procesado,
                est.get("dispatched_radar_timestamp"),
                est.get("dispatched_utc"),
                ahora,
                self.redespacho_min,
            )
            accion = decision
            if decision == "despachar":
                try:
                    self.despachar_radar(nuevo)
                    est["dispatched_radar_timestamp"] = nuevo
                    est["dispatched_utc"] = re_.iso(ahora)
                    est["collector_state"] = "despachado"
                except Exception as exc:
                    accion = "error_despacho"
                    est["ultimo_error"] = str(exc)[:200]

        # Entre frames la IA sigue con el ultimo radar valido.
        if accion in ("sin_cambio", "en_proceso", "error_sinarame") and refresco_debido(
            ahora, est.get("last_refresh_utc"), self.refresco_min
        ):
            try:
                if not self.hay_actividad():
                    self.despachar_refresco()
                    est["last_refresh_utc"] = re_.iso(ahora)
                    accion = accion + "+refresco"
            except Exception as exc:
                est["ultimo_error"] = str(exc)[:200]

        self._guardar(est)
        return accion


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--loop", action="store_true", help="servicio persistente")
    p.add_argument("--intervalo-seg", type=float, default=INTERVALO_SEG)
    p.add_argument(
        "--duracion-max-min", type=float, default=0,
        help="0 = sin limite (servicio); en Actions < limite del job",
    )
    p.add_argument(
        "--relevo", action="store_true",
        help="al terminar, relanza el workflow del detector (Actions)",
    )
    p.add_argument("--sin-despachar", action="store_true")
    a = p.parse_args(argv)

    repo = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    gh = GitHub(repo, token, os.environ.get("CLIMAAR_REF", "main"))

    if a.sin_despachar:
        despachar_radar = lambda ts: print(f"[dry-run] radar {ts}")  # noqa: E731
        despachar_refresco = lambda: print("[dry-run] refresco")  # noqa: E731
        hay_actividad = lambda: False  # noqa: E731
    else:
        if not repo or not token:
            print("ERROR: faltan GITHUB_REPOSITORY y GH_TOKEN/GITHUB_TOKEN")
            return 2
        despachar_radar = lambda ts: gh.despachar(  # noqa: E731
            WORKFLOW_RADAR, {"radar_timestamp": ts}
        )
        despachar_refresco = lambda: gh.despachar(  # noqa: E731
            WORKFLOW_INTELLIGENCE, {"modo": "refresco"}
        )
        hay_actividad = lambda: (  # noqa: E731
            gh.activos(WORKFLOW_RADAR) + gh.activos(WORKFLOW_INTELLIGENCE)
        ) > 0

    det = Detector(
        consultar=consultar_sinarame,
        leer_procesado=gh.ultimo_procesado if token and repo else (lambda: None),
        despachar_radar=despachar_radar,
        despachar_refresco=despachar_refresco,
        hay_actividad=hay_actividad,
        redespacho_min=REDESPACHO_MIN,
    )

    if not a.loop:
        print(det.tick())
        return 0

    parar = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: parar.update(v=True))
    signal.signal(signal.SIGINT, lambda *_: parar.update(v=True))

    inicio = time.monotonic()
    while not parar["v"]:
        t0 = time.monotonic()
        print(
            f"{datetime.now(timezone.utc).isoformat()} tick: {det.tick()}",
            flush=True,
        )
        if a.duracion_max_min and (
            time.monotonic() - inicio
        ) >= a.duracion_max_min * 60:
            break
        resto = max(5.0, a.intervalo_seg - (time.monotonic() - t0))
        fin = time.monotonic() + resto
        while not parar["v"] and time.monotonic() < fin:
            time.sleep(min(1.0, fin - time.monotonic()))

    if a.relevo and not parar["v"] and not a.sin_despachar:
        try:
            gh.despachar(WORKFLOW_DETECTOR)
            print("Relevo del detector solicitado.")
        except Exception as exc:
            print(f"ADVERTENCIA: relevo fallido: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
