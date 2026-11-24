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

Despliegue: servicio persistente (systemd) en una VM/Raspberry Pi:
    python scripts/detector_sinarame.py --loop --latido
Ver docs/DESPLIEGUE_DETECTOR.md. GitHub Actions solo hace de watchdog:
lanza este mismo script unos minutos si el latido del servicio se apaga.
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
VARIABLE_LATIDO = "CLIMAAR_DETECTOR_LATIDO"
# Etapa C (23/11/2026): qué código corre en el VPS, cuándo vence su clave de
# GitHub y cómo salió la última actualización automática. Lo leen la
# vigilancia (scripts/vigilancia.py) y el control de salud de la cadena.
VARIABLE_INFO = "CLIMAAR_DETECTOR_INFO"
INFO_CADA_SEG = 6 * 3600.0
ACTUALIZACION_LOCAL = Path(".climaar/actualizacion.json")
# El watchdog considera caido al servicio si el latido supera esto.
LATIDO_MAX_MIN = 5.0
LATIDO_CADA_SEG = 120.0
# 23/11/2026: una actualización de Playwright dejó al detector sin navegador y
# falló cada consulta durante 27 min sin que nadie se enterara (el latido seguía).
# Si falla seguido (~10 min), deja de dar latido: el respaldo de GitHub toma la
# captura y la vigilancia avisa con el error.
ERRORES_SIN_LATIDO = 20


def _env_float(nombre, defecto):
    try:
        return float(os.environ.get(nombre, defecto))
    except ValueError:
        return float(defecto)


INTERVALO_SEG = _env_float("CLIMAAR_DETECTOR_INTERVALO_SEG", 30)
REFRESCO_MIN = _env_float("CLIMAAR_REFRESCO_MIN", 5)
# 23/11/2026: 10 -> 6 min. El Radar tarda ~1,5 min; si en 6 min el frame no
# quedó procesado y no hay ningún Radar en marcha, se vuelve a pedir.
REDESPACHO_MIN = _env_float("CLIMAAR_REDESPACHO_MIN", 6)
ACTIVIDAD_MAX_MIN = _env_float("CLIMAAR_ACTIVIDAD_MAX_MIN", 60)


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


def contar_runs_recientes(runs, ahora, max_min=60):
    """Cuenta runs creados hace menos de max_min minutos (ignora fantasmas)."""
    total = 0
    for run in runs or []:
        creado = re_.parse_utc((run or {}).get("created_at"))
        if creado is not None and (ahora - creado) <= timedelta(minutes=max_min):
            total += 1
    return total


def refresco_debido(ahora, ultimo_refresco_utc, refresco_min=REFRESCO_MIN):
    if refresco_min <= 0:
        return False
    ultimo = re_.parse_utc(ultimo_refresco_utc)
    return ultimo is None or (
        ahora - ultimo
    ) >= timedelta(minutes=refresco_min)


def servicio_vivo(latido, ahora=None, max_min=LATIDO_MAX_MIN):
    """True si el ultimo latido del servicio es reciente."""
    ahora = ahora or re_.ahora_utc()
    dt = re_.parse_utc(latido)
    if dt is None:
        return False
    return timedelta(0) - timedelta(minutes=1) <= (ahora - dt) <= timedelta(
        minutes=max_min
    )


# ------------------------------------------------------------
# GitHub API (solo stdlib)
# ------------------------------------------------------------

class GitHub:

    def __init__(self, repo, token, ref="main"):
        self.repo = repo
        self.token = token
        self.ref = ref
        # GitHub informa en cada respuesta cuándo vence una clave con fecha
        # (encabezado github-authentication-token-expiration).
        self.clave_vence = None
        self.clave_rechazada_utc = None

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
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                vence = resp.headers.get("github-authentication-token-expiration")
                if vence:
                    self.clave_vence = vence.strip()
                self.clave_rechazada_utc = None
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                # 23/11/2026: la clave venció y el VPS quedó 6 h sin latido.
                self.clave_rechazada_utc = re_.iso(re_.ahora_utc())
                print("ERROR: GitHub rechazó la clave del detector (401): vencida o borrada. "
                      "Crear una nueva y ponerla en /etc/climaar-detector.env", flush=True)
            raise

    def despachar(self, workflow, inputs=None):
        self._req(
            "POST",
            f"actions/workflows/{workflow}/dispatches",
            {"ref": self.ref, "inputs": inputs or {}},
        )

    def activos(self, workflow, ahora=None, max_min=ACTIVIDAD_MAX_MIN):
        """Runs en cola o en curso creados en los últimos max_min minutos.

        GitHub puede dejar runs "fantasma" en estado queued para siempre
        (hay 17 de radar del 7/11/2026 que no se pueden cancelar ni borrar).
        Contarlos hacía creer que siempre había actividad y bloqueaba el
        refresco de la IA entre frames.
        """
        ahora = ahora or re_.ahora_utc()
        total = 0
        for estado in ("queued", "pending", "waiting", "in_progress"):
            raw = self._req(
                "GET",
                f"actions/workflows/{workflow}/runs?status={estado}&per_page=30",
            )
            total += contar_runs_recientes(
                json.loads(raw).get("workflow_runs", []), ahora, max_min
            )
        return total

    def escribir_latido(self, valor, nombre=VARIABLE_LATIDO):
        """Variable de repositorio que el watchdog de Actions lee (vars.*)."""
        cuerpo = {"name": nombre, "value": valor}
        try:
            self._req("PATCH", f"actions/variables/{nombre}", cuerpo)
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            self._req("POST", "actions/variables", cuerpo)

    def escribir_info(self, info, nombre=VARIABLE_INFO):
        self.escribir_latido(json.dumps(info, ensure_ascii=False, separators=(",", ":")), nombre)

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
        radar_activo=None,
        estado_path=ESTADO_LOCAL,
        refresco_min=REFRESCO_MIN,
        redespacho_min=REDESPACHO_MIN,
    ):
        self.consultar = consultar
        self.leer_procesado = leer_procesado
        self.despachar_radar = despachar_radar
        self.despachar_refresco = despachar_refresco
        self.hay_actividad = hay_actividad
        # Un redespacho con un Radar todavía en marcha o en espera solo
        # duplicaría el run (y podría cancelar una IA en espera).
        self.radar_activo = radar_activo or (lambda: False)
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
            est["ultimo_error_sinarame"] = str(exc)[:200]
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
            es_redespacho = re_.parse_utc(est.get("dispatched_radar_timestamp")) == re_.parse_utc(nuevo)
            if decision == "despachar" and es_redespacho:
                try:
                    if self.radar_activo():
                        decision = accion = "en_proceso"
                except Exception:
                    pass  # sin respuesta de GitHub: se redespacha (es idempotente)
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


def actualizacion_con_error(raiz=None):
    """Hora de la última actualización automática fallida, o None."""
    raiz = Path(raiz) if raiz else Path(__file__).resolve().parents[1]
    try:
        act = json.loads((raiz / ACTUALIZACION_LOCAL).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(act, dict) and act.get("resultado") == "error":
        return act.get("utc") or "error"
    return None


def info_servicio(gh, raiz=None, error_consultas=None):
    """Qué código corre, cuándo vence la clave y la última actualización automática."""
    raiz = Path(raiz) if raiz else Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "-C", str(raiz), "rev-parse", "--short=12", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        commit = None
    try:
        actualizacion = json.loads((raiz / ACTUALIZACION_LOCAL).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        actualizacion = None
    return {"utc": re_.iso(re_.ahora_utc()), "commit": commit, "clave_vence": gh.clave_vence,
            "actualizacion": actualizacion, "python": sys.prefix,
            "error_consultas": error_consultas}


def mostrar_estado(ahora=None, estado_path=ESTADO_LOCAL, max_min=LATIDO_MAX_MIN):
    """Salud local: ultima consulta reciente y sin error persistente."""
    ahora = ahora or re_.ahora_utc()
    try:
        est = json.loads(Path(estado_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print("SIN ESTADO: el detector nunca corrio en esta maquina.")
        return 1
    ultimo = re_.parse_utc(est.get("last_check_utc"))
    edad = (ahora - ultimo).total_seconds() / 60.0 if ultimo else None
    sano = edad is not None and edad <= max_min
    print(json.dumps({
        "sano": sano,
        "ultima_consulta_utc": est.get("last_check_utc"),
        "edad_ultima_consulta_min": round(edad, 1) if edad is not None else None,
        "collector_state": est.get("collector_state"),
        "last_seen_radar_timestamp": est.get("last_seen_radar_timestamp"),
        "last_processed_radar_timestamp": est.get(
            "last_processed_radar_timestamp"
        ),
        "errores_consecutivos": est.get("errores_consecutivos", 0),
        "ultimo_error": est.get("ultimo_error"),
    }, ensure_ascii=False, indent=2))
    return 0 if sano else 1


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--loop", action="store_true", help="servicio persistente")
    p.add_argument("--intervalo-seg", type=float, default=INTERVALO_SEG)
    p.add_argument(
        "--duracion-max-min", type=float, default=0,
        help="0 = sin limite (servicio); en Actions < limite del job",
    )
    p.add_argument(
        "--latido", action="store_true",
        help="publica el latido del servicio (variable de repo) para el watchdog",
    )
    p.add_argument("--sin-despachar", action="store_true")
    p.add_argument(
        "--servicio-vivo", metavar="LATIDO", default=None,
        help="imprime true/false segun el latido recibido y sale",
    )
    p.add_argument(
        "--estado", action="store_true",
        help="salud local del detector; sale 0 si esta sano, 1 si no",
    )
    a = p.parse_args(argv)

    if a.servicio_vivo is not None:
        print("true" if servicio_vivo(a.servicio_vivo) else "false")
        return 0

    if a.estado:
        return mostrar_estado()

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
        radar_activo=(lambda: gh.activos(WORKFLOW_RADAR, max_min=30) > 0) if not a.sin_despachar else None,
        redespacho_min=REDESPACHO_MIN,
    )

    if not a.loop:
        print(det.tick())
        return 0

    parar = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: parar.update(v=True))
    signal.signal(signal.SIGINT, lambda *_: parar.update(v=True))

    inicio = time.monotonic()
    ultimo_latido = 0.0
    ultima_info = 0.0
    error_publicado = False
    error_actualizacion_publicado = None
    ticks = consultas_ok = 0
    while not parar["v"]:
        t0 = time.monotonic()
        accion = det.tick()
        ticks += 1
        consultas_ok += not str(accion).startswith("error_sinarame")
        print(
            f"{datetime.now(timezone.utc).isoformat()} tick: {accion}",
            flush=True,
        )
        errores = int(det._estado().get("errores_consecutivos", 0) or 0)
        if a.latido and not a.sin_despachar and errores >= ERRORES_SIN_LATIDO:
            if not error_publicado:
                try:
                    gh.escribir_info(info_servicio(gh, error_consultas={
                        "consecutivos": errores,
                        "ultimo_error": det._estado().get("ultimo_error_sinarame"),
                        "utc": re_.iso(re_.ahora_utc())}))
                    error_publicado = True
                except Exception as exc:
                    print(f"ADVERTENCIA: error no publicado: {exc}", flush=True)
            print(f"ERROR: {errores} consultas seguidas fallaron: sin latido, toma el respaldo.", flush=True)
        elif (
            a.latido
            and not a.sin_despachar
            and time.monotonic() - ultimo_latido >= LATIDO_CADA_SEG
        ):
            if error_publicado:
                ultima_info = 0.0  # se recuperó: se publica enseguida sin el error
                error_publicado = False
            try:
                gh.escribir_latido(re_.iso(re_.ahora_utc()))
                ultimo_latido = time.monotonic()
            except Exception as exc:  # el latido nunca frena al detector
                print(f"ADVERTENCIA: latido no publicado: {exc}", flush=True)
            # Una actualización automática que falla se publica enseguida (no
            # a las 6 h): la vigilancia avisa con la causa (24/11/2026).
            act_error = actualizacion_con_error()
            if act_error and act_error != error_actualizacion_publicado:
                ultima_info = 0.0
            if ultimo_latido and time.monotonic() - ultima_info >= INFO_CADA_SEG:
                try:
                    gh.escribir_info(info_servicio(gh))
                    ultima_info = time.monotonic()
                    error_actualizacion_publicado = act_error
                except Exception as exc:
                    print(f"ADVERTENCIA: información del servicio no publicada: {exc}", flush=True)
        if a.duracion_max_min and (
            time.monotonic() - inicio
        ) >= a.duracion_max_min * 60:
            break
        resto = max(5.0, a.intervalo_seg - (time.monotonic() - t0))
        fin = time.monotonic() + resto
        while not parar["v"] and time.monotonic() < fin:
            time.sleep(min(1.0, fin - time.monotonic()))

    if a.duracion_max_min and ticks and not consultas_ok:
        # Respaldo de GitHub: si ninguna consulta a SINARAME funcionó, la corrida
        # termina con error para que se vea (23/11/2026: fallaba en cada consulta
        # y la corrida igual salía "bien").
        print(f"ERROR: ninguna de las {ticks} consultas a SINARAME funcionó.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
