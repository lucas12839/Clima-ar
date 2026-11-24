#!/usr/bin/env python3
"""Etapa C, punto 11: vigilancia que avisa a tiempo, aunque se corte todo.

El 23/11/2026 no entró radar de 5:10 a 11:20 (hora de Argentina) y nadie se
enteró: la clave de GitHub del VPS había vencido (error 401, sin latido) y el
respaldo de GitHub fallaba en cada consulta. El control de salud de la cadena
(salud_cadena.py) corre dentro de Intelligence, que justamente deja de correr
cuando no entra radar, así que no podía avisar.

Esto corre en el vigilante del detector (climaar-detector-sinarame.yml), que
lanzan el reloj encadenado de CWOP, el programado de GitHub y el reloj
externo (docs/RELOJ_EXTERNO.md). Revisa:

- latido del detector del VPS (y por qué puede faltar);
- clave de GitHub del VPS: avisa 14 días antes de que venza;
- actualización automática del VPS desde main (deploy/actualizar_vps.sh);
- edad del radar en main;
- respaldo de GitHub: si sus últimas corridas fallan;
- reloj encadenado de CWOP: si se cortó, lo vuelve a lanzar;
- alertas del control de salud de la cadena que siguen más de 20 min (antes
  quedaban escritas en salud_cadena.json y nadie las veía: re-auditoría 23/11);
- la IA (Intelligence): que analice el último radar y que no falle seguido
  (23/11/2026: 4 imágenes quedaron sin analizar por un error y nadie avisó);
- reloj externo (cron-job.org): si venía llamando y dejó de llamar;
- la app pública (página y datos desde el VPS): que responda (24/11/2026).

Avisa abriendo un "issue" de GitHub (le llega a Lucas como notificación al
celular y por correo), agrega un comentario si cambian los problemas y lo
cierra solo cuando todo vuelve a la normalidad. Nunca corta nada.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[1]
ARG = timezone(timedelta(hours=-3))
TITULO = "Vigilancia ClimaAR: hay problemas en la cadena"
DUENO = "lucas12839"

LATIDO_MAX_MIN = 15.0          # el latido se publica cada 2 min
RADAR_MAX_MIN = 40.0           # con el detector caído
RADAR_MAX_SINARAME_MIN = 60.0  # con el detector sano: SINARAME a veces llega 30 min o más tarde
CLAVE_AVISO_DIAS = 14
ACTUALIZACION_MAX_HORAS = 3.0  # el VPS se actualiza cada 10 min
CWOP_MAX_MIN = 30.0
RESPALDO_FALLAS = 2
EXTERNO_MIN_LLAMADAS_DIA = 10
EXTERNO_MAX_MIN = 30.0
IA_ATRASO_MAX_MIN = 15.0       # la IA tarda ~2 min después del radar; refresca cada 5
IA_FALLAS = 2
SALUD_PERSISTENTE_MIN = 20.0
# Dirección pública de la app (variable de repo CLIMAAR_APP_URL si se cambia
# por un nombre propio; la sslip.io sigue andando igual).
APP_URL = "https://104-237-9-251.sslip.io"
APP_INTENTOS = 3


def parse(texto) -> datetime | None:
    if not texto:
        return None
    t = str(texto).strip().replace("Z", "+00:00").replace(" UTC", "+00:00")
    try:
        d = datetime.fromisoformat(t)
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def minutos(desde: datetime | None, ahora: datetime) -> float | None:
    return None if desde is None else (ahora - desde).total_seconds() / 60.0


def hora_arg(t: datetime) -> str:
    return t.astimezone(ARG).strftime("%d/%m %H:%M")


# --------------------------------------------------------------------------
# Revisión (pura: tests/test_vigilancia.py)
# --------------------------------------------------------------------------
def revisar(ahora: datetime, latido: str | None, info: dict | None, radar_ts: str | None,
            respaldo: list[dict], cwop: list[dict], externo: list[dict] | None = None,
            ia: dict | None = None, ia_corridas: list[dict] | None = None,
            salud: dict | None = None, app: dict | None = None) -> list[dict[str, str]]:
    """Problemas encontrados: [{"clave", "texto"}]. La clave sirve para saber si
    el problema es el mismo de antes (no se repite el aviso)."""
    problemas = []
    info = info or {}

    edad_latido = minutos(parse(latido), ahora)
    # Si el detector avisó que no puede consultar SINARAME, cortó el latido a
    # propósito: se avisa apenas el respaldo de GitHub lo toma (5 min).
    con_error = isinstance(info.get("error_consultas"), dict)
    latido_ok = edad_latido is not None and edad_latido <= (5.0 if con_error else LATIDO_MAX_MIN)
    if not latido_ok:
        hace = "nunca dio latido" if edad_latido is None else f"no da latido hace {edad_latido:.0f} min"
        err = info.get("error_consultas") if isinstance(info.get("error_consultas"), dict) else None
        if err:
            problemas.append({"clave": "latido", "texto":
                f"El detector del VPS {hace} porque no puede consultar SINARAME "
                f"({err.get('consecutivos')} consultas seguidas fallaron; último error: "
                f"{str(err.get('ultimo_error') or 'sin detalle')[:160]}). Si dice que falta el navegador: "
                "`runuser -u climaar -- env HOME=/opt/clima-ar /opt/Clima-ar/.venv-ia/bin/python -m "
                "playwright install chromium`. Mientras tanto trabaja el respaldo de GitHub."})
        else:
            problemas.append({"clave": "latido", "texto":
            f"El detector del VPS {hace}. Causas posibles: la clave de GitHub del VPS venció o la "
            "rechazan (error 401), el VPS está apagado o sin internet, o el servicio se cayó. "
            "Mientras tanto trabaja el respaldo de GitHub. Para ver la causa, por SSH: "
            "`journalctl -u climaar-detector -n 30`."})

    vence = parse(info.get("clave_vence"))
    if vence is not None:
        dias = (vence - ahora).total_seconds() / 86400.0
        if dias <= 0:
            problemas.append({"clave": "clave_vencida", "texto":
                f"La clave de GitHub del VPS venció el {hora_arg(vence)}. Crear una nueva y ponerla en "
                "`/etc/climaar-detector.env` (ver docs/DESPLIEGUE_DETECTOR.md)."})
        elif dias <= CLAVE_AVISO_DIAS:
            problemas.append({"clave": "clave_por_vencer", "texto":
                f"La clave de GitHub del VPS vence el {vence.astimezone(ARG):%d/%m/%Y} (faltan {dias:.0f} días). "
                "Crear una nueva antes y ponerla en `/etc/climaar-detector.env`."})

    act = info.get("actualizacion")
    if isinstance(act, dict):
        # La información se publica cada 6 h: la edad se mide al publicarla.
        edad_act = minutos(parse(act.get("utc")), parse(info.get("utc")) or ahora)
        if act.get("resultado") != "ok":
            problemas.append({"clave": "actualizacion_error", "texto":
                f"El VPS no se pudo actualizar desde main: {act.get('detalle') or 'sin detalle'}. "
                "Ver por SSH: `journalctl -u climaar-actualizar -n 30`."})
        elif latido_ok and (edad_act is None or edad_act > ACTUALIZACION_MAX_HORAS * 60):
            problemas.append({"clave": "actualizacion_parada", "texto":
                "El VPS no se actualiza solo desde hace más de 3 h (reloj `climaar-actualizar.timer` "
                "apagado). Ver por SSH: `systemctl list-timers climaar-actualizar`."})

    edad_radar = minutos(parse(radar_ts), ahora)
    if edad_radar is None or edad_radar > (RADAR_MAX_SINARAME_MIN if latido_ok else RADAR_MAX_MIN):
        if not latido_ok and any(r.get("conclusion") == "failure" for r in respaldo[:RESPALDO_FALLAS]):
            causa = "el detector del VPS está caído y el respaldo de GitHub también falla"
        elif not latido_ok:
            causa = "el detector del VPS está caído; el respaldo de GitHub no logra traer imágenes"
        else:
            causa = "el detector funciona: o SINARAME no publica, o falla el procesamiento del radar"
        problemas.append({"clave": "radar", "texto":
            f"No entra radar hace {edad_radar:.0f} min ({causa})." if edad_radar is not None
            else f"No se puede leer la hora del último radar ({causa})."})

    terminadas = [r for r in respaldo if r.get("status") == "completed"]
    fallas = terminadas[:RESPALDO_FALLAS]
    if len(fallas) == RESPALDO_FALLAS and all(r.get("conclusion") == "failure" for r in fallas):
        problemas.append({"clave": "respaldo", "texto":
            f"Las últimas {RESPALDO_FALLAS} corridas del respaldo de GitHub (vigilante del detector) "
            "terminaron con error: si el VPS se cae, no hay quien traiga el radar."})

    ultima_cwop = max((parse(r.get("created_at")) for r in cwop if parse(r.get("created_at"))), default=None)
    activa = any(r.get("status") in ("queued", "in_progress", "pending", "waiting") for r in cwop)
    edad_cwop = minutos(ultima_cwop, ahora)
    if not activa and (edad_cwop is None or edad_cwop > CWOP_MAX_MIN):
        problemas.append({"clave": "reloj_cwop", "texto":
            "El reloj encadenado (estaciones CWOP) se había cortado: se volvió a lanzar."})
    # Reloj externo (docs/RELOJ_EXTERNO.md): lanza el vigilante con la clave de Lucas.
    # Si en el último día llamó varias veces y hace 30 min que no llama, se cortó.
    llamadas = [t for r in (externo or []) if (t := parse(r.get("created_at")))]
    if len(llamadas) >= EXTERNO_MIN_LLAMADAS_DIA and minutos(max(llamadas), ahora) > EXTERNO_MAX_MIN:
        problemas.append({"clave": "reloj_externo", "texto":
            f"El reloj externo (cron-job.org) dejó de llamar hace {minutos(max(llamadas), ahora):.0f} min: "
            "revisar la tarea en cron-job.org (¿venció su clave?)."})
    # IA (Intelligence) -------------------------------------------------------
    # ia = {"radar": hora del radar que analizó, "utc": cuándo publicó}.
    if ia is not None:
        radar_ia, publicada = parse(ia.get("radar")), parse(ia.get("utc"))
        radar_main = parse(radar_ts)
        sin_ia = radar_main is not None and (radar_ia is None or radar_ia < radar_main)
        edad_ia = minutos(publicada, ahora)
        if (sin_ia and (edad_ia is None or edad_ia > IA_ATRASO_MAX_MIN)) or (
                edad_ia is not None and edad_ia > IA_ATRASO_MAX_MIN * 2):
            detalle = (f"el último radar ({hora_arg(radar_main)}) no tiene análisis" if sin_ia
                       else f"la IA no publica hace {edad_ia:.0f} min")
            problemas.append({"clave": "ia_atrasada", "texto":
                f"La IA está atrasada: {detalle}. Ver las últimas corridas de "
                "\"ClimaAR Intelligence Central\" en Actions."})
    # Alertas del control de salud que siguen ------------------------------------
    if isinstance(salud, dict):
        desde = salud.get("alertas_desde") or {}
        for texto in salud.get("alertas") or []:
            clave, hace = None, None
            for c, inicio in desde.items():
                if _misma_alerta(c, texto):
                    clave = c
                    hace = minutos(parse(inicio), ahora)
                    break
            if clave is not None and hace is not None and hace >= SALUD_PERSISTENTE_MIN:
                problemas.append({"clave": "salud_" + _corta(clave), "texto":
                    f"Control de salud (hace {hace:.0f} min): {texto}"})
    # App pública ----------------------------------------------------------------
    # app = {"url", "ok", "detalle"}: resultado de probar /api/health (varios intentos).
    if isinstance(app, dict) and app.get("ok") is False:
        problemas.append({"clave": "app_publica", "texto":
            f"La app pública ({app.get('url')}) no responde: {app.get('detalle') or 'sin detalle'}. "
            "Ver por SSH: `systemctl status climaar-api caddy` y `journalctl -u climaar-api -n 30`."})
    corridas_ia = [r for r in (ia_corridas or []) if r.get("status") == "completed"
                   and r.get("conclusion") in ("success", "failure")][:IA_FALLAS]
    if len(corridas_ia) == IA_FALLAS and all(r.get("conclusion") == "failure" for r in corridas_ia):
        problemas.append({"clave": "ia_falla", "texto":
            f"Las últimas {IA_FALLAS} corridas de la IA (Intelligence) terminaron con error: "
            "las imágenes de radar quedan sin analizar."})
    return problemas


def _misma_alerta(clave: str, texto: str) -> bool:
    import re
    return re.sub(r"[0-9]+(?:[.,][0-9]+)?", "N", texto)[:90] == clave


def _corta(clave: str) -> str:
    """Clave estable y corta (va en un comentario HTML del aviso)."""
    import hashlib
    return hashlib.sha1(clave.encode("utf-8")).hexdigest()[:10]


def cuerpo(problemas: list[dict], ahora: datetime) -> str:
    lineas = [f"@{DUENO} la vigilancia encontró esto ({hora_arg(ahora)}, hora de Argentina):", ""]
    lineas += [f"- {p['texto']}" for p in problemas]
    lineas += ["", "Este aviso se cierra solo cuando todo vuelve a la normalidad.",
               f"<!-- claves: {','.join(sorted(p['clave'] for p in problemas))} -->"]
    return "\n".join(lineas)


def claves_de(texto: str) -> set[str]:
    marca = "<!-- claves: "
    if marca not in (texto or ""):
        return set()
    return {c for c in texto.split(marca, 1)[1].split(" -->", 1)[0].split(",") if c}


# --------------------------------------------------------------------------
# GitHub (REST)
# --------------------------------------------------------------------------
class GitHub:
    def __init__(self, repo: str, token: str):
        self.repo, self.token = repo, token

    def api(self, metodo: str, ruta: str, cuerpo_json: Any = None) -> Any:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{self.repo}/{ruta}",
            data=json.dumps(cuerpo_json).encode() if cuerpo_json is not None else None, method=metodo,
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            datos = resp.read()
        return json.loads(datos) if datos else None

    def corridas(self, workflow: str, n: int = 10) -> list[dict]:
        return (self.api("GET", f"actions/workflows/{workflow}/runs?per_page={n}") or {}).get("workflow_runs") or []


def avisar(gh: GitHub, problemas: list[dict], ahora: datetime) -> str:
    abiertos = [i for i in gh.api("GET", "issues?state=open&per_page=50") or []
                if i.get("title") == TITULO and "pull_request" not in i]
    issue = abiertos[0] if abiertos else None
    # El reloj de CWOP ya se arregló solo al relanzarlo: no hace falta abrir un aviso por eso.
    importantes = [p for p in problemas if p["clave"] != "reloj_cwop"]
    if not importantes:
        if issue:
            gh.api("POST", f"issues/{issue['number']}/comments",
                   {"body": f"Todo normal otra vez ({hora_arg(ahora)}, hora de Argentina). Se cierra el aviso."})
            gh.api("PATCH", f"issues/{issue['number']}", {"state": "closed", "state_reason": "completed"})
            return "cerrado"
        return "sin_problemas"
    texto = cuerpo(problemas, ahora)
    if issue is None:
        gh.api("POST", "issues", {"title": TITULO, "body": texto})
        return "abierto"
    if claves_de(issue.get("body") or "") != {p["clave"] for p in problemas}:
        gh.api("PATCH", f"issues/{issue['number']}", {"body": texto})
        gh.api("POST", f"issues/{issue['number']}/comments", {"body": "Cambiaron los problemas:\n\n" + texto})
        return "actualizado"
    return "sin_cambios"


def probar_app(url: str, intentos: int = APP_INTENTOS, espera: float = 5.0, abrir=None) -> dict:
    """Prueba /api/health de la app pública. Un solo fallo no alcanza: varios intentos."""
    import time

    abrir = abrir or urllib.request.urlopen
    detalle = None
    for i in range(intentos):
        try:
            req = urllib.request.Request(url.rstrip("/") + "/api/health",
                                         headers={"User-Agent": "ClimaAR-vigilancia"})
            with abrir(req, timeout=15) as resp:
                if 200 <= getattr(resp, "status", 200) < 300:
                    return {"url": url, "ok": True, "detalle": None}
                detalle = f"respondió {getattr(resp, 'status', '?')}"
        except urllib.error.HTTPError as exc:
            detalle = f"respondió {exc.code}"
        except Exception as exc:
            detalle = f"{type(exc).__name__}: {str(exc)[:120]}"
        if i + 1 < intentos:
            time.sleep(espera)
    return {"url": url, "ok": False, "detalle": detalle}


def main() -> int:
    try:
        ahora = datetime.now(timezone.utc)
        repo = os.environ.get("GITHUB_REPOSITORY", "lucas12839/Clima-ar")
        gh = GitHub(repo, os.environ.get("GH_TOKEN", ""))
        try:
            info = json.loads(os.environ.get("CLIMAAR_DETECTOR_INFO") or "null")
        except ValueError:
            info = None
        ts = BASE / "data" / "radar" / "radar_timestamp.txt"
        radar_ts = ts.read_text(encoding="utf-8").strip() if ts.exists() else None
        propio = os.environ.get("GITHUB_RUN_ID")
        respaldo = [r for r in gh.corridas("climaar-detector-sinarame.yml")
                    if str(r.get("id")) != str(propio)]
        cwop = gh.corridas("climaar-cwop.yml", 5)
        desde = (ahora - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        externo = (gh.api("GET", f"actions/workflows/climaar-detector-sinarame.yml/runs?actor={DUENO}"
                          f"&event=workflow_dispatch&created=%3E%3D{desde}&per_page=50") or {}).get("workflow_runs") or []
        ia = None
        try:
            final = json.loads((BASE / "data" / "ia" / "resultado_final.json").read_text(encoding="utf-8"))
            ia = {"radar": (final.get("radar") or {}).get("timestamp"), "utc": final.get("actualizado_utc")}
        except (OSError, ValueError, AttributeError):
            ia = {}
        ia_corridas = gh.corridas("climaar-intelligence.yml", 6)
        try:
            salud = json.loads((BASE / "data" / "ia" / "salud_cadena.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            salud = None
        app = probar_app(os.environ.get("CLIMAAR_APP_URL") or APP_URL)
        problemas = revisar(ahora, os.environ.get("CLIMAAR_DETECTOR_LATIDO"), info, radar_ts, respaldo, cwop,
                            externo, ia, ia_corridas, salud, app)
        if any(p["clave"] == "reloj_cwop" for p in problemas):
            try:
                gh.api("POST", "actions/workflows/climaar-cwop.yml/dispatches", {"ref": "main"})
            except Exception as exc:
                print(f"::warning::No se pudo volver a lanzar el reloj de CWOP: {exc}")
        for p in problemas:
            print(f"::warning::Vigilancia: {p['texto']}")
        try:
            estado = avisar(gh, problemas, ahora)
        except urllib.error.HTTPError as exc:
            estado = f"no se pudo avisar ({exc.code})"
            print(f"::warning::Vigilancia: no se pudo abrir o cerrar el aviso: {exc}")
        print(f"Vigilancia: {len(problemas)} problemas, aviso: {estado}.")
    except Exception as exc:  # nunca corta nada
        print(f"::warning::Vigilancia no disponible: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
