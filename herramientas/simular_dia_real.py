#!/usr/bin/env python3
"""Reproduce un día real de SINARAME por la cadena COMPLETA, imagen por imagen.

A diferencia de simular_evento_real.py (imágenes del SMN, otra paleta), usa las
imágenes crudas de SINARAME guardadas en data/radar/historico, tal cual llegaron.
Para cada imagen del período (en orden):
1. La deja como frame recién llegado, con el atraso real de SINARAME (17 min).
2. Corre los pasos reales de los workflows (radar e Intelligence, con todas sus
   validaciones) con el reloj puesto en ese momento.
3. Guarda lo que ClimaAR mostraba en cada localidad: llegada, lluvia, granizo,
   viento, tormenta eléctrica, nivel y lo que está "en evaluación".

Sin conexión: no hay satélite, rayos ni estaciones de ese día (los pasos que los
necesitan siguen sin ellos, como en vivo cuando una fuente se cae). Lo que se
mide acá es la parte del radar, que es la que decide la llegada y la intensidad.

Uso (sobre una COPIA del repo, nunca sobre el repo real):
  python herramientas/simular_dia_real.py <copia> --desde 2026-11-13T15:00 --hasta 2026-11-13T21:00 --salida x.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EMU = ROOT / "herramientas" / "emular_workflow.py"
RELOJ = ROOT / "herramientas" / "reloj_simulado"
ATRASO_MIN = 17  # mediana real de SINARAME (llegadas_sinarame.jsonl)


def correr(copia: Path, wf: str, desde: str, hasta: str, env: dict) -> tuple[int, list[str]]:
    r = subprocess.run([sys.executable, str(EMU), str(copia), f".github/workflows/{wf}", sys.executable,
                        "--desde", desde, "--hasta", hasta], cwd=copia, env=env, capture_output=True, text=True)
    lineas = (r.stdout + r.stderr).splitlines()
    fallas = []
    for i, l in enumerate(lineas):
        if l.startswith("FALLA"):
            motivo = [x.strip() for x in lineas[i + 1:i + 30] if "ERROR" in x or "Error" in x]
            fallas.append(l + (" :: " + motivo[-1][:200] if motivo else ""))
    return sum(l.startswith("OK") for l in lineas), fallas


def resumen_localidad(loc: dict) -> dict:
    f = loc.get("fenomenos") or {}
    ll = loc.get("llegada") or {}
    sobre = loc.get("sobre_la_localidad") or {}
    return {
        "minutos_hasta_llegada": loc.get("minutos_hasta_llegada"),
        "llegada": {k: ll.get(k) for k in ("tipo", "minutos", "dbz", "grado", "fuente") if k in ll} if isinstance(ll, dict) else ll,
        "resumen_ahora": loc.get("resumen_ahora"), "resumen_al_llegar": loc.get("resumen_al_llegar"),
        "lluvia": {k: (f.get("lluvia") or {}).get(k) for k in ("grado", "categoria", "mm_1h", "mm_3h", "cuando")},
        "granizo": {k: (f.get("granizo") or {}).get(k) for k in ("grado", "categoria", "confirmado", "cuando")},
        "viento": {k: (f.get("viento") or {}).get(k) for k in ("grado", "categoria", "rafaga_kmh", "cuando")},
        "electrica": {k: (f.get("electrica") or {}).get(k) for k in ("grado", "categoria", "cuando")},
        "sobre_la_localidad": sobre if isinstance(sobre, dict) else None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("copia")
    ap.add_argument("--desde", required=True, help="hora UTC de la primera imagen")
    ap.add_argument("--hasta", required=True)
    ap.add_argument("--previas", type=int, default=6, help="imágenes anteriores para arrancar el seguimiento")
    ap.add_argument("--salida", default=None)
    a = ap.parse_args(argv)
    copia = Path(a.copia).resolve()
    desde = datetime.fromisoformat(a.desde).replace(tzinfo=timezone.utc)
    hasta = datetime.fromisoformat(a.hasta).replace(tzinfo=timezone.utc)
    hist = copia / "data/radar/historico"
    todos = {}
    for p in hist.glob("20*.png"):
        try:
            todos[datetime.strptime(p.stem, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)] = p
        except ValueError:
            pass
    guardadas = Path(copia / ".dia_real")
    if guardadas.exists():
        shutil.rmtree(guardadas)
    guardadas.mkdir()
    for t, p in todos.items():
        shutil.move(str(p), guardadas / p.name)
    previas = sorted(t for t in todos if desde - timedelta(minutes=10 * a.previas) <= t < desde)
    periodo = sorted(t for t in todos if desde <= t <= hasta)

    # Estado limpio, como un día que empieza; las imágenes previas arrancan el seguimiento.
    for rel in ("data/radar/seguimiento/estado.json", ".climaar/estado_rma10.json", "data/radar/historial_ciudad.json",
                "data/radar/historial_localidades.json", "data/radar/historial_puntos_control.json",
                "data/ia/estaciones/historial_superficie.json", "data/ia/estaciones/observaciones/bcp_actual.json",
                "data/ia/estaciones/observaciones/cwop_actual.json", "data/ia/goes19_intelligence.json",
                "data/radar/.flujo_campo.npz", "data/ia/resultado_final.json", "data/ia/celdas_localidad.json",
                # Lo de hoy quedaría "del futuro" para el día simulado: se arranca sin eso.
                "data/ia/modelos_region.json", "data/ia/ambiente_satelite.json", "data/ia/formacion_tormentas.json",
                "data/goes19/goes19_state.json", "data/ia/indicios_severos.json", "data/ia/puntaje_tormentas.json"):
        (copia / rel).unlink(missing_ok=True)

    # El pronóstico europeo guardado (el de hoy) se corre a la fecha del día simulado
    # (sin conexión no hay pronóstico histórico): el ambiente es aproximado.
    amb_path = copia / "data/meteorologia/ambiente_futuro.json"
    original = amb_path.read_text(encoding="utf-8")
    ref = datetime.fromisoformat(str(json.loads(original)["actualizado_utc"]).replace("Z", "+00:00"))

    # Los pronósticos guardados de hoy (WRF, ensambles) también quedarían "del futuro".
    shutil.rmtree(copia / "data/ia/modelos", ignore_errors=True)

    registros = []
    for i, t in enumerate(previas + periodo):
        nombre = t.strftime("%Y%m%d%H%M%S") + ".png"
        shutil.copy(guardadas / nombre, hist / nombre)
        shutil.copy(guardadas / nombre, copia / "data/radar/actual.png")
        (copia / "data/radar/radar_timestamp.txt").write_text(t.isoformat(), encoding="utf-8")
        ahora = t + timedelta(minutes=ATRASO_MIN)
        ambiente = json.loads(original)
        corrimiento = (ahora - timedelta(minutes=5)) - ref
        for punto in ambiente.get("puntos", []):
            for reg_h in punto.get("datos", []):
                h = datetime.fromisoformat(str(reg_h["tiempo_utc"]).replace("Z", "+00:00"))
                h = h if h.tzinfo else h.replace(tzinfo=timezone.utc)
                reg_h["tiempo_utc"] = (h + corrimiento).strftime("%Y-%m-%dT%H:%M")
        ambiente["actualizado_utc"] = (ahora - timedelta(minutes=5)).isoformat()
        amb_path.write_text(json.dumps(ambiente), encoding="utf-8")
        env = dict(os.environ, CLIMAAR_RELOJ_SIMULADO=ahora.isoformat(), CLIMAAR_AHORA_UTC=ahora.isoformat(),
                   PYTHONPATH=os.pathsep.join([str(RELOJ), os.environ.get("PYTHONPATH", "")]))
        r_ok, r_f = correr(copia, "probar-radar-smn.yml", "Procesar radar", "Validar cadena", env)
        i_ok, i_f = correr(copia, "climaar-intelligence.yml", "Nowcast por flujo", "Validar integridad", env)
        if t < desde:
            print(f"(arranque) {t:%H:%M} radar {r_ok} intel {i_ok} {r_f + i_f}", flush=True)
            continue
        reg = {"radar_utc": t.isoformat(), "ahora_utc": ahora.isoformat(), "pasos_ok": r_ok + i_ok, "fallas": r_f + i_f}
        try:
            if i_f:
                raise RuntimeError("Intelligence falló: resultado no actualizado")
            fin = json.loads((copia / "data/ia/resultado_final.json").read_text(encoding="utf-8"))
            tt = fin.get("tipo_tiempo") or {}
            reg["etiqueta"] = tt.get("etiqueta")
            reg["localidades"] = {k: resumen_localidad(v) for k, v in (tt.get("localidades") or {}).items()}
            ind = json.loads((copia / "data/ia/indicios_severos.json").read_text(encoding="utf-8")) \
                if (copia / "data/ia/indicios_severos.json").exists() else {}
            reg["indicios_severos"] = {k: ind.get(k) for k in ("disponible", "resumen", "eco_en_arco") if k in ind}
        except Exception as exc:
            reg["error_resultado"] = str(exc)
        registros.append(reg)
        pa = (reg.get("localidades") or {}).get("punta_alta") or {}
        print(f"{t:%d/%m %H:%M} UTC | fallas {len(reg['fallas'])} | Punta Alta: {pa.get('resumen_ahora')} | "
              f"llega en {pa.get('minutos_hasta_llegada')} | lluvia {pa.get('lluvia', {}).get('categoria')} "
              f"| granizo {pa.get('granizo', {}).get('categoria')}", flush=True)
    for p in guardadas.iterdir():
        shutil.move(str(p), hist / p.name)
    shutil.rmtree(guardadas)
    if a.salida:
        Path(a.salida).write_text(json.dumps(registros, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
