#!/usr/bin/env python3
"""Reproduce un temporal pasado por la cadena COMPLETA como si ocurriera ahora.

Para cada imagen del evento (en orden):
1. La convierte al PNG de paleta que publica SINARAME (mismos índices).
2. La deja como frame recién llegado, ATRASO_MIN después de la hora del radar.
3. Ejecuta los pasos reales de los workflows (radar e Intelligence, con sus
   validaciones) con el reloj puesto en ese momento.
4. Registra lo que ClimaAR mostraría: tipo de tiempo, amenazas, llegada.

Uso (sobre una COPIA del repo, nunca sobre el repo real):
  python herramientas/simular_evento_real.py <copia> <eventos.pkl> <nombre_evento> --salida x.json
El .pkl tiene {nombre: [(fecha_utc, matriz_dbz 512x512), ...]} (ver
decodificar_imagen_smn.py). Limitación: el ambiente ECMWF es el de hoy (no
hay pronóstico histórico sin conexión); no hay estaciones ni rayos GOES del día.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from decodificador_rma10_dbz import DBZ_MAX, DBZ_MIN, PALETTE_INDEX_MAX, PALETTE_INDEX_MIN  # noqa: E402

ATRASO_MIN = 7
EMU = ROOT / "herramientas" / "emular_workflow.py"
RELOJ = ROOT / "herramientas" / "reloj_simulado"


def plantilla(copia: Path) -> Image.Image:
    for p in sorted((copia / "data/radar/historico").glob("20*.png"), reverse=True):
        with Image.open(p) as im:
            if im.mode == "P" and isinstance(im.info.get("transparency"), int):
                im.load()
                return im.copy()
    raise SystemExit("No hay un PNG RMA10 de plantilla en el histórico.")


def a_png(dbz: np.ndarray, base: Image.Image) -> Image.Image:
    transp = base.info["transparency"]
    idx = PALETTE_INDEX_MIN + (np.nan_to_num(dbz, nan=-999) - DBZ_MIN) / (DBZ_MAX - DBZ_MIN) * (
        PALETTE_INDEX_MAX - PALETTE_INDEX_MIN)
    idx = np.clip(np.rint(idx), PALETTE_INDEX_MIN, PALETTE_INDEX_MAX).astype(np.uint8)
    idx[~np.isfinite(dbz)] = transp
    im = Image.fromarray(idx, mode="P")
    im.putpalette(base.getpalette())
    im.info["transparency"] = transp
    return im


def correr(copia: Path, wf: str, desde: str, hasta: str, env: dict) -> tuple[int, int, list[str]]:
    r = subprocess.run([sys.executable, str(EMU), str(copia), f".github/workflows/{wf}", sys.executable,
                        "--desde", desde, "--hasta", hasta], cwd=copia, env=env, capture_output=True, text=True)
    lineas = (r.stdout + r.stderr).splitlines()
    ok = sum(l.startswith("OK") for l in lineas)
    fallas = [l for l in lineas if l.startswith("FALLA")]
    return ok, ok + len(fallas), fallas


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    salida = argv[argv.index("--salida") + 1] if "--salida" in argv else None
    copia, pkl, nombre = Path(argv[0]).resolve(), argv[1], argv[2]
    frames = pickle.load(open(pkl, "rb"))[nombre]
    base = plantilla(copia)

    # Marcas fijas del mapa del SMN (textos, bordes): >=55 dBZ en >30 % de las imágenes.
    fijo = np.mean([np.nan_to_num(g, nan=-99) >= 55 for _, g in frames], axis=0) > 0.3

    # Estado limpio: el evento arranca de cero, como un día sin tormentas previas.
    radar = copia / "data/radar"
    for p in (radar / "historico").glob("*.png"):
        p.unlink()
    for rel in ("data/radar/seguimiento/estado.json", ".climaar/estado_rma10.json", "data/radar/historial_ciudad.json",
                "data/ia/estaciones/historial_superficie.json", "data/ia/estaciones/observaciones/bcp_actual.json",
                "data/ia/estaciones/observaciones/smn_synop_actual.json", "data/ia/goes19_intelligence.json", "data/radar/.flujo_campo.npz", "data/ia/resultado_final.json"):
        (copia / rel).unlink(missing_ok=True)
    amb_path = copia / "data/meteorologia/ambiente_futuro.json"
    ambiente = json.loads(amb_path.read_text(encoding="utf-8"))
    original = json.dumps(ambiente)
    ref_original = datetime.fromisoformat(str(ambiente["actualizado_utc"]).replace("Z", "+00:00"))

    registros = []
    for i, (t, grid) in enumerate(frames):
        g = grid.copy()
        g[fijo] = np.nan
        nombre_png = t.strftime("%Y%m%d%H%M%S")
        im = a_png(g, base)
        im.save(radar / "historico" / f"{nombre_png}.png", transparency=base.info["transparency"])
        im.save(radar / "actual.png", transparency=base.info["transparency"])
        (radar / "radar_timestamp.txt").write_text(t.isoformat(), encoding="utf-8")
        ahora = t + timedelta(minutes=ATRASO_MIN)
        # El pronóstico ECMWF guardado (el de hoy) se corre a la fecha del evento.
        ambiente = json.loads(original)
        corrimiento = (ahora - timedelta(minutes=5)) - ref_original
        for punto in ambiente.get("puntos", []):
            for reg_h in punto.get("datos", []):
                h = datetime.fromisoformat(str(reg_h["tiempo_utc"]).replace("Z", "+00:00"))
                if h.tzinfo is None:
                    h = h.replace(tzinfo=timezone.utc)
                reg_h["tiempo_utc"] = (h + corrimiento).strftime("%Y-%m-%dT%H:%M")
        ambiente["actualizado_utc"] = (ahora - timedelta(minutes=5)).isoformat()
        amb_path.write_text(json.dumps(ambiente), encoding="utf-8")

        env = dict(os.environ, CLIMAAR_RELOJ_SIMULADO=ahora.isoformat(), CLIMAAR_AHORA_UTC=ahora.isoformat(),
                   PYTHONPATH=os.pathsep.join([str(RELOJ), os.environ.get("PYTHONPATH", "")]))
        r_ok, r_tot, r_f = correr(copia, "probar-radar-smn.yml", "Procesar frames intermedios", "Validar cadena", env)
        i_ok, i_tot, i_f = correr(copia, "climaar-intelligence.yml", "Señales de superficie", "Validar integridad", env)

        reg = {"radar_utc": t.isoformat(), "ahora_utc": ahora.isoformat(), "radar_ok": f"{r_ok}/{r_tot}",
               "intelligence_ok": f"{i_ok}/{i_tot}", "fallas": r_f + i_f}
        try:
            fin = json.loads((copia / "data/ia/resultado_final.json").read_text(encoding="utf-8"))
            if i_f:  # si Intelligence falló, el resultado es del ciclo anterior
                raise RuntimeError("Intelligence falló: resultado no actualizado")
            tt = fin.get("tipo_tiempo") or {}
            reg.update({k: tt.get(k) for k in ("nivel", "etiqueta", "alcance", "amenazas", "indicios_en_evaluacion",
                                                "eta_min", "distancia_km")})
            reg["detalle_tipo"] = {k: v for k, v in tt.items() if k not in reg}
            reg["estado_motor"] = (fin.get("motor_decision") or {}).get("estado")
        except Exception as exc:
            reg["error_resultado"] = str(exc)
        registros.append(reg)
        print(f"[{i + 1}/{len(frames)}] {t:%d/%m %H:%M} radar {reg['radar_ok']} intel {reg['intelligence_ok']} "
              f"| {reg.get('etiqueta')} | {reg.get('alcance')} | amenazas {reg.get('amenazas')} "
              f"| indicios {reg.get('indicios_en_evaluacion')}", flush=True)

    if salida:
        Path(salida).write_text(json.dumps(registros, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
