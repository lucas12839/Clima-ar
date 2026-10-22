#!/usr/bin/env python3
"""
Guardas de ejecucion unica. Ya no hay ciclo Radar: lo usa
climaar-ciclo-radar.yml solo para no lanzar dos detectores.

Garantiza un unico ciclo vivo. El workflow pasa por stdin el JSON de
`gh run list --json databaseId,status` y este modulo decide con reglas
puras (facilmente testeables):

  inicio      puede esta ejecucion ser el ciclo? (no si hay otro ciclo
              activo mas antiguo; el mas antiguo gana siempre)
  radar       lanzar Radar? (no si ya hay uno en cola o ejecutandose)
  encadenar   lanzar el proximo ciclo? (no si ya hay otro ciclo activo)

Imprime "true" o "false".
"""

import argparse
import json
import sys

ACTIVE = {"queued", "in_progress", "waiting", "requested", "pending"}


def _runs(raw):
    return [
        {"id": int(r["databaseId"]), "status": r.get("status")}
        for r in raw
        if isinstance(r, dict) and r.get("databaseId") is not None
    ]


def otros_activos(runs, mi_id=None):
    return [
        r for r in _runs(runs)
        if r["status"] in ACTIVE and r["id"] != mi_id
    ]


def debe_continuar_ciclo(runs, mi_id):
    """False si existe otro ciclo activo mas antiguo (duplicado)."""
    return not any(r["id"] < mi_id for r in otros_activos(runs, mi_id))


def debe_lanzar_radar(runs):
    return not otros_activos(runs)


def debe_encadenar(runs, mi_id):
    return not otros_activos(runs, mi_id)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "modo", choices=("inicio", "radar", "encadenar")
    )
    parser.add_argument("--run-id", type=int, default=None)
    args = parser.parse_args(argv)

    runs = json.load(sys.stdin)

    if args.modo == "inicio":
        result = debe_continuar_ciclo(runs, args.run_id)
    elif args.modo == "radar":
        result = debe_lanzar_radar(runs)
    else:
        result = debe_encadenar(runs, args.run_id)

    print("true" if result else "false")


if __name__ == "__main__":
    main()
