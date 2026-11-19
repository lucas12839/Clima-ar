#!/usr/bin/env python3
"""Ejecuta localmente los pasos 'run' de un workflow de GitHub Actions.

Uso: emular_workflow.py <repo> <workflow.yml> <python> [--desde NOMBRE] [--hasta NOMBRE]
Se saltean pasos con 'uses:' y los que dependen de GitHub/red (lista SALTEAR).
Respeta 'if:' simples del tipo env.X != '1' / env.X == '1'.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

SALTEAR = (
    "Checkout", "Configurar Python", "Preparar directorios", "Obtener artefacto",
    "Verificar radar activo", "Actualizar observación SAZB", "Dependencias",
    "Capturar red de estaciones", "Validar frescura y publicar estado",
    "Preparar artefacto", "Publicar artefacto", "Persistir",
)


def evaluar_if(cond, env):
    if not cond:
        return True
    cond = str(cond).strip()
    m = re.fullmatch(r"env\.(\w+)\s*(==|!=)\s*'([^']*)'", cond)
    if m:
        valor = env.get(m.group(1), "")
        return (valor == m.group(3)) if m.group(2) == "==" else (valor != m.group(3))
    if "inputs.modo" in cond:
        return "!= 'refresco'" in cond or "!=" in cond
    return True


def main():
    repo, wf, python = sys.argv[1], sys.argv[2], sys.argv[3]
    desde = sys.argv[sys.argv.index("--desde") + 1] if "--desde" in sys.argv else None
    hasta = sys.argv[sys.argv.index("--hasta") + 1] if "--hasta" in sys.argv else None
    data = yaml.safe_load(Path(wf).read_text(encoding="utf-8"))
    job = data["jobs"].get("intelligence") or list(data["jobs"].values())[-1]
    env = dict(os.environ)
    env["PATH"] = str(Path(python).parent) + os.pathsep + env["PATH"]
    genv = Path("/tmp/github_env_emulado")
    genv.write_text("")
    env["GITHUB_ENV"] = str(genv)
    env["GITHUB_OUTPUT"] = "/tmp/github_output_emulado"
    activo = desde is None
    fallos = 0
    for step in job["steps"]:
        nombre = step.get("name", "")
        if desde and nombre.startswith(desde):
            activo = True
        if not activo:
            continue
        if "run" not in step or any(nombre.startswith(s) for s in SALTEAR):
            print(f"-- saltado: {nombre}")
        else:
            for linea in genv.read_text().splitlines():
                if "=" in linea:
                    k, v = linea.split("=", 1)
                    env[k] = v
            if not evaluar_if(step.get("if"), env):
                print(f"-- omitido por if: {nombre}")
            else:
                r = subprocess.run(["bash", "-c", step["run"]], cwd=repo, env=env,
                                   capture_output=True, text=True)
                estado = "OK " if r.returncode == 0 else "FALLA"
                print(f"{estado} {nombre}")
                if r.returncode != 0:
                    fallos += 1
                    salida = (r.stdout + r.stderr).strip().splitlines()
                    print("     " + "\n     ".join(salida[-6:]))
                    break
        if hasta and nombre.startswith(hasta):
            break
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()
