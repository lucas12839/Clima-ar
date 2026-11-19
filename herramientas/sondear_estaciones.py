import json, subprocess, sys
r = subprocess.run([sys.executable, "scripts/estaciones/colector_cwop.py", "--max-segundos", "630"])
print(json.dumps(json.load(open("data/ia/estaciones/observaciones/cwop_actual.json")), ensure_ascii=False, indent=1))
sys.exit(r.returncode)
