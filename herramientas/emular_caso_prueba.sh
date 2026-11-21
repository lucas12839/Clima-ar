#!/bin/bash
# Prueba de la cadena completa en una COPIA del repo (no toca el repo real).
# Uso: herramientas/emular_caso_prueba.sh <copia> <imagen_fuente|vacia> <historial_estaciones.json|->
# La imagen entra como frame nuevo de hace 7 min; el historial de estaciones (si se da)
# se corre en el tiempo para que su último dato sea de hace 3 min.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
COPIA=$1; FUENTE=$2; HIST=$3
rm -rf "$COPIA" && mkdir -p "$COPIA" && (cd "$REPO" && tar --exclude=.git -cf - . | tar -xf - -C "$COPIA")
cd "$COPIA"
python3 - "$FUENTE" "$HIST" <<'PY'
import json, sys
from datetime import datetime, timedelta, timezone
import numpy as np
from PIL import Image
ahora = datetime.now(timezone.utc)
t = (ahora - timedelta(minutes=7)).replace(second=0, microsecond=0)
t = t.replace(minute=t.minute - t.minute % 10)
base = Image.open('data/radar/actual.png')
if sys.argv[1] == 'vacia':
    im = Image.fromarray(np.full((512, 512), 253, dtype=np.uint8), 'P'); im.putpalette(base.getpalette())
else:
    im = Image.open(sys.argv[1])
for p in ('data/radar/actual.png', f'data/radar/historico/{t:%Y%m%d%H%M%S}.png'):
    im.save(p)
open('data/radar/radar_timestamp.txt', 'w').write(t.isoformat())
print("frame nuevo:", t.isoformat())
if sys.argv[2] != '-':
    h = json.load(open(sys.argv[2]))
    ultimo = max(datetime.fromisoformat(r['utc'].replace('Z', '+00:00')) for v in h.values() for r in v)
    d = (ahora - timedelta(minutes=3)) - ultimo
    out = {k: [dict(r, utc=(datetime.fromisoformat(r['utc'].replace('Z', '+00:00')) + d).isoformat()) for r in v]
           for k, v in h.items() if v}
    json.dump(out, open('data/ia/estaciones/historial_superficie.json', 'w'))
    print("historial corrido", d)
PY
PY=$(which python)
python herramientas/emular_workflow.py "$COPIA" .github/workflows/probar-radar-smn.yml "$PY" --desde "Procesar radar" --hasta "Validar cadena Radar"; r1=$?
python herramientas/emular_workflow.py "$COPIA" .github/workflows/climaar-intelligence.yml "$PY" --desde "Conectar RMA10" --hasta "Validar integridad"; r2=$?
echo "== resultado radar=$r1 intelligence=$r2"
python - <<'PY'
import json
for f in ("data/ia/celdas_localidad.json", "data/ia/estaciones/alarma_llegada.json", "data/ia/salud_cadena.json"):
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f, "NO", e); continue
    if "celdas" in f:
        print(f, d.get("version"), d.get("disponible"), d.get("motivo"),
              {k: v.get("fuerte_o_rayos") for k, v in (d.get("localidades") or {}).items()})
    elif "alarma" in f:
        print(f, d.get("disponible"), d.get("motivo"), d.get("estaciones_frescas"),
              [(e["nombre"], e["senales"], e["tormenta_cerca"]) for e in d.get("estaciones_con_firma", [])],
              {k: (v["estado"], v["minutos"]) for k, v in (d.get("localidades") or {}).items() if v.get("estado")})
    else:
        print(f, json.dumps(d.get("alertas"), ensure_ascii=False)[:600])
PY
exit $(( r1 + r2 ))
