#!/usr/bin/env bash
# Actualiza solo el VPS desde main (etapa C, 23/11/2026). Lo corre cada 10 min
# el reloj climaar-actualizar.timer (instalar con deploy/instalar_actualizacion.sh).
#
# - Baja main del repositorio público: no necesita ninguna clave.
# - Si el VPS tenía cambios propios (lo que antes hacía que nunca quedara igual
#   a main), los guarda en una rama local "respaldo-vps-FECHA" y deja el VPS
#   exactamente igual a main.
# - Reinicia el detector SOLO si cambió su propio código (detector_sinarame.py
#   o radar_estado.py). El capturador de SINARAME se vuelve a abrir en cada
#   consulta, así que sus cambios entran sin reiniciar. Reiniciar no pierde
#   imágenes: al volver, la primera consulta ve la imagen más nueva de SINARAME
#   y la despacha si todavía no se procesó (cada imagen se procesa una sola vez).
# - Si cambiaron las versiones de numpy, Pillow o Playwright, las instala.
# - Deja el resultado en .climaar/actualizacion.json; el detector lo publica en
#   GitHub y la vigilancia avisa si falla.
set -uo pipefail

DEST="${CLIMAAR_DIR:-/opt/Clima-ar}"
URL="${CLIMAAR_REPO_URL:-https://github.com/lucas12839/Clima-ar.git}"
# Usuario dueño de la copia del repositorio (normalmente climaar).
USUARIO="${CLIMAAR_USUARIO:-$(stat -c %U "$DEST/.git" 2>/dev/null || echo climaar)}"
SERVICIO="${CLIMAAR_SERVICIO:-climaar-detector}"
ESTADO="$DEST/.climaar/actualizacion.json"
AHORA="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

como() {  # corre un comando como el usuario del detector (si hace falta)
  if [ "$(id -un)" = "$USUARIO" ]; then "$@"; else runuser -u "$USUARIO" -- "$@"; fi
}

resultado() {  # resultado detalle commit
  mkdir -p "$DEST/.climaar"
  printf '{"utc":"%s","resultado":"%s","detalle":"%s","commit":"%s"}\n' \
    "$AHORA" "$1" "$(printf '%s' "$2" | tr '"\\\n' "'/ " | cut -c1-200)" "$3" > "$ESTADO.tmp"
  mv "$ESTADO.tmp" "$ESTADO"
  chown "$USUARIO" "$ESTADO" 2>/dev/null || true
  echo "actualizacion: $1 - $2"
}

cd "$DEST" || { echo "No existe $DEST"; exit 1; }
antes="$(como git rev-parse HEAD 2>/dev/null || echo ninguno)"

# Último commit de main que dejó este mismo programa (para no necesitar la
# historia completa: main suma ~600 guardados de datos por día).
sincronizado="$(sed -n 's/.*"commit":"\([0-9a-f]*\)".*/\1/p' "$ESTADO" 2>/dev/null | head -1)"

# Descarga corta (23/11/2026): antes bajaba toda la historia (cientos de MB)
# cada 10 min. Solo hace falta la historia si no se sabe qué quedó la última vez.
profundidad="--depth=1"
if [ -z "$sincronizado" ] || [ "${antes:0:12}" != "$sincronizado" ]; then
  profundidad=""
fi
# shellcheck disable=SC2086
if ! salida="$(como git fetch --quiet $profundidad "$URL" main 2>&1)"; then
  resultado error "no se pudo bajar main: $salida" "$antes"
  exit 1
fi
nuevo="$(como git rev-parse FETCH_HEAD)"

# Cambios propios del VPS (archivos del repositorio modificados o commits que main no tiene).
propios=""
if [ -n "$(como git status --porcelain --untracked-files=no)" ]; then
  propios="archivos modificados"
fi
if [ "$antes" != "ninguno" ] && [ "${antes:0:12}" != "$sincronizado" ] \
    && ! como git merge-base --is-ancestor "$antes" "$nuevo"; then
  propios="${propios:+$propios y }commits que main no tiene"
fi
if [ -n "$propios" ]; then
  rama="respaldo-vps-$(date -u +%Y%m%d-%H%M)"
  como git stash push --quiet -m "$rama" >/dev/null 2>&1 || true
  como git branch -f "$rama" "$antes" >/dev/null 2>&1 || true
  echo "El VPS tenía $propios: guardados en la rama local $rama (y en git stash)."
fi

if [ "$antes" = "$nuevo" ] && [ -z "$propios" ]; then
  resultado ok "sin cambios" "${nuevo:0:12}"
  exit 0
fi

cambiados="$(como git diff --name-only "$antes" "$nuevo" 2>/dev/null || echo TODO)"
if ! salida="$(como git reset --quiet --hard "$nuevo" 2>&1)"; then
  resultado error "no se pudo dejar igual a main: $salida" "$antes"
  exit 1
fi

# Python del servicio (el VPS usa .venv-ia; la instalación nueva, .venv).
PY="$DEST/.venv-ia/bin/python"
[ -x "$PY" ] || PY="$DEST/.venv/bin/python"
detalle="actualizado ${antes:0:12} -> ${nuevo:0:12}"

if printf '%s\n' "$cambiados" | grep -qx -e requirements.txt -e TODO; then
  paquetes="$(grep -iE '^(numpy|pillow|playwright)==' requirements.txt | tr '\n' ' ')"
  if [ -n "$paquetes" ] && [ -x "$PY" ]; then
    # shellcheck disable=SC2086
    if ! salida="$(como "$PY" -m pip install --quiet $paquetes 2>&1)"; then
      resultado error "no se pudieron instalar $paquetes: $salida" "${nuevo:0:12}"
      exit 1
    fi
    detalle="$detalle; paquetes: $paquetes"
    # 23/11/2026: Playwright nuevo sin su navegador dejó al detector 27 min sin
    # poder consultar SINARAME. Se baja el navegador de esa versión (si ya está, no
    # hace nada) en la carpeta del usuario del detector, que es donde lo busca.
    if printf '%s\n' "$paquetes" | grep -qi playwright; then
      CASA="$(getent passwd "$USUARIO" | cut -d: -f6)"
      if ! salida="$(como env HOME="${CASA:-$HOME}" "$PY" -m playwright install chromium 2>&1)"; then
        resultado error "no se pudo bajar el navegador de Playwright: $salida" "${nuevo:0:12}"
        exit 1
      fi
      detalle="$detalle; navegador al día"
    fi
  fi
fi

if printf '%s\n' "$cambiados" | grep -qx -e scripts/detector_sinarame.py -e scripts/radar_estado.py \
    -e requirements.txt -e TODO; then
  if systemctl restart "$SERVICIO"; then
    sleep 20
    if systemctl is-active --quiet "$SERVICIO"; then
      detalle="$detalle; detector reiniciado"
    else
      resultado error "el detector no volvió a arrancar después de actualizar" "${nuevo:0:12}"
      exit 1
    fi
  else
    resultado error "no se pudo reiniciar el detector" "${nuevo:0:12}"
    exit 1
  fi
fi

# Servidor de la app (si está instalado): se reinicia si cambió su código o
# sus programas. Tarda unos segundos.
if systemctl list-unit-files climaar-api.service >/dev/null 2>&1 \
    && systemctl is-enabled --quiet climaar-api 2>/dev/null; then
  if printf '%s\n' "$cambiados" | grep -qx -e requirements.txt -e TODO; then
    if ! salida="$(como "$PY" -m pip install --quiet -r requirements.txt 2>&1)"; then
      resultado error "no se pudieron instalar los programas del servidor: $salida" "${nuevo:0:12}"
      exit 1
    fi
  fi
  # El servicio del servidor es solo nuestro (no tiene ajustes del VPS): si
  # cambió, se instala. Solo con Caddy andando (Caddy es la puerta de entrada).
  api_servicio=no
  if printf '%s\n' "$cambiados" | grep -qx -e deploy/climaar-api.service -e TODO \
      && systemctl is-active --quiet caddy 2>/dev/null \
      && ! cmp -s "$DEST/deploy/climaar-api.service" /etc/systemd/system/climaar-api.service; then
    install -m 644 "$DEST/deploy/climaar-api.service" /etc/systemd/system/climaar-api.service
    systemctl daemon-reload
    api_servicio=si
  fi
  if [ "$api_servicio" = si ] || printf '%s\n' "$cambiados" | grep -q -e '\.py$' -e '^requirements.txt$' -e '^config/' -e '^TODO$'; then
    if systemctl restart climaar-api; then
      detalle="$detalle; servidor de la app reiniciado"
    else
      resultado error "no se pudo reiniciar el servidor de la app" "${nuevo:0:12}"
      exit 1
    fi
  fi
fi

if printf '%s\n' "$cambiados" | grep -qx deploy/climaar-detector.service; then
  # El servicio instalado puede tener ajustes propios (.venv-ia): no se pisa solo.
  detalle="$detalle; AVISO: cambió deploy/climaar-detector.service, revisarlo a mano"
fi
resultado ok "$detalle" "${nuevo:0:12}"
