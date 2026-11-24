#!/usr/bin/env bash
# Instala el servidor de la app en el VPS (reemplaza a Render, 24/11/2026).
# Uso (una sola vez, por SSH):  sudo bash /opt/Clima-ar/deploy/instalar_api.sh
# Apagar:                       sudo systemctl disable --now climaar-api
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "Ejecutar con sudo."; exit 1; }
DEST=/opt/Clima-ar
PY="$DEST/.venv-ia/bin/python"
[ -x "$PY" ] || { echo "No existe $PY"; exit 1; }

libre_mb="$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)"
echo "Memoria libre: ${libre_mb} MB"
if [ "$libre_mb" -lt 350 ]; then
  echo "Muy poca memoria libre (hace falta ~350 MB). No se instala nada."
  exit 1
fi

echo "Instalando los programas del servidor (una sola vez)..."
runuser -u climaar -- "$PY" -m pip install --quiet -r "$DEST/requirements.txt"

# Los registros del sistema nunca pueden pasar de 200 MB.
mkdir -p /etc/systemd/journald.conf.d
printf '[Journal]\nSystemMaxUse=200M\n' > /etc/systemd/journald.conf.d/climaar.conf
systemctl restart systemd-journald

# Puerto 8000 abierto (solo si el cortafuegos está activo).
if command -v ufw >/dev/null 2>&1 && ufw status | grep -q "Status: active"; then
  ufw allow 8000/tcp >/dev/null && echo "Puerto 8000 abierto en el cortafuegos."
fi

install -m 644 "$DEST/deploy/climaar-api.service" /etc/systemd/system/climaar-api.service
systemctl daemon-reload
systemctl enable --now climaar-api
sleep 15

if curl -fsS -m 30 http://127.0.0.1:8000/resumen >/dev/null; then
  echo "Servidor funcionando."
else
  echo "El servidor no respondió. Últimas líneas:"
  journalctl -u climaar-api -n 20 --no-pager -o cat || true
  exit 1
fi
echo "Memoria que usa: $(systemctl show climaar-api -p MemoryCurrent --value | awk '{print int($1/1048576)}') MB"
echo "Detector: $(systemctl is-active climaar-detector)"
echo "IP pública: $(curl -fsS -m 10 https://api.ipify.org || echo 'no se pudo ver')"
