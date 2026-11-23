#!/usr/bin/env bash
# Instala la actualización automática del VPS desde main (etapa C, 23/11/2026).
# Uso (una sola vez, por SSH):
#   cd /opt/Clima-ar && sudo git pull && sudo bash deploy/instalar_actualizacion.sh
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "Ejecutar con sudo."; exit 1; }
DEST=/opt/Clima-ar
install -m 644 "$DEST/deploy/climaar-actualizar.service" /etc/systemd/system/climaar-actualizar.service
install -m 644 "$DEST/deploy/climaar-actualizar.timer" /etc/systemd/system/climaar-actualizar.timer
systemctl daemon-reload
systemctl enable --now climaar-actualizar.timer
# Primera actualización ahora mismo (deja el VPS igual a main y reinicia el detector si hace falta).
systemctl start climaar-actualizar.service || true
echo
echo "Resultado:"
cat "$DEST/.climaar/actualizacion.json" 2>/dev/null || echo "(todavía sin resultado)"
echo
echo "Listo. Se actualiza solo cada 10 min. Ver:  systemctl list-timers climaar-actualizar"
echo "Historial:  journalctl -u climaar-actualizar -n 30"
