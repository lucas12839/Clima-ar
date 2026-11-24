#!/usr/bin/env bash
# Instala el recolector de radares de otras zonas (23/11/2026).
# Uso (una sola vez, por SSH):   sudo bash /opt/Clima-ar/deploy/instalar_recoleccion.sh
# Apagar:                        sudo systemctl disable --now climaar-recoleccion.timer
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "Ejecutar con sudo."; exit 1; }
DEST=/opt/Clima-ar
install -m 644 "$DEST/deploy/climaar-recoleccion.service" /etc/systemd/system/climaar-recoleccion.service
install -m 644 "$DEST/deploy/climaar-recoleccion.timer" /etc/systemd/system/climaar-recoleccion.timer
systemctl daemon-reload
systemctl enable --now climaar-recoleccion.timer
# Primera vuelta ahora mismo (baja las últimas 3 horas).
systemctl start climaar-recoleccion.service || true
echo
journalctl -u climaar-recoleccion -n 3 --no-pager -o cat || true
echo
echo "Detector de Bahía: $(systemctl is-active climaar-detector)"
echo "Listo. Ver estado:  cat /var/lib/climaar-recoleccion/estado.json"
