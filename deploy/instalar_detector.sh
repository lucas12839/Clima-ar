#!/usr/bin/env bash
# Instala el detector SINARAME como servicio systemd (Debian/Ubuntu/Raspberry Pi OS).
# Uso:  sudo bash deploy/instalar_detector.sh
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "Ejecutar con sudo."; exit 1; }

REPO_URL="${REPO_URL:-https://github.com/lucas12839/Clima-ar.git}"
DEST=/opt/Clima-ar
ENV_FILE=/etc/climaar-detector.env

apt-get update
apt-get install -y git python3 python3-venv python3-pip

id climaar >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin climaar

if [ ! -d "$DEST/.git" ]; then
  git clone "$REPO_URL" "$DEST"
fi
chown -R climaar:climaar "$DEST"

sudo -u climaar python3 -m venv "$DEST/.venv"
sudo -u climaar "$DEST/.venv/bin/pip" install --upgrade pip
sudo -u climaar "$DEST/.venv/bin/pip" install playwright pillow
"$DEST/.venv/bin/python" -m playwright install-deps chromium
sudo -u climaar env PLAYWRIGHT_BROWSERS_PATH=/home/climaar/.cache/ms-playwright \
  "$DEST/.venv/bin/python" -m playwright install chromium

if [ ! -f "$ENV_FILE" ]; then
  install -m 600 -o root -g root "$DEST/deploy/climaar-detector.env.example" "$ENV_FILE"
  echo ">>> Edite $ENV_FILE y coloque el GH_TOKEN real antes de iniciar."
fi

install -m 644 "$DEST/deploy/climaar-detector.service" /etc/systemd/system/climaar-detector.service
systemctl daemon-reload
systemctl enable climaar-detector

echo "Instalado. Tras configurar el token:  sudo systemctl restart climaar-detector"
echo "Verificar:  systemctl status climaar-detector  |  journalctl -u climaar-detector -f"
