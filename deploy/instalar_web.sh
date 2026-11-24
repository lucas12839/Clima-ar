#!/usr/bin/env bash
# Publica la página de ClimaAR desde el VPS con https (24/11/2026).
# Uso (por SSH):  sudo bash /opt/Clima-ar/deploy/instalar_web.sh [dirección]
#   Sin dirección usa la gratuita basada en la IP: <ip-con-guiones>.sslip.io
#   Con un nombre propio (ej. climaarbahiablanca.duckdns.org) usa ese.
# Se puede volver a correr para cambiar la dirección.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "Ejecutar con sudo."; exit 1; }
DEST=/opt/Clima-ar

systemctl is-active --quiet climaar-api || { echo "Primero hay que instalar el servidor: deploy/instalar_api.sh"; exit 1; }

DOMINIO="${1:-}"
if [ -z "$DOMINIO" ]; then
  IP="$(curl -fsS -m 10 https://api.ipify.org)"
  DOMINIO="$(printf '%s' "$IP" | tr '.' '-').sslip.io"
fi
echo "Dirección: https://$DOMINIO"

if ! command -v caddy >/dev/null 2>&1; then
  echo "Instalando Caddy (una sola vez)..."
  apt-get update -qq
  if ! apt-get install -y -qq caddy >/dev/null 2>&1; then
    # Repositorio oficial de Caddy si el sistema no lo trae.
    apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https curl gnupg >/dev/null
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' > /etc/apt/sources.list.d/caddy-stable.list
    apt-get update -qq
    apt-get install -y -qq caddy >/dev/null
  fi
fi

# Puertos de la web (solo si el cortafuegos está activo).
if command -v ufw >/dev/null 2>&1 && ufw status | grep -q "Status: active"; then
  ufw allow 80/tcp >/dev/null; ufw allow 443/tcp >/dev/null
  echo "Puertos 80 y 443 abiertos en el cortafuegos."
fi

sed "s/^DOMINIO {/$DOMINIO {/" "$DEST/deploy/Caddyfile" > /etc/caddy/Caddyfile
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null
systemctl enable caddy >/dev/null 2>&1
systemctl restart caddy

echo "Esperando el certificado de seguridad (hasta 1 minuto)..."
for _ in $(seq 1 12); do
  if curl -fsS -m 10 "https://$DOMINIO/api/health" >/dev/null 2>&1; then
    echo
    echo "Listo. La app está en:  https://$DOMINIO"
    echo "Memoria de Caddy: $(systemctl show caddy -p MemoryCurrent --value | awk '{print int($1/1048576)}') MB"
    exit 0
  fi
  sleep 5
done
echo "Todavía no responde con https. Últimas líneas de Caddy:"
journalctl -u caddy -n 15 --no-pager -o cat || true
exit 1
