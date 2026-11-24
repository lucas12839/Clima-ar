#!/usr/bin/env bash
# Publica la página de ClimaAR desde el VPS con https (24/11/2026).
# Uso (por SSH):  sudo bash /opt/Clima-ar/deploy/instalar_web.sh [dirección]
#   Sin dirección usa la gratuita basada en la IP: <ip-con-guiones>.sslip.io
#   Con un nombre propio (ej. climaar-bahiablanca.duckdns.org) usa ese y
#   la dirección sslip.io sigue andando (los enlaces viejos no se rompen).
# Se puede volver a correr para cambiar la dirección.
# Nombre gratis en DuckDNS: entrar a https://www.duckdns.org, crear el
# subdominio "climaar-bahiablanca" y poner la IP del VPS. La IP es fija, así
# que no hace falta ningún programa que la actualice.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "Ejecutar con sudo."; exit 1; }
DEST=/opt/Clima-ar

systemctl is-active --quiet climaar-api || { echo "Primero hay que instalar el servidor: deploy/instalar_api.sh"; exit 1; }

DOMINIO="$(printf '%s' "${1:-}" | tr 'A-Z' 'a-z' | sed 's#^https\?://##; s#/.*$##')"
IP="$(curl -fsS -m 10 https://api.ipify.org)"
SSLIP="$(printf '%s' "$IP" | tr '.' '-').sslip.io"
if [ -z "$DOMINIO" ]; then
  DOMINIO="$SSLIP"
  DIRECCIONES="$SSLIP"
else
  # El nombre tiene que apuntar a este VPS antes de pedir el certificado.
  APUNTA="$(getent ahostsv4 "$DOMINIO" | awk 'NR==1{print $1}')"
  if [ "$APUNTA" != "$IP" ]; then
    echo "El nombre $DOMINIO apunta a '${APUNTA:-nada}' y el VPS es $IP."
    echo "Revisar en DuckDNS que el subdominio tenga la IP $IP y volver a correr."
    exit 1
  fi
  DIRECCIONES="$DOMINIO, $SSLIP"
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

# Puertos de la web (solo si el cortafuegos está activo). El 8000 (el que
# usaba Netlify) se cierra: ahora se entra solo por Caddy.
if command -v ufw >/dev/null 2>&1 && ufw status | grep -q "Status: active"; then
  ufw allow 80/tcp >/dev/null; ufw allow 443/tcp >/dev/null
  ufw delete allow 8000/tcp >/dev/null 2>&1 || true
  echo "Puertos 80 y 443 abiertos en el cortafuegos (8000 cerrado)."
fi

# Servidor de la app escuchando solo dentro del VPS.
if ! cmp -s "$DEST/deploy/climaar-api.service" /etc/systemd/system/climaar-api.service; then
  install -m 644 "$DEST/deploy/climaar-api.service" /etc/systemd/system/climaar-api.service
  systemctl daemon-reload
  systemctl restart climaar-api
  sleep 10
fi

sed "s/^DOMINIO {/$DIRECCIONES {/" "$DEST/deploy/Caddyfile" > /etc/caddy/Caddyfile
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
