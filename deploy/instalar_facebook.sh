#!/usr/bin/env bash
# Instala las publicaciones automáticas en Facebook a través de Make (24/11/2026).
# Uso (por SSH):  sudo bash /opt/Clima-ar/deploy/instalar_facebook.sh
#   Pide la dirección del webhook de Make (la da el escenario de Make). Se
#   guarda solo en /etc/climaar-facebook.env (permiso 600), nunca en GitHub.
#   Arma una publicación de prueba, pregunta si publicarla, y prende el reloj.
#   Volver a correrlo sirve para cambiar la dirección.
# Apagar:         sudo systemctl disable --now climaar-facebook.timer
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "Ejecutar con sudo."; exit 1; }
DEST=/opt/Clima-ar
ENV=/etc/climaar-facebook.env
PY="$DEST/.venv-ia/bin/python"

read -r -p "Dirección del webhook de Make (empieza con https://hook.): " WEBHOOK
case "$WEBHOOK" in
  https://hook.*make.com/*) ;;
  *) echo "Esa dirección no parece de Make (https://hook....make.com/...)."; exit 1 ;;
esac

umask 077
printf 'MAKE_WEBHOOK_URL=%s\nFB_ACTIVO=1\n' "$WEBHOOK" > "$ENV"
chmod 600 "$ENV"; chown root:root "$ENV"
umask 022

# Carpeta de las imágenes (la página las sirve en /publicaciones/ para que Make las baje).
install -d -o climaar -g climaar -m 755 /var/lib/climaar-facebook /var/lib/climaar-facebook/publico

# La página tiene que servir /publicaciones/: se rearma Caddy con la misma dirección.
NOMBRE="$(awk '/^[^#[:space:]].* \{$/ {sub(/ \{$/, ""); print; exit}' /etc/caddy/Caddyfile | tr ',' '\n' | tr -d ' ' | grep -v 'sslip.io' | head -n1 || true)"
bash "$DEST/deploy/instalar_web.sh" ${NOMBRE:+"$NOMBRE"} >/dev/null
echo "Página actualizada (sirve /publicaciones/)."

install -m 644 "$DEST/deploy/climaar-facebook.service" /etc/systemd/system/climaar-facebook.service
install -m 644 "$DEST/deploy/climaar-facebook.timer" /etc/systemd/system/climaar-facebook.timer
systemctl daemon-reload

echo
echo "Armando la publicación de la mañana de prueba (todavía NO se publica)..."
rm -rf /tmp/climaar-facebook-prueba
sudo -u climaar env CLIMAAR_API=http://127.0.0.1:8000 "$PY" "$DEST/scripts/facebook_climaar.py" \
  --probar --forzar manana --salida /tmp/climaar-facebook-prueba || true
echo "-----------------------------------------------------"
cat /tmp/climaar-facebook-prueba/manana.txt 2>/dev/null || echo "(no se pudo armar)"
echo
echo "-----------------------------------------------------"
read -r -p "¿Publicar esta en la página de Facebook ahora, como prueba? (s/n): " RESP
if [ "$RESP" = "s" ] || [ "$RESP" = "S" ]; then
  set -a; . "$ENV"; set +a
  if sudo -u climaar env MAKE_WEBHOOK_URL="$MAKE_WEBHOOK_URL" CLIMAAR_API=http://127.0.0.1:8000 \
      "$PY" "$DEST/scripts/facebook_climaar.py" --forzar manana; then
    echo "Enviada a Make. En 1 minuto tiene que aparecer en la página."
  else
    echo "No se pudo enviar. Revisar la dirección del webhook y que el escenario de Make esté prendido."
  fi
fi

systemctl enable --now climaar-facebook.timer >/dev/null
echo
echo "Listo. Publica la de la mañana desde las 7:30 y las advertencias de tormenta cuando corresponda."
echo "Ver qué hizo:  journalctl -u climaar-facebook -n 20 --no-pager"
echo "Apagar:        sudo systemctl disable --now climaar-facebook.timer"
