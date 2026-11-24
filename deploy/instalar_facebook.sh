#!/usr/bin/env bash
# Instala las publicaciones automáticas en Facebook (24/11/2026).
# Uso (por SSH):  sudo bash /opt/Clima-ar/deploy/instalar_facebook.sh
#   Pide el número de la página y la clave de la página (no se muestra al
#   pegarla). Se guardan solo en /etc/climaar-facebook.env (permiso 600).
#   Primero arma una publicación de prueba SIN publicarla; después prende el
#   reloj. Volver a correrlo sirve para cambiar la clave.
# Apagar:         sudo systemctl disable --now climaar-facebook.timer
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "Ejecutar con sudo."; exit 1; }
DEST=/opt/Clima-ar
ENV=/etc/climaar-facebook.env
PY="$DEST/.venv-ia/bin/python"

read -r -p "Número de la página de Facebook (FB_PAGE_ID): " PAGINA
read -r -s -p "Clave de la página (FB_PAGE_TOKEN, no se ve al pegar): " CLAVE; echo
[ -n "$PAGINA" ] && [ -n "$CLAVE" ] || { echo "Faltan datos."; exit 1; }

umask 077
cat > "$ENV" <<FIN
FB_PAGE_ID=$PAGINA
FB_PAGE_TOKEN=$CLAVE
FB_ACTIVO=1
FIN
chmod 600 "$ENV"; chown root:root "$ENV"

# La clave tiene que poder publicar en esa página (pregunta a Facebook, no publica nada).
if ! "$PY" - "$PAGINA" "$CLAVE" <<'FIN'
import sys, requests
pagina, clave = sys.argv[1], sys.argv[2]
r = requests.get(f"https://graph.facebook.com/v23.0/{pagina}", params={"fields": "name", "access_token": clave}, timeout=20)
d = r.json()
if r.status_code != 200 or "name" not in d:
    print("Facebook no acepta la clave:", (d.get("error") or {}).get("message", r.text[:200]))
    sys.exit(1)
print("Página:", d["name"])
FIN
then
  echo "Revisá el número de página y la clave, y volvé a correr el instalador."
  exit 1
fi

install -m 644 "$DEST/deploy/climaar-facebook.service" /etc/systemd/system/climaar-facebook.service
install -m 644 "$DEST/deploy/climaar-facebook.timer" /etc/systemd/system/climaar-facebook.timer
systemctl daemon-reload

# Prueba sin publicar: arma la de la mañana y la de tormenta en /tmp/climaar-facebook-prueba.
sudo -u climaar env CLIMAAR_API=http://127.0.0.1:8000 "$PY" "$DEST/scripts/facebook_climaar.py" \
  --probar --forzar manana --salida /tmp/climaar-facebook-prueba || true
echo
echo "----- Texto de la mañana (prueba, NO publicado) -----"
cat /tmp/climaar-facebook-prueba/manana.txt 2>/dev/null || echo "(no se pudo armar)"
echo "-----------------------------------------------------"

systemctl enable --now climaar-facebook.timer
echo
echo "Listo. Publica la de la mañana desde las 7:30 y las de tormentas cuando corresponda."
echo "Publicar la de la mañana ahora (para probar):  sudo systemctl start climaar-facebook  (solo si ya son más de las 7:30)"
echo "Ver qué hizo:  journalctl -u climaar-facebook -n 20 --no-pager"
echo "Apagar:        sudo systemctl disable --now climaar-facebook.timer"
