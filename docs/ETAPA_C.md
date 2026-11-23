# Etapa C, punto 11: vigilancia (23/11/2026)

Objetivo: que no vuelva a pasar lo del 23/11 (6 h sin radar sin que nadie se entere).

## Lo que pasó el 23/11 y qué lo cubre ahora

| Qué pasó | Qué avisa ahora |
|---|---|
| La clave de GitHub del VPS venció (error 401) | 14 días antes: "la clave vence el ..." (GitHub informa la fecha en cada respuesta y el detector la publica). Si igual vence: "el detector no da latido" a los 15 min, con las causas posibles. |
| El respaldo de GitHub fallaba en cada consulta y la corrida salía "bien" | Ahora, si ninguna consulta funciona, la corrida termina con error, y con 2 seguidas así llega el aviso. |
| Nadie se enteró | Los avisos llegan como **issue de GitHub** a la app de GitHub del celular y por correo. Se cierran solos cuando se arregla. |

## 1. El VPS se actualiza solo desde main

`deploy/actualizar_vps.sh` + `climaar-actualizar.timer` (cada 10 min):
- Baja main del repositorio público, sin clave.
- Si el VPS tiene cambios propios (por eso "main siempre quedaba 1 commit adelante"), los guarda
  en una rama local `respaldo-vps-FECHA` y deja el VPS igual a main.
- Reinicia el detector **solo** si cambió su código (`detector_sinarame.py`, `radar_estado.py`).
  No se pierden imágenes: al volver, la primera consulta ve la más nueva y la despacha si falta
  procesarla (cada imagen se procesa una sola vez). El capturador se abre de nuevo en cada
  consulta, así que sus cambios entran sin reiniciar.
- Usa el Python del VPS (`.venv-ia`; si no está, `.venv`). Si cambian numpy, Pillow o Playwright
  en `requirements.txt`, los instala.
- El resultado va a `.climaar/actualizacion.json`; el detector lo publica en GitHub
  (variable `CLIMAAR_DETECTOR_INFO`, cada 6 h) y la vigilancia avisa si falla.

**Instalación (una sola vez, por SSH):**
```
sudo bash -c "$(curl -fsSL https://raw.githubusercontent.com/lucas12839/Clima-ar/main/deploy/actualizar_vps.sh)"
sudo bash /opt/Clima-ar/deploy/instalar_actualizacion.sh
```
La primera línea deja el VPS igual a main (aunque tenga cambios propios) y reinicia el detector.
La segunda instala el reloj de 10 min.

## 2. Control de cada fuente: `scripts/vigilancia.py`

Corre en el vigilante del detector (`climaar-detector-sinarame.yml`) cada ~10 min, **aunque no
entre radar** (el control de salud de la cadena corre dentro de Intelligence, que se detiene sin
radar). Revisa: latido del VPS, clave por vencer o vencida, actualización del VPS, edad del radar
(40 min con el detector caído; 60 min con el detector sano, porque SINARAME a veces llega 30 min
o más tarde), respaldo de GitHub con errores, reloj de CWOP cortado (lo relanza solo) y reloj
externo que dejó de llamar. `salud_cadena.py` suma también la clave y la actualización.

## 3. Reloj externo de respaldo

Ver `docs/RELOJ_EXTERNO.md` (cron-job.org; Lucas tiene que crear la clave `climaar-reloj-externo`).

## Claves "ClimaAR VPS Sync" (vencen el 6 y el 7/12)

La actualización nueva no usa ninguna clave (el repositorio es público). Para ver si el VPS las
usa para algo, por SSH:
```
git -C /opt/Clima-ar remote -v
sudo grep -rl "github_pat_" /etc /opt/Clima-ar/.git/config /root /home 2>/dev/null
```
Si solo aparece `/etc/climaar-detector.env` (la clave del detector, que vence el 21/02/2027),
las "VPS Sync" ya no hacen falta y se pueden dejar vencer.
