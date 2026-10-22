# Despliegue del detector SINARAME como servicio persistente

El detector (`scripts/detector_sinarame.py --loop`) es **el único proceso que vigila
SINARAME**. Consulta cada **30 s** (solo lista timestamps RMA10; no descarga ni procesa
meteorología) y, cuando aparece un `radar_timestamp` nuevo, despacha
`probar-radar-smn.yml`. No depende de ningún cron.

```
SINARAME → Detector (systemd, 30 s) → Radar → Tracking → Nowcast → IA
                 │ latido (variable CLIMAAR_DETECTOR_LATIDO)
                 ▼
GitHub Actions = watchdog (cron cada 10 min): solo actúa si el latido está apagado
```

Cadena y colas sin cambios (1 en curso + 1 pendiente por grupo de concurrencia).

## Requisitos
- VM o Raspberry Pi (64 bits recomendado, ≥ 1 GB RAM libres: Chromium) con salida a Internet.
- Debian/Ubuntu/Raspberry Pi OS con systemd, Python 3.10+.
- Un **PAT fine-grained** limitado a `lucas12839/Clima-ar` con:
  `Actions: Read and write`, `Variables: Read and write`, `Contents: Read-only`.

## Instalación
```bash
git clone https://github.com/lucas12839/Clima-ar.git /tmp/Clima-ar
sudo bash /tmp/Clima-ar/deploy/instalar_detector.sh
sudo nano /etc/climaar-detector.env        # GH_TOKEN=<tu PAT> (modo 600, ya creado)
sudo systemctl restart climaar-detector
```
El instalador crea el usuario `climaar`, clona en `/opt/Clima-ar`, crea `.venv`, instala
Playwright + Chromium, copia `deploy/climaar-detector.service` y lo habilita al arranque.

Parámetros (en el `.service`): `CLIMAAR_DETECTOR_INTERVALO_SEG=30`, `CLIMAAR_REFRESCO_MIN=10`
(refresco de la IA entre frames), `CLIMAAR_REDESPACHO_MIN=10`. Tras editarlos:
`sudo systemctl daemon-reload && sudo systemctl restart climaar-detector`.

**Reinicio automático:** `Restart=always`, `RestartSec=15`, sin límite de reinicios, y
`enable` lo arranca con la máquina.

## Verificación
1. Servicio activo: `systemctl is-active climaar-detector` → `active`;
   `systemctl status climaar-detector`.
2. Ticks cada ~30 s: `journalctl -u climaar-detector -f` → líneas
   `… tick: sin_cambio` / `despachar` / `en_proceso`.
3. Salud local (sale 0 si sano, 1 si no):
   `sudo -u climaar bash -c 'cd /opt/Clima-ar && .venv/bin/python scripts/detector_sinarame.py --estado'`
   (`sano: true`, `edad_ultima_consulta_min` < 1, `collector_state: ok`).
4. Latido visible en GitHub: repo → *Settings → Secrets and variables → Actions → Variables* →
   `CLIMAAR_DETECTOR_LATIDO` debe tener un timestamp de hace < 2–3 min.
5. Watchdog en reposo: en *Actions → ClimaAR Detector SINARAME (watchdog)* los runs deben
   mostrar “Servicio sano - no se hace nada” (sin instalar Chromium).
6. Frame nuevo de punta a punta: cuando SINARAME publique, aparece un run de
   *ClimaAR Radar RMA10* con `radar_timestamp` y luego *Intelligence*;
   `data/radar/estado_radar.json` en `main` avanza `last_processed_radar_timestamp`.
7. Reinicio automático: terminar el proceso del detector de forma abrupta (por ejemplo con
   la señal SIGKILL sobre el `MainPID` que muestra `systemctl status climaar-detector`);
   systemd lo vuelve a levantar en ~15 s (`journalctl -u climaar-detector`).
8. Recuperación ante servicio caído: `sudo systemctl stop climaar-detector`; a los ≤ 5 min el
   latido caduca y el siguiente watchdog (≤ 10 min) ejecuta el detector temporal (~9 min) con
   `::warning::`. `sudo systemctl start climaar-detector` y vuelve a reposo.

## Rol de GitHub Actions
- **Sin schedule** en Radar, Intelligence, SAZB, ciclo, watchdog y auto-radar.
- Único schedule: `climaar-detector-sinarame.yml` (`3,13,23,33,43,53 * * * *`), solo
  WATCHDOG/recovery. Lee `vars.CLIMAAR_DETECTOR_LATIDO`; si tiene menos de 5 min no hace
  nada. Si no, corre el mismo detector ~9 min (puede duplicar una consulta pero no un
  procesamiento: el reclamo por `radar_timestamp` es idempotente).
- Peor caso con servicio caído: hasta ~5 min (caducidad) + ~10 min (cron) sin vigilancia, más
  retrasos propios de `schedule`; durante la recuperación el intervalo vuelve a ser 30 s.

## Actualizar el código del servicio
```bash
sudo -u climaar git -C /opt/Clima-ar pull --ff-only
sudo systemctl restart climaar-detector
```

## Seguridad
`/etc/climaar-detector.env` solo legible por root (modo 600). Rotar el PAT periódicamente; el
servicio corre como usuario sin shell con `NoNewPrivileges`.

## Limitaciones
- El PAT vence: si lo hace, el latido y los despachos fallan (queda en `journalctl`) y
  el watchdog toma el relevo en Actions. Renovar el token y reiniciar.
- Si la máquina pierde red, no hay detección; vuelve sola al reconectar. El watchdog cubre el hueco.
- No se verificó contra SINARAME real desde el entorno de desarrollo (sin red).
