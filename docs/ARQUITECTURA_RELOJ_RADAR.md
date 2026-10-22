# Reloj radar de ClimaAR

El reloj meteorológico es el **timestamp real del frame RMA10 de SINARAME**.
Ningún `schedule` decide cuándo se procesa un frame.

```
SINARAME publica frame T
  └─ detector (scripts/detector_sinarame.py, consulta liviana cada N s)
       └─ T > last_processed ?  ── no ──> (entre frames) refresco de IA cada M min
            └─ sí: despacha probar-radar-smn.yml (radar_timestamp=T)
                 └─ scraper: descarga → valida PNG → reemplazo atómico del radar activo
                 └─ reclamar(T)   (idempotencia: un T = un procesamiento)
                 └─ procesador_radar → filtrado → seguimiento_temporal → completar(T)
                 └─ artefacto radar-rma10 → climaar-intelligence.yml (radar_run_id exacto)
                      └─ nowcast → ML → evolución → fusión → Storm Engine → Decision → resultado
```

## Estado (`data/radar/estado_radar.json`)
`last_seen_radar_timestamp`, `last_processed_radar_timestamp`, `radar_timestamp`,
`radar_age_minutos` / `vigencia_radar` (se recalculan al leer con `radar_estado.snapshot`),
`estado_collector`, `estado_procesamiento`, `procesando_radar_timestamp`,
`pendiente_radar_timestamp`. El detector guarda además su estado local en
`.climaar/detector_estado.json`.

## Garantías
- **Un frame, un procesamiento**: `radar_estado.reclamar` bajo lock de archivo; el grupo de
  concurrencia `climaar-production-writes` serializa Radar, Intelligence y escrituras a `main`.
- **Frame nuevo durante un procesamiento**: se encola solo el más nuevo (`pendiente_*`);
  GitHub mantiene como máximo 1 run en curso + 1 pendiente por grupo.
- **Radar activo siempre completo**: temporal → validación PNG → `os.replace`.
  El refresco de IA exige `radar_timestamp.txt == last_processed_radar_timestamp`.
- **Entre frames** la IA sigue: el detector pide `climaar-intelligence.yml modo=refresco`
  (sin reprocesar radar) usando el último radar válido; `apply_radar_age` extrapola hasta
  180 min y luego marca `radar_obsoleto` (límites sin cambios).
- **Salida de IA**: `radar_timestamp`, `radar_age`, `vigencia_radar` y `origen_datos`
  (`observacion_radar` / `extrapolacion_nowcast` / `prediccion_ml`).

## Schedules
Solo `climaar-detector-sinarame.yml` (`11,41 * * * *`), que es **watchdog del detector**.
Radar, Intelligence, SAZB, ciclo, watchdog y auto-radar no tienen schedule.

## Limitaciones reales de GitHub Actions
- No hay servicios persistentes: un job dura ≤ 6 h. El detector corre ~5 h 35 min y se relanza
  solo (`--relevo`); el cron lo reinicia si se perdió. Entre relevos puede haber un hueco corto.
- `schedule` puede retrasarse o perderse; por eso solo vigila, no procesa.
- Un job largo consume minutos de Actions (gratis en repos públicos).
- Cada consulta abre Chromium (~25–35 s): el intervalo efectivo mínimo es ~60 s en Actions.
- **Componente que debería ser servicio persistente**: `scripts/detector_sinarame.py --loop`
  en una VM/Raspberry (ver `deploy/climaar-detector.service`), con intervalo de 30 s.
  Es seguro correr ambos a la vez: el reclamo por timestamp evita duplicados.
- No se pudo verificar contra SINARAME real desde el entorno de desarrollo (sin red);
  la consulta reutiliza la extracción ya existente del scraper.
