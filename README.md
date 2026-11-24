# ClimaAR

Detector de tormentas en tiempo real para Bahía Blanca, basado en el radar **RMA10 (SINARAME, producto ZH MAX)** y en datos meteorológicos de superficie y modelos.

**Funciona con reglas y física, sin modelos entrenados.** No hay entrenamiento, ni revisión manual de celdas, ni redes neuronales (DGMR, MetNet, etc.).

> ClimaAR no reemplaza la información oficial del SMN ni la evaluación de profesionales.

## Cómo funciona

```
SINARAME publica un frame (cada ~10 min; a veces con 30 min o más de atraso)
 └─ detector_sinarame.py (servicio en el VPS, consulta cada 30 s)
     └─ probar-radar-smn.yml      captura → valida PNG → dBZ → filtrado → seguimiento de celdas
         └─ climaar-intelligence.yml
              nowcast → georreferenciación (240 km) → evolución por reglas
              → trayectoria → fusión meteorológica → Storm Engine → motor de decisión
              → validación de integridad
```

Entre frames, la IA sigue corriendo con el último radar válido y marca cuántos minutos tiene: hasta 15 min es normal, de 15 a 40 min baja la confianza y con más de 40 min no se proyectan celdas (solo riesgo por ambiente). A más de 150 km del radar la confianza también baja.

## Estructura

| Carpeta / archivo | Qué hace |
|---|---|
| `scraper/capturar_radar_smn.py` | Descarga y valida el PNG del RMA10 |
| `scripts/detector_sinarame.py`, `scripts/radar_estado.py` | Reloj del radar: un frame = un procesamiento |
| `scripts/procesador_radar.py`, `scripts/decodificador_rma10_dbz.py` | Colores → dBZ, máscara de ecos fijos |
| `seguimiento_temporal.py` | Seguimiento de celdas entre frames |
| `scripts/adaptar_rma10_intelligence.py`, `scripts/georreferenciar_rma10.py` | Nowcast y georreferenciación |
| `scripts/prediccion_evolucion.py`, `scripts/conectar_evolucion_trayectoria.py` | Evolución y trayectoria a 5/10/20/30 min |
| `scripts/nowcast_flujo_optico.py` | Nowcast por flujo óptico a 10/20/30/60 min y llegada a Bahía Blanca (ver `docs/VALIDACION_NOWCAST.md`) |
| `scripts/prediccion_rma10_ml.py` | Geometría por track (sin modelo; mantiene el contrato del archivo) |
| `scripts/meteorologia_fusion.py`, `scripts/smn_observaciones.py`, `scripts/estaciones/` | Datos meteorológicos y estaciones |
| `goes19/` | Datos satelitales GOES-19 |
| `scripts/tormenta_intelligence*.py`, `scripts/motor_decision_climaar.py` | Storm Engine y decisión final |
| `scripts/clasificacion_tiempo.py` | Tipo de tiempo: llovizna → tormenta potencialmente severa, con granizo / viento / inundación (sin colores de alerta) |
| `scripts/validar_integridad_climaar.py` | Verifica que todo el ciclo esté sincronizado |
| `app.py` | API (FastAPI) |
| `frontend/` | Web (servida desde el VPS con Caddy, `deploy/instalar_web.sh`) |
| `deploy/`, `docs/` | Servicio del detector y documentación |

## Workflows

| Workflow | Disparo |
|---|---|
| `climaar-detector-sinarame.yml` | Vigilante: lo lanza cada ~10 min el reloj de CWOP; solo actúa si el servicio del VPS no da latido |
| `climaar-cwop.yml` | Reloj encadenado (se relanza solo): escucha CWOP y lanza al vigilante del detector |
| `probar-radar-smn.yml` | Lo despacha el detector con cada frame nuevo |
| `climaar-intelligence.yml` | Después de cada frame (o refresco entre frames) |
| `climaar-calidad.yml` | Al terminar Intelligence |
| `observaciones-sazb.yml`, `probar-app.yml` | Manuales / soporte |

## API

`/health`, `/estado`, `/pronostico`, `/radar`, `/radar/imagen`, `/secuencia`, `/nowcast`, `/prediccion`, `/prediccion/en-vivo`, `/tormenta`, `/inteligencia`, `/modelo`, `/modelo/nowcast`.

## Desarrollo

```bash
pip install -r requirements.txt pytest
python -m pytest -q
```

Reglas del repo: `main` es producción. Todo cambio va por rama y pull request.
