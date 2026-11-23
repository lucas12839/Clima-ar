# Punto 10 (etapa A, punto 5): modo sin radar probado con tormentas reales (21-23/11/2026)

`herramientas/simular_sin_radar.py`. Se usaron 26 días de tormentas reales (radar SINARAME,
GOES-19 y rayos guardados), 2.164 imágenes, cada una en las 7 localidades. En cada imagen se
hace de cuenta que el radar dejó de llegar hace 40, 60 o 90 min. Se arma lo que ClimaAR
mostraría con lo publicado a esa hora y se compara con lo que mostraba el radar real.

## Resultado (acierto CSI: 1 = perfecto; "falsas" = de cada 100 avisos, cuántos sobraron)

| ¿Llueve en la localidad? | sin radar 40 min | 60 min | 90 min | falsas (40 min) |
|---|---|---|---|---|
| Antes: tope <= -50 °C | 0,08 | 0,08 | 0,08 | 70 |
| Radar viejo movido | 0,40 | 0,33 | 0,29 | 45 |
| **Ahora: radar viejo movido + tope <= -60 °C con rayos** | **0,41** | **0,34** | **0,30** | 45 |

| ¿Lluvia moderada o más? | 40 min | 60 min | 90 min | falsas |
|---|---|---|---|---|
| Antes: tope <= -50 °C | 0,09 | 0,09 | 0,09 | 88 |
| Tope <= -60 °C y >= 1 rayo a 15 km (sin radar) | 0,33 | 0,33 | 0,33 | 54 |
| **Ahora: radar viejo movido + esa regla** | **0,32** | **0,28** | **0,26** | 64 |

La regla de antes casi nunca acertaba: detectaba 9 de cada 100 lluvias y 7 de cada 10 avisos
eran falsos. La última imagen de radar, movida con el nowcast, sigue siendo lo mejor para
saber si llueve, incluso con 90 min de atraso.

## Qué cambió

- **Radar de 40 a 90 min:** la lluvia de cada localidad sale del radar viejo movido hasta
  ahora (`nowcast_capas.json`, `radar_movido_a_ahora`), marcada como estimada. Si el satélite y
  los rayos dicen algo más fuerte, manda eso.
- **Sin radar o con más de 90 min:** la lluvia probable sale de un tope de -60 °C o más frío
  Y al menos un rayo a 15 km o menos (antes alcanzaba con -50 °C).
- La lluvia medida por las estaciones y los rayos (tormenta eléctrica) siguen igual.
- Granizo y viento sin radar: siguen como estaban (estaciones y modelos). No hay datos para
  probarlos sin radar.

Probado: `pytest` (352 OK); `emular_workflow.py` de Intelligence OK; caso real del 20/11 con el
radar de 22:20 a las 23:10 (50 min): Bahía, White y Cerri dan "lluvia débil", y en Bahía Centro
llovía débil a esa hora.
