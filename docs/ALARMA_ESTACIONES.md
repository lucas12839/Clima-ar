# Idea 3: estaciones como alarma de "ya llega" (20-21/11/2026)

Decisión de Lucas (20/11, noche): usar las estaciones como alarma de "ya llega", respetando el
límite de 15 min. Se valida solo en vivo.

## Cómo funciona (`scripts/estaciones/alarma_llegada.py`, en paralelo)

Cuando una tormenta empuja aire frío hacia adelante (frente de ráfagas), las estaciones que
toca registran en minutos un salto del viento, una caída de la temperatura y una subida de la
presión. Las estaciones BCP miden cada minuto y publican al instante; el radar llega 10-30
min tarde.

1. **Solo datos de 15 min o menos**, con la edad calculada al usar el dato (`frescura.py`).
2. **Firma en la estación** (últimos 30 min): pico de viento en los últimos 15 min (ráfaga de
   35 km/h o más, o el viento sube 10 km/h o más y llega a 15 km/h) y, desde el pico, caída
   de temperatura de 1,5 °C o más o subida de presión de 1,5 hPa o más (promedios de dos
   datos: el sensor de presión salta ±1 hPa solo).
3. **Tiene que haber tormenta cerca**: eco de 35 dBZ o más a 40 km o menos en la última imagen
   del radar, o rayos a 30 km o menos en los últimos 10 min. Sin eso, una racha de viento sola
   no es una tormenta (el 20/11, Puan y Pigüé dieron firmas por el viento general, sin tormenta).
4. **Localidad aguas abajo**: a lo largo del movimiento de las tormentas a 5 km o más de la
   estación y de costado 30 km o menos. Llegada = distancia / velocidad, menos la edad de la
   firma; solo hasta 60 min.
5. Estación a 15 km o menos de la localidad con firma: "llegó".

Salida: `data/ia/estaciones/alarma_llegada.json` (estado por localidad: `ya_llega` con
minutos y estación, o `llego`). Validación en vivo: cada alarma se verifica con lo que pasó
después en la localidad (35 dBZ o más en la zona de 10 km en las imágenes del radar, o firma
en una estación de la localidad) entre la hora de la alarma y la hora estimada + 20 min
(`resumen`, `ultimas_verificadas`).

## Prueba con días reales (`herramientas/simular_alarma_estaciones.py`)

Solo hay historial de estaciones guardado desde el 19/11, así que se probó con dos casos:

- **20/11 (tormentas de la tarde, ráfaga de 71 km/h en Bahía):** 3 alarmas para Punta Alta
  desde Bahía Blanca Centro (8-13 min antes), las 3 acertadas; la tormenta llegó unos 8 min
  antes de lo estimado (el frente de ráfagas va más rápido que la tormenta). Antes del ajuste
  de la zona aguas abajo, Arroyo Pantanoso -> Sierra de la Ventana falló 4 de 4 (la estación
  estaba al costado, no adelante): por eso el mínimo de 5 km a lo largo.
- **19/11 (noche, nube rodillo):** ninguna alarma. Solo Puan dio una firma débil y el
  movimiento iba hacia el este, lejos de las localidades. En Bahía no pasó nada (correcto).

**Ojo:** son pocos casos y los ajustes se hicieron con ellos. La validación real es en vivo.

## Límites

- Las estaciones están casi todas al norte de Bahía. Al oeste y suroeste (de donde vienen
  muchas tormentas: Médanos, Villarino) no hay ninguna estación BCP funcionando (Teniente
  Origone está fuera de servicio desde abril).
- Sin el movimiento del radar no hay "ya llega", solo "llegó".
