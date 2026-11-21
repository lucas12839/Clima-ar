# Etapa A, puntos 1 a 4: compensar el atraso del radar con lo medido (21/11/2026)

Pedido de Lucas: hacer los puntos 1 a 4 de la etapa A y llevar cada uno al máximo posible.
Regla: solo se usa lo que mejora en la simulación con datos reales.

## Cómo se probó

`herramientas/simular_etapa_a.py`. Como en vivo: a la hora "ahora", la última imagen de radar
tiene un atraso de 10, 20 o 30 min. Se usa solo lo que ya estaría publicado a esa hora:
barridos de GOES-19 (salen ~10 min después de empezar), rayos de los últimos 20 min y
estaciones. Con eso se pronostica qué hay ahora y dentro de 10, 20, 30 y 60 min, y se compara
con la imagen real de esa hora (limpia de ecos falsos).

- 26 días con lluvia (24/10 al 20/11), **1.419 imágenes de partida**, todas (no una de cada dos).
  Se corrió en GitHub, un día por trabajo (rama temporal `prueba-etapa-a`, 6 trabajos a la vez
  para no frenar producción).
- Se mide en las 7 localidades y en 150 puntos de la región. CSI: 1 = perfecto, más alto = mejor.
- "Producción" = lo que corre hoy (escenarios + corrector con rayos y satélite).
- Las trayectorias se midieron con `herramientas/comparar_trayectorias.py --goes` (8 días de
  tormentas, ~3.700 casos a 10 min), con el radar llegando 20 min tarde.

Tablas: tormenta (35 dBZ), radar atrasado 20 min (lo normal) y 30 min.

## Punto 1: corrector con estaciones (`scripts/corrector_estaciones.py`)

Hace lo pedido: si una estación midió lluvia fuerte en los últimos 10 min, la pone en el
pronóstico. Si no midió nada en 15 min, saca la lluvia fuerte que el pronóstico tenía encima
desde hacía 10 min o más. Esa corrección después se mueve con la tormenta.

**Resultado:** en la simulación no cambió nada medible (CSI igual hasta el tercer decimal).
El historial de estaciones guardado empieza recién el 6/11 y casi no tiene lluvia (6 días,
230 casos; en ese tiempo el corrector actuó 32 veces). No alcanza para decir si mejora o empeora.

**Cómo queda:** en paralelo. En cada ciclo se guarda el pronóstico con y sin el corrector y se
compara con el radar real (`verificacion_nowcast.json`, método `estaciones`). Se activa
cuando los datos en vivo muestren que mejora.

**Arreglado en el camino:** una lectura errónea de La Vitícola (21/11 02:47: 6,0 → 0,0 → 6,0 mm)
se contaba como 6 mm de lluvia nueva en `senales_superficie.py` (lluvia de 1 h y 3 h por
localidad). Ahora el acumulado solo puede reiniciarse a la medianoche local.

## Punto 2: posición de ahora con el satélite (`scripts/corrector_satelite.py`)

Se probaron tres formas de usar el barrido más nuevo (8-18 min más fresco que el radar):

| Tormenta, localidades | 20/0 | 20/10 | 20/30 | 30/0 | 30/30 |
|---|---|---|---|---|---|
| Producción | **0,485** | **0,420** | 0,285 | **0,423** | 0,287 |
| Corrimiento de las nubes (flujo de los topes) | 0,417 | 0,379 | 0,296 | 0,353 | 0,249 |
| Ídem, mitad del corrimiento | 0,457 | 0,381 | 0,292 | 0,370 | 0,291 |

(primera ronda, 478 casos; atraso/plazo en min)

| Tormenta, localidades (1.419 casos) | 20/0 | 20/10 | 20/30 | 30/0 | 30/30 |
|---|---|---|---|---|---|
| Producción | 0,512 | 0,415 | 0,315 | 0,424 | 0,290 |
| Núcleo frío de cada tormenta | 0,497 | 0,407 | 0,313 | 0,406 | 0,286 |
| Ídem, mitad | 0,510 | 0,419 | 0,308 | 0,418 | 0,284 |
| Centro de los rayos de cada tormenta | 0,515 | 0,419 | 0,317 | 0,435 | 0,288 |

En la región, las tres quedan iguales o un poco peor.

**Trayectorias** (error típico a 10 / 20 / 30 min): producción 4,5 / 9,5 / 14,7 km;
corrimiento de las nubes 5,6 / 9,9 / 15,0 km; núcleo frío 4,6 / 9,6 / 14,7 km.

**Por qué no sirve:** el satélite ve la parte de arriba de la nube, el yunque, a 10-12 km de
altura. Los vientos de allá arriba no son los que mueven la lluvia, y el yunque se estira y se
desparrama hacia adelante. Por eso, correr la lluvia con el movimiento de la nube la aleja de
donde está de verdad. El núcleo frío y los rayos quedan casi pegados al radar (diferencias de
±0,01, ruido). **No se activa.**

## Punto 3: movimiento medido por el satélite

- Mezclado con el del radar para lo que viene: **empeora** en todo (localidades 20/0:
  0,485 → 0,421).
- Solo cuando el radar no da movimiento (falta la imagen anterior: pasó en 6 de 883 imágenes
  con lluvia), comparado con dejar la tormenta quieta:

| | Localidades 20/0 | Localidades 30/30 | Región 20/0 | Región 30/30 |
|---|---|---|---|---|
| Tormenta quieta (lo de hoy) | **0,460** | **0,263** | 0,413 | 0,242 |
| Movida con el satélite | 0,412 | 0,254 | **0,422** | **0,251** |

Mejora un poco lejos, pero empeora en las localidades, que es lo que importa. **No se
activa** (queda `movimiento_si_falta_radar: false`).

## Punto 4: crece o se debilita (enfriamiento del tope medido y rayos)

Siguiendo a cada tope entre el barrido de la hora del radar y el más nuevo: si se enfría, la
tormenta crece y se sube su intensidad; si se calienta, se baja. Se probó con varias fuerzas
(0,5 a 1,5 dBZ por °C), solo subir, solo bajar, tope máximo de 5 dBZ, umbral de -30 °C y con rayos.

| Tormenta, localidades (1.419 casos) | 20/0 | 20/10 | 20/30 | 30/0 | 30/30 |
|---|---|---|---|---|---|
| Producción | 0,512 | 0,415 | 0,315 | 0,424 | 0,290 |
| Solo sube (1 dBZ/°C) | 0,503 | 0,407 | 0,310 | 0,409 | 0,291 |
| Solo baja | 0,514 | 0,405 | 0,319 | 0,415 | 0,286 |
| Los dos (0,5) | 0,512 | 0,402 | 0,319 | 0,412 | 0,282 |

En la región pasa lo mismo, con diferencias de ±0,01. Gana en unos plazos y pierde en otros:
es ruido. La tendencia de los rayos (primera ronda) también dio igual. **No se activa.**
Ya estaba probado (20/11) que la tendencia del radar empeora, y lo que sí funciona ya está en
producción: el corrector borra la lluvia donde el satélite no ve nube y agrega tormenta donde
hay rayos con tope frío.

## Conclusión (sin rodeos)

Con estos datos (GOES-19 cada 10 min a 2 km, rayos GLM y 13 estaciones), los puntos 2, 3 y 4
**no bajan el error**: la trayectoria sigue en 4,5 / 9,5 / 14,7 km. El satélite sirve para
decir si hay o no hay tormenta (y eso ya se usa), pero no para ubicarla mejor que el radar
viejo movido. El punto 1 está hecho y queda validándose en vivo, porque falta historia de
lluvia en las estaciones.

El código queda en el repo y apagado (`config/nowcast_capas.json`: `satelite_etapa_a`),
probado (`tests/test_etapa_a.py`), con la simulación lista para repetir cuando haya más datos:

    python herramientas/simular_etapa_a.py --goes CARPETA --variantes '{...}' --salida x.json
    python herramientas/simular_etapa_a.py --goes x --sumar res/*.json
    python herramientas/comparar_trayectorias.py --goes CARPETA --llegada 20 --variantes '{...}'
