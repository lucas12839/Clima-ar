# Inspección de herramientas y fuentes (19/11/2026, 17:00 hora de Argentina)

Revisión del código de ClimaAR y prueba real, desde GitHub Actions, de qué datos hay
disponibles sobre Bahía Blanca. Herramientas para repetirla:
`herramientas/inspeccion_fuentes.py` y `herramientas/inspeccion_fuentes2.py` (necesitan red).

## 1. Qué usa hoy ClimaAR y qué desaprovecha

### GOES-19 (`goes19/`)
| Producto | Hoy | Problema |
|---|---|---|
| Canal 13 (infrarrojo, topes) | mínimo y enfriamiento por localidad | solo a 15/30 km de cada localidad, no por tormenta |
| Canal 8 (vapor de agua) | estadística de todo el círculo de 240 km | no se usa por localidad ni por tormenta |
| Altura y presión de topes | estadística de todo el círculo (resolución 10 km) | existe la versión de **2 km** (`ACHA2KMF`) y no se usa |
| Rayos (GLM) | cantidad por localidad en 20 min | no usa el **área** ni la energía de cada rayo; el salto se mide con dos mitades de 10 min, no con el método estándar |
| Vientos por satélite (DMW) | promedio general | llegan cada hora con ~40 min de atraso: poco útiles para el ciclo de 10 min |

Además, cada ciclo **baja el disco completo** de cada producto (canal 13: 24,5 MB; altura de
topes: 29,6 MB; visible: 382 MB). Probado hoy: se puede leer **solo la zona de Bahía Blanca**
pidiendo los pedazos del archivo que hacen falta (1,1 s para el canal 13). Eso permite sumar
muchos productos sin hacer más lento el ciclo.

### Pronóstico (`scripts/nowcast_flujo_optico.py`)
- Movimiento calculado con **solo 2 imágenes** y un único movimiento por zona.
- Traslada la tormenta tal como está: **no crece ni se debilita** en el pronóstico.
- Un solo escenario. La "confiabilidad de llegada" se saca de estadística aparte (30 días).
- La validación (`docs/VALIDACION_NOWCAST.md`) muestra que las fallas más graves son
  **tormentas que se forman o intensifican en el lugar**: ninguna traslación las anticipa.

### Radar
- Solo llega la imagen PNG de reflectividad máxima de la columna (ZH MAX), ~1 km por píxel.
  La página de SINARAME no publica otros productos (comprobado hoy): sin Doppler, sin altura
  de ecos, sin agua líquida integrada. Para granizo eso obliga a apoyarse en el satélite y en
  el nivel de congelamiento.

### Modelo
- Solo ECMWF por Open-Meteo, un único escenario.

## 2. Qué hay disponible (comprobado hoy en GitHub)

### GOES-19: productos que sirven y no usamos
| Producto | Qué da | Para qué nos sirve |
|---|---|---|
| `ACHA2KMF` altura de topes 2 km | altura de la nube cada 2 km | **granizo**: altura del tope por encima del nivel de congelamiento |
| `ACTPF` fase del tope | agua, agua sobreenfriada, mixta, hielo | **inicio de tormenta**: el tope que se congela anuncia convección antes que el radar |
| Canales 11, 14, 15 y 16 (infrarrojos) | diferencias entre canales | inicio de tormenta (reglas físicas de enfriamiento y congelamiento) |
| Canal 2 (visible 0,5 km, de día) | textura del tope | topes sobresalientes de día |
| `DSIF` inestabilidad observada | CAPE, índice de elevación, K, Total Totals, Showalter (cada 10 km, solo cielo despejado) | ambiente **medido** antes de que se nuble, para comparar con el modelo |
| `TPWF` agua precipitable | humedad de la columna | lluvia muy eficiente (inundación) |
| `LVTPF` / `LVMPF` perfiles | temperatura y humedad en 101 niveles (cielo despejado) | ambiente medido en altura |
| GLM: `flash_area`, `flash_energy`, grupos | tamaño y energía de cada rayo | **rayos chicos y muchos = corriente ascendente fuerte** (granizo, ráfagas) |

Descartados:
- **Mesoescala de 1 minuto (M1/M2):** en las últimas 48 h estuvo siempre sobre Estados Unidos.
  Solo se movió a Argentina en campañas científicas (RELAMPAGO 2018-2019). Se puede usar si
  algún día cubre la zona, pero no se puede contar con ella.
- **Lluvia estimada por satélite (`RRQPEF`):** NOAA dejó de publicarla el 2/11/2026.
- Vientos por satélite: ya integrados; llegan cada hora, poco más para sacar.

### Otras fuentes
| Fuente | Qué da | Comprobado |
|---|---|---|
| **WRF del SMN** (AWS, abierto, `smn-ar-wrf`) | modelo argentino de **4 km**, lluvia **cada 10 min** hasta 72 h; además viento, temperatura y nivel de congelamiento cada hora | sí: el de las 12 UTC estuvo a las 11:41 hora de Argentina |
| **Varios modelos por Open-Meteo** | GFS (con índice de elevación e inhibición), ICON (nivel de congelamiento), GEM, UKMO, ARPEGE | sí |
| **Ensambles de Open-Meteo** | ECMWF 51 escenarios, GFS 31, ICON 40: probabilidad de CAPE alto, lluvia y ráfagas | sí (122 series de CAPE para Bahía) |
| **pysteps** (biblioteca abierta de pronóstico inmediato, física y estadística, sin entrenamiento) | movimiento con varias imágenes, escenarios múltiples, crecimiento y debilitamiento, mezcla con modelo | compatible con las reglas del repo (no es aprendizaje automático) |
| Datos completos del RMA10 (con Doppler) | volumen 3D, velocidad del viento | **no son públicos**: hay que pedirlos a SMN / Subsecretaría de Recursos Hídricos |
| Blitzortung (rayos de red ciudadana) | rayos en tiempo real | **no da acceso a terceros**; solo a quien instala un receptor |

## 3. Cómo lo resolvieron los sistemas de referencia (y qué tomamos)

- **NowCastMIX (servicio meteorológico alemán):** cada 5 min junta radar, rayos, estaciones y
  modelos y calcula, con reglas graduales ("lógica difusa": cada dato suma de a poco, sin cortes de
  sí/no), el potencial de granizo, ráfagas y lluvia intensa de cada tormenta; después la mueve con
  su desplazamiento para la próxima hora. **Sin modelos entrenados: es nuestra base.**
- **RDT (satélites europeos):** describe y sigue cada tormenta solo con satélite (tope, topes
  sobresalientes, rayos, modelo); se usa donde no hay radar. **Base del modo sin radar.**
- **ProbSevere (Estados Unidos):** mismos ingredientes por tormenta, pero combinados con modelos
  entrenados. Solo como guía de qué señales mirar (por ejemplo, el área mínima de los rayos).

Adaptación: todo se calcula **solo dentro de los 240 km** y **solo sobre las tormentas que ya sigue
el radar**. Nada se adopta si no se mide antes su costo en el ciclo y su mejora en aciertos.

## 4. Reglas de confiabilidad (valen para todos los puntos)

Objetivo: un sistema confiable que en el futuro pueda dar avisos de tormentas severas.

1. **Sincronización:** cada dato nuevo lleva su hora de captura y solo se usa si corresponde a la
   imagen de radar del ciclo (satélite hasta ~15 min antes; rayos del mismo intervalo). Nunca se
   mezclan ciclos.
2. **Compatibilidad:** a los archivos que leen la API y la pantalla solo se les **agregan** campos;
   nunca se cambian ni se borran los existentes.
3. **Integridad:** `validar_integridad_climaar.py` controla cada fuente nueva (hora, atraso,
   ciclo).
4. **Falla segura:** si falta una fuente, se dice y baja la confianza. **Nunca** se muestra "sin
   tormenta" por falta de datos.
5. **Prueba en paralelo:** cada mejora corre primero "en sombra" (calcula y guarda, pero no se
   muestra) junto a la versión actual, y la reemplaza solo si acierta más.
6. **Medición de aciertos:** por fenómeno, porcentaje de tormentas detectadas, falsas alarmas y
   minutos de anticipación, contra estaciones y rayos de GOES-19. **Decisión de Lucas
   (19/11): se mide con simulaciones de casos reales al terminar los 11 puntos**, y recién ahí
   se decide sobre avisos.
7. **Casos guardados:** cada tormenta real queda como caso de prueba; todo cambio se vuelve a
   correr sobre esos casos antes de unirse a main. Radar: `data/radar/historico`. Satélite y
   rayos por tormenta: `data/goes19/historial/AAAA-MM-DD.jsonl` (desde el 19/11/2026, #44;
   `goes19/historial.py` tiene `leer()` para las simulaciones).
8. **Pruebas:** con tormenta y sin actividad, `pytest` y emulación de los procesos reales.

## 5. Orden de trabajo

Historia: se empezó por la corrección de umbrales (Punta Alta, Tornquist); Lucas pidió revisar
todas las herramientas y salió este plan. La corrección de umbrales quedó dentro del punto 4,
porque necesita antes los datos de los puntos 1 a 3. Es un solo plan.

1. **Base del satélite:** leer solo la zona de 240 km de GOES-19 y calcular por tormenta seguida
   por el radar. **Terminado (#42)**: además se corrigió la ubicación de los píxeles (los datos
   salían del Pacífico), se sincronizó por barrido y hora, y se corrige la perspectiva.
2. **Señales de tormenta severa por satélite:** topes sobresalientes (píxel ≤ −58 °C y ≥ 6,5 °C
   más frío que el yunque a 8 km), altura del tope sobre el nivel de congelamiento (altura 2 km),
   hielo en el tope. **Terminado (#43)**: además vapor − infrarrojo y el nivel de congelamiento
   calculado del perfil del modelo (venía vacío).
3. **Rayos mejorados:** salto de rayos con el método estándar (más de 2 desvíos en intervalos de
   2 min) y área de los rayos, por tormenta. **Terminado (#45)**: además cada rayo cuenta para
   una sola tormenta.
4. **Puntaje de gravedad por tormenta (estilo NowCastMIX):** granizo, ráfagas y lluvia intensa con
   radar + satélite + rayos + modelo. **Reemplaza la corrección de umbrales** (Punta Alta,
   Tornquist). Mientras tanto se mantiene "posible severa". **Terminado en paralelo (#46)**:
   `scripts/puntaje_gravedad.py`, no cambia lo que se ve hasta las simulaciones.
5. **Aviso de tormenta que se forma en el lugar,** antes del radar (enfriamiento de topes,
   congelamiento, diferencias entre canales; estilo RDT). **Terminado en paralelo (#48)**:
   `goes19/formacion.py` + `scripts/aviso_formacion.py`, no cambia lo que se ve. Simulado con
   3 días reales: grado "probable" o más acierta 58-85 % con ~20 min de ventaja.
5b. **Nunca a ciegas cuando el radar se atrasa o falta** (agregado el 20/11 por pedido de Lucas,
   va antes del 6). Noche del 19/11: SINARAME publicó con 16-18 min de atraso casi toda la
   noche y faltaron las imágenes de 22:20 y 00:00 (hora de Argentina), justo cuando la
   tormenta pasaba por Bahía Blanca (llegó débil, se intensificó y se deshizo entre dos
   imágenes). Qué hacer: (1) medir y mostrar la edad del radar en cada dato; (2) mover las
   tormentas a la hora actual con su velocidad (radar adelantado); (3) cubrir el hueco con lo
   que llega casi en tiempo real: rayos GLM (cada 20 s), satélite (cada 10 min, tendencia del
   tope por localidad) y estaciones (cada ~5 min: lluvia, ráfaga, salto de presión, caída de
   temperatura = frente de ráfagas); (4) buscar otra fuente de radar que llegue antes
   (otros productos de SINARAME, mosaico del SMN) y recuperar imágenes que faltan;
   (5) detectar **convección elevada sobre capa estable** (inversión cerca del suelo:
   temperatura a 925 hPa mayor que la de 2 m, inhibición alta, capa límite baja, de noche) y
   bajar el viento y el granizo esperados en superficie; (6) leer la **firma de las
   estaciones** (temperatura que sube + humedad que baja + presión que sube de a poco = frente
   de ráfagas débil; bajón fuerte de temperatura + salto de presión + ráfaga fuerte =
   reventón). Caso guardado: `docs/casos/2026-11-19_nube_rodillo_bahia.md`.
6. **Mejorar nuestro pronóstico** (mismo proceso y mismos archivos): movimiento con 3 o más
   imágenes, crecimiento o debilitamiento según el puntaje, varios escenarios y porcentaje de
   llegada (pysteps sobre nuestra grilla de 512 × 512, solo si su costo en el ciclo lo permite).
7. **Modelos:** WRF del SMN y escenarios de varios modelos, solo puntos dentro de 240 km.
8. **Ambiente medido por el satélite** (inestabilidad, agua precipitable, perfiles) antes de que
   se nuble.
9. **Calibración y medición continua:** lluvia con estaciones, puntaje con estaciones y rayos,
   informe mensual de aciertos.
10. **Nunca a ciegas:** probar el modo sin radar (satélite + rayos + estaciones) con una tormenta
    real.
11. **Vigilancia:** VPS que se actualice solo, control de cada fuente.

Al final: actualizar Render (datos) y Netlify (pantalla).

Descartado: pedidos a instituciones (casi imposible); se resuelve con lo que tenemos. Los datos
completos del RMA10 ya fueron pedidos por Lucas: si llegan, se suman.

Uso estimado de lo que GOES-19 ofrece y nos sirve: hoy ~25-30 %; con los puntos 1-3 ~60 %; con el
5 ~75 %; con el 8 ~85-90 %. El resto no se puede (mesoescala de 1 min casi nunca sobre Argentina;
lluvia satelital discontinuada; visible 0,5 km solo de día y pesado).
