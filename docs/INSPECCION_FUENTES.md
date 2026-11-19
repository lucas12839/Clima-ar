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

## 3. Plan ordenado por prioridad

Prioridad = cuánto mejora la precisión, por cuánto trabajo, y qué destraba lo siguiente.

1. **Corregir los umbrales de tormentas, con apoyo satelital.** (Ya era el próximo trabajo.)
   Granizo severo hoy salta con 60 dBZ crudo (el 13/11 en Punta Alta dio "posible severo" con
   granizo chico). Propuesta: además del radar, exigir confirmación física: tope por encima del
   nivel de congelamiento (altura de topes 2 km + nivel de congelamiento de ICON/WRF/ECMWF),
   tope sobresaliente, enfriamiento rápido o salto de rayos. Revisar Tornquist (posible efecto
   de las sierras). Necesita las respuestas de Lucas sobre granizo de octubre y noviembre.
2. **Lectura parcial de GOES-19 y datos por tormenta.** Leer solo la zona de 240 km (no el disco
   completo) y calcular todo por localidad **y por tormenta seguida por el radar**, no por el
   círculo entero. Es la base técnica de los puntos 3 a 5 sin alargar el ciclo.
3. **Señales de tormenta severa por satélite:** topes sobresalientes (método de textura del
   infrarrojo: píxel ≤ −58 °C y ≥ 6,5 °C más frío que el yunque a 8 km), altura de topes 2 km y
   fase hielo, por tormenta.
4. **Rayos mejorados:** salto de rayos con el método estándar (aumento de más de 2 desvíos en
   intervalos de 2 min) por tormenta, y área de los rayos.
5. **Aviso de tormenta que se forma en el lugar (antes del radar):** reglas físicas de
   enfriamiento de topes, congelamiento (fase) y diferencias entre canales. Ataca la principal
   falla medida del pronóstico.
6. **Pronóstico con pysteps:** movimiento con 3 o más imágenes, varios escenarios →
   probabilidad real de llegada por localidad (reemplaza "llega / puede llegar" por un
   porcentaje calculado), crecimiento y debilitamiento, y a 1-3 h mezcla con el WRF del SMN.
   Se mide contra el pronóstico actual con `validar_nowcast_flujo.py` antes de reemplazarlo.
7. **Modelos:** sumar el WRF del SMN (lluvia cada 10 min a 4 km por localidad) y varios modelos
   y ensambles de Open-Meteo: probabilidad de ambiente severo en vez de un solo valor, índice de
   elevación, inhibición y nivel de congelamiento para granizo.
8. **Ambiente medido por satélite** (inestabilidad, agua precipitable y perfiles en cielo
   despejado) para corregir al modelo antes de que se formen las tormentas.
9. **Calibración continua** (ya planificada): lluvia chaparrón por chaparrón con estaciones,
   cortes de granizo con casos reales, repetir `calibrar_llegadas.py` cada mes.
10. **Nunca a ciegas** (ya planificado): probar el modo sin radar con una tormenta real; con los
    puntos 2 a 5 queda mucho más completo.
11. **Vigilancia:** VPS que se actualice solo, control de cada fuente.
12. **Pedidos a instituciones:** datos completos del RMA10 con Doppler (SMN / Recursos
    Hídricos), estaciones de UNS, CONICET, Municipalidad y red provincial. Opcional: instalar un
    receptor de Blitzortung para tener acceso a esa red de rayos.

Al final: actualizar Render (datos) y Netlify (pantalla).
