# Índices por fenómeno y análisis por localidad (19/11/2026)

## Qué cambió

1. **Cada fenómeno por separado** (`scripts/indices_fenomenos.py`). Lluvia, viento,
   granizo y tormenta eléctrica tienen su propia escala y pueden estar o no estar.
   Se informa la combinación real: "Viento fuerte, sin lluvia", "Rayos aislados, sin
   lluvia", "Lluvia fuerte con viento fuerte".
2. **Cada localidad por separado** (`config/localidades.json`, `scripts/localidades_radar.py`):
   Bahía Blanca, Ingeniero White, General Daniel Cerri, Punta Alta, Tornquist, Médanos y
   Sierra de la Ventana. Cada una tiene su observación, su llegada, su acumulado, sus
   estaciones (a ≤ 15 km), sus rayos y sus topes de nube.
3. **La llegada no se da por hecha**: cada llegada dice cuántas veces llegó de verdad un
   pronóstico parecido en 30 días de radar (`config/calibracion_llegadas.json`).
4. **GOES-19 más aprovechado**: rayos de los últimos 20 min por localidad con el salto
   brusco (`goes19/glm_ventana.py`) y topes de nube por localidad con su enfriamiento
   (`goes19/topes_localidad.py`).
5. **Ambiente en altura** (`scripts/ambiente_severo.py`): cizalladura, gradiente térmico
   y aire seco del modelo europeo, para saber qué puede dar una tormenta.

## Escalas

Grado común 0–4 para comparar fenómenos. La calificación general es el grado más alto y
dice de qué fenómeno se trata: "Fuerte (viento)", "Posible severa (lluvia)".

| Grado | Lluvia | Viento | Granizo | Tormenta eléctrica |
|---|---|---|---|---|
| 0 | sin lluvia | sin viento destacable (< 30 km/h) | sin señal | sin rayos |
| 1 | débil / llovizna (< 3 mm/h) | – | – | aislada (< 2 rayos/min a 30 km) |
| 2 | moderada (3–12 mm/h) | moderado (30–50 km/h) | posible (50–55 dBZ) | – |
| 3 | fuerte (≥ 12 mm/h) | fuerte (50–90 km/h, o línea a 60–75 km/h) | probable (55–60 dBZ) | frecuente (≥ 2 rayos/min) |
| 4 | severa: ≥ 30 mm en 1 h, ≥ 50 mm en 3 h, fuerte > 1 h o acumulado alto mientras sigue lloviendo | severo: ráfaga ≥ 90 km/h, o línea ≥ 75 km/h en dos frames seguidos | severo (≥ 60 dBZ) | severa: salto brusco de rayos |

- **Grado 4 sin confirmación de una estación = "posible severa"** (decisión de Lucas,
  19/11/2026), hasta calibrar con datos de SINARAME.
- La lluvia instantánea llega como mucho a "fuerte": la severa depende de cuánto dura o
  acumula. Si ya paró, el acumulado se informa aparte ("posibles anegamientos").
- El modelo europeo solo anticipa viento hasta "fuerte": no mide.
- Topes muy fríos (≤ −60 °C) o enfriándose ≥ 8 °C en 10 min suben un grado el granizo
  de una tormenta que ya tiene ≥ 50 dBZ. El ambiente (CAPE ≥ 1000 J/kg y cizalladura
  ≥ 55 km/h) también. Ninguno de los dos crea fenómenos solo.
- Sin radar: lluvia por estaciones o "probable" por topes ≤ −50 °C; viento por
  estaciones y modelo; rayos por GOES-19.

## Lluvia por localidad: promedio del área, no el píxel más intenso

Viernes 13/11/2026 (día local), lluvia acumulada estimada por el radar crudo:

| Localidad | Píxel más intenso | Promedio del área |
|---|---|---|
| Bahía Blanca | 175 mm | **7 mm** |
| Ingeniero White | 150 mm | 26 mm |
| Punta Alta | 509 mm | **191 mm** |
| Médanos | 75 mm | 18 mm |

Según Lucas, en Bahía Blanca llovió poco y en Punta Alta más de 70 mm: el promedio del
área reproduce la diferencia. El radar crudo de SINARAME parece **sobreestimar** (191 mm
contra "más de 70"): hace falta el dato real de Punta Alta para corregir la fórmula.

## Calibración de llegadas (20/10–19/11/2026, 2913 frames)

`herramientas/calibrar_llegadas.py`. Pronósticos de eco ≥ 35 dBZ llegando a una
localidad en ≤ 60 min, todas las localidades juntas:

| Plazo | 35–40 dBZ | 40–45 dBZ | ≥ 45 dBZ |
|---|---|---|---|
| 0–20 min | 37 % (153) | 58 % (48) | **68 %** (31) |
| 20–40 min | 32 % (195) | 28 % (61) | 29 % (28) |
| 40–60 min | 26 % (192) | 38 % (58) | 36 % (42) |

Cuando no llegó, **casi siempre pasó al costado** (a ≤ 20 km). Pocas veces se deshizo.
Por eso:

- ClimaAR dice "**llega**" solo si en casos parecidos llegó al menos 5 de cada 10; si no,
  "**puede llegar**", con el texto "en casos parecidos llegó 3 de cada 10; las demás
  pasaron al costado".
- Una llegada poco confiable se muestra, pero **no sube la calificación** de la localidad.
- Tornquist y Médanos tienen muchas "llegadas sin pronóstico" (55 de 75 y 62 de 99):
  ecos que se forman en el lugar o ruido del radar en esa zona. A revisar.

## Viento por radar: velocidad local y estable

El flujo óptico a veces da saltos imposibles (13/11 05:20 UTC: 35 → 105 km/h en 10 min).
Ahora:

- se usa la velocidad de los ecos a ≤ 20 km de cada localidad, no el promedio del radar;
- "posible severo" exige ≥ 75 km/h en **dos frames seguidos** (16/12/2023: 75–84 km/h);
- 60–75 km/h queda en "fuerte" (ráfagas fuertes posibles).

## Simulación del viernes 13/11/2026 (cadena completa)

119 frames del día local (00:10 a 23:30, hora de Argentina), todos los pasos de los
workflows sin errores. Cantidad de frames con cada calificación:

| Localidad | Calificación más alta | Lluvia máx. en 6 h (radar) |
|---|---|---|
| Bahía Blanca | Fuerte (viento) en 21 frames; nunca severa | 3,5 mm |
| Ingeniero White | Fuerte (lluvia / viento) | 13 mm |
| Punta Alta | **Posible severa (lluvia)** en 48 frames, granizo posible severo | **117 mm** |
| Médanos | Fuerte (lluvia, viento, granizo) | 16 mm |
| Tornquist, Cerri, Sierra de la Ventana | Fuerte como máximo | < 16 mm |

Antes de estos cambios Bahía Blanca habría quedado con "indicio de inundación"
(48 mm en 3 h por el píxel más intenso) y "viento posible severo" por la velocidad
promedio del radar: las dos cosas eran falsas para la ciudad.

## Pendiente de calibrar

- Lluvia real en Punta Alta y Bahía Blanca el 13/11 (corrección de la fórmula del radar crudo).
- Días con granizo o tormenta fuerte entre el 20/10 y el 19/11 (cortes de granizo con
  radar crudo: hubo ≥ 60 dBZ sobre la ciudad el 28/10, 4/11, 12/11 y 13/11).
- Validar los cortes de rayos con tormentas reales (la ventana de 20 min empezó hoy).

## Calidad de estaciones y calibración automática de la lluvia (19/11/2026, tarde)

Ni el radar ni las estaciones se creen a ciegas (pedido de Lucas: las estaciones de Bahía
están en el aeropuerto y en Grünbein, lejos del centro, y no sabemos si miden bien).

- **Puntos de control:** cada estación dentro de los 240 km del radar se compara con lo que
  el radar ve **justo encima de ella** (3 km), no con la ciudad. Así se calibra la fórmula
  del radar, que después sirve para cualquier localidad.
- **Control de calidad** (`scripts/estaciones/calidad_y_calibracion.py`, en cada ciclo):
  dato atrasado, valores imposibles, temperatura trabada 3 h, lluvia del día que baja, y
  desacuerdos con el radar (el radar ve ≥ 5 mm encima y la estación 0, o la estación ≥ 10 mm
  sin eco cerca). Con 2 desacuerdos en 14 días la lluvia de esa estación queda "dudosa".
  Una estación dudosa no entra en los índices ni en la calibración.
- **Registro de calibración:** una fila por hora y estación confiable a ≤ 100 km
  (`data/ia/calibracion/registro_lluvia.json`, 120 días).
- **Factor automático:** lluvia medida / lluvia del radar en las horas con lluvia. Se usa
  cuando junta ≥ 20 horas de ≥ 3 días distintos (`data/ia/calibracion/factor_lluvia.json`).
- **Factor inicial 0,4** (`config/calibracion_lluvia.json`): Punta Alta 13/11, 72 mm medidos
  contra ~190 mm del radar crudo.

Viernes 13/11 con la corrección (119 frames, sin errores): Bahía Blanca máx. 1,4 mm en 6 h;
Punta Alta "Posible severa (lluvia)" entre las 15:20 y las 16:00 (lo fuerte real fue de 14 a
15:30, con 35 mm). Con un solo factor el día da bien (~76 mm contra 72), pero el chaparrón
de la tarde queda corto (19 mm en 3 h contra 35): por eso hace falta la calibración con
estaciones, chaparrón por chaparrón.
