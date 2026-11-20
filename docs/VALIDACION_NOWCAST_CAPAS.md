# Validación del nowcast en capas (punto 5b, capa 6)

Fecha: 20/11/2026 (madrugada, hora de Argentina). Herramienta: `herramientas/simular_huecos.py`.

## Cómo se probó

Con el radar guardado (`data/radar/historico`, 26 días con lluvia entre el 24/10 y el 20/11/2026)
se simuló que SINARAME se atrasa: se toma una imagen (y las 2 anteriores) y se pronostica qué hay
10, 20, 30, 40, 60 y 90 minutos después, como si no hubiera llegado ninguna imagen nueva. Después
se compara con la imagen real de esa hora.

- **655 imágenes de partida** (una de cada dos imágenes con lluvia en el radar).
- Dónde se mide: las 7 localidades y 150 puntos de la región (cada 25 km, de 20 a 200 km del radar).
- Umbrales: 20 dBZ (lluvia) y 35 dBZ (tormenta).
- Puntaje (CSI): aciertos / (aciertos + falsas alarmas + omisiones). 1 = perfecto. Más alto = mejor.
- **"Hoy"** = lo que se mostraba como "ahora" hasta este cambio: la imagen vieja tal como llegó.

## Resultado: tormentas (35 dBZ), las 7 localidades

| Atraso del radar | 10 min | 20 min | 30 min | 40 min | 60 min | 90 min |
|---|---|---|---|---|---|---|
| Hoy (imagen vieja) | 0,478 | 0,393 | 0,348 | 0,231 | 0,175 | 0,165 |
| Nowcast en capas | **0,535** | **0,463** | **0,383** | **0,334** | **0,249** | **0,232** |
| Mejora | +12 % | +18 % | +10 % | +45 % | +42 % | +41 % |

Región (150 puntos), tormentas: 0,517 → **0,580** (10 min), 0,254 → **0,352** (40 min),
0,196 → **0,295** (60 min). Lluvia (20 dBZ) también mejora en todos los atrasos.

Detección y falsas alarmas en las localidades, tormentas, a 60 min: detecta el **42 %** (antes 29 %)
con **62 %** de falsas alarmas (antes 69 %). Mejora las dos cosas a la vez.

Por localidad (tormentas a 30 min): Bahía Blanca 0,375 → 0,389; Ingeniero White 0,220 → 0,361;
Punta Alta 0,391 → 0,500; Médanos 0,355 → 0,362; Cerri igual (0,448); Tornquist igual (0,356 → 0,354);
Sierra de la Ventana 0,190 → 0,200.

## Qué se probó y qué quedó (cada capa entra solo si mejora)

| Variante | Resultado | Decisión |
|---|---|---|
| Capa 1: imagen movida a la hora actual (2 imágenes) | mejor que "hoy" en la región en todos los atrasos | entra |
| Movimiento con 3 imágenes (punto 6) | un poco mejor que con 2 en casi todo | **entra** |
| Capa 2: tendencia del radar (crece/se debilita siguiendo al eco) | **peor** en casi todo (la tendencia de 10 min no se mantiene) | **queda apagada** |
| Capa 5: 9 escenarios (velocidad ±20 %, dirección ±20°), "hay eco si 3 de 9 lo dan" (hasta 40 min) y "2 de 9" (después) | la mejor en todos los atrasos | **entra** |

## Hallazgo: ecos fijos intermitentes cerca del radar

En las imágenes **sin lluvia**, Bahía Blanca tenía ≥ 20 dBZ en el 37 % de ellas (también Punta
Alta, Médanos y Cerri): son ecos del terreno que aparecen y desaparecen y que la máscara vieja
(`config/clutter_rma10.png`) no sacaba. Se agregó `config/clutter_intermitente_rma10.png`
(479 píxeles con eco en más del 5 % de las imágenes sin lluvia), generada con
`herramientas/generar_clutter_intermitente.py`. La usa el nowcast en capas; lo que se ve hoy
todavía no (se cambia cuando el nowcast en capas pase a mostrarse).

## pysteps (punto 6: "mejor tecnología", solo si mejora)

`herramientas/comparar_pysteps.py`, 268 imágenes de partida, mismos huecos y puntos.

| Tormentas, región | 10 min | 20 min | 30 min | 40 min | 60 min | Tiempo por ciclo |
|---|---|---|---|---|---|---|
| Hoy | 0,527 | 0,401 | 0,328 | 0,257 | 0,199 | - |
| **Nowcast en capas** | **0,586** | 0,474 | **0,406** | **0,357** | **0,301** | 0,7 s |
| pysteps Lucas-Kanade | 0,566 | 0,467 | 0,376 | 0,322 | 0,256 | 0,6 s |
| pysteps S-PROG | 0,518 | 0,420 | 0,335 | 0,286 | 0,221 | 0,9 s |
| pysteps ANVIL (crece/se debilita) | 0,553 | **0,479** | 0,393 | 0,338 | 0,274 | 1,3 s |

Nuestro nowcast en capas gana en la región en casi todo. ANVIL (el único que ve crecer y
debilitarse la lluvia) se acerca y en algunas localidades es algo mejor.

**Combinación (punto 6):** "hay eco si lo dan los escenarios o ANVIL". Se probó en dos muestras
distintas (~270 casos cada una):

| Tormentas | 20 min | 30 min | 40 min | 60 min |
|---|---|---|---|---|
| Localidades, muestra 1: capas / capas + ANVIL | 0,448 / 0,453 | 0,354 / **0,378** | 0,333 / **0,401** | 0,278 / **0,311** |
| Localidades, muestra 2: capas / capas + ANVIL | **0,527** / 0,507 | 0,414 / **0,425** | 0,350 / **0,369** | 0,286 / **0,305** |
| Región, muestra 2: capas / capas + ANVIL | 0,461 / **0,472** | 0,411 / **0,419** | 0,355 / **0,369** | 0,294 / **0,299** |

Decisión: **ANVIL entra desde 30 min** (a 10 y 20 min no aporta de forma pareja). Con su propio
movimiento (Lucas-Kanade de pysteps) rindió más que con el nuestro. Costo: ~2 s más por ciclo.
pysteps va en `requirements-nowcast.txt` y se instala solo en Intelligence (la cadena del radar no se demora).
Si pysteps falta o falla, siguen los escenarios solos y el archivo lo dice (`anvil.usado`).

## Validación automática (en vivo)

Cada ciclo guarda lo pronosticado para las próximas horas de imagen y, cuando llega la imagen real,
compara por localidad y en 150 puntos de la región, separado por atraso. Archivo:
`data/ia/verificacion_nowcast.json` (resumen también en `data/radar/nowcast_capas.json`,
`validacion_automatica`). Así se ve, con datos en vivo, si el nowcast en capas sigue ganando.

## Límites (dicho sin rodeos)

- Rayos, satélite y estaciones (capas 2, 3 y 4) **no tienen historia suficiente** para simular
  (el historial por tormenta existe desde el 19/11). Se calculan y se guardan en paralelo; su
  efecto se va a medir con la validación automática y con las simulaciones finales (regla 6).
- Con 90 min de atraso el pronóstico acierta poco (0,23): por eso la confianza baja de a poco y
  cuanto más viejo el radar más pesan rayos, satélite y estaciones (capa 3).

## Validación de todo el sistema (pedido de Lucas), 20/11/2026 ~03:00 hora de Argentina

Revisado en producción después de unir #53, #54 y #55:

| Pregunta | Resultado |
|---|---|
| ¿Las herramientas se pisan? | No. Radar e Intelligence escriben en el mismo grupo `climaar-production-writes`: corren de a una. Las herramientas nuevas solo **agregan** archivos (`nowcast_capas.json`, `verificacion_nowcast.json`, `modelos_region.json`, `data/ia/modelos/`) y no cambian ninguno de los que lee la pantalla. |
| ¿Alguna se bloquea? | No. Ciclo de Intelligence: ~3 min 30 s (antes ~2 min 30 s; tope de 15 min). Nowcast en capas ~9 s, modelos ~1 s (13-25 s cuando sale una corrida nueva del WRF, tope 5 min). Todas con falla segura. pysteps solo en Intelligence: la cadena del radar no se demora. |
| ¿Se pierden datos? | No. Todo se guarda en git en el paso "Persistir". El archivo de validación automática se achicó de 300 kB a ~70 kB por ciclo (sin sangría y con números enteros en la región). |
| ¿Están sincronizadas? | Sí. La integridad exige que el nowcast en capas sea de la misma imagen de radar, con la edad del radar bien calculada y ninguna fuente del futuro. Hora de cada fuente en `fuentes` (radar, rayos, satélite, estaciones, ambiente). Ciclos reales con integridad OK. |

## Corrector con rayos y satélite medidos + control de calidad del radar (20/11/2026, mañana)

Con el **historial real de GOES-19** de los mismos 26 días (`herramientas/historial_goes.py`, rama
`datos-goes-historial`) se simularon los huecos con radar, rayos y satélite juntos, usando solo lo
que ya estaría publicado a cada hora (satélite con ~8 min de demora, rayos de los últimos 10 min).

**Hallazgo: ecos falsos del radar.** Varias noches (26 y 30/10, 5, 7, 8 y 9/11) el radar mostraba
ecos de hasta 75 dBZ con el cielo despejado según el satélite (ej.: 8/11 00:10 hora de Argentina,
ninguna nube más fría que +2 °C en todo el radio). Es propagación anómala (el haz se curva hacia el
suelo con aire estable). Ahora el nowcast en capas borra los ecos sin ninguna nube encima
(`control_calidad_satelite`) y la simulación también los saca de la "verdad".
**Lo que se ve hoy en la pantalla todavía no tiene este control.**

Resultado (457 casos; CSI, más alto = mejor):

| Tormentas (35 dBZ), localidades | 10 min | 20 min | 30 min | 40 min | 60 min | 90 min |
|---|---|---|---|---|---|---|
| Hoy (imagen vieja) | 0,475 | 0,398 | 0,315 | 0,306 | 0,187 | 0,175 |
| Capas (antes del corrector) | 0,519 | 0,407 | 0,338 | 0,313 | 0,223 | 0,194 |
| **Capas + corrector** | **0,607** | **0,466** | **0,428** | **0,397** | **0,292** | **0,276** |

Lluvia (20 dBZ) en localidades: 0,342 → **0,483** a 30 min y 0,294 → **0,431** a 60 min.
Región: también mejora en todos los atrasos.

- **ANVIL:** con el corrector casi no aporta (gana en unos plazos y pierde en otros): queda
  **apagado** y pysteps ya no se instala. El código queda por si más adelante sirve.
- **Rayos:** marcan **tormenta eléctrica** como fenómeno aparte (`fenomenos.tormenta_electrica`).
  Solo agregan lluvia de tormenta si el satélite ve un tope de -25 °C o más frío encima.
- Regla que se probó y quedó apagada: bajar tormentas sin rayos (empeora a 10-40 min).
- Límite: la limpieza de la "verdad" usa el mismo criterio del satélite, así que parte de la mejora
  es justamente haber sacado los ecos falsos; eso es real (no había nube), pero se va a confirmar con
  la validación automática en vivo.

## La pregunta útil: "¿llega a mi localidad en la próxima hora?" (20/11/2026, 11:00 hora de Argentina)

`herramientas/simular_llegadas.py`, 458 casos (26 días), radar atrasado 10, 20 y 30 min, verdad
limpia de ecos falsos. Nowcast en capas con corrector (lo que corre hoy en paralelo):

| Próxima hora | Detecta (llegó y avisó) | Falsas alarmas | Acierto total* | Hora ±15 min |
|---|---|---|---|---|
| Tormenta, en la localidad | 48-53 % (antes 29-35 %) | 25-35 % (antes 33-41 %) | 91-92 % | 72-79 % |
| Tormenta, hasta 10 km | 63-64 % (antes 50-53 %) | 16-22 % (antes 38-46 %) | 86-87 % | 74-80 % |
| Lluvia, hasta 10 km | 68-73 % (antes 74-76 %) | **5-7 %** (antes 39-41 %) | 81-84 % | 80-86 % |

\* Incluye las horas en que no pasó nada y no se avisó: por eso es alto.

**Veredicto: el 5b NO queda cerrado para el objetivo de 80-100 %.** Mejoró mucho (sobre todo
menos falsas alarmas), pero la detección de tormentas en la próxima hora está en ~50 % (localidad)
y ~63 % (con 10 km de margen). Lo que falta son, sobre todo, tormentas que **nacen o crecen durante
la hora** (el radar viejo no las tiene).

## Por qué se escapaban las tormentas y cómo se arregló (20/11/2026, tarde)

**Análisis de las omisiones** (`herramientas/analizar_omisiones.py`, 26 días, 684 imágenes de
partida, 9.597 casos localidad × atraso). De las tormentas que llegaron a menos de 10 km y no
se avisaron:

| Qué había en el radar viejo | Omisiones | Aciertos |
|---|---|---|
| Tormenta (≥ 35 dBZ) a menos de 15 km | 20 % | 75 % |
| Tormenta a 15-30 km | 44 % | 17 % |
| Tormenta a 30-50 km | 22 % | 7 % |
| Solo lluvia cerca (creció durante la hora) | 10 % | 1 % |
| Nada a 30 km (nació o vino de muy lejos) | 5 % | 0 % |

**Lo contrario de lo que se pensaba:** 2 de cada 3 tormentas que se escapaban **ya estaban en el
radar a menos de 30 km**, casi siempre del lado de donde venía el movimiento (529 de 617). El
pronóstico las tenía, pero en el campo "3 de 9 escenarios" no llegaban: se desviaron,
aceleraron o se estiraron. Las que nacen o crecen durante la hora son 1 de cada 7.

**Rayos y aviso de formación (probados con las mismas filas):** hay rayos a menos de 30 km solo en
el 5 % de los casos; sumarlos casi no cambia nada (+0,2 puntos de detección). La señal de
formación por satélite (tope que se enfría de verdad, bajo −10 °C, con eco débil) agrega
detecciones pero con **dos falsas alarmas por cada acierto** (detección 69 % con 25 % de falsas
alarmas): rinde igual que agrandar la zona y no se conecta. Quedan como están (formación en
paralelo; rayos = tormenta eléctrica).

**Arreglo: envolvente de caminos** (`nc.campos_envolvente`). 25 caminos posibles (velocidad
0,6-1,4 × dirección ±40°) desde la imagen limpia, con el mismo corrector de satélite y rayos.
"Llega" si lo dan k de los 25. `herramientas/simular_envolvente.py`, los mismos 684 casos, radar
atrasado 10-30 min:

| Tormenta en la próxima hora, hasta 10 km | Detecta | Falsas alarmas | CSI | Hora ±15 min |
|---|---|---|---|---|
| Hoy (3 de 9) | 63,5 % | 16,2 % | 56,6 | 79 % |
| 3 de 25 | 67,3 % | 18,6 % | 58,3 | 83 % |
| **2 de 25 = "probable"** | **73,9 %** | **23,7 %** | **60,1** (el mejor) | 81 % |
| **1 de 25 = "posible"** | **80,8 %** | **33,8 %** | 57,2 | 79 % |

| Lluvia en la próxima hora, hasta 10 km | Detecta | Falsas alarmas | CSI |
|---|---|---|---|
| Hoy | 71,5 % | 5,9 % | 68,4 |
| **"probable" (2 de 25)** | **82,0 %** | **10,0 %** | 75,2 |
| **"posible" (1 de 25)** | **87,1 %** | **14,4 %** | 76,0 |

En la localidad misma (sin margen) la tormenta queda en 70 % / 43 % ("probable") y 77 % / 55 %
("posible"): acertar el círculo exacto de una localidad con una hora de anticipación es mucho
más difícil que acertar a 10 km.

**Límite (dicho sin rodeos):** con estas fuentes, detectar más tormentas cuesta más falsas
alarmas; el CSI (que junta las dos cosas) no pasa de ~60 en ninguna variante probada
(satélite, rayos, radar más lejos, más caminos). Para pasar de 80 % sin subir las falsas
alarmas haría falta otro tipo de dato (Doppler del RMA10, más estaciones).

**Cómo queda** (en paralelo, no cambia lo que se ve): `nowcast_capas.json` →
`localidades.<id>.llegada_proxima_hora.{lluvia,tormenta}.{probable,posible}.{localidad,hasta_10km}`
= minutos desde ahora hasta la primera llegada (`null` = no llega en la hora), y
`tormenta_electrica_ahora` (rayos a ≤ 10 km en los últimos 10 min). La validación automática en
vivo guarda cada pronóstico de llegada y lo cierra con las imágenes reales de su hora
(`verificacion_nowcast.json` → `llegadas_acumulado`; resumen en
`validacion_automatica.llegadas_proxima_hora`).

**Con el radar muy atrasado (45, 60 y 90 min, mismos 684 casos):** la envolvente sigue
mejorando. Tormenta hasta 10 km: hoy 50 % detectada / 29 % falsas alarmas (CSI 41,9);
"probable" 62 % / 35 % (CSI 46,7); "posible" 73 % / 45 % (CSI 46,1). Lluvia hasta 10 km: hoy
65 % / 9 %; "probable" 77 % / 13 %; "posible" 85 % / 17 %. Con más de 90 min de atraso no está
probado y el archivo lo marca (`llegada_proxima_hora.fuera_de_lo_validado`).
