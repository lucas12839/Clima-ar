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
pysteps se agregó a `requirements.txt` (se instala en ~1 min en GitHub, probado).
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
