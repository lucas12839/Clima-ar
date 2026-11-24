# ClimaAR — estado y plan (actualizado 20/11/2026, tarde, hora de Argentina)

Leer junto con `CLAUDE.md` (reglas fijas del repo).

## Cómo trabajar con Lucas
- Todo en español, sin términos técnicos en inglés (o explicados).
- Respuestas breves: qué se hizo, qué falta, qué decidir.
- Horarios siempre en **hora de Argentina (UTC−3)**.
- Claude genera, arregla, prueba y sube los cambios (rama + pull request + merge).
- El VPS no se actualiza solo desde GitHub: si hay que tocarlo, darle el comando a Lucas.

## Pasos del 23/11/2026, noche (orden de Lucas: 1 reloj externo, 2 servidor, 3 seguimiento, 4 estaciones O/SO, interfaz al final)
- **1 Reloj externo: hecho.** cron-job.org "ClimaAR vigilante" cada 10 min, clave `climaar-reloj-externo`
  (solo Actions, vence 21/02/2027, igual que la del detector). Prueba 204 y corrida en GitHub a las 20:37.
- **2 Servidor: hecho.** El VPS se actualiza solo desde main (`climaar-actualizar.timer`). El servicio
  instalado queda con `.venv-ia` y tiempos 10/10 (distinto del de `deploy/`, a propósito: no tocar).
- **3 Seguimiento, divisiones y uniones: hecho.** `herramientas/medir_division_union.py` (8 días).
  La mayoría de lo que el seguimiento marca como división o unión son pedazos chicos que aparecen al lado
  o parpadeos del detector; las de verdad (los pedazos suman lo que había y caen dentro de la tormenta de
  antes) son ~24 por día. En esas, el centro del pedazo saltaba y el tamaño cambiaba aunque el sistema no.
  Ahora se conserva el centro ponderado del sistema: salto de velocidad 67-70 → 24 km/h; cambio de tamaño
  falso +66 % (unión) / −43 % (división) → ~0 % (ya no "crece" ni "se debilita" por eso); la hija hereda
  dirección y pasado. Las trayectorias que se muestran (5b, flujo) no usan esto y no cambian.
- **Falla del 23/11, 20:41-21:08:** la actualización instaló Playwright 1.65.2 sin su navegador; el
  detector falló cada consulta 27 min y seguía dando latido. Arreglado a mano (`playwright install
  chromium` como `climaar`). Ahora: el actualizador baja el navegador cuando cambia Playwright; si el
  detector falla ~10 min seguidos deja de dar latido (toma el respaldo de GitHub) y publica el error, y
  la vigilancia avisa a los 5 min con la causa y el comando.
- **4 Estaciones al oeste/suroeste: cerrado sin cambios (decisión de Lucas).** Relevado el 23/11: red
  BCP completa (ids 1-130, ninguna activa al O/SO; Teniente Origone y Perú muertas), CWOP (nada),
  Netatmo (una sola, en Bahía y desactualizada), SMN Río Colorado (horaria, ya descartada). Lucas no
  tiene a quién pedirle que active CWOP en su estación. Fuera de 240 km no sirven (sin control con
  radar, sin tormenta asociada, horas de anticipación). No quiere aviso automático de estaciones nuevas.
  La zona O/SO queda con radar, satélite y modelo europeo. No volver a buscar lo mismo.
- **Sigue: interfaz (Render/Netlify), cuando Lucas diga.** Rama para borrar a mano: `prueba-estaciones-oeste`.

## Reparación de la sincronización de la cadena (23/11/2026, 17:00 hora de Argentina)
Revisión de punta a punta con las corridas reales de GitHub. Lo que se encontró y se arregló:
- **La IA rechazaba tormentas recién nacidas**: la validación comparaba la trayectoria del 5b con el
  predictor viejo ("sin movimiento" en la primera imagen). Hoy dejó sin análisis las imágenes de
  14:30 (2 corridas) y de 21:10/21:20 del 22/11. Ahora se valida contra el 5b.
- **Un corte de red de GitHub tiraba el guardado** (sin reintento al bajar main). Radar, IA y CWOP
  reintentan 5-6 veces con espera creciente.
- **CWOP compartía la fila de escritura** con Radar e IA (el tercero que llega cancela al que espera):
  sale de la fila (solo escribe su archivo). Una prueba impide que otro reloj vuelva a entrar.
- **La IA se dispara después de guardar el radar** (antes quedaba esperando y podía correr con un
  frame que no llegó a main).
- **Redespacho del radar: 6 min** (antes 10) y solo si no hay un Radar en marcha.
- **Error de la trayectoria que se ve (5b), medido en vivo**: `data/ia/verificacion_trayectorias.json`;
  el control de salud avisa si a 10 min supera 9 km (medido en pruebas: 4,5). Antes se medía el viejo.
- **La vigilancia avisa** si el último radar queda sin análisis o si la IA falla 2 veces seguidas.
- **VPS**: el actualizador baja solo lo último (antes, toda la historia cada 10 min).
- Control de salud: el total de estaciones cuenta solo las activas (11), no las dadas de baja.
- **Para Lucas (VPS)**: el servicio instalado tiene `CLIMAAR_REDESPACHO_MIN=10`; cambiarlo a 6 en
  `/etc/systemd/system/climaar-detector.service` (o dejarlo: con 10 también funciona).
- **Para decidir**: Pigüé llega tarde seguido (20-70 min) y queda afuera en casi la mitad de los ciclos.

## Mejoras 1 a 4 (23/11/2026, noche) — `docs/MEJORAS_23_11.md`
- Lluvia: calibración con el historial oficial del SMN (6 h). Mes casi seco: 6 ventanas con lluvia,
  el radar vio el 100 %; factor medido 0,25-0,29 contra 0,40 en uso: no alcanza para cambiarlo. Ahora
  se junta solo dos veces por día (`climaar-calibracion.yml`). Viento y granizo: sin datos oficiales.
- Falsas alarmas (27 %): 8 de cada 10 son casi aciertos (pasó a 10-25 km o llegó un poco más débil).
  Ningún filtro mejora sin perder detección: queda igual.
- Trayectoria: en vivo coincide con la simulación del mismo día; mezclar con la velocidad propia de
  cada tormenta no mejora: queda igual (4,5 / 9,5 / 14,7 km).
- 13/11 Punta Alta por la cadena completa: avisó ~13 min antes, lluvia fuerte y posible severa con
  granizo probable (cayó chico). Encontró y se arreglaron 3 errores ocultos que cortaban todo el ciclo.

## Re-auditoría de la cadena (23/11/2026, 18:00 hora de Argentina)
Con las 1000 corridas reales de GitHub desde el 22/11 y los avisos de las corridas "exitosas":
- **Alertas que nadie veía**: el control de salud escribía alertas en `salud_cadena.json` (estaciones
  sin datos: 147 ciclos; VPS sin latido: 36) y solo quedaban ahí. Ahora guarda desde cuándo sigue cada
  una y la vigilancia abre el aviso al celular si sigue más de 20 min.
- **El Radar se contaba a sí mismo como "frame esperando"** (la lista de GitHub tarda en actualizarse)
  y no disparaba la IA: 1 vez en 60 (frame 15:50, 8,6 min tarde). Arreglado.
- **Pasos "en paralelo" que fallan en silencio**: el control de salud revisa que los 10 resultados del
  ciclo se hayan regenerado y sean de la misma imagen. Aplicado a 238 ciclos pasados: ninguno falló.
- Estaciones BCP: todas a la vez y con un reintento (antes de a una: 26 s por ciclo).
- Revisado y bien: flujo óptico "no disponible" = sin ecos o radar viejo (correcto); satélite
  sincronizado con la imagen; una sola hora de radar en todos los archivos; un solo escritor por
  archivo de estado; numeración y seguimiento continuos.
- No depende de nosotros: SINARAME no publicó las imágenes de 11:30 y 16:40 (error 404).

## Decisiones de Lucas
- Sin modelos entrenados, todo por reglas y física (ver `CLAUDE.md`).
- **Sin alertas de colores ni avisos a corto plazo.** Los avisos oficiales son del SMN.
- **No existe la categoría "destructivo".** Lo máximo es "Severa".
- Escala por fenómeno, **cinco niveles, sin "muy fuerte"**: de fuerte se pasa a severo.
  Los cortes se revisan más adelante.
- Render (API) y Netlify (pantalla) están desactualizados a propósito: Lucas los
  actualiza al final, cuando la estructura esté sólida.
- El Doppler viene después.

## Terminado y verificado
- Plan original completo (1 a 7): sin aprendizaje automático, estructura limpia, reloj del
  radar, pronóstico por flujo óptico, validación con eventos, datos podados.
- Velocidades imposibles eliminadas (tope físico 120 km/h).
- Escala de tipo de tiempo 0–8 con amenazas (granizo calibrado; viento e inundación como
  "indicio en evaluación"), estaciones conectadas, márgenes 15/40 min, refresco cada 5 min.
- Modo "sin actividad" en toda la cadena.
- PR #29: se recuperan los frames que la página de SINARAME no muestra (se perdía 1 de 4).
- PR #30: el pronóstico ECMWF se reutiliza si tiene menos de 30 min; espera máxima 20 s.
- PR #31: el ciclo ya no se corta si GOES-19 no responde. Simulador de eventos.

## Simulación de los temporales (cadena completa, como si pasaran ahora)
173 imágenes sin errores en ningún paso. Detalle en `docs/VALIDACION_EVENTOS.md`.

| Temporal | Se acerca | Sobre la ciudad | Qué mostró ClimaAR |
|---|---|---|---|
| Viento 16/12/2023 (destructivo) | 15:52 | 16:19 | indicio de viento desde 15:43 (llega 15:50: ~29 min antes); calificación quedó en "Tormenta fuerte" |
| Granizo 2/2/2025 | 14:32 | 14:49 | granizo 14:58, ya sobre la ciudad |
| Inundación 7/3/2025 | 23:31 del 6/3 | 22:46 del 6/3 | indicio de inundación desde 06:26 del 7/3 |

Herramienta: `herramientas/simular_evento_real.py` (necesita las imágenes del SMN de cada
evento: Lucas tiene los zips `Temporal_viento`, `Temporal_granizo`, `Temporal_inundacion`).
Lucas dijo que por ahora no hacen falta más simulaciones.

## Hecho el 19/11/2026 a la mañana (PR #33 y siguiente) — ver `docs/INDICES_Y_LOCALIDADES.md`
- **Punto 1 terminado: índices por fenómeno.** Lluvia, viento, granizo y tormenta eléctrica
  por separado, combinación real ("Viento fuerte, sin lluvia"), calificación general = el más
  alto ("Fuerte (viento)"). Lucas eligió: grado 4 sin estación = **"Posible severa"**.
- **Localidades por separado** (pedido de Lucas por el 13/11, cuando en Bahía llovió poco y en
  Punta Alta más de 70 mm): Bahía Blanca, Ingeniero White, General Daniel Cerri, Punta Alta,
  Tornquist, Médanos, Sierra de la Ventana (Lucas sacó Monte Hermoso, Dorrego y Pehuen-Co).
- Lluvia por localidad = **promedio del área** (no el píxel más intenso).
- **Confiabilidad de cada llegada** con 30 días de radar: después de 20 min solo llegó ~1 de
  cada 3; casi siempre pasó al costado. Se dice "llega" o "puede llegar", y una llegada poco
  confiable no sube la calificación.
- Viento severo por radar: velocidad local, ≥ 75 km/h en dos frames seguidos.
- **GOES-19:** rayos de los últimos 20 min por localidad con salto brusco (antes: un archivo
  de 20 s); topes de nube (canal 13) por localidad con enfriamiento; topes ≤ −50 °C dan
  "lluvia probable" cuando no hay radar. Funcionando en producción desde las 08:47.
- **Modelo europeo en altura** (850/700/500 hPa): cizalladura, gradiente térmico y aire seco
  → "ambiente favorable a tormentas severas". En producción desde las 09:12.
- Tendencia de las tormentas a ≤ 20 km de cada localidad (se intensifican / se debilitan).
- Simulación del viernes 13/11 (119 frames, cadena completa, sin errores): Bahía Blanca nunca
  severa (3,5 mm en 6 h); Punta Alta "Posible severa (lluvia)" con 117 mm en 6 h.

### Revisión de 30 días (solo radar, `herramientas/revisar_30_dias.py`)
Días con grado 4 ("posible severa") por localidad, del 20/10 al 19/11:
- Bahía Blanca: granizo 28/10, 4/11, 12/11. Nunca lluvia ni viento severo.
- Punta Alta: lluvia 12/11 y 13/11; granizo 28/10, 12/11, 13/11.
- Médanos: lluvia 3/11, 6/11, 12/11; granizo 28/10, 2/11, 12/11; viento 12/11.
- **Tornquist: granizo en 10 días** (30/10 al 9/11) — sospechoso: tormentas sobre las sierras
  o efecto del terreno en el radar. A confirmar con Lucas.

### Hecho a la tarde
- Control de calidad de estaciones y calibración automática de la lluvia con estaciones
  confiables (ver `docs/INDICES_Y_LOCALIDADES.md`). Factor inicial 0,4 por Punta Alta 13/11
  (72 mm reales, 35 mm entre 14 y 15:30; en Bahía casi no llovió: la tormenta se desvió).
- Pendiente: Lucas quiere corregir los umbrales de tormentas (próximo paso).
- `estado_fuentes` en `resultado_final.json`: edad de radar, satélite, rayos, modelo y
  estaciones, cuáles están al día y si ClimaAR quedó "a ciegas" (ninguna observación al día).
- Pendiente: enlace de la estación de MeteoPunta en Punta Alta para sumarla como punto de control.

### Red CWOP (estaciones ciudadanas abiertas)
- Única red de estaciones caseras que se puede leer de forma automática con permiso (modo solo
  escucha de APRS-IS). Weather Underground, Weathercloud, Ecowitt y WeatherLink no lo permiten
  sin la clave de cada dueño.
- Conectada **LU4DRH-11** (Ecowitt, 6 km al noroeste del centro): transmite cada 10 min; frente
  a BCP Bahía Centro da +1,5 °C, viento y lluvia parecidos. Workflow `climaar-cwop.yml` cada
  10 min; entra en señales, puntos de control y control de calidad como cualquier estación.
- Descartadas por ahora: FW6154 y EW9010 (~70 km NE), sin transmisiones en 30 min.

### Datos reales verificados (`config/eventos_verificados.json`)
- 13/11: Punta Alta 76,7 mm (80,4 en dos días) con **granizo chico**; Bahía 0 a 0,5 mm;
  Pehuen-Co y Monte Hermoso > 50 mm; Dorrego > 60 mm (MeteoPunta). Confirma el factor 0,4
  (191 mm del radar crudo × 0,4 = 76 mm). El granizo "posible severo" de ClimaAR fue
  exagerado: el corte de 60 dBZ crudo es bajo. A corregir con los umbrales de Lucas.

### Preguntas para Lucas (calibración)
1. ¿Hubo granizo o tormenta fuerte el 28/10, 4/11, 12/11 o 13/11 en Bahía Blanca? ¿Y en
   Tornquist entre el 30/10 y el 9/11?
2. Viernes 13/11: ¿cuánto llovió exactamente en Punta Alta y en Bahía Blanca, y a qué hora fue
   lo fuerte? (el radar crudo dio ~190 mm en Punta Alta: parece sobreestimar).

## Punto 5 del plan terminado EN PARALELO (20/11/2026, 00:30 hora de Argentina) — esperando a Lucas para el 6
- **Satélite** (`goes19/formacion.py`, en `goes19_state.json` → `formacion_satelite`, versión
  1.8): el ciclo lee además el canal 13 de 10 y 20 min antes (mismo recorte, ~0,3 s cada uno).
  Junta RDT (objetos seguidos), UW-CI (enfriamiento que no confunde movimiento con
  crecimiento) y SATCAST (cruce de 0 °C, enfriamiento en 10 y 20 min, vapor − infrarrojo):
  cada píxel se compara con el **más frío a 12 km (20 km) en la imagen anterior**; si al lado
  hay nube igual de fría que no creció es **borde de yunque o banda alta** y se descarta.
  Por objeto: tope, enfriamiento, hielo, altura, vapor − infrarrojo, rayos, posición en el
  suelo (paralaje).
- **Aviso** (`scripts/aviso_formacion.py`, paso nuevo de Intelligence): control con la imagen de
  radar (sin contar ecos fijos) y los núcleos: si ya hay ≥ 35 dBZ es "tormenta que crece"
  (punto 4), si no es "en formación". Grados **calibrados con la simulación**: 0 tope sobre
  −10 °C o tope muy frío sin nada de precipitación a ≤ 160 km (nube alta); 1 vigilancia;
  2 probable (eco débil 20-35 dBZ, rayo, o tope −20/−25 °C con hielo cerca del radar);
  3 inminente (eco débil o rayo con tope bajo −20 °C). Ambiente estable o frente: máximo 1.
  Seguimiento entre ciclos con número; **ventaja** = minutos entre el primer aviso y el eco de
  tormenta; sin eco en 60 min = probable falsa alarma; cada tormenta nueva del radar se anota
  con o sin aviso previo. Salida `data/ia/formacion_tormentas.json` (`modo:
  en_paralelo_no_se_muestra`, la integridad lo exige) e historial `data/ia/formacion/`.
- **Simulación con datos reales** (`herramientas/simular_formacion.py desde hasta --rayos`,
  imágenes de `data/radar/historico` + NOAA): 19/11, 13/11 y 3/11 (este último no se usó para
  calibrar). Grado ≥ 2: acierto 24 % / 85 % / 58 %, ventaja mediana ~20 min (hasta 60-70);
  grado 3: 67 % / 79 % / 68 %. Vigilancia sola: 13-49 % (es solo "atención"). Del total de
  tormentas nuevas del radar se anticiparon 9-30 %: las que nacen pegadas a otra tormenta o
  bajo su yunque el satélite no las ve (límite físico). Los números de la simulación
  arman núcleos desde la imagen (≥ 40 dBZ), así que las "tormentas nuevas" están algo infladas.
- Arreglos del mismo PR: **velocidad con varias imágenes** (`nowcast_flujo_optico.py`): si un
  par de imágenes da una velocidad muy distinta de los dos pares anteriores, se usa la mediana
  (19/11 23:10: un par dio 58 km/h y los vecinos 22-25; ClimaAR mostró "viento fuerte" por
  eso). Prueba `explicit_no_motion_state` ya no depende de los datos en vivo.
- **Granizo que se muestra (decidido con Lucas el 20/11, #49):** "probable" o más que sale
  solo del radar baja un grado si el satélite y el ambiente están disponibles y nada lo
  confirma (topes ≤ −50 °C o enfriándose ≥ 8 °C/10 min, rayos frecuentes o salto, ambiente
  favorable). Nunca saca el aviso; sin datos no baja. Caso 19/11 23:20: 56 dBZ, CAPE 0, tope
  −37 °C, casi sin rayos → "posible". El reemplazo por el puntaje sigue para el final.

## Punto 4 del plan terminado EN PARALELO (19/11/2026, 22:30 hora de Argentina) — esperando a Lucas para el 5
- `scripts/puntaje_gravedad.py` (paso nuevo de Intelligence, antes de la integridad): por
  tormenta, potencial 0-1 y grado (0, 2 posible, 3 probable, 4 severo) de **granizo, ráfagas y
  lluvia intensa**, estilo NowCastMIX (reglas graduales, sin entrenamiento) con radar (p99 de
  dBZ, velocidad), satélite (tope frío, tope sobresaliente, tope sobre 0 °C, enfriamiento),
  rayos (salto 2σ, frecuencia, rayos chicos) y ambiente ECMWF (CAPE, cizalladura, gradiente,
  aire seco, 0 °C, agua precipitable).
- Reglas: sin núcleo fuerte en el radar no hay granizo/ráfagas; **"severo" exige confirmación
  de satélite o rayos** (el radar solo llega a "probable": caso Punta Alta 13/11 → "posible");
  eco fuerte + tope > −20 °C + 0 rayos = **probable eco no meteorológico** (sierras/Tornquist,
  cerca del radar); 0 °C muy alto baja el granizo; si falta una fuente baja la confianza.
- **EN PARALELO (regla 5): NO cambia lo que se ve.** Salida `data/ia/puntaje_tormentas.json`
  (`modo: en_paralelo_no_se_muestra`, la integridad exige ese modo) y comparación por
  localidad con lo que mostró ClimaAR en `data/ia/puntaje/AAAA-MM-DD.jsonl`. Pasar a mostrarlo
  es decisión de Lucas después de las simulaciones.
- Prueba real (21:50 hora de Argentina): 10 tormentas comunes, ninguna con granizo/ráfagas; el eco de 62 dBZ
  cerca del radar marcado como sospechoso.

## Punto 3 del plan terminado (19/11/2026, 22:00 hora de Argentina) — esperando a Lucas para el 4
- `goes19/rayos_tormenta.py`, dentro de `rayos` de cada tormenta en `por_tormenta`:
  - **asignación exclusiva:** cada rayo cuenta para una sola tormenta (la más cercana en el
    minuto del rayo, con su movimiento); nunca rayos posteriores a la imagen de radar;
  - **salto de rayos 2σ** (Schultz 2009, 2011): 14 min hasta la imagen de radar en tramos de
    2 min; salto si el último cambio supera 2 desvíos de los 5 anteriores con ≥ 10 rayos/min.
    Se informa `nivel_sigma` para calibrar el umbral de 10 (puede ser alto para GLM);
  - **tamaño de rayos:** área mínima, mediana y fracción de rayos chicos (< 300 km², guía GLM de
    NOAA), energía total.
- GLM: la ventana de la imagen de radar pasó de 10 a 14 min (una sola descarga con la de
  localidades); se lee área y energía de cada rayo. El conteo de 10 min se mantiene igual.
- Casos guardados: tasas de 2 min, salto, nivel, área mínima y fracción de chicos.
- Integridad: los rayos de cada tormenta terminan en la imagen de radar.
- Prueba real 00:20 UTC (21:20 hora de Argentina): 14 min completos (42 archivos), áreas 70-900 km²,
  sin saltos (una tormenta subió 2,5 desvíos con 3 rayos/min). Dos ecos fuertes cerca del radar
  (54 y 62 dBZ) con 0 rayos y topes de −16 °C: probables ecos no meteorológicos (punto 4).

## Regla 7 (casos guardados) hecha (19/11/2026, 21:40 hora de Argentina) — sigue el punto 3
- `goes19/historial.py`: en cada ciclo con tormentas, una línea por imagen de radar en
  `data/goes19/historial/AAAA-MM-DD.jsonl` (día UTC) con satélite, rayos y señales de cada
  tormenta (~0,3 KB por tormenta). Una sola vez por imagen; se reemplaza solo si después llegan
  más productos del satélite. 400 días. Nunca corta el ciclo. `leer(desde, hasta)` para
  simulaciones. El workflow Intelligence agrega la carpeta al guardar resultados.

## Punto 2 del plan terminado (19/11/2026, 20:50 hora de Argentina) — esperando a Lucas para el 3
- **Señales de tormenta severa por satélite**, por tormenta (`senales_severas` dentro de cada
  tormenta en `por_tormenta`), con la misma sincronización, movimiento y perspectiva del punto 1:
  - topes sobresalientes (textura del infrarrojo, Bedka 2010: ≤ −58 °C, mínimo local a 4 km,
    yunque frío a 6-10 km, ≥ 6,5 °C más frío que el yunque);
  - vapor de agua menos infrarrojo (canal 8 − canal 13, mismo barrido, píxel a píxel);
  - altura del tope con detalle de 2 km (`ABI-L2-ACHA2KMF`; NOAA la publica ~10 min más tarde:
    si no llega sincronizada se usa la de 10 km);
  - fase del tope (`ABI-L2-ACTPF`: agua, sobreenfriada, mixta, hielo);
  - tope por encima del nivel de congelamiento.
- **Nivel de congelamiento arreglado:** Open-Meteo `/v1/ecmwf` no trae `freezing_level_height`
  (llegaba vacío, así que la regla "isoterma alta: el granizo chico se derrite" de
  `indices_fenomenos.py` nunca actuaba). Ahora se calcula del perfil ECMWF (superficie y
  925-500 hPa con su altura real) en `meteorologia_fusion.py`. Desde ahora esa regla funciona
  (actúa con el 0 °C a ≥ 4500 m, típico de pleno verano).
- Prueba real 23:10 UTC (20:10 hora de Argentina): 7 tormentas a 160-220 km, topes 100 % hielo, 9-11 km,
  5,8-7,5 km sobre el 0 °C (2830 m), sin topes sobresalientes (ninguna llegó a −58 °C).
- Estado GOES versión 1.7.

## Punto 1 del plan terminado (19/11/2026, 20:15 hora de Argentina)
- **Error grave corregido:** las fórmulas que pasan cada píxel de GOES-19 a latitud/longitud
  estaban mal (signos y senos/cosenos invertidos). Hasta este cambio, los datos "a 240 km de
  Bahía" salían del océano Pacífico (~32°S 85°O, a ~2200 km): topes del canal 13 (y por
  localidad), vapor de agua, altura y presión de topes. Rayos y vientos estaban bien (traen
  su propia latitud/longitud). Ahora: fórmulas oficiales de NOAA, verificadas con el ejemplo
  del manual y con PROJ. Los estados anteriores a la versión 1.6 no se comparan.
- **Lectura parcial:** solo se leen los pedazos del archivo de la zona (idéntico a bajar el
  archivo entero, comprobado píxel por píxel en GitHub). Si falla, baja el archivo completo.
  Recorte exacto del círculo (antes la ventana era de ~2000 km).
- **Sincronización:** todos los productos del mismo barrido que el canal 13; hora en que el
  satélite pasó por Bahía (no el fin del barrido); nunca datos posteriores a la imagen de radar;
  hasta 20 min antes, si no, no se usa por tormenta.
- **Por tormenta** (`por_tormenta` en `data/goes19/goes19_state.json`): por cada núcleo del
  radar, topes (canal 13), vapor (canal 8), altura y presión de topes, enfriamiento (misma
  tormenta) y rayos de los 10 min que terminan en la imagen de radar. La tormenta se mueve con
  su desplazamiento hasta la hora del satélite y se corrige la perspectiva (un tope de 10 km
  aparece ~11 km al sureste).
- Control de integridad: datos por tormenta solo de tormentas del radar, sincronizados, nunca
  del futuro. GOES puede faltar sin cortar el ciclo.
- Herramienta: `herramientas/verificar_goes19.py` (en GitHub, con internet).
- Prueba real 22:40 UTC (19:40 hora de Argentina): 4 tormentas fuertes con topes de −45 a
  −49 °C, 9-10 km y 9 a 46 rayos; dos ecos de 55-56 dBZ con cielo despejado y sin rayos
  (probables ecos no meteorológicos: útil para el punto 4). Estado GOES en 7,5 s.

## Siguiente — orden de trabajo vigente: `docs/INSPECCION_FUENTES.md` (19/11/2026, 17:00)
Inspección completa de GOES-19, pronóstico y fuentes, con reglas de confiabilidad y un plan de 11 puntos por prioridad.
Lo de abajo queda como detalle de los puntos 9 a 11 de ese plan (el viejo "D. Más estaciones" quedó descartado).

- **A. Nunca a ciegas:** falta probar el modo sin radar con una tormenta real (rayos + topes
  + estaciones); vientos satelitales todavía solo como dato general.
- **B. Mejor pronóstico:** horizonte 2–3 h con varios escenarios, formas (líneas, ecos en
  arco), crecimiento por celda (hoy: tendencia a ≤ 20 km).
- **C. Calibración continua:** con las respuestas de arriba, ajustar la fórmula de lluvia del
  radar crudo y los cortes de granizo; repetir `calibrar_llegadas.py` cada mes.
- **D. Vigilancia:** VPS que se actualice solo, monitoreo de cada fuente, Doppler más adelante.
- Hecho: `getdata` de Pillow reemplazado (mismo resultado comprobado en 40 frames).


## Contexto rápido para un chat nuevo (20/11/2026, 00:45 hora de Argentina)
- Lucas trabaja desde el celular, no sabe inglés ni validar pull requests: Claude prueba,
  valida, abre el PR y lo une a main (permiso dado). Todo en español, horarios en hora de
  Argentina. Avisar al terminar cada punto y **no avanzar al siguiente sin que Lucas lo diga**.
- **Plan único y vigente: `docs/INSPECCION_FUENTES.md`** (objetivo: sistema confiable que en el
  futuro pueda avisar tormentas severas). 8 reglas de confiabilidad + 11 puntos. La corrección
  de umbrales (Punta Alta, Tornquist) quedó dentro del punto 4.
- **Actualización 20/11/2026, ~03:15 hora de Argentina (Lucas durmiendo, trabajo en segundo
  plano autorizado):** terminados **5b (#53), 6 (#54) y 7 (#55)**, todos EN PARALELO (no
  cambian lo que se ve) y validados (`docs/VALIDACION_NOWCAST_CAPAS.md`). Archivos nuevos:
  `data/radar/nowcast_capas.json`, `data/ia/verificacion_nowcast.json` (validación automática en
  vivo), `data/ia/modelos_region.json`, `data/ia/modelos/`. Sigue el **punto 8** (esperar a Lucas).
- **20/11 mañana:** máscara de clutter que no borra tormentas reales (#57); corrector con rayos y
  satélite medidos + control de calidad del radar con el satélite (ecos falsos de noche); ANVIL
  apagado (no aporta con el corrector). Lucas pidió después medir con la pregunta útil ("¿llega a
  mi localidad en la próxima hora?") para ver si el 5b queda cerrado: HECHO (#59,
  `herramientas/simular_llegadas.py`). **El 5b NO queda cerrado**: tormenta en la próxima hora,
  detección ~50 % en la localidad y ~63 % hasta 10 km (falsas alarmas 16-35 %); lluvia hasta
  10 km: detección ~70 %, falsas alarmas 5-7 %. Objetivo de Lucas: 80-100 %.
- **20/11 tarde (Lucas: "reparar y validar 5b, después el 7"):**
  - **5b:** las omisiones NO eran tormentas que nacían: 2 de cada 3 ya estaban en el radar a
    menos de 30 km y no llegaban en el escenario central (`herramientas/analizar_omisiones.py`).
    Rayos y aviso de formación casi no aportan para esto (probado). Arreglo: **envolvente de 25
    caminos** (`nc.campos_envolvente`) con dos grados por localidad, `probable` (2 de 25) y
    `posible` (1 de 25), en `nowcast_capas.json` → `llegada_proxima_hora` (en paralelo) +
    validación en vivo de las llegadas. Tormenta hasta 10 km: 63,5 % → **74 % "probable"**
    (24 % falsas alarmas) y **81 % "posible"** (34 %); lluvia hasta 10 km: **82 % / 87 %**
    (10 % / 14 %). Detalle: `docs/VALIDACION_NOWCAST_CAPAS.md`. Límite: más detección cuesta
    más falsas alarmas (CSI tope ~60 con estas fuentes).
  - **7:** error corregido en `modelos_region.py` (la lluvia de 10 min del WRF se leía como
    acumulado); medido con 26 días: el WRF no acierta cuándo ni dónde llueve por localidad
    (~1 de cada 3 horas con lluvia), sirve como ambiente. `docs/VALIDACION_MODELOS.md`.
    Historial en la rama `datos-wrf-historial`.
- **5b CERRADO (20/11, 16:25, Lucas lo deja a criterio de Claude):** cerrado con la envolvente
  (#61). Primer ciclo en vivo OK (19:00 UTC, tormentas con 413 rayos). Se sigue midiendo solo en
  vivo (`verificacion_nowcast.json` → `llegadas_acumulado`); se revisa en las simulaciones finales.
- **Auditoría de la cadena (20/11, 17:00-19:00, pedido de Lucas antes del 8):** 12 arreglos y
  control de salud en cada ciclo (`data/ia/salud_cadena.json`). El más importante: el
  seguimiento se reconstruía desde cero en casi cada imagen (las tormentas cambiaban de número)
  porque el capturador escribía su estado. Detalle: `docs/AUDITORIA_2026-11-20.md`.
- **Punto 8 TERMINADO en paralelo (20/11, ~18:15):** ambiente medido por GOES-19
  (`data/ia/ambiente_satelite.json`), validado con 25 días (`docs/VALIDACION_AMBIENTE_SATELITE.md`).
- **Paso 1 HECHO (20/11, noche, decisión de Lucas):** el 5b pasa a lo que se ve: llegadas a la zona
  de cada localidad (tormenta a <= 10 km, probable) y trayectorias por tormenta (16-25 % menos error
  que el predictor viejo). "Justo sobre la localidad" no se muestra. `docs/VALIDACION_NOWCAST_CAPAS.md`.
- **Paso 2 HECHO EN PARALELO (20/11, noche):** celdas seguidas con sus rayos para la localidad exacta
  (`data/ia/celdas_localidad.json`, validación en vivo). Simulado con 703 casos: 5b y celdas juntos
  = 62 % detectado / 34 % falsas alarmas (5b solo 73/46); el objetivo 70-80 % no se alcanzó.
  `docs/VALIDACION_CELDAS_RAYOS.md`.
- **21/11/2026, madrugada (Lucas: "empezá con la idea 1 y la 2, terminalas, y buscá más estaciones"):**
  - **Idea 2 HECHA EN PARALELO:** aviso de la localidad exacta solo para tormentas fuertes o con
    rayos (`celdas_localidad.json` → `fuerte_o_rayos`, versión 1.1, validación en vivo propia en
    `resumen.aviso_fuerte`). Simulado con 718 casos: 60 % detectado / 43 % falsas alarmas (5b
    solo 78/67). **El objetivo 70-80 % no se alcanzó.** Con rayos es muy seguro (7 % falsas) pero
    cubre 1 de cada 5. Sierra de la Ventana: 81 % falsas alarmas. `docs/VALIDACION_AVISO_FUERTE.md`.
  - **Idea 3 HECHA EN PARALELO:** alarma de "ya llega" por estaciones aguas arriba
    (`scripts/estaciones/alarma_llegada.py` → `data/ia/estaciones/alarma_llegada.json`, validación
    en vivo). Probada con el 19/11 y el 20/11: 3 de 3 alarmas acertadas (Bahía Centro → Punta Alta),
    pocos casos. `docs/ALARMA_ESTACIONES.md`.
  - **5 estaciones BCP nuevas:** Punta Alta, La Vitícola, Villa Ventana, Sierras Grandes y
    Macachín (19 estaciones en total). Radio del resumen de Bahía: 20 km (antes 25) para no mezclar
    Punta Alta. Falta cobertura al oeste y suroeste. Pasman sin datos hace 20 días.
    `docs/ESTACIONES_NUEVAS.md`.
  - Doppler en vivo: comprobado que la página de SINARAME no lo publica (rama `prueba-doppler`).
    El gancho se descarta (dura poco y el PNG no lo resuelve lejos del radar).
- **Investigación del 21/11 (madrugada):** el atraso del radar es casi todo de SINARAME (mediana
  16 min hasta que lo vemos; nuestra cadena suma ~4,5 min: radar 2 + Intelligence 2,4; sin cola).
  SINARAME publica en dos tandas (1/3 de las imágenes a 4-7 min, 2/3 a 14-17). Descartados por Lucas:
  más radares (vecinos de SINARAME = mismo atraso; INTA Anguil se rompe seguido; no hay otros que
  lleguen a Bahía), radar simulado por satélite (GREMLIN: es una estimación), enlaces de celular
  (datos privados), prueba del detector más rápido.
- **ETAPA G HECHA EN PARALELO (23/11/2026, tarde): `docs/ETAPA_G.md`.** Medida con 46 jornadas severas
  verificadas de EE. UU. (radar MRMS + base oficial NCEI). Entra solo el eco en arco: 6 de cada 10
  dieron ráfagas severas en la hora siguiente (base 1 de 10). Desvío de supercelda y duración no
  entran. La velocidad sola ("línea rápida", hoy en lo que se ve) acierta 15 %: decidir con Lucas.
- **PENDIENTES QUE DECIDE LUCAS (23/11/2026, 14:30):**
  - Calibrar la lluvia con el historial oficial del SMN Bahía Blanca Aero (a 2 km del radar, zona sin
    ecos fijos). Probado: el servicio WIS2 del SMN (`w2b.smn.gov.ar`, mismo que usaba el colector) da
    lluvia cada 6 h y 24 h y viento medio horario desde al menos el 24/10 (`datetime` con intervalo).
    Solo para calibrar, no entra en vivo. Es una sola estación: si no junta el mínimo de casos, no se
    cambia el factor. Se suma el historial de la Bolsa de Cereales si Alex lo consigue.
  - Su parte de la etapa C (VPS que se actualiza solo y reloj externo).
  - Estaciones: desactivadas las que no cumplen (#78). Revisadas y descartadas por Lucas: SIMPARH (solo
    repite la Bolsa de Cereales y no guarda historial), Bajo Hondo, Monte Hermoso, Pedro Luro (SNIH),
    Redes Climáticas La Pampa, calidad del aire municipal (EMCABB), INTA SIGA, SNIH.
  - Ramas de prueba para borrar a mano en GitHub: `prueba-calibracion`, `prueba-etapa-d`, `prueba-etapa-d2`,
    `prueba-fuentes-bahia`, `prueba-etapa-a` y las viejas de la lista de abajo.
- **ETAPAS D, E y F HECHAS (23/11/2026, tarde): `docs/SIMULACION_FINAL.md`.** Simulación final con
  29 días: llegada a 10 km 72 % / 26 % (tormenta), 81 % / 11 % (lluvia). Pasa a verse la tormenta
  eléctrica sobre la localidad (2-9 % de avisos de más). Arreglos: ecos de terreno de la noche fuera de
  lo que se ve (canal 13 antes del análisis por localidad), validación en vivo de "sobre la localidad",
  viento con cortes del SMN, granizo severo con confirmación, Pasman desactivada. Queda: etapa G
  (Doppler sin gancho) y Render/Netlify; la parte de Lucas de la etapa C.
- **ETAPAS B y C HECHAS (23/11/2026, tarde).** Etapa B: `docs/ETAPA_B.md` (#75): calibración de la
  lluvia con estación y radar en la misma hora (había un desfase), puntaje de gravedad contra lo
  medido (`verificar_puntaje.py`), informe mensual automático (`data/ia/informe/AAAA-MM.md`).
  Etapa C: `docs/ETAPA_C.md`: VPS que se actualiza solo (falta que Lucas lo instale por SSH, 2
  comandos), `vigilancia.py` con avisos por issue de GitHub (clave por vencer, sin latido, sin radar,
  respaldo que falla, relojes), reloj externo `docs/RELOJ_EXTERNO.md` (falta que Lucas cree la
  clave `climaar-reloj-externo` y la tarea en cron-job.org). Sigue la etapa D cuando Lucas diga.
- **PARA EL CHAT NUEVO (23/11/2026, 12:10 hora de Argentina) — pedido de Lucas: "valida y termina
  la etapa B (punto 9) y la etapa C (punto 11); controlá que esté todo bien, corregí los errores que
  vayan surgiendo y validá todo".** Permiso permanente para abrir y unir los pull requests a main.
  Respuestas breves y en español, sin términos en inglés, horarios en hora de Argentina.
  - **Etapa B, punto 9: calibración y medición continua.**
    1. Lluvia: radar contra pluviómetros (estaciones BCP, CWOP; `calidad_y_calibracion.py`,
       `historial_superficie.json`). Corregir cuánta lluvia marca el radar solo si los datos alcanzan
       (el historial de estaciones empieza el 6/11 y casi no tiene lluvia): dejarlo armado para que
       se recalcule solo a medida que llueve, con un mínimo de casos antes de cambiar nada.
    2. Puntaje de gravedad (en paralelo, #46/#47): compararlo con lo medido después (ráfagas y
       lluvia de las estaciones, rayos), para ajustar los cortes más adelante.
    3. Informe mensual de aciertos automático: nowcast (`verificacion_nowcast.json`, también los
       métodos de la etapa A en paralelo: `estaciones`), llegadas por localidad, modo sin radar,
       alarma "ya llega".
  - **Etapa C, punto 11: vigilancia.**
    1. Que el VPS se actualice solo desde main (hoy hay que hacer `git pull` a mano; el VPS usa
       `/opt/Clima-ar/.venv-ia`) y reinicie el detector sin perder imágenes.
    2. Control de cada fuente (`salud_cadena.py`): sumar avisos claros de lo que pasó el 23/11.
    3. Reloj externo de respaldo, afuera de GitHub (Lucas tiene que crear una clave).
  - **Lo que pasó el 23/11 (arreglado):** de 5:10 a 11:20 no entró radar. (a) La clave de GitHub del
    VPS venció: error 401, sin latido. Lucas creó `climaar-detector-2026-11` (vence el 21/02/2027) y
    la puso en `/etc/climaar-detector.env`. (b) El respaldo de GitHub (`climaar-detector-sinarame.yml`)
    no instalaba numpy y fallaba en cada consulta (#73). La etapa C tiene que avisar estas dos cosas
    a tiempo: clave por vencer o rechazada, y respaldo que falla.
  - **Claves por vencer:** "ClimaAR VPS Sync" y "ClimaAR VPS Sync 2026" vencen el 6 y el 7/12/2026
    (recordatorio en el celular de Lucas el 4/12 a las 10:00). Ver qué usa cada una en el VPS.
  - **Después (Lucas: "de esas estaciones nos encargamos después"):** estaciones nuevas. Faltan 11 BCP
    dentro de 240 km (Saavedra, Faro, 17 de Agosto, El Zorro, San Miguel Arcángel, Carhué, Claromecó,
    Guaminí, Cristiano Muerto, San Cayetano, Daireaux): no sirven para la alarma (sin viento o con
    viento dudoso), sí para calibrar la lluvia. Redes Climáticas La Pampa: la lista de estaciones es
    pública (`/api/web/v1/site-map/stations`), los datos piden usuario. SIMPARH (Provincia de Buenos
    Aires, public-map-simparh.ada.gba.gov.ar): desde GitHub no se encuentra la dirección.
  - Rama de prueba para borrar: `prueba-etapa-a`.
- **PLAN DE TRABAJO (21/11/2026, 00:30 hora de Argentina) — en este orden, esperar el "arrancá" de
  Lucas en cada etapa:**
  - **Etapa A, puntos 1 a 4: HECHOS y medidos el 21/11 (madrugada), `docs/VALIDACION_ETAPA_A.md`.**
    Punto 1 (estaciones): queda en paralelo, validándose en vivo (en la simulación no cambió nada:
    falta historia de lluvia en las estaciones). Puntos 2-4 (posición, movimiento y crecimiento con
    el satélite): probados con 26 días y 1.419 casos, **ninguno mejora** (el yunque se mueve distinto
    que la lluvia); quedan apagados. Trayectoria sin cambios: 4,5 / 9,5 / 14,7 km. Arreglado: una
    lectura errónea de una estación (bajada sin medianoche) sumaba lluvia falsa. Falta el punto 5.
  - **Etapa A, punto 5 (punto 10, modo sin radar): HECHO el 23/11, `docs/VALIDACION_SIN_RADAR.md`.**
    Con el radar cortado 40-90 min, la lluvia por localidad sale del radar viejo movido (acierto
    0,41 contra 0,08 de antes); sin radar, "lluvia probable" pide tope <= -60 °C y un rayo cerca
    (falsas alarmas de 88 % a 54 %). **Etapa A terminada.** Sigue la etapa B (punto 9).
  - **Etapa A — Compensar el atraso del radar** (solo datos medidos; cada mejora se mide con
    `herramientas/simular_huecos.py` antes de usarla; en paralelo hasta validarla):
    1. Corrector con estaciones: lluvia medida cada minuto por las 13 estaciones (agregar donde
       miden lluvia fuerte, bajar donde no miden nada).
    2. Posición de ahora con el satélite: topes fríos de la imagen más nueva de GOES-19 (~10 min más
       fresca que el radar) para corregir dónde está cada tormenta, también las que no tienen rayos.
    3. Movimiento medido por el satélite (topes fríos entre imágenes) para controlar el del radar.
    4. Crece o se debilita: enfriamiento del tope y aumento de rayos para subir o bajar la
       intensidad pronosticada.
    5. Punto 10: probar el modo sin radar (satélite + rayos + estaciones) con una tormenta real.
    - Medida de la etapa: además de los huecos del radar, el error de la trayectoria de cada
      tormenta (`herramientas/comparar_trayectorias.py`; hoy 4,5 / 9,5 / 15 km a 10 / 20 / 30 min).
      Lo esperable con estos datos: bajar 20-30 % (~3,5 / 7 / 11 km); por debajo de ~3 km a 10 min
      y ~8 km a 30 min no se llega con un radar cada 10 min (las tormentas cambian de forma).
  - **Etapa B — Punto 9: calibración y medición continua:** lluvia con estaciones, puntaje de
    gravedad con estaciones y rayos, informe mensual de aciertos.
  - **Etapa C — Punto 11: vigilancia:** VPS que se actualice solo, control de cada fuente, reloj
    externo de respaldo (clave que tiene que crear Lucas).
  - **Etapa D — Simulaciones finales de aciertos** con casos reales.
  - **Etapa E — Decidir qué se muestra** con los números en vivo y de la etapa D: "justo sobre la
    localidad" (`celdas_localidad.json`, ideas 2 y 3), puntaje de gravedad, avisos.
  - **Etapa F — Repasos:** umbrales de la escala de gravedad; pasar a lo que se ve lo que esté
    medido; borrar el predictor viejo y las ramas de prueba; revisar Sierra de la Ventana (falsas
    alarmas); Pasman sin datos.
  - **Etapa G — Reemplazo del Doppler sin el gancho** (Lucas: "lo dejamos para lo último"): arco en
    ZH MAX, rayos y topes de GOES-19, viento que gira con la altura desde los modelos.
  - **Al final — Render y Netlify** (interfaz).
- Estado anterior: **puntos 1 a 5 terminados** (#42 GOES-19 bien ubicado y por tormenta, #43 señales
  severas y 0 °C, #44 casos guardados, #45 rayos 2σ, #46/#47 puntaje de gravedad EN PARALELO,
  #48 tormentas en formación EN PARALELO + velocidad con varias imágenes).
  Después 5b, 6 y 7 (terminados el 20/11). Después 7 a 11, luego simulaciones con casos reales para medir aciertos (regla 6, decisión de Lucas)
  y recién ahí decidir avisos y mostrar el puntaje. Al final Render y Netlify.
- Cómo probar: `pytest`; `herramientas/emular_workflow.py` en una copia, con tormenta y sin
  actividad (para "sin actividad": imagen de radar toda de índice 253, guardarla también en
  `data/radar/historico/AAAAmmddHHMMSS.png`, nuevo `radar_timestamp.txt`, correr
  `probar-radar-smn.yml` de "Procesar radar" a "Validar cadena Radar" y después Intelligence).
  Este entorno no tiene internet hacia NOAA/Open-Meteo: las pruebas reales se hacen en GitHub
  con la rama temporal `prueba-inspeccion` (workflow con `on: push`), y después de unir se
  revisa el primer ciclo en producción. Nunca commitear `data/` en PR de código
  (`git checkout origin/main -- data .climaar` antes de commitear). PR por la API REST
  (`gh api repos/lucas12839/clima-ar/pulls`), GraphQL no está disponible. **Antes de unir,
  revisar la lista de archivos del PR** (`gh api .../pulls/N/files`): el #46 se unió sin 3
  archivos (paso del workflow, integridad, plan) y se completó en el #47.
- Simulaciones: `herramientas/simular_formacion.py` (punto 5) vuelve a correr días pasados con
  el radar guardado y el satélite de NOAA; en GitHub con la rama `prueba-inspeccion` (matriz de
  días en paralelo). Días con tormentas en el radar guardado: 27/10-6/11, 12/11, 13/11.
- **Punto 5b nuevo (va antes del 6): nunca a ciegas cuando no hay imagen de radar nueva.**
  Decisión de Lucas (20/11): NO crear otro nowcast, mejorar al 100 % y validar el que ya existe
  (`nowcast_flujo_optico.py`); 8 tareas y **estructura en 6 capas acordada** en
  `docs/INSPECCION_FUENTES.md`. El nowcast actual NO es IA: mueve la última imagen y supone
  intensidad constante (por eso no ve crecer ni deshacerse una tormenta). **Hecho el 20/11
  (#53, #54): nowcast en capas + ANVIL, en paralelo.** Noche 19/11: atraso de SINARAME 16-18 min y faltaron
  las imágenes de 22:20 y 00:00 justo cuando la tormenta pasaba por Bahía. En la ciudad no
  hubo nada: convección elevada sobre capa estable, nube de arco al norte (caso y foto en
  `docs/casos/2026-11-19_nube_rodillo_bahia.md`, con el análisis de Alex Gonzalez).
- Datos por tormenta: `data/goes19/goes19_state.json` (`por_tormenta` y `formacion_satelite`,
  versión 1.8), `data/ia/puntaje_tormentas.json`, `data/ia/formacion_tormentas.json`; casos guardados en `data/goes19/historial/`, `data/ia/formacion/` y
  `data/ia/puntaje/`.
- Localidades: Bahía Blanca, Ingeniero White, General Daniel Cerri, Punta Alta, Tornquist,
  Médanos, Sierra de la Ventana (Lucas sacó Monte Hermoso, Dorrego y Pehuen-Co).
- Estaciones: solo fuentes con permiso de lectura automática (BCP, SMN, CWOP). No usar el
  usuario de Lucas ni claves internas de Ecowitt, WeatherLink, Weather Underground o
  Weathercloud. MeteoPunta descartado. Pedidos a instituciones descartados por Lucas (casi
  imposible); los datos completos del RMA10 (Doppler) ya los pidió Lucas: si llegan, se suman.

## Ramas de prueba para borrar (Claude no puede borrarlas)

- `prueba-etapa-a` (simulaciones de la etapa A, 21/11).
`auditoria-cadena`, `paso1-5b-visible`, `punto8-ambiente-satelite`, `paso2-celdas-rayos`, `reloj-cwop-tiempo`, `plan-pendientes`, `prueba-intermedios`, `prueba-intermedios2`, `prueba-captura-chrome`, `prueba-goes-red`,
`indices-fenomenos-localidades`, `prueba-estaciones`, `prueba-doppler`, `idea2-aviso-fuerte` (ya unida), `prueba-wrf`, `datos-wrf-historial` (historial del WRF, borrar al final), `prueba-inspeccion` (temporal de pruebas reales: borrar al final), `nowcast-sin-radar`, `nowcast-capas`, `punto6-anvil`, `punto7-modelos`, `validacion-sistema` y las ramas ya unidas de cada PR (`goes-zona-tormentas`, `senales-severas`, `casos-guardados`, `rayos-mejorados`, `puntaje-gravedad`, etc.).
