# ClimaAR — estado y plan (actualizado 20/11/2026, 03:15 hora de Argentina)

Leer junto con `CLAUDE.md` (reglas fijas del repo).

## Cómo trabajar con Lucas
- Todo en español, sin términos técnicos en inglés (o explicados).
- Respuestas breves: qué se hizo, qué falta, qué decidir.
- Horarios siempre en **hora de Argentina (UTC−3)**.
- Claude genera, arregla, prueba y sube los cambios (rama + pull request + merge).
- El VPS no se actualiza solo desde GitHub: si hay que tocarlo, darle el comando a Lucas.

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
  mi localidad en la próxima hora?") para ver si el 5b queda cerrado: TODAVÍA NO HECHO.
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
`prueba-intermedios`, `prueba-intermedios2`, `prueba-captura-chrome`, `prueba-goes-red`,
`indices-fenomenos-localidades`, `prueba-estaciones`, `prueba-inspeccion` (temporal de pruebas reales: borrar al final), `nowcast-sin-radar`, `nowcast-capas`, `punto6-anvil`, `punto7-modelos`, `validacion-sistema` y las ramas ya unidas de cada PR (`goes-zona-tormentas`, `senales-severas`, `casos-guardados`, `rayos-mejorados`, `puntaje-gravedad`, etc.).
