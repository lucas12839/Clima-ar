# ClimaAR — estado y plan (actualizado 19/11/2026, 09:30 hora de Argentina)

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


## Contexto rápido para un chat nuevo (19/11/2026, 16:40)
- Lucas trabaja desde el celular, no sabe validar pull requests: Claude prueba, valida, abre
  el PR y lo une a main (permiso dado). Probar siempre con `pytest`, `herramientas/emular_workflow.py`
  (con tormenta y sin actividad) y, si cambia la clasificación, el 13/11 con
  `herramientas/simular_evento_real.py` (frames en `data/radar/historico`).
- Hechos con Claude y en producción: PR #15 a #38 (ayer: plan 1-7 y puntos 1-5; hoy: índices
  por fenómeno, localidades, GOES-19, ambiente, calidad de estaciones, calibración de lluvia,
  estado de fuentes, estación CWOP).
- Localidades: Bahía Blanca, Ingeniero White, General Daniel Cerri, Punta Alta, Tornquist,
  Médanos, Sierra de la Ventana (Lucas sacó Monte Hermoso, Dorrego y Pehuen-Co).
- Estaciones: solo fuentes con permiso de lectura automática (BCP, SMN, CWOP). No usar el
  usuario de Lucas ni claves internas de Ecowitt, WeatherLink, Weather Underground o
  Weathercloud. MeteoPunta descartado (no respondió). Sí se puede pedir acceso a
  instituciones (UNS, CONICET, Municipalidad, red provincial de 11 estaciones).
- **Próximo trabajo de Lucas:** corrección de los umbrales de tormentas (granizo severo con
  radar crudo está bajo: el 13/11 en Punta Alta cayó granizo chico y ClimaAR dijo "posible
  severo"). Ahora es el punto 4 de `docs/INSPECCION_FUENTES.md`: un puntaje de gravedad por tormenta
  (radar + satélite + rayos + modelo), después de los puntos 1 a 3.

## Ramas de prueba para borrar (Claude no puede borrarlas)
`prueba-intermedios`, `prueba-intermedios2`, `prueba-captura-chrome`, `prueba-goes-red`,
`indices-fenomenos-localidades`, `prueba-estaciones`, `prueba-inspeccion` y las ramas ya unidas de cada PR.
