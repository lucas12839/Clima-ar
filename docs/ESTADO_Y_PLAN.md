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
- Pendiente: enlace de la estación de MeteoPunta en Punta Alta para sumarla como punto de control.

### Preguntas para Lucas (calibración)
1. ¿Hubo granizo o tormenta fuerte el 28/10, 4/11, 12/11 o 13/11 en Bahía Blanca? ¿Y en
   Tornquist entre el 30/10 y el 9/11?
2. Viernes 13/11: ¿cuánto llovió exactamente en Punta Alta y en Bahía Blanca, y a qué hora fue
   lo fuerte? (el radar crudo dio ~190 mm en Punta Alta: parece sobreestimar).

## Siguiente
- **A. Nunca a ciegas:** falta probar el modo sin radar con una tormenta real (rayos + topes
  + estaciones); vientos satelitales todavía solo como dato general.
- **B. Mejor pronóstico:** horizonte 2–3 h con varios escenarios, formas (líneas, ecos en
  arco), crecimiento por celda (hoy: tendencia a ≤ 20 km).
- **C. Calibración continua:** con las respuestas de arriba, ajustar la fórmula de lluvia del
  radar crudo y los cortes de granizo; repetir `calibrar_llegadas.py` cada mes.
- **D. Vigilancia:** VPS que se actualice solo, monitoreo de cada fuente, Doppler más adelante.
- Menor: reemplazar `getdata` de Pillow (deja de existir en 2027-10).

## Ramas de prueba para borrar (Claude no puede borrarlas)
`prueba-intermedios`, `prueba-intermedios2`, `prueba-captura-chrome`, `prueba-goes-red`,
`indices-fenomenos-localidades` (ya unida).
