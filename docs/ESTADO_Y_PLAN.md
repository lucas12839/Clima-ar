# ClimaAR — estado y plan (traspaso al 19/11/2026)

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

## Próximo: punto 1 — índices por fenómeno
Cada fenómeno con su índice; la calificación general es el índice más alto y dice cuál
es ("Severa (viento)"). Propuesta (Lucas aprobó la idea, cortes a revisar):

- **Lluvia** (radar, mm/h por Marshall-Palmer; estaciones confirman): llovizna <1,
  débil 1–3, moderada 3–12, fuerte 12–50, **severa** >50 mm/h o fuerte sostenida >1 h
  o 30 mm en 1 h / 50 mm en 3 h.
- **Viento** (ráfagas de estaciones; radar anticipa): leve <30, moderado 30–50,
  fuerte 50–90, **severo** ráfaga ≥90 km/h o línea ≥60 km/h en bloque (coherencia ≥0,85,
  ≥45 dBZ). Con esto el 16/12 habría quedado "Severa (viento)" desde las 15:50.
- **Granizo** (radar): sin señal <50 dBZ, posible 50–55, probable 55–60, **severo** ≥60.
- **Tormenta eléctrica** (rayos GOES-19): sin rayos, aislada, frecuente, **severa** con
  salto brusco de rayos.

Pendiente de respuesta de Lucas: si viento y lluvia severa por radar se muestran como
"severa" o "posible severa" hasta calibrarlos.

## Después
2. Probar viento severo y lluvia severa contra los últimos 30 días de radar (falsas alarmas).
- **A. GOES-19 y nunca a ciegas:** rayos de los últimos 5–10 min y saltos de rayos,
  enfriamiento de topes de nubes, vientos satelitales, modo completo sin radar.
- **B. Mejor pronóstico:** crecimiento/debilitamiento por celda, horizonte 2–3 h con
  varios escenarios, potencial de severidad del ambiente, formas (líneas, ecos en arco).
- **C. Calibración y validación continua** con cada temporal nuevo.
- **D. Vigilancia:** VPS que se actualice solo, monitoreo de cada fuente (hoy el vigilante
  de GitHub tarda horas si el VPS se cae), Doppler más adelante.
- Menor: reemplazar `getdata` de Pillow (deja de existir en 2027-10).

## Ramas de prueba para borrar (Claude no puede borrarlas)
`prueba-intermedios`, `prueba-intermedios2`, `prueba-captura-chrome`.
