# Auditoría de la cadena de producción (20/11/2026, tarde, hora de Argentina)

Pedido de Lucas antes del punto 8: revisar toda la cadena y lo que se fue
agregando, buscar detalles pasados por alto, cosas para optimizar y
correcciones chicas. Todo arreglado en un solo cambio y verificado.

## Lo que se encontró y cómo quedó

| # | Problema | Arreglo |
|---|---|---|
| 0 | **Causa raíz del seguimiento:** el capturador marcaba el frame nuevo en `data/radar/seguimiento/estado.json` ANTES de procesarlo. El seguimiento creía que ya estaba hecho y reconstruía todo desde cero en casi cada imagen: **los números de las tormentas cambiaban en cada frame** y se perdía la historia de cada una (predicción, enfriamiento del tope, puntaje). | El capturador ya no toca ese archivo (usa solo `.climaar/estado_rma10.json`). Simulado con las imágenes del 20/11 (14:00 a 19:10): continuidad en todos los frames. |
| 1 | Estaciones del SMN: miden por hora y el dato llega 20-25 min tarde, pero el límite era 15 min para todas. **Nunca entraba ninguna** (ni Bahía Blanca Aero). El informe del aeropuerto (SAZB) valía ~10 min por hora. | Límite por red en `scripts/estaciones/frescura.py`: BCP 15, CWOP 20, SMN 90, SAZB 75 min. |
| 2a | La estación CWOP de hace 2 h seguía contando como "al día": la marca "válida" se guardaba al escuchar y no se recalculaba. | La edad se calcula de nuevo al usar el dato (`senales_superficie.py`). |
| 2b | GitHub corre los relojes "cada 10 min" solo ~3 veces por día: CWOP y el **respaldo del VPS** casi nunca corrían (si el VPS caía, horas sin radar). | `climaar-cwop.yml` es un reloj encadenado: cada vuelta lanza la siguiente (mínimo 8 min) y pide al vigilante del detector que mire el latido. El reloj programado queda solo para rearrancar la cadena. |
| 3 | Al reconstruir, la numeración volvía a 1: una predicción vieja de la "tormenta 5" se comparaba con otra "5" (error publicado ~40 km a 10 min). Las predicciones sin observación quedaban pendientes para siempre. | Numeración continua; una predicción no se valida contra una tormenta nacida después; pendientes vencen a los 60 min; registro de errores viejo descartado (versión 2). |
| 4 | Cada corrida bajaba toda la historia del repo (~540 MB, 25-36 s) y reinstalaba paquetes. | Descarga corta (`fetch-depth: 1`, probado el rebase) y caché de paquetes. Python 3.12 en todos. |
| 5 | `resultado_final.json` pesaba 1 MB (con espacios) y se guarda ~290 veces por día. | Los 4 archivos grandes se escriben compactos (mismo contenido, ~mitad de tamaño). |
| 6 | "Riesgo severo" se calculaba en Calidad y nunca se guardaba ni se leía. | Borrado. |
| 7 | Restos viejos: estado RainViewer de octubre (radio 150 km), capturas de error, `continuidad.txt`, `data/goes/`, dataset para entrenar. | Borrados. Las capturas de error van a `logs/` (fuera de main). |
| 8 | 3 procesos manuales sin uso (Auto Radar, Ciclo Radar, Watchdog). | Borrados. |
| 9 | pysteps anotado pero no instalado. | Aclarado: no se instala porque ANVIL está apagado. |
| 10 | Versiones de paquetes sin fijar. | Fijadas a las que corren hoy en producción. |
| 11 | 5 de 14 estaciones sin datos hace días. | Se informan en `estaciones_sin_datos_24h`. |

## Para que no se pase por alto de nuevo

- **Control de salud en cada ciclo:** `scripts/salud_cadena.py` →
  `data/ia/salud_cadena.json` y avisos en Actions. Revisa: estaciones que entran
  por red, estaciones muertas, reloj CWOP, latido del VPS, SAZB, edad del radar,
  numeración que vuelve atrás, seguimiento reconstruido desde cero 3 imágenes
  seguidas, error de la predicción por tormenta y tamaño de los archivos.
- **Pruebas:** `tests/test_auditoria_20_11.py` (una por arreglo).
- **Lista en `CLAUDE.md`** ("Antes de cerrar cualquier punto").

## Queda anotado (no se tocó)

- La predicción vieja por tormenta (`prediccion_evolucion.py`) con la continuidad
  arreglada sigue errando 10-45 km a 20 min en la simulación (pocas muestras): sigue
  el centro de sistemas grandes que se juntan y separan. El pronóstico validado es
  el nowcast en capas (5b). El control de salud lo va a medir en vivo.
- El grupo de escrituras compartido (radar, IA, CWOP) deja una sola corrida en
  espera: hoy no cancela radares (sí el 18 y 19/11), pero es un riesgo si la IA tarda.
