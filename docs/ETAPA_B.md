# Etapa B, punto 9: calibración y medición continua (23/11/2026)

Todo corre en paralelo en cada ciclo de Intelligence: se mide, no cambia lo que se ve.

## 1. Lluvia del radar contra las estaciones (`scripts/estaciones/calidad_y_calibracion.py`)

**Error encontrado y corregido:** el registro comparaba horas distintas. La estación se medía
en la hora que terminaba "ahora" y el radar en la hora que terminaba en la imagen, que llega
10 a 30 min tarde. Además se registraba aunque el radar hubiera tenido huecos (el 23/11, con
6 h sin radar, una "hora" del radar sumaba solo 50 min de lluvia).

Ahora:
- Estación y radar en **la misma hora**: la que termina en la primera imagen de cada hora
  (la de en punto, o hasta y 20 si esa se salteó).
- Solo se registra si el radar tuvo la hora completa (>= 5 imágenes, sin huecos de más de
  20 min) y la estación tiene datos al principio y al final de la hora.
- Las 943 filas viejas quedan con `"ventana": "desfasada"` y no se usan para calibrar.
- El factor se calcula solo, en cada ciclo, recién con **>= 20 horas con lluvia, de >= 3 días
  distintos y >= 30 mm medidos**, en estaciones confiables a <= 100 km del radar. Hasta
  entonces rige el inicial (0,4, Punta Alta 13/11). Siempre queda entre 0,25 y 2.
- Se guarda también la relación por distancia al radar (0-50, 50-100, 100-160, 160-250 km),
  para ver cuánto menos ve el radar lejos (el haz pasa por arriba de la lluvia).

Dato a seguir: con las filas viejas (desfasadas), cerca del radar la estación y el radar sin
corregir daban casi lo mismo (relación 1,0, 5 horas de lluvia débil del 20/11), es decir que
el 0,4 podría estar bajando de más la lluvia débil. Con lluvia fuerte (Punta Alta) daba 0,38.
No se cambia nada a mano: el factor nuevo entra solo cuando se junten los casos.

## 2. Puntaje de gravedad contra lo medido (`scripts/verificar_puntaje.py`)

Cada imagen guardada en `data/ia/puntaje/`, a los 45 min, se compara con las estaciones que
estaban a <= 10 km de cada tormenta: viento máximo (ráfaga o viento medio) y lluvia desde
10 min antes hasta 40 min después de la imagen, y los rayos de la misma tormenta en las 3
imágenes siguientes. Se acumula por mes, por fenómeno y por grado (0 a 4): cuántas veces se
midió >= 60 / >= 90 km/h, >= 10 / >= 20 mm, >= 1 / >= 10 rayos. Con eso se ajustan los cortes
en la etapa F. El granizo no se puede verificar (no hay mediciones).

Salidas: `data/ia/verificacion_puntaje.json` y cada caso en `data/ia/verificacion_puntaje/AAAA-MM.jsonl`.

## 3. Informe mensual de aciertos (`scripts/informe_aciertos.py`)

`data/ia/informe/AAAA-MM.md` (para leer en el celular) y `.json`. Se actualiza en cada ciclo y
se cierra solo al cambiar el mes (hora de Argentina). Incluye: nowcast contra la imagen sin
mover, corrector con estaciones, **modo sin radar** (nuevo contador en
`verificacion_nowcast.json`: pronósticos hechos con el radar de 40 min o más), llegadas por
localidad, alarma "ya llega", puntaje de gravedad y calibración de la lluvia.

Primeras cifras (desde el inicio de las mediciones, 23/11): el corrector con estaciones todavía
no mejora (tormenta a 60 min: 0,08 contra 0,14 del nowcast solo), por eso sigue apagado.

## Pruebas

`pytest` (tests nuevos: `test_calidad_estaciones`, `test_verificar_puntaje`,
`test_informe_aciertos`, modo sin radar en `test_nowcast_capas`) y la cadena emulada con
tormenta y sin actividad (`herramientas/emular_caso_prueba.sh`): sin alertas nuevas.
