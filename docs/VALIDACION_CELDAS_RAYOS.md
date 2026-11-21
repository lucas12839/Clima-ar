# Paso 2: tormenta SOBRE la localidad, celda por celda con rayos (20/11/2026, noche)

Pedido de Lucas: la llegada de tormenta "justo sobre la localidad" fallaba casi siempre en
vivo. Objetivo planteado: en los próximos 30 min, 70-80 % detectado con menos de 40 % de
falsas alarmas.

## Qué se hizo (en paralelo, no cambia lo que se ve)

- `scripts/celdas_rayos.py`: los rayos del satélite (GLM, ~1 min de atraso) de los últimos
  10 min se agrupan en celdas (dónde está cada tormenta AHORA y su movimiento propio). Cada
  núcleo de >= 35 dBZ del radar (atrasado 10-30 min) se mueve con el 5b hasta ahora; si hay
  una celda de rayos donde debería estar, el lugar sale de los rayos. Las celdas de rayos sin
  núcleo en el radar entran como nuevas. Muchos caminos posibles (lugar, velocidad, dirección)
  -> porcentaje de que pase sobre la localidad y minuto estimado.
- `scripts/celdas_localidad.py` -> `data/ia/celdas_localidad.json` en cada ciclo, con
  validación en vivo contra las imágenes de radar que llegan después (`resumen`).
- `goes19/glm_ventana.py`: los rayos de los últimos 10 min ahora llevan su edad.

## Simulación (`herramientas/simular_celdas_rayos.py`)

26 días guardados con el historial real de rayos y satélite; radar atrasado 10, 20 y 30 min;
703 casos, 13.000 localidad-casos. Verdad: 35 dBZ o más dentro de la localidad en las
imágenes de ahora a +30 min. "Nuevas" = la tormenta no estaba encima al empezar (lo difícil).

| Método | Todas: detectadas / falsas alarmas | Nuevas: detectadas / falsas alarmas |
|---|---|---|
| 5b localidad (lo de hoy) | 73 % / 46 % | 59 % / 68 % |
| Celdas >= 50 % | 64 % / 39 % | 49 % / 63 % |
| **5b y celdas >= 50 %** | **62 % / 34 %** | 46 % / 58 % |
| Solo celdas con rayos >= 70 % | 14 % / 8,5 % | 8 % / 24 % |

Calibración de la celda con rayos (nuevas): cuando dice 90-100 %, pasó el 82 %.

## Conclusión honesta

- **El objetivo no se alcanzó.** Combinar el 5b con las celdas baja las falsas alarmas de 46 %
  a 34 % (cumple el "menos de 40 %"), pero detecta 62 %, no 70-80 %. Para las llegadas nuevas
  la mejora es chica.
- **Los rayos aportan poco para esta pregunta:** la mayoría de las tormentas de 35 dBZ que
  pasan sobre una localidad no tienen rayos cerca. Cuando la celda tiene rayos el aviso es
  muy seguro (8 de cada 10 pasan), pero cubre 1 de cada 7 casos.
- El límite es el atraso del radar (20-30 min) más la vida corta de cada celda: las que se
  deshacen o nacen al costado no se pueden prever en un punto exacto.

Queda corriendo en paralelo con validación en vivo de cinco criterios (`combinado`, `rayos`,
`celdas_probable`, `celdas_posible`, `5b_localidad`). Se decide qué mostrar con esos números.
