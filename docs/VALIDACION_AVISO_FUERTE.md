# Idea 2: aviso de la localidad EXACTA solo para tormentas fuertes o con rayos (20-21/11/2026)

Decisión de Lucas (20/11, noche): el aviso para la localidad exacta queda solo para tormentas
fuertes (45 dBZ o más) o con rayos. Lo demás (lluvia, tormentas débiles sin rayos) lo sigue
cubriendo la zona de 10 km.

## Qué se hizo (en paralelo, no cambia lo que se ve)

- `scripts/celdas_rayos.py`: parámetro `solo_fuertes_dbz`. Con él, de las celdas del radar
  (>= 35 dBZ) solo cuentan las que tienen un pico de 45 dBZ o más, o rayos encima ahora;
  las celdas de rayos sin núcleo en el radar entran como siempre.
- `scripts/celdas_localidad.py` (versión 1.1): en cada localidad, `fuerte_o_rayos` con la
  probabilidad, si avisa (umbral 0,7) y la hora estimada. Validación en vivo con su propia
  verdad (`resumen.aviso_fuerte`): 45 dBZ o más dentro de la localidad en las imágenes que
  llegan después, o rayos dentro de la localidad (radio + 4 km). Los rayos de cada ciclo se
  anotan en `rayos_en_localidad` (3 h).
- `herramientas/simular_aviso_fuerte.py`: la simulación.

## Simulación

26 días guardados con el historial real de rayos y satélite; radar atrasado 10, 20 y 30 min;
718 casos, 13.000 localidad-casos. Verdad: 45 dBZ o más dentro de la localidad, o rayos
dentro de ella, entre ahora y +30 min. Hubo 724 eventos (695 fuertes, 212 con rayos). Además
hubo 554 tormentas de 35-44 dBZ sin rayos: esas quedan para la zona de 10 km.

| Método | Todas: detectadas / falsas alarmas | Nuevas: detectadas / falsas alarmas |
|---|---|---|
| 5b en la localidad (lo de hoy) | 78 % / 67 % | 66 % / 82 % |
| Fuertes o con rayos >= 0,5 | 67 % / 53 % | 52 % / 72 % |
| **Fuertes o con rayos >= 0,7 (elegido)** | **60 % / 43 %** | 42 % / 66 % |
| Solo celdas con rayos >= 0,7 | 20 % / 7 % | 9 % / 12 % |

Con el umbral elegido, por atraso del radar: 10 min -> 67 % / 42 %; 20 min -> 60 % / 45 %;
30 min -> 53 % / 43 %. Por localidad: Bahía 66 % / 39 %, Punta Alta 67 % / 37 %, Médanos
66 % / 37 %, Cerri 69 % / 43 %, White 53 % / 51 %, Tornquist 44 % / 36 %, Sierra de la Ventana
52 % / 81 % (en las sierras el radar falla más).

## Conclusión honesta

- **El objetivo no se alcanzó.** Restringir el aviso a tormentas fuertes o con rayos baja las
  falsas alarmas del 5b (de 67 % a 43 %), pero detecta 60 %, no 70-80 %, y las falsas alarmas
  quedan apenas arriba del 40 %.
- **Cuando hay rayos el aviso es muy seguro** (7 % de falsas alarmas), pero cubre 1 de cada 5
  casos: la mayoría de las tormentas fuertes que pasan sobre una localidad no tienen rayos.
- El límite sigue siendo el mismo: el radar llega 10-30 min tarde y las celdas nacen, se
  deshacen o se corren en ese tiempo. Con 10 min de atraso funciona mejor (67 %).
- Sierra de la Ventana necesita revisión aparte (81 % de falsas alarmas).

Queda corriendo en paralelo con validación en vivo. Qué se muestra lo decide Lucas (punto 5
de la lista de pendientes) con los números en vivo.
