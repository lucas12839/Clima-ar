# Etapa G: indicios de tormenta severa sin Doppler (23/11/2026)

El RMA10 no publica el Doppler. Se probaron tres indicios que salen solo de la reflectividad,
del seguimiento de cada tormenta y del viento del modelo (`scripts/indicios_severos.py`):
eco en arco, desvío de supercelda (a la izquierda en el hemisferio sur) y duración.

## Con qué se midió (verdad oficial, sin inventar nada)

En Argentina no hay una base pública oficial de tormentas severas, y el radar de Córdoba de
RELAMPAGO tiene un contrato que prohíbe usarlo fuera de esa campaña. Se usó Estados Unidos
(`herramientas/simular_indicios_eeuu.py`, en GitHub):
- **Radar:** MRMS de NOAA, máximo de la columna con control de calidad (como el ZH MAX),
  pasado a la grilla de ClimaAR (512 x 512, 240 km, 0,94 km) y una imagen cada 10 min.
- **Verdad:** base oficial NCEI Storm Events (verificada por el servicio meteorológico de
  EE. UU.): granizo de 2,5 cm o más, ráfagas de 93 km/h o más, tornados, con hora y lugar.
  Cuenta si el reporte toca la tormenta (su radio + 10 km, siguiendo su movimiento) entre
  10 min antes y 60 min después. Las tormentas sin reporte cuentan como "no severas" aunque
  quizás nadie lo reportó: los aciertos medidos son un piso, no un techo.
- **Ambiente:** análisis del modelo GFS (viento en 10 m, 850, 700 y 500 hPa, CAPE).
- **46 jornadas severas** de 2023 y 2024 en las llanuras (20 de viento, 21 de granizo, 5 de
  tornados), 21.157 imágenes de tormentas, con dos umbrales de envolvente (35 y 40 dBZ).

## Resultado

| Indicio (tormentas con núcleo de 45 dBZ) | Activaciones | Viento severo en la hora siguiente | Cualquier severo |
|---|---|---|---|
| Ninguno (todas las tormentas) | 21.157 | 9 % | 19 % |
| Solo velocidad >= 60 km/h | 5.298 | 15 % | — |
| **Eco en arco** (largo >= 60 km, curvatura >= 0,05, 30-120 km/h) | 506 | **59 %** | **70 %** |
| Desvío de supercelda >= 20 km/h | — | 14 % (base 7 %) en granizo grande/tornado | 36 % |
| Duración >= 90 min | — | — | 46 %, casi siempre después de lo severo |

- **Eco en arco: entra (en paralelo).** De cada 10 ecos en arco, 6 dieron ráfagas severas
  oficiales en la hora siguiente y 7 algún fenómeno severo: 6 veces más que una tormenta
  cualquiera. Funciona en 41 de las 46 jornadas, no depende de un caso. Es estable con
  envolvente de 35 y de 40 dBZ (en el RMA10 se usa 40: el radar crudo marca alto).
  Anticipación: cuando avisa antes, ~30 min (mediana).
- **Desvío de supercelda: no entra.** Apenas duplica la base (14 % contra 7 %).
- **Duración: no entra.** Llega cuando lo severo ya pasó.
- **Dato importante:** la velocidad sola (lo que hoy usa ClimaAR para "línea rápida") acierta
  15 %, casi igual que cualquier tormenta (9 %). El arco es 4 veces mejor. No se cambió lo
  que se ve: queda para decidir con Lucas.

## Control con Bahía (radar crudo, 24/10 al 23/11, 3.421 imágenes)
Con los mismos cortes, el arco aparece 52 veces en 5 días: 28/10, 3/11, 5/11, 12/11 y 13/11
(los días más tormentosos del período, el 13/11 con alerta naranja). Ninguno en días sin
tormentas fuertes. No hay mediciones de ráfagas de esos días para confirmar el viento.

## Cómo queda
`scripts/indicios_severos.py` corre en cada ciclo de Intelligence **en paralelo**
(`data/ia/indicios_severos.json`, historial en `data/ia/indicios/`). No cambia lo que se ve.
Lo que no se traslada de EE. UU.: los números de intensidad y del ambiente.
