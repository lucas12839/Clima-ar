# Etapas D, E y F (23/11/2026)

## Etapa D: simulación final de aciertos con casos reales

Todo el radar guardado: **29 días con lluvia (24/10 al 23/11), 1.583 imágenes de partida (todas)**,
con el historial real de GOES-19 (satélite y rayos; se sumaron el 21, 22 y 23/11 a la rama
`datos-goes-historial`). Como en vivo: el radar llega 10, 20 o 30 min tarde y solo se usa lo
que ya estaría publicado a esa hora. La verdad son las imágenes reales que llegaron después,
limpias de ecos falsos con el satélite. Corrió en GitHub, un día por trabajo
(`herramientas/simular_envolvente.py` y `simular_aviso_fuerte.py` con `--dias`).

### 1. ¿Llega una tormenta o lluvia a la zona de mi localidad (hasta 10 km) en la próxima hora? (lo que se ve)

| | Detectadas | Falsas alarmas | Casos |
|---|---|---|---|
| Tormenta (35 dBZ), "probable" | **72 %** | **26 %** | 6.194 llegadas |
| Lluvia (20 dBZ), "probable" | **81 %** | **11 %** | 11.665 llegadas |
| Antes del 5b (para comparar), tormenta | 61 % | 17 % | |

Tormenta por localidad: Bahía Blanca 78/20, General Cerri 74/26, Ingeniero White 78/38,
Médanos 66/23, Punta Alta 73/29, Sierra de la Ventana 68/33, Tornquist 72/20.
Iguales a los del 20/11 con 26 días (74/24): el resultado es estable.

### 2. ¿Pasa una tormenta justo SOBRE la localidad en los próximos 30 min?

| Método | Detectadas | Falsas alarmas |
|---|---|---|
| 5b en la localidad exacta | 79 % | 66 % |
| Celdas fuertes o con rayos (idea 2) | 60 % | 41 % |
| **Celdas con rayos >= 0,7** | 22 % | **7 %** |

La localidad exacta sigue sin llegar a lo pedido (70-80 % con pocas falsas alarmas): es un
límite de los datos (radar cada 10 min que llega tarde, tormentas que cambian de forma).
Con rayos es muy segura: de cada 100 avisos sobraron 2 a 9 en cada localidad, salvo Sierra de
la Ventana (47 %).

### 3. Modo sin radar
Medido en la etapa A con 26 días (`docs/VALIDACION_SIN_RADAR.md`). Desde la etapa B se mide
también en vivo (informe mensual, "sin radar").

### 4. Índices por fenómeno (escala de gravedad), 31 días, 3.418 imágenes
`herramientas/revisar_30_dias.py`, solo radar, con tres limpiezas del radar:

| Tornquist | Limpieza de antes | + ecos sueltos | **+ satélite (ecos sin nube)** |
|---|---|---|---|
| Días con granizo "posible severo" | 12 | 12 | **5** |
| Días con lluvia fuerte | 4 | 4 | **0** |
| Días con lluvia severa | 1 | 1 | **0** |

En Médanos baja el granizo "probable" de 4 a 2 días. En las demás localidades no cambia nada:
el control solo saca ecos de terreno de la noche, nunca tormentas reales.

### 5. Validación en vivo (lo que se midió solo, 20 al 23/11)
- Llegadas a 10 km (tormenta "probable"): 57 % detectadas, 42 % falsas alarmas, 147 llegadas.
  Menos que en la simulación: son pocos días, con tormentas sueltas.
- **Error encontrado:** la validación en vivo de "sobre la localidad" daba 5 % de detección porque
  contaba como tormentas los ecos de terreno de la noche sobre Tornquist y Médanos (sin ninguna
  nube encima): 136 "pasó sin aviso" falsos. Corregido (ver F).

## Etapa E: qué se muestra (decidido con los números)

| Qué | Decisión |
|---|---|
| Llegada a la zona de cada localidad (10 km) | Sigue igual (72 % / 26 %). |
| **Tormenta eléctrica sobre la localidad (30 min)** | **Pasa a verse** (celdas con rayos >= 0,7: 2-9 % de avisos de más). Campo `sobre_la_localidad` de cada localidad en `resultado_final.json`. Sin Sierra de la Ventana. |
| Tormenta sin rayos sobre la localidad | No se muestra (41-66 % de avisos de más). Sigue en paralelo. |
| Puntaje de gravedad | Sigue en paralelo: todavía no hay casos medidos (la comparación con estaciones empezó el 23/11). |
| Alarma "ya llega" por estaciones | Sigue en paralelo: 1 caso verificado. |
| Avisos | Ninguno (decisión de Lucas: los avisos oficiales son del SMN). |

## Etapa F: repasos y arreglos

- **Ecos de terreno de la noche en lo que se ve:** paso nuevo `goes19/c13_radar.py` (canal 13 a la
  hora del radar, ~1 s) antes del análisis por localidad; `nowcast_flujo_optico.py` saca los ecos
  sin nube encima y los ecos fijos sueltos (lo mismo que ya hacía el nowcast en capas). También
  mejora la calibración de la lluvia (el radar ya no suma lluvia falsa sobre las estaciones).
- **Validación en vivo de "sobre la localidad"** (`celdas_localidad.py` 1.2): la verdad sale de cada
  imagen limpia con el satélite; lo acumulado se reinició.
- **Viento con los cortes del SMN:** moderado desde 40 km/h de ráfaga, fuerte desde 70 (o 45 de
  viento medio), severo desde 90. Antes "fuerte" desde 50: Bahía mostraba viento fuerte en toda la
  región casi todas las tardes ventosas (1 de cada 3 imágenes del 20 al 22/11).
- **Granizo severo solo con prueba aparte del radar** (tope <= -60 °C o muchos rayos). El 13/11 Punta
  Alta tuvo 75 dBZ en el radar crudo y cayó granizo chico.
- **Pasman** desactivada (sin datos desde fines de octubre). El control de salud ya no avisa por ella.
- **SMN:** el aviso ahora dice cuánto atraso trae su dato más nuevo (hoy ~4-5 h: el SMN publica tarde).
- **Predictor viejo:** se deja como respaldo de las trayectorias cuando el 5b no está (borrarlo
  dejaría sin respaldo a la cadena). Ya no se usa para lo que se ve.
- **Ramas de prueba:** no se pueden borrar desde esta sesión; Lucas puede hacerlo en GitHub (Branches).
