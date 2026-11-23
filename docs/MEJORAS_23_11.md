# Mejoras del 23/11/2026 (noche): calibración, falsas alarmas, trayectoria y tormenta real

Pedido de Lucas: "hacé de la 1 a la 4, dejá todo bien definido y mejorado, sin errores ocultos".
Todo medido con datos reales; nada inventado. Horarios en hora de Argentina salvo que diga UTC.

## 1. Calibrar la intensidad con datos reales

**Lluvia.** Se bajó el historial oficial del SMN (servicio WIS2) del 23/10 al 23/11 para las
estaciones dentro de 170 km (Bahía Blanca Aero, Pigüé, Coronel Suárez, Tres Arroyos, Río Colorado)
y se comparó la lluvia de cada 6 h con la que estima el radar crudo en el mismo lugar y las mismas
horas (134 ventanas con radar completo).

| | Resultado |
|---|---|
| Ventanas en que llovió (>= 1 mm) | 6 (27 mm en total, 4 días) |
| El radar vio lluvia cuando llovió | **100 %** |
| El radar marcó lluvia sin que lloviera | 6 ventanas |
| Lluvia real / lluvia del radar sin corregir | 0,25 (mediana) a 0,29 (total) |
| Factor en uso | 0,40 (Punta Alta 13/11) |

Fue un mes casi seco: **no alcanza para cambiar el factor** (mínimo: 12 ventanas, 4 días, 40 mm).
Los dos números apuntan al mismo lado (el radar crudo exagera la lluvia 2,5 a 4 veces) y el 0,40
queda dentro de lo medido. Lo nuevo es que ahora se junta solo:

- `scripts/estaciones/calibracion_smn.py` + `climaar-calibracion.yml` (2 veces por día, mira 4 días
  para atrás). Registro: `data/ia/calibracion/registro_smn.json`; factor:
  `data/ia/calibracion/factor_lluvia_smn.json`.
- Orden del factor: estaciones en vivo (hora por hora) -> SMN (cada 6 h) -> inicial (0,40).
- El informe mensual lo muestra; el control de salud avisa si deja de correr más de 3 días.

**Viento y granizo.** No se pueden calibrar con lo que hay: en 31 días el SMN publicó solo 7 ráfagas
en Bahía Aero (máxima 54 km/h) y ningún reporte de granizo. El viento usa los cortes oficiales del SMN
(40 / 70 / 90 km/h) y el granizo severo necesita una prueba aparte del radar (tope muy frío o muchos
rayos). Se juntan casos solos: puntaje de gravedad contra lo medido (`verificar_puntaje.py`).

## 2. Falsas alarmas de "llega una tormenta a la zona (10 km)"

`herramientas/analizar_falsas_alarmas.py`, 6 días con tormentas, 242 casos, radar 20 min tarde,
mismo método que lo que se ve (2 de 25 caminos): detectadas 73 %, falsas alarmas 27 %.

| Qué pasó de verdad en los avisos que sobraron | |
|---|---|
| La tormenta pasó a 10-25 km (de costado) | 47 % |
| Llegó, pero un poco más débil (30-35 dBZ) | 36 % |
| Se deshizo antes de llegar | 5 % |
| Nada cerca | 13 % |

**8 de cada 10 "falsas alarmas" son casi aciertos**: la tormenta pasó cerca o llegó lluvia. Los avisos
sin nada cerca son ~5 de cada 100. Se probaron filtros con lo que ya se sabe a la hora del aviso
(3 o 4 caminos de 25, no avisar si se está debilitando): bajan las falsas alarmas apenas 1 punto y
se pierden detecciones (hasta 13 puntos). **Se deja como está.** Para mostrar: "llega a la zona o
pasa cerca" describe mejor lo que pasa (decisión de Lucas para la interfaz).

## 3. Trayectoria en vivo

- El control en vivo (8 km a 10 min) y la simulación del mismo día (5,5 km; 23/11 de 14 a 18 h)
  coinciden: no hay error de la cadena. Ese día las tormentas cambiaban mucho de forma.
- Con los 8 días de referencia la trayectoria del 5b da **4,5 / 9,5 / 14,7 km a 10 / 20 / 30 min**,
  igual que lo documentado.
- Se probó mezclar el movimiento del campo con la velocidad propia de cada tormenta (30 % y 60 %):
  igual o peor (4,7 y 5,4 km a 10 min). **Se deja como está.**

## 4. Tormenta fuerte real: 13/11 en Punta Alta

`herramientas/simular_dia_real.py`: las imágenes crudas de SINARAME de ese día, una por una, por la
cadena completa (radar e Intelligence con todas sus validaciones), con 17 min de atraso como en vivo.
Sin conexión: sin satélite, rayos ni estaciones de ese día (como cuando esas fuentes se caen).
Real: dos tormentas intensas, 35 mm entre las 14:00 y las 15:30, 76,7 mm en el día, granizo chico.

| Hora en que se veía | Punta Alta en ClimaAR |
|---|---|
| 13:47 | "llega en 28 min" |
| 13:57 | llovizna, "llega en 23 min" |
| 14:37 | lluvia moderada con granizo posible |
| 14:47 a 14:57 | **lluvia fuerte con granizo probable** |
| 15:27 a 15:37 | **lluvia posible severa con granizo probable** |
| 16:37 | segunda tormenta: "llega en 3 min" |
| 16:57 a 17:07 | lluvia moderada a fuerte con granizo probable |

- Avisó la llegada ~13 min antes de que empezara a llover, con el radar 17 min atrasado.
- Granizo: "probable", nunca "severo" (cayó granizo chico): correcto.
- 0 pasos fallados en 32 imágenes.

### Errores ocultos que encontró esta prueba (arreglados)
- **Un paso "en paralelo" que fallaba cortaba TODO el ciclo**: la validación final rechazaba los
  archivos viejos del 5b, las trayectorias, los indicios severos, las celdas por localidad y el
  ambiente del satélite, aunque lo que se ve ya usaba el respaldo. Ahora se aceptan solo si lo que se
  ve no los usó; el control de salud avisa.
- **Un fallo grave del 5b (memoria, cuelgue) cortaba el ciclo**: el 5b, las trayectorias, el puntaje y
  las tormentas en formación ahora tienen tiempo máximo y respaldo.
- **Si Open-Meteo no respondía más de 3 h, no se publicaba nada** (tampoco el radar). Ahora se usa el
  pronóstico anterior hasta 9 h (cubre de 3 h antes a 12 h después) y el control de salud avisa si
  pasa más de 1,5 h sin actualizarse.
