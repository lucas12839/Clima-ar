# Validación del punto 7: modelos por localidad (20/11/2026, tarde)

Herramientas: `herramientas/historial_wrf.py` (baja el WRF del SMN de los 26 días, en GitHub,
rama `datos-wrf-historial`) y `herramientas/medir_wrf.py` (compara con la lluvia del radar
guardado en cada localidad, limpia de clutter y de ecos falsos con el satélite).

## Error encontrado y corregido

En los archivos de 10 min del WRF del SMN, `PP` es **la lluvia de esos 10 minutos** (la
documentación del SMN lo dice: "10-min accumulated precipitation"), no el acumulado desde que
empieza la corrida. `modelos_region.py` lo restaba como si fuera acumulado, así que la lluvia del
WRF por localidad salía mal (casi siempre 0 o con saltos). Además algunas horas vienen como
14:09:59 en vez de 14:10. Corregido en la versión `modelos-region-1.1`; la corrida guardada con
la versión vieja se vuelve a bajar sola en el primer ciclo.

## ¿El WRF acierta la lluvia por localidad?

Se usa la corrida que había en vivo (la última publicada, ~4 h de demora: plazo 4-10 h).
Lluvia ≥ 1 mm en la hora, las 7 localidades, 177 horas con lluvia:

| | Detecta | Falsas alarmas |
|---|---|---|
| Hora exacta | 23 % | 65 % |
| Con ±1 h de margen | 33 % | 48 % |
| Bloque de 3 h (como `lluvia_3h`) | 27 % | 53 % |
| Región (alguna localidad, bloque de 6 h) | 37 % | 31 % |

Por localidad (±1 h): Punta Alta 48 % / 26 %, Ingeniero White 43 % / 36 %, Sierra de la Ventana
42 % / 64 %, Bahía Blanca 39 % / 50 %, Cerri 37 % / 50 %, Médanos 31 % / 48 %, Tornquist 14 % / 65 %.

Ejemplo 13/11 (Punta Alta, más de 70 mm medidos): la corrida de las 00 UTC dio 26 mm en el
punto más lluvioso de la zona y 15 mm de media; la de las 06 UTC, 11 mm.

**Veredicto:** el WRF **no sirve para decir cuándo ni dónde llueve** en una localidad (acierta 1
de cada 3 horas con lluvia y la mitad de lo que da no pasa). Sirve para el **ambiente** del día
(temperatura, humedad, presión, viento, nivel de congelamiento) y para "día con tormentas en la
región". Así se usa: en paralelo, sin decidir nada por la lluvia del WRF. Los ensambles
(ECMWF + GFS + ICON) siguen como probabilidad de ambiente inestable en 12 h; no hay historial
para medir su lluvia por localidad.

Nota: en la "verdad" del radar los núcleos con granizo dan valores de lluvia exagerados
(Marshall-Palmer sin tope); no cambia las tablas, que miden si llovió o no.
