# Validación con eventos históricos (paso 6)

Se usaron las imágenes públicas del SMN **"Bahía Blanca-SINARAME ZH MAX"** (mismo
radar RMA10 y mismo producto que ClimaAR) de tres temporales. Las imágenes no son el
PNG de paleta de SINARAME, así que se decodifican con
`herramientas/decodificar_imagen_smn.py`:

- Hora: título de la imagen (OCR, hora local + 3 h = UTC).
- Geometría: círculo de 240 km ajustado en las imágenes (centro 328,5 / 417,8 px,
  radio 310 px), remuestreado a la grilla de ClimaAR (512×512).
- dBZ: barra de colores de la imagen; etiquetas, rutas y polígonos de aviso se
  descartan por no ser colores de la escala.
- Cadencia real de las imágenes: **9 minutos** (con huecos de 18–36 min).

Se reproduce con:

```bash
python herramientas/validar_eventos.py <carpeta_evento> [...] --salida resultados.json
```

En cada frame se corre el mismo nowcast por flujo óptico de producción (frame
anterior a 5–30 min) y se compara con lo que después llegó a Bahía Blanca (radio
10 km, eco que cubra ≥5 km²). "Anticipación" = cuántos minutos antes el nowcast
pronosticó la llegada con un error de ±20 min.

## Resultados

| Evento | Frames | Llegada ≥35 dBZ | Aviso | Llegada ≥45 dBZ | Aviso | Máx. sobre la ciudad | Velocidad (mediana / p90) |
|---|---|---|---|---|---|---|---|
| Inundación 7/3/2025 | 76 | 01:46 UTC | – (ya llovía al primer frame) | 06:34 UTC | **36 min** | 49,9 dBZ | 29 / 100 km/h |
| Viento (bow echo) 16/12/2023 | 79 | 19:19 UTC | **18 min** | 21:50 UTC | 18 min | 49,3 dBZ | 50 / 79 km/h |
| Granizo 2/2/2025 | 18 | 17:49 UTC | 0 | 17:58 UTC | **9 min** | 55,3 dBZ | 57 / 69 km/h |

## Lectura

- El nowcast detecta los tres eventos y estima un movimiento coherente con lo
  observado (bow echo hacia el SE a 55–84 km/h, coherencia 0,72–0,98).
- La anticipación queda limitada a 9–36 min por la física del caso: en el bow
  echo los núcleos ≥45 dBZ estaban a 20–30 km desde las 18:00 UTC pero pasaron
  al costado; lo que llegó a la ciudad fue una parte de la línea que se
  intensificó al acercarse. En la inundación la convección se formó y quedó
  estacionaria sobre la ciudad durante horas. Ninguna extrapolación anticipa
  formación ni intensificación.
- **Señal no aprovechada hoy:** en el bow echo, entre 19:37 y 19:46 UTC la línea
  avanzaba a 75–84 km/h con coherencia 0,94–0,98 — la firma típica de viento
  destructivo. ClimaAR todavía no la usa para alertar viento.
- El máximo de reflectividad sobre la ciudad (≈50 dBZ) no refleja por sí solo la
  severidad (290 mm de lluvia el 7/3, ráfagas el 16/12): la clasificación de
  severidad necesita además duración, velocidad del sistema, rayos y estaciones.

## Pendiente para el próximo plan

- Alerta de viento por líneas rápidas y coherentes (bow echo).
- Acumulado de lluvia por duración del eco sobre la ciudad (inundación).
- Escala completa de tipos de tiempo (llovizna → tormenta potencialmente severa).
- Conectar las estaciones y los rayos de GOES-19 a la severidad.
