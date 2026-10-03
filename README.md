# ClimaAR

ClimaAR es un prototipo de seguimiento de tormentas para Bahía Blanca basado en el **RMA10 del SMN**, con captura automática y análisis temporal.

## Arquitectura actual

1. **RMA10 oficial:** GitHub Actions captura el producto `ZH_MAX` aproximadamente cada 10 minutos desde la página oficial del SMN.
2. **Archivo propio:** cada captura se guarda con timestamp en `data/radar/raw/` y se mantiene `data/radar/latest.png`.
3. **dBZ:** `scripts/procesar_radar.py` decodifica los colores usando la paleta calibrada a partir de la leyenda RMA10 de Bahía Blanca.
4. **Características:** se calculan reflectividad media/máxima, áreas por umbral y centroides de las zonas convectivas.
5. **Secuencia temporal:** la API expone los últimos 4 frames para construir entradas `T-30, T-20, T-10, T`.
6. **Nowcast inicial:** `/nowcast` hace una extrapolación temporal simple de la evolución reciente. Está marcado como `baseline_experimental`; todavía no es el modelo de IA final.
7. **Primer modelo de nowcasting:** `scripts/entrenar_nowcast.py` aprende a predecir el próximo frame (10 min) a partir de las últimas cuatro observaciones RMA10. El workflow `entrenar-nowcast.yml` se ejecuta cuando ya hay suficientes secuencias.
8. **Modelo ambiental:** el workflow `procesar-historico.yml` conserva el modelo de temperatura como baseline independiente.

## API

- `/health`
- `/radar`
- `/radar/imagen`
- `/secuencia`
- `/tormenta`
- `/nowcast`
- `/modelo`
- `/modelo/nowcast`
- `/prediccion`
- `/estado`

## Fuente

El radar utilizado es el RMA10 de Bahía Blanca del Servicio Meteorológico Nacional.

> La interpretación de radar meteorológico requiere conocimiento especializado. ClimaAR no reemplaza la información oficial ni la evaluación de profesionales.

## Siguiente etapa

Con suficientes secuencias RMA10, se entrenará el modelo de nowcasting de ClimaAR usando las secuencias temporales y observaciones/eventos históricos disponibles.
