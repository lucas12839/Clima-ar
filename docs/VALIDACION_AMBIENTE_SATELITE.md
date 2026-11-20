# Punto 8: ambiente medido por el satélite (20/11/2026, hora de Argentina)

## Qué hace (en paralelo, no cambia lo que se ve)

`scripts/ambiente_satelite.py` → `data/ia/ambiente_satelite.json`, en cada ciclo de
Intelligence. Lee de GOES-19 solo la zona de 240 km (lectura parcial, ~2 s en total):

| Producto | Qué mide | Cada | Atraso |
|---|---|---|---|
| DSIF | energía para tormentas (CAPE), índice de elevación, K, Total Totals, Showalter | 10 min | ~3 min |
| TPWF | agua precipitable de la columna | 10 min | ~3 min |
| LVTPF / LVMPF | temperatura y humedad en 101 niveles | 10 min | ~13 min |

Píxel de 10 km; solo hay dato con cielo despejado (calidad buena = `DQF_Overall` 0).
Del perfil se calcula: temperatura y humedad en 850/700/500 hPa, punto de rocío en 850,
gradiente 700-500 hPa (granizo), nivel de 0 °C (granizo que llega al suelo) y aire seco en
capas medias (ráfagas descendentes).

- **Por localidad** (25 km alrededor) y **para toda la región** (mediana, extremo y dónde
  está el aire más inestable).
- **Antes de que se nuble:** con nubes se usa la última medición despejada (hasta 6 h),
  con su hora y su edad (`origen`).
- **Comparación con el ECMWF** en Bahía Blanca a la misma hora (`comparacion_modelo_bahia`).
- **Reglas de confiabilidad:** mismo barrido que el canal 13 del ciclo (o el anterior más
  cercano, hasta 20 min antes del radar), nunca posterior; si falta un producto se dice; en
  los refrescos entre imágenes se reutiliza lo leído (0,5 s, sin bajar nada).
  `validar_integridad_climaar.py` lo controla y el control de salud avisa si no se pudo leer.
- Detalle encontrado: LVMPF dice "percent" pero trae fracciones (0 a 1); se pasa a %.

Primer ciclo real (20/11, 17:20): temperaturas medidas a 1 °C del ECMWF en 850/700/500 hPa;
nivel de 0 °C 300 m más bajo que el modelo; CAPE medido 362 J/kg contra 570 del modelo.

## Validación con los días guardados (`herramientas/validar_ambiente_satelite.py`)

25 días (25/10 al 19/11). Satélite a las 12 y 15 hora de Argentina, ECMWF y GFS del día,
y lo que pasó en el radar entre las 12 y las 03 hora de Argentina. "Día con tormenta en la
región" = alguna imagen con 300 píxeles o más de 45 dBZ (11 de 25 días).

| Regla | Detecta | Falsas alarmas |
|---|---|---|
| Satélite: energía (CAPE) ≥ 100 J/kg a las 15 h | 60 % | 33 % |
| ECMWF: CAPE ≥ 100 J/kg a las 12 h | 55 % | 25 % |
| Satélite: índice de elevación ≤ 3,5 a las 12 h | 82 % | 47 % |

**Conclusión honesta:** ningún índice del ambiente, ni del satélite ni de los modelos,
anticipa solo qué día va a haber tormenta (con 25 días, todos quedan cerca de "decir que sí
siempre"). Del 2 al 5/11 el radar tuvo ecos fuertes con energía cero en el satélite y en
el modelo (lluvia de frente, no convección), así que la cuenta mezcla eventos distintos.

Lo que el satélite **sí aporta**:
- **Corrige al modelo:** el 11/11 y el 16/11 el ECMWF daba 1120 y 850 J/kg y no hubo
  tormentas en la región; el satélite medía 18-105. En los dos días más fuertes (12 y 13/11)
  el satélite midió 680-736 J/kg y el ECMWF 330-440: el modelo los subestimaba a la mitad.
- **Agua en la columna:** el 13/11 (más de 70 mm en Punta Alta) el satélite midió 38,5 mm,
  el valor más alto del mes; el 12/11, 31 mm. Ningún otro día pasó de 29 mm.
- Es un dato **medido** con hora, que llega 3 min después del barrido, cada 10 min.

Por eso queda en paralelo y entra en la calibración y la medición de aciertos (punto 9)
como dato del ambiente para cada fenómeno (granizo, lluvia muy eficiente, ráfagas), no
como pronóstico por sí solo. Las lecturas (`lectura`) son reglas iniciales en evaluación.
