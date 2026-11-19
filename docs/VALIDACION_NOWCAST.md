# Validación del nowcast por flujo óptico

Validación retrospectiva (*hindcast*) con los frames RMA10 de SINARAME guardados
en `data/radar/historico/`, del 12/10/2026 al 19/11/2026. Se reproduce con:

```bash
python scripts/validar_nowcast_flujo.py --desde 20261012 --hasta 20261120
```

Para cada frame T con convección (≥150 píxeles ≥35 dBZ, sin clutter) y con el
frame T−10 disponible, se pronostica con flujo óptico y se compara contra lo que
el radar observó después. Referencia: **persistencia** (suponer que nada se mueve).

## Campo completo — CSI a 35 dBZ (1.020 casos convectivos)

| Plazo | Flujo óptico | Persistencia | Mejora |
|---|---|---|---|
| 10 min | 0,442 | 0,350 | +26 % |
| 20 min | 0,315 | 0,230 | +37 % |
| 30 min | 0,242 | 0,172 | +41 % |
| 60 min | 0,139 | 0,099 | +40 % |

En los días más convectivos (12–13/11) la mejora es mayor: 0,55 vs 0,41 a 10 min
y 0,32 vs 0,21 a 30 min.

## Bahía Blanca (radio 10 km, llegada de eco ≥35 dBZ en 60 min)

- 128 casos sin eco previo sobre la ciudad; 21 llegadas reales.
- Flujo óptico: 3 aciertos, 11 falsas alarmas, 18 omisiones.
- Las falsas alarmas pronosticaban 35–39 dBZ (lluvia en el umbral). Las llegadas
  reales detectadas llegaron con 40–44 dBZ.
- Las omisiones son sobre todo lluvia que **se intensifica en el lugar**, algo que
  ninguna extrapolación puede anticipar.

**Decisión:** el motor de decisión solo toma el flujo óptico como evidencia de
tormenta cuando pronostica **≥40 dBZ** llegando y la confianza es **≥0,5**. El
pronóstico ≥35 dBZ se publica como información (`data/radar/nowcast_flujo.json`).

## Limitaciones

- Extrapolación pura: no anticipa formación ni intensificación.
- Más allá de 150 km el haz pasa alto y subestima la reflectividad (confianza
  reducida hasta 0,5 a 240 km).
- Con el radar atrasado más de 15 min la confianza baja; con más de 40 min no se
  proyectan celdas.
