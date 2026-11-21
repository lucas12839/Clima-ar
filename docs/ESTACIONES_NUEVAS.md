# Estaciones nuevas (búsqueda del 20/11/2026)

Pedido de Lucas: más estaciones que den información actualizada y confiable, a las que nos
podamos conectar sin problemas y en puntos estratégicos.

## Dónde se buscó

- **Red BCP (Bolsa de Cereales y Productos de Bahía Blanca):** el mapa público tiene 55
  estaciones (40 habilitadas). Ya usábamos 9. Es la misma página y el mismo formato que ya lee
  `colector_bcp.py`, así que conectarse no tiene problemas. Miden cada minuto.
- **Red CWOP (estaciones ciudadanas abiertas):** escucha de 12 min con filtro de 250 km. Dentro
  del alcance solo transmite la que ya usamos (LU4DRH-11, Bahía Blanca noroeste). Nada nuevo.

## Sumadas (5)

| Estación | Distancia y rumbo desde Bahía | Por qué es estratégica | Datos |
|---|---|---|---|
| Punta Alta (76) | 25 km al sureste | Punta Alta no tenía estación (13/11: 70 mm) | temp., presión, viento, ráfaga |
| La Vitícola (3) | 28 km al norte | Entre Bahía y Tornquist, aguas arriba de Bahía | temp., viento, ráfaga |
| Villa Ventana (59) | 76 km al nornoreste | Junto a Sierra de la Ventana y Tornquist | temp., viento, ráfaga |
| Sierras Grandes (97) | 70 km al nornoreste | Sobre las sierras, junto a Sierra de la Ventana | temp., presión, viento, ráfaga |
| Macachín (95) | 205 km al noroeste (La Pampa) | Aguas arriba: de ahí vienen muchas tormentas | temp., presión, viento, ráfaga |

Todas tenían el último dato de hace menos de 10 min. Entran al control de calidad automático
(`calidad_y_calibracion.py`) y como puntos de control del radar; si una mide mal, se marca
dudosa y no entra. La de Punta Alta queda fuera del resumen de Bahía (radio de Bahía: 20 km).

## Descartadas

- **Sin datos hace días o meses:** Ing. White (25), Cnel. Pringles (16), Cnel. Falcón (19),
  Teniente Origone (5), Cumbre Cordón Ventana (27), Perú (38), Funke (9, 3 h de atraso).
  También **Pasman (28), que ya estaba en la lista**: sin datos hace 20 días.
- **Datos atrasados más de 15 min:** Bajo Hondo (24 min), Monte Hermoso (17 min).
- **Sin viento (no sirven para la alarma):** 17 de Agosto, Saavedra, Guaminí, El Zorro.
- **Viento en cero y sin presión (dudosas):** Carhué, San Miguel Arcángel.
- **Al este (las tormentas casi nunca vienen de ahí):** Frapal, Faro, Claromecó, González
  Chaves, San Cayetano, Cristiano Muerto, Lamadrid, Daireaux.
- **Fuera de los 240 km:** Anchorena, Salliqueló.

## Lo que falta

Al oeste y suroeste de Bahía (Médanos, Villarino, Río Colorado) no hay estaciones abiertas que
publiquen seguido. La del SMN en Río Colorado es horaria (no sirve para "ya llega").
