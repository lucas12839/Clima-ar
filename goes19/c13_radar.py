"""Canal 13 de GOES-19 a la hora de la imagen de radar, ANTES del análisis por localidad.

Etapa F (23/11/2026). Con 31 días de radar (herramientas/revisar_30_dias.py --limpieza
satelite): de noche el radar crudo marca ecos de terreno sin ninguna nube encima
(propagación anómala) sobre Tornquist y Médanos, y ClimaAR los mostraba como tormenta
o granizo. Con el control del satélite, Tornquist pasa de 12 a 5 días con "granizo
posible severo" y de 4 a 0 días con lluvia fuerte. El nowcast en capas ya hacía ese
control, pero el análisis por localidad (lo que se ve) corre antes de leer GOES-19.

Este paso lee solo el canal 13 (lectura parcial, ~1 s) y lo deja en el archivo temporal
que usan el análisis por localidad (nowcast_flujo_optico.py) y después el nowcast en
capas. El estado completo de GOES-19 se arma después, igual que antes. Nunca corta el
ciclo: si falla, el análisis sigue sin este control.

Uso: python -m goes19.c13_radar
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def main() -> int:
    try:
        from .goes19_engine import latest_product_object
        from .state import C13_RADAR, _hora_satelite, guardar_c13_ahora, leer_dominio, load_radar_reference_time

        referencia = load_radar_reference_time()
        meta = latest_product_object("abi_c13", now=referencia, hours_back=1)
        if meta is None:
            print("Canal 13 a la hora del radar: no hay imagen publicada.")
            return 0
        with tempfile.TemporaryDirectory() as tmp:
            dominio, lectura = leer_dominio(meta, ["CMI"], Path(tmp), "abi_c13")
        hora = _hora_satelite(meta, dominio)
        guardar_c13_ahora(dominio, hora, C13_RADAR)
        dif = None if hora is None else abs((hora - referencia).total_seconds()) / 60.0
        print(f"Canal 13 a la hora del radar: {hora} (diferencia {dif if dif is None else round(dif, 1)} min, "
              f"lectura {lectura.get('modo')} {lectura.get('segundos')} s).")
    except Exception as exc:  # nunca corta el ciclo
        print(f"AVISO: canal 13 a la hora del radar no disponible: {type(exc).__name__}: {str(exc)[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
