"""Recorte de los registros que crecen sin fin (24/11/2026).

historial_meteorologico.csv, historial_inteligencia_tormenta.csv y
data/sazb/ingesta_log.jsonl sumaban una fila por ciclo desde octubre y nadie
los recortaba (1-2 MB cada uno y creciendo; cada guardado en main copia el
archivo entero). Se conservan los últimos DIAS días.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

DIAS = 30


def _fecha(texto):
    try:
        dt = datetime.fromisoformat(str(texto).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _escribir(path: Path, lineas: list[str]) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.writelines(lineas)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def recortar_csv(path, dias=DIAS, ahora=None) -> int:
    """Deja el encabezado y las filas de los últimos `dias` (primera columna = fecha).
    Devuelve cuántas filas sacó. Las filas sin fecha legible se conservan."""
    path = Path(path)
    if not path.exists():
        return 0
    limite = (ahora or datetime.now(timezone.utc)) - timedelta(days=dias)
    with path.open(encoding="utf-8", newline="") as f:
        lineas = f.readlines()
    if len(lineas) < 2:
        return 0
    cuerpo = []
    for linea in lineas[1:]:
        dt = _fecha(linea.split(",", 1)[0])
        if dt is None or dt >= limite:
            cuerpo.append(linea)
    sacadas = len(lineas) - 1 - len(cuerpo)
    if sacadas:
        _escribir(path, [lineas[0]] + cuerpo)
    return sacadas


def recortar_jsonl(path, campo="obsTime_utc", dias=DIAS, ahora=None, otros=("utc", "timestamp_utc")) -> int:
    """Deja las líneas de los últimos `dias` según `campo` (o los otros nombres)."""
    path = Path(path)
    if not path.exists():
        return 0
    limite = (ahora or datetime.now(timezone.utc)) - timedelta(days=dias)
    with path.open(encoding="utf-8") as f:
        lineas = f.readlines()
    quedan = []
    for linea in lineas:
        try:
            dato = json.loads(linea)
        except ValueError:
            continue  # línea rota: se descarta
        dt = None
        if isinstance(dato, dict):
            for c in (campo, *otros):
                dt = _fecha(dato.get(c)) if dato.get(c) else None
                if dt is not None:
                    break
        if dt is None or dt >= limite:
            quedan.append(linea)
    sacadas = len(lineas) - len(quedan)
    if sacadas:
        _escribir(path, quedan)
    return sacadas
