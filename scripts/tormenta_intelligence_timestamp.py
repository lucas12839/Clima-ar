from __future__ import annotations

import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]

RADAR_FILE = (
    BASE
    / "data"
    / "radar"
    / "radar_nowcast.json"
)


def parse_timestamp(value: str) -> datetime:
    value = value.strip()

    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    dt = datetime.fromisoformat(value)

    if dt.tzinfo is None:
        dt = dt.replace(
            tzinfo=timezone.utc
        )

    return dt.astimezone(timezone.utc)


def load_radar_timestamp() -> datetime:
    if not RADAR_FILE.exists():
        raise FileNotFoundError(
            f"No existe {RADAR_FILE}"
        )

    import json

    with RADAR_FILE.open(
        "r",
        encoding="utf-8",
    ) as fh:
        data = json.load(fh)

    nowcast = data.get("nowcast")

    if not isinstance(nowcast, dict):
        raise RuntimeError(
            "radar_nowcast.json no contiene "
            "un bloque nowcast válido."
        )

    timestamp = nowcast.get(
        "frame_actual_utc"
    )

    if not timestamp:
        raise RuntimeError(
            "No existe nowcast.frame_actual_utc."
        )

    return parse_timestamp(timestamp)


def main() -> None:
    radar_timestamp = load_radar_timestamp()

    print(
        "Timestamp real del radar:",
        radar_timestamp.isoformat(),
    )

    # Importamos Storm Engine después de haber
    # preparado el timestamp real del radar.
    sys.path.insert(
        0,
        str(BASE / "scripts"),
    )

    engine = importlib.import_module(
        "tormenta_intelligence"
    )

    original_now_dt = getattr(
        engine,
        "now_dt",
        None,
    )

    if original_now_dt is None:
        raise RuntimeError(
            "tormenta_intelligence.py no expone "
            "la función now_dt()."
        )

    try:
        engine.now_dt = (
            lambda: radar_timestamp
        )

        print(
            "Storm Engine ejecutándose con "
            "el timestamp del frame Radar."
        )

        result = engine.main()

    finally:
        engine.now_dt = original_now_dt

    print(
        "Storm Engine finalizado correctamente."
    )

    return result


if __name__ == "__main__":
    main()
