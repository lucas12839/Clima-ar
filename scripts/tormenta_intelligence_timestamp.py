import sys
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

RADAR_FILE = (
    BASE_DIR
    / "data"
    / "radar"
    / "radar_nowcast.json"
)


def parse_dt(value):
    if not value:
        return None

    text = str(value).strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    dt = datetime.fromisoformat(text)

    if dt.tzinfo is None:
        dt = dt.replace(
            tzinfo=timezone.utc
        )

    return dt.astimezone(
        timezone.utc
    )


def main():
    if not RADAR_FILE.exists():
        raise RuntimeError(
            f"No existe {RADAR_FILE}"
        )

    import json

    with RADAR_FILE.open(
        "r",
        encoding="utf-8",
    ) as fh:
        radar = json.load(fh)

    nowcast = radar.get(
        "nowcast"
    )

    if not isinstance(
        nowcast,
        dict,
    ):
        raise RuntimeError(
            "radar_nowcast.json no contiene nowcast"
        )

    radar_timestamp = parse_dt(
        nowcast.get(
            "frame_actual_utc"
        )
    )

    if radar_timestamp is None:
        raise RuntimeError(
            "No existe frame_actual_utc válido"
        )

    import tormenta_intelligence as engine

    original_now_dt = engine.now_dt

    try:
        engine.now_dt = (
            lambda: radar_timestamp
        )

        print(
            "Storm Engine sincronizado con "
            f"radar: {radar_timestamp.isoformat()}"
        )

        return int(
            engine.main()
        )

    finally:
        engine.now_dt = original_now_dt


if __name__ == "__main__":
    sys.exit(main())
