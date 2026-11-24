import json
from datetime import datetime, timedelta, timezone

from scripts.recorte_historial import recortar_csv, recortar_jsonl

AHORA = datetime(2026, 11, 24, 12, tzinfo=timezone.utc)


def test_recortar_csv_deja_encabezado_y_ultimos_dias(tmp_path):
    p = tmp_path / "h.csv"
    filas = ["timestamp_utc,a\n"]
    for d in (40, 31, 29, 1):
        filas.append(f"{(AHORA - timedelta(days=d)).isoformat()},{d}\n")
    filas.append("sin-fecha,9\n")
    p.write_text("".join(filas), encoding="utf-8")
    assert recortar_csv(p, ahora=AHORA) == 2
    lineas = p.read_text(encoding="utf-8").splitlines()
    assert lineas[0] == "timestamp_utc,a"
    assert [l.split(",")[1] for l in lineas[1:]] == ["29", "1", "9"]
    assert recortar_csv(p, ahora=AHORA) == 0


def test_recortar_jsonl(tmp_path):
    p = tmp_path / "log.jsonl"
    regs = [{"actualizado_utc": (AHORA - timedelta(days=d)).isoformat()} for d in (45, 2)]
    regs.append({"obsTime_utc": (AHORA - timedelta(days=50)).isoformat()})
    p.write_text("".join(json.dumps(r) + "\n" for r in regs) + "roto\n", encoding="utf-8")
    assert recortar_jsonl(p, campo="actualizado_utc", otros=("obsTime_utc",), ahora=AHORA) == 3
    assert len(p.read_text(encoding="utf-8").splitlines()) == 1


def test_archivo_que_no_existe(tmp_path):
    assert recortar_csv(tmp_path / "no.csv") == 0
    assert recortar_jsonl(tmp_path / "no.jsonl") == 0
