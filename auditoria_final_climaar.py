#!/usr/bin/env python3
"""ClimaAR - auditoría y cierre final, conservador y transaccional.

El script NO escribe ningún archivo hasta que todas las modificaciones
esperadas coinciden exactamente y todos los archivos Python modificados
superan la validación de sintaxis.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TARGETS = [
    ".github/workflows/climaar-auto-radar.yml",
    ".github/workflows/probar-radar-smn.yml",
    ".github/workflows/climaar-intelligence.yml",
    "scripts/prediccion_rma10_ml.py",
    "seguimiento_temporal.py",
    "app.py",
]


def fail(message: str) -> None:
    raise SystemExit(f"ABORTADO: {message}")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        fail(f"{label}: se esperaban 1 coincidencia y hay {count}.")
    return text.replace(old, new, 1)


def move_block(text: str, start: str, end: str, destination: str, label: str) -> str:
    if text.count(start) != 1:
        fail(f"{label}: marcador inicial no único.")
    if text.count(end) != 1:
        fail(f"{label}: marcador final no único.")
    if text.count(destination) != 1:
        fail(f"{label}: destino no único.")

    start_i = text.index(start)
    end_i = text.index(end, start_i)
    block = text[start_i:end_i]
    without = text[:start_i] + text[end_i:]
    dest_i = without.index(destination)
    return without[:dest_i] + block + without[dest_i:]


def patch_auto_radar(text: str) -> str:
    return replace_once(
        text,
        '''on:\n  schedule:\n    - cron: "3-59/10 * * * *"\n  workflow_dispatch:\n''',
        '''on:\n  workflow_dispatch:\n''',
        "scheduler duplicado de RMA10",
    )


def patch_radar_workflow(text: str) -> str:
    text = replace_once(
        text,
        '''permissions:\n  contents: write\n  actions: read\n\njobs:\n''',
        '''permissions:\n  contents: write\n  actions: read\n\nconcurrency:\n  group: climaar-rma10-captura\n  cancel-in-progress: false\n\njobs:\n''',
        "concurrencia del radar",
    )

    continuity_start = '''      # ============================================================\n      # 13. VERIFICAR CONTINUIDAD\n      # ============================================================\n'''
    continuity_end = '''      # ============================================================\n      # 14. PREPARAR HISTORIAL CONTINUO\n      # ============================================================\n'''
    mask_marker = '''      # ============================================================\n      # 11. GENERAR MÁSCARA ESTÁTICA\n      # ============================================================\n'''

    text = move_block(
        text,
        continuity_start,
        continuity_end,
        mask_marker,
        "orden de validación de continuidad",
    )

    text = replace_once(
        text,
        "# 13. VERIFICAR CONTINUIDAD",
        "# 11. VERIFICAR CONTINUIDAD",
        "renumeración continuidad",
    )
    text = replace_once(
        text,
        "# 11. GENERAR MÁSCARA ESTÁTICA",
        "# 12. GENERAR MÁSCARA ESTÁTICA",
        "renumeración máscara",
    )
    text = replace_once(
        text,
        "# 12. PROCESAR RADAR",
        "# 13. PROCESAR RADAR",
        "renumeración procesamiento",
    )

    reject_step = '''      # ============================================================\n      # RECHAZAR FRAME FUERA DE SECUENCIA\n      # ============================================================\n\n      - name: Rechazar frame fuera de secuencia\n        if: >\n          steps.frame.outputs.new_frame == 'true' &&\n          steps.continuidad.outputs.accepted != 'true'\n        shell: bash\n        run: |\n          set -e\n          echo "ERROR: RMA10 entregó un frame repetido o atrasado."\n          echo "No se procesará ni se persistirá como nuevo frame."\n          exit 1\n\n\n'''
    destination = '''      # ============================================================\n      # 12. GENERAR MÁSCARA ESTÁTICA\n      # ============================================================\n'''
    text = replace_once(
        text,
        destination,
        reject_step + destination,
        "rechazo de frame fuera de secuencia",
    )

    # Estas son exactamente las dos etapas que procesan el frame.
    old_if = "if: steps.frame.outputs.new_frame == 'true'"
    if text.count(old_if) < 2:
        fail("condiciones de máscara/procesamiento no encontradas.")
    text = text.replace(
        old_if,
        "if: steps.continuidad.outputs.accepted == 'true'",
        2,
    )

    # La persistencia final también exige continuidad aceptada.
    text = replace_once(
        text,
        '''if: steps.frame.outputs.new_frame == 'true'\n        shell: bash\n        run: |\n\n          set -e\n\n          git config user.name''',
        '''if: steps.continuidad.outputs.accepted == 'true'\n        shell: bash\n        run: |\n\n          set -e\n\n          git config user.name''',
        "persistencia de estado RMA10",
    )

    return text


def patch_ml(text: str) -> str:
    text = replace_once(
        text,
        '''    timestamp = (\n        read_radar_timestamp()\n        or resultado.get(\n            "imagen_procesada"\n        )\n    )\n\n    output = {\n''',
        '''    timestamp = read_radar_timestamp()\n\n    if not timestamp:\n        raise SystemExit(\n            "No existe un timestamp real de RMA10. "\n            "No se generará una predicción ML sin frame radar válido."\n        )\n\n    output = {\n''',
        "timestamp ML estricto",
    )

    text = replace_once(
        text,
        '''        "calibracion_probabilistica": {\n            "aplicada": calibrated_values > 0,\n            "metodo": "PLATT_SCALING_LOGIT",\n''',
        '''        "calibracion_probabilistica": {\n            "aplicada": any(\n                bool(model_info.get("enabled"))\n                for model_info in probability_calibration["models"].values()\n                if isinstance(model_info, dict)\n            ),\n            "metodo": "PLATT_SCALING_LOGIT",\n''',
        "estado de calibración ML",
    )
    return text


def patch_intelligence_workflow(text: str) -> str:
    text = replace_once(
        text,
        '''          if tracks != len(predicciones):\n              raise SystemExit(\n                  "ERROR: cantidad de tracks y predicciones no coincide."\n              )\n\n          print(\n              "===== ML RMA10 ====="\n          )\n''',
        '''          if tracks != len(predicciones):\n              raise SystemExit(\n                  "ERROR: cantidad de tracks y predicciones no coincide."\n              )\n\n          radar_ts = Path(\n              "data/radar/radar_timestamp.txt"\n          ).read_text(encoding="utf-8").strip()\n\n          ml_ts = data.get("frame_radar")\n\n          if not ml_ts:\n              raise SystemExit(\n                  "ERROR: ML RMA10 no contiene frame_radar."\n              )\n\n          if str(ml_ts).strip() != radar_ts:\n              raise SystemExit(\n                  "ERROR: frame_radar ML no coincide con el timestamp RMA10."\n              )\n\n          print(\n              "===== ML RMA10 ====="\n          )\n''',
        "validación estricta ML/RMA10",
    )

    text = replace_once(
        text,
        '''              if radar_normalizado != nowcast_normalizado:\n                  print(\n                      "AVISO: timestamp radar/nowcast diferente."\n                  )\n''',
        '''              if radar_normalizado != nowcast_normalizado:\n                  raise SystemExit(\n                      "ERROR: timestamp radar/nowcast diferente."\n                  )\n''',
        "validación estricta radar/nowcast",
    )
    return text


def patch_tracking(text: str) -> str:
    text = replace_once(
        text,
        '''    activos_previos = [\n\n        track\n\n        for track in estado[\n            "tracks"\n        ]\n\n        if (\n            track.get(\n                "activo"\n            ) is True\n\n            and\n\n            track.get(\n                "ultimo_frame"\n            ) == anterior_ts\n        )\n    ]\n\n    candidatos = []\n''',
        '''    activos_previos = []\n    intervalos_por_track = {}\n\n    for track in estado["tracks"]:\n\n        if track.get("activo") is not True:\n            continue\n\n        track_ts = parse_iso(\n            track.get("ultimo_frame")\n        )\n\n        if track_ts is None:\n            continue\n\n        if track_ts > actual_dt:\n            continue\n\n        gap_minutos = (\n            actual_dt - track_ts\n        ).total_seconds() / 60.0\n\n        if gap_minutos <= 0:\n            continue\n\n        if gap_minutos > MAX_INTERVALO * (MAX_MISSED_FRAMES + 1):\n            continue\n\n        missed = entero(\n            track.get("missed_frames", 0)\n        )\n\n        if missed > MAX_MISSED_FRAMES:\n            continue\n\n        activos_previos.append(track)\n        intervalos_por_track[track["track_id"]] = gap_minutos\n\n    candidatos = []\n''',
        "recuperación de tracks perdidos",
    )

    text = replace_once(
        text,
        '''            comp = compatibilidad(\n                anterior,\n                actual,\n                intervalo,\n                track\n            )\n''',
        '''            intervalo_track = intervalos_por_track.get(\n                track["track_id"],\n                intervalo,\n            )\n\n            comp = compatibilidad(\n                anterior,\n                actual,\n                intervalo_track,\n                track\n            )\n''',
        "intervalo de asociación de track",
    )

    text = replace_once(
        text,
        '''        actualizar_track(\n            track,\n            nucleo,\n            radar_ts,\n            intervalo,\n            calidad\n        )\n''',
        '''        intervalo_track = intervalos_por_track.get(\n            track["track_id"],\n            intervalo,\n        )\n\n        actualizar_track(\n            track,\n            nucleo,\n            radar_ts,\n            intervalo_track,\n            calidad\n        )\n''',
        "actualización de track recuperado",
    )
    return text


def patch_app(text: str) -> str:
    old = '''@app.get("/prediccion/en-vivo")\ndef prediccion_en_vivo() -> dict[str, Any]:\n    radar_ts = radar_timestamp()\n    age = frame_age_minutes(radar_ts)\n    ml = read_json(ML_PREDICTION_FILE) or {}\n\n    return {\n        "estado": "ok" if ml else "sin_datos",\n        "fuente_radar": FUENTE,\n        "radar": "RMA10",\n        "modo": "PREDICCION_IA_ENTRE_FRAMES",\n        "ultimo_radar_real_utc": radar_ts,\n        "ultimo_radar_edad_minutos": round(age, 2) if age is not None else None,\n        "actualizado_utc": datetime.now(timezone.utc).isoformat(),\n        "mensaje": "Predicción IA en curso · estimación entre datos de radar",\n        "prediccion": ml,\n    }\n'''
    new = '''def _interp(a: float | None, b: float | None, fraction: float) -> float | None:\n    if a is None and b is None:\n        return None\n    if a is None:\n        return b\n    if b is None:\n        return a\n    return a + (b - a) * fraction\n\n\ndef _live_ml_projection(ml: dict[str, Any], elapsed: float) -> dict[str, Any]:\n    tracks = ml.get("predicciones", [])\n    if not isinstance(tracks, list):\n        return {"tracks": []}\n\n    horizons = [5, 10, 20, 30]\n    result = []\n\n    for track in tracks:\n        if not isinstance(track, dict):\n            continue\n\n        raw = track.get("horizontes", {})\n        if not isinstance(raw, dict):\n            continue\n\n        points = []\n        for h in horizons:\n            item = raw.get(str(h))\n            if isinstance(item, dict):\n                points.append((h, item))\n\n        if not points:\n            continue\n\n        if elapsed <= points[0][0]:\n            left_h, left = points[0]\n            right_h, right = points[0]\n        elif elapsed >= points[-1][0]:\n            left_h, left = points[-1]\n            right_h, right = points[-1]\n        else:\n            left_h, left = points[0]\n            right_h, right = points[-1]\n            for index in range(len(points) - 1):\n                h1, p1 = points[index]\n                h2, p2 = points[index + 1]\n                if h1 <= elapsed <= h2:\n                    left_h, left = h1, p1\n                    right_h, right = h2, p2\n                    break\n\n        fraction = 0.0 if right_h == left_h else (elapsed - left_h) / (right_h - left_h)\n        fraction = max(0.0, min(1.0, fraction))\n\n        def value(target: str) -> float | None:\n            a = left.get(target)\n            b = right.get(target)\n            try:\n                a = float(a) if a is not None else None\n            except (TypeError, ValueError):\n                a = None\n            try:\n                b = float(b) if b is not None else None\n            except (TypeError, ValueError):\n                b = None\n            return _interp(a, b, fraction)\n\n        result.append({\n            "track_id": track.get("track_id"),\n            "lat": track.get("lat"),\n            "lon": track.get("lon"),\n            "distance_km": track.get("distance_km"),\n            "rumbo_grados": track.get("motion_bearing_deg"),\n            "velocidad_px_min": track.get("speed_px_min"),\n            "probabilidades": {\n                "fortalecimiento": value("strengthened"),\n                "debilitamiento": value("weakened"),\n                "hacia_bahia": value("toward_bahia"),\n            },\n            "horizonte_base_min": [left_h, right_h],\n        })\n\n    return {"tracks": result}\n\n\n@app.get("/prediccion/en-vivo")\ndef prediccion_en_vivo() -> dict[str, Any]:\n    radar_ts = radar_timestamp()\n    age = frame_age_minutes(radar_ts)\n    ml = read_json(ML_PREDICTION_FILE) or {}\n\n    if not ml or not radar_ts:\n        return {\n            "estado": "sin_datos",\n            "fuente_radar": FUENTE,\n            "radar": "RMA10",\n            "modo": "PREDICCION_ENTRE_FRAMES",\n            "ultimo_radar_real_utc": radar_ts,\n            "ultimo_radar_edad_minutos": round(age, 2) if age is not None else None,\n            "actualizado_utc": datetime.now(timezone.utc).isoformat(),\n            "prediccion": ml,\n        }\n\n    elapsed = max(0.0, min(30.0, age or 0.0))\n\n    return {\n        "estado": "ok",\n        "fuente_radar": FUENTE,\n        "radar": "RMA10",\n        "modo": "PREDICCION_ENTRE_FRAMES",\n        "base": "ML_RMA10",\n        "ultimo_radar_real_utc": radar_ts,\n        "ultimo_radar_edad_minutos": round(age, 2) if age is not None else None,\n        "minutos_proyectados": round(elapsed, 2),\n        "actualizado_utc": datetime.now(timezone.utc).isoformat(),\n        "mensaje": "Predicción entre frames · proyección basada en el último radar real",\n        "prediccion_ml": ml,\n        "proyeccion_en_vivo": _live_ml_projection(ml, elapsed),\n    }\n'''
    return replace_once(text, old, new, "predicción en vivo")


def validate_python(path: str, text: str) -> None:
    try:
        ast.parse(text, filename=path)
    except SyntaxError as exc:
        fail(f"sintaxis inválida en {path}: {exc}")


def main() -> None:
    original = {}
    for path in TARGETS:
        p = ROOT / path
        if not p.exists():
            fail(f"no existe {path}")
        original[path] = p.read_text(encoding="utf-8")

    updated = dict(original)

    updated[".github/workflows/climaar-auto-radar.yml"] = patch_auto_radar(
        updated[".github/workflows/climaar-auto-radar.yml"]
    )
    updated[".github/workflows/probar-radar-smn.yml"] = patch_radar_workflow(
        updated[".github/workflows/probar-radar-smn.yml"]
    )
    updated["scripts/prediccion_rma10_ml.py"] = patch_ml(
        updated["scripts/prediccion_rma10_ml.py"]
    )
    updated[".github/workflows/climaar-intelligence.yml"] = patch_intelligence_workflow(
        updated[".github/workflows/climaar-intelligence.yml"]
    )
    updated["seguimiento_temporal.py"] = patch_tracking(
        updated["seguimiento_temporal.py"]
    )
    updated["app.py"] = patch_app(updated["app.py"])

    # Preflight completo: ningún archivo se escribe hasta llegar acá.
    validate_python("app.py", updated["app.py"])
    validate_python(
        "scripts/prediccion_rma10_ml.py",
        updated["scripts/prediccion_rma10_ml.py"],
    )
    validate_python(
        "seguimiento_temporal.py",
        updated["seguimiento_temporal.py"],
    )

    checks = {
        ".github/workflows/climaar-auto-radar.yml": [
            'on:\n  workflow_dispatch:',
        ],
        ".github/workflows/probar-radar-smn.yml": [
            "group: climaar-rma10-captura",
            "# 11. VERIFICAR CONTINUIDAD",
            "# 12. GENERAR MÁSCARA ESTÁTICA",
            "# 13. PROCESAR RADAR",
            "Rechazar frame fuera de secuencia",
            "steps.continuidad.outputs.accepted == 'true'",
        ],
        ".github/workflows/climaar-intelligence.yml": [
            "frame_radar ML no coincide con el timestamp RMA10",
            "timestamp radar/nowcast diferente",
        ],
        "scripts/prediccion_rma10_ml.py": [
            "No existe un timestamp real de RMA10",
            '"aplicada": any(',
        ],
        "seguimiento_temporal.py": [
            "intervalos_por_track",
            "MAX_MISSED_FRAMES + 1",
        ],
        "app.py": [
            "def _live_ml_projection",
            '"modo": "PREDICCION_ENTRE_FRAMES"',
        ],
    }

    for path, tokens in checks.items():
        for token in tokens:
            if token not in updated[path]:
                fail(f"validación final: falta en {path}: {token}")

    if 'cron: "3-59/10 * * * *"' in updated[".github/workflows/climaar-auto-radar.yml"]:
        fail("el scheduler duplicado de RMA10 sigue activo")

    changed = [p for p in TARGETS if updated[p] != original[p]]
    if len(changed) != len(TARGETS):
        fail(
            "la auditoría no produjo exactamente los 6 cambios esperados: "
            + ", ".join(changed)
        )

    for path in changed:
        (ROOT / path).write_text(updated[path], encoding="utf-8")

    print("OK: auditoría final aplicada.")
    print("OK: 6 archivos modificados.")
    print("OK: sintaxis Python validada antes de escribir.")
    print("OK: no se tocó la estructura de captura, máscara o Storm Engine fuera de los puntos auditados.")


if __name__ == "__main__":
    main()
