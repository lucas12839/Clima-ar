#!/usr/bin/env python3
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parent

def patch_file(rel, replacements):
    path = ROOT / rel
    text = path.read_text(encoding='utf-8')
    for label, old, new in replacements:
        n = text.count(old)
        if n != 1:
            raise RuntimeError(f'{rel}: {label}: esperadas 1 coincidencia, encontradas {n}. No se modifica.')
        text = text.replace(old, new, 1)
    path.write_text(text, encoding='utf-8')

patch_file('scripts/prediccion_rma10_ml.py', [
('ruta timestamp',
'OUTPUT_PATH = ROOT / "data/radar/seguimiento/prediccion_ml_rma10.json"\n\nRADAR_DIAMETRO_KM',
'OUTPUT_PATH = ROOT / "data/radar/seguimiento/prediccion_ml_rma10.json"\nRADAR_TIMESTAMP_PATH = ROOT / "data/radar/radar_timestamp.txt"\n\nRADAR_DIAMETRO_KM'),
('lector timestamp',
'''def get_probability(\n    model,\n    row,\n):''',
'''def read_radar_timestamp():\n    if not RADAR_TIMESTAMP_PATH.exists():\n        return None\n    try:\n        value = RADAR_TIMESTAMP_PATH.read_text(encoding="utf-8").strip()\n        return value or None\n    except Exception:\n        return None\n\n\ndef get_probability(\n    model,\n    row,\n):'''),
('selector calibracion',
'''            calibration[key] = {\n                "enabled": True,\n                "method": "PLATT_SCALING_LOGIT",\n                "samples": int(len(y)),\n                "positive_samples": int(\n                    y.sum()\n                ),\n                "a": round(\n                    a,\n                    10\n                ),\n                "b": round(\n                    b,\n                    10\n                ),\n            }\n\n            metrics[key] = {\n''',
'''            ece_raw = expected_calibration_error(y_evaluation, raw_evaluation)\n            ece_calibrated = expected_calibration_error(y_evaluation, calibrated)\n\n            use_calibrated = (\n                len(y_evaluation) >= CALIBRATION_MIN_SAMPLES\n                and ece_raw is not None\n                and ece_calibrated is not None\n                and brier_calibrated <= brier_raw\n                and ece_calibrated <= ece_raw\n            )\n\n            calibration[key] = {\n                "enabled": bool(use_calibrated),\n                "method": "PLATT_SCALING_LOGIT",\n                "samples": int(len(y)),\n                "positive_samples": int(y.sum()),\n                "a": round(a, 10),\n                "b": round(b, 10),\n                "selection": "CALIBRATED" if use_calibrated else "RAW",\n            }\n\n            metrics[key] = {\n'''),
('metricas seleccion',
'''                "ece_raw": expected_calibration_error(\n                    y_evaluation,\n                    raw_evaluation,\n                ),\n                "ece_calibrated": expected_calibration_error(\n                    y_evaluation,\n                    calibrated,\n                ),\n''',
'''                "ece_raw": ece_raw,\n                "ece_calibrated": ece_calibrated,\n                "selection": "CALIBRATED" if use_calibrated else "RAW",\n'''),
('timestamp salida',
'''    timestamp = (\n        resultado.get(\n            "imagen_procesada"\n        )\n    )\n''',
'''    timestamp = (\n        read_radar_timestamp()\n        or resultado.get(\n            "imagen_procesada"\n        )\n    )\n'''),
('flag calibracion',
'''        "calibracion_probabilistica": {\n            "aplicada": True,\n''',
'''        "calibracion_probabilistica": {\n            "aplicada": calibrated_values > 0,\n''')
])

patch_file('app.py', [
('ruta ML',
'PREDICTION_FILE = RADAR_DIR / "seguimiento" / "prediccion_evolucion.json"\nTIMESTAMP_FILE',
'PREDICTION_FILE = RADAR_DIR / "seguimiento" / "prediccion_evolucion.json"\nML_PREDICTION_FILE = RADAR_DIR / "seguimiento" / "prediccion_ml_rma10.json"\nTIMESTAMP_FILE'),
('endpoint vivo',
'''@app.get("/tormenta")\ndef tormenta() -> dict[str, Any]:''',
'''@app.get("/prediccion/en-vivo")\ndef prediccion_en_vivo() -> dict[str, Any]:\n    radar_ts = radar_timestamp()\n    age = frame_age_minutes(radar_ts)\n    ml = read_json(ML_PREDICTION_FILE) or {}\n\n    return {\n        "estado": "ok" if ml else "sin_datos",\n        "fuente_radar": FUENTE,\n        "radar": "RMA10",\n        "modo": "PREDICCION_IA_ENTRE_FRAMES",\n        "ultimo_radar_real_utc": radar_ts,\n        "ultimo_radar_edad_minutos": round(age, 2) if age is not None else None,\n        "actualizado_utc": datetime.now(timezone.utc).isoformat(),\n        "mensaje": "Predicción IA en curso · estimación entre datos de radar",\n        "prediccion": ml,\n    }\n\n\n@app.get("/tormenta")\ndef tormenta() -> dict[str, Any]:''')
])

for rel in ('scripts/prediccion_rma10_ml.py', 'app.py'):
    ast.parse((ROOT / rel).read_text(encoding='utf-8'), filename=rel)

print('OK: parche aplicado y sintaxis validada.')
