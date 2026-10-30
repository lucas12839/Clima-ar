from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
from PIL import Image
from scipy import ndimage

HISTORICO = Path("data/radar/historico")
RADAR_TIMESTAMP = Path("data/radar/radar_timestamp.txt")
SALIDA = Path("data/radar/mascara_ecos_estaticos.png")
INFO = Path("data/radar/mascara_ecos_estaticos.json")
CONFIANZA = Path("data/radar/mascara_ecos_estaticos_confianza.png")

MAX_FRAMES = 40
PERSISTENCIA_MINIMA = 0.65
RADIO_TOLERANCIA_PX = 6
MIN_FRAMES_HISTORICOS = 6


def cargar_timestamp():
    if not RADAR_TIMESTAMP.exists():
        raise RuntimeError("No existe data/radar/radar_timestamp.txt")
    value = RADAR_TIMESTAMP.read_text(encoding="utf-8").strip()
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception as exc:
        raise RuntimeError(f"Timestamp de radar inválido: {value}") from exc


def timestamp_archivo(path):
    try:
        nombre = path.stem
        if len(nombre) == 14 and nombre.isdigit():
            return datetime.strptime(nombre, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        if len(nombre) == 15 and nombre[8] == "_":
            return datetime.strptime(nombre, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        pass
    return None


def intensidad_radar(rgb):
    """Misma clasificación operativa que scripts/procesador_radar.py."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("La imagen radar no tiene formato RGB válido.")
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    intensidad = np.zeros(r.shape, dtype=np.uint8)
    intensidad[(b > r + 20) & (b > g + 10)] = 1
    intensidad[(g > r + 15) & (g > b + 10)] = 2
    intensidad[(r > 140) & (g > 100) & (b < 120) & (r > b + 40)] = 3
    intensidad[(r > 170) & (g > 60) & (g < 180) & (b < 100) & (r > g + 25)] = 4
    intensidad[(r > 170) & (g < 90) & (b < 100) & (r > g + 70)] = 5
    intensidad[(r > 150) & (b > 80) & (g < 100) & (r > g + 50)] = 6
    return intensidad


def elemento_tolerancia(radius):
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def obtener_archivos():
    radar_dt = cargar_timestamp()
    archivos = []
    for path in HISTORICO.glob("*.png"):
        dt = timestamp_archivo(path)
        if dt is None:
            continue
        # El frame objetivo NO participa en la máscara. Así una tormenta
        # nueva no puede contaminar el mapa de ecos estáticos.
        if dt >= radar_dt:
            continue
        archivos.append((dt, path))
    archivos.sort(key=lambda x: x[0])
    return radar_dt, [path for _, path in archivos[-MAX_FRAMES:]]


def construir_mascara(matrices):
    historial = np.stack(matrices, axis=0)
    presente = historial > 0
    se = elemento_tolerancia(RADIO_TOLERANCIA_PX)
    soporte = np.zeros(historial.shape[1:], dtype=np.uint16)
    for frame in presente:
        soporte += ndimage.binary_dilation(frame, structure=se).astype(np.uint16)
    minimo = max(MIN_FRAMES_HISTORICOS, int(np.ceil(historial.shape[0] * PERSISTENCIA_MINIMA)))
    mascara = soporte >= minimo
    confianza = np.clip(soporte.astype(np.float32) / historial.shape[0], 0.0, 1.0)
    return mascara, confianza, soporte, minimo


def main():
    print("======================================")
    print("ClimaAR - Máscara estática espacial/temporal")
    print("======================================")
    radar_dt, archivos = obtener_archivos()
    if len(archivos) < MIN_FRAMES_HISTORICOS:
        raise RuntimeError(
            f"Se necesitan al menos {MIN_FRAMES_HISTORICOS} frames históricos anteriores al frame actual. "
            f"Encontrados: {len(archivos)}"
        )
    matrices = []
    tamaño = None
    for numero, archivo in enumerate(archivos, start=1):
        print(f"[{numero}/{len(archivos)}] {archivo.name}")
        with Image.open(archivo).convert("RGB") as imagen:
            if tamaño is None:
                tamaño = imagen.size
            if imagen.size != tamaño:
                raise RuntimeError(f"Tamaño diferente en {archivo.name}: {imagen.size} != {tamaño}")
            matrices.append(intensidad_radar(np.asarray(imagen, dtype=np.uint8)))

    mascara_bool, confianza, soporte, min_frames = construir_mascara(matrices)
    mascara_data = mascara_bool.astype(np.uint8) * 255
    confianza_data = np.rint(confianza * 255).astype(np.uint8)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mascara_data, mode="L").save(SALIDA)
    Image.fromarray(confianza_data, mode="L").save(CONFIANZA)

    estaticos = int(mascara_bool.sum())
    info = {
        "schema_version": 5,
        "metodo": "SOPORTE_TEMPORAL_ESPACIAL_TOLERANTE",
        "radar_timestamp_utc": radar_dt.isoformat().replace("+00:00", "Z"),
        "frames_analizados": len(archivos),
        "frames": [p.name for p in archivos],
        "persistencia_minima": PERSISTENCIA_MINIMA,
        "min_frames_activos": min_frames,
        "radio_tolerancia_px": RADIO_TOLERANCIA_PX,
        "pixeles_estaticos": estaticos,
        "porcentaje_mascara": round(estaticos / mascara_bool.size * 100, 3),
        "ancho": tamaño[0],
        "alto": tamaño[1],
        "mascara_no_geografica": True,
        "frame_actual_excluido": True,
        "permite_movimiento_y_deformacion": True,
        "no_bloquea_zonas_sin_eco_actual": True,
    }
    INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Píxeles estáticos candidatos: {estaticos}")
    print(f"Persistencia mínima: {min_frames}/{len(archivos)}")
    print(f"Radio espacial: {RADIO_TOLERANCIA_PX}px")
    print(f"Máscara: {SALIDA}")
    return info


if __name__ == "__main__":
    main()
