import json
import math
import heapq
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from scipy import ndimage

BASE = Path("data/radar")
ACTUAL = BASE / "actual.png"
MASCARA = BASE / "mascara_ecos_estaticos.png"
MASCARA_JSON = BASE / "mascara_ecos_estaticos.json"
SALIDA = BASE / "resultado_procesamiento.json"

# ============================================================
# BLOQUE 1 - LIMPIEZA Y DETECCION OPERATIVA RMA10
# ============================================================
# Los ecos débiles y estáticos siguen conservándose para análisis
# interno, pero solamente los núcleos operativos pasan a tracking/ML.

INTENSIDAD_MINIMA = 3
MIN_AREA_NUCLEO = 20
MIN_AREA_NUCLEO_MODERADO = 35
MIN_AREA_CELDA_GRANDE = 120
UMBRAL_REGION_GRANDE = 1500
DISTANCIA_MIN_SEMILLAS = 24
CIERRE_ECOS_ITERACIONES = 1

NIVELES = {
    0: "SIN_ECO",
    1: "DEBIL",
    2: "DEBIL",
    3: "MODERADO",
    4: "MODERADO",
    5: "FUERTE",
    6: "MUY_FUERTE",
}

TEMPORAL_MAX_FRAMES = 3
TEMPORAL_MAX_GAP_MINUTOS = 60.0
TEMPORAL_DELTA_INTENSIDAD = 2
TEMPORAL_NUEVO_ECO_MIN_FRAMES = 2


def cargar_imagen(path):
    if isinstance(path, Image.Image):
        return np.array(path.convert("RGB"))
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    return np.array(Image.open(path).convert("RGB"))


def intensidad_radar(rgb):
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("La imagen radar no tiene formato RGB valido.")

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


def cargar_mascara():
    if not MASCARA.exists():
        print("AVISO: no existe la máscara estática.")
        return None
    try:
        imagen = cargar_imagen(MASCARA)
        return np.mean(imagen, axis=2) > 100
    except Exception as exc:
        print(f"AVISO: no se pudo cargar información de máscara: {exc}")
        return None


def cargar_info_mascara():
    if not MASCARA_JSON.exists():
        return {}
    try:
        with open(MASCARA_JSON, "r", encoding="utf-8") as f:
            datos = json.load(f)
        return datos if isinstance(datos, dict) else {}
    except Exception as exc:
        print(f"AVISO: no se pudo leer información de máscara: {exc}")
        return {}


def _parse_timestamp_text(texto):
    if texto is None:
        return None
    texto = str(texto).strip()
    if not texto:
        return None
    if texto.endswith("Z"):
        texto = texto[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(texto)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    for formato in (
        "%Y%m%d%H%M%S",
        "%Y%m%d_%H%M%S",
        "%Y-%m-%dT%H_%M_%S+00_00",
        "%Y-%m-%dT%H_%M_%S",
        "%Y-%m-%dT%H-%M-%S",
    ):
        try:
            return datetime.strptime(texto, formato).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    if texto.isdigit() and len(texto) == 10:
        try:
            dt = datetime.fromtimestamp(int(texto), tz=timezone.utc)
            if 2000 <= dt.year <= 2100:
                return dt
        except (OverflowError, OSError, ValueError):
            pass
    return None


def _timestamp_de_path(path):
    try:
        return _parse_timestamp_text(Path(path).stem)
    except Exception:
        return None


def _timestamp_actual():
    path = BASE / "radar_timestamp.txt"
    if not path.exists():
        return None
    try:
        return _parse_timestamp_text(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _frames_previos(ruta_actual=None):
    objetivo = _timestamp_de_path(ruta_actual) if ruta_actual is not None else _timestamp_actual()
    if objetivo is None:
        objetivo = _timestamp_actual()
    if objetivo is None:
        return []

    candidatos = {}
    for carpeta in (BASE / "historico", BASE / "historico_tracking_backup"):
        if not carpeta.exists():
            continue
        for path in carpeta.glob("*.png"):
            stamp = _timestamp_de_path(path)
            if stamp is None or stamp >= objetivo:
                continue
            diferencia = (objetivo - stamp).total_seconds() / 60.0
            if diferencia <= 0 or diferencia > TEMPORAL_MAX_GAP_MINUTOS:
                continue
            candidatos.setdefault(stamp.isoformat(), (stamp, path))

    salida = sorted(candidatos.values(), key=lambda item: item[0], reverse=True)
    return [path for _, path in salida[:TEMPORAL_MAX_FRAMES]]


def _cargar_intensidades_previas(actual, ruta_actual=None):
    previas = []
    for path in _frames_previos(ruta_actual):
        try:
            matriz = intensidad_radar(cargar_imagen(path))
            if matriz.shape == actual.shape:
                previas.append(matriz)
        except Exception as exc:
            print(f"AVISO: no se pudo usar frame previo {path}: {exc}")
    return previas


def _mascara_temporal_dinamica(actual, mascara, ruta_actual=None):
    if mascara is None:
        return np.zeros(actual.shape, dtype=bool), 0

    previas = _cargar_intensidades_previas(actual, ruta_actual)
    if not previas:
        return np.zeros(actual.shape, dtype=bool), 0

    apilado = np.stack(previas, axis=0)
    presentes = apilado > 0
    cantidad_presente = np.sum(presentes, axis=0)
    mediana_previa = np.median(apilado, axis=0)
    maximo_previo = np.max(apilado, axis=0)
    actual_presente = actual > 0

    nuevo_eco = actual_presente & (
        cantidad_presente <= max(0, len(previas) - TEMPORAL_NUEVO_ECO_MIN_FRAMES)
    )
    cambio_intensidad = actual_presente & (
        np.abs(actual.astype(np.int16) - mediana_previa.astype(np.int16)) >= TEMPORAL_DELTA_INTENSIDAD
    )
    intensificacion = actual_presente & (
        actual.astype(np.int16) >= maximo_previo.astype(np.int16) + TEMPORAL_DELTA_INTENSIDAD
    )

    evidencia = mascara & (nuevo_eco | cambio_intensidad | intensificacion)
    return evidencia, len(previas)


def aplicar_mascara_estatica(actual, ruta_actual=None):
    mascara = cargar_mascara()
    if mascara is None:
        print("AVISO: se procesa el radar sin máscara estática.")
        return actual.copy(), 0, 0, 0

    if mascara.shape != actual.shape:
        print("AVISO: tamaño de máscara diferente al radar actual.")
        print("Máscara:", mascara.shape)
        print("Radar:", actual.shape)
        print("Se ignora la máscara para no producir un error.")
        return actual.copy(), 0, 0, 0

    estaticos = mascara & (actual > 0)
    evidencia_temporal, frames_temporales = _mascara_temporal_dinamica(
        actual, mascara, ruta_actual
    )
    eliminar = estaticos & ~evidencia_temporal

    resultado = actual.copy()
    resultado[eliminar] = 0

    info = cargar_info_mascara()
    frames_mascara = int(info.get("frames_analizados", 0))
    cantidad = int(np.sum(eliminar))
    rescatados = int(np.sum(evidencia_temporal & estaticos))

    print("Máscara estática externa utilizada.")
    print("Píxeles estáticos eliminados:", cantidad)
    print("Píxeles dinámicos dentro de máscara conservados:", rescatados)
    print("Frames temporales comparados:", frames_temporales)
    print("Frames utilizados para máscara:", frames_mascara)

    return resultado, cantidad, frames_mascara, frames_temporales


def distancia_px(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def detectar_semillas(matriz, componente):
    semillas = []
    maximo = ndimage.maximum_filter(matriz, size=21, mode="constant", cval=0)

    for nivel in (6, 5, 4, 3):
        candidatos = componente & (matriz >= nivel) & (matriz == maximo)
        if not np.any(candidatos):
            continue
        ys, xs = np.where(candidatos)
        puntos = sorted(
            [(int(matriz[y, x]), int(y), int(x)) for y, x in zip(ys, xs)],
            key=lambda p: (-p[0], p[1], p[2]),
        )
        for intensidad, y, x in puntos:
            if all(
                distancia_px(x, y, sx, sy) >= DISTANCIA_MIN_SEMILLAS
                for _, sy, sx in semillas
            ):
                semillas.append((intensidad, y, x))
    return semillas


def separar_region(matriz, componente):
    area = int(np.sum(componente))
    if area < UMBRAL_REGION_GRANDE:
        return [componente]

    semillas = detectar_semillas(matriz, componente)
    if len(semillas) <= 1:
        return [componente]

    h, w = matriz.shape
    etiquetas = np.zeros((h, w), dtype=np.int32)
    heap = []

    for indice, (_, y, x) in enumerate(semillas, start=1):
        etiquetas[y, x] = indice
        heapq.heappush(heap, (-int(matriz[y, x]), 0.0, y, x, indice))

    vecinos = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )

    while heap:
        _, distancia, y, x, etiqueta = heapq.heappop(heap)
        intensidad_actual = int(matriz[y, x])

        for dy, dx in vecinos:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if not componente[ny, nx] or matriz[ny, nx] <= 0 or etiquetas[ny, nx] != 0:
                continue

            nueva = int(matriz[ny, nx])
            nuevo_costo = distancia + 1 + abs(intensidad_actual - nueva) * 100
            etiquetas[ny, nx] = etiqueta
            heapq.heappush(heap, (-nueva, nuevo_costo, ny, nx, etiqueta))

    regiones = []
    for etiqueta in range(1, len(semillas) + 1):
        region = etiquetas == etiqueta
        if int(np.sum(region)) >= MIN_AREA_CELDA_GRANDE:
            regiones.append(region)

    return regiones if regiones else [componente]


def detectar_componentes(matriz):
    # Solo niveles operativos. Los niveles 1-2 quedan fuera de tracking/ML.
    mascara = matriz >= INTENSIDAD_MINIMA

    # Une pequeños cortes de 1 píxel dentro de una misma estructura de precipitación.
    if CIERRE_ECOS_ITERACIONES > 0:
        mascara = ndimage.binary_closing(
            mascara,
            structure=np.ones((3, 3), dtype=bool),
            iterations=CIERRE_ECOS_ITERACIONES,
        )

    etiquetas, cantidad = ndimage.label(
        mascara,
        structure=np.ones((3, 3), dtype=np.uint8),
    )

    componentes = []
    for etiqueta in range(1, cantidad + 1):
        componente = etiquetas == etiqueta
        area = int(np.sum(componente))
        if area >= MIN_AREA_NUCLEO:
            componentes.append(componente)
    return componentes


def crear_nucleo(matriz, region):
    ys, xs = np.where(region)
    area = len(xs)
    if area < MIN_AREA_NUCLEO:
        return None

    valores = matriz[region]
    intensidad = int(valores.max())
    ancho = int(xs.max() - xs.min() + 1)
    alto = int(ys.max() - ys.min() + 1)
    area_caja = max(ancho * alto, 1)
    densidad = area / area_caja

    # Un eco nivel 3 muy pequeño/aislado no es una célula operativa.
    if intensidad == 3 and area < MIN_AREA_NUCLEO_MODERADO:
        return None

    if densidad < 0.12:
        return None

    # Para nivel 3 exigimos estructura espacial mínima.
    if intensidad == 3 and max(ancho, alto) < 5:
        return None

    return {
        "x": round(float(np.mean(xs)), 2),
        "y": round(float(np.mean(ys)), 2),
        "pixeles": area,
        "ancho": ancho,
        "alto": alto,
        "intensidad": intensidad,
        "intensidad_max": intensidad,
        "intensidad_promedio": round(float(np.mean(valores)), 3),
        "nivel_intensidad": NIVELES.get(intensidad, "DESCONOCIDO"),
        "coherencia_espacial": round(float(densidad), 3),
        "estado": "DINAMICO",
    }


def detectar_nucleos(matriz):
    nucleos = []
    for componente in detectar_componentes(matriz):
        for region in separar_region(matriz, componente):
            nucleo = crear_nucleo(matriz, region)
            if nucleo is not None:
                nucleos.append(nucleo)

    nucleos.sort(key=lambda n: (-n["intensidad"], -n["pixeles"]))
    return nucleos


def _region_nucleo(nucleo, shape):
    x = int(round(nucleo["x"]))
    y = int(round(nucleo["y"]))
    ancho = max(int(nucleo["ancho"]), 1)
    alto = max(int(nucleo["alto"]), 1)
    h, w = shape
    x1 = max(0, x - ancho // 2)
    x2 = min(w, x + ancho // 2 + 1)
    y1 = max(0, y - alto // 2)
    y2 = min(h, y + alto // 2 + 1)
    return y1, y2, x1, x2


def analizar_mascara(nucleo, mascara, evidencia_temporal=None):
    if mascara is None:
        return "DINAMICO", 0.70

    try:
        y1, y2, x1, x2 = _region_nucleo(nucleo, mascara.shape)
        region = mascara[y1:y2, x1:x2]
        if region.size == 0:
            return "DINAMICO", 0.70

        porcentaje = float(np.mean(region))
        if porcentaje >= 0.70:
            if evidencia_temporal is not None:
                evidencia = evidencia_temporal[y1:y2, x1:x2]
                if evidencia.size and float(np.mean(evidencia)) >= 0.15:
                    return "CAMBIO", 0.78
            return "ECO_ESTATICO", 0.95
        if porcentaje >= 0.40:
            return "CAMBIO", 0.65
        return "DINAMICO", 0.75
    except Exception:
        return "DINAMICO", 0.50


def calcular_confianza(nucleo, estado, confianza_base):
    if estado == "ECO_ESTATICO":
        return 0.0

    confianza = float(confianza_base)
    if int(nucleo.get("intensidad", 0)) >= 5:
        confianza += 0.08
    if int(nucleo.get("pixeles", 0)) >= 100:
        confianza += 0.05
    if float(nucleo.get("coherencia_espacial", 0)) >= 0.30:
        confianza += 0.03

    return round(min(1.0, confianza), 3)


def clasificacion_radar(nucleos):
    """Describe intensidad radar. NO es una clasificación de severidad meteorológica."""
    if not nucleos:
        return "SIN_ECOS_OPERATIVOS"
    intensidad_max = max(n["intensidad"] for n in nucleos)
    if intensidad_max >= 6:
        return "RADAR_MUY_FUERTE"
    if intensidad_max >= 5:
        return "RADAR_FUERTE"
    if intensidad_max >= 4:
        return "RADAR_MODERADO_FUERTE"
    return "RADAR_MODERADO"


def main():
    print("=== ClimaAR - Procesador RMA10 ===")

    if not ACTUAL.exists():
        raise FileNotFoundError(f"No existe {ACTUAL}")

    imagen = cargar_imagen(ACTUAL)
    print("Tamaño radar actual:", imagen.shape)

    matriz_original = intensidad_radar(imagen)
    pixeles_originales = int(np.sum(matriz_original > 0))

    matriz_dinamica, pixeles_estaticos, frames_mascara, frames_temporales = aplicar_mascara_estatica(
        matriz_original,
        ruta_actual=ACTUAL,
    )

    pixeles_dinamicos = int(np.sum(matriz_dinamica > 0))
    mascara = cargar_mascara()
    evidencia_temporal, _ = _mascara_temporal_dinamica(
        matriz_original,
        mascara,
        ruta_actual=ACTUAL,
    )

    # --------------------------------------------------------
    # 1) Ecos estáticos: se identifican en el frame original,
    #    se conservan en el JSON interno y NO pasan downstream.
    # --------------------------------------------------------
    nucleos_candidatos_originales = detectar_nucleos(matriz_original)
    nucleos_estaticos = []
    for nucleo in nucleos_candidatos_originales:
        estado, confianza_base = analizar_mascara(
            nucleo, mascara, evidencia_temporal
        )
        nucleo["estado"] = estado
        nucleo["confianza"] = calcular_confianza(
            nucleo, estado, confianza_base
        )
        if estado == "ECO_ESTATICO":
            nucleos_estaticos.append(nucleo)

    # --------------------------------------------------------
    # 2) Solo la matriz limpia entra en detección operativa.
    # --------------------------------------------------------
    nucleos_operativos = detectar_nucleos(matriz_dinamica)
    nucleos_dinamicos = []

    for nucleo in nucleos_operativos:
        estado, confianza_base = analizar_mascara(
            nucleo, mascara, evidencia_temporal
        )

        # Doble seguridad: un eco clasificado como estático jamás pasa.
        if estado == "ECO_ESTATICO":
            if nucleo not in nucleos_estaticos:
                nucleo["estado"] = estado
                nucleo["confianza"] = 0.0
                nucleos_estaticos.append(nucleo)
            continue

        nucleo["estado"] = "DINAMICO"
        nucleo["confianza"] = calcular_confianza(
            nucleo, "DINAMICO", confianza_base
        )
        nucleos_dinamicos.append(nucleo)

    nucleos_estaticos.sort(key=lambda n: (-n.get("intensidad", 0), -n.get("pixeles", 0)))
    nucleos_dinamicos.sort(key=lambda n: (-n.get("intensidad", 0), -n.get("pixeles", 0)))

    # ESTE es el único conjunto que recibe tracking/ML/Intelligence.
    nucleos = nucleos_dinamicos

    radar_estado = clasificacion_radar(nucleos)

    resultado = {
        "version": "RMA10-procesador-11.0-bloque1",
        "timestamp_procesamiento": datetime.now(timezone.utc).isoformat(),
        "fuente": "SMN RMA10",
        "metodo_segmentacion": "mascara_estatica_temporal_conectividad_coherencia_operativa",
        "umbral_intensidad_nucleo": INTENSIDAD_MINIMA,
        "min_area_nucleo": MIN_AREA_NUCLEO,
        "min_area_nucleo_moderado": MIN_AREA_NUCLEO_MODERADO,
        "distancia_minima_semillas": DISTANCIA_MIN_SEMILLAS,
        "frames_historicos_utilizados": frames_mascara,
        "frames_temporales_comparados": frames_temporales,
        "pixeles_con_eco_original": pixeles_originales,
        "pixeles_ecos_estaticos_eliminados": pixeles_estaticos,
        "pixeles_identificados_como_estaticos": pixeles_estaticos,
        "pixeles_con_eco_final": pixeles_dinamicos,
        "pixeles_dinamicos": pixeles_dinamicos,
        "nucleos_estaticos_excluidos": len(nucleos_estaticos),
        "nucleos_detectados": len(nucleos_dinamicos) + len(nucleos_estaticos),
        "nucleos_dinamicos": len(nucleos_dinamicos),
        "nucleos_estaticos": nucleos_estaticos,
        "nucleos": nucleos,
        "estado_general": radar_estado,
        "clasificacion_severidad": "NA_NO_SE_DETERMINA_SOLO_CON_RADAR",
        "radar_intensidad_es_senal_no_severidad": True,
        "ecos_estaticos_se_conservan_internamente": True,
        "ecos_estaticos_se_excluyen_antes_del_tracking": True,
        "ecos_estaticos_se_excluyen_del_ml": True,
        "ecos_debil_nivel_1_2_excluidos_de_nucleos_operativos": True,
        "recuperacion_de_ecos_nuevos": True,
        "mascara_estatica_externa": True,
        "historiales_reprocesados": False,
    }

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(SALIDA, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    print("Pixeles originales:", pixeles_originales)
    print("Estáticos eliminados:", pixeles_estaticos)
    print("Pixeles dinámicos:", pixeles_dinamicos)
    print("Núcleos detectados operativos:", len(nucleos_dinamicos))
    print("Núcleos estáticos excluidos:", len(nucleos_estaticos))
    print("Estado radar:", radar_estado)
    print("Severidad meteorológica: NO_DETERMINADA_EN_ESTE_BLOQUE")
    print("Procesamiento RMA10 completado.")


if __name__ == "__main__":
    main()
