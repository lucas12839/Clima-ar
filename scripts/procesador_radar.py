import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from scipy import ndimage

try:
    from decodificador_rma10_dbz import validar_paleta, indice_a_dbz
except ImportError:
    validar_paleta = None
    indice_a_dbz = None

BASE = Path("data/radar")
ACTUAL = BASE / "actual.png"
MASCARA = BASE / "mascara_ecos_estaticos.png"
MASCARA_JSON = BASE / "mascara_ecos_estaticos.json"
MASCARA_CLUTTER = Path(__file__).resolve().parents[1] / "config" / "clutter_rma10.png"
SALIDA = BASE / "resultado_procesamiento.json"

# ============================================================
# BLOQUE 1 - LIMPIEZA Y DETECCION OPERATIVA RMA10
# ============================================================

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
# Píxeles (~0,9 km² cada uno) de la misma área de lluvia que tienen que
# estar afuera de la máscara de falsos ecos para aceptar cambio adentro.
APOYO_MIN_PIXELES_FUERA = 15


def cargar_imagen(path):
    if isinstance(path, Image.Image):
        return np.array(path.convert("RGB"))
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    with Image.open(path) as imagen:
        return np.array(imagen.convert("RGB"))


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


def cargar_dbz_rma10(path):
    """
    Decodifica el mismo PNG RMA10 usado por el procesador RGB.

    Si la imagen no es una paleta RMA10 validada, devuelve None para
    mantener intacta la cadena operativa RGB.
    """
    if validar_paleta is None or indice_a_dbz is None:
        print("AVISO: decodificador dBZ RMA10 no disponible.")
        return None

    try:
        path = Path(path)
        with Image.open(path) as imagen:
            validar_paleta(imagen)
            transparencia = imagen.info.get("transparency")
            if not isinstance(transparencia, int):
                raise ValueError("No se encontro el indice de transparencia esperado.")

            # Índices de la paleta en orden de filas (igual que getdata, que Pillow quita en 2027).
            indices = np.asarray(imagen, dtype=np.int16).reshape(-1)
            matriz = np.fromiter(
                (
                    np.nan
                    if int(indice) == transparencia
                    else indice_a_dbz(int(indice))
                    for indice in indices
                ),
                dtype=np.float32,
                count=indices.size,
            ).reshape(imagen.size[1], imagen.size[0])

        return matriz
    except Exception as exc:
        print(f"AVISO: no se pudo decodificar dBZ RMA10: {exc}")
        return None


def resumen_dbz(matriz_dbz):
    if matriz_dbz is None:
        return None

    validos = np.isfinite(matriz_dbz)
    if not np.any(validos):
        return None

    valores = matriz_dbz[validos]
    return {
        "pixeles_validos": int(valores.size),
        "min": round(float(np.min(valores)), 3),
        "max": round(float(np.max(valores)), 3),
        "media": round(float(np.mean(valores)), 3),
        "p95": round(float(np.percentile(valores, 95)), 3),
        "p99": round(float(np.percentile(valores, 99)), 3),
    }


def estadisticas_dbz_region(matriz_dbz, region):
    if matriz_dbz is None:
        return None

    try:
        valores = matriz_dbz[region]
        valores = valores[np.isfinite(valores)]
    except (IndexError, TypeError, ValueError):
        return None

    if valores.size == 0:
        return None

    return {
        "min": round(float(np.min(valores)), 3),
        "max": round(float(np.max(valores)), 3),
        "media": round(float(np.mean(valores)), 3),
        "p95": round(float(np.percentile(valores, 95)), 3),
        "p99": round(float(np.percentile(valores, 99)), 3),
        "pixeles": int(valores.size),
    }


def cargar_mascara_clutter(shape):
    """Clutter persistente del RMA10 (píxeles con eco en >30 % de los frames).

    Se genera con: python scripts/nowcast_flujo_optico.py --generar-mascara
    """
    if not MASCARA_CLUTTER.exists():
        return None
    try:
        with Image.open(MASCARA_CLUTTER) as img:
            clutter = np.asarray(img.convert("L")) > 127
        return clutter if clutter.shape == shape else None
    except Exception as exc:
        print(f"AVISO: no se pudo cargar la máscara de clutter: {exc}")
        return None


def cargar_mascara():
    """Máscara estática = ecos fijos conocidos + clutter persistente.

    Los píxeles enmascarados que muestran cambio temporal (una tormenta real
    pasando sobre las sierras) se rescatan en _mascara_temporal_dinamica.
    """
    base = None
    if MASCARA.exists():
        try:
            imagen = cargar_imagen(MASCARA)
            base = np.mean(imagen, axis=2) > 100
        except Exception as exc:
            print(f"AVISO: no se pudo cargar información de máscara: {exc}")
    else:
        print("AVISO: no existe la máscara estática.")

    shape = base.shape if base is not None else (512, 512)
    clutter = cargar_mascara_clutter(shape)
    if base is None:
        return clutter
    if clutter is None:
        return base
    return base | clutter


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

    # La mediana de hasta tres frames se calcula por operaciones elementales.
    # np.median(axis=0) generaba picos de CPU muy altos en algunas imágenes.
    cantidad_presente = np.add.reduce([frame > 0 for frame in previas])
    minimo_previo = np.minimum.reduce(previas)
    maximo_previo = np.maximum.reduce(previas)
    if len(previas) == 1:
        mediana_previa = previas[0].astype(np.float32)
    elif len(previas) == 2:
        mediana_previa = (minimo_previo.astype(np.float32) + maximo_previo.astype(np.float32)) / 2.0
    else:
        suma_previa = np.add.reduce([frame.astype(np.uint16) for frame in previas])
        mediana_previa = (suma_previa - minimo_previo.astype(np.uint16) - maximo_previo.astype(np.uint16)).astype(np.float32)
    actual_presente = actual > 0

    nuevo_eco = actual_presente & (
        cantidad_presente <= max(0, len(previas) - TEMPORAL_NUEVO_ECO_MIN_FRAMES)
    )
    cambio_intensidad = actual_presente & (
        np.abs(actual.astype(np.int16) - mediana_previa.astype(np.int16))
        >= TEMPORAL_DELTA_INTENSIDAD
    )
    intensificacion = actual_presente & (
        actual.astype(np.int16)
        >= maximo_previo.astype(np.int16) + TEMPORAL_DELTA_INTENSIDAD
    )

    evidencia = mascara & (nuevo_eco | cambio_intensidad | intensificacion)
    # El falso eco de las sierras parpadea de una imagen a otra y eso se
    # confundía con "cambio" (24/11/2026: en 474 imágenes sin lluvia
    # aparecieron 158 tormentas sobre la máscara). Una tormenta real sobre
    # las sierras es parte de un área de lluvia que también sigue afuera
    # de la máscara; el falso eco, no. Solo cuenta el cambio con ese apoyo.
    evidencia &= _apoyo_fuera_de_mascara(actual, mascara)
    return evidencia, len(previas)


def _apoyo_fuera_de_mascara(actual, mascara):
    eco = ndimage.binary_closing(actual >= INTENSIDAD_MINIMA, iterations=1)
    etiquetas, cantidad = ndimage.label(eco)
    if cantidad == 0:
        return np.zeros(actual.shape, dtype=bool)
    fuera = np.bincount(
        etiquetas[eco & ~mascara].ravel(), minlength=cantidad + 1
    )
    validas = fuera >= APOYO_MIN_PIXELES_FUERA
    validas[0] = False
    return validas[etiquetas]


def aplicar_mascara_estatica(
    actual,
    ruta_actual=None,
    *,
    mascara=None,
    evidencia_temporal=None,
    frames_temporales=None,
    quiet=False,
):
    mascara = cargar_mascara() if mascara is None else mascara
    if mascara is None:
        if not quiet:
            print("AVISO: se procesa el radar sin máscara estática.")
        return actual.copy(), 0, 0

    if mascara.shape != actual.shape:
        if not quiet:
            print("AVISO: tamaño de máscara diferente al radar actual.")
            print("Máscara:", mascara.shape)
            print("Radar:", actual.shape)
            print("Se ignora la máscara para no producir un error.")
        return actual.copy(), 0, 0

    estaticos = mascara & (actual > 0)
    if evidencia_temporal is None:
        evidencia_temporal, frames_temporales = _mascara_temporal_dinamica(
            actual, mascara, ruta_actual
        )
    frames_temporales = int(frames_temporales or 0)
    eliminar = estaticos & ~evidencia_temporal

    resultado = actual.copy()
    resultado[eliminar] = 0

    info = cargar_info_mascara()
    frames_mascara = int(info.get("frames_analizados", 0))
    cantidad = int(np.sum(eliminar))
    rescatados = int(np.sum(evidencia_temporal & estaticos))

    if not quiet:
        print("Máscara estática externa utilizada.")
        print("Píxeles estáticos eliminados:", cantidad)
        print("Píxeles dinámicos dentro de máscara conservados:", rescatados)
        print("Frames temporales comparados:", frames_temporales)
        print("Frames utilizados para máscara:", frames_mascara)

    return resultado, cantidad, frames_mascara


def distancia_px(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def detectar_semillas(matriz, componente, maximo=None):
    semillas = []
    if maximo is None:
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


def separar_region(matriz, componente, maximo=None):
    area = int(np.sum(componente))
    if area < UMBRAL_REGION_GRANDE:
        return [componente]

    semillas = detectar_semillas(matriz, componente, maximo=maximo)
    if len(semillas) <= 1:
        return [componente]

    # Watershed de SciPy ejecutado en C: conserva la separación por máximos
    # locales sin el heap Python que se volvía muy lento en ecos extensos.
    y_coords, x_coords = np.where(componente)
    y1, y2 = int(y_coords.min()), int(y_coords.max()) + 1
    x1, x2 = int(x_coords.min()), int(x_coords.max()) + 1
    component_crop = componente[y1:y2, x1:x2]
    matrix_crop = matriz[y1:y2, x1:x2]
    markers = np.zeros(component_crop.shape, dtype=np.int32)

    for indice, (_, y, x) in enumerate(semillas, start=1):
        markers[y - y1, x - x1] = indice

    topography = (255 - np.clip(matrix_crop, 0, 255)).astype(np.uint8)
    topography[~component_crop] = 255
    try:
        labels = ndimage.watershed_ift(
            topography,
            markers,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
    except Exception:
        return [componente]

    regiones = []
    for etiqueta in range(1, len(semillas) + 1):
        local_region = (labels == etiqueta) & component_crop
        if int(np.sum(local_region)) < MIN_AREA_CELDA_GRANDE:
            continue
        region = np.zeros(matriz.shape, dtype=bool)
        region[y1:y2, x1:x2] = local_region
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


def crear_nucleo(matriz, region, matriz_dbz=None):
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

    nucleo = {
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

    # El dBZ se calcula sobre los píxeles EXACTOS del núcleo,
    # no sobre su caja delimitadora.
    if matriz_dbz is not None and matriz_dbz.shape == matriz.shape:
        dbz = estadisticas_dbz_region(matriz_dbz, region)
        if dbz is not None:
            nucleo["dbz"] = dbz

    return nucleo


def detectar_nucleos(matriz, matriz_dbz=None):
    nucleos = []
    componentes = detectar_componentes(matriz)
    maximo = None
    if any(int(np.sum(componente)) >= UMBRAL_REGION_GRANDE for componente in componentes):
        maximo = ndimage.maximum_filter(matriz, size=21, mode="constant", cval=0)
    for componente in componentes:
        for region in separar_region(matriz, componente, maximo=maximo):
            nucleo = crear_nucleo(matriz, region, matriz_dbz)
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

    # --------------------------------------------------------
    # dBZ: mismo frame, segundo canal de lectura.
    # Si falla la validación cuantitativa, el procesador RGB
    # continúa exactamente como antes.
    # --------------------------------------------------------
    matriz_dbz = cargar_dbz_rma10(ACTUAL)
    resumen = resumen_dbz(matriz_dbz)

    if matriz_dbz is not None and matriz_dbz.shape == imagen.shape[:2]:
        print("dBZ RMA10: OK")
        print("dBZ global:", resumen)
    else:
        if matriz_dbz is not None:
            print("AVISO: tamaño dBZ diferente al RGB. Se ignora dBZ.")
        matriz_dbz = None
        resumen = None

    matriz_original = intensidad_radar(imagen)
    pixeles_originales = int(np.sum(matriz_original > 0))

    matriz_dinamica, pixeles_estaticos, frames_mascara = aplicar_mascara_estatica(
        matriz_original,
        ruta_actual=ACTUAL,
    )

    # Compatibilidad con seguimiento_temporal.py: aplicar_mascara_estatica
    # conserva la firma histórica de 3 valores. El conteo de frames se
    # obtiene aquí sin alterar la cadena de tracking.
    frames_temporales = len(
        _cargar_intensidades_previas(
            matriz_original,
            ruta_actual=ACTUAL,
        )
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
    nucleos_candidatos_originales = detectar_nucleos(
        matriz_original,
        matriz_dbz=matriz_dbz,
    )
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
    nucleos_operativos = detectar_nucleos(
        matriz_dinamica,
        matriz_dbz=matriz_dbz,
    )
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

    nucleos_estaticos.sort(
        key=lambda n: (-n.get("intensidad", 0), -n.get("pixeles", 0))
    )
    nucleos_dinamicos.sort(
        key=lambda n: (-n.get("intensidad", 0), -n.get("pixeles", 0))
    )

    # ESTE es el único conjunto que recibe tracking/ML/Intelligence.
    nucleos = nucleos_dinamicos

    radar_estado = clasificacion_radar(nucleos)

    resultado = {
        "version": "RMA10-procesador-11.1-dbz",
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
        "dbz_decodificado": matriz_dbz is not None,
        "dbz_global": resumen,
        "dbz_fuente": "paleta_RMA10_ZH_MAX_validada",
        "dbz_metodo": "interpolacion_lineal_indice_64_207_a_-15_75",
        "dbz_por_nucleo": matriz_dbz is not None,
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
