from pathlib import Path
from collections import deque
from datetime import datetime, timezone, timedelta
import json
import re
from PIL import Image

RADAR_DIR = Path("data/radar")
HISTORICO_DIR = RADAR_DIR / "historico"
MASK_PATH = RADAR_DIR / "mascara_ecos_estaticos.png"
OUTPUT_PATH = RADAR_DIR / "resultado_procesamiento.json"
TIMESTAMP_FILE = RADAR_DIR / "radar_timestamp.txt"
STATE_PATH = RADAR_DIR / "seguimiento" / "estado.json"
PENDING_PATH = RADAR_DIR / "seguimiento" / "predicciones_pendientes.json"

TIMESTAMP_RE = re.compile(r"^(\d{14})\.png$")
MAX_FUTURE_SKEW_MINUTES = 2.0

# ============================================================
# CONFIGURACIÓN DE MÁSCARA DINÁMICA
# ============================================================

MAX_HISTORICOS_COMPARAR = 5
MAX_ANTIGUEDAD_HISTORICO_MIN = 60

# Para considerar un píxel como eco estático:
# debe existir en al menos 3 frames anteriores y
# mantenerse prácticamente igual.
MIN_FRAMES_COMPARABLES = 3
MIN_FRAMES_ESTABLES = 3
TOLERANCIA_INTENSIDAD_ESTATICA = 1

# Si el píxel actual cambia claramente respecto del histórico,
# se recupera como posible precipitación.
MIN_CAMBIOS_DINAMICOS = 1

# ============================================================
# NIVELES
# ============================================================

NIVELES = {
    0: "FONDO",
    1: "MUY_DEBIL",
    2: "DEBIL",
    3: "MODERADA",
    4: "FUERTE",
    5: "MUY_FUERTE",
    6: "SEVERA",
}


# ============================================================
# CLASIFICACIÓN DE COLORES
# ============================================================

def clasificar_color(r, g, b):
    if max(r, g, b) < 25:
        return "FONDO", 0

    if b > g and g > r * 1.5:
        return "AZUL_CYAN", 1

    if g > r * 1.35 and g >= b * 0.95:
        return "VERDE", 2

    if (
        r > 100
        and g > 100
        and b < 100
        and abs(r - g) < 100
    ):
        return "AMARILLO", 3

    if (
        r > 130
        and 60 < g < 180
        and b < 90
        and r > g * 1.15
    ):
        return "NARANJA", 4

    if (
        r > 130
        and r > g * 1.35
        and r > b * 1.35
    ):
        return "ROJO", 5

    if (
        r > 100
        and b > 100
        and r > g * 1.3
    ):
        return "VIOLETA_MAGENTA", 6

    return "OTRO", 0


# ============================================================
# CONVERTIR IMAGEN A MATRIZ DE INTENSIDADES
# ============================================================

def imagen_a_matriz_intensidad(imagen):
    ancho, alto = imagen.size
    pix = imagen.load()

    matriz = [
        [0] * ancho
        for _ in range(alto)
    ]

    for y in range(alto):
        for x in range(ancho):
            valor = pix[x, y]

            if len(valor) >= 3:
                r, g, b = valor[:3]
            else:
                r = g = b = valor[0]

            _, intensidad = clasificar_color(
                r,
                g,
                b
            )

            matriz[y][x] = intensidad

    return matriz


# ============================================================
# TIMESTAMP
# ============================================================

def parse_timestamp_file():
    if not TIMESTAMP_FILE.exists():
        raise RuntimeError(
            "No existe data/radar/radar_timestamp.txt"
        )

    value = TIMESTAMP_FILE.read_text(
        encoding="utf-8"
    ).strip()

    if not value:
        raise RuntimeError(
            "radar_timestamp.txt esta vacio"
        )

    try:
        texto = value.replace(
            "Z",
            "+00:00"
        )

        dt = datetime.fromisoformat(
            texto
        )

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt.astimezone(
            timezone.utc
        )

    except Exception as exc:
        raise RuntimeError(
            f"Timestamp radar invalido: {value}"
        ) from exc


def timestamp_compacto(dt):
    return dt.strftime(
        "%Y%m%d%H%M%S"
    )


# ============================================================
# HISTÓRICO RMA10
# ============================================================

def obtener_historicos_validos(radar_dt):
    candidatos = []

    if not HISTORICO_DIR.exists():
        return []

    limite_inferior = (
        radar_dt
        - timedelta(
            minutes=MAX_ANTIGUEDAD_HISTORICO_MIN
        )
    )

    for archivo in HISTORICO_DIR.glob("*.png"):
        match = TIMESTAMP_RE.match(
            archivo.name
        )

        if not match:
            continue

        try:
            dt = datetime.strptime(
                match.group(1),
                "%Y%m%d%H%M%S"
            ).replace(
                tzinfo=timezone.utc
            )
        except Exception:
            continue

        # Nunca usar un frame futuro.
        if dt >= radar_dt:
            continue

        # Evitar frames demasiado antiguos.
        if dt < limite_inferior:
            continue

        candidatos.append(
            (dt, archivo)
        )

    candidatos.sort(
        key=lambda item: item[0],
        reverse=True
    )

    seleccionados = []

    for dt, archivo in candidatos:
        if len(seleccionados) >= MAX_HISTORICOS_COMPARAR:
            break

        try:
            imagen = Image.open(
                archivo
            ).convert("RGBA")

            seleccionados.append(
                (
                    dt,
                    imagen
                )
            )

        except Exception as exc:
            print(
                f"No se pudo leer historico "
                f"{archivo.name}: {exc}"
            )

    seleccionados.sort(
        key=lambda item: item[0]
    )

    return seleccionados


# ============================================================
# LIMPIEZA DE FRAMES FUTUROS
# ============================================================

def limpiar_frames_futuros(radar_dt):
    eliminados = 0

    limite = (
        radar_dt
        + timedelta(
            minutes=MAX_FUTURE_SKEW_MINUTES
        )
    )

    for archivo in HISTORICO_DIR.glob("*.png"):

        match = TIMESTAMP_RE.match(
            archivo.name
        )

        if not match:
            continue

        try:
            dt = datetime.strptime(
                match.group(1),
                "%Y%m%d%H%M%S"
            ).replace(
                tzinfo=timezone.utc
            )

        except Exception:
            continue

        if dt > limite:
            archivo.unlink(
                missing_ok=True
            )

            eliminados += 1

    return eliminados


# ============================================================
# REPARAR ESTADO FUTURO
# ============================================================

def reparar_estado_futuro(radar_dt):
    if not STATE_PATH.exists():
        return False

    try:
        data = json.loads(
            STATE_PATH.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        return False

    if not isinstance(data, dict):
        return False

    value = data.get(
        "ultimo_radar_timestamp"
    )

    if not value:
        return False

    try:
        texto = str(value).replace(
            "Z",
            "+00:00"
        )

        ultimo = datetime.fromisoformat(
            texto
        )

        if ultimo.tzinfo is None:
            ultimo = ultimo.replace(
                tzinfo=timezone.utc
            )

        ultimo = ultimo.astimezone(
            timezone.utc
        )

    except Exception:
        return False

    if ultimo <= (
        radar_dt
        + timedelta(
            minutes=MAX_FUTURE_SKEW_MINUTES
        )
    ):
        return False

    data["ultimo_radar_timestamp"] = None
    data["tracks"] = []
    data["tracks_activos"] = []
    data["ultimo_frame_nucleos"] = []
    data["siguiente_track_id"] = 1

    data["reparacion_timestamp"] = (
        datetime.now(timezone.utc)
        .isoformat()
        .replace(
            "+00:00",
            "Z"
        )
    )

    data["motivo_reparacion"] = (
        "Estado anterior tenia un radar "
        "futuro invalido. Se reinicio el "
        "seguimiento para evitar contaminar "
        "la prediccion."
    )

    STATE_PATH.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    return True


# ============================================================
# LIMPIAR PREDICCIONES FUTURAS
# ============================================================

def limpiar_predicciones_futuras(radar_dt):
    if not PENDING_PATH.exists():
        return 0

    try:
        data = json.loads(
            PENDING_PATH.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        return 0

    if not isinstance(data, dict):
        return 0

    pendientes = data.get(
        "pendientes",
        []
    )

    if not isinstance(pendientes, list):
        return 0

    validos = []
    eliminados = 0

    for item in pendientes:

        if not isinstance(item, dict):
            continue

        value = item.get(
            "timestamp_origen"
        )

        futuro = False

        if value:

            try:
                texto = str(value).replace(
                    "Z",
                    "+00:00"
                )

                origen = datetime.fromisoformat(
                    texto
                )

                if origen.tzinfo is None:
                    origen = origen.replace(
                        tzinfo=timezone.utc
                    )

                origen = origen.astimezone(
                    timezone.utc
                )

                futuro = (
                    origen
                    > radar_dt
                    + timedelta(
                        minutes=MAX_FUTURE_SKEW_MINUTES
                    )
                )

            except Exception:
                pass

        if futuro:
            eliminados += 1
        else:
            validos.append(item)

    data["pendientes"] = validos[-500:]

    data["actualizado"] = (
        radar_dt
        .isoformat()
        .replace(
            "+00:00",
            "Z"
        )
    )

    PENDING_PATH.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    return eliminados


# ============================================================
# MÁSCARA ESTÁTICA
# ============================================================

def cargar_mascara():
    if not MASK_PATH.exists():
        return None

    try:
        return Image.open(
            MASK_PATH
        ).convert("L")

    except Exception:
        return None


# ============================================================
# FILTRO DINÁMICO POR PÍXEL
#
# NO elimina un núcleo completo.
#
# Solamente elimina píxeles que:
# - están dentro de la máscara estática
# - aparecen en suficientes frames anteriores
# - mantienen la misma intensidad
#
# Si cambian, se recuperan como dinámicos.
# ============================================================

def construir_mascara_dinamica(
    imagen_actual,
    mascara_estatica,
    historicos
):
    ancho, alto = imagen_actual.size

    intensidad_actual = (
        imagen_a_matriz_intensidad(
            imagen_actual
        )
    )

    historicos_matrices = []

    for dt, imagen in historicos:

        if imagen.size != imagen_actual.size:
            print(
                "Historico ignorado por "
                "resolucion diferente: "
                f"{imagen.size}"
            )
            continue

        historicos_matrices.append(
            (
                dt,
                imagen_a_matriz_intensidad(
                    imagen
                )
            )
        )

    comparables = len(
        historicos_matrices
    )

    # Si no tenemos suficientes frames,
    # NO eliminamos píxeles.
    if (
        mascara_estatica is None
        or comparables < MIN_FRAMES_COMPARABLES
    ):
        return (
            intensidad_actual,
            0,
            0,
            comparables
        )

    if mascara_estatica.size != (
        ancho,
        alto
    ):
        print(
            "Mascara estatica con "
            "resolucion diferente. "
            "Se ignora en este frame."
        )

        return (
            intensidad_actual,
            0,
            0,
            comparables
        )

    mascara_resultado = [
        fila[:]
        for fila in intensidad_actual
    ]

    pixeles_estaticos = 0
    pixeles_dinamicos = 0

    for y in range(alto):
        for x in range(ancho):

            if (
                mascara_estatica.getpixel(
                    (x, y)
                ) <= 0
            ):
                continue

            actual = intensidad_actual[y][x]

            # Fondo actual no necesita filtrarse.
            if actual <= 0:
                continue

            valores = []

            for _, matriz in historicos_matrices:
                valores.append(
                    matriz[y][x]
                )

            presentes = [
                valor
                for valor in valores
                if valor > 0
            ]

            if len(presentes) < MIN_FRAMES_COMPARABLES:
                continue

            estables = 0
            cambios = 0

            for anterior in presentes:

                diferencia = abs(
                    actual - anterior
                )

                if (
                    diferencia
                    <= TOLERANCIA_INTENSIDAD_ESTATICA
                ):
                    estables += 1
                else:
                    cambios += 1

            # Eco persistente:
            # mismo píxel, misma zona,
            # intensidad estable.
            if (
                estables
                >= MIN_FRAMES_ESTABLES
                and cambios
                < MIN_CAMBIOS_DINAMICOS
            ):
                mascara_resultado[y][x] = 0
                pixeles_estaticos += 1

            # Cambio respecto del histórico:
            # conservar el píxel.
            elif cambios >= MIN_CAMBIOS_DINAMICOS:
                pixeles_dinamicos += 1

    return (
        mascara_resultado,
        pixeles_estaticos,
        pixeles_dinamicos,
        comparables
    )


# ============================================================
# DETECTAR NÚCLEOS
# ============================================================

def detectar_nucleos_desde_matriz(
    matriz,
    imagen_original
):
    alto = len(matriz)

    if alto == 0:
        return []

    ancho = len(matriz[0])

    pix = imagen_original.load()

    colores = [
        [None] * ancho
        for _ in range(alto)
    ]

    for y in range(alto):
        for x in range(ancho):

            if matriz[y][x] <= 0:
                continue

            valor = pix[x, y]

            if len(valor) >= 3:
                r, g, b = valor[:3]
            else:
                r = g = b = valor[0]

            color, _ = clasificar_color(
                r,
                g,
                b
            )

            colores[y][x] = color

    visitado = [
        [False] * ancho
        for _ in range(alto)
    ]

    nucleos = []

    vecinos = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ]

    for y0 in range(alto):
        for x0 in range(ancho):

            if visitado[y0][x0]:
                continue

            if matriz[y0][x0] <= 0:
                visitado[y0][x0] = True
                continue

            cola = deque(
                [(x0, y0)]
            )

            visitado[y0][x0] = True

            puntos = []

            while cola:

                x, y = cola.popleft()

                puntos.append(
                    (x, y)
                )

                for dx, dy in vecinos:

                    nx = x + dx
                    ny = y + dy

                    if nx < 0 or nx >= ancho:
                        continue

                    if ny < 0 or ny >= alto:
                        continue

                    if visitado[ny][nx]:
                        continue

                    if matriz[ny][nx] <= 0:
                        continue

                    visitado[ny][nx] = True

                    cola.append(
                        (nx, ny)
                    )

            if len(puntos) < 10:
                continue

            xs = [
                p[0]
                for p in puntos
            ]

            ys = [
                p[1]
                for p in puntos
            ]

            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)

            centro_x = (
                sum(xs)
                / len(xs)
            )

            centro_y = (
                sum(ys)
                / len(ys)
            )

            max_intensidad = max(
                matriz[y][x]
                for x, y in puntos
            )

            colores_presentes = sorted(
                set(
                    colores[y][x]
                    for x, y in puntos
                )
                - {
                    None,
                    "OTRO"
                }
            )

            nucleos.append({
                "x": round(
                    centro_x,
                    2
                ),
                "y": round(
                    centro_y,
                    2
                ),
                "pixeles": len(puntos),
                "ancho": (
                    max_x
                    - min_x
                    + 1
                ),
                "alto": (
                    max_y
                    - min_y
                    + 1
                ),
                "intensidad": max_intensidad,
                "nivel_intensidad": (
                    NIVELES.get(
                        max_intensidad,
                        "OTRO"
                    )
                ),
                "colores": (
                    colores_presentes
                ),
            })

    return nucleos


# ============================================================
# COMPATIBILIDAD
# ============================================================

def detectar_nucleos(imagen):
    matriz = (
        imagen_a_matriz_intensidad(
            imagen
        )
    )

    return detectar_nucleos_desde_matriz(
        matriz,
        imagen
    )


# ============================================================
# CONFIANZA
# ============================================================

def calcular_confianza(
    nucleo,
    estado,
    base
):
    confianza = float(base)

    intensidad = nucleo[
        "intensidad"
    ]

    confianza += intensidad * 2

    if intensidad >= 5:
        confianza += 2

    confianza = max(
        1,
        min(
            99,
            confianza
        )
    )

    if estado == "ECO_ESTATICO":
        confianza = min(
            confianza,
            15
        )

    return round(
        confianza,
        1
    )


# ============================================================
# ESTADO DEL NÚCLEO
# ============================================================

def clasificar_estado_dinamico(
    nucleo,
    pixeles_estaticos,
    pixeles_dinamicos
):
    total = max(
        1,
        nucleo["pixeles"]
    )

    proporcion_estatica = (
        pixeles_estaticos
        / total
    )

    proporcion_dinamica = (
        pixeles_dinamicos
        / total
    )

    if (
        proporcion_estatica >= 0.90
        and proporcion_dinamica < 0.10
    ):
        return (
            "ECO_ESTATICO",
            15
        )

    if proporcion_dinamica >= 0.10:
        return (
            "CAMBIO",
            70
        )

    return (
        "PRECIPITACION",
        85
    )


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def main():

    HISTORICO_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    STATE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    radar_dt = parse_timestamp_file()

    ahora = datetime.now(
        timezone.utc
    )

    if radar_dt > (
        ahora
        + timedelta(
            minutes=MAX_FUTURE_SKEW_MINUTES
        )
    ):
        raise RuntimeError(
            "El radar_timestamp.txt "
            "es futuro respecto al reloj "
            "del runner: "
            f"{radar_dt.isoformat()}"
        )

    actual_path = (
        RADAR_DIR
        / "actual.png"
    )

    if not actual_path.exists():
        raise RuntimeError(
            "No existe "
            "data/radar/actual.png"
        )

    eliminados = (
        limpiar_frames_futuros(
            radar_dt
        )
    )

    estado_reparado = (
        reparar_estado_futuro(
            radar_dt
        )
    )

    pendientes_eliminados = (
        limpiar_predicciones_futuras(
            radar_dt
        )
    )

    if eliminados:
        print(
            "Frames historicos futuros "
            f"eliminados: {eliminados}"
        )

    if estado_reparado:
        print(
            "Estado persistente futuro "
            "reparado y reiniciado."
        )

    if pendientes_eliminados:
        print(
            "Predicciones pendientes "
            f"invalidas eliminadas: "
            f"{pendientes_eliminados}"
        )

    print(
        "Procesando SIEMPRE "
        "data/radar/actual.png"
    )

    print(
        "Frame RMA10 valido: "
        f"{timestamp_compacto(radar_dt)}"
    )

    imagen = Image.open(
        actual_path
    ).convert("RGBA")

    print(
        f"Resolucion: "
        f"{imagen.width}x{imagen.height}"
    )

    mascara = cargar_mascara()

    if mascara is not None:
        print(
            "Mascara estatica encontrada: "
            f"{MASK_PATH}"
        )
    else:
        print(
            "No hay mascara estatica."
        )

    historicos = (
        obtener_historicos_validos(
            radar_dt
        )
    )

    print(
        "Frames historicos disponibles "
        "para comparación: "
        f"{len(historicos)}"
    )

    for dt, _ in historicos:
        print(
            "  - "
            f"{dt.isoformat()}"
        )

    (
        matriz_filtrada,
        pixeles_estaticos,
        pixeles_dinamicos,
        frames_comparados
    ) = construir_mascara_dinamica(
        imagen,
        mascara,
        historicos
    )

    print(
        "Frames comparados: "
        f"{frames_comparados}"
    )

    print(
        "Pixeles eliminados como "
        "eco estático: "
        f"{pixeles_estaticos}"
    )

    print(
        "Pixeles recuperados como "
        "dinámicos: "
        f"{pixeles_dinamicos}"
    )

    nucleos = (
        detectar_nucleos_desde_matriz(
            matriz_filtrada,
            imagen
        )
    )

    print(
        "Nucleos detectados después "
        "del filtro dinámico: "
        f"{len(nucleos)}"
    )

    resultado = []
    ecos_estaticos = []

    suma_peso = 0.0
    suma_confianza = 0.0

    cantidad_estaticos = 0
    cantidad_precipitacion = 0
    cantidad_cambios = 0

    nucleo_id = 0
    nucleo_id_estatico = 0

    for nucleo in nucleos:

        # ====================================================
        # ESTADO DINÁMICO
        #
        # IMPORTANTE:
        # no usamos bounding box de la máscara.
        # El filtrado ya ocurrió píxel por píxel.
        # ====================================================

        pixeles_nucleo = nucleo[
            "pixeles"
        ]

        # Como el núcleo fue construido después
        # de eliminar los píxeles estáticos,
        # el núcleo restante se considera dinámico.
        #
        # Esto evita eliminar una tormenta completa
        # simplemente porque atraviesa una zona de montaña.

        estado = "PRECIPITACION"
        confianza_base = 85

        if pixeles_dinamicos > 0:
            estado = "CAMBIO"
            confianza_base = 70

        confianza = calcular_confianza(
            nucleo,
            estado,
            confianza_base
        )

        nucleo_id += 1

        registro = {
            "id": nucleo_id,
            "x": nucleo["x"],
            "y": nucleo["y"],
            "pixeles": nucleo["pixeles"],
            "ancho": nucleo["ancho"],
            "alto": nucleo["alto"],
            "intensidad": nucleo["intensidad"],
            "nivel_intensidad": (
                nucleo["nivel_intensidad"]
            ),
            "colores": nucleo["colores"],
            "estado": estado,
            "confianza": confianza,
            "es_trackeable": True,
            "filtro_estatico": (
                "pixelwise_temporal"
            ),
        }

        resultado.append(
            registro
        )

        suma_peso += float(
            nucleo["intensidad"]
        )

        suma_confianza += confianza

        if estado == "CAMBIO":
            cantidad_cambios += 1
        else:
            cantidad_precipitacion += 1

    # ========================================================
    # RESUMEN
    # ========================================================

    cantidad_nucleos = len(
        resultado
    )

    if cantidad_nucleos:
        confianza_promedio = (
            suma_confianza
            / cantidad_nucleos
        )
    else:
        confianza_promedio = 0.0

    # ========================================================
    # GUARDAR RESULTADO
    # ========================================================

    salida = {
        "version": 5,
        "timestamp": (
            radar_dt
            .isoformat()
            .replace(
                "+00:00",
                "Z"
            )
        ),
        "timestamp_compacto": (
            timestamp_compacto(
                radar_dt
            )
        ),
        "imagen": (
            "data/radar/actual.png"
        ),
        "resolucion": {
            "ancho": imagen.width,
            "alto": imagen.height,
        },
        "mascara_estatica": (
            MASK_PATH.exists()
        ),
        "metodo_filtro_estatico": (
            "pixelwise_temporal"
        ),
        "parametros_filtro": {
            "max_historicos": (
                MAX_HISTORICOS_COMPARAR
            ),
            "antiguedad_max_min": (
                MAX_ANTIGUEDAD_HISTORICO_MIN
            ),
            "min_frames_comparables": (
                MIN_FRAMES_COMPARABLES
            ),
            "min_frames_estables": (
                MIN_FRAMES_ESTABLES
            ),
            "tolerancia_intensidad": (
                TOLERANCIA_INTENSIDAD_ESTATICA
            ),
        },
        "frames_historicos_comparados": (
            frames_comparados
        ),
        "nucleos_detectados": (
            cantidad_nucleos
        ),
        "nucleos_trackeables": (
            cantidad_nucleos
        ),
        "nucleos_estaticos_excluidos": (
            0
        ),
        "ecos_estaticos": (
            ecos_estaticos
        ),
        "nucleos": resultado,
        "resumen": {
            "nucleos": (
                cantidad_nucleos
            ),
            "precipitacion": (
                cantidad_precipitacion
            ),
            "cambios": (
                cantidad_cambios
            ),
            "ecos_estaticos": (
                cantidad_estaticos
            ),
            "pixeles_ecos_estaticos_excluidos": (
                pixeles_estaticos
            ),
            "pixeles_dinamicos_recuperados": (
                pixeles_dinamicos
            ),
            "confianza_promedio": round(
                confianza_promedio,
                1
            ),
            "peso_total": round(
                suma_peso,
                2
            ),
        },
        "limpieza": {
            "frames_futuros_eliminados": (
                eliminados
            ),
            "estado_reparado": (
                estado_reparado
            ),
            "predicciones_futuras_eliminadas": (
                pendientes_eliminados
            ),
        },
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            salida,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "Resultado guardado en: "
        f"{OUTPUT_PATH}"
    )

    print(
        "Procesamiento RMA10 "
        "finalizado correctamente."
    )


if __name__ == "__main__":
    main()
