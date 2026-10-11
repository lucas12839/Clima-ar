#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import json
import math
import re


RADAR_DIR = Path("data/radar")
RESULTADO_FILE = RADAR_DIR / "resultado_procesamiento.json"

SEGUIMIENTO_DIR = RADAR_DIR / "seguimiento"
ESTADO_FILE = SEGUIMIENTO_DIR / "estado.json"
RESULTADO_SEGUIMIENTO = SEGUIMIENTO_DIR / "seguimiento_temporal.json"

HISTORICO_DIR = RADAR_DIR / "historico"

# Parámetros de seguimiento
DISTANCIA_BASE = 25.0
DISTANCIA_MAXIMA = 90.0

CAMBIO_AREA_CRECE = 15.0
CAMBIO_AREA_DEBILITA = -15.0

MOVIMIENTO_MINIMO = 5.0

MAX_HISTORIAL_POSICIONES = 12


# ============================================================
# UTILIDADES
# ============================================================

def ahora_utc():
    return datetime.utcnow().isoformat() + "Z"


def cargar_json(path):
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"No se pudo leer {path}: {e}")
        return None


def guardar_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporal = path.with_suffix(".tmp")

    with open(temporal, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    temporal.replace(path)


# ============================================================
# OBTENER TIMESTAMP DEL RADAR
# ============================================================

def obtener_timestamp_radar():

    archivos = sorted(
        HISTORICO_DIR.glob("*.png"),
        key=lambda p: p.name
    )

    if not archivos:
        return None

    nombre = archivos[-1].stem

    encontrado = re.search(
        r"(\d{14})",
        nombre
    )

    if not encontrado:
        return None

    texto = encontrado.group(1)

    try:
        return datetime.strptime(
            texto,
            "%Y%m%d%H%M%S"
        )
    except Exception:
        return None


# ============================================================
# EXTRAER NÚCLEOS DEL RESULTADO ACTUAL
# ============================================================

def obtener_nucleos(resultado):

    if not isinstance(resultado, dict):
        return []

    posibles = [
        resultado.get("nucleos"),
        resultado.get("núcleos"),
        resultado.get("cells"),
        resultado.get("ecos"),
        resultado.get("detected_cells"),
    ]

    for candidato in posibles:

        if isinstance(candidato, list):
            return candidato

    # Búsqueda recursiva por si el JSON tiene otra estructura
    def buscar(obj):

        if isinstance(obj, dict):

            for clave, valor in obj.items():

                clave_normalizada = str(clave).lower()

                if clave_normalizada in (
                    "nucleos",
                    "núcleos",
                    "cells",
                    "ecos",
                    "detected_cells"
                ) and isinstance(valor, list):

                    return valor

                encontrado = buscar(valor)

                if encontrado is not None:
                    return encontrado

        elif isinstance(obj, list):

            for elemento in obj:

                encontrado = buscar(elemento)

                if encontrado is not None:
                    return encontrado

        return None

    encontrado = buscar(resultado)

    return encontrado if encontrado else []


# ============================================================
# NORMALIZAR NÚCLEO
# ============================================================

def normalizar_nucleo(nucleo):

    if not isinstance(nucleo, dict):
        return None

    try:

        x = float(
            nucleo.get(
                "centro_x",
                nucleo.get("x", 0)
            )
        )

        y = float(
            nucleo.get(
                "centro_y",
                nucleo.get("y", 0)
            )
        )

        pixeles = int(
            nucleo.get(
                "pixeles",
                nucleo.get("area", 0)
            )
        )

        ancho = float(
            nucleo.get(
                "ancho",
                nucleo.get("width", 1)
            )
        )

        alto = float(
            nucleo.get(
                "alto",
                nucleo.get("height", 1)
            )
        )

        intensidad = nucleo.get(
            "intensidad_maxima",
            nucleo.get(
                "intensity",
                "DESCONOCIDA"
            )
        )

        nivel = nucleo.get(
            "nivel_intensidad",
            "DESCONOCIDO"
        )

        estado = nucleo.get(
            "estado",
            "DESCONOCIDO"
        )

        confianza = float(
            nucleo.get(
                "confianza",
                0
            )
        )

        return {
            "x": x,
            "y": y,
            "pixeles": pixeles,
            "ancho": ancho,
            "alto": alto,
            "intensidad": intensidad,
            "nivel_intensidad": nivel,
            "estado": estado,
            "confianza": confianza
        }

    except Exception:
        return None


# ============================================================
# DISTANCIA
# ============================================================

def distancia(a, b):

    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]

    return math.sqrt(
        dx * dx +
        dy * dy
    )


# ============================================================
# INTERSECCIÓN DE RECTÁNGULOS
# ============================================================

def calcular_iou(a, b):

    a_left = a["x"] - a["ancho"] / 2
    a_right = a["x"] + a["ancho"] / 2
    a_top = a["y"] - a["alto"] / 2
    a_bottom = a["y"] + a["alto"] / 2

    b_left = b["x"] - b["ancho"] / 2
    b_right = b["x"] + b["ancho"] / 2
    b_top = b["y"] - b["alto"] / 2
    b_bottom = b["y"] + b["alto"] / 2

    izquierda = max(a_left, b_left)
    derecha = min(a_right, b_right)
    arriba = max(a_top, b_top)
    abajo = min(a_bottom, b_bottom)

    if derecha <= izquierda:
        return 0.0

    if abajo <= arriba:
        return 0.0

    interseccion = (
        (derecha - izquierda) *
        (abajo - arriba)
    )

    area_a = (
        a["ancho"] *
        a["alto"]
    )

    area_b = (
        b["ancho"] *
        b["alto"]
    )

    union = (
        area_a +
        area_b -
        interseccion
    )

    if union <= 0:
        return 0.0

    return interseccion / union


# ============================================================
# VELOCIDAD
# ============================================================

def calcular_umbral_distancia(minutos):

    if minutos <= 0:
        return DISTANCIA_MAXIMA

    return min(
        DISTANCIA_MAXIMA,
        DISTANCIA_BASE +
        minutos * 5
    )


# ============================================================
# COMPARAR NÚCLEOS
# ============================================================

def compatibilidad(actual, anterior, minutos):

    dist = distancia(
        actual,
        anterior
    )

    limite = calcular_umbral_distancia(
        minutos
    )

    if dist > limite:
        return None

    iou = calcular_iou(
        actual,
        anterior
    )

    # Área
    area_anterior = max(
        1,
        anterior["pixeles"]
    )

    area_actual = max(
        1,
        actual["pixeles"]
    )

    cambio_area = (
        (area_actual - area_anterior)
        / area_anterior
        * 100
    )

    # Puntaje
    puntaje = 0

    # Cercanía
    puntaje += max(
        0,
        50 * (
            1 -
            dist / limite
        )
    )

    # Solapamiento
    puntaje += min(
        30,
        iou * 30
    )

    # Área similar
    diferencia_area = abs(
        cambio_area
    )

    if diferencia_area < 50:
        puntaje += 20

    elif diferencia_area < 100:
        puntaje += 10

    return {
        "puntaje": puntaje,
        "distancia": dist,
        "iou": iou,
        "cambio_area": cambio_area
    }


# ============================================================
# ASIGNAR IDs
# ============================================================

def siguiente_id(tracks):

    ids = []

    for track in tracks:

        try:
            ids.append(
                int(track.get("id", 0))
            )
        except Exception:
            pass

    if not ids:
        return 1

    return max(ids) + 1


# ============================================================
# ANALIZAR
# ============================================================

def analizar():

    print("")
    print("======================================")
    print("ClimaAR - SEGUIMIENTO TEMPORAL")
    print("======================================")

    resultado = cargar_json(
        RESULTADO_FILE
    )

    if resultado is None:
        raise RuntimeError(
            "No existe resultado_procesamiento.json"
        )

    nucleos_originales = obtener_nucleos(
        resultado
    )

    actuales = []

    for nucleo in nucleos_originales:

        normalizado = normalizar_nucleo(
            nucleo
        )

        if normalizado:
            actuales.append(
                normalizado
            )

    print(
        f"Núcleos actuales: {len(actuales)}"
    )

    estado_anterior = cargar_json(
        ESTADO_FILE
    )

    if estado_anterior is None:

        print(
            "No existe frame anterior."
        )

        print(
            "Este será el primer frame de referencia."
        )

        tracks = []

        for actual in actuales:

            tracks.append({
                "id": siguiente_id(tracks),
                "primer_frame": ahora_utc(),
                "ultimo_frame": ahora_utc(),
                "frames": 1,
                "x": actual["x"],
                "y": actual["y"],
                "pixeles": actual["pixeles"],
                "intensidad": actual["intensidad"],
                "nivel_intensidad": actual["nivel_intensidad"],
                "historial": [
                    {
                        "x": actual["x"],
                        "y": actual["y"],
                        "pixeles": actual["pixeles"],
                        "intensidad": actual["intensidad"]
                    }
                ]
            })

        nuevo_estado = {
            "timestamp": ahora_utc(),
            "radar_timestamp": (
                obtener_timestamp_radar().isoformat()
                if obtener_timestamp_radar()
                else None
            ),
            "tracks": tracks,
            "nucleos": actuales
        }

        guardar_json(
            ESTADO_FILE,
            nuevo_estado
        )

        guardar_json(
            RESULTADO_SEGUIMIENTO,
            {
                "timestamp": ahora_utc(),
                "frame_anterior_disponible": False,
                "nucleos_actuales": len(actuales),
                "nucleos_nuevos": len(actuales),
                "nucleos_seguimiento": 0,
                "nucleos_debilitados": 0,
                "nucleos_creciendo": 0,
                "nucleos_movimiento": 0,
                "nucleos_estables": 0,
                "nucleos_desaparecidos": 0,
                "tracks": tracks
            }
        )

        return

    anteriores = estado_anterior.get(
        "nucleos",
        []
    )

    tracks = estado_anterior.get(
        "tracks",
        []
    )

    # Convertir anteriores por seguridad
    anteriores_normalizados = []

    for anterior in anteriores:

        normalizado = normalizar_nucleo(
            anterior
        )

        if normalizado:
            anteriores_normalizados.append(
                normalizado
            )

    timestamp_actual = obtener_timestamp_radar()
    timestamp_anterior = None

    texto_anterior = estado_anterior.get(
        "radar_timestamp"
    )

    if texto_anterior:

        try:
            timestamp_anterior = datetime.fromisoformat(
                texto_anterior
            )

        except Exception:
            timestamp_anterior = None

    if (
        timestamp_actual is not None
        and timestamp_anterior is not None
    ):

        minutos = (
            timestamp_actual -
            timestamp_anterior
        ).total_seconds() / 60

        minutos = max(
            0.1,
            minutos
        )

    else:

        minutos = 10.0

    print(
        f"Intervalo estimado: {minutos:.1f} minutos"
    )

    usados_anteriores = set()

    nuevos_tracks = []

    nuevos = 0
    creciendo = 0
    debilitando = 0
    movimiento = 0
    estables = 0

    coincidencias = []

    # ========================================================
    # BUSCAR CORRESPONDENCIAS
    # ========================================================

    for indice_actual, actual in enumerate(
        actuales
    ):

        mejor = None
        mejor_indice = None

        for indice_anterior, anterior in enumerate(
            anteriores_normalizados
        ):

            if indice_anterior in usados_anteriores:
                continue

            resultado_match = compatibilidad(
                actual,
                anterior,
                minutos
            )

            if resultado_match is None:
                continue

            if (
                mejor is None
                or resultado_match["puntaje"]
                > mejor["puntaje"]
            ):

                mejor = resultado_match
                mejor_indice = indice_anterior

        if mejor is None:

            nuevos += 1

            coincidencias.append({
                "actual": actual,
                "anterior": None,
                "match": None
            })

            continue

        usados_anteriores.add(
            mejor_indice
        )

        anterior = anteriores_normalizados[
            mejor_indice
        ]

        coincidencias.append({
            "actual": actual,
            "anterior": anterior,
            "match": mejor
        })


    # ========================================================
    # PROCESAR TRACKS
    # ========================================================

    id_siguiente = siguiente_id(
        tracks
    )

    tracks_por_posicion = {}

    for track in tracks:

        tracks_por_posicion[
            (
                round(track.get("x", 0), 1),
                round(track.get("y", 0), 1)
            )
        ] = track


    for coincidencia in coincidencias:

        actual = coincidencia["actual"]
        anterior = coincidencia["anterior"]
        match = coincidencia["match"]

        # ----------------------------------------------------
        # NÚCLEO NUEVO
        # ----------------------------------------------------

        if anterior is None:

            track = {
                "id": id_siguiente,
                "primer_frame": ahora_utc(),
                "ultimo_frame": ahora_utc(),
                "frames": 1,
                "x": actual["x"],
                "y": actual["y"],
                "pixeles": actual["pixeles"],
                "intensidad": actual["intensidad"],
                "nivel_intensidad": actual["nivel_intensidad"],
                "estado": "NUEVO",
                "historial": [
                    {
                        "x": actual["x"],
                        "y": actual["y"],
                        "pixeles": actual["pixeles"],
                        "intensidad": actual["intensidad"],
                        "estado": "NUEVO"
                    }
                ]
            }

            nuevos_tracks.append(
                track
            )

            id_siguiente += 1

            continue


        # ----------------------------------------------------
        # EXISTENTE
        # ----------------------------------------------------

        dx = (
            actual["x"] -
            anterior["x"]
        )

        dy = (
            actual["y"] -
            anterior["y"]
        )

        desplazamiento = math.sqrt(
            dx * dx +
            dy * dy
        )

        velocidad_pixeles_min = (
            desplazamiento /
            minutos
        )

        cambio_area = (
            (
                actual["pixeles"] -
                anterior["pixeles"]
            )
            /
            max(
                1,
                anterior["pixeles"]
            )
            *
            100
        )

        # ----------------------------------------------------
        # ESTADO DE EVOLUCIÓN
        # ----------------------------------------------------

        if cambio_area >= CAMBIO_AREA_CRECE:

            estado_evolucion = "CRECIENDO"
            creciendo += 1

        elif cambio_area <= CAMBIO_AREA_DEBILITA:

            estado_evolucion = "DEBILITANDOSE"
            debilitando += 1

        else:

            estado_evolucion = "ESTABLE"
            estables += 1


        # ----------------------------------------------------
        # MOVIMIENTO
        # ----------------------------------------------------

        se_mueve = (
            desplazamiento >=
            MOVIMIENTO_MINIMO
        )

        if se_mueve:
            movimiento += 1


        # ----------------------------------------------------
        # BUSCAR TRACK EXISTENTE
        # ----------------------------------------------------

        track = None

        mejor_distancia = float("inf")

        for candidato in tracks:

            distancia_track = math.sqrt(
                (
                    candidato.get("x", 0)
                    - anterior["x"]
                ) ** 2
                +
                (
                    candidato.get("y", 0)
                    - anterior["y"]
                ) ** 2
            )

            if distancia_track < mejor_distancia:

                mejor_distancia = (
                    distancia_track
                )

                track = candidato


        if track is None:

            track = {
                "id": id_siguiente,
                "primer_frame": ahora_utc(),
                "frames": 0,
                "historial": []
            }

            id_siguiente += 1


        track["ultimo_frame"] = ahora_utc()

        track["frames"] = (
            track.get("frames", 0) +
            1
        )

        track["x"] = actual["x"]
        track["y"] = actual["y"]

        track["pixeles"] = actual["pixeles"]

        track["intensidad"] = (
            actual["intensidad"]
        )

        track["nivel_intensidad"] = (
            actual["nivel_intensidad"]
        )

        track["estado"] = (
            estado_evolucion
        )

        track["desplazamiento_pixeles"] = round(
            desplazamiento,
            2
        )

        track["velocidad_pixeles_min"] = round(
            velocidad_pixeles_min,
            3
        )

        track["direccion_grados"] = round(
            (
                math.degrees(
                    math.atan2(
                        dy,
                        dx
                    )
                )
                + 360
            ) % 360,
            1
        )

        track["cambio_area_porcentaje"] = round(
            cambio_area,
            1
        )

        track["iou"] = round(
            match["iou"],
            3
        )

        track["historial"] = (
            track.get("historial", [])
        )

        track["historial"].append({

            "x": actual["x"],
            "y": actual["y"],

            "pixeles":
                actual["pixeles"],

            "intensidad":
                actual["intensidad"],

            "estado":
                estado_evolucion,

            "desplazamiento_pixeles":
                round(
                    desplazamiento,
                    2
                ),

            "velocidad_pixeles_min":
                round(
                    velocidad_pixeles_min,
                    3
                ),

            "direccion_grados":
                round(
                    (
                        math.degrees(
                            math.atan2(
                                dy,
                                dx
                            )
                        )
                        + 360
                    ) % 360,
                    1
                ),

            "cambio_area_porcentaje":
                round(
                    cambio_area,
                    1
                )
        })

        track["historial"] = track[
            "historial"
        ][
            -MAX_HISTORIAL_POSICIONES:
        ]

        nuevos_tracks.append(
            track
        )


    # ========================================================
    # DESAPARECIDOS
    # ========================================================

    desaparecidos = 0

    for indice_anterior, anterior in enumerate(
        anteriores_normalizados
    ):

        if indice_anterior in usados_anteriores:
            continue

        desaparecidos += 1


    # ========================================================
    # RESULTADO
    # ========================================================

    seguimiento = {

        "timestamp": ahora_utc(),

        "radar_timestamp": (
            timestamp_actual.isoformat()
            if timestamp_actual
            else None
        ),

        "intervalo_minutos": round(
            minutos,
            2
        ),

        "frame_anterior_disponible": True,

        "nucleos_actuales": len(
            actuales
        ),

        "nucleos_nuevos": nuevos,

        "nucleos_creciendo": creciendo,

        "nucleos_debilitandose":
            debilitando,

        "nucleos_movimiento":
            movimiento,

        "nucleos_estables":
            estables,

        "nucleos_desaparecidos":
            desaparecidos,

        "tracks": nuevos_tracks
    }

    guardar_json(
        RESULTADO_SEGUIMIENTO,
        seguimiento
    )

    nuevo_estado = {

        "timestamp": ahora_utc(),

        "radar_timestamp": (
            timestamp_actual.isoformat()
            if timestamp_actual
            else None
        ),

        "nucleos": actuales,

        "tracks": nuevos_tracks
    }

    guardar_json(
        ESTADO_FILE,
        nuevo_estado
    )

    print("")
    print("======================================")
    print("SEGUIMIENTO TEMPORAL")
    print("======================================")

    print(
        f"Núcleos actuales: {len(actuales)}"
    )

    print(
        f"Nuevos: {nuevos}"
    )

    print(
        f"Creciendo: {creciendo}"
    )

    print(
        f"Debilitándose: {debilitando}"
    )

    print(
        f"Moviéndose: {movimiento}"
    )

    print(
        f"Estables: {estables}"
    )

    print(
        f"Desaparecidos: {desaparecidos}"
    )

    print(
        "Seguimiento guardado correctamente."
    )


if __name__ == "__main__":
    analizar()
