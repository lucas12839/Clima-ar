from pathlib import Path
from collections import deque
import json
import math
from PIL import Image


RADAR_DIR = Path("data/radar")
HISTORICO_DIR = RADAR_DIR / "historico"
MASK_PATH = RADAR_DIR / "mascara_ecos_estaticos.png"
OUTPUT_PATH = RADAR_DIR / "resultado_procesamiento.json"


# ---------------------------------------------------------
# CLASIFICACIÓN DE COLORES DEL RADAR
# ---------------------------------------------------------

def clasificar_color(r, g, b):
    # Fondo
    if max(r, g, b) < 25:
        return "FONDO", 0

    # Azul / cyan
    if b > g and g > r * 1.5:
        return "AZUL_CYAN", 1

    # Verde
    if g > r * 1.35 and g >= b * 0.95:
        return "VERDE", 2

    # Amarillo
    if r > 100 and g > 100 and b < 100 and abs(r - g) < 100:
        return "AMARILLO", 3

    # Naranja
    if r > 130 and 60 < g < 180 and b < 90 and r > g * 1.15:
        return "NARANJA", 4

    # Rojo
    if r > 130 and r > g * 1.35 and r > b * 1.35:
        return "ROJO", 5

    # Violeta / magenta
    if r > 100 and b > 100 and r > g * 1.3:
        return "VIOLETA_MAGENTA", 6

    return "OTRO", 0


NIVELES = {
    0: "FONDO",
    1: "MUY_DEBIL",
    2: "DEBIL",
    3: "MODERADA",
    4: "FUERTE",
    5: "MUY_FUERTE",
    6: "SEVERA",
}


# ---------------------------------------------------------
# DETECCIÓN DE NÚCLEOS
# ---------------------------------------------------------

def detectar_nucleos(imagen):
    ancho, alto = imagen.size
    pix = imagen.load()

    clasificacion = [[0] * ancho for _ in range(alto)]
    colores = [[None] * ancho for _ in range(alto)]

    for y in range(alto):
        for x in range(ancho):
            r, g, b, *resto = pix[x, y]

            color, intensidad = clasificar_color(r, g, b)

            clasificacion[y][x] = intensidad
            colores[y][x] = color

    visitado = [[False] * ancho for _ in range(alto)]
    nucleos = []

    vecinos = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),           (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    ]

    for y0 in range(alto):
        for x0 in range(ancho):

            if visitado[y0][x0]:
                continue

            if clasificacion[y0][x0] <= 0:
                visitado[y0][x0] = True
                continue

            cola = deque([(x0, y0)])
            visitado[y0][x0] = True
            puntos = []

            while cola:
                x, y = cola.popleft()
                puntos.append((x, y))

                for dx, dy in vecinos:
                    nx = x + dx
                    ny = y + dy

                    if nx < 0 or nx >= ancho or ny < 0 or ny >= alto:
                        continue

                    if visitado[ny][nx]:
                        continue

                    if clasificacion[ny][nx] <= 0:
                        continue

                    visitado[ny][nx] = True
                    cola.append((nx, ny))

            # Eliminar ruido mínimo
            if len(puntos) < 10:
                continue

            xs = [p[0] for p in puntos]
            ys = [p[1] for p in puntos]

            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)

            centro_x = sum(xs) / len(xs)
            centro_y = sum(ys) / len(ys)

            max_intensidad = max(
                clasificacion[y][x]
                for x, y in puntos
            )

            colores_presentes = sorted(
                set(colores[y][x] for x, y in puntos)
                - {None, "OTRO"}
            )

            nucleos.append({
                "x": round(centro_x, 2),
                "y": round(centro_y, 2),
                "pixeles": len(puntos),
                "ancho": max_x - min_x + 1,
                "alto": max_y - min_y + 1,
                "intensidad": max_intensidad,
                "nivel_intensidad": NIVELES.get(
                    max_intensidad,
                    "OTRO"
                ),
                "colores": colores_presentes,
            })

    return nucleos


# ---------------------------------------------------------
# MÁSCARA DE ECOS ESTÁTICOS
# ---------------------------------------------------------

def cargar_mascara():
    if not MASK_PATH.exists():
        return None

    try:
        mask = Image.open(MASK_PATH).convert("L")
        return mask
    except Exception:
        return None


def analizar_mascara(nucleo, mascara):
    if mascara is None:
        return {
            "estado": "SIN_MASCARA",
            "confianza": 65
        }

    mw, mh = mascara.size

    x = int(round(nucleo["x"]))
    y = int(round(nucleo["y"]))

    if not (0 <= x < mw and 0 <= y < mh):
        return {
            "estado": "PRECIPITACION",
            "confianza": 75
        }

    # Analizamos una zona alrededor del núcleo,
    # no solamente su centro.
    mitad_x = max(1, nucleo["ancho"] // 2)
    mitad_y = max(1, nucleo["alto"] // 2)

    x1 = max(0, x - mitad_x)
    x2 = min(mw - 1, x + mitad_x)
    y1 = max(0, y - mitad_y)
    y2 = min(mh - 1, y + mitad_y)

    total = 0
    marcados = 0

    for yy in range(y1, y2 + 1):
        for xx in range(x1, x2 + 1):
            total += 1

            if mascara.getpixel((xx, yy)) > 0:
                marcados += 1

    if total == 0:
        return {
            "estado": "PRECIPITACION",
            "confianza": 75
        }

    proporcion = marcados / total

    # Muy dentro de la zona estática
    if proporcion >= 0.80:
        return {
            "estado": "ECO_ESTATICO",
            "confianza": 92
        }

    # Parcialmente dentro: no bloquearlo
    if proporcion >= 0.30:
        return {
            "estado": "CAMBIO",
            "confianza": 70
        }

    # Mayormente fuera de la máscara
    return {
        "estado": "PRECIPITACION",
        "confianza": 85
    }


# ---------------------------------------------------------
# PROBABILIDAD
# ---------------------------------------------------------

def calcular_confianza(nucleo, estado, confianza_base):
    intensidad = nucleo["intensidad"]

    confianza = float(confianza_base)

    # Intensidad más alta = mayor señal meteorológica
    confianza += intensidad * 2

    if intensidad >= 5:
        confianza += 2

    # No permitir valores absurdos
    confianza = max(1, min(99, confianza))

    # Un eco estático nunca se presenta como precipitación
    if estado == "ECO_ESTATICO":
        confianza = min(confianza, 15)

    return round(confianza, 1)


# ---------------------------------------------------------
# PROCESAMIENTO PRINCIPAL
# ---------------------------------------------------------

def encontrar_imagen():
    archivos = sorted(
        HISTORICO_DIR.glob("*.png")
    )

    if not archivos:
        raise RuntimeError(
            "No se encontró ningún PNG histórico."
        )

    return archivos[-1]


def main():

    imagen_path = encontrar_imagen()

    print(f"Procesando: {imagen_path}")

    imagen = Image.open(imagen_path).convert("RGBA")

    nucleos = detectar_nucleos(imagen)

    mascara = cargar_mascara()

    print(f"Núcleos detectados: {len(nucleos)}")

    resultado = []

    suma_peso = 0
    suma_confianza = 0

    cantidad_estaticos = 0
    cantidad_precipitacion = 0
    cantidad_cambios = 0

    for i, nucleo in enumerate(nucleos, start=1):

        info = analizar_mascara(
            nucleo,
            mascara
        )

        estado = info["estado"]
        confianza = calcular_confianza(
            nucleo,
            estado,
            info["confianza"]
        )

        if estado == "ECO_ESTATICO":
            cantidad_estaticos += 1

        elif estado == "PRECIPITACION":
            cantidad_precipitacion += 1

        elif estado == "CAMBIO":
            cantidad_cambios += 1

        peso = nucleo["pixeles"] * max(
            1,
            nucleo["intensidad"]
        )

        # Los ecos estáticos pesan muchísimo menos
        # en la probabilidad meteorológica.
        if estado == "ECO_ESTATICO":
            peso *= 0.05

        suma_peso += peso
        suma_confianza += peso * confianza

        item = dict(nucleo)

        item["nucleo_id"] = i
        item["estado"] = estado
        item["confianza"] = confianza

        resultado.append(item)

    if suma_peso > 0:
        probabilidad_global = suma_confianza / suma_peso
    else:
        probabilidad_global = 0

    probabilidad_global = round(
        max(0, min(99, probabilidad_global)),
        1
    )

    salida = {
        "version": 2,

        "imagen_procesada": str(
            imagen_path
        ),

        "ancho_imagen": imagen.width,
        "alto_imagen": imagen.height,

        "nucleos_detectados": len(resultado),

        "nucleos": resultado,

        "resumen": {
            "ecos_estaticos": cantidad_estaticos,
            "precipitacion": cantidad_precipitacion,
            "cambios": cantidad_cambios,
            "probabilidad_global": probabilidad_global
        }
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            salida,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print("===== PROCESAMIENTO =====")
    print(
        f"Núcleos: {len(resultado)}"
    )
    print(
        f"Estáticos: {cantidad_estaticos}"
    )
    print(
        f"Precipitación: {cantidad_precipitacion}"
    )
    print(
        f"Cambios: {cantidad_cambios}"
    )
    print(
        f"Probabilidad global: "
        f"{probabilidad_global}%"
    )
    print(
        f"Salida: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
