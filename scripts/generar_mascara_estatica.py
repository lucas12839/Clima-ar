from pathlib import Path
from PIL import Image, ImageFilter
import json

HISTORICO = Path("data/radar/historico")
SALIDA = Path("data/radar/mascara_ecos_estaticos.png")
INFO = Path("data/radar/mascara_ecos_estaticos.json")


def es_echo(r, g, b):
    if max(r, g, b) < 25:
        return False

    diferencia = max(r, g, b) - min(r, g, b)

    if diferencia < 20:
        return False

    if b > g and g > r * 1.5:
        return True

    if g > r * 1.35 and g >= b * 0.95:
        return True

    if r > 100 and g > 100 and b < 100 and abs(r - g) < 100:
        return True

    if r > 130 and 60 < g < 180 and b < 90 and r > g * 1.15:
        return True

    if r > 130 and r > g * 1.35 and r > b * 1.35:
        return True

    if r > 100 and b > 100 and r > g * 1.3:
        return True

    return False


archivos = sorted(
    HISTORICO.glob("202610*.png"),
    key=lambda p: p.name
)[-3:]


if len(archivos) < 2:
    raise RuntimeError(
        "No hay al menos 2 frames recientes para crear la máscara."
    )


print("FRAMES UTILIZADOS:")
for archivo in archivos:
    print(" -", archivo)


imagenes = [
    Image.open(archivo).convert("RGB")
    for archivo in archivos
]


ancho, alto = imagenes[0].size


if any(imagen.size != (ancho, alto) for imagen in imagenes):
    raise RuntimeError(
        "Los frames tienen tamaños diferentes."
    )


votos = [
    [0] * ancho
    for _ in range(alto)
]


for imagen in imagenes:

    pixeles = imagen.load()

    for y in range(alto):

        for x in range(ancho):

            r, g, b = pixeles[x, y]

            if es_echo(r, g, b):
                votos[y][x] += 1


# Un eco que aparece en al menos 2 frames
# se considera candidato a eco estático.

mascara = Image.new(
    "L",
    (ancho, alto),
    0
)

pixeles_mascara = mascara.load()

for y in range(alto):

    for x in range(ancho):

        if votos[y][x] >= 2:
            pixeles_mascara[x, y] = 255


# Pequeño margen de seguridad de 2 píxeles.
mascara = mascara.filter(
    ImageFilter.MaxFilter(5)
)


SALIDA.parent.mkdir(
    parents=True,
    exist_ok=True
)


mascara.save(SALIDA)


informacion = {
    "frames": [
        archivo.name
        for archivo in archivos
    ],
    "umbral": 2,
    "margen_pixeles": 2,
    "ancho": ancho,
    "alto": alto
}


with open(INFO, "w") as archivo:

    json.dump(
        informacion,
        archivo,
        indent=2
    )


print()
print("======================================")
print("MASCARA DE ECOS ESTATICOS CREADA")
print("======================================")
print(SALIDA)
print(INFO)
