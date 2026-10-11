from pathlib import Path
from PIL import Image, ImageFilter
import json

HISTORICO = Path("data/radar/historico")
SALIDA = Path("data/radar/mascara_ecos_estaticos.png")

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
    if r > 100 and g > 100 and b < 100 and abs(r-g) < 100:
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
    raise RuntimeError("No hay al menos 2 frames 202610 disponibles.")

print("FRAMES UTILIZADOS:")
for f in archivos:
    print(" -", f)

imagenes = [Image.open(f).convert("RGB") for f in archivos]

ancho, alto = imagenes[0].size

if any(im.size != (ancho, alto) for im in imagenes):
    raise RuntimeError("Los frames tienen tamaños diferentes.")

votos = [[0] * ancho for _ in range(alto)]

for imagen in imagenes:
    pix = imagen.load()

    for y in range(alto):
        for x in range(ancho):
            r, g, b = pix[x, y]

            if es_echo(r, g, b):
                votos[y][x] += 1

# Un pixel se considera estático si aparece
# en al menos 2 de los frames.
mask = Image.new("L", (ancho, alto), 0)
mp = mask.load()

umbral = 2

for y in range(alto):
    for x in range(ancho):
        if votos[y][x] >= umbral:
            mp[x, y] = 255

# Pequeño margen para variaciones de algunos píxeles
mask = mask.filter(ImageFilter.MaxFilter(5))

SALIDA.parent.mkdir(parents=True, exist_ok=True)
mask.save(SALIDA)

info = {
    "frames": [f.name for f in archivos],
    "umbral": umbral,
    "margen_pixeles": 2,
    "ancho": ancho,
    "alto": alto
}

with open("data/radar/mascara_ecos_estaticos.json", "w") as f:
    json.dump(info, f, indent=2)

print()
print("MASCARA CREADA:")
print(SALIDA)
