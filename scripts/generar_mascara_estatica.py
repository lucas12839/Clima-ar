from pathlib import Path
from PIL import Image, ImageFilter
import json
import math

HISTORICO = Path("data/radar/historico")
SALIDA = Path("data/radar/mascara_ecos_estaticos.png")
INFO = Path("data/radar/mascara_ecos_estaticos.json")
CONFIANZA = Path("data/radar/mascara_ecos_estaticos_confianza.png")

MAX_FRAMES = 40
PERSISTENCIA_MINIMA = 0.75
ESTABILIDAD_MINIMA = 0.75


def intensidad(r, g, b):
    if max(r, g, b) < 25:
        return 0
    if max(r, g, b) - min(r, g, b) < 20:
        return 0

    if b > g and g > r * 1.5:
        return 1
    if g > r * 1.35 and g >= b * 0.95:
        return 2
    if r > 100 and g > 100 and b < 100 and abs(r - g) < 100:
        return 3
    if r > 130 and 60 < g < 180 and b < 90 and r > g * 1.15:
        return 4
    if r > 130 and r > g * 1.35 and r > b * 1.35:
        return 5
    if r > 100 and b > 100 and r > g * 1.3:
        return 6
    return 0


def main():
    archivos = sorted(HISTORICO.glob("202610*.png"), key=lambda p: p.name)
    archivos = archivos[-MAX_FRAMES:]

    if len(archivos) < 3:
        raise RuntimeError(
            f"Se necesitan al menos 3 frames históricos. Encontrados: {len(archivos)}"
        )

    imagenes = [Image.open(p).convert("RGB") for p in archivos]
    ancho, alto = imagenes[0].size

    if any(im.size != (ancho, alto) for im in imagenes):
        raise RuntimeError("Los frames históricos tienen tamaños diferentes.")

    total = ancho * alto
    activos = [0] * total
    clases = [[0] * total for _ in range(7)]

    for numero, imagen in enumerate(imagenes, 1):
        print(f"Analizando frame {numero}/{len(imagenes)}: {archivos[numero-1].name}")
        for i, (r, g, b) in enumerate(imagen.getdata()):
            c = intensidad(r, g, b)
            if c:
                activos[i] += 1
                clases[c][i] += 1

    min_activos = max(2, math.ceil(len(imagenes) * PERSISTENCIA_MINIMA))
    mascara_data = bytearray(total)
    confianza_data = bytearray(total)

    estaticos = 0

    for i in range(total):
        a = activos[i]
        if a < min_activos:
            continue

        dominante = max(clases[c][i] for c in range(1, 7))
        estabilidad = dominante / a if a else 0.0
        persistencia = a / len(imagenes)

        if estabilidad >= ESTABILIDAD_MINIMA:
            confianza = int(round(
                min(1.0, 0.65 * persistencia + 0.35 * estabilidad) * 255
            ))
            mascara_data[i] = 255
            confianza_data[i] = confianza
            estaticos += 1

    mascara = Image.frombytes("L", (ancho, alto), bytes(mascara_data))
    # Solo suaviza pequeños huecos; no expande agresivamente zonas geográficas.
    mascara = mascara.filter(ImageFilter.MedianFilter(3))

    confianza = Image.frombytes("L", (ancho, alto), bytes(confianza_data))

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    mascara.save(SALIDA)
    confianza.save(CONFIANZA)

    info = {
        "schema_version": 2,
        "metodo": "PERSISTENCIA_TEMPORAL_Y_ESTABILIDAD_DE_INTENSIDAD",
        "frames_analizados": len(archivos),
        "frames": [p.name for p in archivos],
        "persistencia_minima": PERSISTENCIA_MINIMA,
        "estabilidad_minima": ESTABILIDAD_MINIMA,
        "min_frames_activos": min_activos,
        "pixeles_estaticos": estaticos,
        "porcentaje_mascara": round(estaticos / total * 100, 3),
        "ancho": ancho,
        "alto": alto,
        "mascara_no_geografica": True,
        "permite_recuperacion_de_ecos_nuevos": True,
    }

    INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print("======================================")
    print("APRENDIZAJE ESTATICO COMPLETADO")
    print("======================================")
    print("Frames:", len(archivos))
    print("Pixeles estáticos:", estaticos)
    print("Porcentaje máscara:", info["porcentaje_mascara"], "%")
    print("Máscara:", SALIDA)
    print("Confianza:", CONFIANZA)
    print("Info:", INFO)


if __name__ == "__main__":
    main()
