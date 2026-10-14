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

NIVELES = {
    0: "FONDO",
    1: "MUY_DEBIL",
    2: "DEBIL",
    3: "MODERADA",
    4: "FUERTE",
    5: "MUY_FUERTE",
    6: "SEVERA",
}


def clasificar_color(r, g, b):
    if max(r, g, b) < 25:
        return "FONDO", 0
    if b > g and g > r * 1.5:
        return "AZUL_CYAN", 1
    if g > r * 1.35 and g >= b * 0.95:
        return "VERDE", 2
    if r > 100 and g > 100 and b < 100 and abs(r - g) < 100:
        return "AMARILLO", 3
    if r > 130 and 60 < g < 180 and b < 90 and r > g * 1.15:
        return "NARANJA", 4
    if r > 130 and r > g * 1.35 and r > b * 1.35:
        return "ROJO", 5
    if r > 100 and b > 100 and r > g * 1.3:
        return "VIOLETA_MAGENTA", 6
    return "OTRO", 0


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
        (-1, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
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
                    nx, ny = x + dx, y + dy
                    if nx < 0 or nx >= ancho or ny < 0 or ny >= alto:
                        continue
                    if visitado[ny][nx] or clasificacion[ny][nx] <= 0:
                        continue
                    visitado[ny][nx] = True
                    cola.append((nx, ny))

            if len(puntos) < 10:
                continue

            xs = [p[0] for p in puntos]
            ys = [p[1] for p in puntos]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            centro_x = sum(xs) / len(xs)
            centro_y = sum(ys) / len(ys)

            max_intensidad = max(
                clasificacion[y][x] for x, y in puntos
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
                "nivel_intensidad": NIVELES.get(max_intensidad, "OTRO"),
                "colores": colores_presentes,
            })

    return nucleos


def cargar_mascara():
    if not MASK_PATH.exists():
        return None
    try:
        return Image.open(MASK_PATH).convert("L")
    except Exception:
        return None


def analizar_mascara(nucleo, mascara):
    if mascara is None:
        return "SIN_MASCARA", 65

    mw, mh = mascara.size
    x = int(round(nucleo["x"]))
    y = int(round(nucleo["y"]))

    if not (0 <= x < mw and 0 <= y < mh):
        return "PRECIPITACION", 75

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
        return "PRECIPITACION", 75

    proporcion = marcados / total

    if proporcion >= 0.80:
        return "ECO_ESTATICO", 92
    if proporcion >= 0.30:
        return "CAMBIO", 70
    return "PRECIPITACION", 85


def calcular_confianza(nucleo, estado, base):
    confianza = float(base)
    intensidad = nucleo["intensidad"]
    confianza += intensidad * 2
    if intensidad >= 5:
        confianza += 2
    confianza = max(1, min(99, confianza))
    if estado == "ECO_ESTATICO":
        confianza = min(confianza, 15)
    return round(confianza, 1)


def parse_timestamp_file():
    if not TIMESTAMP_FILE.exists():
        raise RuntimeError(
            "No existe data/radar/radar_timestamp.txt"
        )

    value = TIMESTAMP_FILE.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError("radar_timestamp.txt esta vacio")

    try:
        texto = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(texto)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception as exc:
        raise RuntimeError(
            f"Timestamp radar invalido: {value}"
        ) from exc


def timestamp_compacto(dt):
    return dt.strftime("%Y%m%d%H%M%S")


def limpiar_frames_futuros(radar_dt):
    eliminados = 0
    limite = radar_dt + timedelta(minutes=MAX_FUTURE_SKEW_MINUTES)

    for archivo in HISTORICO_DIR.glob("*.png"):
        match = TIMESTAMP_RE.match(archivo.name)
        if not match:
            continue

        try:
            dt = datetime.strptime(
                match.group(1),
                "%Y%m%d%H%M%S"
            ).replace(tzinfo=timezone.utc)
        except Exception:
            continue

        if dt > limite:
            archivo.unlink(missing_ok=True)
            eliminados += 1

    return eliminados


def reparar_estado_futuro(radar_dt):
    if not STATE_PATH.exists():
        return False

    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False

    if not isinstance(data, dict):
        return False

    value = data.get("ultimo_radar_timestamp")
    if not value:
        return False

    try:
        texto = str(value).replace("Z", "+00:00")
        ultimo = datetime.fromisoformat(texto)
        if ultimo.tzinfo is None:
            ultimo = ultimo.replace(tzinfo=timezone.utc)
        ultimo = ultimo.astimezone(timezone.utc)
    except Exception:
        return False

    if ultimo <= radar_dt + timedelta(minutes=MAX_FUTURE_SKEW_MINUTES):
        return False

    data["ultimo_radar_timestamp"] = None
    data["tracks"] = []
    data["tracks_activos"] = []
    data["ultimo_frame_nucleos"] = []
    data["siguiente_track_id"] = 1
    data["reparacion_timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    data["motivo_reparacion"] = "Estado anterior tenia un radar futuro invalido. Se reinicio el seguimiento para evitar contaminar la prediccion."

    STATE_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return True


def limpiar_predicciones_futuras(radar_dt):
    if not PENDING_PATH.exists():
        return 0

    try:
        data = json.loads(PENDING_PATH.read_text(encoding="utf-8"))
    except Exception:
        return 0

    if not isinstance(data, dict):
        return 0

    pendientes = data.get("pendientes", [])
    if not isinstance(pendientes, list):
        return 0

    validos = []
    eliminados = 0

    for item in pendientes:
        if not isinstance(item, dict):
            continue

        value = item.get("timestamp_origen")
        futuro = False

        if value:
            try:
                texto = str(value).replace("Z", "+00:00")
                origen = datetime.fromisoformat(texto)
                if origen.tzinfo is None:
                    origen = origen.replace(tzinfo=timezone.utc)
                origen = origen.astimezone(timezone.utc)
                futuro = origen > radar_dt + timedelta(minutes=MAX_FUTURE_SKEW_MINUTES)
            except Exception:
                pass

        if futuro:
            eliminados += 1
        else:
            validos.append(item)

    data["pendientes"] = validos[-500:]
    data["actualizado"] = radar_dt.isoformat().replace("+00:00", "Z")

    PENDING_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return eliminados


def main():
    HISTORICO_DIR.mkdir(parents=True, exist_ok=True)

    radar_dt = parse_timestamp_file()
    ahora = datetime.now(timezone.utc)

    if radar_dt > ahora + timedelta(minutes=MAX_FUTURE_SKEW_MINUTES):
        raise RuntimeError(
            "El radar_timestamp.txt es futuro respecto al reloj del runner: "
            f"{radar_dt.isoformat()}"
        )

    actual_path = RADAR_DIR / "actual.png"
    if not actual_path.exists():
        raise RuntimeError(
            "No existe data/radar/actual.png"
        )

    eliminados = limpiar_frames_futuros(radar_dt)
    estado_reparado = reparar_estado_futuro(radar_dt)
    pendientes_eliminados = limpiar_predicciones_futuras(radar_dt)

    if eliminados:
        print(f"Frames historicos futuros eliminados: {eliminados}")
    if estado_reparado:
        print("Estado persistente futuro reparado y reiniciado.")
    if pendientes_eliminados:
        print(
            "Predicciones pendientes invalidas eliminadas:",
            pendientes_eliminados
        )

    imagen_path = actual_path
    print(
        "Procesando SIEMPRE data/radar/actual.png; "
        "no se selecciona el historico por nombre."
    )
    print(
        f"Frame RMA10 valido: {timestamp_compacto(radar_dt)}"
    )

    imagen = Image.open(imagen_path).convert("RGBA")

    print(f"Resolucion: {imagen.width}x{imagen.height}")

    nucleos = detectar_nucleos(imagen)
    mascara = cargar_mascara()

    print(f"Nucleos detectados: {len(nucleos)}")

    resultado = []
    suma_peso = 0.0
    suma_confianza = 0.0
    cantidad_estaticos = 0
    cantidad_precipitacion = 0
    cantidad_cambios = 0

    for i, nucleo in enumerate(nucleos, start=1):
        estado, confianza_base = analizar_mascara(nucleo, mascara)
        confianza = calcular_confianza(
            nucleo,
            estado,
            confianza_base
        )

        if estado == "ECO_ESTATICO":
            cantidad_estaticos += 1
        elif estado == "PRECIPITACION":
            cantidad_precipitacion += 1
        elif estado == "CAMBIO":
            cantidad_cambios += 1

        peso = nucleo["pixeles"] * max(1, nucleo["intensidad"])
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

    radar_iso = radar_dt.isoformat().replace("+00:00", "Z")
    compacto = timestamp_compacto(radar_dt)

    salida = {
        "version": 3,
        "imagen_procesada": f"{compacto}.png",
        "radar_timestamp": radar_iso,
        "fuente_imagen": "data/radar/actual.png",
        "ancho_imagen": imagen.width,
        "alto_imagen": imagen.height,
        "nucleos_detectados": len(resultado),
        "nucleos": resultado,
        "resumen": {
            "ecos_estaticos": cantidad_estaticos,
            "precipitacion": cantidad_precipitacion,
            "cambios": cantidad_cambios,
            "probabilidad_global": probabilidad_global,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            salida,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8"
    )

    print("===== PROCESAMIENTO =====")
    print(f"Radar timestamp UTC: {radar_iso}")
    print(f"Nucleos: {len(resultado)}")
    print(f"Estaticos: {cantidad_estaticos}")
    print(f"Precipitacion: {cantidad_precipitacion}")
    print(f"Cambios: {cantidad_cambios}")
    print(f"Probabilidad global: {probabilidad_global}%")
    print(f"Salida: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
