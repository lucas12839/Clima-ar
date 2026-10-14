#!/usr/bin/env python3
"""
ClimaAR - Adaptador operativo RMA10 -> Storm Engine.

Conecta el pipeline RMA10 actual con los motores de meteorología e IA
que históricamente trabajaban sobre radar_nowcast.json.

No reemplaza captura, detección, tracking ni predicción.
Solo construye el contrato de datos que necesita el Storm Engine.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("data/radar")
RESULTADO = BASE / "resultado_procesamiento.json"
SEGUIMIENTO = BASE / "seguimiento" / "seguimiento_temporal.json"
ESTADO = BASE / "seguimiento" / "estado.json"
PREDICCION = BASE / "seguimiento" / "prediccion_evolucion.json"
SALIDA = BASE / "radar_nowcast.json"

# Aproximación del diámetro del dominio RMA10 para convertir movimiento
# de imagen a velocidad física. Se conserva explícita para poder calibrarla.
RADAR_RANGE_KM = 240.0


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def timestamp_from_result(resultado):
    name = str(resultado.get("imagen_procesada", ""))
    if len(name) >= 14 and name[:14].isdigit():
        dt = datetime.strptime(name[:14], "%Y%m%d%H%M%S")
        return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def active_tracks(estado):
    tracks = estado.get("tracks", [])
    active_ids = estado.get("tracks_activos", [])

    if not isinstance(tracks, list):
        return []

    active_set = {str(x) for x in active_ids} if isinstance(active_ids, list) else set()
    result = []

    for track in tracks:
        if not isinstance(track, dict):
            continue
        if active_set and str(track.get("track_id")) not in active_set:
            continue
        if track.get("activo") is False:
            continue
        result.append(track)

    return result


def main():
    resultado = load_json(RESULTADO, {})
    seguimiento = load_json(SEGUIMIENTO, {})
    estado = load_json(ESTADO, {})
    prediccion = load_json(PREDICCION, {})

    if not isinstance(resultado, dict):
        raise SystemExit("ERROR: resultado_procesamiento.json inválido")
    if not isinstance(estado, dict):
        raise SystemExit("ERROR: estado.json inválido")

    width = max(1.0, number(resultado.get("ancho_imagen"), 512))
    height = max(1.0, number(resultado.get("alto_imagen"), 512))
    center_x = width / 2.0
    center_y = height / 2.0
    km_per_px = RADAR_RANGE_KM / max(width, height)

    nucleos = resultado.get("nucleos", [])
    if not isinstance(nucleos, list):
        nucleos = []

    precip = [
        n for n in nucleos
        if isinstance(n, dict)
        and n.get("estado") in ("PRECIPITACION", "CAMBIO")
    ]

    tracks = active_tracks(estado)
    actividad = bool(precip)

    area_px = sum(
        max(0.0, number(n.get("pixeles")))
        for n in precip
    )

    intensidad = max(
        (
            int(number(n.get("intensidad"), 0))
            for n in precip
        ),
        default=0,
    )

    principal = None

    if tracks:
        principal = max(
            tracks,
            key=lambda t:
                max(1.0, number(t.get("pixeles")))
                * max(1.0, number(t.get("intensidad"), 1)),
        )

    dx = number(principal.get("dx_pixeles"), 0) if principal else 0
    dy = number(principal.get("dy_pixeles"), 0) if principal else 0
    x = number(principal.get("x"), center_x) if principal else center_x
    y = number(principal.get("y"), center_y) if principal else center_y

    to_center_x = center_x - x
    to_center_y = center_y - y

    # Indicador conservador: el movimiento del núcleo tiene componente
    # hacia el centro del dominio. No se presenta como una coordenada
    # geográfica exacta hasta calibrar la proyección del RMA10.
    toward = (
        dx * to_center_x + dy * to_center_y > 0
        if principal
        else False
    )

    speed_px_min = (
        number(principal.get("velocidad_pixeles_min"), 0)
        if principal
        else 0
    )

    speed_kmh = (
        round(speed_px_min * km_per_px * 60.0, 1)
        if speed_px_min > 0
        else None
    )

    distance_km = (
        round(
            math.hypot(to_center_x, to_center_y)
            * km_per_px,
            1,
        )
        if principal
        else None
    )

    eta_minutos = None

    if (
        toward
        and speed_kmh
        and speed_kmh > 5
        and distance_km is not None
    ):
        eta_minutos = round(
            distance_km / speed_kmh * 60.0,
            1,
        )

    tendencia = (
        principal.get("tendencia_area")
        if principal
        else None
    )

    fortalecimiento = {
        "CRECIENDO": "fortaleciendose",
        "DEBILITANDOSE": "debilitandose",
    }.get(tendencia, "estable")

    radar_ts = timestamp_from_result(resultado)

    salida = {
        "version": "RMA10-operativo-1.0",
        "fuente": "SMN RMA10",
        "actualizado": radar_ts,
        "nowcast": {
            "actualizado": radar_ts,
            "frame_actual_utc": radar_ts,
            "frames_analizados": len(nucleos),
            "tiles_ok": True,
            "actividad": actividad,
            "area_px": round(area_px, 2),
            "dbz_max": intensidad,
            "dbz_mean": intensidad,
            "distancia_km": distance_km,
            "fortalecimiento": fortalecimiento,
            "cambio_area_pct": round(
                number(
                    principal.get("cambio_area_porcentaje")
                ),
                2,
            ) if principal else 0,
            "velocidad_kmh": speed_kmh,
            "direccion": (
                principal.get("direccion")
                if principal
                else None
            ),
            "direccion_grados": (
                principal.get("rumbo_grados")
                if principal
                else None
            ),
            "movimiento_hacia_bahia": toward,
            "eta_minutos": eta_minutos,
            "confianza_movimiento": round(
                number(
                    principal.get("confianza_movimiento")
                ),
                3,
            ) if principal else 0,
            "estado": (
                "actividad_detectada"
                if actividad
                else "sin_precipitacion"
            ),
            "nucleos": precip,
            "nucleo_principal_id": (
                principal.get("track_id")
                if principal
                else None
            ),
            "proyecciones": {},
        },
        "diagnostico": {
            "sin_precipitacion": not actividad,
            "cobertura_pct": round(
                min(
                    100.0,
                    area_px / (width * height) * 100.0,
                ),
                2,
            ),
            "nucleos_precipitacion": len(precip),
            "nucleos_total": len(nucleos),
        },
        "tracking": {
            "tracks_activos": len(tracks),
            "track_principal": (
                principal.get("track_id")
                if principal
                else None
            ),
            "intervalo_minutos": seguimiento.get(
                "intervalo_minutos"
            ),
        },
        "prediccion": {
            "disponible": bool(prediccion),
            "radar_timestamp": (
                prediccion.get("radar_timestamp")
                if isinstance(prediccion, dict)
                else None
            ),
            "resumen": (
                prediccion.get("resumen", {})
                if isinstance(prediccion, dict)
                else {}
            ),
        },
        "calibracion": (
            prediccion.get("calibracion", {})
            if isinstance(prediccion, dict)
            else {}
        ),
        "escala": {
            "ancho_px": width,
            "alto_px": height,
            "km_por_pixel_estimado": round(
                km_per_px,
                5,
            ),
            "rango_diametro_km": RADAR_RANGE_KM,
        },
    }

    SALIDA.write_text(
        json.dumps(
            salida,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("===== ADAPTADOR RMA10 -> INTELLIGENCE =====")
    print("Actividad:", actividad)
    print("Núcleos precipitación:", len(precip))
    print("Tracks activos:", len(tracks))
    print("Velocidad estimada km/h:", speed_kmh)
    print("Dirección:", (
        principal.get("rumbo_grados") if principal else None
    ))
    print("Hacia centro:", toward)
    print("ETA estimada min:", eta_minutos)
    print("Salida:", SALIDA)


if __name__ == "__main__":
    main()
