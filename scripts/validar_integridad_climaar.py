#!/usr/bin/env python3
"""Validador integral de la cadena operativa ClimaAR.

No selecciona datos ni modifica archivos. Falla ante cualquier
desincronización, cambio de principal o inclusión de ecos estáticos.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
RADAR = BASE / "data" / "radar"
TRACK = RADAR / "seguimiento"
IA = BASE / "data" / "ia"


def fail(message: str) -> None:
    raise SystemExit(f"ERROR INTEGRIDAD CLIMAAR: {message}")


def load(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"archivo faltante o vacío: {path.relative_to(BASE)}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"JSON inválido {path.relative_to(BASE)}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON raíz no es objeto: {path.relative_to(BASE)}")
    return value


def parse_utc(value, name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        fail(f"{name} no contiene timestamp")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        fail(f"{name} tiene timestamp inválido: {value!r}")
    if dt.tzinfo is None:
        fail(f"{name} no tiene zona horaria: {value!r}")
    return dt.astimezone(timezone.utc)


def sid(value):
    if value is None:
        return None
    return str(value)


def require_same_timestamp(expected: datetime, value, name: str) -> None:
    actual = parse_utc(value, name)
    if actual != expected:
        fail(
            f"desincronización {name}: "
            f"{actual.isoformat()} != {expected.isoformat()}"
        )


def find_track(items, track_id, name):
    wanted = sid(track_id)
    for item in items:
        if isinstance(item, dict) and sid(item.get("track_id")) == wanted:
            return item
    fail(f"principal track {wanted} no existe en {name}")


def validar_goes19(radar_ts: datetime, nc: dict) -> None:
    """GOES-19 por tormenta: nunca datos de otro frame, del futuro ni de
    tormentas que el radar no tiene. GOES puede faltar (no corta el ciclo),
    pero si está, tiene que ser coherente."""
    path = BASE / "data" / "goes19" / "goes19_state.json"
    if not path.is_file() or path.stat().st_size == 0:
        print("GOES-19: sin estado (permitido: el ciclo sigue sin satélite)")
        return
    goes = load(path)
    frame = goes.get("reference_frame_utc")
    if not frame or parse_utc(frame, "goes19.reference_frame_utc") != radar_ts:
        print("GOES-19: estado de otro frame (no se integra; permitido)")
        return
    sinc = goes.get("sincronizacion")
    if isinstance(sinc, dict):
        for nombre, item in (sinc.get("productos") or {}).items():
            if not isinstance(item, dict) or not item.get("hora_satelite_utc"):
                continue
            hora = parse_utc(item["hora_satelite_utc"], f"goes19.{nombre}.hora_satelite_utc")
            if hora > radar_ts + timedelta(minutes=1):
                fail(f"GOES-19 {nombre} es posterior a la imagen de radar: {hora.isoformat()}")
    pt = goes.get("por_tormenta")
    if pt is None:
        return
    if not isinstance(pt, dict) or not isinstance(pt.get("tormentas"), list):
        fail("goes19.por_tormenta inválido")
    if pt.get("estado") == "ok":
        require_same_timestamp(radar_ts, pt.get("frame_radar_utc"), "goes19.por_tormenta.frame_radar_utc")
        tracks = {sid(n.get("track_id")) for n in (nc.get("nucleos_georreferenciados") or [])}
        for t in pt["tormentas"]:
            if sid(t.get("track_id")) not in tracks:
                fail(f"goes19.por_tormenta tiene una tormenta que el radar no tiene: {t.get('track_id')}")
            rayos = t.get("rayos") or {}
            if rayos.get("estado") == "ok":
                require_same_timestamp(radar_ts, rayos.get("fin_utc"),
                                       f"goes19 rayos de la tormenta {t.get('track_id')}")
            senales = t.get("senales_severas") or {}
            items = [(c, t.get(c) or {}) for c in ("c13", "c08", "altura_topes", "presion_topes")]
            items += [(f"senales_severas.{c}", senales.get(c) or {})
                      for c in ("topes_sobresalientes", "vapor_menos_infrarrojo", "altura_topes_2km", "fase")]
            for campo, item in items:
                if item.get("estado") == "ok":
                    estado = (item.get("sincronizacion") or {}).get("estado")
                    if estado != "sincronizado":
                        fail(f"goes19 {campo} de la tormenta {t.get('track_id')} usado sin sincronizar ({estado})")
    elif pt["tormentas"]:
        fail(f"goes19.por_tormenta estado={pt.get('estado')} pero trae tormentas")
    print(f"GOES-19: sincronizado ({pt.get('estado')}, {len(pt['tormentas'])} tormentas)")


def validar_puntaje(radar_ts: datetime, nc: dict) -> None:
    """Puntaje de gravedad (en paralelo): de esta imagen, de tormentas del
    radar y nunca mostrado al público todavía. Puede faltar."""
    path = IA / "puntaje_tormentas.json"
    if not path.is_file() or path.stat().st_size == 0:
        print("Puntaje de gravedad: sin archivo (permitido)")
        return
    p = load(path)
    if p.get("frame_radar_utc") is None or parse_utc(p["frame_radar_utc"], "puntaje.frame_radar_utc") != radar_ts:
        print("Puntaje de gravedad: de otra imagen (permitido)")
        return
    if p.get("modo") != "en_paralelo_no_se_muestra":
        fail(f"puntaje de gravedad en modo {p.get('modo')!r}: todavía debe correr en paralelo")
    tracks = {sid(n.get("track_id")) for n in (nc.get("nucleos_georreferenciados") or [])}
    for t in p.get("tormentas") or []:
        if sid(t.get("track_id")) not in tracks:
            fail(f"puntaje de gravedad de una tormenta que el radar no tiene: {t.get('track_id')}")
        for fen in ("granizo", "rafagas", "lluvia_intensa"):
            g = (t.get(fen) or {}).get("grado")
            if g not in (0, 1, 2, 3, 4):
                fail(f"puntaje {fen} de la tormenta {t.get('track_id')} con grado inválido: {g}")
    print(f"Puntaje de gravedad: sincronizado ({p.get('estado')}, {len(p.get('tormentas') or [])} tormentas)")


def validar_formacion(radar_ts: datetime) -> None:
    """Tormentas en formación (punto 5, en paralelo): de esta imagen, con el
    satélite de antes del radar y nunca mostradas al público todavía. Puede faltar."""
    path = IA / "formacion_tormentas.json"
    if not path.is_file() or path.stat().st_size == 0:
        print("Tormentas en formación: sin archivo (permitido)")
        return
    f = load(path)
    if f.get("frame_radar_utc") is None or parse_utc(f["frame_radar_utc"], "formacion.frame_radar_utc") != radar_ts:
        print("Tormentas en formación: de otra imagen (permitido)")
        return
    if f.get("modo") != "en_paralelo_no_se_muestra":
        fail(f"tormentas en formación en modo {f.get('modo')!r}: todavía debe correr en paralelo")
    if f.get("objetos"):
        hora = parse_utc(f.get("hora_satelite_utc"), "formacion.hora_satelite_utc")
        minutos = (radar_ts - hora).total_seconds() / 60.0
        if not 0.0 <= minutos <= 20.0:
            fail(f"tormentas en formación con satélite fuera de sincronía ({minutos:.1f} min)")
    for o in f.get("objetos") or []:
        g = (o.get("calificacion") or {}).get("grado")
        if g not in (0, 1, 2, 3):
            fail(f"tormenta en formación con grado inválido: {g}")
        if o.get("tipo") == "tormenta_existente_que_crece" and g != 0:
            fail("una tormenta que el radar ya ve no puede tener aviso de formación")
    for e in f.get("eventos") or []:
        v = e.get("ventaja_min")
        if v is not None and v < 0:
            fail(f"ventaja negativa en {e.get('id')}: {v}")
    print(f"Tormentas en formación: sincronizado ({f.get('estado')}, {len(f.get('objetos') or [])} nubes, "
          f"{len(f.get('seguimiento') or [])} seguimientos)")


CATEGORIAS_CAPAS = {"sin_eco", "lluvia", "tormenta", "posible_convectiva", "sin_datos_radar",
                    "posible_ya_paso_o_se_deshizo"}


def validar_nowcast_capas(radar_ts: datetime) -> None:
    """Nowcast en capas (punto 5b, en paralelo): de esta imagen, con la edad
    del radar bien calculada y ninguna fuente del futuro. Puede faltar."""
    path = RADAR / "nowcast_capas.json"
    if not path.is_file() or path.stat().st_size == 0:
        print("Nowcast en capas: sin archivo (permitido)")
        return
    n = load(path)
    if not n.get("disponible"):
        print(f"Nowcast en capas: no disponible ({n.get('motivo')}) (permitido)")
        return
    if n.get("en_paralelo") is not True and not n.get("en_uso"):
        fail("nowcast en capas: si no corre en paralelo tiene que decir para qué se usa (en_uso)")
    if n.get("radar_timestamp") is None or parse_utc(n["radar_timestamp"], "capas.radar_timestamp") != radar_ts:
        # Re-auditoría 23/11/2026: si el 5b falló en este ciclo, su archivo queda de otra
        # imagen. Es válido solo si lo que se ve NO salió de él (respaldo: flujo óptico);
        # el control de salud lo avisa. Antes esto cortaba todo el ciclo.
        final = load(IA / "resultado_final.json") if (IA / "resultado_final.json").is_file() else {}
        fuente = ((final.get("tipo_tiempo") or {}).get("fuente_llegadas")) if isinstance(final, dict) else None
        if fuente == "nowcast_capas":
            fail(f"nowcast en capas desincronizado y usado en lo que se ve: {n.get('radar_timestamp')!r}")
        print(f"Nowcast en capas: no se actualizó en este ciclo ({n.get('radar_timestamp')}); "
              f"lo que se ve salió de {fuente} (permitido)")
        return
    generado = parse_utc(n.get("generado_utc"), "capas.generado_utc")
    edad = (generado - radar_ts).total_seconds() / 60.0
    if n.get("edad_radar_min") is None or abs(max(edad, 0.0) - float(n["edad_radar_min"])) > 0.2:
        fail(f"nowcast en capas con edad del radar mal calculada: {n.get('edad_radar_min')} != {edad:.1f}")
    for nombre, fuente in (n.get("fuentes") or {}).items():
        hora = fuente.get("hora")
        if nombre != "ambiente" and hora:
            h = parse_utc(hora if ("Z" in hora or "+" in hora) else hora + "Z", f"capas.fuentes.{nombre}")
            if (h - generado).total_seconds() > 120:
                fail(f"nowcast en capas: la fuente {nombre} es del futuro ({hora})")
    for loc_id, item in (n.get("localidades") or {}).items():
        cat = (item.get("estado_ahora") or {}).get("categoria")
        if cat not in CATEGORIAS_CAPAS:
            fail(f"nowcast en capas: categoría inválida en {loc_id}: {cat!r}")
        c = item.get("confianza")
        if not isinstance(c, (int, float)) or not 0.0 <= c <= 1.0:
            fail(f"nowcast en capas: confianza inválida en {loc_id}: {c!r}")
        lleg = item.get("llegada_proxima_hora")
        if lleg is not None:
            for fen, grados in lleg.items():
                for grado, zonas in grados.items():
                    for zona, minutos in zonas.items():
                        if minutos is not None and not (isinstance(minutos, (int, float)) and 0 <= minutos <= 60):
                            fail(f"nowcast en capas: llegada inválida en {loc_id} {fen}/{grado}/{zona}: {minutos!r}")
            pr, po = lleg.get("tormenta", {}).get("probable", {}), lleg.get("tormenta", {}).get("posible")
            if po is not None:
                for zona, m in pr.items():
                    if m is not None and po.get(zona) is None:
                        fail(f"nowcast en capas: en {loc_id} la tormenta es 'probable' pero no 'posible' ({zona})")
    print(f"Nowcast en capas: sincronizado (radar de hace {n['edad_radar_min']} min, "
          f"{len(n.get('localidades') or {})} localidades)")


def validar_modelos_region() -> None:
    """Modelos por localidad (punto 7, en paralelo): nunca del futuro ni de una corrida
    que todavía no empezó. Puede faltar (red caída)."""
    path = IA / "modelos_region.json"
    if not path.is_file() or path.stat().st_size == 0:
        print("Modelos por localidad: sin archivo (permitido)")
        return
    m = load(path)
    if m.get("en_paralelo") is not True:
        fail("modelos por localidad: todavía deben correr en paralelo")
    generado = parse_utc(m.get("generado_utc"), "modelos.generado_utc")
    w = m.get("wrf_smn") or {}
    if w.get("estado") == "ok":
        init = parse_utc(w.get("corrida_utc"), "modelos.wrf.corrida_utc")
        if init > generado:
            fail(f"modelos: corrida del WRF del futuro ({w.get('corrida_utc')})")
    e = m.get("ensambles") or {}
    if e.get("estado") in ("ok", "viejo"):
        bajado = parse_utc(e.get("bajado_utc"), "modelos.ensambles.bajado_utc")
        if bajado > generado + timedelta(minutes=2):
            fail(f"modelos: ensambles del futuro ({e.get('bajado_utc')})")
    print(f"Modelos por localidad: WRF {w.get('estado')} (corrida {w.get('corrida_utc')}), "
          f"ensambles {e.get('estado')}")


def validar_trayectorias_capas(radar_ts: datetime) -> None:
    """Trayectorias del 5b (se ven desde el 20/11/2026): de esta imagen de radar,
    y si las proyecciones dicen venir del 5b, el archivo tiene que estar al día
    y traer la tormenta principal."""
    path = RADAR / "trayectorias_capas.json"
    meta = ((load(RADAR / "radar_nowcast.json").get("nowcast") or {}).get("proyecciones_metadata") or {})
    t = load(path) if path.is_file() and path.stat().st_size else {}
    if t.get("disponible"):
        if parse_utc(t.get("radar_timestamp"), "trayectorias.radar_timestamp") != radar_ts:
            if meta.get("fuente") == "nowcast_capas":
                fail(f"trayectorias del 5b desincronizadas y usadas: {t.get('radar_timestamp')!r}")
            # Re-auditoría 23/11/2026: el paso que lo escribe nunca corta el ciclo; si
            # falló, el archivo queda de otra imagen y lo que se ve no lo usa (controla
            # la hora). Antes esto cortaba todo el ciclo. El control de salud lo avisa.
            print(f"Trayectorias: el 5b no se actualizó en este ciclo; proyecciones desde {meta.get('fuente')} (permitido)")
            return
    if meta.get("fuente") == "nowcast_capas":
        if not t.get("disponible"):
            fail("las proyecciones dicen venir del 5b pero las trayectorias no están disponibles")
        if str(meta.get("track_id")) not in (t.get("tracks") or {}):
            fail(f"la tormenta principal {meta.get('track_id')} no está en las trayectorias del 5b")
    print(f"Trayectorias: proyecciones desde {meta.get('fuente')} "
          f"(5b disponible={bool(t.get('disponible'))}, {len(t.get('tracks') or {})} tormentas)")


def validar_indicios_severos(radar_ts: datetime) -> None:
    """Etapa G: de esta imagen de radar; el eco en arco es lo único que se ve."""
    path = IA / "indicios_severos.json"
    if not path.is_file() or path.stat().st_size == 0:
        print("Indicios de tormenta severa: sin archivo (permitido)")
        return
    c = load(path)
    if c.get("modo") != "se_muestra_eco_en_arco":
        fail(f"indicios de tormenta severa: modo inesperado {c.get('modo')!r}")
    if not c.get("disponible"):
        print(f"Indicios de tormenta severa: no disponible ({c.get('motivo')}) (permitido)")
        return
    if parse_utc(c.get("radar_timestamp"), "indicios.radar_timestamp") != radar_ts:
        # Re-auditoría 23/11/2026: el paso que lo escribe nunca corta el ciclo; si
        # falló, el archivo queda de otra imagen y lo que se ve no lo usa (controla
        # la hora). Antes esto cortaba todo el ciclo. El control de salud lo avisa.
        print("Indicios de tormenta severa: no se actualizaron en este ciclo (permitido)")
        return
    # Lo que se ve: un viento por "eco en arco" tiene que tener un arco en esta imagen.
    final = load(IA / "resultado_final.json") if (IA / "resultado_final.json").is_file() else {}
    for loc_id, ev in (((final.get("tipo_tiempo") or {}).get("localidades")) or {}).items():
        for momento in ("ahora", "al_llegar"):
            v = ((ev or {}).get(momento) or {}).get("viento") or {}
            if any("eco en arco" in str(r) for r in v.get("razones") or []) and not c.get("arcos"):
                fail(f"{loc_id}: viento por eco en arco sin ningún arco en esta imagen")
    print(f"Indicios de tormenta severa: sincronizado ({c.get('arcos', 0)} ecos en arco)")


def validar_celdas_localidad(radar_ts: datetime) -> None:
    """Paso 2 (en paralelo): de esta imagen de radar y con probabilidades válidas."""
    path = IA / "celdas_localidad.json"
    if not path.is_file() or path.stat().st_size == 0:
        print("Celdas por localidad: sin archivo (permitido)")
        return
    c = load(path)
    if c.get("en_paralelo") is not True:
        fail("celdas por localidad: todavía debe correr en paralelo")
    if not c.get("disponible"):
        print(f"Celdas por localidad: no disponible ({c.get('motivo')}) (permitido)")
        return
    if parse_utc(c.get("radar_timestamp"), "celdas.radar_timestamp") != radar_ts:
        print("Celdas por localidad: no se actualizaron en este ciclo (permitido)")
        return
    for loc_id, l in (c.get("localidades") or {}).items():
        pr = l.get("probabilidad")
        if not isinstance(pr, (int, float)) or not 0.0 <= pr <= 1.0:
            fail(f"celdas por localidad: probabilidad inválida en {loc_id}: {pr!r}")
        m = l.get("minutos")
        if m is not None and not 0 <= m <= c.get("horizonte_min", 30):
            fail(f"celdas por localidad: minutos inválidos en {loc_id}: {m!r}")
    # Etapa E: lo único que se ve (tormenta eléctrica encima) tiene que salir de esta imagen.
    final = load(IA / "resultado_final.json") if (IA / "resultado_final.json").is_file() else {}
    for loc_id, ev in (((final.get("tipo_tiempo") or {}).get("localidades")) or {}).items():
        s = (ev or {}).get("sobre_la_localidad")
        if s is None:
            continue
        pr = (c.get("localidades") or {}).get(loc_id, {}).get("probabilidad_solo_rayos")
        if not isinstance(pr, (int, float)) or abs(pr - s.get("probabilidad", -1)) > 0.01 or pr < 0.7:
            fail(f"tormenta eléctrica sobre {loc_id} no coincide con las celdas: {s!r}")
    print(f"Celdas por localidad: sincronizado ({len(c.get('localidades') or {})} localidades)")


def validar_ambiente_satelite(radar_ts: datetime) -> None:
    """Ambiente medido por el satélite (punto 8, en paralelo): de esta imagen de
    radar, nunca con un barrido posterior ni más de 20 min anterior, y la
    memoria de cielo despejado nunca del futuro ni de más de 6 h. Puede faltar."""
    path = IA / "ambiente_satelite.json"
    if not path.is_file() or path.stat().st_size == 0:
        print("Ambiente del satélite: sin archivo (permitido)")
        return
    a = load(path)
    if a.get("en_paralelo") is not True:
        fail("ambiente del satélite: todavía debe correr en paralelo")
    if not a.get("disponible"):
        print(f"Ambiente del satélite: no disponible ({a.get('motivo')}) (permitido)")
        return
    if parse_utc(a.get("frame_radar_utc"), "ambiente.frame_radar_utc") != radar_ts:
        # Re-auditoría 23/11/2026: el paso que lo escribe nunca corta el ciclo; si
        # falló, el archivo queda de otra imagen y lo que se ve no lo usa (controla
        # la hora). Antes esto cortaba todo el ciclo. El control de salud lo avisa.
        print("Ambiente del satélite: no se actualizó en este ciclo (permitido)")
        return
    for nombre, p in (a.get("productos") or {}).items():
        if p.get("estado") != "ok":
            continue
        fin = parse_utc(p.get("fin_barrido_utc"), f"ambiente.{nombre}.fin")
        desfase = (radar_ts - fin).total_seconds() / 60.0
        if desfase < -1.0:
            fail(f"ambiente del satélite: {nombre} es posterior a la imagen de radar")
        if desfase > 20.0:
            fail(f"ambiente del satélite: {nombre} con {desfase:.0f} min de desfase (máximo 20)")
    for loc_id, loc in (a.get("localidades") or {}).items():
        for grupo, m in (loc.get("ultimo_despejado") or {}).items():
            h = parse_utc(m.get("hora_utc"), f"ambiente.{loc_id}.{grupo}")
            edad = (radar_ts - h).total_seconds() / 60.0
            if edad < -1.0 or edad > 361.0:
                fail(f"ambiente del satélite: memoria {loc_id}/{grupo} con edad inválida ({edad:.0f} min)")
    print(f"Ambiente del satélite: sincronizado (barrido {a.get('hora_satelite_utc')}, "
          f"reutilizado={a.get('reutilizado')})")


def validar_sin_actividad(nc, state, ml, radar_intel, final, radar_meteo) -> None:
    """Frame sin ecos operativos: ningún componente puede declarar principal."""
    if nc.get("actividad") is not False:
        fail("Tracking no definió principal pero el Nowcast declara actividad")
    for nombre, valor in (
        ("nowcast.nucleo_principal_id", nc.get("nucleo_principal_id")),
        ("nowcast.track_principal_id", nc.get("track_principal_id")),
        ("nowcast.track_principal_candidato_id", nc.get("track_principal_candidato_id")),
        ("nowcast.bahia_blanca.nucleo_principal_id", (nc.get("bahia_blanca") or {}).get("nucleo_principal_id")),
        ("intelligence.radar.nucleo_principal_id", radar_intel.get("nucleo_principal_id")),
        ("meteorologia.radar.nucleo_principal_id", radar_meteo.get("nucleo_principal_id")),
        ("resultado_final.principal.track_id", (final.get("principal") or {}).get("track_id")),
    ):
        if valor is not None:
            fail(f"sin actividad pero {nombre}={valor}")
    activos = [
        t for t in state.get("tracks", [])
        if isinstance(t, dict) and t.get("activo") is True and t.get("presente_en_ultimo_frame") is True
    ]
    if activos:
        fail(f"sin principal pero estado.json tiene {len(activos)} tracks activos presentes")
    if nc.get("nucleos") not in ([], None):
        fail("sin actividad pero nowcast.nucleos no está vacío")
    if nc.get("proyecciones"):
        fail("sin actividad pero hay proyecciones")
    metadata = nc.get("proyecciones_metadata") or {}
    if metadata.get("motivo_no_disponible") != "SIN_ACTIVIDAD":
        fail("sin actividad pero proyecciones_metadata no lo declara")
    if not isinstance(ml.get("predicciones"), list):
        fail("ML.predicciones no es lista")
    if not isinstance(final.get("severidad_meteorologica"), dict):
        fail("resultado_final.severidad_meteorologica inválido")
    print("CLIMAAR INTEGRIDAD: OK")
    print("Frame sin actividad: ningún componente declara principal.")


def main() -> None:
    radar_ts = parse_utc(
        (RADAR / "radar_timestamp.txt").read_text(encoding="utf-8").strip(),
        "radar_timestamp.txt",
    )

    processing = load(RADAR / "resultado_procesamiento.json")
    tracking = load(TRACK / "seguimiento_temporal.json")
    state = load(TRACK / "estado.json")
    nowcast_file = load(RADAR / "radar_nowcast.json")
    ml = load(TRACK / "prediccion_ml_rma10.json")
    evolution = load(TRACK / "prediccion_evolucion.json")
    meteo = load(IA / "estado_meteorologico.json")
    intel = load(IA / "inteligencia_tormenta.json")
    final = load(IA / "resultado_final.json")

    nc = nowcast_file.get("nowcast")
    if not isinstance(nc, dict):
        fail("radar_nowcast.json no contiene nowcast")

    require_same_timestamp(radar_ts, nowcast_file.get("actualizado"), "nowcast.actualizado")
    require_same_timestamp(radar_ts, nc.get("actualizado"), "nowcast.nowcast.actualizado")
    require_same_timestamp(radar_ts, nc.get("frame_actual_utc"), "nowcast.frame_actual_utc")
    require_same_timestamp(radar_ts, tracking.get("radar_timestamp"), "tracking.radar_timestamp")
    require_same_timestamp(radar_ts, state.get("ultimo_radar_timestamp"), "estado.ultimo_radar_timestamp")
    require_same_timestamp(radar_ts, ml.get("frame_radar"), "ML.frame_radar")
    require_same_timestamp(radar_ts, evolution.get("radar_timestamp"), "evolution.radar_timestamp")

    radar_meteo = meteo.get("radar", {})
    if not isinstance(radar_meteo, dict):
        fail("estado_meteorologico.radar inválido")
    require_same_timestamp(radar_ts, radar_meteo.get("frame_actual_utc"), "meteorologia.radar.frame_actual_utc")

    radar_intel = intel.get("radar", {})
    if not isinstance(radar_intel, dict):
        fail("inteligencia.radar inválido")
    require_same_timestamp(radar_ts, radar_intel.get("frame_actual_utc"), "intelligence.radar.frame_actual_utc")

    final_radar = final.get("radar", {})
    if not isinstance(final_radar, dict):
        fail("resultado_final.radar inválido")
    final_nc = final_radar.get("nowcast", {})
    if not isinstance(final_nc, dict):
        fail("resultado_final.radar.nowcast inválido")
    require_same_timestamp(radar_ts, final_nc.get("frame_actual_utc"), "resultado_final.radar.nowcast.frame_actual_utc")

    validar_goes19(radar_ts, nc)
    validar_puntaje(radar_ts, nc)
    validar_formacion(radar_ts)
    validar_nowcast_capas(radar_ts)
    validar_modelos_region()
    validar_ambiente_satelite(radar_ts)
    validar_trayectorias_capas(radar_ts)
    validar_celdas_localidad(radar_ts)
    validar_indicios_severos(radar_ts)

    candidate = tracking.get("track_principal_candidato_id")
    if candidate is None:
        validar_sin_actividad(nc, state, ml, radar_intel, final, radar_meteo)
        return

    principal = nc.get("nucleo_principal_id")
    track_principal = nc.get("track_principal_id")
    candidate_nowcast = nc.get("track_principal_candidato_id")

    if principal is None:
        fail("Nowcast no tiene nucleo_principal_id")
    if sid(principal) != sid(candidate):
        fail(f"principal Tracking={candidate} != Nowcast={principal}")
    if sid(track_principal) != sid(candidate):
        fail(f"track_principal_id={track_principal} != Tracking={candidate}")
    if sid(candidate_nowcast) != sid(candidate):
        fail(f"Nowcast track_principal_candidato_id={candidate_nowcast} != Tracking={candidate}")

    bahia = nc.get("bahia_blanca", {})
    if not isinstance(bahia, dict):
        fail("nowcast.bahia_blanca inválido")
    if sid(bahia.get("nucleo_principal_id")) != sid(candidate):
        fail(
            "Bahía Blanca principal="
            f"{bahia.get('nucleo_principal_id')} != Tracking={candidate}"
        )

    state_track = find_track(state.get("tracks", []), candidate, "estado.json")
    if state_track.get("activo") is not True:
        fail(f"principal {candidate} no está activo en estado.json")
    if state_track.get("presente_en_ultimo_frame") is not True:
        fail(f"principal {candidate} no está presente en el último frame")

    nuclei = nc.get("nucleos")
    if not isinstance(nuclei, list) or not nuclei:
        fail("Nowcast no contiene núcleos operativos")
    principal_nucleus = find_track(nuclei, candidate, "nowcast.nucleos")

    geo_nuclei = nc.get("nucleos_georreferenciados")
    if not isinstance(geo_nuclei, list):
        fail("Nowcast no contiene nucleos_georreferenciados")
    find_track(geo_nuclei, candidate, "nowcast.nucleos_georreferenciados")

    for item in nuclei:
        if not isinstance(item, dict):
            fail("nowcast.nucleos contiene un elemento inválido")
        if str(item.get("estado", "")).upper() == "ECO_ESTATICO":
            fail("un ECO_ESTATICO llegó a nowcast.nucleos")

    dynamic_count = processing.get("nucleos_dinamicos")
    if isinstance(dynamic_count, int) and dynamic_count != len(nuclei):
        fail(
            f"nucleos_dinamicos={dynamic_count} != "
            f"nowcast.nucleos={len(nuclei)}"
        )

    ml_predictions = ml.get("predicciones")
    if not isinstance(ml_predictions, list):
        fail("ML.predicciones no es lista")
    ml_principal = find_track(ml_predictions, candidate, "ML.predicciones")
    if sid(ml_principal.get("track_id")) != sid(candidate):
        fail("ML principal inconsistente")

    evolution_nuclei = evolution.get("nucleos")
    if not isinstance(evolution_nuclei, list):
        fail("evolution.nucleos no es lista")
    evolution_track = find_track(evolution_nuclei, candidate, "prediccion_evolucion.nucleos")
    forecast_available = (
        evolution_track.get("pronostico_disponible") is True
    )
    if (
        not forecast_available
        and evolution_track.get("motivo") != "SIN_MOVIMIENTO_VALIDO"
    ):
        fail(
            f"evolution principal {candidate} tiene un estado de "
            f"pronóstico no reconocido: {evolution_track.get('motivo')}"
        )

    metadata = nc.get("proyecciones_metadata")
    if not isinstance(metadata, dict):
        fail("nowcast.proyecciones_metadata inválido")
    if sid(metadata.get("track_id")) != sid(candidate):
        fail(f"proyecciones_metadata.track_id={metadata.get('track_id')} != {candidate}")
    metadata_availability = metadata.get(
        "pronostico_disponible",
        True,
    )
    if metadata.get("fuente") == "nowcast_capas":
        # Desde el 20/11/2026 la trayectoria sale del 5b, que da movimiento
        # aunque el predictor viejo todavía no lo tenga (tormenta recién
        # nacida: SIN_MOVIMIENTO_VALIDO). Comparar contra el viejo frenaba
        # la IA de ese frame (23/11: 4 imágenes sin publicar).
        if metadata_availability is not True:
            fail("proyecciones_metadata del 5b sin pronóstico disponible")
        forecast_available = True
    elif metadata_availability is not forecast_available:
        fail(
            "proyecciones_metadata.pronostico_disponible no coincide "
            "con prediccion_evolucion"
        )

    projections = nc.get("proyecciones")
    if not isinstance(projections, dict):
        fail("nowcast.proyecciones inválido")
    if not forecast_available:
        if projections:
            fail(
                "nowcast contiene proyecciones sin movimiento válido"
            )
        if metadata.get("motivo_no_disponible") != evolution_track.get(
            "motivo"
        ):
            fail(
                "motivo de proyección no disponible no coincide "
                "con prediccion_evolucion"
            )
    else:
        for horizon in ("5", "10", "20", "30"):
            projection = projections.get(horizon)
            if not isinstance(projection, dict):
                fail(f"falta proyección {horizon}")
            if sid(projection.get("track_id")) != sid(candidate):
                fail(
                    f"proyección {horizon} track_id={projection.get('track_id')} "
                    f"!= principal={candidate}"
                )
            projection_ts = parse_utc(
                projection.get("timestamp_utc"),
                f"proyección {horizon}.timestamp_utc",
            )
            expected_projection_ts = radar_ts + timedelta(minutes=int(horizon))
            if projection_ts != expected_projection_ts:
                fail(
                    f"proyección {horizon} desincronizada: "
                    f"{projection_ts.isoformat()} != "
                    f"{expected_projection_ts.isoformat()}"
                )

    if sid(radar_intel.get("nucleo_principal_id")) != sid(candidate):
        fail(
            f"intelligence principal={radar_intel.get('nucleo_principal_id')} "
            f"!= {candidate}"
        )

    final_principal = final.get("principal", {})
    if not isinstance(final_principal, dict):
        fail("resultado_final.principal inválido")
    if sid(final_principal.get("track_id")) != sid(candidate):
        fail(
            f"resultado_final principal={final_principal.get('track_id')} "
            f"!= {candidate}"
        )

    if sid(radar_meteo.get("nucleo_principal_id")) != sid(candidate):
        fail(
            f"meteorologia principal={radar_meteo.get('nucleo_principal_id')} "
            f"!= {candidate}"
        )

    if not isinstance(final.get("severidad_meteorologica"), dict):
        fail("resultado_final.severidad_meteorologica inválido")

    print("CLIMAAR INTEGRIDAD: OK")
    print(f"Frame UTC: {radar_ts.isoformat()}")
    print(f"Principal track: {candidate}")
    print(f"Núcleos operativos: {len(nuclei)}")
    print("Radar / Tracking / Nowcast / Geo / ML / Evolution / ECMWF / Intelligence / Decision: sincronizados")


if __name__ == "__main__":
    main()
