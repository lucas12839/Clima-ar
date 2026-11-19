"""Verificación real de GOES-19 (necesita internet; correr en GitHub Actions).

1. Arma el estado GOES-19 completo para la imagen de radar actual y muestra,
   por producto, el modo de lectura, la sincronización y, por tormenta, topes,
   altura, presión y rayos.
2. Compara píxel por píxel la lectura parcial (solo la zona de 240 km) con la
   descarga completa del archivo: tienen que ser idénticas.

Uso: python herramientas/verificar_goes19.py
"""
import json, sys, time, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, requests
from goes19 import state as st
from goes19.geospatial import prepare_goes_domain_multi

t0 = time.time()
ref = st.load_radar_reference_time()
print("Frame radar:", ref.isoformat())
s = st.build_state(ref)
print(f"build_state: {time.time()-t0:.1f} s")
for n, v in {**s["products"], **s["cloudtop"]}.items():
    print(n, v.get("status"), v.get("lectura"), v.get("sincronizacion"), (v.get("domain") or {}).get("points"),
          {k: round(x, 1) for k, x in (v.get("statistics") or {}).items() if k in ("min", "median", "max")}, v.get("error", ""))
print("GLM:", s["glm"].get("rayos_frame_radar"), s["glm"]["ventana"].get("archivos_leidos"), s["glm"]["ventana"].get("archivos_esperados"))
pt = s["por_tormenta"]
print("POR TORMENTA:", pt["estado"], len(pt["tormentas"]))
for t in pt["tormentas"]:
    print(f"  tormenta {t['track_id']}: radar {t['dbz_max']} r {t['radio_km']} km | C13 {t['c13'].get('estado')} min {t['c13'].get('min_c')} p10 {t['c13'].get('p10_c')} corr {t['c13'].get('corrimiento_km')} px {t['c13'].get('pixeles')} | C08 min {t['c08'].get('min_c')} | altura max {t['altura_topes'].get('max_km')} ({t['altura_paralaje_km']} {t['altura_paralaje_origen']}) | pres min {t['presion_topes'].get('min_hpa')} | rayos {t['rayos'].get('cantidad')}")
    sev = t.get("senales_severas") or {}
    ot = sev.get("topes_sobresalientes") or {}
    vi = sev.get("vapor_menos_infrarrojo") or {}
    fa = sev.get("fase") or {}
    a2 = sev.get("altura_topes_2km") or {}
    co = sev.get("congelamiento") or {}
    print(f"     señales: topes sobresalientes {ot.get('cantidad')} (dif {ot.get('max_diferencia_yunque_c')}) | "
          f"vapor-IR max {vi.get('max_c')} | hielo {fa.get('fraccion_hielo_en_nubes')} | "
          f"altura 2 km {a2.get('estado')} {a2.get('max_km')} | congelamiento {(co.get('nivel') or {}).get('metros')} m, "
          f"tope sobre 0 °C {co.get('tope_sobre_congelamiento_km')} km")
for n, v in ((s.get("senales_severas") or {}).get("productos") or {}).items():
    print(n, v.get("status"), v.get("lectura"), v.get("sincronizacion"), v.get("pixeles_buenos_240km"), v.get("error", ""))
print("Topes sobresalientes en la zona:", (s.get("senales_severas") or {}).get("topes_sobresalientes_zona"))
loc = s["products"]["abi_c13"].get("por_localidad", {})
print("Topes por localidad (muestra):", {k: v for k, v in list(loc.items())[:4]})
json.dumps(s)

# Comparación exacta parcial vs completa para cada producto.
print("\nCOMPARACIÓN parcial vs completa")
for n, v in {**s["products"], **s["cloudtop"], **(s.get("senales_severas") or {}).get("productos", {})}.items():
    url = (v.get("source") or {}).get("url")
    if not url:
        continue
    var = {"abi_c08": ["CMI"], "abi_c13": ["CMI"], "cloudtop_height": ["HT", "DQF"], "cloudtop_pressure": ["PRES", "DQF"],
           "cloudtop_height_2km": ["HT", "DQF"], "cloud_phase": ["Phase", "DQF"]}[n]
    t1 = time.time(); a = prepare_goes_domain_multi(url, var); tp = time.time() - t1
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "f.nc"; t1 = time.time(); p.write_bytes(requests.get(url, timeout=300).content)
        mb = p.stat().st_size / 1e6; b = prepare_goes_domain_multi(p, var); tc = time.time() - t1
    iguales = all(np.array_equal(a["datos"][x], b["datos"][x], equal_nan=True) for x in var)
    print(f"{n}: parcial {tp:.1f} s, completa {tc:.1f} s ({mb:.1f} MB) -> idénticos: {iguales}, píxeles {a['datos'][var[0]].size}")
