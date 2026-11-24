/* ClimaAR · interfaz 24/11/2026
   Vista General: un solo pedido chico (/resumen) por minuto, y solo
   mientras la página está a la vista. El pronóstico se pide cada 30 min.
   Vista Meteorólogos: se piden los datos completos solo al abrirla. */
{
const API = (window.CLIMAAR_CONFIG && CLIMAAR_CONFIG.API_BASE_URL) || "";
const MIN_RESUMEN = 60 * 1000;
const MIN_PRONOSTICO = 30 * 60 * 1000;
const MIN_METEO = 2 * 60 * 1000;

const $ = id => document.getElementById(id);
const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const num = v => (v === null || v === undefined || v === "" || !isFinite(Number(v))) ? null : Number(v);
const fmt = (v, d = 0) => num(v) === null ? "—" : Number(v).toLocaleString("es-AR", { maximumFractionDigits: d, minimumFractionDigits: d });

const estado = {
  vista: "general",
  localidad: leer("climaar_localidad") || "bahia_blanca",
  resumen: null,
  pronostico: null,
  frameMostrado: null,
  tiempos: {},
  animando: null
};

function leer(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
function guardar(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }

async function pedir(ruta, ms = 20000) {
  const c = new AbortController();
  const t = setTimeout(() => c.abort(), ms);
  try {
    const r = await fetch(API + ruta, { cache: "no-store", signal: c.signal, headers: { Accept: "application/json" } });
    if (!r.ok) throw new Error(ruta + " " + r.status);
    return await r.json();
  } finally { clearTimeout(t); }
}

/* ---------- Textos ---------- */

const PUNTOS = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSO","SO","OSO","O","ONO","NO","NNO"];
const NOMBRE_PUNTO = { N:"norte", NNE:"norte-noreste", NE:"noreste", ENE:"este-noreste", E:"este", ESE:"este-sureste", SE:"sureste", SSE:"sur-sureste", S:"sur", SSO:"sur-suroeste", SO:"suroeste", OSO:"oeste-suroeste", O:"oeste", ONO:"oeste-noroeste", NO:"noroeste", NNO:"norte-noroeste" };
const punto = g => num(g) === null ? null : PUNTOS[Math.round(((Number(g) % 360) + 360) % 360 / 22.5) % 16];
const FENOMENOS = [["lluvia","Lluvia"],["viento","Viento"],["granizo","Granizo"],["electrica","Rayos"]];

function horaLocal(iso) {
  const d = iso ? new Date(iso) : null;
  if (!d || isNaN(d)) return "—";
  return d.toLocaleTimeString("es-AR", { hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "America/Argentina/Buenos_Aires" });
}

/* Cielo: código del tiempo de Open-Meteo; si falta, la nubosidad. */
function cielo(codigo, nubes, lluvia) {
  const c = num(codigo);
  if (c !== null) {
    if (c >= 95) return ["Tormenta", "tormenta"];
    if (c >= 80) return ["Chaparrones", "lluvia"];
    if (c >= 71 && c <= 77) return ["Nieve", "lluvia"];
    if (c >= 51) return ["Lluvia", "lluvia"];
    if (c >= 45) return ["Niebla", "nublado"];
    if (c === 3) return ["Nublado", "nublado"];
    if (c === 2) return ["Parcialmente nublado", "parcial"];
    if (c === 1) return ["Mayormente despejado", "sol"];
    return ["Despejado", "sol"];
  }
  if (num(lluvia) > 0) return ["Lluvia", "lluvia"];
  const n = num(nubes);
  if (n === null) return ["—", "parcial"];
  if (n < 20) return ["Despejado", "sol"];
  if (n < 50) return ["Algo nublado", "parcial"];
  if (n < 85) return ["Mayormente nublado", "nublado"];
  return ["Nublado", "nublado"];
}

function icono(tipo) {
  const sol = '<circle cx="24" cy="24" r="9" fill="#ffd45e"/><g stroke="#ffd45e" stroke-width="3" stroke-linecap="round"><path d="M24 5v5M24 38v5M5 24h5M38 24h5M10.5 10.5l3.5 3.5M34 34l3.5 3.5M10.5 37.5l3.5-3.5M34 14l3.5-3.5"/></g>';
  const nube = (x, c) => `<path transform="translate(${x} 0)" d="M15 38h20a8 8 0 0 0 0-16 11 11 0 0 0-21 3 6.5 6.5 0 0 0 1 13z" fill="${c}"/>`;
  const s = {
    sol: sol,
    parcial: '<g transform="translate(-6 -6) scale(.8)">' + sol + "</g>" + nube(4, "#dfe7f3"),
    nublado: nube(-6, "#8fa2bd") + nube(2, "#dfe7f3"),
    lluvia: nube(0, "#b9c6d9") + '<g stroke="#6fa4f2" stroke-width="3" stroke-linecap="round"><path d="M18 42l-2 4M26 42l-2 4M34 42l-2 4"/></g>',
    tormenta: nube(0, "#8fa2bd") + '<path d="M26 36l-6 8h5l-3 6 8-9h-5l3-5z" fill="#ffd45e"/>'
  }[tipo] || "";
  return `<svg viewBox="0 0 48 48" aria-hidden="true">${s}</svg>`;
}

/* ---------- Vista General ---------- */

function localidadActual() {
  const lista = (estado.resumen && estado.resumen.localidades) || [];
  return lista.find(l => l.id === estado.localidad) || lista[0] || null;
}

function llenarSelector() {
  const sel = $("localidad");
  const lista = (estado.resumen && estado.resumen.localidades) || [];
  if (sel.options.length === lista.length && lista.length) return;
  sel.innerHTML = lista.map(l => `<option value="${esc(l.id)}">${esc(l.nombre)}</option>`).join("");
  if (!lista.some(l => l.id === estado.localidad) && lista[0]) estado.localidad = lista[0].id;
  sel.value = estado.localidad;
}

function pintarTiempo() {
  const t = (estado.resumen && estado.resumen.tiempo_actual) || {};
  const [texto, tipo] = cielo(t.codigo_tiempo, t.nubosidad, t.precipitacion);
  $("temp").textContent = fmt(t.temperatura);
  $("cielo").textContent = t.disponible ? texto : "Sin datos del tiempo en este momento";
  $("cieloIcono").innerHTML = icono(tipo);
  $("sensacion").textContent = num(t.sensacion_termica) === null ? "—" : fmt(t.sensacion_termica) + " °C";
  const dir = punto(t.viento_dir);
  $("viento").textContent = num(t.viento_vel) === null ? "—" : fmt(t.viento_vel) + " km/h" + (dir ? " " + dir : "");
  $("viento").title = dir ? "Viene del " + NOMBRE_PUNTO[dir] : "";
  $("rafagas").textContent = num(t.rafagas) === null ? "—" : fmt(t.rafagas) + " km/h";
  $("humedad").textContent = num(t.humedad) === null ? "—" : fmt(t.humedad) + " %";
  $("presion").textContent = num(t.presion) === null ? "—" : fmt(t.presion) + " hPa";
  $("lluviaAhora").textContent = num(t.precipitacion) === null ? "—" : fmt(t.precipitacion, 1) + " mm";
  $("fuenteTiempo").textContent = t.fuente ? "Datos de Bahía Blanca · fuente: " + t.fuente : "";
}

function pintarTormenta() {
  const loc = localidadActual();
  const nivel = $("nivelGeneral");
  if (!loc || !loc.con_datos) {
    nivel.dataset.n = "0"; nivel.textContent = "Sin datos";
    $("resumenLocal").textContent = "Todavía no hay análisis para esta localidad.";
    $("fenomenos").innerHTML = ""; $("llegada").classList.add("oculto"); $("celdaCercana").textContent = "";
    return;
  }
  const n = num(loc.nivel) || 0;
  nivel.dataset.n = String(n);
  nivel.textContent = n ? "Nivel " + n : "Sin tormentas";
  $("resumenLocal").textContent = loc.resumen_ahora || (n ? loc.etiqueta : "Sin lluvia ni tormentas");

  const min = num(loc.minutos_hasta_llegada);
  const caja = $("llegada");
  if (min !== null) {
    const hora = new Date(Date.now() + min * 60000);
    caja.innerHTML = `Llega a ${esc(loc.nombre)} en <b>~${esc(fmt(min))} min</b> (cerca de las ${esc(horaLocal(hora.toISOString()))})` +
      (loc.resumen_al_llegar ? `<br>Al llegar: ${esc(loc.resumen_al_llegar)}` : "");
    caja.classList.remove("oculto");
  } else {
    caja.classList.add("oculto");
  }

  $("fenomenos").innerHTML = FENOMENOS.map(([k, nombre]) => {
    const f = (loc.fenomenos && loc.fenomenos[k]) || {};
    const g = num(f.grado) || 0;
    const barras = [1,2,3,4].map(i => `<i class="${i <= g ? "on" + g : ""}"></i>`).join("");
    let cat = f.categoria || "sin datos";
    if (g > 0 && !f.confirmado) cat = "posible · " + cat;
    if (f.en_calibracion) cat += " · en calibración";
    return `<li><span class="nombre">${esc(nombre)}</span><span class="barra" role="img" aria-label="${esc(nombre)} nivel ${g} de 4">${barras}</span><span class="cat">${esc(cat)}</span></li>`;
  }).join("");

  $("celdaCercana").textContent = textoCelda(loc);
}

function distKm(a, b) {
  const r = Math.PI / 180, dLat = (b[0] - a[0]) * r, dLon = (b[1] - a[1]) * r;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a[0] * r) * Math.cos(b[0] * r) * Math.sin(dLon / 2) ** 2;
  return 12742 * Math.asin(Math.sqrt(h));
}
function rumbo(a, b) {
  const r = Math.PI / 180;
  const y = Math.sin((b[1] - a[1]) * r) * Math.cos(b[0] * r);
  const x = Math.cos(a[0] * r) * Math.sin(b[0] * r) - Math.sin(a[0] * r) * Math.cos(b[0] * r) * Math.cos((b[1] - a[1]) * r);
  return (Math.atan2(y, x) / r + 360) % 360;
}

/* Describe la tormenta más cercana: dónde está y hacia dónde va.
   La hora de llegada solo la da el sistema (no se calcula acá). */
function textoCelda(loc) {
  const ts = (estado.resumen && estado.resumen.tormentas) || [];
  if (!ts.length) return "El radar no ve tormentas en la zona.";
  const p = [loc.lat, loc.lon];
  const c = ts.map(t => ({ t, d: distKm(p, [t.lat, t.lon]) })).sort((a, b) => a.d - b.d)[0];
  const donde = NOMBRE_PUNTO[punto(rumbo(p, [c.t.lat, c.t.lon]))];
  let txt = `Tormenta más cercana: a ${fmt(c.d)} km al ${donde}`;
  if (c.t.direccion && num(c.t.velocidad_kmh) > 2) txt += `, se mueve hacia el ${NOMBRE_PUNTO[c.t.direccion] || c.t.direccion} a ${fmt(c.t.velocidad_kmh)} km/h`;
  if (c.t.tendencia_area === "DEBILITANDOSE") txt += " y se está debilitando";
  else if (c.t.tendencia_area === "FORTALECIENDOSE" || c.t.tendencia_area === "CRECIENDO") txt += " y está creciendo";
  return txt + ".";
}

/* ---------- Mapa ---------- */

const mapa = { m: null, radar: null, capas: null };

// Bahía Blanca, White y Cerri están pegadas: sus nombres se muestran
// solo cuando están elegidas (si no, al tocar el círculo).
const CERCANAS = new Set(["bahia_blanca", "ingeniero_white", "general_cerri"]);
const LADO = { general_cerri: "left", ingeniero_white: "bottom", punta_alta: "right", medanos: "left", tornquist: "left", sierra_de_la_ventana: "right" };

function colorNivel(n) {
  return ["#33465f", "#9fd8f0", "#6fa4f2", "#6a63e6", "#b03bd4"][Math.max(0, Math.min(4, num(n) || 0))];
}

function iniciarMapa() {
  if (mapa.m || !window.L) return;
  mapa.m = L.map("mapa", { zoomControl: true, attributionControl: true }).setView([-38.55, -62.25], 8);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 12, attribution: "© OpenStreetMap" }).addTo(mapa.m);
  mapa.capas = L.layerGroup().addTo(mapa.m);
}

function limitesRadar() {
  const r = estado.resumen && estado.resumen.radar;
  if (r && Array.isArray(r.bounds)) return r.bounds;
  const c = (r && r.centro) || [-38.7342, -62.1654], km = (r && r.alcance_km) || 240;
  const dLat = km / 110.574, dLon = km / (111.32 * Math.cos(c[0] * Math.PI / 180));
  return [[c[0] - dLat, c[1] - dLon], [c[0] + dLat, c[1] + dLon]];
}

function urlImagen(ruta, clave) {
  return API + ruta + (ruta.includes("?") ? "&" : "?") + "f=" + encodeURIComponent(clave || "");
}

function ponerImagenRadar(ruta, clave) {
  if (!mapa.m) return;
  const url = urlImagen(ruta, clave);
  if (mapa.radar) mapa.radar.setUrl(url);
  else mapa.radar = L.imageOverlay(url, limitesRadar(), { opacity: .72 }).addTo(mapa.m);
}

function pintarMapa() {
  iniciarMapa();
  if (!mapa.m) return;
  const r = estado.resumen || {};
  const frame = r.frame_actual_utc || "";
  if (frame !== estado.frameMostrado && !estado.animando) {
    ponerImagenRadar((r.radar && r.radar.imagen) || "/radar/imagen", frame);
    estado.frameMostrado = frame;
  }
  mapa.capas.clearLayers();

  (r.localidades || []).forEach(l => {
    if (num(l.lat) === null) return;
    const sel = l.id === estado.localidad;
    L.circle([l.lat, l.lon], { radius: (num(l.radio_km) || 4) * 1000, color: sel ? "#fff" : colorNivel(l.nivel), weight: sel ? 3 : 2, fillOpacity: .08 })
      .bindTooltip(l.nombre, { permanent: sel || !CERCANAS.has(l.id), direction: LADO[l.id] || "top", className: "etiqueta-loc" })
      .on("click", () => elegirLocalidad(l.id))
      .addTo(mapa.capas);
  });

  (r.tormentas || []).forEach(t => {
    const camino = (t.recorrido || []).filter(p => Array.isArray(p));
    if (camino.length > 1) L.polyline(camino, { color: "#ffffff", weight: 3, opacity: .9 }).addTo(mapa.capas);
    const fut = (t.pronostico || []).filter(p => num(p.lat) !== null);
    if (fut.length) {
      L.polyline([[t.lat, t.lon], ...fut.map(p => [p.lat, p.lon])], { color: "#5fd0e0", weight: 3, dashArray: "6 6" }).addTo(mapa.capas);
      fut.forEach(p => L.circleMarker([p.lat, p.lon], { radius: 3, color: "#5fd0e0", fillOpacity: 1 })
        .bindTooltip(fmt(p.minutos) + "′", { permanent: true, direction: "right", className: "etiqueta-min" }).addTo(mapa.capas));
    }
    const info = `Intensidad ${esc(fmt(t.intensidad))}/6` +
      (t.direccion ? `<br>Va hacia el ${esc(NOMBRE_PUNTO[t.direccion] || t.direccion)} a ${esc(fmt(t.velocidad_kmh))} km/h` : "") +
      (num(t.tamano_km2) ? `<br>Tamaño ~${esc(fmt(t.tamano_km2))} km²` : "");
    L.circleMarker([t.lat, t.lon], { radius: 7, color: "#fff", weight: 2, fillColor: "#ff6b5e", fillOpacity: 1 }).bindPopup(info).addTo(mapa.capas);
  });

  const edad = num(r.frame_edad_minutos);
  $("radarHora").textContent = frame ? `Imagen del radar de las ${horaLocal(frame)}` + (edad !== null ? ` (hace ${fmt(edad)} min)` : "") + (r.radar_retrasado ? " · el radar viene atrasado" : "") : "Sin imagen del radar.";
}

async function animar() {
  const btn = $("animar");
  if (estado.animando) { clearInterval(estado.animando); estado.animando = null; btn.textContent = "Animar 2 h"; estado.frameMostrado = null; pintarMapa(); return; }
  btn.textContent = "Cargando…";
  try {
    const s = await pedir("/secuencia");
    const frames = (s.frames || []).filter(f => f.imagen);
    if (!frames.length) throw new Error("sin frames");
    frames.forEach(f => { const i = new Image(); i.src = urlImagen(f.imagen, f.timestamp); });
    let i = 0;
    btn.textContent = "Detener";
    estado.animando = setInterval(() => {
      const f = frames[i % frames.length];
      ponerImagenRadar(f.imagen, f.timestamp);
      const t = String(f.timestamp);
      const iso = `${t.slice(0,4)}-${t.slice(4,6)}-${t.slice(6,8)}T${t.slice(8,10)}:${t.slice(10,12)}:00Z`;
      $("radarHora").textContent = "Animación · " + horaLocal(iso);
      i++;
    }, 700);
  } catch (e) {
    btn.textContent = "Animar 2 h";
    $("radarHora").textContent = "No se pudo cargar la animación.";
  }
}

/* ---------- Pronóstico ---------- */

function pintarDias() {
  const p = estado.pronostico;
  const dias = (p && p.dias) || [];
  if (!dias.length) { $("dias").innerHTML = '<p class="pie">El pronóstico no está disponible en este momento.</p>'; return; }
  $("dias").innerHTML = dias.slice(0, 3).map(d => {
    const fecha = new Date(d.fecha + "T12:00:00-03:00");
    const nombre = isNaN(fecha) ? d.fecha : fecha.toLocaleDateString("es-AR", { weekday: "long", day: "numeric", timeZone: "America/Argentina/Buenos_Aires" });
    const [texto, tipo] = cielo(d.codigo_tiempo, null, null);
    const dir = punto(d.viento_direccion);
    return `<div class="dia"><div class="fecha">${esc(nombre)}</div>${icono(tipo)}
      <div class="temps">${esc(fmt(d.maxima))}° <span>${esc(fmt(d.minima))}°</span></div>
      <div class="mas">${esc(texto)}</div>
      <div class="mas">Lluvia ${esc(fmt(d.prob_lluvia))} %${num(d.lluvia_mm) > 0 ? " · " + esc(fmt(d.lluvia_mm, 1)) + " mm" : ""}</div>
      <div class="mas">Viento ${esc(fmt(d.viento))} km/h${dir ? " " + esc(dir) : ""}</div>
      <div class="mas">Ráfagas ${esc(fmt(d.rafagas))} km/h</div></div>`;
  }).join("") + (p.ejemplo ? '<p class="pie" style="grid-column:1/-1">Datos de ejemplo (solo en el boceto).</p>' : "");
}

/* ---------- Vista Meteorólogos ---------- */

const NOMBRES_ES = {
  temperature_2m: "temperatura (°C)", dew_point_2m: "punto de rocío (°C)", pressure_msl: "presión a nivel del mar (hPa)",
  cape: "CAPE, energía para tormentas (J/kg)", freezing_level_height: "altura de congelamiento (m)",
  total_column_integrated_water_vapour: "agua precipitable (mm)",
  temperature_850hPa: "temperatura 850 hPa (°C)", temperature_700hPa: "temperatura 700 hPa (°C)", temperature_500hPa: "temperatura 500 hPa (°C)",
  wind_speed_850hPa: "viento 850 hPa (km/h)", wind_direction_850hPa: "dirección 850 hPa (°)",
  wind_speed_500hPa: "viento 500 hPa (km/h)", wind_direction_500hPa: "dirección 500 hPa (°)",
  relative_humidity_700hPa: "humedad 700 hPa (%)", relative_humidity_500hPa: "humedad 500 hPa (%)",
  obsTime_utc: "hora de la observación", viento_kt: "viento (nudos)", visibilidad_millas: "visibilidad (millas)",
  cizalladura_850_500_kmh: "cizalladura 850-500 hPa (km/h)", gradiente_850_500_c_km: "gradiente térmico 850-500 (°C/km)",
  humedad_media_minima_pct: "humedad media mínima (%)"
};
const ETIQ = s => NOMBRES_ES[s] || String(s).replace(/_utc$/, "").replace(/_/g, " ");
const valor = (k, v) => typeof v === "number" ? fmt(v, Number.isInteger(v) ? 0 : 2) : (/_utc$/.test(k) ? horaLocal(v) : v);
function kv(obj, claves) {
  if (!obj || typeof obj !== "object") return '<p class="pie">Sin datos.</p>';
  const filas = (claves || Object.keys(obj)).filter(k => obj[k] !== null && obj[k] !== undefined && typeof obj[k] !== "object")
    .map(k => `<dt>${esc(ETIQ(k))}</dt><dd>${esc(valor(k, obj[k]))}</dd>`).join("");
  return filas ? `<dl class="kv">${filas}</dl>` : '<p class="pie">Sin datos.</p>';
}
function tabla(cab, filas) {
  return `<div class="tabla"><table><thead><tr>${cab.map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${
    filas.map(f => `<tr>${f.map(c => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}
const bloque = (titulo, html, abierto) => `<details${abierto ? " open" : ""}><summary>${esc(titulo)}</summary><div class="cuerpo">${html}</div></details>`;

async function cargarMeteo() {
  const st = $("meteoEstado");
  try {
    const [intel, radar, pred] = await Promise.all([pedir("/inteligencia", 40000), pedir("/radar"), pedir("/prediccion/en-vivo").catch(() => null)]);
    pintarMeteo(intel || {}, radar || {}, pred || {});
    st.textContent = "Actualizado " + horaLocal(new Date().toISOString()) + " · frame del radar " + horaLocal(radar.frame_actual_utc);
  } catch (e) {
    st.textContent = "No se pudieron cargar los datos completos. Se reintenta solo.";
  }
}

function pintarMeteo(intel, radar, pred) {
  const abiertos = new Set([...document.querySelectorAll("#meteoContenido details[open] summary")].map(s => s.textContent));
  const final = intel.resultado_final || {};
  const tipo = final.tipo_tiempo || {};
  const storm = intel.inteligencia_tormenta || {};
  const meteo = intel.estado_meteorologico || {};
  const evol = intel.prediccion_evolucion || {};
  const partes = [];

  partes.push(bloque("Estado del radar y del sistema", kv({
    frame_radar: horaLocal(radar.frame_actual_utc), edad_min: radar.frame_edad_minutos, estado_radar: radar.estado_radar,
    retrasado: radar.radar_retrasado ? "sí" : "no", nucleos_detectados: (radar.nucleos || []).length,
    sincronizados: intel.sincronizacion ? (intel.sincronizacion.sincronizados ? "sí" : "no") : "—",
    nowcast: horaLocal(intel.sincronizacion && intel.sincronizacion.nowcast), calidad_datos: storm.calidad_datos,
    calidad_seguimiento: storm.calidad_tracking, confianza_modelo: storm.confianza_modelo
  })));

  const nuc = (radar.nucleos || []).map(n => {
    const g = n.georreferencia || {}, m = n.movimiento || {}, d = n.dbz || {};
    return [n.track_id ?? n.nucleo_id, fmt(d.max, 1), fmt(d.p95, 1), n.nivel_intensidad || "—", fmt(g.distancia_bahia_km), punto(m.direccion_grados) || "—", m.tendencia || "—", fmt(m.cambio_area_pct), fmt(m.confianza_track, 2)];
  });
  partes.push(bloque("Núcleos del radar", nuc.length ? tabla(["Celda","dBZ máx","dBZ p95","Intensidad","Dist. BB km","Va hacia","Tendencia","Área %","Conf."], nuc) : '<p class="pie">Sin núcleos.</p>'));

  const locs = tipo.localidades || {};
  const filasLoc = Object.keys(locs).map(k => {
    const l = locs[k], f = l.fenomenos || {}, c = x => { const v = f[x] || {}; return `${v.grado ?? "—"} ${v.categoria || ""}${v.grado > 0 && !v.confirmado ? " (posible)" : ""}`; };
    const ac = l.acumulado_radar_mm || {};
    return [l.nombre || k, (l.calificacion && l.calificacion.grado) ?? "—", c("lluvia"), c("viento"), c("granizo"), c("electrica"), num(l.minutos_hasta_llegada) === null ? "—" : fmt(l.minutos_hasta_llegada) + " min", fmt(ac["1h"], 1) + " / " + fmt(ac["3h"], 1), fmt(l.dbz_entorno_20km, 1)];
  });
  partes.push(bloque("Por localidad", tabla(["Localidad","Nivel","Lluvia","Viento","Granizo","Rayos","Llega en","Radar mm 1h/3h","dBZ 20 km"], filasLoc)));

  const fuentes = Object.keys(locs).map(k => {
    const f = (locs[k].fenomenos) || {};
    return `<p class="subt">${esc(locs[k].nombre || k)}</p><ul class="lista">${FENOMENOS.map(([x, n]) => {
      const v = f[x] || {}; const r = (v.razones || []).join("; ");
      return `<li>${esc(n)}: fuente ${esc(v.fuente || "—")}${v.en_calibracion ? " · en calibración" : ""}${r ? " · " + esc(r) : ""}</li>`; }).join("")}</ul>`;
  }).join("");
  partes.push(bloque("Fuente de cada fenómeno", fuentes || '<p class="pie">Sin datos.</p>'));

  const amb = meteo.ambiente_actual || {};
  partes.push(bloque("Ambiente (modelo europeo)", `<p class="subt">${esc((tipo.ambiente && tipo.ambiente.etiqueta) || "")}</p>` + kv(tipo.ambiente, ["cape","cizalladura_850_500_kmh","gradiente_850_500_c_km","humedad_media_minima_pct"]) +
    '<p class="subt">Perfil</p>' + kv(amb, ["temperature_2m","dew_point_2m","pressure_msl","cape","freezing_level_height","total_column_integrated_water_vapour","temperature_850hPa","temperature_700hPa","temperature_500hPa","wind_speed_850hPa","wind_direction_850hPa","wind_speed_500hPa","wind_direction_500hPa","relative_humidity_700hPa","relative_humidity_500hPa"])));

  partes.push(bloque("Aeropuerto (SAZB)", kv(meteo.observacion_sazb, ["estacion","obsTime_utc","edad_minutos","temperatura_c","punto_rocio_c","viento_kt","direccion_viento","presion_hpa","visibilidad_millas"])));

  const res = evol.resumen || {};
  const met = res.metricas_por_horizonte || {};
  const filasH = Object.keys(met).map(h => { const m = met[h] || {}; return [h + " min", ...Object.keys(m).slice(0, 4).map(k => typeof m[k] === "number" ? fmt(m[k], 1) : String(m[k]))]; });
  const cabH = Object.keys(met).length ? ["Horizonte", ...Object.keys(met[Object.keys(met)[0]] || {}).slice(0, 4).map(ETIQ)] : [];
  partes.push(bloque("Predicción entre frames", `<p class="pie">${esc(pred.nota || "")}</p>` + kv(res) + (filasH.length ? '<p class="subt">Por horizonte</p>' + tabla(cabH, filasH) : "")));

  const g = storm.goes19 || {};
  const prod = g.productos || {};
  const filasG = Object.keys(prod).map(k => { const p = prod[k] || {}, s = p.statistics || {}, src = p.source || {}; return [k, p.status || "—", fmt(src.age_minutes, 1), fmt(s.min, 1), fmt(s.p10, 1), fmt(s.mean, 1)]; });
  partes.push(bloque("Satélite GOES-19", kv({ adquisicion: g.acquisition, generado: horaLocal(g.generated_at_utc) }) + (filasG.length ? tabla(["Producto","Estado","Edad min","Mín","p10","Media"], filasG) : "")));

  partes.push(bloque("Motor de decisión", kv({ estado: storm.estado, clasificacion: storm.clasificacion_meteorologica, indice_hibrido: storm.indice_hibrido_severidad, duracion_estimada_min: storm.duracion_estimada_minutos, probabilidades: storm.interpretacion_probabilidades }) +
    ((storm.razones || []).length ? '<p class="subt">Razones</p><ul class="lista">' + storm.razones.map(r => `<li>${esc(r)}</li>`).join("") + "</ul>" : "")));

  $("meteoContenido").innerHTML = partes.join("");
  document.querySelectorAll("#meteoContenido details").forEach((d, i) => {
    const t = d.querySelector("summary").textContent;
    if (abiertos.size ? abiertos.has(t) : i === 0) d.open = true;
  });
}

/* ---------- Ciclo de actualización ---------- */

async function actualizarGeneral() {
  try {
    estado.resumen = await pedir("/resumen");
    llenarSelector(); pintarTiempo(); pintarTormenta(); pintarMapa();
  } catch (e) {
    $("cielo").textContent = "Sin conexión con el servidor. Se reintenta solo.";
  }
}
async function actualizarPronostico() {
  try { estado.pronostico = await pedir("/pronostico"); } catch (e) { estado.pronostico = null; }
  pintarDias();
}

function programar() {
  Object.values(estado.tiempos).forEach(clearInterval);
  estado.tiempos = {};
  if (document.hidden) return;   // nadie mira: no se pide nada
  if (estado.vista === "general") {
    estado.tiempos.resumen = setInterval(actualizarGeneral, MIN_RESUMEN);
    estado.tiempos.pron = setInterval(actualizarPronostico, MIN_PRONOSTICO);
  } else {
    estado.tiempos.meteo = setInterval(cargarMeteo, MIN_METEO);
  }
}

function elegirLocalidad(id) {
  estado.localidad = id; guardar("climaar_localidad", id);
  $("localidad").value = id;
  pintarTormenta(); pintarMapa();
}

function cambiarVista(v) {
  estado.vista = v;
  $("general").classList.toggle("oculto", v !== "general");
  $("meteo").classList.toggle("oculto", v !== "meteo");
  $("tabGeneral").setAttribute("aria-selected", String(v === "general"));
  $("tabMeteo").setAttribute("aria-selected", String(v === "meteo"));
  if (v === "meteo") cargarMeteo();
  else { actualizarGeneral(); if (mapa.m) setTimeout(() => mapa.m.invalidateSize(), 50); }
  programar();
}

$("localidad").addEventListener("change", e => elegirLocalidad(e.target.value));
document.querySelectorAll(".vistas button").forEach(b => b.addEventListener("click", () => cambiarVista(b.dataset.vista)));
$("animar").addEventListener("click", animar);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) { if (estado.vista === "general") actualizarGeneral(); else cargarMeteo(); }
  programar();
});

actualizarGeneral();
actualizarPronostico();
programar();
}
