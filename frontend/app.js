/* ClimaAR · interfaz 24/11/2026
   Vista General: un solo pedido chico (/resumen) por minuto, y solo
   mientras la página está a la vista. El pronóstico se pide cada 30 min.
   Vista Meteorólogos: se piden los datos completos solo al abrirla. */
{
// config.js declara CLIMAAR_CONFIG con "const", que no queda en window:
// hay que preguntarlo directo (24/11/2026: la app pedía /resumen sin /api).
const API = (typeof CLIMAAR_CONFIG !== "undefined" && CLIMAAR_CONFIG.API_BASE_URL) || "/api";
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
  tiempos: {}
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

function esDeNoche() {
  const h = Number(new Date().toLocaleString("en-US", { hour: "numeric", hour12: false, timeZone: "America/Argentina/Buenos_Aires" }));
  return h >= 21 || h < 7;
}

function icono(tipo, dia) {
  const sol = '<circle cx="24" cy="24" r="9" fill="#ffd45e"/><g stroke="#ffd45e" stroke-width="3" stroke-linecap="round"><path d="M24 5v5M24 38v5M5 24h5M38 24h5M10.5 10.5l3.5 3.5M34 34l3.5 3.5M10.5 37.5l3.5-3.5M34 14l3.5-3.5"/></g>';
  const nube = (x, c) => `<path transform="translate(${x} 0)" d="M15 38h20a8 8 0 0 0 0-16 11 11 0 0 0-21 3 6.5 6.5 0 0 0 1 13z" fill="${c}"/>`;
  const luna = '<path d="M30 8a15 15 0 1 0 10 26A13 13 0 0 1 30 8z" fill="#e6ecf7"/>';
  if (!dia && esDeNoche()) {
    if (tipo === "sol") tipo = "luna";
    else if (tipo === "parcial") tipo = "luna_parcial";
  }
  const s = {
    luna: luna,
    luna_parcial: '<g transform="translate(-6 -6) scale(.8)">' + luna + "</g>" + nube(4, "#dfe7f3"),
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
  $("sensacion").parentElement.classList.toggle("oculto", num(t.sensacion_termica) === null);
  const dir = punto(t.viento_dir);
  $("viento").textContent = num(t.viento_vel) === null ? "—" : fmt(t.viento_vel) + " km/h" + (dir ? " " + dir : "");
  $("viento").title = dir ? "Viene del " + NOMBRE_PUNTO[dir] : "";
  $("rafagas").textContent = num(t.rafagas) === null ? "—" : fmt(t.rafagas) + " km/h";
  $("humedad").textContent = num(t.humedad) === null ? "—" : fmt(t.humedad) + " %";
  $("presion").textContent = num(t.presion) === null ? "—" : fmt(t.presion) + " hPa";
  $("lluviaAhora").textContent = num(t.precipitacion) === null ? "—" : fmt(t.precipitacion, 1) + " mm";
  $("fuenteTiempo").textContent = t.fuente ? "Datos de Bahía Blanca · fuente: " + t.fuente : "";
}

const ICONO_FEN = {
  lluvia: '<path d="M12 3c-3 4.5-6 8-6 11a6 6 0 0 0 12 0c0-3-3-6.5-6-11z"/>',
  viento: '<path d="M3 9h11a3 3 0 1 0-3-3M3 14h15a3 3 0 1 1-3 3M3 19h7" fill="none" stroke-width="2" stroke-linecap="round"/>',
  granizo: '<circle cx="7" cy="8" r="3"/><circle cx="16" cy="7" r="2.5"/><circle cx="11" cy="15" r="3.2"/><circle cx="18" cy="16" r="2.2"/>',
  electrica: '<path d="M13 2 4 14h6l-2 8 10-13h-6l3-7z"/>'
};
const NIVEL_TXT = ["Sin tormentas", "Nivel 1 · Débil", "Nivel 2 · Moderado", "Nivel 3 · Fuerte", "Nivel 4 · Severo"];

function pintarTormenta() {
  const loc = localidadActual();
  const nivel = $("nivelGeneral");
  $("focoNombre").textContent = loc ? loc.nombre : "—";
  if (!loc || !loc.con_datos) {
    nivel.dataset.n = "0"; nivel.textContent = "Sin datos"; $("foco").dataset.n = "0";
    $("resumenLocal").textContent = "Todavía no hay análisis para esta localidad.";
    $("fenomenos").innerHTML = ""; $("llegada").classList.add("oculto"); $("celdaCercana").textContent = "";
    return;
  }
  const n = num(loc.nivel) || 0;
  nivel.dataset.n = String(n);
  $("foco").dataset.n = String(n);
  // Un 4 sin confirmar por una estación es "posible severo".
  nivel.textContent = n === 4 && /^posible/i.test(loc.etiqueta || "") ? "Nivel 4 · Posible severo" : (NIVEL_TXT[n] || "Nivel " + n);
  // El título va con el nivel (incluye lo de los últimos minutos); si ahora
  // es distinto, se aclara abajo para que no parezca una contradicción.
  const titulo = loc.resumen || loc.resumen_ahora || (n ? loc.etiqueta : "Sin lluvia ni tormentas");
  $("resumenLocal").innerHTML = esc(titulo) +
    (loc.resumen_ahora && loc.resumen && loc.resumen_ahora !== loc.resumen ? `<span class="foco-ahora">En este momento: ${esc(loc.resumen_ahora.charAt(0).toLowerCase() + loc.resumen_ahora.slice(1))}</span>` : "");

  const min = num(loc.minutos_hasta_llegada);
  const caja = $("llegada");
  if (min !== null) {
    const hora = new Date(Date.now() + min * 60000);
    caja.innerHTML = `<div class="cuenta-min"><b>${esc(fmt(min))}</b><span>min</span></div>
      <div class="cuenta-txt">para que llegue a ${esc(loc.nombre)}<br><strong>cerca de las ${esc(horaLocal(hora.toISOString()))}</strong>` +
      (loc.resumen_al_llegar ? `<br><span class="al-llegar">Al llegar: ${esc(loc.resumen_al_llegar)}</span>` : "") + `</div>`;
    caja.classList.remove("oculto");
  } else if (loc.sobre_la_localidad) {
    caja.innerHTML = `<div class="cuenta-txt"><strong>La tormenta está sobre ${esc(loc.nombre)}</strong></div>`;
    caja.classList.remove("oculto");
  } else {
    caja.classList.add("oculto");
  }

  $("fenomenos").innerHTML = FENOMENOS.map(([k, nombre]) => {
    const f = (loc.fenomenos && loc.fenomenos[k]) || {};
    const g = num(f.grado) || 0;
    const barras = [1,2,3,4].map(i => `<i class="${i <= g ? "on" + g : ""}"></i>`).join("");
    let cat = f.categoria || "sin datos";
    if (g > 0 && !f.confirmado && !/^posible/i.test(cat)) cat = "posible · " + cat;
    if (num(f.minutos_desde_deteccion) > 0) cat += ` · hace ${fmt(f.minutos_desde_deteccion)} min`;
    if (f.en_calibracion) cat += " · en calibración";
    const [dato, detalles] = cifrasFenomeno(k, f, loc);
    return `<li data-g="${g}"><svg class="fen-icono" viewBox="0 0 24 24" aria-hidden="true">${ICONO_FEN[k]}</svg>
      <span class="nombre">${esc(nombre)}</span>
      ${dato ? `<span class="fen-dato">${esc(dato)}</span>` : ""}
      <span class="barra" role="img" aria-label="${esc(nombre)} nivel ${g} de 4">${barras}</span>
      <span class="cat">${esc(cat)}</span>
      ${detalles.length ? `<span class="fen-det">${detalles.map(esc).join("<br>")}</span>` : ""}</li>`;
  }).join("");

  $("celdaCercana").textContent = textoCelda(loc);
}

const FUENTE_TXT = { radar: "radar", estaciones: "estaciones", modelo: "modelo europeo", goes19: "satélite", radar_movido: "radar estimado" };
const fuenteTxt = f => String(f || "").split("+").map(x => FUENTE_TXT[x] || x).filter(Boolean).join(" + ");

/* Números que acompañan a cada nivel: [dato grande, renglones de detalle].
   Todo sale de lo que midió o calculó la estructura; si no hay número, no se inventa. */
function cifrasFenomeno(k, f, loc) {
  const det = [];
  const llega = (loc.al_llegar && loc.al_llegar[k]) || null;
  const minLlega = num(loc.minutos_hasta_llegada);
  let dato = null;
  if (k === "lluvia") {
    if (num(f.mm_h) !== null && num(f.grado) > 0) dato = `${fmt(f.mm_h, 1)} mm/h`;
    else if (num(f.mm_1h) !== null) dato = `${fmt(f.mm_1h, 1)} mm en 1 h`;
    const ac = loc.acumulado_radar_mm || {};
    if (num(f.mm_3h) !== null) det.push(`Medido: ${fmt(f.mm_1h, 1)} mm en 1 h · ${fmt(f.mm_3h, 1)} mm en 3 h`);
    else if (num(ac["1h"]) >= 0.5 || num(ac["3h"]) >= 0.5) det.push(`Ya llovió: ${fmt(ac["1h"], 1)} mm en 1 h · ${fmt(ac["3h"], 1)} mm en 3 h (radar)`);
    if (llega && num(llega.mm_h) > 0 && minLlega !== null && f.cuando !== "al_llegar") det.push(`Puede llover al llegar: ~${fmt(llega.mm_h, 0)} mm/h`);
    const hoy = estado.pronostico && estado.pronostico.dias && estado.pronostico.dias[0];
    if (hoy && num(hoy.prob_lluvia) !== null) det.push(`Hoy en la zona: ${fmt(hoy.prob_lluvia)} % de probabilidad` + (num(hoy.lluvia_mm) > 0 ? ` · ${fmt(hoy.lluvia_mm, 1)} mm (pronóstico)` : ""));
  } else if (k === "viento") {
    if (num(f.rafaga_kmh) !== null) dato = `ráfagas ${fmt(f.rafaga_kmh)} km/h`;
    if (num(f.viento_kmh) !== null) det.push(`Viento medio: ${fmt(f.viento_kmh)} km/h`);
    if (llega && num(llega.rafaga_kmh) > num(f.rafaga_kmh || 0) && minLlega !== null) det.push(`Al llegar: ráfagas de ~${fmt(llega.rafaga_kmh)} km/h`);
  } else if (k === "granizo") {
    const m = /(\d+)\s*dBZ/.exec(f.razon || "");
    if (m && num(f.grado) > 0) det.push(`Señal del radar: ${m[1]} dBZ (el granizo es posible desde 50)`);
    if (llega && num(llega.grado) > num(f.grado || 0) && minLlega !== null) det.push(`Al llegar: granizo ${llega.categoria}`);
  } else if (k === "electrica") {
    if (num(f.tasa_por_min) > 0) dato = `${fmt(f.tasa_por_min, 1)} rayos/min`;
    if (num(f.grado) > 0) det.push("A menos de 30 km de la localidad");
  }
  // El nivel puede venir de la tormenta que llega: que se note.
  if (dato && f.cuando === "al_llegar") dato = dato.replace(/(\d[\d.,]*)/, "~$1") + " al llegar";
  if (num(f.grado) > 0 && f.fuente) det.push(`Fuente: ${fuenteTxt(f.fuente)}`);
  return [dato, det];
}

/* Lo que ClimaAR está vigilando en este momento (todo con la edad calculada ahora). */
function pintarVivo() {
  const r = estado.resumen || {};
  const frame = r.frame_actual_utc, edad = num(r.frame_edad_minutos);
  const chip = (id, valor, detalle, estadoChip) => {
    const c = $(id);
    c.querySelector("b").textContent = valor;
    if (detalle !== undefined) c.querySelector("small").textContent = detalle;
    c.dataset.estado = estadoChip || "ok";
  };
  chip("chipRadar", frame ? horaLocal(frame) : "sin imagen",
    edad !== null ? `hace ${fmt(edad)} min` : "—",
    !frame ? "mal" : r.radar_retrasado || (edad !== null && edad > 20) ? "atraso" : "ok");
  const ts = r.tormentas || [];
  const fuertes = ts.filter(t => num(t.intensidad) >= 5).length;
  chip("chipTormentas", String(ts.length), fuertes ? `${fuertes} tormenta${fuertes > 1 ? "s" : ""} fuerte${fuertes > 1 ? "s" : ""}` : "en 240 km", fuertes ? "activo" : "ok");
  const ry = r.rayos || {};
  chip("chipRayos", ry.disponible ? String(ry.ultimos_10_min ?? 0) : "sin datos",
    ry.disponible ? `últimos 10 min · ${fmt(ry.ultima_hora)} en la hora` : "satélite sin datos nuevos",
    !ry.disponible ? "atraso" : ry.ultimos_10_min > 0 ? "activo" : "ok");
  const amb = r.ambiente || "—";
  chip("chipAmbiente", amb.replace(/^Ambiente\s+/i, "").replace(/\s+a tormentas$/i, ""), "para tormentas",
    /muy favorable|favorable/i.test(amb) && !/poco/i.test(amb) ? "activo" : "ok");

  const aviso = $("avisoAtraso");
  if (frame && edad !== null && edad > 15) {
    aviso.textContent = `El radar se publicó hace ${fmt(edad)} min. Mientras llega la imagen nueva, ClimaAR sigue estimando dónde están las tormentas con el satélite, los rayos y las estaciones.`;
    aviso.classList.remove("oculto");
  } else aviso.classList.add("oculto");

  const t = r.tiempo_actual || {};
  const dir = punto(t.viento_dir);
  $("ahoraCorto").textContent = t.disponible && num(t.temperatura) !== null
    ? `${fmt(t.temperatura)} °C · viento ${fmt(t.viento_vel)} km/h${dir ? " " + dir : ""}` : "";
}

function pintarLocalidades() {
  const lista = (estado.resumen && estado.resumen.localidades) || [];
  $("localidadesGrid").innerHTML = lista.map(l => {
    const n = num(l.nivel) || 0, min = num(l.minutos_hasta_llegada);
    const sub = min !== null ? `llega en ~${fmt(min)} min` : l.sobre_la_localidad ? "tormenta encima" : (l.resumen || l.resumen_ahora || (n ? l.etiqueta : "sin tormentas"));
    return `<button class="loc${l.id === estado.localidad ? " elegida" : ""}" data-id="${esc(l.id)}" data-n="${n}">
      <span class="loc-nivel">${n || "–"}</span><span class="loc-nombre">${esc(l.nombre)}</span>
      <span class="loc-sub${min !== null ? " llega" : ""}">${esc(sub)}</span></button>`;
  }).join("");
  $("localidadesGrid").querySelectorAll(".loc").forEach(b => b.addEventListener("click", () => {
    elegirLocalidad(b.dataset.id);
    $("foco").scrollIntoView({ behavior: "smooth", block: "start" });
  }));
}

const INTENSIDAD_TXT = { DEBIL: "Lluvia débil", MODERADO: "Lluvia moderada", FUERTE: "Tormenta fuerte", MUY_FUERTE: "Tormenta muy fuerte" };
const TENDENCIA_TXT = { FORTALECIENDOSE: "creciendo", CRECIENDO: "creciendo", DEBILITANDOSE: "debilitándose", ESTABLE: "estable" };
function nivelIntensidad(t) {
  const i = num(t.intensidad) || 0;
  return i >= 6 ? 4 : i >= 5 ? 3 : i >= 3 ? 2 : 1;   // escala del radar 1-6 → 1-4
}

function pintarListaTormentas() {
  const ts = (estado.resumen && estado.resumen.tormentas) || [];
  const loc = localidadActual();
  $("cuentaTormentas").textContent = ts.length ? `${ts.length} en 240 km` : "";
  if (!ts.length) {
    $("listaTormentas").innerHTML = '<li class="vacia">El radar no ve lluvia ni tormentas en los 240 km. ClimaAR sigue vigilando cada imagen nueva.</li>';
    return;
  }
  const p = loc ? [loc.lat, loc.lon] : RADAR_CENTRO;
  const orden = ts.map(t => ({ t, d: distKm(p, [t.lat, t.lon]) }))
    .sort((a, b) => (nivelIntensidad(b.t) >= 3) - (nivelIntensidad(a.t) >= 3) || a.d - b.d).slice(0, 6);
  $("listaTormentas").innerHTML = orden.map(({ t, d }) => {
    const n = nivelIntensidad(t);
    const donde = NOMBRE_PUNTO[punto(rumbo(p, [t.lat, t.lon]))] || "";
    const mov = t.direccion && num(t.velocidad_kmh) > 2
      ? `va hacia el ${NOMBRE_PUNTO[t.direccion] || t.direccion}` + (t.movimiento_dudoso ? " (velocidad incierta)" : ` a ${fmt(t.velocidad_kmh)} km/h`)
      : "casi quieta";
    const tend = TENDENCIA_TXT[t.tendencia_area];
    return `<li data-n="${n}"><span class="num">N.º ${esc(t.id)}</span>
      <div><b>${esc(INTENSIDAD_TXT[t.nivel_intensidad] || "Eco de lluvia")}</b>${num(t.tamano_km2) ? ` · ${esc(fmt(t.tamano_km2))} km²` : ""}${tend ? ` · ${esc(tend)}` : ""}
      <br><span class="suave">a ${esc(fmt(d))} km al ${esc(donde)} de ${esc(loc ? loc.nombre : "el radar")} · ${esc(mov)}</span></div></li>`;
  }).join("");
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
  if (c.t.direccion && num(c.t.velocidad_kmh) > 2) {
    txt += `, se mueve hacia el ${NOMBRE_PUNTO[c.t.direccion] || c.t.direccion}`;
    txt += c.t.movimiento_dudoso ? " (velocidad incierta)" : ` a ${fmt(c.t.velocidad_kmh)} km/h`;
  }
  if (c.t.tendencia_area === "DEBILITANDOSE") txt += " y se está debilitando";
  else if (c.t.tendencia_area === "FORTALECIENDOSE" || c.t.tendencia_area === "CRECIENDO") txt += " y está creciendo";
  return txt + ".";
}

/* ---------- Mapa ---------- */

const mapa = { m: null, radar: null, capas: null };
let animRadar = null, animSat = null;

// Bahía Blanca, White y Cerri están pegadas: sus nombres se muestran
// solo cuando están elegidas (si no, al tocar el círculo).
const OFFSET = { right: [4, 0], left: [-4, 0], top: [0, -4], bottom: [0, 4] };
const CERCANAS = new Set(["bahia_blanca", "ingeniero_white", "general_cerri"]);
const LADO = { bahia_blanca: "top", general_cerri: "left", ingeniero_white: "bottom", punta_alta: "right", medanos: "left", tornquist: "left", sierra_de_la_ventana: "right" };

const RADAR_CENTRO = [-38.7342, -62.1654];
const RADAR_KM = 240;

// Mapa base (cartografía, anillos y nombres según el acercamiento). Lo usan
// el radar de la vista General y el satélite de la vista Meteorólogos.
function crearMapaBase(id, fuente) {
  const m = L.map(id, { zoomControl: true, attributionControl: true, zoomSnap: 0.25, minZoom: 6, maxZoom: 12 });
  m.fitBounds(cuadroRadar(), { padding: [0, 0] });
  m.attributionControl.addAttribution("Cartografía: Natural Earth, IGN · " + fuente);
  // Capas: tierra abajo, la imagen en el medio, y rutas, límites, ríos y
  // nombres arriba (finos) para ubicar lo que se ve sin taparlo.
  [["tierra", 200], ["encima", 450]].forEach(([n, z]) => {
    m.createPane(n); m.getPane(n).style.zIndex = z; m.getPane(n).style.pointerEvents = "none";
  });
  // Detalle de calles solo al acercarse mucho (zoom 10 o más).
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    minZoom: 10, maxZoom: 12, subdomains: "abcd", pane: "tierra",
    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> · © <a href="https://carto.com/attributions">CARTO</a>'
  }).addTo(m);
  cargarCartografia(m);
  anillos(m);
  const zoomInicial = m.getZoom();
  const zoomClase = () => {
    const z = m.getZoom(), c = m.getContainer().classList;
    c.toggle("lejos", z < zoomInicial + 0.75);
    c.toggle("medio", z < zoomInicial + 1.75);   // White y Cerri, pegadas a Bahía
  };
  m.on("zoomend", zoomClase); zoomClase();
  return m;
}

function iniciarMapa() {
  if (mapa.m || !window.L) return;
  mapa.m = crearMapaBase("mapa", "Radar: SINARAME RMA10");
  mapa.capas = L.layerGroup().addTo(mapa.m);
}

function cuadroRadar() {
  const dLat = RADAR_KM / 110.574, dLon = RADAR_KM / (111.32 * Math.cos(RADAR_CENTRO[0] * Math.PI / 180));
  return [[RADAR_CENTRO[0] - dLat, RADAR_CENTRO[1] - dLon], [RADAR_CENTRO[0] + dLat, RADAR_CENTRO[1] + dLon]];
}

// Anillos de distancia desde el radar, hasta su alcance (240 km).
function anillos(m) {
  [50, 100, 150, 200, 240].forEach(km => {
    L.circle(RADAR_CENTRO, { radius: km * 1000, pane: "encima", color: km === RADAR_KM ? "#5fd0e0" : "#9fb1cc", weight: km === RADAR_KM ? 1.5 : 1, opacity: km === RADAR_KM ? .7 : .35, fill: false, dashArray: km === RADAR_KM ? null : "2 6", interactive: false }).addTo(m);
    // Etiqueta al noroeste, lejos de las ciudades y del centro.
    const ang = -40 * Math.PI / 180, dLat = km * Math.cos(ang) / 110.574, dLon = km * Math.sin(ang) / (111.32 * Math.cos(RADAR_CENTRO[0] * Math.PI / 180));
    L.marker([RADAR_CENTRO[0] + dLat, RADAR_CENTRO[1] + dLon], { pane: "encima", interactive: false, icon: L.divIcon({ className: "anillo-km", html: km + " km", iconSize: [48, 14] }) }).addTo(m);
  });
  L.circleMarker(RADAR_CENTRO, { pane: "encima", radius: 3, color: "#5fd0e0", fillOpacity: 1, interactive: false }).addTo(m);
}

const ESTILO_CARTO = {
  tierra: { pane: "tierra", stroke: true, color: "#34507a", weight: 1, fillColor: "#16233a", fillOpacity: 1 },
  lagos: { pane: "tierra", stroke: false, fillColor: "#0b1524", fillOpacity: 1 },
  rios: { pane: "encima", color: "#3f7fbf", weight: 1.2, opacity: .7 },
  provincias: { pane: "encima", color: "#c9d6ea", weight: 1.2, opacity: .55, dashArray: "6 4" },
  rutas: { pane: "encima", color: "#e2b86b", weight: 1, opacity: .45 },
  partidos: { pane: "encima", color: "#8fa6c8", weight: .8, opacity: .5 }
};

let cartografia = null;
async function cargarCartografia(m) {
  try {
    if (!cartografia) cartografia = fetch("cartografia.json").then(r => r.json());
    const g = await cartografia;
    // Las localidades de ClimaAR se dibujan aparte (con su nombre).
    const ya = new Set(["Bahía Blanca", "Punta Alta", "Tornquist", "Médanos", "Ingeniero White", "General Daniel Cerri", "Sierra de la Ventana"]);
    L.geoJSON(g, {
      filter: f => f.properties.capa !== "ciudades" || !ya.has(f.properties.nombre),
      onEachFeature: (f, capa) => {
        if (f.properties.capa === "nombre_partido" && !ya.has(f.properties.nombre)) capa.bindTooltip(f.properties.nombre, { permanent: true, direction: "center", className: "etiqueta-partido" });
      },
      style: f => ESTILO_CARTO[f.properties.capa] || {},
      pointToLayer: (f, ll) => f.properties.capa === "nombre_partido"
        ? L.circleMarker(ll, { pane: "encima", radius: 0, opacity: 0, fillOpacity: 0, interactive: false })
        : L.circleMarker(ll, { pane: "encima", radius: 2.5, color: "#c9d6ea", weight: 1, fillOpacity: 1, interactive: false })
        .bindTooltip(f.properties.nombre, { permanent: true, direction: "right", className: "etiqueta-ciudad" + ((f.properties.rango ?? 9) > 8 ? " menor" : ""), offset: [3, 0] }),
      interactive: false
    }).addTo(m);
  } catch (e) { /* sin cartografía: el mapa sigue funcionando */ }
}

function limitesRadar() {
  const r = estado.resumen && estado.resumen.radar;
  if (r && Array.isArray(r.bounds)) return r.bounds;
  return cuadroRadar();
}

function urlImagen(ruta, clave) {
  return API + ruta + (ruta.includes("?") ? "&" : "?") + "f=" + encodeURIComponent(clave || "");
}

function ponerImagenRadar(ruta, clave) {
  if (!mapa.m) return;
  const url = urlImagen(ruta, clave);
  if (mapa.radar) mapa.radar.setUrl(url);
  else mapa.radar = L.imageOverlay(url, limitesRadar(), { opacity: .9 }).addTo(mapa.m);
}

function pintarMapa() {
  iniciarMapa();
  if (!mapa.m) return;
  const r = estado.resumen || {};
  const frame = r.frame_actual_utc || "";
  if (frame !== estado.frameMostrado && !(animRadar && animRadar.activa())) {
    ponerImagenRadar((r.radar && r.radar.imagen) || "/radar/mapa", frame);
    estado.frameMostrado = frame;
  }
  mapa.capas.clearLayers();

  // Solo el nombre de cada localidad, sin círculos ni trazos: la lluvia
  // se ve sola en el radar.
  (r.localidades || []).forEach(l => {
    if (num(l.lat) === null) return;
    const sel = l.id === estado.localidad;
    L.circleMarker([l.lat, l.lon], { pane: "encima", radius: sel ? 4 : 3, color: "#fff", weight: 1, fillColor: sel ? "#5fd0e0" : "#fff", fillOpacity: 1 })
      .bindTooltip(l.nombre, { permanent: true, direction: LADO[l.id] || "right", className: "etiqueta-ciudad localidad" + (sel ? " elegida" : "") + (CERCANAS.has(l.id) && !sel ? " menor" : ""), offset: OFFSET[LADO[l.id] || "right"] })
      .on("click", () => elegirLocalidad(l.id))
      .addTo(mapa.capas);
  });

  if (animRadar && animRadar.activa()) return;
  const edad = num(r.frame_edad_minutos);
  $("radarHora").textContent = frame ? `Imagen del radar de las ${horaLocal(frame)}` + (edad !== null ? ` (hace ${fmt(edad)} min)` : "") + (r.radar_retrasado ? " · el radar viene atrasado" : "") : "Sin imagen del radar.";
}

// Animación de imágenes con botón "Último" para volver al instante más nuevo
// sin esperar a que termine la vuelta.
function crearAnimador({ boton, botonUltimo, textoBoton, cargar, poner, rotulo, alVolver }) {
  let reloj = null;
  const volver = () => {
    if (reloj) clearInterval(reloj);
    reloj = null;
    boton.textContent = textoBoton;
    alVolver();
  };
  boton.addEventListener("click", async () => {
    if (reloj) { volver(); return; }
    boton.textContent = "Cargando…";
    try {
      const frames = await cargar();
      if (!frames.length) throw new Error("sin imágenes");
      frames.forEach(f => { const i = new Image(); i.src = f.url; });
      let i = 0;
      boton.textContent = "Pausa";
      reloj = setInterval(() => {
        const f = frames[i % frames.length];
        poner(f); rotulo("Animación · " + horaLocal(f.hora));
        i++;
      }, 700);
    } catch (e) {
      boton.textContent = textoBoton;
      rotulo("No se pudo cargar la animación.");
    }
  });
  botonUltimo.addEventListener("click", volver);
  return { activa: () => reloj !== null };
}

function isoDeClave(t) {
  t = String(t);
  return `${t.slice(0,4)}-${t.slice(4,6)}-${t.slice(6,8)}T${t.slice(8,10)}:${t.slice(10,12)}:00Z`;
}

/* ---------- Satélite GOES-19 (vista Meteorólogos) ---------- */

const sat = { m: null, nubes: null, rayos: null, radar: null, mostrada: null, datosRayos: [], finRayos: null, ultimaImagen: null };

function iniciarSatelite() {
  if (sat.m || !window.L) return;
  sat.m = crearMapaBase("mapaSat", "Satélite: NOAA GOES-19");
  sat.rayos = L.layerGroup().addTo(sat.m);
  sat.m.on("move", medirMira);
}

function ponerNubes(ruta, clave) {
  if (!sat.m) return;
  const url = urlImagen(ruta, clave);
  if (sat.nubes) sat.nubes.setUrl(url);
  else sat.nubes = L.imageOverlay(url, cuadroRadar(), { opacity: .95 }).addTo(sat.m);
}

// Rayos como en Windy: un rayito chico por cada rayo (la cantidad se ve
// sola por cuántos hay juntos). Se muestran los 10 min que terminan en la
// imagen que se está viendo, así acompañan la animación. Destellan los de
// los últimos 5 min de esa imagen; los de los últimos 2 min, con brillo.
const RAYO_SVG = '<svg viewBox="0 0 16 22" aria-hidden="true"><path d="M10 0 2 12h5l-2 10 9-13H9l3-9z"/></svg>';
const RAYOS_VENTANA_S = 600;
const MIRA_KM = 15;

function rayosHasta(finSeg) {
  return (sat.datosRayos || []).filter(r => r[2] <= finSeg && r[2] > finSeg - RAYOS_VENTANA_S);
}

function pintarRayos(finSeg) {
  sat.finRayos = finSeg;
  sat.rayos.clearLayers();
  const lista = rayosHasta(finSeg).sort((a, b) => a[2] - b[2]);   // los nuevos quedan arriba
  lista.forEach(p => {
    const edad = (finSeg - p[2]) / 60;
    const clase = edad <= 2 ? "rayo nuevo brillo" : edad <= 5 ? "rayo nuevo" : "rayo";
    L.marker([p[0], p[1]], {
      pane: "encima", interactive: false, keyboard: false,
      icon: L.divIcon({ className: "icono-rayo", html: `<span class="${clase}" style="animation-delay:-${(Math.random() * 1.6).toFixed(2)}s">${RAYO_SVG}</span>`, iconSize: [10, 14], iconAnchor: [5, 7] })
    }).addTo(sat.rayos);
  });
  medirMira();
}

// Mira fija en el centro del mapa: cuántos rayos hay alrededor (15 km) y su
// densidad, en los 10 min de la imagen que se está viendo.
function medirMira() {
  if (!sat.m || sat.finRayos == null) return;
  const c = sat.m.getCenter(), centro = [c.lat, c.lng];
  const n = rayosHasta(sat.finRayos).filter(r => distKm(centro, [r[0], r[1]]) <= MIRA_KM).length;
  const area = Math.PI * MIRA_KM * MIRA_KM;
  $("miraDato").textContent = n
    ? `${n} rayo${n > 1 ? "s" : ""} a ${MIRA_KM} km · ${fmt(n / area * 100, 1)} por 100 km² (10 min)`
    : `Sin rayos a ${MIRA_KM} km (10 min)`;
}

function radarSobreSatelite() {
  if (!sat.m) return;
  if ($("radarEncima").checked) {
    const url = urlImagen((estado.resumen && estado.resumen.radar && estado.resumen.radar.imagen) || "/radar/mapa", estado.frameMostrado || "");
    if (sat.radar) sat.radar.setUrl(url);
    else sat.radar = L.imageOverlay(url, limitesRadar(), { opacity: .85 }).addTo(sat.m);
  } else if (sat.radar) {
    sat.m.removeLayer(sat.radar); sat.radar = null;
  }
}

async function cargarSatelite() {
  iniciarSatelite();
  if (!sat.m) return;
  try {
    const s = await pedir("/satelite", 30000);
    const lista = s.imagenes || [];
    const ultima = lista[lista.length - 1];
    if (ultima && ultima.clave !== sat.mostrada && !(animSat && animSat.activa())) {
      ponerNubes(ultima.imagen, ultima.clave);
      sat.mostrada = ultima.clave;
    }
    sat.datosRayos = ((s.rayos && s.rayos.lista) || []).filter(r => Array.isArray(r) && num(r[0]) !== null && num(r[2]) !== null);
    sat.ultimaImagen = ultima || null;
    radarSobreSatelite();
    if (animSat && animSat.activa()) return;
    // Vista "Último": los rayos más nuevos que haya (aunque el satélite tarde).
    const masNuevo = sat.datosRayos.reduce((m, r) => Math.max(m, r[2]), 0);
    const finImagen = ultima ? Date.parse(ultima.hora_utc) / 1000 : 0;
    pintarRayos(Math.max(masNuevo, finImagen) || Date.now() / 1000);
    const n = rayosHasta(sat.finRayos).length;
    $("satHora").textContent = ultima
      ? `Imagen del satélite de las ${horaLocal(ultima.hora_utc)} · ${n ? n + " rayos en los últimos 10 min" : "sin rayos en los últimos 10 min"}`
      : "El satélite no está disponible en este momento.";
  } catch (e) {
    $("satHora").textContent = "El satélite no está disponible en este momento.";
  }
}

/* ---------- Pronóstico ---------- */

function pintarDias() {
  const p = estado.pronostico;
  const dias = (p && p.dias) || [];
  if (!dias.length) { $("dias").innerHTML = '<p class="pie" style="grid-column:1/-1">El pronóstico no está disponible en este momento.</p>'; return; }
  $("dias").innerHTML = dias.slice(0, 3).map(d => {
    const fecha = new Date(d.fecha + "T12:00:00-03:00");
    const nombre = isNaN(fecha) ? d.fecha : fecha.toLocaleDateString("es-AR", { weekday: "long", day: "numeric", timeZone: "America/Argentina/Buenos_Aires" });
    const [texto, tipo] = cielo(d.codigo_tiempo, null, null);
    const dir = punto(d.viento_direccion);
    return `<div class="dia"><div class="fecha">${esc(nombre)}</div>${icono(tipo, true)}
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
  cargarSatelite();
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
    llenarSelector(); pintarVivo(); pintarTormenta(); pintarLocalidades(); pintarMapa(); pintarListaTormentas(); pintarTiempo();
  } catch (e) {
    $("cielo").textContent = "Sin conexión con el servidor. Se reintenta solo.";
  }
}
async function actualizarPronostico() {
  try { estado.pronostico = await pedir("/pronostico"); } catch (e) { estado.pronostico = null; }
  pintarDias(); pintarTormenta();
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
  pintarTormenta(); pintarLocalidades(); pintarMapa(); pintarListaTormentas();
}

function cambiarVista(v) {
  estado.vista = v;
  $("general").classList.toggle("oculto", v !== "general");
  $("meteo").classList.toggle("oculto", v !== "meteo");
  $("tabGeneral").setAttribute("aria-selected", String(v === "general"));
  $("tabMeteo").setAttribute("aria-selected", String(v === "meteo"));
  if (v === "meteo") { cargarMeteo(); if (sat.m) setTimeout(() => sat.m.invalidateSize(), 50); }
  else { actualizarGeneral(); if (mapa.m) setTimeout(() => mapa.m.invalidateSize(), 50); }
  programar();
}

$("localidad").addEventListener("change", e => elegirLocalidad(e.target.value));
document.querySelectorAll(".vistas button").forEach(b => b.addEventListener("click", () => cambiarVista(b.dataset.vista)));
animRadar = crearAnimador({
  boton: $("animar"), botonUltimo: $("ultimo"), textoBoton: "Animar",
  cargar: async () => {
    const s = await pedir("/secuencia");
    // Misma imagen limpia que el mapa, no la cruda del radar.
    return (s.frames || []).filter(f => f.timestamp).map(f => ({
      ruta: "/radar/mapa/" + f.timestamp, clave: f.timestamp, url: urlImagen("/radar/mapa/" + f.timestamp, f.timestamp), hora: isoDeClave(f.timestamp)
    }));
  },
  poner: f => ponerImagenRadar(f.ruta, f.clave),
  rotulo: t => { $("radarHora").textContent = t; },
  alVolver: () => { estado.frameMostrado = null; pintarMapa(); }
});
animSat = crearAnimador({
  boton: $("animarSat"), botonUltimo: $("ultimoSat"), textoBoton: "Animar",
  cargar: async () => {
    const s = await pedir("/satelite", 30000);
    return (s.imagenes || []).map(f => ({ ruta: f.imagen, clave: f.clave, url: urlImagen(f.imagen, f.clave), hora: f.hora_utc }));
  },
  poner: f => { ponerNubes(f.ruta, f.clave); pintarRayos(Date.parse(f.hora) / 1000); },
  rotulo: t => { $("satHora").textContent = t; },
  alVolver: () => { sat.mostrada = null; cargarSatelite(); }
});
$("radarEncima").addEventListener("change", radarSobreSatelite);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) { if (estado.vista === "general") actualizarGeneral(); else cargarMeteo(); }
  programar();
});

actualizarGeneral();
actualizarPronostico();
programar();
}
