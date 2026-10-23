"use strict";

/* ========================================================= */
/* CONFIG */
/* ========================================================= */

const API_BASE_URL =
  typeof CLIMAAR_CONFIG !== "undefined" &&
  typeof CLIMAAR_CONFIG.API_BASE_URL === "string"
    ? CLIMAAR_CONFIG.API_BASE_URL.replace(/\/+$/, "")
    : "";

const state = {
  radar: null,
  health: null,
  weather: null,
  livePrediction: null,
  storm: null,
  intelligence: null,
  model: null,
  forecast: null,
  sequence: [],

  radarZoom: 1,
  radarPanX: 0,
  radarPanY: 0,

  playing: false,
  playTimer: null,
  playIndex: 0,

  dragging: false,
  dragStartX: 0,
  dragStartY: 0,
  dragPanX: 0,
  dragPanY: 0
};


/* ========================================================= */
/* DOM */
/* ========================================================= */

const $ = (selector) =>
  document.querySelector(selector);

const $$ = (selector) =>
  Array.from(document.querySelectorAll(selector));


/* ========================================================= */
/* API */
/* ========================================================= */

function apiUrl(path) {
  if (!path) {
    return API_BASE_URL;
  }

  if (/^https?:\/\//i.test(path)) {
    return path;
  }

  if (!API_BASE_URL) {
    return path;
  }

  return `${API_BASE_URL}${path}`;
}


async function getJSON(path, timeout = 15000) {
  const controller = new AbortController();

  const timer = setTimeout(() => {
    controller.abort();
  }, timeout);

  try {
    const response = await fetch(
      apiUrl(path),
      {
        method: "GET",
        cache: "no-store",
        headers: {
          Accept: "application/json"
        },
        signal: controller.signal
      }
    );

    if (!response.ok) {
      throw new Error(
        `${path} → HTTP ${response.status}`
      );
    }

    return await response.json();

  } finally {
    clearTimeout(timer);
  }
}


/* ========================================================= */
/* UTILIDADES */
/* ========================================================= */

function number(value) {
  if (
    value === null ||
    value === undefined ||
    value === ""
  ) {
    return null;
  }

  const n = Number(value);

  return Number.isFinite(n)
    ? n
    : null;
}


function text(value, fallback = "—") {
  if (
    value === null ||
    value === undefined ||
    value === ""
  ) {
    return fallback;
  }

  return String(value);
}


function firstNumber(...values) {
  for (const value of values) {
    const n = number(value);

    if (n !== null) {
      return n;
    }
  }

  return null;
}


function firstValue(...values) {
  for (const value of values) {
    if (
      value !== null &&
      value !== undefined &&
      value !== ""
    ) {
      return value;
    }
  }

  return null;
}


function formatNumber(
  value,
  decimals = 0
) {
  const n = number(value);

  if (n === null) {
    return "—";
  }

  return n.toFixed(decimals);
}


function formatPercent(value) {
  const n = number(value);

  if (n === null) {
    return "—";
  }

  return `${n.toFixed(1)}%`;
}


function formatAge(value) {
  const n = number(value);

  if (n === null) {
    return "—";
  }

  if (n < 1) {
    return "menos de 1 min";
  }

  return `${Math.round(n)} min`;
}


function formatTimestamp(value) {
  if (!value) {
    return "—";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return String(value);
  }

  return date.toLocaleString(
    "es-AR",
    {
      day: "2-digit",
      month: "2-digit",
      hour: "2-digit",
      minute: "2-digit"
    }
  );
}


function setText(
  selector,
  value
) {
  const element = $(selector);

  if (element) {
    element.textContent = text(value);
  }
}


function setClass(
  selector,
  ...classes
) {
  const element = $(selector);

  if (!element) {
    return;
  }

  element.className = classes.join(" ");
}


/* ========================================================= */
/* STATUS RADAR */
/* ========================================================= */

function radarStatusInfo(data) {
  const age = number(
    data?.frame_edad_minutos
  );

  const status =
    data?.estado_radar ||
    "";

  if (
    data?.radar_actualizado === true ||
    status === "actualizado"
  ) {
    return {
      label: "EN VIVO",
      className: "statusLive"
    };
  }

  if (
    data?.radar_retrasado === true ||
    status === "retrasado" ||
    (age !== null && age > 45)
  ) {
    return {
      label: "RETRASADO",
      className: "statusDelayed"
    };
  }

  if (
    data?.radar_disponible === false ||
    status === "no_disponible"
  ) {
    return {
      label: "SIN DATOS",
      className: "statusOffline"
    };
  }

  return {
    label: "SIN DATOS",
    className: "statusUnknown"
  };
}


/* ========================================================= */
/* RADAR */
/* ========================================================= */

function radarImageUrl(path = "/radar/imagen") {
  const base = apiUrl(path);

  return `${base}${
    base.includes("?")
      ? "&"
      : "?"
  }t=${Date.now()}`;
}


function applyRadarTransform() {
  const image = $("#radarImage");

  if (!image) {
    return;
  }

  image.style.setProperty(
    "--radar-scale",
    state.radarZoom
  );

  image.style.setProperty(
    "--pan-x",
    `${state.radarPanX}px`
  );

  image.style.setProperty(
    "--pan-y",
    `${state.radarPanY}px`
  );

  setText(
    "#zoomValue",
    `${state.radarZoom.toFixed(1)}×`
  );
}


function resetRadarView() {
  state.radarZoom = 1;
  state.radarPanX = 0;
  state.radarPanY = 0;

  applyRadarTransform();
}


function changeRadarZoom(delta) {
  state.radarZoom = Math.max(
    1,
    Math.min(
      3,
      state.radarZoom + delta
    )
  );

  const maxPan =
    (state.radarZoom - 1) * 120;

  state.radarPanX = Math.max(
    -maxPan,
    Math.min(
      maxPan,
      state.radarPanX
    )
  );

  state.radarPanY = Math.max(
    -maxPan,
    Math.min(
      maxPan,
      state.radarPanY
    )
  );

  applyRadarTransform();
}


function showRadarImage(path) {
  const image = $("#radarImage");
  const loading = $("#radarLoading");

  if (!image) {
    return;
  }

  if (loading) {
    loading.classList.remove("hidden");
  }

  image.onload = () => {
    if (loading) {
      loading.classList.add("hidden");
    }
  };

  image.onerror = () => {
    if (loading) {
      loading.textContent =
        "Radar no disponible";
      loading.classList.remove("hidden");
    }
  };

  image.src = radarImageUrl(path);
}


function renderRadar(data) {
  state.radar = data || {};

  const status =
    radarStatusInfo(state.radar);

  const radarStatus =
    $("#radarStatus");

  if (radarStatus) {
    radarStatus.textContent =
      status.label;

    radarStatus.className =
      `statusBadge ${status.className}`;
  }

  const globalStatus =
    $("#globalStatus");

  if (globalStatus) {
    globalStatus.textContent =
      status.label;

    globalStatus.className =
      `statusBadge ${status.className}`;
  }

  const age = number(
    state.radar.frame_edad_minutos
  );

  setText(
    "#radarAge",
    age !== null
      ? formatAge(age)
      : "—"
  );

  setText(
    "#radarTimestamp",
    formatTimestamp(
      state.radar.frame_actual_utc
    )
  );

  setText(
    "#radarFrame",
    state.radar.frame_actual_utc
      || "—"
  );

  setText(
    "#radarHistory",
    state.radar.frames_historicos !==
      undefined
      ? `${state.radar.frames_historicos} frames`
      : "—"
  );

  setText(
    "#summaryRadar",
    status.label
  );

  setText(
    "#summaryRadarAge",
    age !== null
      ? `hace ${formatAge(age)}`
      : "sin antigüedad"
  );

  showRadarImage(
    state.radar.imagen_local ||
    "/radar/imagen"
  );

  applyRadarTransform();
}


/* ========================================================= */
/* WEATHER */
/* ========================================================= */

function renderWeather(data) {
  state.weather = data || {};

  const temperature =
    number(state.weather.temperatura);

  setText(
    "#temp",
    temperature !== null
      ? `${Math.round(temperature)}°`
      : "—"
  );

  const skyNow = skyDescription();

  setText(
    "#condition",
    skyNow !== "—"
      ? skyNow
      : "Sin datos suficientes"
  );

  const secondary = [];

  const dew =
    number(state.weather.punto_rocio);

  if (dew !== null) {
    secondary.push(
      `Punto de rocío ${dew.toFixed(1)}°`
    );
  }

  const direction =
    state.weather.viento_direccion;

  if (
    direction !== null &&
    direction !== undefined &&
    direction !== ""
  ) {
    secondary.push(
      `Dirección ${direction}°`
    );
  }

  setText(
    "#weatherSecondary",
    secondary.length
      ? secondary.join(" · ")
      : "Sin datos suficientes"
  );

  const wind =
    number(state.weather.viento);

  setText(
    "#wind",
    wind !== null
      ? `${Math.round(wind)} km/h`
      : "—"
  );

  const pressure =
    number(state.weather.presion);

  setText(
    "#pressure",
    pressure !== null
      ? `${Math.round(pressure)} hPa`
      : "—"
  );

  const rain =
    number(state.weather.precipitacion);

  setText(
    "#rain",
    rain !== null
      ? `${rain} mm`
      : "—"
  );

  renderConditionCards();
}


/* ========================================================= */
/* CONDICIONES ACTUALES (tarjetas) */
/* ========================================================= */

const CARDINALS = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"];

function degreesToCardinal(value) {
  const deg = number(value);

  if (deg === null) {
    return "—";
  }

  const normalized = ((deg % 360) + 360) % 360;

  return CARDINALS[Math.round(normalized / 45) % 8];
}


function skyDescription() {
  const w = state.weather || {};
  const direct = firstValue(w.cielo, w.estado_cielo);

  if (typeof direct === "string") {
    return direct;
  }

  const raw = getIntelligenceClassification();

  if (raw && raw !== "sin_tormenta") {
    return "Tormenta";
  }

  const fromCode = weatherCodeText(w.codigo_tiempo);

  if (fromCode) {
    return fromCode;
  }

  const rain = number(w.precipitacion);

  if (rain !== null && rain > 0) {
    return "Lluvia";
  }

  return "—";
}


// Códigos WMO de Open-Meteo traducidos a lenguaje simple.
function weatherCodeText(code) {
  const c = number(code);

  if (c === null) return null;
  if (c === 0) return "Despejado";
  if (c === 1) return "Mayormente despejado";
  if (c === 2) return "Parcialmente nublado";
  if (c === 3) return "Nublado";
  if (c === 45 || c === 48) return "Niebla";
  if (c >= 51 && c <= 57) return "Llovizna";
  if (c >= 61 && c <= 67) return "Lluvia";
  if (c >= 71 && c <= 77) return "Nieve";
  if (c >= 80 && c <= 82) return "Chaparrones";
  if (c === 85 || c === 86) return "Nevadas";
  if (c >= 95) return "Tormenta";

  return null;
}


function renderConditionCards() {
  const w = state.weather || {};
  const unit = (value, suffix, digits = 0) =>
    value !== null
      ? `${value.toFixed(digits)}${suffix}`
      : "—";

  setText("#cardTemp", unit(number(w.temperatura), "°C", 1));
  setText("#cardFeels", unit(firstNumber(w.sensacion_termica), "°C", 1));
  setText("#windDir", degreesToCardinal(w.viento_direccion));
  setText("#gusts", unit(firstNumber(w.rafagas, w.rafaga), " km/h"));
  setText("#humidity", unit(firstNumber(w.humedad), "%"));
  setText("#sky", skyDescription());
}


/* ========================================================= */
/* ALERTA PRINCIPAL */
/* ========================================================= */

function intensityLabel(level) {
  const n = number(level);

  // Intensidad del eco radar (RMA10 0-6): no es la clasificación
  // meteorológica ni la severidad de la tormenta.
  if (n === null) return "Sin datos";
  if (n <= 0) return "Sin eco";
  return `Eco radar nivel ${n}/6`;
}


const CLASSIFICATION_LABELS = {
  sin_tormenta: "Sin tormenta para Bahía Blanca",
  tormenta: "Tormenta",
  tormenta_fuerte: "Tormenta fuerte",
  tormenta_severa: "Tormenta severa"
};


function getIntelligenceClassification() {
  const intel = state.intelligence || {};
  const raw =
    intel.clasificacion_meteorologica ||
    intel?.resultado_final?.clasificacion_meteorologica ||
    intel?.resultado_final?.motor_decision?.clasificacion_meteorologica ||
    intel?.inteligencia_tormenta?.clasificacion_meteorologica;

  return CLASSIFICATION_LABELS[raw] ? raw : null;
}


function getIntelligenceSeverity() {
  const intel = state.intelligence || {};
  const final = intel.resultado_final || {};
  const raw =
    intel.severidad ||
    intel.severity ||
    final?.severidad_meteorologica?.valor ||
    final?.motor_decision?.severidad_meteorologica ||
    intel?.inteligencia_tormenta?.severidad ||
    intel?.motor_decision?.severidad_meteorologica ||
    intel?.severidad_meteorologica?.valor;

  if (raw === null || raw === undefined) {
    return null;
  }

  const value = String(raw).trim().toLowerCase() === "roja"
    ? "rojo"
    : String(raw).trim().toLowerCase();
  return ["verde", "amarillo", "naranja", "rojo"].includes(value)
    ? value
    : null;
}

function severityLabel(value) {
  const labels = {
    verde: "Sin severidad determinada",
    amarillo: "Severidad meteorológica: Amarillo",
    naranja: "Severidad meteorológica: Naranja",
    rojo: "Severidad meteorológica: Rojo"
  };

  return labels[value] || "Severidad meteorológica: No determinada";
}

function severityClass(value) {
  if (value === "rojo") return "riskHigh";
  if (value === "naranja") return "riskMedium";
  if (value === "amarillo") return "riskMedium";
  return "riskLow";
}

function principalTrackId() {
  const intel = state.intelligence || {};
  const radar = state.radar || {};

  const candidates = [
    radar.nucleo_principal_id,
    radar?.bahia_blanca?.nucleo_principal_id,
    intel?.principal?.track_id,
    intel?.resultado_final?.principal?.track_id,
    state.model?.principal_track_id
  ];

  for (const value of candidates) {
    if (value !== null && value !== undefined && value !== "") {
      return value;
    }
  }

  return null;
}


function principalNucleus() {
  const radar = state.radar || {};
  const principalId = principalTrackId();

  const nuclei = Array.isArray(radar.nucleos)
    ? radar.nucleos
    : [];

  if (principalId === null || principalId === undefined) {
    return null;
  }

  return nuclei.find(
    item => String(item?.track_id) === String(principalId)
  ) || null;
}

function classificationText() {
  const value = getIntelligenceClassification();
  return value
    ? CLASSIFICATION_LABELS[value]
    : "Clasificación no determinada";
}


function renderAlert() {
  const level =
    number(
      state.radar?.intensidad_nivel_max
    );

  const cores =
    Array.isArray(
      state.radar?.nucleos
    )
      ? state.radar.nucleos.length
      : 0;

  const activity =
    cores > 0 ||
    state.weather?.cobertura_radar === true;

  if (!activity) {
    setText(
      "#alertTitle",
      "Sin actividad detectada"
    );

    setText(
      "#alertDescription",
      "No hay núcleos identificados en el frame disponible."
    );

    setText(
      "#alertEyebrow",
      "ESTADO CLIMAAR"
    );

    return;
  }

  setText(
    "#alertTitle",
    "Actividad detectada en radar"
  );

  const severity = getIntelligenceSeverity();

  setText(
    "#alertDescription",
    `${cores} núcleo${
      cores === 1
        ? ""
        : "s"
    } operativo${
      cores === 1
        ? ""
        : "s"
    } · ${intensityLabel(level)} · ${classificationText()} · ${severityLabel(severity)}.`
  );

  setText(
    "#alertEyebrow",
    "SEGUIMIENTO ACTIVO"
  );
}


/* ========================================================= */
/* STORM / NOWCAST */
/* ========================================================= */

function renderStorm() {
  const radar = state.radar || {};
  const cores = Array.isArray(radar.nucleos)
    ? radar.nucleos
    : [];
  const principal = principalNucleus();
  const level = number(radar.intensidad_nivel_max);
  const radarLabel = radar.intensidad || intensityLabel(level);
  const severity = getIntelligenceSeverity();

  setText(
    "#stormTitle",
    principal
      ? `Núcleo principal · Track ${principal.track_id ?? "—"}`
      : "Actividad detectada"
  );

  setText(
    "#stormRisk",
    `${classificationText()} · ${severityLabel(severity)}`
  );

  const risk = $("#stormRisk");
  if (risk) {
    risk.className = "riskBadge";
    risk.classList.add(severityClass(severity));
  }

  setText(
    "#stormIntensity",
    level !== null ? String(level) : "—"
  );

  setText(
    "#stormIntensityLevel",
    `Radar: ${radarLabel}`
  );

  setText(
    "#stormCores",
    String(cores.length)
  );

  const movement = principal?.movimiento || {};
  const direction = number(
    movement.direccion_grados ?? radar.direccion_grados
  );

  setText(
    "#stormMovement",
    radar.movimiento_hacia_bahia === true
      ? "Acercándose a Bahía Blanca"
      : radar.movimiento_hacia_bahia === false
        ? "No se dirige hacia Bahía Blanca"
        : "Sin movimiento hacia Bahía Blanca confirmado"
  );

  const distance = number(
    radar.distancia_km ??
    radar?.bahia_blanca?.distancia_principal_km
  );

  setText(
    "#stormDistance",
    distance !== null
      ? `${distance.toFixed(1)} km`
      : "Sin datos"
  );

  const eta = number(
    radar.eta_minutos ??
    radar?.bahia_blanca?.eta_minutos
  );

  setText(
    "#stormEta",
    eta !== null
      ? `${Math.round(eta)} min`
      : "Sin ETA válida"
  );

  setText(
    "#stormDirection",
    direction !== null
      ? `${Math.round(direction)}°`
      : "Sin datos"
  );

  const trend = movement.tendencia || radar.fortalecimiento;
  const trendMap = {
    fortaleciendose: "El núcleo principal se está fortaleciendo",
    debilitandose: "El núcleo principal se está debilitando",
    estable: "La evolución del núcleo principal es estable",
    fortaleciendo: "El núcleo principal se está fortaleciendo",
    debilitando: "El núcleo principal se está debilitando"
  };

  setText(
    "#stormTrend",
    trendMap[trend] || "Evolución no determinada"
  );

  setText(
    "#stormTrendDetail",
    principal
      ? `Track ${principal.track_id ?? "—"} · ${cores.length} núcleos operativos en el frame.`
      : "Sin núcleo principal confirmado."
  );

  setText(
    "#summaryCores",
    cores.length
  );

  setText(
    "#summaryIntensity",
    level !== null
      ? `${level}/6 · ${radarLabel}`
      : "—"
  );
}


/* ========================================================= */
/* NUCLEOS */
/* ========================================================= */

function renderNuclei() {
  const list =
    $("#nucleiList");

  if (!list) {
    return;
  }

  const nuclei =
    Array.isArray(
      state.radar?.nucleos
    )
      ? state.radar.nucleos
      : [];

  setText(
    "#nucleiCount",
    nuclei.length
      ? `${nuclei.length}`
      : "0"
  );

  if (!nuclei.length) {
    list.innerHTML = `
      <div class="emptyState">
        No hay núcleos disponibles en el frame actual.
      </div>
    `;

    return;
  }

  const sorted =
    [...nuclei]
      .sort(
        (a,b) =>
          (number(b?.intensidad_max) || 0) -
          (number(a?.intensidad_max) || 0)
      )
      .slice(0, 8);

  list.innerHTML =
    sorted
      .map(
        (core, index) => {

          const intensity =
            firstNumber(
              core.intensidad_max,
              core.intensidad
            );

          const level =
            core.nivel_intensidad ||
            intensityLabel(
              intensity
            );

          const track =
            core.track_id !== undefined
              ? `Track ${core.track_id}`
              : "Track sin ID";

          const confidence =
            number(
              core.confianza
            );

          return `
            <div class="nucleusRow">

              <div class="nucleusIndex">
                ${index + 1}
              </div>

              <div class="nucleusInfo">

                <strong>
                  ${escapeHtml(level)}
                </strong>

                <span>
                  ${escapeHtml(track)}
                  ${
                    confidence !== null
                      ? ` · confianza track ${Math.round(confidence * 100)}%`
                      : ""
                  }
                </span>

              </div>

              <div class="nucleusIntensity">
                ${
                  intensity !== null
                    ? `${intensity}/6`
                    : "—"
                }
              </div>

            </div>
          `;
        }
      )
      .join("");
}


/* ========================================================= */
/* MODELO ML */
/* ========================================================= */

function getPredictions(
  model
) {
  if (
    !model ||
    !Array.isArray(
      model.predicciones
    )
  ) {
    return [];
  }

  return model.predicciones
    .filter(
      item =>
        item &&
        typeof item === "object"
    );
}


function findPrincipalTrack(predictions) {
  if (!predictions.length) {
    return null;
  }

  const principalId = principalTrackId();

  if (principalId !== null && principalId !== undefined) {
    return predictions.find(
      item => String(item?.track_id) === String(principalId)
    ) || null;
  }

  return predictions.length === 1 ? predictions[0] : null;
}


function renderModel() {
  const model =
    state.model || {};

  const predictions =
    getPredictions(model);

  const principal =
    findPrincipalTrack(
      predictions
    );

  setText(
    "#mlFrame",
    model.frame_radar ||
    "—"
  );

  if (!principal) {
    setText(
      "#predictionTrack",
      "Sin track disponible"
    );

    setText(
      "#predictionDistance",
      "—"
    );

    [
      "#pred5",
      "#pred10",
      "#pred20",
      "#pred30",
      "#strength5",
      "#trackBearing"
    ].forEach(
      selector =>
        setText(selector, "—")
    );

    setText(
      "#predictionMode",
      "No hay predicciones ML disponibles para mostrar."
    );

    return;
  }

  setText(
    "#predictionTrack",
    principal.track_id !== undefined
      ? `Track ${principal.track_id}`
      : "Track"
  );

  const distance =
    number(principal.distance_km);

  setText(
    "#predictionDistance",
    distance !== null
      ? `${distance.toFixed(1)} km`
      : "—"
  );

  const horizons =
    principal.horizontes || {};

  for (
    const minute of [5,10,20,30]
  ) {
    const item =
      horizons[String(minute)] ||
      horizons[minute];

    const value =
      number(
        item?.toward_bahia
      );

    setText(
      `#pred${minute}`,
      value !== null
        ? formatPercent(value)
        : "—"
    );
  }

  const strengthened5 =
    number(
      horizons["5"]?.strengthened
    );

  setText(
    "#strength5",
    strengthened5 !== null
      ? formatPercent(strengthened5)
      : "—"
  );

  const bearing =
    number(
      principal.motion_bearing_deg
    );

  setText(
    "#trackBearing",
    bearing !== null
      ? `${bearing.toFixed(1)}°`
      : "—"
  );

  setText(
    "#predictionMode",
    "Predicción ML del track más cercano · valores del modelo sin reinterpretarlos."
  );
}


/* ========================================================= */
/* SINCRONIZACIÓN / SISTEMA */
/* ========================================================= */

function renderSystem() {
  const health =
    state.health || {};

  const sync =
    health.sincronizacion ||
    state.radar?.sincronizacion ||
    {};

  const backend =
    health.status;

  setText(
    "#backendStatus",
    backend || "—"
  );

  setText(
    "#systemRadar",
    health.estado_radar ||
    state.radar?.estado_radar ||
    "—"
  );

  setText(
    "#systemRadarFrame",
    sync.radar || "—"
  );

  setText(
    "#systemNowcastFrame",
    sync.nowcast || "—"
  );

  setText(
    "#systemMlFrame",
    sync.ml || "—"
  );

  setText(
    "#systemSync",
    sync.sincronizados === true
      ? "SINCRONIZADOS"
      : "NO SINCRONIZADOS"
  );

  const backendElement =
    $("#backendStatus");

  const syncElement =
    $("#systemSync");

  const radarElement =
    $("#systemRadar");

  if (backendElement) {
    backendElement.className =
      health.status === "ok"
        ? "systemGood"
        : health.status === "degraded"
          ? "systemWarn"
          : "systemBad";
  }

  if (syncElement) {
    syncElement.className =
      sync.sincronizados === true
        ? "systemGood"
        : "systemWarn";
  }

  if (radarElement) {
    radarElement.className =
      health.estado_radar === "actualizado"
        ? "systemGood"
        : health.estado_radar === "retrasado"
          ? "systemWarn"
          : "systemBad";
  }

  setText(
    "#summarySync",
    sync.sincronizados === true
      ? "OK"
      : "—"
  );
}


/* ========================================================= */
/* SECUENCIA RADAR */
/* ========================================================= */

async function loadSequence() {
  try {
    const data =
      await getJSON("/secuencia");

    if (
      data &&
      Array.isArray(data.frames)
    ) {
      state.sequence =
        data.frames
          .filter(
            frame =>
              frame &&
              frame.imagen
          );
    }

    if (state.radar) {
      setText(
        "#radarHistory",
        state.sequence.length
          ? `${state.sequence.length} frames`
          : "Sin historial"
      );
    }

  } catch (error) {
    console.warn(
      "ClimaAR /secuencia:",
      error
    );
  }
}


function stopRadarPlayback() {
  state.playing = false;

  if (state.playTimer) {
    clearInterval(
      state.playTimer
    );

    state.playTimer = null;
  }

  const button =
    $("#radarPlay");

  if (button) {
    button.textContent = "▶";
    button.classList.remove(
      "playing"
    );
  }
}


function startRadarPlayback() {
  if (
    !state.sequence.length
  ) {
    stopRadarPlayback();

    setText(
      "#radarTimestamp",
      "Sin historial disponible"
    );

    return;
  }

  state.playing = true;
  state.playIndex = 0;

  const button =
    $("#radarPlay");

  if (button) {
    button.textContent = "■";
    button.classList.add(
      "playing"
    );
  }

  showRadarImage(
    state.sequence[
      state.playIndex
    ].imagen
  );

  state.playTimer =
    setInterval(
      () => {

        state.playIndex =
          (state.playIndex + 1) %
          state.sequence.length;

        const frame =
          state.sequence[
            state.playIndex
          ];

        if (frame) {
          showRadarImage(
            frame.imagen
          );

          setText(
            "#radarTimestamp",
            formatTimestamp(
              frame.timestamp
            )
          );

          setText(
            "#radarFrame",
            frame.timestamp
          );
        }

      },
      850
    );
}


function toggleRadarPlayback() {
  if (state.playing) {
    stopRadarPlayback();
  } else {
    startRadarPlayback();
  }
}


/* ========================================================= */
/* DRAG DEL RADAR */
/* ========================================================= */

function initRadarDrag() {
  const viewport =
    $("#radarViewport");

  if (!viewport) {
    return;
  }

  viewport.addEventListener(
    "pointerdown",
    event => {

      if (
        state.radarZoom <= 1
      ) {
        return;
      }

      state.dragging = true;

      state.dragStartX =
        event.clientX;

      state.dragStartY =
        event.clientY;

      state.dragPanX =
        state.radarPanX;

      state.dragPanY =
        state.radarPanY;

      viewport.setPointerCapture(
        event.pointerId
      );
    }
  );

  viewport.addEventListener(
    "pointermove",
    event => {

      if (!state.dragging) {
        return;
      }

      const maxPan =
        (state.radarZoom - 1) *
        140;

      state.radarPanX =
        Math.max(
          -maxPan,
          Math.min(
            maxPan,
            state.dragPanX +
              (
                event.clientX -
                state.dragStartX
              )
          )
        );

      state.radarPanY =
        Math.max(
          -maxPan,
          Math.min(
            maxPan,
            state.dragPanY +
              (
                event.clientY -
                state.dragStartY
              )
          )
        );

      applyRadarTransform();
    }
  );

  viewport.addEventListener(
    "pointerup",
    event => {
      state.dragging = false;

      try {
        viewport.releasePointerCapture(
          event.pointerId
        );
      } catch (_) {}
    }
  );

  viewport.addEventListener(
    "pointercancel",
    () => {
      state.dragging = false;
    }
  );
}


/* ========================================================= */
/* NAVEGACIÓN */
/* ========================================================= */

function scrollToSection(
  id
) {
  const section =
    document.getElementById(id);

  if (!section) {
    return;
  }

  section.scrollIntoView({
    behavior: "smooth",
    block: "start"
  });
}


function initNavigation() {
  $$(".navItem").forEach(
    button => {

      button.addEventListener(
        "click",
        () => {

          const target =
            button.dataset.target;

          if (target) {
            scrollToSection(
              target
            );
          }
        }
      );
    }
  );

  const menu =
    $("#menuButton");

  if (menu) {
    menu.addEventListener(
      "click",
      () => {
        scrollToSection("mas");
      }
    );
  }

  const notification =
    $("#notificationButton");

  if (notification) {
    notification.addEventListener(
      "click",
      () => {
        scrollToSection(
          "adversas"
        );
      }
    );
  }

  const alert =
    $("#alertCard");

  if (alert) {
    alert.addEventListener(
      "click",
      () => {
        scrollToSection(
          "adversas"
        );
      }
    );
  }

  const observer =
    new IntersectionObserver(
      entries => {

        const visible =
          entries
            .filter(
              entry =>
                entry.isIntersecting
            )
            .sort(
              (a,b) =>
                b.intersectionRatio -
                a.intersectionRatio
            )[0];

        if (!visible) {
          return;
        }

        const id =
          visible.target.dataset.section;

        $$(".navItem")
          .forEach(
            button => {

              button.classList.toggle(
                "active",
                button.dataset.target === id
              );
            }
          );
      },
      {
        root: null,
        threshold: [0.2,0.4,0.6],
        rootMargin:
          "-15% 0px -55% 0px"
      }
    );

  $$(".section")
    .forEach(
      section =>
        observer.observe(section)
    );
}


/* ========================================================= */
/* RADAR CONTROLS */
/* ========================================================= */

function initRadarControls() {
  const plus =
    $("#zoomIn");

  const minus =
    $("#zoomOut");

  const reset =
    $("#radarReset");

  const play =
    $("#radarPlay");

  if (plus) {
    plus.addEventListener(
      "click",
      () => changeRadarZoom(.2)
    );
  }

  if (minus) {
    minus.addEventListener(
      "click",
      () => changeRadarZoom(-.2)
    );
  }

  if (reset) {
    reset.addEventListener(
      "click",
      resetRadarView
    );
  }

  if (play) {
    play.addEventListener(
      "click",
      toggleRadarPlayback
    );
  }
}


/* ========================================================= */
/* HTML SAFE */
/* ========================================================= */

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}



/* ========================================================= */
/* CONDICIONES ADVERSAS */
/* ========================================================= */

const CLASS_LABELS = {
  1: "Clase 1 — fenómeno débil",
  2: "Clase 2 — fenómeno moderado",
  3: "Clase 3 — fenómeno fuerte",
  4: "Clase 4 — fenómeno severo"
};

const CLASS_FROM_CLASSIFICATION = {
  tormenta: 2,
  tormenta_fuerte: 3,
  tormenta_severa: 4
};

const CLASS_FROM_SEVERITY = {
  amarillo: 2,
  naranja: 3,
  rojo: 4
};


function intelBlock() {
  const intel = state.intelligence || {};
  const storm = intel.inteligencia_tormenta || {};
  const final = intel.resultado_final || {};

  return { intel, storm, final };
}


function adverseClass() {
  const raw = getIntelligenceClassification();

  return CLASS_FROM_CLASSIFICATION[raw] ||
    CLASS_FROM_SEVERITY[getIntelligenceSeverity()] ||
    null;
}


function renderAdverse() {
  const { intel, storm, final } = intelBlock();

  const threat = firstValue(
    intel.amenaza_bahia,
    final.amenaza_bahia,
    storm.amenaza_bahia
  ) || {};

  const classification = getIntelligenceClassification();

  const active =
    threat.amenaza === true ||
    (classification !== null && classification !== "sin_tormenta");

  const none = $("#advNone");
  const detail = $("#advDetail");

  if (none) none.classList.toggle("hidden", active);
  if (detail) detail.classList.toggle("hidden", !active);

  renderRadarHumanText(active, threat);

  if (!active) {
    return;
  }

  setText(
    "#advType",
    classification ? CLASSIFICATION_LABELS[classification] : "Tormenta"
  );

  const cls = adverseClass();

  setText("#advClass", cls ? CLASS_LABELS[cls] : "Clase no determinada");

  const radar = state.radar || {};
  const stormRadar = storm.radar || {};

  // Bearing desde el fenómeno hacia Bahía Blanca; llega desde el opuesto.
  const toCity = number(
    stormRadar.bahia_blanca?.rumbo_desde_principal_grados
  );

  const heading = firstNumber(
    stormRadar.direccion_grados,
    radar.direccion_grados
  );

  const from =
    toCity !== null
      ? toCity + 180
      : heading !== null &&
        (stormRadar.movimiento_hacia_bahia ?? radar.movimiento_hacia_bahia) === true
        ? heading + 180
        : null;

  setText(
    "#advDirection",
    from !== null
      ? `Llega desde el ${degreesToCardinal(from)}`
      : "—"
  );

  const distance = firstNumber(threat.distancia_km, radar.distancia_km);

  setText(
    "#advDistance",
    distance !== null ? `${distance.toFixed(0)} km` : "—"
  );

  const eta = firstNumber(
    threat.eta_minutos,
    intel.eta_minutos,
    storm.eta_minutos,
    radar.eta_minutos
  );

  setText(
    "#advEta",
    eta !== null ? `Posible llegada en ${Math.round(eta)} min` : "—"
  );

  const duration = firstNumber(
    storm.duracion_estimada_minutos,
    final.duracion_estimada_minutos
  );

  setText(
    "#advDuration",
    duration !== null ? `${Math.round(duration)} min` : "—"
  );

  setText("#advTrend", trendText(
    firstValue(stormRadar.fortalecimiento, radar.fortalecimiento)
  ));

  const probability = firstNumber(
    storm.probabilidades?.llegada_bahia,
    storm.probabilidades?.impacto,
    storm.probabilidades?.tormenta
  );

  setText(
    "#advProbability",
    probability !== null
      ? `Probabilidad: ${formatPercent(toPercent(probability))}`
      : "Probabilidad: no disponible"
  );

  const confidence = firstNumber(
    storm.confianza_modelo,
    final.motor_decision?.componentes?.ia_confianza
  );

  setText(
    "#advConfidence",
    confidence !== null
      ? `Confianza: ${formatPercent(toPercent(confidence))}`
      : "Confianza: no disponible"
  );
}


function toPercent(value) {
  return value <= 1 ? value * 100 : value;
}


function trendText(value) {
  const map = {
    fortaleciendose: "Fortaleciéndose",
    fortaleciendo: "Fortaleciéndose",
    debilitandose: "Debilitándose",
    debilitando: "Debilitándose",
    estable: "Estable"
  };

  return map[value] || "—";
}


function renderRadarHumanText(active, threat) {
  const box = $("#radarHumanText");

  if (!box) {
    return;
  }

  if (!active) {
    box.innerHTML = "<p>No se esperan condiciones adversas.</p>";
    return;
  }

  const lines = [];
  const eta = number(threat.eta_minutos);
  const distance = number(threat.distancia_km);
  const cls = adverseClass();
  const classification = getIntelligenceClassification();

  if (classification) {
    lines.push(`${CLASSIFICATION_LABELS[classification]} cerca de Bahía Blanca.`);
  }

  if (distance !== null) lines.push(`Distancia: ${distance.toFixed(0)} km.`);
  if (eta !== null) lines.push(`Posible llegada en ${Math.round(eta)} minutos.`);
  if (cls) lines.push(`Intensidad: ${CLASS_LABELS[cls]}.`);

  box.innerHTML = lines.length
    ? lines.map(line => `<p>${escapeHtml(line)}</p>`).join("")
    : "<p>Sin información de tormentas para mostrar.</p>";
}


/* ========================================================= */
/* MAPA (Leaflet) */
/* ========================================================= */

// Coordenadas públicas de Bahía Blanca, solo como respaldo si la API no responde.
const FALLBACK_CENTER = [-38.7183, -62.2663];

const map = {
  instance: null,
  city: null,
  radarOverlay: null,
  nuclei: null,
  track: null
};


function validLatLon(lat, lon) {
  const a = number(lat);
  const b = number(lon);

  return a !== null && b !== null &&
    Math.abs(a) <= 90 && Math.abs(b) <= 180
    ? [a, b]
    : null;
}


function cityCoordinates() {
  const storm = state.intelligence?.inteligencia_tormenta || {};

  return validLatLon(state.health?.lat, state.health?.lon) ||
    validLatLon(storm.ubicacion?.latitud, storm.ubicacion?.longitud);
}


function initMap() {
  const element = $("#map");

  if (!element) {
    return;
  }

  if (typeof L === "undefined") {
    setText("#mapNote", "No se pudo cargar el mapa.");
    return;
  }

  map.instance = L.map(element, { zoomControl: true })
    .setView(FALLBACK_CENTER, 9);

  L.tileLayer(
    "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    {
      maxZoom: 18,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    }
  ).addTo(map.instance);

  map.nuclei = L.layerGroup().addTo(map.instance);
  map.track = L.layerGroup().addTo(map.instance);
}


// Bounds reales del radar: solo si la API los entrega.
function radarBounds() {
  const r = state.radar || {};
  const b = r.bounds || r.limites || r.radar_bounds;

  if (Array.isArray(b) && b.length === 2) {
    const sw = validLatLon(b[0]?.[0], b[0]?.[1]);
    const ne = validLatLon(b[1]?.[0], b[1]?.[1]);

    return sw && ne ? [sw, ne] : null;
  }

  return null;
}


function renderMap() {
  if (!map.instance) {
    return;
  }

  const city = cityCoordinates();

  if (city && !map.city) {
    map.instance.setView(city, map.instance.getZoom());
  }

  const position = city || FALLBACK_CENTER;

  if (!map.city) {
    map.city = L.marker(position)
      .addTo(map.instance)
      .bindPopup("Bahía Blanca");
  } else {
    map.city.setLatLng(position);
  }

  const bounds = radarBounds();

  if (map.radarOverlay) {
    map.instance.removeLayer(map.radarOverlay);
    map.radarOverlay = null;
  }

  if (bounds) {
    map.radarOverlay = L.imageOverlay(
      radarImageUrl("/radar/imagen"),
      bounds,
      { opacity: 0.7 }
    ).addTo(map.instance);
  }

  setText(
    "#mapNote",
    bounds
      ? "Radar sobre el mapa."
      : "El radar sobre el mapa se mostrará cuando ClimaAR entregue su ubicación geográfica real."
  );

  map.nuclei.clearLayers();
  map.track.clearLayers();

  const nuclei = Array.isArray(state.radar?.nucleos)
    ? state.radar.nucleos
    : [];

  nuclei.forEach(item => {
    const geo = item?.georreferencia;
    const pos = validLatLon(geo?.latitud, geo?.longitud);

    if (pos) {
      L.circleMarker(pos, { radius: 8, color: "#ff9f43" })
        .addTo(map.nuclei)
        .bindPopup(
          `Núcleo detectado${
            number(geo?.distancia_bahia_km) !== null
              ? ` · a ${Math.round(geo.distancia_bahia_km)} km de Bahía Blanca`
              : ""
          }`
        );
    }
  });

  const projections =
    state.intelligence?.inteligencia_tormenta?.radar?.proyecciones;

  if (projections && typeof projections === "object") {
    const points = Object.values(projections)
      .map(p => validLatLon(p?.latitud, p?.longitud))
      .filter(Boolean);

    if (points.length > 1) {
      L.polyline(points, { color: "#ff5e72", dashArray: "6 6" })
        .addTo(map.track);
    }
  }
}


/* ========================================================= */
/* PRONÓSTICO DE 3 DÍAS */
/* ========================================================= */

// No existe aún un endpoint de pronóstico diario: estado vacío.
const WEEKDAYS = ["domingo", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado"];

function dayName(iso) {
  const d = new Date(`${iso}T12:00:00`);

  return Number.isNaN(d.getTime())
    ? null
    : WEEKDAYS[d.getDay()];
}


function dayDescription(d) {
  const parts = [];
  const rain = number(d.prob_lluvia);
  const gust = number(d.rafagas);
  const mm = number(d.lluvia_mm);

  if (rain !== null) {
    parts.push(
      rain >= 60 ? "Es probable que llueva."
        : rain >= 30 ? "Puede llover."
        : "Poco probable que llueva."
    );
  }

  if (mm !== null && mm > 0) parts.push(`Se esperan unos ${mm.toFixed(1)} mm.`);
  if (gust !== null && gust >= 60) parts.push(`Ráfagas fuertes de hasta ${Math.round(gust)} km/h.`);

  return parts.join(" ") || null;
}


function mapForecast(payload) {
  const list = Array.isArray(payload?.dias) ? payload.dias : [];

  return list.map(d => ({
    dia: dayName(d.fecha),
    estado: weatherCodeText(d.codigo_tiempo),
    minima: d.minima,
    maxima: d.maxima,
    viento: d.viento,
    prob_lluvia: d.prob_lluvia,
    descripcion: dayDescription(d),
    severo: number(d.codigo_tiempo) >= 95
  }));
}


function renderForecast(days) {
  const list = Array.isArray(days) ? days.slice(0, 3) : [];
  const empty = $("#forecastEmpty");

  if (empty) empty.classList.toggle("hidden", list.length > 0);

  $$("#forecastGrid .dayCard").forEach((card, index) => {
    const day = list[index] || {};
    const set = (field, value) => {
      const el = card.querySelector(`[data-field="${field}"]`);
      if (el) el.textContent = value ?? "—";
    };

    set("day", day.dia || `Día ${index + 1}`);
    set("state", day.estado);
    set("min", number(day.minima) !== null ? `${Math.round(day.minima)}°` : "—");
    set("max", number(day.maxima) !== null ? `${Math.round(day.maxima)}°` : "—");
    set("wind", number(day.viento) !== null ? `${Math.round(day.viento)} km/h` : "—");
    set("rain", number(day.prob_lluvia) !== null ? `${Math.round(day.prob_lluvia)}%` : "—");
    set("desc", day.descripcion);
    card.classList.toggle("dayAlert", day.severo === true);
  });
}

/* ========================================================= */
/* CARGA */
/* ========================================================= */

async function cargarDatos() {

  const results =
    await Promise.allSettled([
      getJSON("/radar"),
      getJSON("/health"),
      getJSON("/estado"),
      getJSON("/prediccion/en-vivo"),
      getJSON("/tormenta"),
      getJSON("/inteligencia"),
      getJSON("/modelo"),
      getJSON("/pronostico")
    ]);

  const [
    radar,
    health,
    weather,
    livePrediction,
    storm,
    intelligence,
    model,
    forecast
  ] = results;

  state.forecast =
    forecast.status === "fulfilled"
      ? forecast.value
      : state.forecast;

  state.radar =
    radar.status === "fulfilled"
      ? radar.value
      : state.radar;

  state.health =
    health.status === "fulfilled"
      ? health.value
      : state.health;

  state.weather =
    weather.status === "fulfilled"
      ? weather.value
      : state.weather;

  state.livePrediction =
    livePrediction.status === "fulfilled"
      ? livePrediction.value
      : state.livePrediction;

  state.storm =
    storm.status === "fulfilled"
      ? storm.value
      : state.storm;

  state.intelligence =
    intelligence.status === "fulfilled"
      ? intelligence.value
      : state.intelligence;

  state.model =
    model.status === "fulfilled"
      ? model.value
      : state.model;


  renderRadar(
    state.radar
  );

  renderWeather(
    state.weather
  );

  renderAlert();

  renderStorm();

  renderNuclei();

  renderModel();

  renderSystem();

  renderAdverse();

  renderMap();

  renderForecast(mapForecast(state.forecast));
}


/* ========================================================= */
/* INIT */
/* ========================================================= */

async function init() {

  initNavigation();

  initMap();

  initRadarControls();

  initRadarDrag();

  resetRadarView();

  await cargarDatos();

  await loadSequence();

  setInterval(
    cargarDatos,
    60000
  );
}


document.addEventListener(
  "DOMContentLoaded",
  init
);
