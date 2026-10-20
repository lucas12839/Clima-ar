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

  setText(
    "#condition",
    state.weather.nowcast ||
    "Sin datos suficientes"
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
}


/* ========================================================= */
/* ALERTA PRINCIPAL */
/* ========================================================= */

function intensityLabel(level) {
  const n = number(level);

  const labels = {
    0: "Sin precipitación",
    1: "Muy débil",
    2: "Débil",
    3: "Moderada",
    4: "Fuerte",
    5: "Muy fuerte",
    6: "Severa"
  };

  return labels[n] || "Sin datos";
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

  setText(
    "#alertDescription",
    `${cores} núcleo${
      cores === 1
        ? ""
        : "s"
    } detectado${
      cores === 1
        ? ""
        : "s"
    } · Intensidad ${
      intensityLabel(level)
    }.`
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
  const radar =
    state.radar || {};

  const cores =
    Array.isArray(radar.nucleos)
      ? radar.nucleos
      : [];

  const level =
    number(
      radar.intensidad_nivel_max
    );

  const label =
    radar.intensidad ||
    intensityLabel(level);

  setText(
    "#stormTitle",
    cores.length
      ? "Actividad detectada"
      : "Sin actividad detectada"
  );

  setText(
    "#stormRisk",
    label
  );

  const risk =
    $("#stormRisk");

  if (risk) {
    risk.className =
      "riskBadge";

    if (level !== null && level >= 5) {
      risk.classList.add(
        "riskHigh"
      );
    } else if (
      level !== null &&
      level >= 3
    ) {
      risk.classList.add(
        "riskMedium"
      );
    } else if (
      level !== null &&
      level > 0
    ) {
      risk.classList.add(
        "riskLow"
      );
    }
  }

  setText(
    "#stormIntensity",
    level !== null
      ? String(level)
      : "—"
  );

  setText(
    "#stormIntensityLevel",
    label
  );

  setText(
    "#stormCores",
    cores.length
      ? String(cores.length)
      : "0"
  );

  const weather =
    state.weather || {};

  const intelligenceRadar =
    state.intelligence?.radar || {};

  setText(
    "#stormMovement",
    weather.movimiento ||
    (intelligenceRadar.movimiento_hacia_bahia === true
      ? "Movimiento hacia Bahía Blanca"
      : "Sin movimiento confirmado")
  );

  const distance =
    number(intelligenceRadar.distancia_km);

  setText(
    "#stormDistance",
    distance !== null
      ? `${distance.toFixed(1)} km`
      : "Sin datos"
  );

  const eta =
    number(intelligenceRadar.eta_minutos);

  setText(
    "#stormEta",
    eta !== null
      ? `${Math.round(eta)} min`
      : "Sin datos"
  );

  const direction =
    number(intelligenceRadar.direccion_grados);

  setText(
    "#stormDirection",
    direction !== null
      ? `${direction.toFixed(1)}°`
      : "Sin datos"
  );

  const trends = cores
    .map(
      core =>
        core?.movimiento?.tendencia
    )
    .filter(Boolean);

  let trend =
    "Sin datos suficientes";

  if (trends.length) {
    const counts = {};

    for (const item of trends) {
      counts[item] =
        (counts[item] || 0) + 1;
    }

    trend =
      Object.entries(counts)
        .sort(
          (a,b) => b[1] - a[1]
        )[0][0];
  }

  const intelligenceTrend =
    intelligenceRadar.fortalecimiento;

  if (intelligenceTrend) {
    trend = intelligenceTrend;
  }

  const trendMap = {
    fortaleciendose:
      "Se observan núcleos fortaleciéndose",

    debilitandose:
      "Se observan núcleos debilitándose",

    estable:
      "La evolución observada es estable",

    fortaleciéndose:
      "Se observan núcleos fortaleciéndose",

    debilitándose:
      "Se observan núcleos debilitándose"
  };

  setText(
    "#stormTrend",
    trendMap[trend] ||
    trend
  );

  setText(
    "#stormTrendDetail",
    trends.length
      ? `${trends.length} tracks con tendencia disponible.`
      : intelligenceTrend
        ? "Tendencia tomada del motor de Inteligencia."
        : "Sin tendencia disponible."
  );

  setText(
    "#summaryCores",
    cores.length
  );

  setText(
    "#summaryIntensity",
    level !== null
      ? `${level}/6`
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


function findPrincipalThreatTrack(
  predictions
) {
  if (!predictions.length) {
    return null;
  }

  const intelligenceRadar =
    state.intelligence?.radar || {};

  const intelligenceNuclei =
    Array.isArray(intelligenceRadar.nucleos)
      ? intelligenceRadar.nucleos
      : [];

  const targetDirection =
    number(intelligenceRadar.direccion_grados);

  const intelligenceTracks =
    intelligenceNuclei
      .map(core => ({
        trackId: core?.track_id,
        direction: firstNumber(
          core?.direction,
          core?.direccion,
          core?.movimiento?.direccion_grados
        )
      }))
      .filter(
        item =>
          item.trackId !== undefined &&
          item.trackId !== null
      );

  if (intelligenceTracks.length) {
    const directMatch =
      predictions.find(prediction =>
        intelligenceTracks.some(
          track =>
            String(track.trackId) ===
            String(prediction.track_id)
        )
      );

    if (directMatch) {
      if (targetDirection === null) {
        return directMatch;
      }

      const candidates =
        predictions.filter(prediction =>
          intelligenceTracks.some(
            track =>
              String(track.trackId) ===
              String(prediction.track_id)
          )
        );

      const angularDistance = (a, b) => {
        const diff =
          Math.abs(a - b) % 360;
        return Math.min(diff, 360 - diff);
      };

      const best =
        candidates
          .map(prediction => {
            const track =
              intelligenceTracks.find(
                item =>
                  String(item.trackId) ===
                  String(prediction.track_id)
              );

            const predictionDirection =
              firstNumber(
                prediction.motion_bearing_deg,
                track?.direction
              );

            return {
              prediction,
              directionError:
                predictionDirection !== null
                  ? angularDistance(
                      predictionDirection,
                      targetDirection
                    )
                  : 999,
              distance:
                number(prediction.distance_km) ??
                999999
            };
          })
          .sort(
            (a, b) =>
              a.directionError -
              b.directionError ||
              a.distance - b.distance
          )[0];

      if (best) {
        return best.prediction;
      }
    }
  }

  return [...predictions]
    .filter(
      item =>
        number(item.distance_km) !== null
    )
    .sort(
      (a,b) =>
        number(a.distance_km) -
        number(b.distance_km)
    )[0] || null;
}


function renderModel() {
  const model =
    state.model || {};

  const predictions =
    getPredictions(model);

  const nearest =
    findPrincipalThreatTrack(
      predictions
    );

  setText(
    "#mlFrame",
    model.frame_radar ||
    "—"
  );

  if (!nearest) {
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
    nearest.track_id !== undefined
      ? `Track ${nearest.track_id}`
      : "Track"
  );

  const distance =
    number(nearest.distance_km);

  setText(
    "#predictionDistance",
    distance !== null
      ? `${distance.toFixed(1)} km`
      : "—"
  );

  const horizons =
    nearest.horizontes || {};

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
      nearest.motion_bearing_deg
    );

  setText(
    "#trackBearing",
    bearing !== null
      ? `${bearing.toFixed(1)}°`
      : "—"
  );

  setText(
    "#predictionMode",
    "Predicción ML del track principal de amenaza · valores del modelo sin reinterpretarlos."
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
          "tormentas"
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
          "tormentas"
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
      getJSON("/modelo")
    ]);

  const [
    radar,
    health,
    weather,
    livePrediction,
    storm,
    intelligence,
    model
  ] = results;

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
}


/* ========================================================= */
/* INIT */
/* ========================================================= */

async function init() {

  initNavigation();

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
