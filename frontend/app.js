const API_BASE_URL =
  typeof CLIMAAR_CONFIG !== "undefined" &&
  typeof CLIMAAR_CONFIG.API_BASE_URL === "string"
    ? CLIMAAR_CONFIG.API_BASE_URL.replace(/\/+$/, "")
    : "";

const $ = (selector) => document.querySelector(selector);

function apiUrl(path) {
  if (!API_BASE_URL) {
    return path;
  }

  return `${API_BASE_URL}${path}`;
}

async function getJSON(path) {
  const response = await fetch(apiUrl(path), {
    cache: "no-store",
    headers: {
      Accept: "application/json"
    }
  });

  if (!response.ok) {
    throw new Error(`${path} → HTTP ${response.status}`);
  }

  return response.json();
}

function numero(value, fallback = null) {
  const n = Number(value);

  return Number.isFinite(n) ? n : fallback;
}

function texto(value, fallback = "Sin datos suficientes") {
  if (
    value === null ||
    value === undefined ||
    value === ""
  ) {
    return fallback;
  }

  return String(value);
}

function actualizarRadar(data) {
  const metadata = data || {};

  const imagen =
    metadata.imagen_local ||
    metadata.imagen_url ||
    metadata.image_url ||
    "/radar/imagen";

  const radar = $("#radar");

  if (radar) {
    radar.src =
      `${apiUrl(imagen)}${imagen.includes("?") ? "&" : "?"}t=${Date.now()}`;
  }

  const edad = numero(metadata.frame_edad_minutos);

  if ($("#updated")) {
    if (edad !== null) {
      $("#updated").textContent =
        `Actualizado hace ${Math.round(edad)} min`;
    } else {
      $("#updated").textContent =
        "Sin datos suficientes";
    }
  }

  const live = document.querySelector(".live");

  if (live) {
    if (metadata.radar_actualizado === true) {
      live.textContent = "● EN VIVO";
    } else if (metadata.radar_retrasado === true) {
      live.textContent = "● RETRASADO";
    } else {
      live.textContent = "● SIN DATOS";
    }
  }
}

function actualizarEstado(data) {
  if (!data) {
    return;
  }

  const temperatura = numero(data.temperatura);
  const viento = numero(data.viento);
  const presion = numero(data.presion);
  const lluvia = numero(data.precipitacion);

  if ($("#temp")) {
    $("#temp").textContent =
      temperatura !== null
        ? `${Math.round(temperatura)}°`
        : "--";
  }

  if ($("#condition")) {
    $("#condition").textContent =
      texto(data.nowcast, "Sin datos suficientes");
  }

  const weatherSpan = document.querySelector(".weather span");

  if (weatherSpan) {
    const rocio = numero(data.punto_rocio);

    const partes = [];

    if (rocio !== null) {
      partes.push(`Punto de rocío ${rocio.toFixed(1)}°`);
    }

    if (data.meteorologia_externa === "no_disponible") {
      partes.push("Meteorología externa sin datos");
    }

    weatherSpan.textContent =
      partes.length > 0
        ? partes.join(" · ")
        : "Sin datos suficientes";
  }

  const metricValues = document.querySelectorAll(".metrics b");

  if (metricValues.length >= 3) {
    metricValues[0].textContent =
      viento !== null
        ? `${Math.round(viento)} km/h`
        : "--";

    metricValues[1].textContent =
      presion !== null
        ? `${Math.round(presion)} hPa`
        : "--";

    metricValues[2].textContent =
      lluvia !== null
        ? `${lluvia} mm`
        : "--";
  }

  const stormStats = document.querySelectorAll(".stormStats > div");

  if (stormStats.length >= 3) {
    const movementValue =
      stormStats[0].querySelector("b");

    const distanceValue =
      stormStats[1].querySelector("strong");

    const etaValue =
      stormStats[2].querySelector("strong");

    const etaLabel =
      stormStats[2].querySelector("b");

    if (movementValue) {
      movementValue.textContent =
        texto(data.movimiento);
    }

    if (distanceValue) {
      distanceValue.textContent =
        "—";
    }

    if (etaValue) {
      const eta = numero(data.eta_minutos);

      etaValue.textContent =
        eta !== null
          ? `${Math.round(eta)} min`
          : "—";
    }

    if (etaLabel) {
      etaLabel.textContent =
        data.movimiento_hacia_bahia === true
          ? "Posible llegada"
          : "Sin confirmación";
    }
  }
}

function actualizarRadarEstado(data) {
  if (!data) {
    return;
  }

  const risk = document.querySelector(".risk");

  if (risk) {
    const intensity = numero(
      data.intensidad_nivel_max
    );

    if (intensity === null) {
      risk.textContent = "SIN DATOS";
    } else if (intensity >= 6) {
      risk.textContent = "SEVERA";
    } else if (intensity >= 4) {
      risk.textContent = "FUERTE";
    } else if (intensity >= 3) {
      risk.textContent = "MODERADA";
    } else {
      risk.textContent = "DÉBIL";
    }
  }

  const stormTitle =
    document.querySelector(".stormCard .title b");

  if (stormTitle) {
    stormTitle.textContent =
      texto(data.intensidad, "Sin datos suficientes");
  }

  const prediction =
    document.querySelector(".prediction");

  if (prediction) {
    const synchronized =
      data.sincronizacion &&
      data.sincronizacion.sincronizados === true;

    prediction.textContent = synchronized
      ? "Datos sincronizados · Radar / Nowcast / ML"
      : "Sin datos sincronizados suficientes";
  }
}

async function cargarDatos() {
  try {
    const [radar, estado] = await Promise.all([
      getJSON("/radar"),
      getJSON("/estado")
    ]);

    actualizarRadar(radar);
    actualizarRadarEstado(radar);
    actualizarEstado(estado);

  } catch (error) {
    console.error("ClimaAR:", error);

    if ($("#updated")) {
      $("#updated").textContent =
        "Sin datos suficientes";
    }

    const live = document.querySelector(".live");

    if (live) {
      live.textContent = "● SIN DATOS";
    }
  }
}

cargarDatos();

setInterval(cargarDatos, 60000);
