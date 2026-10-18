const $ = (selector) => document.querySelector(selector);

async function getJSON(path) {
  const response = await fetch(path, {
    cache: "no-store"
  });

  if (!response.ok) {
    throw new Error(`${path} → HTTP ${response.status}`);
  }

  return response.json();
}

function actualizarRadar(data) {
  const metadata = data.metadata || {};

  const imagen =
    data.imagen_url ||
    data.image_url ||
    "/radar/imagen";

  $("#radar").src =
    `${imagen}${imagen.includes("?") ? "&" : "?"}t=${Date.now()}`;

  const edad = metadata.frame_edad_minutos;

  if (Number.isFinite(Number(edad))) {
    $("#updated").textContent =
      `Actualizado hace ${Math.round(Number(edad))} min`;
  } else {
    $("#updated").textContent =
      "Frame disponible";
  }
}

function cargarDatos() {
  getJSON("/radar")
    .then(actualizarRadar)
    .catch((error) => {
      console.error("ClimaAR:", error);

      $("#updated").textContent =
        "Sin datos suficientes";
    });
}

cargarDatos();

setInterval(cargarDatos, 60000);
