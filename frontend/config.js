// Los pedidos van a /api/... en la misma dirección de la página: Caddy, en el
// VPS, los pasa al servidor de la app (deploy/Caddyfile). Sin Netlify ni Render
// desde el 24/11/2026.
const CLIMAAR_CONFIG = {
  API_BASE_URL: "/api"
};
