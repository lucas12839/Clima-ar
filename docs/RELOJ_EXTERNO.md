# Reloj externo de respaldo (etapa C, afuera de GitHub)

GitHub no es un reloj confiable: corre los programados "cada 10 min" solo unas 3 veces por
día. Hoy el vigilante del detector lo lanza el reloj encadenado de CWOP. Si esa cadena se
corta, el reloj externo la vuelve a levantar: cada 10 min llama al vigilante, y el vigilante
revisa todo y relanza la cadena si hace falta.

Se usa **cron-job.org** (gratis, anda desde el celular). Tarda unos 10 minutos.

## 1. Crear la clave en GitHub (solo puede lanzar procesos)

1. GitHub → foto de perfil → **Settings** → **Developer settings** → **Personal access tokens**
   → **Fine-grained tokens** → **Generate new token**.
2. Nombre: `climaar-reloj-externo`. Vencimiento: 90 días (como la del detector).
3. **Repository access**: *Only select repositories* → `lucas12839/Clima-ar`.
4. **Permissions** → *Repository permissions* → **Actions: Read and write**. Nada más.
5. **Generate token** y copiar la clave (empieza con `github_pat_`). No se vuelve a mostrar.

## 2. Crear la tarea en cron-job.org

1. Entrar a https://cron-job.org y crear una cuenta.
2. **Create cronjob**:
   - Title: `ClimaAR vigilante`
   - URL: `https://api.github.com/repos/lucas12839/Clima-ar/actions/workflows/climaar-detector-sinarame.yml/dispatches`
   - Execution schedule: **Every 10 minutes**.
3. Pestaña **Advanced**:
   - Request method: **POST**
   - Headers (agregar 3):
     - `Authorization` = `Bearer ` + la clave (con un espacio después de Bearer)
     - `Accept` = `application/vnd.github+json`
     - `X-GitHub-Api-Version` = `2022-11-28`
   - Request body: `{"ref":"main"}`
4. Guardar y tocar **Test run**: tiene que responder **204** (bien, sin contenido).

## Cómo se controla

- En GitHub → Actions → *ClimaAR Detector SINARAME (watchdog)* aparecen corridas lanzadas por
  `lucas12839` cada 10 min.
- Si el reloj externo venía llamando y deja de hacerlo por más de 30 min (por ejemplo, porque
  venció su clave), la vigilancia abre un aviso.
- Antes de que venza la clave (90 días), crear otra y cambiarla en la tarea de cron-job.org.
