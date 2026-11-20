# ClimaAR — instrucciones para trabajar en este repo

## Decisiones fijas
- Detector de tormentas en tiempo real para Bahía Blanca. Radar RMA10 (SINARAME, ZH MAX, PNG nativo), alcance 240 km.
- **Sin modelos entrenados, sin revisión manual de celdas, sin DGMR/MetNet.** Todo por reglas y física. No agregar `joblib`, `scikit-learn` ni `torch`.
- SINARAME publica cada ~10 min y a veces llega con 30 min o más de atraso.
- El detector corre como servicio systemd en un VPS (consulta cada 30 s). GitHub Actions solo vigila.

## Reglas de trabajo
- `main` = producción. Nunca commitear directo a `main`: rama + pull request.
- Los bots de Actions commitean datos en `data/` constantemente; los PR de código no deben tocar `data/` salvo que sea el objetivo.
- Antes de abrir un PR: `python -m pytest -q` y correr los workflows reales en una copia del repo con `herramientas/emular_workflow.py` (ejecuta los pasos `run:` y sus validaciones; saltea los que necesitan GitHub o red):
  `python herramientas/emular_workflow.py <copia> .github/workflows/climaar-intelligence.yml $(which python) --desde "Conectar RMA10" --hasta "Validar integridad"`
- Probar siempre los dos casos: con tormenta y **sin actividad** (sin núcleos, nowcast `actividad: false`, sin principal).
- Mantener los contratos de archivos JSON que leen la API, Intelligence y `validar_integridad_climaar.py`.

## Antes de cerrar cualquier punto (auditoría 20/11/2026, `docs/AUDITORIA_2026-11-20.md`)
- Mirar `data/ia/salud_cadena.json` en main (control de salud de cada ciclo): sin alertas nuevas.
- Edad de los datos: usar `scripts/estaciones/frescura.py` y calcular la edad **al usar** el dato, nunca confiar en una marca "fresca" guardada antes.
- Un archivo de estado lo escribe **un solo** programa (el capturador no toca el estado del seguimiento).
- Números de tormenta: nunca vuelven atrás; todo lo que compara una tormenta entre ciclos debe controlar que sea la misma.
- Relojes: GitHub corre los programados cada 10 min solo ~3 veces por día; lo que tenga que correr seguido va encadenado (como `climaar-cwop.yml`) o lo dispara el detector.
- Nada nuevo se calcula para tirarlo: si nadie lo lee o no se guarda, no va.
- Descarga corta (`fetch-depth: 1`), Python 3.12, versiones fijas en `requirements.txt`, JSON grandes compactos.

## Ciclo
detector → `probar-radar-smn.yml` (captura, dBZ, seguimiento) → `climaar-intelligence.yml` (nowcast, georreferenciación, evolución, fusión, Storm Engine, decisión, integridad) → `climaar-calidad.yml`.
