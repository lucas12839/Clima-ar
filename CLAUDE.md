# ClimaAR — instrucciones para trabajar en este repo

## Decisiones fijas
- Detector de tormentas en tiempo real para Bahía Blanca. Radar RMA10 (SINARAME, ZH MAX, PNG nativo), alcance 240 km.
- **Sin modelos entrenados, sin revisión manual de celdas, sin DGMR/MetNet.** Todo por reglas y física. No agregar `joblib`, `scikit-learn` ni `torch`.
- SINARAME publica cada ~10 min y a veces llega con 30 min o más de atraso.
- El detector corre como servicio systemd en un VPS (consulta cada 30 s). GitHub Actions solo vigila.

## Reglas de trabajo
- `main` = producción. Nunca commitear directo a `main`: rama + pull request.
- Los bots de Actions commitean datos en `data/` constantemente; los PR de código no deben tocar `data/` salvo que sea el objetivo.
- Antes de abrir un PR: `python -m pytest -q` y correr el ciclo de Intelligence en una copia (ver `climaar-intelligence.yml` para el orden).
- Mantener los contratos de archivos JSON que leen la API, Intelligence y `validar_integridad_climaar.py`.

## Ciclo
detector → `probar-radar-smn.yml` (captura, dBZ, seguimiento) → `climaar-intelligence.yml` (nowcast, georreferenciación, evolución, fusión, Storm Engine, decisión, integridad) → `climaar-calidad.yml`.
