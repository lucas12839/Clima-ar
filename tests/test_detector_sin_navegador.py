"""23/11/2026: Playwright nuevo sin navegador. El detector fallaba en cada consulta
y seguía dando latido: nadie se enteraba y el respaldo de GitHub no actuaba."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import detector_sinarame as det  # noqa: E402
import vigilancia as vg  # noqa: E402


def test_fallas_seguidas_cortan_el_latido_y_publican_el_error(tmp_path, monkeypatch):
    latidos, infos = [], []
    monkeypatch.setattr(det, "ESTADO_LOCAL", tmp_path / "estado.json", raising=False)
    monkeypatch.setattr(det.GitHub, "escribir_latido", lambda self, v, nombre=None: latidos.append(v))
    monkeypatch.setattr(det.GitHub, "escribir_info", lambda self, info, nombre=None: infos.append(info))
    monkeypatch.setattr(det.GitHub, "ultimo_procesado", lambda self: None)
    monkeypatch.setattr(det.GitHub, "activos", lambda self, *a, **k: 0)
    monkeypatch.setattr(det.GitHub, "despachar", lambda self, *a, **k: None)
    monkeypatch.setattr(det.GitHub, "clave_vence", None, raising=False)

    def sin_navegador(timeout=240):
        raise RuntimeError("BrowserType.launch: Executable doesn't exist")

    monkeypatch.setattr(det, "consultar_sinarame", sin_navegador)
    monkeypatch.setattr(det, "ERRORES_SIN_LATIDO", 3)
    monkeypatch.setattr(det, "LATIDO_CADA_SEG", 0.0)
    monkeypatch.setattr(det.time, "sleep", lambda s: None)
    reloj = {"t": 0.0}

    def monotonic():
        reloj["t"] += 1.0
        return reloj["t"]

    monkeypatch.setattr(det.time, "monotonic", monotonic)
    monkeypatch.setenv("GITHUB_REPOSITORY", "x/y")
    monkeypatch.setenv("GH_TOKEN", "t")
    monkeypatch.chdir(tmp_path)
    det.main(["--loop", "--latido", "--intervalo-seg", "1", "--duracion-max-min", "0.3"])
    # Latido solo en las primeras consultas; después, ninguno.
    assert 1 <= len(latidos) < 3
    errores = [i for i in infos if i.get("error_consultas")]
    assert len(errores) == 1
    assert "Executable doesn't exist" in errores[0]["error_consultas"]["ultimo_error"]


def test_vigilancia_explica_la_causa_y_el_arreglo():
    ahora = datetime(2026, 11, 24, 0, 30, tzinfo=timezone.utc)
    info = {"utc": ahora.isoformat(), "clave_vence": "2027-02-21 03:00:00 UTC",
            "error_consultas": {"consecutivos": 20, "ultimo_error": "Executable doesn't exist"}}
    p = vg.revisar(ahora, (ahora - timedelta(minutes=12)).isoformat(), info, None,
                   [{"status": "completed", "conclusion": "success"}], [])
    texto = next(x["texto"] for x in p if x["clave"] == "latido")
    assert "no puede consultar SINARAME" in texto and "playwright install chromium" in texto
