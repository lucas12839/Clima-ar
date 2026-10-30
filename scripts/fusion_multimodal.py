"""ClimaAR multimodal evidence fusion V1 (laboratory).

This module deliberately does not replace Storm Engine logic. It creates a
single evidence score from already-produced radar/RMA10, environmental,
convective and GOES signals, while preserving source freshness and uncertainty.
No output is called a calibrated probability unless the source explicitly says
it is calibrated.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Any


def _num(v: Any, default: float | None = None) -> float | None:
    try:
        x = float(v)
        return x if isfinite(x) else default
    except (TypeError, ValueError):
        return default


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, float(x)))
    return 1.0 / (1.0 + exp(-x))


def freshness_weight(age_min: float | None, fresh: float = 20.0, stale: float = 90.0) -> float:
    if age_min is None:
        return 0.0
    if age_min <= fresh:
        return 1.0
    if age_min >= stale:
        return 0.0
    return clamp((stale - age_min) / (stale - fresh))


def radar_intensity_score(radar: dict[str, Any]) -> float | None:
    """Map validated dBZ summary to an evidence score, not a hazard probability."""
    dbz = radar.get("dbz") if isinstance(radar, dict) else None
    if not isinstance(dbz, dict):
        return None
    p95 = _num(dbz.get("p95"))
    p99 = _num(dbz.get("p99"))
    if p95 is None and p99 is None:
        return None
    x = p99 if p99 is not None else p95
    # 35 dBZ = meaningful convection; 60+ dBZ = very strong core evidence.
    return clamp((x - 30.0) / 35.0)


def radar_evolution_score(prediction: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    """Return (strengthening, weakening, toward-Bahia) from calibrated/raw RMA10 outputs.

    Values are source probabilities when present. The caller must preserve the
    calibration metadata rather than relabeling them as globally calibrated.
    """
    preds = prediction.get("predicciones") if isinstance(prediction, dict) else None
    if not isinstance(preds, list) or not preds:
        return None, None, None
    best = None
    # Prefer the closest track to Bahia; use distance as the selection criterion.
    for item in preds:
        d = _num(item.get("distance_km"), 1e9)
        if best is None or d < best[0]:
            best = (d, item)
    if best is None:
        return None, None, None
    horizons = best[1].get("horizontes") or {}
    candidates = []
    for h in ("5", "10", "20", "30"):
        row = horizons.get(h)
        if isinstance(row, dict):
            candidates.append(row)
    if not candidates:
        return None, None, None
    def avg(key: str) -> float | None:
        vals = [_num(x.get(key)) for x in candidates]
        vals = [x for x in vals if x is not None]
        return sum(vals) / len(vals) if vals else None
    return avg("strengthened"), avg("weakened"), avg("toward_bahia")


def convective_context(conv: dict[str, Any]) -> tuple[float | None, float]:
    """Return a conservative convective context score and confidence.

    The current convective model has only two event days and weak validation, so
    this is intentionally a low-confidence modifier, never a standalone alert.
    """
    if not isinstance(conv, dict):
        return None, 0.0
    cape = _num(conv.get("cape"))
    li = _num(conv.get("lifted_index"))
    cin = _num(conv.get("convective_inhibition"))
    if cape is None and li is None and cin is None:
        return None, 0.0
    s = 0.0
    n = 0
    if cape is not None:
        s += clamp(cape / 2500.0); n += 1
    if li is not None:
        s += clamp((2.0 - li) / 8.0); n += 1
    if cin is not None:
        s += clamp((-cin) / 250.0); n += 1
    return clamp(s / max(n, 1)), 0.30


def fuse(inputs: dict[str, Any]) -> dict[str, Any]:
    radar = inputs.get("radar") or {}
    rma = inputs.get("rma10") or {}
    env = inputs.get("environment") or {}
    conv = inputs.get("convective") or {}
    goes = inputs.get("goes") or {}

    age = _num(radar.get("radar_age"))
    w_radar = freshness_weight(age)
    intensity = radar_intensity_score(radar.get("nucleo_principal") or radar)
    strength, weaken, toward = radar_evolution_score(rma)
    conv_score, conv_conf = convective_context(conv)
    env_impact = _num(env.get("impacto"))
    env_intens = _num(env.get("intensificacion"))
    env_form = _num(env.get("formacion"))
    goes_conv = _num(goes.get("goes19_conveccion"))
    goes_light = _num(goes.get("goes19_rayo"))

    # Radar receives the highest weight because ClimaAR is a nowcasting system.
    evidence = []
    weights = []
    def add(v, w):
        if v is not None:
            evidence.append(clamp(v)); weights.append(float(w))
    add(toward, 0.20 * w_radar)
    add(strength, 0.22 * w_radar)
    add(intensity, 0.18 * w_radar)
    add(env_impact, 0.18)
    add(env_intens, 0.08)
    add(env_form, 0.05)
    add(goes_conv, 0.06)
    add(goes_light, 0.03)
    if conv_score is not None:
        add(conv_score, 0.05 * conv_conf)

    base = sum(v*w for v,w in zip(evidence,weights)) / sum(weights) if weights else 0.0
    disagreement = 0.0
    if len(evidence) >= 2:
        mean = sum(evidence)/len(evidence)
        disagreement = sum(abs(x-mean) for x in evidence)/len(evidence)
    source_count = len(evidence)
    confidence = clamp(
        0.25
        + 0.10 * min(source_count, 5)
        + 0.25 * w_radar
        - 0.30 * disagreement
        - 0.15 * (1.0 - conv_conf if conv_score is not None else 0.0)
    )
    return {
        "version": "multimodal-fusion-v1-lab",
        "evidence_score": round(clamp(base), 4),
        "confidence": round(confidence, 4),
        "radar_freshness_weight": round(w_radar, 4),
        "source_count": source_count,
        "source_disagreement": round(disagreement, 4),
        "components": {
            "radar_intensity": intensity,
            "radar_strengthening": strength,
            "radar_weakening": weaken,
            "radar_toward_bahia": toward,
            "environment_impact_score": env_impact,
            "environment_intensification_score": env_intens,
            "environment_formation_score": env_form,
            "convective_context": conv_score,
            "goes_convection": goes_conv,
            "goes_lightning": goes_light,
        },
        "semantics": {
            "evidence_score_is_calibrated_probability": False,
            "convective_branch_is_primary": False,
            "goes_is_independent_alert": False,
            "radar_age_controls_radar_weight": True,
        },
    }
