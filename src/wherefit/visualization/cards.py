"""Small presentation helpers for city cards."""

from __future__ import annotations

from wherefit.config import RISK_BANDS, SCORE_BANDS


def score_label(score: float) -> tuple[str, str]:
    for minimum, label, color in SCORE_BANDS:
        if score >= minimum:
            return label, color
    return "不推荐", "#dc2626"


def risk_label(score: float) -> str:
    for minimum, label in RISK_BANDS:
        if score >= minimum:
            return label
    return "低风险"
