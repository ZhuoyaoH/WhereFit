"""Preference and score normalization helpers."""

from __future__ import annotations

from wherefit.config import MODE_WEIGHTS


def normalize_score(value: float | int | None) -> float:
    if value is None:
        return 50.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 50.0
    if number < 0:
        return 0.0
    if number > 100:
        return 100.0
    return round(number, 1)


def mode_weights(mode: str) -> dict[str, float]:
    return MODE_WEIGHTS.get(mode, MODE_WEIGHTS["Compare"])
