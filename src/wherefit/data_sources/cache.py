"""Small CSV cache helpers for external data."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def safe_cache_name(*parts: object, suffix: str = ".csv") -> str:
    raw = "_".join(str(part) for part in parts if str(part))
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    return f"{safe}{suffix}"


def read_csv_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def write_csv_cache(path: Path, data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)

