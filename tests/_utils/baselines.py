"""Helpers for loading regression baselines."""

from __future__ import annotations

import json
from pathlib import Path

_BASELINE_ROOT = Path(__file__).resolve().parent.parent / "_baselines"


def load_baseline(name: str) -> object:
    """Return the parsed JSON baseline stored under ``tests/_baselines``.

    Args:
      name: Baseline stem without the ``.json`` suffix.

    Returns:
      Parsed JSON content; callers are expected to know the schema.
    """

    path = _BASELINE_ROOT / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Baseline '{name}' not found at {path}")
    with path.open() as handle:
        return json.load(handle)
