"""Regression baselines for small 2D polytopes (area == capacity)."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from viterbo.math import volume


@pytest.mark.goal_math
@pytest.mark.smoke
def test_modern_2d_baselines() -> None:
    """Compare area and capacity to curated 2D baselines."""
    # tests/_baselines/modern_2d.json relative to this file
    path = str(Path(__file__).resolve().parents[1] / "_baselines" / "modern_2d.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for name, entry in data.items():
        verts = jnp.asarray(entry["vertices"], dtype=jnp.float64)
        a = volume.volume_reference(verts)
        c = a
        assert jnp.isclose(
            c, jnp.asarray(entry["capacity_ehz"], dtype=jnp.float64), rtol=1e-12, atol=0.0
        ), name
        assert jnp.isclose(
            a, jnp.asarray(entry["area"], dtype=jnp.float64), rtol=1e-12, atol=0.0
        ), name
