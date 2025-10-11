"""Regression baselines for small 2D polytopes (area == capacity)."""

from __future__ import annotations

import json
import os

import jax.numpy as jnp
import pytest

from viterbo.modern import polytopes, capacity, volume


@pytest.mark.goal_math
@pytest.mark.smoke
def test_modern_2d_baselines() -> None:
    """Compare area and capacity to curated 2D baselines."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "_baselines", "modern_2d.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for name, entry in data.items():
        verts = jnp.asarray(entry["vertices"], dtype=jnp.float64)
        P = polytopes.build_from_vertices(verts)
        c = capacity.ehz_capacity_reference(P)
        a = volume.volume_reference(P)
        assert jnp.isclose(c, jnp.asarray(entry["capacity_ehz"], dtype=jnp.float64), rtol=1e-12, atol=0.0), name
        assert jnp.isclose(a, jnp.asarray(entry["area"], dtype=jnp.float64), rtol=1e-12, atol=0.0), name

