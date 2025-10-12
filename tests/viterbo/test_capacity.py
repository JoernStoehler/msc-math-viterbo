"""EHZ capacity semantics for reference and per-instance APIs."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo.math.capacity.facet_normals import ehz_capacity_reference_facet_normals


@pytest.mark.goal_code
@pytest.mark.smoke
def test_ehz_capacity_reference_manual_batching_pattern() -> None:
    """Manual loops over instances reproduce batching behaviour without a helper."""

    def _square_halfspaces(radius: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        normals = jnp.asarray(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ],
            dtype=jnp.float64,
        )
        offsets = jnp.full((4,), radius, dtype=jnp.float64)
        return normals, offsets

    instances = [_square_halfspaces(1.0), _square_halfspaces(2.0)]
    capacities = [ehz_capacity_reference_facet_normals(B, c) for (B, c) in instances]

    assert capacities[0] == pytest.approx(4.0, rel=1e-12, abs=0.0)
    assert capacities[1] == pytest.approx(16.0, rel=1e-12, abs=0.0)
    # API exposes only per-instance solvers under viterbo.math.capacity


@pytest.mark.goal_math
@pytest.mark.smoke
def test_ehz_capacity_reference_simplex_4d_value() -> None:
    """Reference capacity on right 4D simplex conv(0,2e_i) equals 1.0."""
    vertices = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    from viterbo._wrapped import spatial as _spatial

    eq = _spatial.convex_hull_equations(vertices)
    B = jnp.asarray(eq[:, :-1], dtype=jnp.float64)
    c = jnp.asarray(-eq[:, -1], dtype=jnp.float64)
    value = ehz_capacity_reference_facet_normals(B, c)
    assert value == pytest.approx(1.0, rel=1e-12, abs=0.0)
