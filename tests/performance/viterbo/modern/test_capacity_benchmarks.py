"""Performance smoke for modern capacity (4D small instance)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo.modern import capacity, polytopes


@pytest.mark.goal_performance
def test_modern_capacity_reference_benchmark_4d(benchmark) -> None:
    """Benchmark modern facet-normal reference on a 4D simplex."""
    verts = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    P = polytopes.build_from_vertices(verts)
    result = benchmark(lambda: capacity.ehz_capacity_reference(P))
    assert jnp.isfinite(result)

