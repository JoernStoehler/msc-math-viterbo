"""Performance smoke for modern capacity (4D small instance)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo import capacity, polytopes

pytestmark = [pytest.mark.smoke]


@pytest.mark.goal_performance
@pytest.mark.smoke
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
    result = benchmark(lambda: capacity.ehz_capacity_reference(P.normals, P.offsets, P.vertices))
    assert jnp.isfinite(result)
