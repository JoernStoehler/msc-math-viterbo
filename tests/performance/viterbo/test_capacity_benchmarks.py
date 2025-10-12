"""Performance smoke for modern capacity (4D small instance)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo.math.capacity.facet_normals import (
    ehz_capacity_reference_facet_normals,
)
from viterbo.datasets.builders import build_from_vertices

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
    P = build_from_vertices(verts)
    result = benchmark(lambda: ehz_capacity_reference_facet_normals(P.normals, P.offsets))
    assert jnp.isfinite(result)
