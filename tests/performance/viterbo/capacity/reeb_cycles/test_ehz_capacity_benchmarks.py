"""Performance benchmarks for the combinatorial Reeb cycle solvers."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import pytest_benchmark.plugin  # type: ignore[reportMissingTypeStubs]
import jax.numpy as jnp
from tests._utils.polytope_samples import load_polytope_instances

from viterbo.math.capacity.reeb_cycles import ehz_capacity_fast_reeb, ehz_capacity_reference_reeb

pytestmark = [pytest.mark.smoke, pytest.mark.deep]

_POLYTOPES = load_polytope_instances()
_POLYTOPE_INSTANCES = _POLYTOPES[0]
_POLYTOPE_IDS = _POLYTOPES[1]


@pytest.mark.goal_performance
@pytest.mark.smoke
@pytest.mark.benchmark(group="reeb-cycles")
@pytest.mark.parametrize(("B", "c"), _POLYTOPE_INSTANCES, ids=_POLYTOPE_IDS)
def test_fast_matches_reference(
    benchmark: pytest_benchmark.plugin.BenchmarkFixture,
    B: np.ndarray,
    c: np.ndarray,
) -> None:
    """Benchmark the optimized solver against the reference implementation."""

    normals = jnp.asarray(B, dtype=jnp.float64)
    offsets = jnp.asarray(c, dtype=jnp.float64)

    try:
        reference = ehz_capacity_reference_reeb(normals, offsets)
    except ValueError as error:
        with pytest.raises(ValueError) as caught:
            benchmark(lambda: ehz_capacity_fast_reeb(normals, offsets))
        assert str(caught.value) == str(error)
    else:
        optimized = cast(float, benchmark(lambda: ehz_capacity_fast_reeb(normals, offsets)))
        assert np.isclose(optimized, reference, atol=1e-8)
