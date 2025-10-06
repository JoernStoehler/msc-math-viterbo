"""Performance benchmarks for the combinatorial Reeb cycle solvers."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import pytest_benchmark.plugin  # type: ignore[reportMissingTypeStubs]
from tests._utils.polytope_samples import load_polytope_instances

from viterbo.symplectic.capacity.reeb_cycles.fast import compute_ehz_capacity_fast
from viterbo.symplectic.capacity.reeb_cycles.reference import compute_ehz_capacity_reference

pytestmark = [pytest.mark.smoke, pytest.mark.deep]

_POLYTOPES = load_polytope_instances()
_POLYTOPE_INSTANCES = _POLYTOPES[0]
_POLYTOPE_IDS = _POLYTOPES[1]


@pytest.mark.benchmark(group="reeb-cycles")
@pytest.mark.parametrize(("B", "c"), _POLYTOPE_INSTANCES, ids=_POLYTOPE_IDS)
def test_fast_matches_reference(
    benchmark: pytest_benchmark.plugin.BenchmarkFixture,
    B: np.ndarray,
    c: np.ndarray,
) -> None:
    """Benchmark the optimized solver against the reference implementation."""

    try:
        reference = compute_ehz_capacity_reference(B, c)
    except ValueError as error:
        with pytest.raises(ValueError) as caught:
            benchmark(lambda: compute_ehz_capacity_fast(B, c))
        assert str(caught.value) == str(error)
    else:
        optimized = cast(float, benchmark(lambda: compute_ehz_capacity_fast(B, c)))
        assert np.isclose(optimized, reference, atol=1e-8)
