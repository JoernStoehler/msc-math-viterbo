"""Performance-focused tests for the EHZ capacity implementations.

These tests live alongside the correctness suite to keep benchmarking
ergonomic for contributors. The additional comments explain how the pytest
plugins compose so future maintainers can extend the pattern to other modules.
"""

from __future__ import annotations

import numpy as np
import pytest
import pytest_benchmark.plugin

from viterbo import compute_ehz_capacity
from viterbo.ehz_fast import compute_ehz_capacity_fast

from .._polytope_samples import load_polytope_instances

# Reuse the exact same catalog of polytopes as the regression tests so that any
# deviation caught here points to a performance-only issue rather than a change
# in sampling. The helper also carries documentation about why the instances are
# structured the way they are. We keep line-profiler opt-in via CLI (`make
# profile-line`) to avoid noisy output during normal pytest runs.
_POLYTOPE_DATA = load_polytope_instances()
_POLYTOPE_INSTANCES = _POLYTOPE_DATA[0]
_POLYTOPE_IDS = _POLYTOPE_DATA[1]


@pytest.mark.benchmark(group="ehz_capacity")
@pytest.mark.parametrize(("B", "c"), _POLYTOPE_INSTANCES, ids=_POLYTOPE_IDS)
def test_fast_ehz_capacity_matches_reference_and_tracks_speed(
    benchmark: pytest_benchmark.plugin.BenchmarkFixture,
    B: np.ndarray,
    c: np.ndarray,
) -> None:
    """Benchmark the optimized kernel and assert it still agrees with reference.

    The ``benchmark`` fixture repeatedly calls the optimized implementation and
    returns the last result so we can validate correctness inline. Pytest's
    normal assertion rewriting remains active, which keeps this single source
    of truth for both performance and accuracy checks.
    """

    try:
        reference = compute_ehz_capacity(B, c)
    except ValueError as error:
        # When the reference rejects an instance (e.g. infeasible constraints)
        # we still run the optimized variant through the benchmark harness so
        # the failure is visible in timing reports. Pytest-benchmark re-raises
        # the underlying exception, letting us assert on parity of the message.
        with pytest.raises(ValueError) as caught:
            benchmark(lambda: compute_ehz_capacity_fast(B, c))
        assert str(caught.value) == str(error)
    else:
        optimized = benchmark(lambda: compute_ehz_capacity_fast(B, c))
        assert np.isclose(optimized, reference, atol=1e-8)
