"""Benchmarks for the volume estimators."""

from __future__ import annotations

import numpy as np
import pytest
import pytest_benchmark.plugin

from viterbo import random_polytope
from viterbo.volume import polytope_volume_fast, polytope_volume_reference


@pytest.mark.benchmark
def test_volume_fast_matches_reference(
    benchmark: pytest_benchmark.plugin.BenchmarkFixture,
) -> None:
    rng = np.random.default_rng(314)
    polytope = random_polytope(4, rng=rng, name="volume-bench")
    B, c = polytope.halfspace_data()

    baseline = polytope_volume_reference(B, c)

    def _run() -> float:
        return polytope_volume_fast(B, c)

    result = benchmark(_run)
    assert pytest.approx(baseline, rel=1e-9) == result
