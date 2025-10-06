"""Benchmarks for the volume estimators."""

from __future__ import annotations

from typing import cast

import jax
import pytest
import pytest_benchmark.plugin  # type: ignore[reportMissingTypeStubs]

from viterbo.geometry.polytopes import random_polytope
from viterbo.geometry.volume import polytope_volume_fast, polytope_volume_reference

pytestmark = [pytest.mark.smoke, pytest.mark.deep]


@pytest.mark.benchmark
def test_volume_fast_matches_reference(
    benchmark: pytest_benchmark.plugin.BenchmarkFixture,
) -> None:
    key = jax.random.PRNGKey(314)
    polytope = random_polytope(4, key=key, name="volume-bench")
    B, c = polytope.halfspace_data()

    baseline = polytope_volume_reference(B, c)

    def _run() -> float:
        return polytope_volume_fast(B, c)

    result = cast(float, benchmark(_run))
    assert pytest.approx(baseline, rel=1e-9) == result  # type: ignore[reportUnknownMemberType]
