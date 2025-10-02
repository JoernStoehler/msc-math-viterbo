"""Shared fixtures for generating representative polytope instances for tests.

We centralize the catalog sampling logic so both correctness and performance
suites iterate over the exact same data. This avoids accidental drift between
benchmark inputs and the regression tests that guard correctness.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np

from viterbo.polytopes import catalog, random_transformations


def load_polytope_instances(
    *,
    rng_seed: int = 2023,
    variant_count: int = 3,
) -> tuple[Sequence[tuple[np.ndarray, np.ndarray]], Sequence[str]]:
    """Return a deterministic list of polytopes plus readable identifiers.

    The optimized and reference implementations are sensitive to both the
    polytope geometry and the affine transformations we apply. Supplying a
    fixed RNG seed keeps tests reproducible while allowing us to reuse the
    sample set across correctness, benchmarking, and profiling runs.
    """

    rng = np.random.default_rng(rng_seed)
    instances: list[tuple[np.ndarray, np.ndarray]] = []
    identifiers: list[str] = []

    for polytope in catalog():
        B, c = polytope.halfspace_data()
        instances.append((B, c))
        identifiers.append(polytope.name)

        variants = random_transformations(polytope, rng=rng, count=variant_count)
        for index, (variant_B, variant_c) in enumerate(variants):
            instances.append((variant_B, variant_c))
            identifiers.append(f"{polytope.name}-variant-{index}")

    return instances, identifiers


__all__: Final = ["load_polytope_instances"]
