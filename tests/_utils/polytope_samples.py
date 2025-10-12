"""Shared fixtures for generating representative polytope instances for tests.

We centralize the catalog sampling logic so both correctness and performance
suites iterate over the exact same data. This avoids accidental drift between
benchmark inputs and the regression tests that guard correctness.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import jax
import jax.numpy as jnp

from viterbo.geom import catalog, random_transformations, polytope_volume_reference


def load_polytope_instances(
    *,
    rng_seed: int = 2023,
    variant_count: int = 3,
    include_metadata: bool = False,
) -> (
    tuple[
        Sequence[tuple[jnp.ndarray, jnp.ndarray]],
        Sequence[str],
    ]
    | tuple[
        Sequence[tuple[jnp.ndarray, jnp.ndarray]],
        Sequence[str],
        Sequence[dict[str, float | None]],
    ]
):
    """Return a deterministic list of polytopes plus readable identifiers.

    The optimized and reference implementations are sensitive to both the
    polytope geometry and the affine transformations we apply. Supplying a
    fixed RNG seed keeps tests reproducible while allowing us to reuse the
    sample set across correctness, benchmarking, and profiling runs.
    """

    key = jax.random.PRNGKey(rng_seed)
    instances: list[tuple[jnp.ndarray, jnp.ndarray]] = []
    identifiers: list[str] = []
    metadata: list[dict[str, float | None]] = []

    for polytope in catalog():
        geometry = polytope.geometry
        B, c = geometry.halfspace_data()
        instances.append((B, c))
        identifiers.append(polytope.metadata.slug)
        if include_metadata:
            metadata.append(
                {
                    "reference_volume": polytope_volume_reference(B, c),
                    "reference_capacity": polytope.metadata.reference_capacity,
                }
            )

        variants = random_transformations(polytope, key=key, count=variant_count)
        for index, variant in enumerate(variants):
            variant_B, variant_c = variant.geometry.halfspace_data()
            instances.append((variant_B, variant_c))
            identifiers.append(f"{polytope.metadata.slug}-variant-{index}")
            if include_metadata:
                metadata.append(
                    {
                        "reference_volume": polytope_volume_reference(variant_B, variant_c),
                        "reference_capacity": polytope.metadata.reference_capacity,
                    }
                )
    if include_metadata:
        return instances, identifiers, metadata
    return instances, identifiers


__all__: Final = ["load_polytope_instances"]
