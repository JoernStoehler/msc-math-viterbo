"""Shared fixtures for generating representative polytope instances for tests.

Minimal replacement used by performance suites to avoid importing the
deprecated viterbo.datasets. Provides a few small 4D instances.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import jax.numpy as jnp
import numpy as np

from viterbo.datasets2 import generators
from viterbo.math import volume


def _as_numpy_pair(sample: generators.PolytopeSample) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(sample.normals), np.asarray(sample.offsets)


def load_polytope_instances(
    *,
    rng_seed: int = 0,
    variant_count: int = 0,
    include_metadata: bool = False,
) -> (
    tuple[Sequence[tuple[np.ndarray, np.ndarray]], Sequence[str]]
    | tuple[
        Sequence[tuple[np.ndarray, np.ndarray]], Sequence[str], Sequence[dict[str, float | None]]
    ]
):
    """Return a small deterministic list of 4D polytopes plus readable identifiers."""

    instances: list[tuple[np.ndarray, np.ndarray]] = []
    identifiers: list[str] = []
    metadata: list[dict[str, float | None]] = []

    # Right 4D simplex conv(0,2e_i)
    simp = generators.simplex(4)
    instances.append(_as_numpy_pair(simp))
    identifiers.append("simplex_4d")
    if include_metadata:
        B, c = jnp.asarray(simp.normals), jnp.asarray(simp.offsets)
        metadata.append(
            {
                "reference_volume": float(volume.polytope_volume_reference(B, c)),
                "reference_capacity": None,
            }
        )

    # Hypercube [-1,1]^4
    cube = generators.hypercube(4, radius=1.0)
    instances.append(_as_numpy_pair(cube))
    identifiers.append("hypercube_4d")
    if include_metadata:
        B, c = jnp.asarray(cube.normals), jnp.asarray(cube.offsets)
        metadata.append(
            {
                "reference_volume": float(volume.polytope_volume_reference(B, c)),
                "reference_capacity": None,
            }
        )

    if include_metadata:
        return instances, identifiers, metadata
    return instances, identifiers


__all__: Final = ["load_polytope_instances"]
