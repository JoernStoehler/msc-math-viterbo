"""Optimized symmetry-reduced EHZ capacity computation (JAX-first)."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.symplectic.capacity.facet_normals.fast import (
    compute_ehz_capacity_fast as compute_ehz_capacity_fast_full,
)
from viterbo.symplectic.capacity.facet_normals.subset_utils import (
    iter_index_combinations,
    prepare_subset,
    subset_capacity_candidate_dynamic,
)
from viterbo.symplectic.capacity.symmetry_reduced.pairs import (
    FacetPairingMetadata,
    detect_opposite_facet_pairs,
)
from viterbo.symplectic.capacity.symmetry_reduced.reference import (
    _enforce_pair_constraints,
    _GroupCache,
)
from viterbo.symplectic.core import standard_symplectic_matrix


def compute_ehz_capacity_fast_symmetry_reduced(
    B_matrix: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    tol: float = 1e-10,
    pairing: FacetPairingMetadata | None = None,
    enforce_detection: bool = True,
) -> float:
    """Compute the EHZ capacity via symmetry-reduced dynamic programming."""
    B = jnp.asarray(B_matrix, dtype=jnp.float64)
    offsets = jnp.asarray(c, dtype=jnp.float64)

    if B.ndim != 2:
        raise ValueError("Facet matrix B must be two-dimensional.")

    if offsets.ndim != 1 or offsets.shape[0] != B.shape[0]:
        raise ValueError("Vector c must have length equal to the number of facets.")

    num_facets, dimension = B.shape
    if int(dimension) % 2 != 0 or int(dimension) < 2:
        raise ValueError("The ambient dimension must satisfy 2n with n >= 1.")

    metadata = pairing or detect_opposite_facet_pairs(B, offsets)
    if enforce_detection and not metadata.pairs:
        return compute_ehz_capacity_fast_full(B, offsets, tol=tol)

    J = standard_symplectic_matrix(dimension)
    subset_size = dimension + 1
    best_capacity = jnp.inf

    group_cache = _GroupCache(metadata=metadata)

    for indices in iter_index_combinations(num_facets, subset_size):
        if not metadata.is_canonical_subset(indices):
            continue

        subset = prepare_subset(B_matrix=B, c=offsets, indices=indices, J=J, tol=tol)
        if subset is None:
            continue

        constrained = _enforce_pair_constraints(
            subset=subset,
            B_matrix=B,
            c=offsets,
            groups=group_cache.group_for_indices(indices),
            tol=tol,
        )
        if constrained is None:
            continue

        candidate_value = subset_capacity_candidate_dynamic(constrained, tol=tol)
        if candidate_value is None:
            continue

        if candidate_value < best_capacity:
            best_capacity = candidate_value

    if not bool(jnp.isfinite(best_capacity)):
        raise ValueError("No admissible facet subset satisfied the symmetry constraints.")

    return float(best_capacity)
