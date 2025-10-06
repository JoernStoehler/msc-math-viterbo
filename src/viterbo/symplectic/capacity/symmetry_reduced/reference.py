"""Reference symmetry-reduced EHZ capacity computation (JAX-first)."""

from __future__ import annotations

from typing import Any, Iterable, Sequence, cast

import cvxpy as cp
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from viterbo.symplectic.capacity.facet_normals.reference import (
    compute_ehz_capacity_reference as compute_ehz_capacity_reference_full,
)
from viterbo.symplectic.capacity.facet_normals.subset_utils import (
    FacetSubset,
    iter_index_combinations,
    prepare_subset,
    subset_capacity_candidate_dynamic,
)
from viterbo.symplectic.capacity.symmetry_reduced.pairs import (
    FacetPairingMetadata,
    detect_opposite_facet_pairs,
)
from viterbo.symplectic.core import standard_symplectic_matrix


def compute_ehz_capacity_reference_symmetry_reduced(
    B_matrix: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    tol: float = 1e-10,
    pairing: FacetPairingMetadata | None = None,
    enforce_detection: bool = True,
) -> float:
    """Compute the EHZ capacity while enforcing opposite-facet symmetries.

    Args:
      B_matrix: Outward facet normals for ``P = {x : Bx <= c}``.
      c: Offsets ``c``.
      tol: Numerical tolerance for feasibility checks.
      pairing: Optional precomputed symmetry metadata. When omitted the
        heuristics from :func:`detect_opposite_facet_pairs` are used.
      enforce_detection: When ``True`` the solver falls back to the
        full reference implementation if no opposite pairs are detected.

    Returns:
      The EHZ capacity of ``P`` under the symmetry constraints.

    Notes:
      The solver mirrors :func:`compute_ehz_capacity_reference_full` but skips
      facet subsets that are symmetric duplicates and re-solves the Reeb-measure
      system with equality constraints ``β_i = β_j`` for detected facet pairs.
    """
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
        return compute_ehz_capacity_reference_full(B, offsets, tol=tol)

    J = standard_symplectic_matrix(dimension)
    subset_size = dimension + 1
    best_capacity = jnp.inf

    cache = _GroupCache(metadata=metadata)

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
            groups=cache.group_for_indices(indices),
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


class _GroupCache:
    """Cache per-subset equality groups derived from symmetry metadata."""

    def __init__(self, *, metadata: FacetPairingMetadata) -> None:
        self._metadata = metadata
        self._cache: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] = {}

    def group_for_indices(self, indices: Iterable[int]) -> tuple[tuple[int, ...], ...]:
        key = tuple(int(i) for i in indices)
        if key not in self._cache:
            self._cache[key] = self._metadata.subset_groups(key)
        return self._cache[key]


# Public aliases for use in other modules without triggering private-usage checks.
GroupCache = _GroupCache


def _enforce_pair_constraints(
    *,
    subset: FacetSubset,
    B_matrix: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    groups: Sequence[Sequence[int]],
    tol: float,
) -> FacetSubset | None:
    """Solve for ``β`` under equality constraints for opposite facets."""
    if not groups:
        return subset

    indices = tuple(int(i) for i in subset.indices)
    if len(groups) == len(indices):
        return subset

    B_np = np.asarray(B_matrix, dtype=np.float64)
    c_np = np.asarray(c, dtype=np.float64)
    B_subset = B_np[np.asarray(indices)]
    c_subset = c_np[np.asarray(indices)]

    index_to_position = {index: pos for pos, index in enumerate(indices)}
    positions_groups: list[tuple[int, ...]] = []
    aggregated_normals: list[np.ndarray] = []
    aggregated_offsets: list[float] = []

    for group in groups:
        if not group:
            continue
        positions = tuple(index_to_position[int(idx)] for idx in group)
        normals = B_subset[np.asarray(positions)]
        offsets = c_subset[np.asarray(positions)]
        aggregated_normals.append(np.sum(normals, axis=0))
        aggregated_offsets.append(float(np.sum(offsets)))
        positions_groups.append(positions)

    aggregated_normals_np = np.stack(aggregated_normals, axis=0)
    aggregated_offsets_np = np.asarray(aggregated_offsets, dtype=np.float64)

    A = np.vstack(
        [
            aggregated_offsets_np[np.newaxis, :],
            aggregated_normals_np.T,
        ]
    )
    rhs = np.zeros(A.shape[0], dtype=np.float64)
    rhs[0] = 1.0

    variable = cp.Variable(len(positions_groups))
    constraints: list[cp.Constraint] = [A @ variable == rhs, variable >= 0]
    problem = cast(Any, cp.Problem(cp.Minimize(0), constraints))
    try:
        problem.solve()
    except cp.SolverError:
        return None

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        return None

    solution_value = variable.value
    if solution_value is None:
        return None

    solution = np.asarray(solution_value, dtype=np.float64).reshape(-1)
    solution = np.where(np.abs(solution) <= float(tol), 0.0, solution)
    if np.any(solution < -float(tol)):
        return None

    beta = np.zeros_like(np.asarray(subset.beta))
    for group_positions, value in zip(positions_groups, solution):
        for position in group_positions:
            beta[position] = value

    if not np.allclose(B_subset.T @ beta, np.zeros(B_subset.shape[1]), atol=float(tol), rtol=0.0):
        return None

    if not np.isclose(float(c_subset @ beta), 1.0, atol=float(tol), rtol=0.0):
        return None

    return FacetSubset(
        indices=indices,
        beta=jnp.asarray(beta, dtype=jnp.float64),
        symplectic_products=subset.symplectic_products,
    )


def enforce_pair_constraints(
    *,
    subset: FacetSubset,
    B_matrix: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    groups: Sequence[Sequence[int]],
    tol: float,
) -> FacetSubset | None:
    """Public wrapper for enforcing symmetry pair constraints."""
    return _enforce_pair_constraints(subset=subset, B_matrix=B_matrix, c=c, groups=groups, tol=tol)
