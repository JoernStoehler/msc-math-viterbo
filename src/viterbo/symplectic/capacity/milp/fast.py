"""Heuristic MILP solver for the EHZ capacity using partial branching."""

from __future__ import annotations

from typing import Mapping

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.symplectic.capacity.milp.model import (
    MilpCapacityResult,
    MilpCertificate,
    build_certificate,
    build_subset_model,
    solve_subset_model,
)


def compute_ehz_capacity_fast(
    B_matrix: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    tol: float = 1e-10,
    highs_options: Mapping[str, float] | None = None,
    node_limit: int = 8192,
) -> MilpCapacityResult:
    """Search for good MILP certificates via a depth-first subset exploration."""

    B = jnp.asarray(B_matrix, dtype=jnp.float64)
    offsets = jnp.asarray(c, dtype=jnp.float64)

    if B.ndim != 2:
        raise ValueError("Facet matrix B must be two-dimensional.")

    if offsets.ndim != 1 or offsets.shape[0] != B.shape[0]:
        raise ValueError("Vector c must have length equal to the number of facets.")

    num_facets, dimension = B.shape
    if int(dimension) % 2 != 0 or int(dimension) < 2:
        raise ValueError("The ambient dimension must satisfy 2n with n >= 1.")

    subset_size = int(dimension) + 1
    if subset_size > int(num_facets):
        raise ValueError("Not enough facets to form a candidate subset.")

    priorities = jnp.argsort(-jnp.abs(offsets))
    ordered_facets: tuple[int, ...] = tuple(int(idx) for idx in priorities)

    stack: list[tuple[tuple[int, ...], int]] = [(tuple(), 0)]
    expanded_nodes = 0
    explored = 0
    best_certificate: MilpCertificate | None = None

    while stack:
        prefix, start = stack.pop()
        remaining = subset_size - len(prefix)

        if remaining == 0:
            subset_indices = tuple(sorted(ordered_facets[position] for position in prefix))
            model = build_subset_model(B_matrix=B, c=offsets, indices=subset_indices)
            solution = solve_subset_model(model, options=highs_options)
            if solution is None:
                continue

            certificate = build_certificate(model=model, solution=solution, tol=float(tol))
            explored += 1
            if certificate is None:
                continue

            if best_certificate is None or certificate.capacity < best_certificate.capacity:
                best_certificate = certificate
            continue

        if node_limit is not None and expanded_nodes >= node_limit and best_certificate is not None:
            continue

        max_start = int(num_facets) - remaining
        for position in range(max_start, start - 1, -1):
            new_prefix = prefix + (position,)
            stack.append((new_prefix, position + 1))
            expanded_nodes += 1
            if node_limit is not None and expanded_nodes >= node_limit:
                break

    if best_certificate is None:
        raise ValueError("No admissible facet subset satisfied the MILP constraints.")

    return MilpCapacityResult(
        upper_bound=float(best_certificate.capacity),
        lower_bound=None,
        certificate=best_certificate,
        explored_subsets=explored,
    )
