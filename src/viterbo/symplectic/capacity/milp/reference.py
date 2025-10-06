"""Reference MILP computation for the EHZ capacity."""

from __future__ import annotations

from typing import Mapping

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.symplectic.capacity.facet_normals.subset_utils import iter_index_combinations
from viterbo.symplectic.capacity.milp.model import (
    MilpCapacityResult,
    MilpCertificate,
    SubsetMilpModel,
    build_certificate,
    build_subset_model,
    compute_gap_ratio,
    estimate_capacity_lower_bound,
    solve_subset_model,
)


def compute_ehz_capacity_reference(
    B_matrix: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    tol: float = 1e-10,
    highs_options: Mapping[str, float] | None = None,
) -> MilpCapacityResult:
    """Enumerate all facet subsets and return the optimal MILP certificate."""

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
    best_certificate: MilpCertificate | None = None
    explored = 0

    for indices in iter_index_combinations(int(num_facets), subset_size):
        explored += 1
        model: SubsetMilpModel = build_subset_model(B_matrix=B, c=offsets, indices=indices)
        solution = solve_subset_model(model, options=highs_options)
        if solution is None:
            continue

        certificate = build_certificate(model=model, solution=solution, tol=float(tol))
        if certificate is None:
            continue
        if best_certificate is None or certificate.capacity < best_certificate.capacity:
            best_certificate = certificate

    if best_certificate is None:
        raise ValueError("No admissible facet subset satisfied the MILP constraints.")

    lower_bound = estimate_capacity_lower_bound(
        B_matrix=B,
        c=offsets,
        tol=float(tol),
        options=highs_options,
    )

    value = float(best_certificate.capacity)
    lower_value = None
    if lower_bound is not None:
        lower_value = float(min(value, lower_bound))
    return MilpCapacityResult(
        upper_bound=value,
        lower_bound=lower_value,
        certificate=best_certificate,
        explored_subsets=explored,
        gap_ratio=compute_gap_ratio(upper_bound=value, lower_bound=lower_value),
    )
