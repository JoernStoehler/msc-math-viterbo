"""Shared MILP model construction utilities for EHZ capacity computations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from viterbo._wrapped.highs import load_highs
from viterbo.symplectic.capacity.facet_normals.subset_utils import FacetSubset
from viterbo.symplectic.core import standard_symplectic_matrix


@dataclass(frozen=True)
class SubsetMilpModel:
    """Dense MILP representation for a fixed facet subset."""

    indices: tuple[int, ...]
    facet_normals: Float[Array, " subset_size dimension"]
    support_numbers: Float[Array, " subset_size"]

    @property
    def subset_size(self) -> int:
        return int(self.facet_normals.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.facet_normals.shape[1])


@dataclass(frozen=True)
class SubsetMilpSolution:
    """Solution data returned by the HiGHS solve."""

    beta: Float[Array, " subset_size"]
    objective_value: float
    status: str


@dataclass(frozen=True)
class MilpCertificate:
    """Certificate describing an admissible facet subset and action value."""

    subset_indices: tuple[int, ...]
    beta: Float[Array, " subset_size"]
    symplectic_products: Float[Array, " subset_size subset_size"]
    capacity: float

    @property
    def subset_size(self) -> int:
        return int(self.beta.shape[0])


@dataclass(frozen=True)
class MilpCapacityResult:
    """Result of a MILP-based capacity computation."""

    upper_bound: float
    lower_bound: float | None
    certificate: MilpCertificate
    explored_subsets: int


def build_subset_model(
    *,
    B_matrix: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    indices: Iterable[int],
) -> SubsetMilpModel:
    """Return the MILP model representing the Reeb constraints for ``indices``."""

    subset = tuple(int(i) for i in indices)
    rows = jnp.asarray(subset, dtype=int)
    facet_normals = jnp.asarray(B_matrix)[rows, :]
    support = jnp.asarray(c)[rows]
    return SubsetMilpModel(indices=subset, facet_normals=facet_normals, support_numbers=support)


def solve_subset_model(
    model: SubsetMilpModel,
    *,
    options: Mapping[str, float] | None = None,
) -> SubsetMilpSolution | None:
    """Solve the MILP corresponding to ``model`` using HiGHS."""

    resources = load_highs()
    highs = resources.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("random_seed", 0)

    if options is not None:
        for key, value in options.items():
            highs.setOptionValue(str(key), float(value))

    infinity = highs.getInfinity()
    subset_size = model.subset_size
    variables = highs.addVariables(
        range(subset_size),
        lb=[0.0] * subset_size,
        ub=[infinity] * subset_size,
        obj=[0.0] * subset_size,
        type=[resources.HighsVarType.kContinuous] * subset_size,
    )

    normals = np.asarray(model.facet_normals, dtype=np.float64)
    support = np.asarray(model.support_numbers, dtype=np.float64)

    # Equality constraints Σ β_i n_i = 0.
    for row in normals.T:
        expression = highs.expr()
        for column, coefficient in enumerate(row):
            if coefficient == 0.0:
                continue
            expression += float(coefficient) * variables[column]
        highs.addConstr(expression == 0.0)

    # Normalisation Σ β_i c_i = 1.
    expression = highs.expr()
    for column, coefficient in enumerate(support):
        if coefficient == 0.0:
            continue
        expression += float(coefficient) * variables[column]
    highs.addConstr(expression == 1.0)

    status = highs.minimize()
    if status != resources.HighsStatus.kOk:
        return None

    model_status = highs.getModelStatus()
    if model_status != resources.HighsModelStatus.kOptimal:
        return None

    solution = highs.getSolution()
    beta = jnp.asarray(solution.col_value[:subset_size], dtype=jnp.float64)
    objective_value = float(highs.getObjectiveValue())
    status_string = str(highs.modelStatusToString(model_status))

    return SubsetMilpSolution(beta=beta, objective_value=objective_value, status=status_string)


def build_certificate(
    *,
    model: SubsetMilpModel,
    solution: SubsetMilpSolution,
    tol: float,
) -> MilpCertificate | None:
    """Construct a capacity certificate from a solved subset model."""

    beta = jnp.asarray(solution.beta, dtype=jnp.float64)
    beta = jnp.where(jnp.abs(beta) <= float(tol), 0.0, beta)

    if bool(jnp.any(beta < -float(tol))):
        return None

    normals = jnp.asarray(model.facet_normals, dtype=jnp.float64)
    J = standard_symplectic_matrix(model.dimension)
    symplectic_products = (normals @ J) @ normals.T

    subset = FacetSubset(
        indices=model.indices,
        beta=beta,
        symplectic_products=symplectic_products,
    )

    # Evaluate the candidate using the dynamic programming shortcut for robustness.
    from viterbo.symplectic.capacity.facet_normals.subset_utils import (  # local import for cycle
        subset_capacity_candidate_dynamic,
    )

    candidate = subset_capacity_candidate_dynamic(subset, tol=tol)
    if candidate is None:
        return None

    return MilpCertificate(
        subset_indices=model.indices,
        beta=beta,
        symplectic_products=symplectic_products,
        capacity=float(candidate),
    )
