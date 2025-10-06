"""Shared MILP model construction utilities for EHZ capacity computations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping, Tuple

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

    subset: FacetSubset
    capacity: float

    @property
    def subset_indices(self) -> tuple[int, ...]:
        return self.subset.indices

    @property
    def beta(self) -> Float[Array, " subset_size"]:
        return self.subset.beta

    @property
    def symplectic_products(self) -> Float[Array, " subset_size subset_size"]:
        return self.subset.symplectic_products

    @property
    def subset_size(self) -> int:
        return int(self.subset.beta.shape[0])


@dataclass(frozen=True)
class MilpCapacityResult:
    """Result of a MILP-based capacity computation."""

    upper_bound: float
    lower_bound: float | None
    certificate: MilpCertificate
    explored_subsets: int
    gap_ratio: float | None


def compute_gap_ratio(*, upper_bound: float, lower_bound: float | None) -> float | None:
    """Return the relative gap ``(upper-lower)/upper`` when finite bounds are available."""

    if lower_bound is None:
        return None
    if upper_bound <= 0.0:
        return None

    gap = max(0.0, upper_bound - max(0.0, lower_bound))
    ratio = gap / upper_bound
    return float(min(1.0, ratio))


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


def _solve_beta_linear_program(
    *,
    objective: np.ndarray,
    normals: np.ndarray,
    support: np.ndarray,
    options: Mapping[str, float] | None,
) -> np.ndarray | None:
    """Solve ``max objective @ beta`` subject to the global Reeb constraints."""

    resources = load_highs()
    highs = resources.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("random_seed", 0)

    if options is not None:
        for key, value in options.items():
            highs.setOptionValue(str(key), float(value))

    infinity = highs.getInfinity()
    num_facets = int(objective.size)

    variables = highs.addVariables(
        range(num_facets),
        lb=[0.0] * num_facets,
        ub=[infinity] * num_facets,
        obj=list(map(float, objective)),
        type=[resources.HighsVarType.kContinuous] * num_facets,
    )

    for row in normals.T:
        expression = highs.expr()
        for column, coefficient in enumerate(row):
            if coefficient == 0.0:
                continue
            expression += float(coefficient) * variables[column]
        highs.addConstr(expression == 0.0)

    expression = highs.expr()
    for column, coefficient in enumerate(support):
        if coefficient == 0.0:
            continue
        expression += float(coefficient) * variables[column]
    highs.addConstr(expression == 1.0)

    status = highs.maximize()
    if status != resources.HighsStatus.kOk:
        return None

    model_status = highs.getModelStatus()
    if model_status != resources.HighsModelStatus.kOptimal:
        return None

    solution = highs.getSolution()
    return np.asarray(solution.col_value[:num_facets], dtype=np.float64)


def _solve_beta_product_linear_program(
    *,
    pair: tuple[int, int],
    normals: np.ndarray,
    support: np.ndarray,
    single_bounds: np.ndarray,
    options: Mapping[str, float] | None,
) -> float | None:
    """Return an upper bound on ``beta_i * beta_j`` via a lifted relaxation."""

    i, j = pair
    upper_i = float(single_bounds[i])
    upper_j = float(single_bounds[j])

    if upper_i <= 0.0 or upper_j <= 0.0:
        return None

    resources = load_highs()
    highs = resources.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("random_seed", 0)

    if options is not None:
        for key, value in options.items():
            highs.setOptionValue(str(key), float(value))

    num_facets = int(support.size)
    infinity = highs.getInfinity()
    total_variables = num_facets + 1

    ub = [float(infinity)] * total_variables
    for index in range(num_facets):
        if single_bounds[index] > 0.0:
            ub[index] = float(single_bounds[index])

    lb = [0.0] * total_variables
    obj = [0.0] * num_facets + [1.0]

    variables = highs.addVariables(
        range(total_variables),
        lb=lb,
        ub=ub,
        obj=obj,
        type=[resources.HighsVarType.kContinuous] * total_variables,
    )

    w_variable = variables[num_facets]

    for row in normals.T:
        expression = highs.expr()
        for column, coefficient in enumerate(row):
            if coefficient == 0.0:
                continue
            expression += float(coefficient) * variables[column]
        highs.addConstr(expression == 0.0)

    expression = highs.expr()
    for column, coefficient in enumerate(support):
        if coefficient == 0.0:
            continue
        expression += float(coefficient) * variables[column]
    highs.addConstr(expression == 1.0)

    expression = highs.expr()
    expression += 1.0 * w_variable
    expression += -upper_i * variables[j]
    highs.addConstr(expression <= 0.0)

    expression = highs.expr()
    expression += 1.0 * w_variable
    expression += -upper_j * variables[i]
    highs.addConstr(expression <= 0.0)

    expression = highs.expr()
    expression += 1.0 * w_variable
    expression += -upper_i * variables[j]
    expression += -upper_j * variables[i]
    highs.addConstr(expression >= -upper_i * upper_j)

    status = highs.maximize()
    if status != resources.HighsStatus.kOk:
        return None

    model_status = highs.getModelStatus()
    if model_status != resources.HighsModelStatus.kOptimal:
        return None

    solution = highs.getSolution()
    return float(solution.col_value[num_facets])


def _compute_beta_upper_bounds(
    *,
    normals: np.ndarray,
    support: np.ndarray,
    abs_products: np.ndarray,
    options: Mapping[str, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-facet, pairwise, and product bounds for Reeb coefficients."""

    num_facets = int(support.size)
    single_bounds = np.zeros(num_facets, dtype=np.float64)
    pair_bounds = np.zeros((num_facets, num_facets), dtype=np.float64)
    product_bounds = np.zeros((num_facets, num_facets), dtype=np.float64)

    standard_objective = np.zeros(num_facets, dtype=np.float64)

    for index in range(num_facets):
        standard_objective.fill(0.0)
        standard_objective[index] = 1.0
        solution = _solve_beta_linear_program(
            objective=standard_objective,
            normals=normals,
            support=support,
            options=options,
        )
        if solution is None:
            continue
        single_bounds[index] = float(np.maximum(0.0, solution[index]))

    for i in range(num_facets):
        for j in range(i + 1, num_facets):
            if abs_products[i, j] == 0.0:
                continue
            standard_objective.fill(0.0)
            standard_objective[i] = 1.0
            standard_objective[j] = 1.0
            solution = _solve_beta_linear_program(
                objective=standard_objective,
                normals=normals,
                support=support,
                options=options,
            )
            if solution is None:
                continue
            pair_sum = float(np.maximum(0.0, solution[i] + solution[j]))
            pair_bounds[i, j] = pair_sum
            pair_bounds[j, i] = pair_sum

            product = _solve_beta_product_linear_program(
                pair=(i, j),
                normals=normals,
                support=support,
                single_bounds=single_bounds,
                options=options,
            )
            if product is not None:
                product_bounds[i, j] = product
                product_bounds[j, i] = product

    return single_bounds, pair_bounds, product_bounds


@lru_cache(maxsize=32)
def _standard_symplectic_matrix_cached(dimension: int) -> np.ndarray:
    """Return the canonical symplectic form as a cached NumPy array."""

    return np.asarray(standard_symplectic_matrix(dimension), dtype=np.float64)


def estimate_capacity_lower_bound(
    *,
    B_matrix: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    tol: float,
    options: Mapping[str, float] | None = None,
) -> float | None:
    """Return a relaxation-based lower bound for the EHZ capacity."""

    normals = np.ascontiguousarray(B_matrix, dtype=np.float64)
    support = np.ascontiguousarray(c, dtype=np.float64)

    options_key = _normalise_options(options)
    upper_estimate = _cached_relaxation_upper_estimate(
        normals_shape=tuple(normals.shape),
        normals_bytes=normals.tobytes(),
        support_size=int(support.size),
        support_bytes=support.tobytes(),
        options_key=options_key,
    )

    if upper_estimate <= float(tol):
        return None

    return 0.5 / upper_estimate


def _normalise_options(options: Mapping[str, float] | None) -> Tuple[Tuple[str, float], ...]:
    """Return a deterministic tuple encoding HiGHS options for caching."""

    if options is None:
        return ()
    return tuple(sorted((str(key), float(value)) for key, value in options.items()))


@lru_cache(maxsize=64)
def _cached_relaxation_upper_estimate(
    *,
    normals_shape: Tuple[int, ...],
    normals_bytes: bytes,
    support_size: int,
    support_bytes: bytes,
    options_key: Tuple[Tuple[str, float], ...],
) -> float:
    """Return the cached lifted-relaxation contribution sum for ``B`` and ``c``."""

    normals = np.frombuffer(normals_bytes, dtype=np.float64).reshape(normals_shape)
    support = np.frombuffer(support_bytes, dtype=np.float64).reshape((support_size,))

    options = None if len(options_key) == 0 else {key: value for key, value in options_key}

    dimension = int(normals.shape[1])
    J = _standard_symplectic_matrix_cached(dimension)
    symplectic_products = normals @ J @ normals.T
    abs_products = np.abs(symplectic_products)

    single_bounds, pair_bounds, product_bounds = _compute_beta_upper_bounds(
        normals=normals,
        support=support,
        abs_products=abs_products,
        options=options,
    )

    outer_products = np.minimum(
        np.multiply.outer(single_bounds, single_bounds),
        0.25 * np.square(pair_bounds),
    )
    valid_products = product_bounds > 0.0
    if np.any(valid_products):
        outer_products = np.where(
            valid_products,
            np.minimum(outer_products, product_bounds),
            outer_products,
        )
    np.fill_diagonal(outer_products, 0.0)

    contribution = abs_products * outer_products
    return float(np.sum(np.triu(contribution, k=1)))


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

    subset = FacetSubset(indices=model.indices, beta=beta, symplectic_products=symplectic_products)

    # Evaluate the candidate using the dynamic programming shortcut for robustness.
    from viterbo.symplectic.capacity.facet_normals.subset_utils import (  # local import for cycle
        subset_capacity_candidate_dynamic,
    )

    candidate = subset_capacity_candidate_dynamic(subset, tol=tol)
    if candidate is None:
        return None

    return MilpCertificate(subset=subset, capacity=float(candidate))
