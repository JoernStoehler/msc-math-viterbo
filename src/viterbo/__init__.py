"""Top-level package for the Viterbo tools."""

from __future__ import annotations

from .core import normalize_vector, standard_symplectic_matrix
from .ehz import compute_ehz_capacity
from .ehz_fast import compute_ehz_capacity_fast
from .hello import hello_numpy
from .polytopes import (
    NormalCone,
    Polytope,
    PolytopeCombinatorics,
    affine_transform,
    cartesian_product,
    catalog,
    cross_polytope,
    halfspaces_from_vertices,
    hypercube,
    mirror_polytope,
    polytope_combinatorics,
    polytope_fingerprint,
    random_affine_map,
    random_polytope,
    random_transformations,
    regular_polygon_product,
    rotate_polytope,
    simplex_with_uniform_weights,
    translate_polytope,
    truncated_simplex_four_dim,
    vertices_from_halfspaces,
    viterbo_counterexample,
)
from .search import enumerate_search_space, iter_search_space
from .solvers import (
    LinearProgram,
    LinearProgramBackend,
    LinearProgramSolution,
    ScipyLinearProgramBackend,
    solve_linear_program,
)
from .systolic import systolic_ratio
from .volume import polytope_volume_fast, polytope_volume_reference

__all__ = [
    "LinearProgram",
    "LinearProgramBackend",
    "LinearProgramSolution",
    "NormalCone",
    "Polytope",
    "PolytopeCombinatorics",
    "affine_transform",
    "cartesian_product",
    "catalog",
    "compute_ehz_capacity",
    "compute_ehz_capacity_fast",
    "cross_polytope",
    "enumerate_search_space",
    "hello_numpy",
    "halfspaces_from_vertices",
    "iter_search_space",
    "hypercube",
    "mirror_polytope",
    "normalize_vector",
    "polytope_combinatorics",
    "polytope_fingerprint",
    "polytope_volume_fast",
    "polytope_volume_reference",
    "random_affine_map",
    "random_polytope",
    "random_transformations",
    "regular_polygon_product",
    "rotate_polytope",
    "simplex_with_uniform_weights",
    "solve_linear_program",
    "ScipyLinearProgramBackend",
    "standard_symplectic_matrix",
    "systolic_ratio",
    "translate_polytope",
    "truncated_simplex_four_dim",
    "vertices_from_halfspaces",
    "viterbo_counterexample",
]
