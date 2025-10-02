"""Top-level package for the Viterbo tools."""

from __future__ import annotations

from .core import normalize_vector
from .ehz import compute_ehz_capacity, standard_symplectic_matrix
from .ehz_fast import compute_ehz_capacity_fast
from .hello import hello_numpy
from .polytopes import (
    Polytope,
    affine_transform,
    cartesian_product,
    catalog,
    cross_polytope,
    hypercube,
    mirror_polytope,
    random_affine_map,
    random_polytope,
    random_transformations,
    regular_polygon_product,
    rotate_polytope,
    simplex_with_uniform_weights,
    translate_polytope,
    truncated_simplex_four_dim,
    viterbo_counterexample,
)
from .search import enumerate_search_space, iter_search_space
from .systolic import systolic_ratio
from .volume import polytope_volume_fast, polytope_volume_reference

__all__ = [
    "Polytope",
    "affine_transform",
    "cartesian_product",
    "catalog",
    "compute_ehz_capacity",
    "compute_ehz_capacity_fast",
    "cross_polytope",
    "enumerate_search_space",
    "hello_numpy",
    "hypercube",
    "iter_search_space",
    "mirror_polytope",
    "normalize_vector",
    "polytope_volume_fast",
    "polytope_volume_reference",
    "random_affine_map",
    "random_polytope",
    "random_transformations",
    "regular_polygon_product",
    "rotate_polytope",
    "simplex_with_uniform_weights",
    "standard_symplectic_matrix",
    "systolic_ratio",
    "translate_polytope",
    "truncated_simplex_four_dim",
    "viterbo_counterexample",
]
