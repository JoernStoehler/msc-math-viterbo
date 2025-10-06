"""Polytope helpers and canonical combinatorics (JAX-first)."""

from viterbo.geometry.polytopes import reference as _reference
from viterbo.geometry.polytopes import samples as _samples

NormalCone = _reference.NormalCone
Polytope = _reference.Polytope
PolytopeCombinatorics = _reference.PolytopeCombinatorics

affine_transform = _reference.affine_transform
cartesian_product = _reference.cartesian_product
clear_polytope_cache = _reference.clear_polytope_cache
haim_kislev_action = _reference.haim_kislev_action
polytope_fingerprint = _reference.polytope_fingerprint
random_affine_map = _reference.random_affine_map
random_polytope = _reference.random_polytope
random_transformations = _reference.random_transformations
rotate_polytope = _reference.rotate_polytope
translate_polytope = _reference.translate_polytope
mirror_polytope = _reference.mirror_polytope
polytope_combinatorics = _reference.polytope_combinatorics
vertices_from_halfspaces = _reference.vertices_from_halfspaces
halfspaces_from_vertices = _reference.halfspaces_from_vertices

catalog = _samples.catalog
cross_polytope = _samples.cross_polytope
hypercube = _samples.hypercube
regular_polygon_product = _samples.regular_polygon_product
simplex_with_uniform_weights = _samples.simplex_with_uniform_weights
truncated_simplex_four_dim = _samples.truncated_simplex_four_dim
viterbo_counterexample = _samples.viterbo_counterexample
