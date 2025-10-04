"""Half-space utilities with reference, optimised, and JAX variants."""

from viterbo.geometry.halfspaces import jax as _jax_impl
from viterbo.geometry.halfspaces import optimized as _optimized_impl
from viterbo.geometry.halfspaces import reference as _reference_impl
from viterbo.geometry.halfspaces import samples as _samples

enumerate_vertices_reference = _reference_impl.enumerate_vertices
remove_redundant_facets_reference = _reference_impl.remove_redundant_facets

enumerate_vertices_optimized = _optimized_impl.enumerate_vertices
remove_redundant_facets_optimized = _optimized_impl.remove_redundant_facets

enumerate_vertices_jax = _jax_impl.enumerate_vertices
remove_redundant_facets_jax = _jax_impl.remove_redundant_facets

unit_hypercube_halfspaces = _samples.unit_hypercube_halfspaces
unit_square_halfspaces = _samples.unit_square_halfspaces

# Backwards-compatible aliases for the legacy API surface.
enumerate_vertices = enumerate_vertices_reference
remove_redundant_facets = remove_redundant_facets_reference
