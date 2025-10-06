"""Half-space utilities exposing reference and fast variants."""

from viterbo.geometry.halfspaces import fast as _fast
from viterbo.geometry.halfspaces import reference as _reference
from viterbo.geometry.halfspaces import samples as _samples

enumerate_vertices_reference = _reference.enumerate_vertices
remove_redundant_facets_reference = _reference.remove_redundant_facets

enumerate_vertices_fast = _fast.enumerate_vertices
remove_redundant_facets_fast = _fast.remove_redundant_facets

# Backwards-compatible default: expose the readable reference variant.
enumerate_vertices = enumerate_vertices_reference
remove_redundant_facets = remove_redundant_facets_reference

unit_hypercube_halfspaces = _samples.unit_hypercube_halfspaces
unit_square_halfspaces = _samples.unit_square_halfspaces
