"""Optimized wrapper delegating to the facet-normal EHZ solver."""

from __future__ import annotations

from typing import Final

from jaxtyping import Array, Float

from viterbo.symplectic.capacity.facet_normals.fast import (
    compute_ehz_capacity_fast as _facet_fast,
)
from viterbo.symplectic.capacity.facet_normals.reference import (
    compute_ehz_capacity_reference as _facet_reference,
)
from viterbo.symplectic.capacity.reeb_cycles.graph import build_oriented_edge_graph


def compute_ehz_capacity_fast(
    B_matrix: Float[Array, " num_facets dimension"],
    c_vector: Float[Array, " num_facets"],
    *,
    atol: float = 1e-9,
) -> float:
    """Compute ``c_EHZ`` via the fast facet-normal search after graph validation."""
    graph = build_oriented_edge_graph(B_matrix, c_vector, atol=atol)
    if graph.graph.number_of_nodes() == 0:
        raise ValueError("Oriented-edge graph is empty; polytope lacks admissible edges.")
    try:
        return _facet_fast(B_matrix, c_vector)
    except ValueError:
        return _facet_reference(B_matrix, c_vector)


__all__: Final = ["compute_ehz_capacity_fast"]
