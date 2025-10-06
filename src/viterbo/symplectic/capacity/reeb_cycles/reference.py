"""Reference wrapper for combinatorial Reeb cycle verification."""

from __future__ import annotations

from typing import Final

from jaxtyping import Array, Float

from viterbo.symplectic.capacity.facet_normals.reference import (
    compute_ehz_capacity_reference as _facet_reference,
)
from viterbo.symplectic.capacity.reeb_cycles.graph import build_oriented_edge_graph


def compute_ehz_capacity_reference(
    B_matrix: Float[Array, " num_facets dimension"],
    c_vector: Float[Array, " num_facets"],
    *,
    atol: float = 1e-9,
) -> float:
    """Compute ``c_EHZ`` while validating the oriented-edge graph."""
    graph = build_oriented_edge_graph(B_matrix, c_vector, atol=atol)
    if graph.graph.number_of_nodes() == 0:
        raise ValueError("Oriented-edge graph is empty; polytope lacks admissible edges.")
    return _facet_reference(B_matrix, c_vector)


__all__: Final = ["compute_ehz_capacity_reference"]
