from __future__ import annotations

import torch
from tests.polytopes import PLANAR_POLYTOPE_PAIRS

from viterbo.math.capacity_ehz.stubs import (
    compute_cF_constant_certified,
    oriented_edge_spectrum_4d,
)
from viterbo.math.constructions import lagrangian_product

torch.set_default_dtype(torch.float64)


def test_cF_constant_certified_positive_and_stable() -> None:
    """Certified C*(X) is strictly positive and deterministic."""
    square_q, square_p = PLANAR_POLYTOPE_PAIRS["square_product"]
    vertices, normals, offsets = lagrangian_product(square_q.vertices, square_p.vertices)

    # Build faces via the oriented-edge routine up to the faces enumeration stage.
    # We deliberately call the internal enumeration by briefly invoking the public
    # solver, which builds faces before DFS; that path is deterministic.
    # To keep the test focused, we reconstruct faces by mirroring the enumeration.
    from viterbo.math.capacity_ehz.stubs import _enumerate_two_faces, _vertex_facet_incidence

    tol = max(float(torch.finfo(vertices.dtype).eps) ** 0.5, 1e-9)
    vertex_facets = _vertex_facet_incidence(vertices, normals, offsets, tol)
    faces = _enumerate_two_faces(vertices, normals, vertex_facets, tol)

    c1 = compute_cF_constant_certified(normals, offsets, faces)
    c2 = compute_cF_constant_certified(normals, offsets, faces)
    assert c1 > 0.0 and c2 > 0.0
    assert abs(c1 - c2) <= 0.0  # bitwise-stable across repeated runs


def test_oriented_edge_budgets_do_not_change_minimal_action() -> None:
    """Enabling CH budgets preserves the minimal action on test polytopes."""
    square_q, square_p = PLANAR_POLYTOPE_PAIRS["square_product"]
    vertices, normals, offsets = lagrangian_product(square_q.vertices, square_p.vertices)

    base = oriented_edge_spectrum_4d(vertices, normals, offsets)
    with_budgets = oriented_edge_spectrum_4d(
        vertices, normals, offsets, use_cF_budgets=True, cF_constant=None
    )
    torch.testing.assert_close(with_budgets, base, atol=1e-8, rtol=0.0)
