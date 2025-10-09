from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.smoke]

from viterbo.exp1.capacity_ehz import capacity_halfspace_optimization
from viterbo.exp1.examples import hypercube
from viterbo.exp1.polytopes import lagrangian_product, matmul, to_halfspaces, to_vertices


@pytest.mark.goal_code
def test_capacity_halfspace_optimization_raises_notimplemented_for_dim8() -> None:
    """Facet solver raises NotImplementedError when subset size > 7 (dim=8 ⇒ 9)."""
    H = hypercube(8)
    A, b = H.as_tuple()
    with pytest.raises(NotImplementedError):
        _ = capacity_halfspace_optimization(A, b, tol=1e-9)


@pytest.mark.goal_code
def test_to_lagrangian_product_rejects_mixed_support_facets() -> None:
    """to_lagrangian_product rejects 4D polytopes with mixed-support facet normals."""
    # Start with a clean 2x2 product (square × square) in R^4
    left = to_vertices(hypercube(2))
    right = to_vertices(hypercube(2))
    prod = lagrangian_product(left, right)
    H4 = to_halfspaces(prod)
    # Apply an invertible linear map that mixes p and q blocks
    I2 = np.eye(2)
    mix = np.block([[I2, 0.5 * I2], [0.5 * I2, I2]])
    H_mixed = matmul(mix, H4)
    with pytest.raises(ValueError):
        # Mixed-support facets should cause factorization to fail by design
        from viterbo.exp1.polytopes import to_lagrangian_product

        _ = to_lagrangian_product(H_mixed)
