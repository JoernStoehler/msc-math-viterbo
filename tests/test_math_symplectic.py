from __future__ import annotations

import pytest
import torch

from viterbo.math.symplectic import (
    capacity_ehz_algorithm1,
    capacity_ehz_algorithm2,
    capacity_ehz_primal_dual,
    lagrangian_product,
    minimal_action_cycle,
    random_symplectic_matrix,
    symplectic_form,
    systolic_ratio,
)


torch.set_default_dtype(torch.float64)


def _sorted_rows(tensor: torch.Tensor) -> torch.Tensor:
    order = torch.arange(tensor.size(0), device=tensor.device)
    for dim in range(tensor.size(1) - 1, -1, -1):
        order = order[torch.argsort(tensor[order, dim])]
    return tensor[order]


def test_symplectic_form_structure() -> None:
    j = symplectic_form(4)
    expected = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]]
    )
    torch.testing.assert_close(j, expected)


def test_random_symplectic_matrix_preserves_form() -> None:
    dimension = 4
    seed = 42
    matrix = random_symplectic_matrix(dimension, seed)
    j = symplectic_form(dimension)
    lhs = matrix.T @ j @ matrix
    torch.testing.assert_close(lhs, j, atol=1e-6, rtol=1e-6)


def test_lagrangian_product_block_structure() -> None:
    vertices_p = torch.tensor([[-1.0], [1.0]])
    vertices_q = torch.tensor([[0.0], [2.0]])
    vertices, normals, offsets = lagrangian_product(vertices_p, vertices_q)
    expected_vertices = torch.tensor(
        [[-1.0, 0.0], [-1.0, 2.0], [1.0, 0.0], [1.0, 2.0]]
    )
    torch.testing.assert_close(vertices, expected_vertices)
    # Halfspaces correspond to |x| <= 1 and 0 <= y <= 2
    expected_normals = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )
    expected_offsets = torch.tensor([1.0, 1.0, 2.0, 0.0])
    torch.testing.assert_close(_sorted_rows(normals), _sorted_rows(expected_normals), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        offsets.sort().values,
        expected_offsets.sort().values,
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.skip(reason="capacity solvers not implemented yet")
def test_capacity_ehz_algorithm1_matches_ellipsoid() -> None:
    """Verify the Artstein–Avidan–Ostrover programme on ellipsoids once available.

    The EHZ capacity of an ellipsoid ``\{x : x^T Q x \le 1\}`` is
    ``\pi / \sqrt{\lambda_{max}(Q)}`` (Hofer & Zehnder 1994, Prop. 4.3.2). The
    future test will discretise ``Q`` via supporting hyperplanes, run
    :func:`capacity_ehz_algorithm1`, and assert that the returned value matches
    the analytic prediction to 1e-9 relative tolerance while also confirming the
    reported active facet set spans the correct eigenspaces.
    """
    pytest.skip("capacity solvers not implemented yet")


@pytest.mark.skip(reason="capacity solvers not implemented yet")
def test_capacity_ehz_algorithm2_detects_shortest_billiard() -> None:
    """Expect the vertex-based search to recover the square billiard action.

    Bezdek & Bezdek (2010) show that the minimal Minkowski billiard for the
    square Lagrangian product ``[-1, 1] × [-1, 1]`` travels along a diagonal
    with action ``2\sqrt{2}``. After constraining
    :func:`capacity_ehz_algorithm2` to the Lagrangian-product setting we will
    enumerate admissible vertex words and assert that the solver returns both
    the action and the vertex cycle matching the analytic solution.
    """
    pytest.skip("capacity solvers not implemented yet")


@pytest.mark.skip(reason="primal-dual solver not implemented yet")
def test_capacity_ehz_primal_dual_handles_general_polytope() -> None:
    """Ensure the hybrid solver returns consistent primal and dual certificates.

    Once :func:`capacity_ehz_primal_dual` is available we will feed it a 4D
    permutahedron where Minkowski billiards are invalid and verify that the
    reported capacity agrees with the result from
    :func:`capacity_ehz_algorithm1`, while also checking that the returned vertex
    cycle satisfies the balance equations to 1e-8.
    """
    pytest.skip("primal-dual solver not implemented yet")


@pytest.mark.skip(reason="minimal action reconstruction not implemented")
def test_minimal_action_cycle_returns_closed_characteristic() -> None:
    """Confirm the returned cycle solves the discrete Hamiltonian system.

    After coupling to the convex optimisation solvers we will call
    :func:`minimal_action_cycle` on both an ellipsoid (facet-based workflow) and
    a square Lagrangian product (vertex-based workflow), ensuring the resulting
    orbit closes, satisfies equal-incidence, and reproduces the input capacity to
    1e-9 relative accuracy.
    """
    pytest.skip("minimal action reconstruction not implemented")


@pytest.mark.skip(reason="systolic ratio helper not implemented")
def test_systolic_ratio_matches_ball_normalisation() -> None:
    """Ensure the systolic ratio agrees with the Viterbo conjecture on balls.

    For the ``2n``-dimensional ball ``B^{2n}(r)`` the volume and capacity satisfy
    ``vol = \pi^n r^{2n} / n!`` and ``c_{EHZ} = \pi r^2``. After implementing
    :func:`systolic_ratio` we will plug these tensors in for ``n = 2`` and
    ``n = 3`` and assert that the helper returns ``2!`` and ``3!`` respectively,
    while also checking that invalid ``capacity_ehz`` inputs raise ``ValueError``.
    """
    pytest.skip("systolic ratio helper not implemented")
