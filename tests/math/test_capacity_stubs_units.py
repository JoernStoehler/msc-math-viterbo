from __future__ import annotations

import math

import pytest
import torch

from viterbo.math.capacity_ehz.stubs import (
    _candidate_betas,
    _maximum_triangular_sum,
    _nullspace_vectors,
    _symplectic_form_matrix,
)


pytestmark = pytest.mark.smoke


def test_symplectic_form_matrix_errors_on_odd_dims() -> None:
    with pytest.raises(ValueError):
        _symplectic_form_matrix(3, dtype=torch.float64, device=torch.device("cpu"))


@pytest.mark.parametrize("d", [2, 4, 6])
def test_symplectic_form_matrix_structure_and_properties(d: int) -> None:
    J = _symplectic_form_matrix(d, dtype=torch.float64, device=torch.device("cpu"))
    assert J.shape == (d, d)
    # Skew-symmetry: J^T = -J
    assert torch.allclose(J.T, -J, atol=1e-12, rtol=0.0)
    # Canonical block structure: J^2 = -I
    I = torch.eye(d, dtype=J.dtype, device=J.device)
    assert torch.allclose(J @ J, -I, atol=1e-12, rtol=0.0)


def test_nullspace_vectors_empty_and_identity_cases() -> None:
    dtype = torch.float64
    device = torch.device("cpu")
    tol = 1e-12

    # Empty matrix (0 x 3) -> no constraints, return (3, 0)
    A_empty_rows = torch.empty((0, 3), dtype=dtype, device=device)
    ns1 = _nullspace_vectors(A_empty_rows, tol)
    assert ns1.shape == (3, 0)

    # Zero-column matrix (3 x 0) -> s.numel() == 0 branch, identity of size 0 (0, 0)
    A_zero_cols = torch.empty((3, 0), dtype=dtype, device=device)
    ns2 = _nullspace_vectors(A_zero_cols, tol)
    assert ns2.shape == (0, 0)


def test_nullspace_vectors_full_rank_and_rank_deficient() -> None:
    dtype = torch.float64
    device = torch.device("cpu")
    tol = 1e-10

    # Full column rank: I_3 => nullspace is empty (3, 0)
    A_full = torch.eye(3, dtype=dtype, device=device)
    ns_full = _nullspace_vectors(A_full, tol)
    assert ns_full.shape == (3, 0)

    # Rank-deficient (2 x 3): rows select first two coordinates => nullspace span{e3}
    A_rd = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype, device=device)
    ns_rd = _nullspace_vectors(A_rd, tol)
    # One-dimensional nullspace
    assert ns_rd.shape == (3, 1)
    # Check that columns form a basis for solutions: A @ v = 0
    assert torch.allclose(A_rd @ ns_rd, torch.zeros((2, 1), dtype=dtype, device=device), atol=1e-9, rtol=0.0)
    # Orthonormality (SVD basis): v^T v = 1
    v = ns_rd[:, 0]
    assert math.isclose(float(v.dot(v).item()), 1.0, rel_tol=0.0, abs_tol=1e-9)


def test_candidate_betas_null_dim_one_feasible_and_normalised() -> None:
    dtype = torch.float64
    device = torch.device("cpu")
    tol = 1e-10
    feas_tol = 1e-9

    # A = [1, -1] (1 x 2), nullspace span{[1, 1]}; offsets positive
    A = torch.tensor([[1.0, -1.0]], dtype=dtype, device=device)
    offsets = torch.tensor([1.0, 2.0], dtype=dtype, device=device)
    null = _nullspace_vectors(A, tol)
    assert null.shape == (2, 1)

    cands = _candidate_betas(A, offsets, null, tol, feas_tol)
    # Expect exactly one non-negative feasible candidate on this ray
    assert len(cands) == 1
    beta = cands[0]
    # Non-negative and non-zero
    assert torch.all(beta >= -1e-12)
    assert float(torch.linalg.norm(beta).item()) > 0.0
    # Unit length because the ray was normalised inside (no negatives to zero)
    assert torch.allclose(torch.linalg.norm(beta), torch.tensor(1.0, dtype=dtype), atol=1e-9, rtol=0.0)
    # Feasibility: A @ beta == 0
    assert torch.allclose(A @ beta, torch.zeros((1,), dtype=dtype), atol=1e-9, rtol=0.0)


def test_candidate_betas_null_dim_two_feasible_and_deduplicated() -> None:
    dtype = torch.float64
    device = torch.device("cpu")
    tol = 1e-10
    feas_tol = 1e-9

    # A = [[0, 0, 1]] (1 x 3), nullspace span{e1, e2}; feasible betas have beta3 = 0, beta1,beta2 >= 0
    A = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype, device=device)
    offsets = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=device)
    null = _nullspace_vectors(A, tol)
    assert null.shape == (3, 2)

    cands = _candidate_betas(A, offsets, null, tol, feas_tol)
    # All candidates must be feasible and have beta3 ~ 0
    for beta in cands:
        assert torch.all(beta >= -1e-12)
        assert float(beta[2].abs().item()) <= 1e-8
        assert float(torch.linalg.norm(beta).item()) > 0.0
        assert torch.allclose(A @ beta, torch.zeros((1,), dtype=dtype), atol=1e-8, rtol=0.0)

    # Deduplication: normalise by offsets dot and ensure uniqueness
    def _normalise_by_offsets(b: torch.Tensor) -> torch.Tensor:
        scale = float(offsets.dot(b).item())
        assert scale > 0.0
        return (b / scale).round(decimals=6)

    normalised = [_normalise_by_offsets(b) for b in cands]
    # Convert to tuples for easy set comparison
    tuples = [tuple(map(float, v.tolist())) for v in normalised]
    assert len(tuples) == len(set(tuples))


def test_maximum_triangular_sum_small_k() -> None:
    dtype = torch.float64
    device = torch.device("cpu")

    # k <= 1 => 0.0
    assert _maximum_triangular_sum(torch.tensor([], dtype=dtype, device=device), torch.empty((0, 0), dtype=dtype, device=device)) == 0.0
    assert _maximum_triangular_sum(torch.tensor([1.0], dtype=dtype, device=device), torch.zeros((1, 1), dtype=dtype, device=device)) == 0.0

    # k = 2 => |b1*b2*w|
    beta2 = torch.tensor([1.0, 1.0], dtype=dtype, device=device)
    omega2 = torch.tensor([[0.0, 2.0], [-2.0, 0.0]], dtype=dtype, device=device)
    val2 = _maximum_triangular_sum(beta2, omega2)
    assert math.isclose(val2, 2.0, rel_tol=0.0, abs_tol=1e-12)

    # k = 3 with positive upper-triangular entries -> optimal order sums them
    beta3 = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)
    omega3 = torch.tensor(
        [[0.0, 2.0, 3.0], [-2.0, 0.0, 5.0], [-3.0, -5.0, 0.0]], dtype=dtype, device=device
    )
    val3 = _maximum_triangular_sum(beta3, omega3)
    assert math.isclose(val3, 2.0 + 3.0 + 5.0, rel_tol=0.0, abs_tol=1e-12)

