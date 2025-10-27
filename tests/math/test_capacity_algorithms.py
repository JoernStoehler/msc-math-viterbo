from __future__ import annotations

import pytest
import torch

import viterbo.math.capacity_ehz.algorithms as algo
from viterbo.math.capacity_ehz.algorithms import (
    capacity_ehz_algorithm1,
    capacity_ehz_algorithm2,
    capacity_ehz_primal_dual,
)

pytestmark = pytest.mark.smoke


def _square_halfspaces_2d(dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    normals = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=dtype,
    )
    offsets = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=dtype)
    return normals, offsets


def _square_vertices_2d(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ],
        dtype=dtype,
    )


def _product_halfspaces_4d(dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    # Q = [-1, 1]^2, P = [-1, 1]^2; product is [-1,1]^4
    rows = []
    for dim in range(4):
        normal_pos = [0.0, 0.0, 0.0, 0.0]
        normal_neg = [0.0, 0.0, 0.0, 0.0]
        normal_pos[dim] = 1.0
        normal_neg[dim] = -1.0
        rows.append(normal_pos)
        rows.append(normal_neg)
    normals = torch.tensor(rows, dtype=dtype)
    offsets = torch.ones((8,), dtype=dtype)
    return normals, offsets


def _simplex_vertices_4d(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    # Standard 4-simplex vertices: origin and standard basis e1..e4 (not a cartesian product)
    eye = torch.eye(4, dtype=dtype)
    origin = torch.zeros((1, 4), dtype=dtype)
    return torch.cat([origin, eye], dim=0)


def test_algorithm1_2d_area_from_halfspaces() -> None:
    normals, offsets = _square_halfspaces_2d()
    cap = capacity_ehz_algorithm1(normals, offsets)
    assert cap.ndim == 0
    torch.testing.assert_close(cap, torch.tensor(4.0, dtype=normals.dtype))


def test_algorithm1_4d_delegates_to_algorithm2(monkeypatch: pytest.MonkeyPatch) -> None:
    normals, offsets = _product_halfspaces_4d()

    captured = {}

    def fake_algo2(vertices: torch.Tensor) -> torch.Tensor:
        captured["called"] = True
        captured["shape"] = tuple(vertices.shape)
        return torch.tensor(123.456, dtype=vertices.dtype, device=vertices.device)

    monkeypatch.setattr(algo, "capacity_ehz_algorithm2", fake_algo2)
    out = capacity_ehz_algorithm1(normals, offsets)
    assert captured.get("called", False) is True
    assert captured["shape"][1] == 4  # ambient dimension
    torch.testing.assert_close(out, torch.tensor(123.456, dtype=normals.dtype))


def test_algorithm2_2d_area_and_errors() -> None:
    # area
    tri = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.get_default_dtype())
    cap = capacity_ehz_algorithm2(tri)
    torch.testing.assert_close(cap, torch.tensor(0.5, dtype=tri.dtype))
    # <3 vertices -> ValueError
    with pytest.raises(ValueError):
        capacity_ehz_algorithm2(tri[:2])
    # odd ambient dimension -> ValueError
    odd = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=tri.dtype)
    with pytest.raises(ValueError):
        capacity_ehz_algorithm2(odd)


def test_algorithm2_4d_fallback_calls_oriented_edge_spectrum(monkeypatch: pytest.MonkeyPatch) -> None:
    vertices = _simplex_vertices_4d()
    called = {}

    def fake_oriented(vertices_in: torch.Tensor, normals: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        called["ok"] = (
            vertices_in.shape[1] == 4
            and normals.shape[1] == 4
            and normals.shape[0] == offsets.shape[0]
        )
        # Return sentinel with same dtype/device
        return torch.tensor(7.89, dtype=vertices_in.dtype, device=vertices_in.device)

    monkeypatch.setattr(algo, "oriented_edge_spectrum_4d", fake_oriented)
    out = capacity_ehz_algorithm2(vertices)
    assert called.get("ok", False)
    torch.testing.assert_close(out, torch.tensor(7.89, dtype=vertices.dtype))


def test_primal_dual_2d_consistent_and_inconsistent() -> None:
    # Consistent inputs: square
    vertices = _square_vertices_2d()
    cap_v = capacity_ehz_algorithm2(vertices)
    normals, offsets = _square_halfspaces_2d()
    out = capacity_ehz_primal_dual(vertices, normals, offsets)
    torch.testing.assert_close(out, cap_v)

    # Inconsistent: triangle vs square
    tri = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=vertices.dtype)
    with pytest.raises(ValueError):
        capacity_ehz_primal_dual(tri, normals, offsets)


def test_primal_dual_4d_consistent_and_inconsistent(monkeypatch: pytest.MonkeyPatch) -> None:
    vertices4 = _simplex_vertices_4d()
    normals4, offsets4 = _product_halfspaces_4d()

    # Consistent: monkeypatch both entries to agree
    monkeypatch.setattr(algo, "capacity_ehz_algorithm2", lambda v: torch.tensor(3.21, dtype=v.dtype))
    monkeypatch.setattr(
        algo,
        "capacity_ehz_algorithm1",
        lambda n, c: torch.tensor(3.21, dtype=n.dtype),
    )
    out = capacity_ehz_primal_dual(vertices4, normals4, offsets4)
    torch.testing.assert_close(out, torch.tensor(3.21, dtype=vertices4.dtype))

    # Inconsistent: algorithm1 vs algorithm2 disagree
    monkeypatch.setattr(algo, "capacity_ehz_algorithm2", lambda v: torch.tensor(1.0, dtype=v.dtype))
    monkeypatch.setattr(algo, "capacity_ehz_algorithm1", lambda n, c: torch.tensor(2.0, dtype=n.dtype))
    with pytest.raises(ValueError):
        capacity_ehz_primal_dual(vertices4, normals4, offsets4)
