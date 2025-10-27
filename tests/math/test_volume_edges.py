from __future__ import annotations

import math

import pytest
import torch

from viterbo.math.volume import (
    volume,
    volume_via_lawrence,
    volume_via_monte_carlo,
    volume_via_triangulation,
)

torch.set_default_dtype(torch.float64)

pytestmark = pytest.mark.smoke


def test_invalid_shapes_raise() -> None:
    with pytest.raises(ValueError):
        volume(torch.randn(10))  # 1D tensor
    with pytest.raises(ValueError):
        volume(torch.randn(2, 2, 2))  # 3D tensor
    with pytest.raises(ValueError):
        volume(torch.empty(4, 0))  # D == 0 not allowed


def test_invalid_dtype_raises() -> None:
    # Integers are not supported (mean/linear algebra require floating types).
    int_triangle = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.int64)
    with pytest.raises(Exception):  # RuntimeError on integer mean/ops
        _ = volume(int_triangle)


def test_dispatch_1d_length_and_dtype() -> None:
    seg = torch.tensor([[-1.5], [2.0]], dtype=torch.float32)
    length = volume(seg)
    assert length.dtype == torch.float32
    torch.testing.assert_close(length, torch.tensor(3.5, dtype=torch.float32))


def test_dispatch_2d_shoelace_order_and_duplicates() -> None:
    # Unordered square with duplicates; area should be 4.0
    sq = torch.tensor(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [-1.0, -1.0],
            [1.0, 1.0],  # duplicate
            [-1.0, -1.0],  # duplicate
        ]
    )
    area = volume(sq)
    torch.testing.assert_close(area, torch.tensor(4.0, dtype=sq.dtype))


def test_dispatch_3d_known_cube_volume() -> None:
    corners = torch.cartesian_prod(
        torch.tensor([-1.0, 1.0]),
        torch.tensor([-1.0, 1.0]),
        torch.tensor([-1.0, 1.0]),
    ).to(torch.get_default_dtype())
    vol = volume(corners)
    torch.testing.assert_close(vol, torch.tensor(8.0, dtype=corners.dtype), atol=1e-9, rtol=0.0)


def test_dispatch_4d_known_simplex_and_cube() -> None:
    # 4-simplex: origin + basis vectors, volume = 1/4!
    simplex = torch.cat((torch.zeros((1, 4)), torch.eye(4)), dim=0)
    vol_simplex = volume(simplex)
    torch.testing.assert_close(
        vol_simplex,
        torch.tensor(1.0 / math.factorial(4), dtype=simplex.dtype),
        atol=1e-12,
        rtol=0.0,
    )
    # 4D hypercube [-1,1]^4 has volume 16
    corners4 = torch.cartesian_prod(
        torch.tensor([-1.0, 1.0]),
        torch.tensor([-1.0, 1.0]),
        torch.tensor([-1.0, 1.0]),
        torch.tensor([-1.0, 1.0]),
    ).to(torch.get_default_dtype())
    vol_cube4 = volume(corners4)
    torch.testing.assert_close(
        vol_cube4, torch.tensor(16.0, dtype=corners4.dtype), atol=1e-9, rtol=0.0
    )


def test_rank_deficient_in_3d_raises() -> None:
    # Coplanar points in R^3 are not full-dimensional; hull conversion should fail.
    coplanar = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    with pytest.raises(ValueError):
        _ = volume(coplanar)


def test_tolerance_near_degenerate_tetra() -> None:
    # Base triangle area = 0.5; small height epsilon => volume ~ (0.5 * eps) / 3
    eps = 5e-8  # a bit above sqrt(eps) for float64; exercises tolerance threshold
    base = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    apex = torch.tensor([[0.0, 0.0, eps]])
    tetra = torch.cat((base, apex), dim=0)
    vol = volume(tetra)
    expected = (0.5 * eps) / 3.0
    torch.testing.assert_close(vol, torch.tensor(expected, dtype=tetra.dtype), atol=1e-12, rtol=0.0)


def test_stubs_raise_not_implemented() -> None:
    verts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(NotImplementedError):
        _ = volume_via_triangulation(verts)
    normals = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    offsets = torch.tensor([1.0, 1.0, 1.0, 1.0])
    with pytest.raises(NotImplementedError):
        _ = volume_via_lawrence(normals, offsets)
    with pytest.raises(NotImplementedError):
        _ = volume_via_monte_carlo(verts, normals, offsets, samples=10, generator=torch.Generator())


def test_internal_project_span_degenerate_and_rank_zero() -> None:
    from viterbo.math.volume import _project_onto_affine_span

    # Single unique vertex → None
    one = torch.zeros((1, 3))
    assert _project_onto_affine_span(one, tol=1e-9) is None

    # Multiple vertices but tol so large that rank drops to zero → None
    pts = torch.tensor([[0.0, 0.0, 0.0], [1e-12, 0.0, 0.0], [0.0, 1e-12, 0.0]])
    assert _project_onto_affine_span(pts, tol=1.0) is None


def test_internal_facet_measure_degenerate_returns_zero() -> None:
    from viterbo.math.volume import _facet_measure

    # Single point facet → zero measure
    v = torch.tensor([[3.14, -2.0, 0.0]])
    fm = _facet_measure(v, tol=1e-9)
    torch.testing.assert_close(fm, torch.zeros((), dtype=v.dtype))
