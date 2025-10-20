from __future__ import annotations

import torch

import pytest

from viterbo.math.capacity_ehz.common import (
    order_vertices_ccw,
    polygon_area,
    validate_halfspaces_planar,
    validate_planar_vertices,
)
from viterbo.math.capacity_ehz.cycle import minimal_action_cycle
from viterbo.math.volume import volume


def test_minimal_action_cycle_odd_dimension_error() -> None:
    # d=1 should raise a clear ValueError about even ambient dimension
    vertices = torch.tensor([[0.0], [1.0], [2.0]])
    normals = torch.tensor([[1.0], [-1.0]])
    offsets = torch.tensor([2.0, 0.0])
    with pytest.raises(ValueError, match="even \\("):
        minimal_action_cycle(vertices, normals, offsets)


def test_minimal_action_cycle_4d_non_product_not_implemented() -> None:
    # A random 4D vertex set will generally not form a cartesian product of two polygons.
    grid = torch.randn(7, 4)
    normals = torch.randn(9, 4)
    offsets = torch.ones(9)
    with pytest.raises(NotImplementedError, match="Lagrangian product"):
        minimal_action_cycle(grid, normals, offsets)


def test_planar_validators_and_ccw_ordering() -> None:
    # validate_planar_vertices
    bad = torch.randn(2, 3)
    with pytest.raises(ValueError, match=r"planar.*\(N, 2\)"):
        validate_planar_vertices(bad, "vertices")
    with pytest.raises(ValueError, match=r"N>=3|at least three"):
        validate_planar_vertices(torch.randn(2, 2), "vertices")

    # validate_halfspaces_planar
    normals = torch.randn(2, 3)
    offsets = torch.ones(2)
    with pytest.raises(ValueError, match=r"D=2|\(F, 2\)"):
        validate_halfspaces_planar(normals, offsets, "normals", "offsets")
    with pytest.raises(ValueError, match="share the first dimension"):
        validate_halfspaces_planar(torch.randn(3, 2), torch.ones(2), "normals", "offsets")
    with pytest.raises(ValueError, match="strictly positive"):
        validate_halfspaces_planar(
            torch.randn(3, 2), torch.tensor([1.0, 0.0, -1.0]), "normals", "offsets"
        )

    # order_vertices_ccw preserves dtype/device and orients CCW
    v = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    ordered = order_vertices_ccw(v)
    assert ordered.dtype == v.dtype and ordered.device == v.device
    rolled = ordered.roll(-1, dims=0)
    signed = 0.5 * torch.sum(ordered[:, 0] * rolled[:, 1] - ordered[:, 1] * rolled[:, 0])
    assert float(signed) > 0.0


def test_volume_backend_paths_basic() -> None:
    # 2D: area2d path equals polygon_area
    poly = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    torch.testing.assert_close(volume(poly), polygon_area(poly))

    # 3D: facets path returns positive scalar for a cube
    coords = [-1.0, 1.0]
    cube = torch.tensor(
        [[x, y, z] for x in coords for y in coords for z in coords], dtype=torch.float64
    )
    vol = volume(cube)
    assert vol.ndim == 0 and float(vol) > 0.0
