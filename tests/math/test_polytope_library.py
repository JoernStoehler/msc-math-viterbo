from __future__ import annotations

import torch

from tests.polytopes import STANDARD_POLYTOPES


def test_standard_polytopes_metadata_consistency() -> None:
    for poly in STANDARD_POLYTOPES:
        assert poly.vertices.ndim == 2
        assert poly.normals.ndim == 2
        assert poly.offsets.ndim == 1
        assert poly.normals.size(0) == poly.offsets.size(0)
        assert poly.vertices.dtype == torch.float64
        assert poly.normals.dtype == torch.float64
        assert poly.offsets.dtype == torch.float64
        assert poly.volume.ndim == 0
        assert poly.volume.item() > 0.0
        if poly.volume_reference is not None:
            torch.testing.assert_close(
                poly.volume,
                torch.tensor(poly.volume_reference, dtype=poly.volume.dtype),
                rtol=1e-9,
                atol=1e-9,
            )
        normals_norm = torch.linalg.norm(poly.normals, dim=1)
        torch.testing.assert_close(normals_norm, torch.ones_like(normals_norm), atol=1e-9, rtol=0.0)
