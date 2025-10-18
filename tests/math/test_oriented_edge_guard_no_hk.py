from __future__ import annotations

import pytest
import torch
from tests.polytopes import PLANAR_POLYTOPE_PAIRS

from viterbo.math.capacity_ehz.stubs import oriented_edge_spectrum_4d
from viterbo.math.constructions import lagrangian_product

torch.set_default_dtype(torch.float64)


def test_oriented_edge_does_not_call_hk(monkeypatch: pytest.MonkeyPatch) -> None:
    # Monkeypatch HK to raise if referenced; oriented-edge must not call it.
    def explode(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("Haim–Kislev must not be called from oriented-edge")

    monkeypatch.setattr(
        "viterbo.math.capacity_ehz.stubs.capacity_ehz_haim_kislev",
        explode,
        raising=True,
    )

    square_q, square_p = PLANAR_POLYTOPE_PAIRS["square_product"]
    vertices, normals, offsets = lagrangian_product(square_q.vertices, square_p.vertices)

    # Small input should succeed deterministically without touching HK.
    capacity = oriented_edge_spectrum_4d(vertices, normals, offsets)
    assert capacity.ndim == 0 and torch.isfinite(capacity)
