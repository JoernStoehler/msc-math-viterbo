from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from viterbo.math import polar


@pytest.mark.smoke
@pytest.mark.xfail(
    reason="Polar helpers deferred; see docs/reviews/gap-matrix-2025-10-21.md",
    raises=NotImplementedError,
    strict=True,
)
@pytest.mark.parametrize(
    "fn,args",
    [
        (polar.polar_from_halfspaces, (torch.eye(2, dtype=torch.float64), torch.ones(2))),
        (polar.mahler_product_approx, (torch.zeros((3, 2), dtype=torch.float64),)),
    ],
)
def test_polar_stubs_raise_not_implemented(
    fn: Callable[..., object], args: tuple[object, ...]
) -> None:
    fn(*args)
