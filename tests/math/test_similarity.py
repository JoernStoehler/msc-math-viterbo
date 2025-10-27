from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from viterbo.math import similarity


@pytest.mark.smoke
@pytest.mark.xfail(
    reason="Similarity metrics deferred; see docs/reviews/gap-matrix-2025-10-21.md",
    raises=NotImplementedError,
    strict=True,
)
@pytest.mark.parametrize(
    "fn,args",
    [
        (similarity.hausdorff_distance, (torch.zeros((2, 2)), torch.ones((2, 2)))),
        (
            similarity.hausdorff_distance_under_symplectic_group,
            (torch.zeros((2, 2)), torch.ones((2, 2))),
        ),
        (
            similarity.support_l2_distance,
            (torch.zeros((2, 2)), torch.ones((2, 2)), 4),
        ),
    ],
)
def test_similarity_stubs_raise_not_implemented(
    fn: Callable[..., torch.Tensor], args: tuple[object, ...]
) -> None:
    fn(*args)
