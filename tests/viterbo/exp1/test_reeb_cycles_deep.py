from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.deep]

from viterbo.exp1.examples import viterbo_counterexample
from viterbo.exp1.polytopes import to_halfspaces
from viterbo.exp1.reeb_cycles.reference import compute_ehz_capacity_and_cycle_reference


@pytest.mark.goal_math
def test_reeb_reference_capacity_and_cycle_on_counterexample_product() -> None:
    """Reference EHZ capacity returns finite value and simple cycle on 5×5 product."""
    prod = viterbo_counterexample()
    H = to_halfspaces(prod)
    A, b = H.as_tuple()
    cap, cycle = compute_ehz_capacity_and_cycle_reference(A, b, atol=1e-9)
    assert np.isfinite(cap)
    assert cycle.shape[1] == 4 and cycle.shape[0] >= 3

