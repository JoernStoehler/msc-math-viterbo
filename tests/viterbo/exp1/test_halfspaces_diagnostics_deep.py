from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

pytestmark = [pytest.mark.deep]

from viterbo.exp1.halfspaces import halfspace_degeneracy_metrics
from viterbo.exp1.examples import hypercube


@pytest.mark.goal_code
def test_near_parallel_facets_flagged_by_diagnostics() -> None:
    """Duplicate/parallel facets trigger high correlation and duplicate fraction."""
    # 2D system with duplicated x<=1 facet; still a valid (unbounded) system for diagnostics
    A = jnp.array(
        [
            [1.0, 0.0],  # x <= 1
            [1.0, 0.0],  # duplicate direction (parallel)
            [0.0, 1.0],  # y <= 1
            [0.0, -1.0],  # -y <= 1  => y >= -1
        ],
        dtype=jnp.float64,
    )
    b = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float64)
    rep = halfspace_degeneracy_metrics(A, b, atol=1e-9)
    assert float(rep.max_abs_row_correlation) >= 0.99
    assert float(rep.duplicate_facet_fraction) > 0.0


@pytest.mark.goal_code
def test_hypercube_high_dim_has_stable_conditioning() -> None:
    """Hypercube H-reps have well-conditioned facet matrices across dimensions."""
    H = hypercube(8)
    A, b = H.as_tuple()
    rep = halfspace_degeneracy_metrics(A, b, atol=1e-12)
    assert rep.rank == 8
    # B = [I; -I] ⇒ singular values equal ⇒ condition ≈ 1
    assert math.isclose(float(rep.condition_number), 1.0, rel_tol=1e-12, abs_tol=1e-12)
    assert float(rep.duplicate_facet_fraction) == 0.0
    assert float(rep.max_abs_row_correlation) <= 1.0
