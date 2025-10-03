"""Unit tests for :mod:`viterbo.symplectic.core`."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from viterbo.symplectic.core import ZERO_TOLERANCE, normalize_vector


def test_normalize_vector_unit_length() -> None:
    vector = np.array([3.0, 4.0])  # shape: (2,)
    normalized = normalize_vector(vector)
    assert (
        pytest.approx(1.0)  # type: ignore[reportUnknownMemberType]  # Pytest stubs incomplete; TODO: refine types
        == float(np.linalg.norm(normalized))
    )


def test_normalize_vector_zero_vector_raises() -> None:
    zero = np.zeros(3)  # shape: (3,)
    with pytest.raises(ValueError):
        normalize_vector(zero)


def test_normalize_vector_accepts_list_input() -> None:
    values = [3.0, 4.0, 12.0]
    normalized = normalize_vector(cast(Any, values))
    assert pytest.approx(1.0) == float(np.linalg.norm(normalized))


def test_zero_tolerance_reasonable() -> None:
    assert 0.0 < ZERO_TOLERANCE < 1e-6
