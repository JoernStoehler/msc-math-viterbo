"""Tests for symmetry-reduced EHZ capacity solvers."""

from __future__ import annotations

import numpy as np
import pytest

from viterbo.geometry.polytopes import cross_polytope, hypercube
from viterbo.symplectic.capacity import (
    FacetPairingMetadata,
    compute_ehz_capacity_fast,
    compute_ehz_capacity_fast_symmetry_reduced,
    compute_ehz_capacity_reference,
    compute_ehz_capacity_reference_symmetry_reduced,
    detect_opposite_facet_pairs,
)

_CENTRAL_POLYTOPES = tuple(
    (*poly.halfspace_data(), poly.name)
    for poly in (
        hypercube(2, name="hypercube-2d"),
        cross_polytope(2, name="cross-polytope-2d"),
        hypercube(3, name="hypercube-3d"),
    )
)


@pytest.mark.goal_code
def test_detect_opposite_pairs_hypercube() -> None:
    """Detector finds two opposite-facet pairs for a 2D hypercube (square)."""
    B, c = hypercube(2, name="hypercube-detection").halfspace_data()
    metadata = detect_opposite_facet_pairs(B, c)
    assert len(metadata.pairs) == 2
    assert metadata.unpaired == ()


@pytest.mark.goal_code
def test_detect_opposite_pairs_cross_polytope() -> None:
    """Detector finds two opposite-facet pairs for a 2D cross-polytope (diamond)."""
    B, c = cross_polytope(2, name="cross-polytope-detection").halfspace_data()
    metadata = detect_opposite_facet_pairs(B, c)
    assert len(metadata.pairs) == 2
    assert metadata.unpaired == ()


@pytest.mark.goal_math
@pytest.mark.parametrize("B, c, name", _CENTRAL_POLYTOPES)
def test_reference_symmetry_matches_full(B: np.ndarray, c: np.ndarray, name: str) -> None:
    """Symmetry-reduced reference equals the full solver on central polytopes."""
    try:
        reference = compute_ehz_capacity_reference(B, c)  # type: ignore[reportArgumentType]
    except ValueError:
        with pytest.raises(ValueError):
            compute_ehz_capacity_reference_symmetry_reduced(B, c)  # type: ignore[reportArgumentType]
    else:
        reduced = compute_ehz_capacity_reference_symmetry_reduced(B, c)  # type: ignore[reportArgumentType]
        assert np.isclose(reference, reduced, atol=1e-8)


@pytest.mark.goal_math
@pytest.mark.parametrize("B, c, name", _CENTRAL_POLYTOPES)
def test_fast_symmetry_matches_full(B: np.ndarray, c: np.ndarray, name: str) -> None:
    """Symmetry-reduced fast solver equals the full fast solver on central polytopes."""
    try:
        fast = compute_ehz_capacity_fast(B, c)  # type: ignore[reportArgumentType]
    except ValueError:
        with pytest.raises(ValueError):
            compute_ehz_capacity_fast_symmetry_reduced(B, c)  # type: ignore[reportArgumentType]
    else:
        reduced = compute_ehz_capacity_fast_symmetry_reduced(B, c)  # type: ignore[reportArgumentType]
        assert np.isclose(fast, reduced, atol=1e-8)


@pytest.mark.goal_code
def test_solver_respects_pairing_overrides() -> None:
    """Explicit pairing overrides are applied and preserve baseline capacity."""
    poly = hypercube(2, name="hypercube-override")
    B, c = poly.halfspace_data()
    metadata = FacetPairingMetadata(pairs=((2, 0), (3, 1)), unpaired=())

    try:
        baseline = compute_ehz_capacity_fast(B, c)
    except ValueError:
        with pytest.raises(ValueError):
            compute_ehz_capacity_fast_symmetry_reduced(
                B,
                c,
                pairing=metadata,
                enforce_detection=False,
            )
    else:
        reduced = compute_ehz_capacity_fast_symmetry_reduced(
            B,
            c,
            pairing=metadata,
            enforce_detection=False,
        )
        assert np.isclose(reduced, baseline, atol=1e-8)


@pytest.mark.goal_code
def test_canonical_subset_allows_paired_facets() -> None:
    """Canonical subset permits paired facets and rejects reversed-order duplicates."""
    metadata = FacetPairingMetadata(pairs=((0, 1), (2, 3)), unpaired=())

    assert metadata.is_canonical_subset((0, 1, 2))
    assert not metadata.is_canonical_subset((1, 2, 3))
    assert not metadata.is_canonical_subset((1, 0, 2))
