"""EHZ spectrum interfaces and expected output shapes."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo import polytopes, spectrum
from viterbo.capacity import reeb_cycles


EXPECTED_SIMPLEX_EDGES = (
    (0, (0, 1, 2), 0, 1, 3, 4),
    (1, (0, 1, 2), 1, 0, 4, 3),
    (2, (0, 1, 3), 0, 2, 2, 4),
    (3, (0, 1, 3), 2, 0, 4, 2),
    (4, (0, 2, 3), 0, 3, 1, 4),
    (5, (0, 2, 3), 3, 0, 4, 1),
    (6, (1, 2, 3), 0, 4, 0, 4),
    (7, (1, 2, 3), 4, 0, 4, 0),
    (8, (0, 1, 4), 1, 2, 2, 3),
    (9, (0, 1, 4), 2, 1, 3, 2),
    (10, (0, 2, 4), 1, 3, 1, 3),
    (11, (0, 2, 4), 3, 1, 3, 1),
    (12, (1, 2, 4), 1, 4, 0, 3),
    (13, (1, 2, 4), 4, 1, 3, 0),
    (14, (0, 3, 4), 2, 3, 1, 2),
    (15, (0, 3, 4), 3, 2, 2, 1),
    (16, (1, 3, 4), 2, 4, 0, 2),
    (17, (1, 3, 4), 4, 2, 2, 0),
    (18, (2, 3, 4), 3, 4, 0, 1),
    (19, (2, 3, 4), 4, 3, 1, 0),
)

EXPECTED_SIMPLEX_SUCCESSORS = (
    (8, 10, 12),
    (2, 4, 6),
    (9, 14, 16),
    (0, 4, 6),
    (11, 15, 18),
    (0, 2, 6),
    (13, 17, 19),
    (0, 2, 4),
    (3, 14, 16),
    (1, 10, 12),
    (5, 15, 18),
    (1, 8, 12),
    (7, 17, 19),
    (1, 8, 10),
    (5, 11, 18),
    (3, 9, 16),
    (7, 13, 19),
    (3, 9, 14),
    (7, 13, 17),
    (5, 11, 15),
)

EXPECTED_SIMPLEX_INCOMING = (
    (3, 5, 7),
    (9, 11, 13),
    (1, 5, 7),
    (8, 15, 17),
    (1, 3, 7),
    (10, 14, 19),
    (1, 3, 5),
    (12, 16, 18),
    (0, 11, 13),
    (2, 15, 17),
    (0, 9, 13),
    (4, 14, 19),
    (0, 9, 11),
    (6, 16, 18),
    (2, 8, 17),
    (4, 10, 19),
    (2, 8, 15),
    (6, 12, 18),
    (4, 10, 14),
    (6, 12, 16),
)

EXPECTED_SIMPLEX_SPECTRUM = (
    6.82842712474619,
    23.313708498984763,
    28.970562748477146,
    35.798989873223334,
    35.798989873223334,
)


@pytest.mark.goal_code
@pytest.mark.smoke
def test_ehz_spectrum_reference_requires_four_dimensional_bundle() -> None:
    """2D bundles trigger a ValueError because the oriented-edge graph is 4D-specific."""

    vertices = jnp.asarray(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=jnp.float64,
    )
    bundle = polytopes.build_from_vertices(vertices)
    with pytest.raises(ValueError, match="dimension four"):
        spectrum.ehz_spectrum_reference(bundle, head=3)


@pytest.mark.goal_code
@pytest.mark.smoke
def test_ehz_spectrum_reference_manual_batching_pattern() -> None:
    """Manual loops construct padded spectra; module exposes per-instance APIs only."""

    vertices4 = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    bundle4 = polytopes.build_from_vertices(vertices4)
    vertices2 = jnp.asarray(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=jnp.float64,
    )
    bundle2 = polytopes.build_from_vertices(vertices2)
    head = 5

    padded_rows: list[jnp.ndarray] = []
    for bundle in (bundle4, bundle2):
        padded = jnp.full((head,), float("nan"), dtype=jnp.float64)
        try:
            seq = spectrum.ehz_spectrum_reference(bundle, head=head)
        except ValueError:
            padded_rows.append(padded)
            continue
        if seq:
            values = jnp.asarray(seq, dtype=jnp.float64)[:head]
            padded = padded.at[: values.shape[0]].set(values)
        padded_rows.append(padded)

    arr = jnp.stack(padded_rows, axis=0)
    assert arr.shape == (2, head)
    assert jnp.any(jnp.isfinite(arr[0]))
    assert jnp.all(jnp.isnan(arr[1]))
    assert not hasattr(spectrum, "ehz_spectrum_batched")


@pytest.mark.goal_code
@pytest.mark.smoke
def test_oriented_edge_graph_matches_expected_metadata() -> None:
    """Modern oriented-edge graph matches the expected facets and adjacency on a 4D simplex."""

    vertices = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    bundle = polytopes.build_from_vertices(vertices)

    graph = reeb_cycles.build_oriented_edge_graph(bundle)

    edges = tuple(
        (
            edge.identifier,
            edge.facets,
            edge.tail_vertex,
            edge.head_vertex,
            edge.tail_missing_facet,
            edge.head_missing_facet,
        )
        for edge in graph.edges
    )
    assert edges == EXPECTED_SIMPLEX_EDGES

    successors = tuple(graph.successors(edge.identifier) for edge in graph.edges)
    assert successors == EXPECTED_SIMPLEX_SUCCESSORS

    assert graph.incoming == EXPECTED_SIMPLEX_INCOMING


@pytest.mark.goal_math
@pytest.mark.smoke
def test_ehz_spectrum_reference_matches_expected_values() -> None:
    """Spectrum enumeration matches the pre-computed simplex action sequence."""

    vertices = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    bundle = polytopes.build_from_vertices(vertices)
    spectrum_values = spectrum.ehz_spectrum_reference(bundle, head=5)
    assert spectrum_values == pytest.approx(EXPECTED_SIMPLEX_SPECTRUM, rel=1e-12, abs=0.0)
