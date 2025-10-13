"""Dataset dispatch utilities for atlas quantity computation.

The helpers here are intentionally thin: they accept an :class:`AtlasJaxRow`
containing precomputed columns and simply forward the stored arrays to the
canonical math-layer algorithms.  This keeps dependencies explicit (callers must
populate prerequisites ahead of time) and avoids re-implementing numerical
logic outside :mod:`viterbo.math`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, TypeVar

import jax
import numpy as np

from viterbo.math.capacity import facet_normals, reeb_cycles, symmetry_reduced
from viterbo.math.volume import (
    polytope_volume_fast,
    polytope_volume_reference,
    volume_reference,
)


_T = TypeVar("_T")


def _no_jit_call(func: Callable[..., _T], *args, **kwargs) -> _T:
    """Execute ``func`` with ``jax.jit`` temporarily disabled."""

    with jax.disable_jit():
        return func(*args, **kwargs)


@dataclass
class AtlasJaxRow:
    """Typed container used by dataset builders.

    The hugging-face representation remains an untyped ``dict``.  Builders
    populate this structure with any columns that have already been computed.
    Canonical ``compute_*`` helpers consume these values without triggering
    implicit conversions – missing prerequisites raise ``KeyError``.
    """

    vertices: np.ndarray | None = None
    normals: np.ndarray | None = None
    offsets: np.ndarray | None = None
    incidence: np.ndarray | None = None
    volume: dict[str, float] | None = None
    capacity_ehz: dict[str, float] | None = None

    def require(self, *fields: str) -> None:
        missing = [name for name in fields if getattr(self, name) is None]
        if missing:
            joined = ", ".join(sorted(missing))
            raise KeyError(f"Missing prerequisite columns: {joined}")

    @property
    def dimension(self) -> int | None:
        if self.vertices is not None and self.vertices.ndim == 2:
            return int(self.vertices.shape[1])
        if self.normals is not None and self.normals.ndim == 2:
            return int(self.normals.shape[1])
        return None


@dataclass(frozen=True)
class ReebCycleSummary:
    """Structured result returned by :func:`compute_reeb_cycles`."""

    edge_count: int
    cycles: tuple[tuple[int, ...], ...]


VOLUME_ALGORITHMS: tuple[str, ...] = (
    "halfspaces_reference",
    "halfspaces_fast",
    "vertices_reference",
)
CAPACITY_EHZ_ALGORITHMS: tuple[str, ...] = (
    "facet_normals_reference",
    "facet_normals_fast",
    "reeb_reference",
    "reeb_fast",
    "symmetry_reduced_reference",
    "symmetry_reduced_fast",
)
SPECTRUM_ALGORITHMS: tuple[str, ...] = ("ehz_reference",)
REEB_CYCLE_ALGORITHMS: tuple[str, ...] = ("oriented_edges",)
SYSTOLIC_RATIO_ALGORITHMS: tuple[str, ...] = CAPACITY_EHZ_ALGORITHMS


def compute_volume(row: AtlasJaxRow, *, algorithm: str, atol: float = 1e-9) -> float:
    if algorithm not in VOLUME_ALGORITHMS:
        raise ValueError(f"Unknown volume algorithm: {algorithm}")
    if algorithm == "vertices_reference":
        row.require("vertices")
        if row.vertices.size == 0:
            return float("nan")
        return float(_no_jit_call(volume_reference, row.vertices))
    row.require("normals", "offsets")
    if algorithm == "halfspaces_reference":
        return float(
            _no_jit_call(polytope_volume_reference, row.normals, row.offsets, atol=atol)
        )
    if algorithm == "halfspaces_fast":
        return float(_no_jit_call(polytope_volume_fast, row.normals, row.offsets, atol=atol))
    raise AssertionError("Unhandled volume algorithm")


def compute_capacity_ehz(row: AtlasJaxRow, *, algorithm: str) -> float:
    if algorithm not in CAPACITY_EHZ_ALGORITHMS:
        raise ValueError(f"Unknown capacity algorithm: {algorithm}")
    row.require("normals", "offsets")
    dimension = row.dimension
    if dimension is None:
        raise KeyError("Missing dimension metadata for capacity computation")
    if dimension > 2:
        raise ValueError(
            "Atlas tiny only records EHZ capacities for planar polytopes; "
            "higher dimensions remain TODO."
        )
    if algorithm == "facet_normals_reference":
        return float(
            _no_jit_call(
                facet_normals.ehz_capacity_reference_facet_normals, row.normals, row.offsets
            )
        )
    if algorithm == "facet_normals_fast":
        return float(
            _no_jit_call(
                facet_normals.ehz_capacity_fast_facet_normals, row.normals, row.offsets
            )
        )
    if algorithm == "reeb_reference":
        return float(
            _no_jit_call(reeb_cycles.ehz_capacity_reference_reeb, row.normals, row.offsets)
        )
    if algorithm == "reeb_fast":
        return float(
            _no_jit_call(reeb_cycles.ehz_capacity_fast_reeb, row.normals, row.offsets)
        )
    if algorithm == "symmetry_reduced_reference":
        return float(
            _no_jit_call(
                symmetry_reduced.ehz_capacity_reference_symmetry_reduced,
                row.normals,
                row.offsets,
            )
        )
    if algorithm == "symmetry_reduced_fast":
        return float(
            _no_jit_call(
                symmetry_reduced.ehz_capacity_fast_symmetry_reduced, row.normals, row.offsets
            )
        )
    raise AssertionError("Unhandled capacity algorithm")


def compute_spectrum(row: AtlasJaxRow, *, algorithm: str, head: int) -> tuple[float, ...]:
    if algorithm not in SPECTRUM_ALGORITHMS:
        raise ValueError(f"Unknown spectrum algorithm: {algorithm}")
    row.require("normals", "offsets")
    if row.dimension != 4:
        raise ValueError("EHZ spectrum currently only tabulated for 4D polytopes")
    raise ValueError("EHZ spectrum computation for atlas_tiny is pending")


def compute_reeb_cycles(row: AtlasJaxRow, *, algorithm: str, limit: int) -> ReebCycleSummary:
    if algorithm not in REEB_CYCLE_ALGORITHMS:
        raise ValueError(f"Unknown cycle algorithm: {algorithm}")
    row.require("normals", "offsets")
    if row.dimension != 4:
        raise ValueError("Reeb cycles currently only defined for 4D polytopes")
    raise ValueError("Reeb cycle computation for atlas_tiny is pending")


def compute_systolic_ratio(row: AtlasJaxRow, *, algorithm: str) -> float:
    if algorithm not in SYSTOLIC_RATIO_ALGORITHMS:
        raise ValueError(f"Unknown systolic algorithm: {algorithm}")
    if row.capacity_ehz is None or algorithm not in row.capacity_ehz:
        raise KeyError(f"Missing capacity value for algorithm {algorithm}")
    if row.volume is None or "halfspaces_reference" not in row.volume:
        raise KeyError("Missing reference volume for systolic ratio")
    dimension = row.dimension
    if dimension is None or dimension % 2 != 0:
        return float("nan")
    capacity_value = row.capacity_ehz[algorithm]
    reference_volume = row.volume["halfspaces_reference"]
    if not math.isfinite(capacity_value) or capacity_value <= 0.0:
        return float("nan")
    if not math.isfinite(reference_volume) or reference_volume <= 0.0:
        return float("nan")
    exponent = dimension // 2
    return float((capacity_value**exponent) / reference_volume)


